#!/usr/bin/env python3
"""Phase 5: Stem-removed training with all cross-validated improvements.

Cross-validated findings from 5 research agents (2026-02-24):
  1. max_det=300 is the hidden killer — silently caps metrics (+3pp mAP50 fix)
  2. Stems are 24% of annotations, 2.6px wide, undetectable → remove
  3. "Dirty train, clean val" — 32K train + cleaned 2,867 val
  4. Mosaic harmful for dense OMR → 0.0
  5. Loss rebalance: box=5.0, cls=1.0 → shift precision→recall
  6. tal_topk=13 (v8DetectionLoss defaults to 10, but assigner default is 13)
  7. Box capping still needed for dense images

Base model: Ultimate v5 Stable (mAP50=0.7743 deploy, 0.7254 strict on 33-class)
Dataset: Phase 5 No-Stem (32 classes, 32K train, 2,867 cleaned val)

Two-stage progressive unfreeze:
  Stage 1: Freeze backbone (layers 0-9), train head — 40 epochs
  Stage 2: Unfreeze all, very low LR — 160 epochs
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

# --- Config ---
_ROOT = Path("/home/thc1006/dev/music-app")
BASE_MODEL = str(
    _ROOT / "training/harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt"
)
DATA = str(
    _ROOT / "training/datasets/yolo_harmony_v2_phase5_nostem/harmony_phase5_nostem.yaml"
)
PROJECT = "harmony_omr_v2_phase5"
NAME = "nostem_v1"

MAX_BOXES = 400  # TAL box capping
TAL_TOPK = 13    # Assigner default is 13, loss overrides to 10 — fix it

# Stage 1: Freeze backbone, train head + neck
STAGE1_ARGS = {
    "data": DATA,
    "imgsz": 1280,
    "batch": 6,
    "nbs": 64,
    "device": "0",
    "workers": 12,
    "cache": False,
    "amp": False,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.1,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,
    "epochs": 40,
    "patience": 20,
    "freeze": 10,
    # Loss weights: RECALL-BIASED (cross-validated recommendation)
    "cls": 1.0,   # 2x default → stronger classification signal
    "box": 5.0,   # lower than 7.5 → less box rigidity, more recall
    "dfl": 1.5,   # default
    # Augmentation: NONE for dense OMR (cross-validated: mosaic harmful)
    "mosaic": 0.0,
    "copy_paste": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "scale": 0.2,       # minimal scale jitter
    "erasing": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.2,
    "close_mosaic": 0,   # N/A since mosaic=0
    # CRITICAL: max_det=1500 (default 300 silently caps metrics)
    "max_det": 1500,
    # Other
    "save": True,
    "save_period": 5,
    "plots": True,
    "val": True,
    "verbose": True,
    "project": PROJECT,
    "name": NAME,
    "exist_ok": True,
    "deterministic": False,
    "compile": True,
}

# Stage 2: Unfreeze all, very low LR
STAGE2_ARGS = {
    **STAGE1_ARGS,
    "freeze": None,
    "lr0": 0.0001,
    "lrf": 0.01,
    "warmup_epochs": 2.0,
    "epochs": 160,
    "patience": 50,
    "save_period": 10,
    "name": f"{NAME}_stage2",
}


# ---------------------------------------------------------------------------
# Monkey-Patches
# ---------------------------------------------------------------------------

def _apply_box_capping_patch(max_boxes: int = 400) -> None:
    """Cap GT boxes per image in TAL to prevent OOM.

    At batch=6, max_boxes=400: TAL ~3.4 GB (safe for 32 GB VRAM).
    """
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.ops import xywh2xyxy

    _original_preprocess = v8DetectionLoss.preprocess

    def _capped_preprocess(
        self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor
    ) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        actual_max = int(counts.max().item())

        if actual_max <= max_boxes:
            return _original_preprocess(self, targets, batch_size, scale_tensor)

        capped_targets = []
        for j in range(batch_size):
            matches = i == j
            n = int(matches.sum().item())
            if n == 0:
                continue
            img_targets = targets[matches]
            if n > max_boxes:
                perm = torch.randperm(n, device=targets.device)[:max_boxes]
                img_targets = img_targets[perm]
            capped_targets.append(img_targets)

        if capped_targets:
            targets = torch.cat(capped_targets, dim=0)
        else:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        for j in range(batch_size):
            matches = i == j
            if n := matches.sum():
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    v8DetectionLoss.preprocess = _capped_preprocess
    logging.getLogger("phase5").info(
        "TAL box capping applied: max_boxes=%d", max_boxes
    )


def _apply_tal_topk_patch(topk: int = 13) -> None:
    """Patch v8DetectionLoss to use tal_topk=13 instead of default 10.

    The TaskAlignedAssigner default is topk=13, but v8DetectionLoss.__init__
    overrides it to 10. This patch restores the assigner's intended default.
    """
    from ultralytics.utils.loss import v8DetectionLoss

    _original_init = v8DetectionLoss.__init__

    def _patched_init(self, model, tal_topk: int = 10, tal_topk2=None):
        _original_init(self, model, tal_topk=topk, tal_topk2=tal_topk2)

    v8DetectionLoss.__init__ = _patched_init
    logging.getLogger("phase5").info(
        "TAL topk patched: tal_topk=%d (was 10)", topk
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(model_path: str, data: str, log: logging.Logger) -> None:
    """Run dual-protocol evaluation with max_det=1500."""
    from ultralytics import YOLO

    eval_model = YOLO(model_path)

    # Deploy protocol (best inference config from Phase A)
    m1 = eval_model.val(
        data=data, imgsz=1280, conf=0.15, iou=0.6,
        device="0", verbose=False, plots=False, max_det=1500,
    )
    log.info(
        "Phase5 (deploy conf=0.15,iou=0.6): mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f",
        float(m1.box.map50), float(m1.box.map),
        float(m1.box.mp), float(m1.box.mr),
    )

    # Strict protocol
    m2 = eval_model.val(
        data=data, imgsz=1280, conf=0.001, iou=0.7,
        device="0", verbose=False, plots=False, max_det=1500,
    )
    log.info(
        "Phase5 (strict conf=0.001,iou=0.7): mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f",
        float(m2.box.map50), float(m2.box.map),
        float(m2.box.mp), float(m2.box.mr),
    )

    return m1, m2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log_dir = _ROOT / "training" / PROJECT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_dir / f"{NAME}.log"), mode="a"),
        ],
    )
    log = logging.getLogger("phase5")

    log.info("=" * 60)
    log.info("Phase 5: Stem-Removed Training (32 classes)")
    log.info("=" * 60)
    log.info("torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info(
            "gpu=%s vram=%.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Check dataset exists
    data_path = Path(DATA)
    if not data_path.exists():
        log.error("Dataset not found: %s", DATA)
        log.error("Run create_phase5_nostem_dataset.py first!")
        return 1

    # Apply patches BEFORE importing YOLO
    _apply_box_capping_patch(MAX_BOXES)
    _apply_tal_topk_patch(TAL_TOPK)

    from ultralytics import YOLO

    # ===== Stage 1: Frozen backbone =====
    log.info("=" * 60)
    log.info("STAGE 1: Freeze backbone (layers 0-9), train head")
    log.info("Key changes: mosaic=0.0, cls=1.0, box=5.0, max_det=1500, tal_topk=13")
    log.info("=" * 60)

    model = YOLO(BASE_MODEL)

    # The base model has 33 classes but dataset is 32 classes.
    # Ultralytics will auto-adapt the head (reset final conv layer).
    log.info("Model: %s (33 classes → 32 class head auto-adapted)", BASE_MODEL)
    log.info("Data: %s", DATA)
    log.info(
        "Config: epochs=%d, lr0=%s, freeze=%s, batch=%d, mosaic=%.1f, cls=%.1f, box=%.1f",
        STAGE1_ARGS["epochs"],
        STAGE1_ARGS["lr0"],
        STAGE1_ARGS["freeze"],
        STAGE1_ARGS["batch"],
        STAGE1_ARGS["mosaic"],
        STAGE1_ARGS["cls"],
        STAGE1_ARGS["box"],
    )

    t0 = time.time()
    model.train(**STAGE1_ARGS)
    stage1_time = time.time() - t0
    log.info("Stage 1 complete in %.1f hours", stage1_time / 3600)

    try:
        stage1_dir = Path(model.trainer.save_dir)
        stage1_best = stage1_dir / "weights" / "best.pt"
    except Exception:
        stage1_best = Path(PROJECT) / NAME / "weights" / "best.pt"

    log.info("Stage 1 best: %s (exists=%s)", stage1_best, stage1_best.exists())

    # ===== Stage 2: Full model, very low LR =====
    log.info("=" * 60)
    log.info("STAGE 2: Unfreeze all, lr0=0.0001, epochs=160, patience=50")
    log.info("=" * 60)

    if stage1_best.exists():
        stage2_model = YOLO(str(stage1_best))
    else:
        log.warning("Stage 1 best not found, using last.pt")
        stage1_last = stage1_dir / "weights" / "last.pt"
        stage2_model = YOLO(str(stage1_last))

    t1 = time.time()
    stage2_model.train(**STAGE2_ARGS)
    stage2_time = time.time() - t1
    log.info("Stage 2 complete in %.1f hours", stage2_time / 3600)

    try:
        stage2_dir = Path(stage2_model.trainer.save_dir)
        stage2_best = stage2_dir / "weights" / "best.pt"
    except Exception:
        stage2_best = Path(PROJECT) / f"{NAME}_stage2" / "weights" / "best.pt"

    log.info("Stage 2 best: %s (exists=%s)", stage2_best, stage2_best.exists())
    log.info("Total training time: %.1f hours", (stage1_time + stage2_time) / 3600)

    # ===== Evaluation =====
    best_pt = stage2_best if stage2_best.exists() else stage1_best
    if best_pt.exists():
        log.info("Running dual-protocol evaluation with max_det=1500...")
        m_deploy, m_strict = run_eval(str(best_pt), DATA, log)

        deploy_map50 = float(m_deploy.box.map50)
        strict_map50 = float(m_strict.box.map50)

        # Note: baselines are from 33-class eval, not directly comparable to 32-class.
        # But the relative improvement is what matters.
        log.info("=" * 60)
        log.info("PHASE 5 RESULTS (32 classes, no stems):")
        log.info("  Deploy mAP50: %.4f", deploy_map50)
        log.info("  Strict mAP50: %.4f", strict_map50)
        log.info("  Deploy Recall: %.4f", float(m_deploy.box.mr))
        log.info("  Strict Recall: %.4f", float(m_strict.box.mr))
        log.info("=" * 60)
    else:
        log.error("No best.pt found for evaluation!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
