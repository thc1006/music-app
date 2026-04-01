#!/usr/bin/env python3
"""Phase 4: Conservative fine-tuning with TAL box capping.

Cross-validated findings from 5 research agents (2026-02-23):
- Phase 3 FAILED because: cls=1.5, copy_paste=0.15, mixup=0.1, scale=0.7
- This run uses DEFAULTS for loss weights and ZERO augmentation on converged model
- Box capping monkey-patch eliminates 100% of TAL OOM events
- Two-stage progressive unfreeze: freeze backbone -> full model

Base model: Ultimate v5 Stable (mAP50=0.7763 deploy, 0.7254 strict on cleaned_v2)
Dataset: Phase 8 Cleaned v2 (26,115 train / 2,867 val)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

# --- Config ---
_ROOT = Path("/home/thc1006/dev/music-app")
BASE_MODEL = str(_ROOT / "training/harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt")
DATA = str(_ROOT / "training/datasets/yolo_harmony_v2_phase8_cleaned_v2/harmony_phase8_cleaned_v2.yaml")
PROJECT = "harmony_omr_v2_phase4"
NAME = "conservative_v1"

MAX_BOXES = 400  # Cap GT boxes per image in TAL to prevent OOM

# Stage 1: Freeze backbone (layers 0-9), train head only
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
    "epochs": 30,
    "patience": 15,
    "freeze": 10,
    # Loss weights: DEFAULTS
    "cls": 0.5,
    "box": 7.5,
    "dfl": 1.5,
    # Augmentation: MINIMAL for converged model
    "copy_paste": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "scale": 0.3,
    "mosaic": 0.5,
    "close_mosaic": 5,
    "erasing": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.2,
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
    "epochs": 120,
    "patience": 40,
    "mosaic": 0.3,
    "close_mosaic": 15,
    "save_period": 10,
    "name": f"{NAME}_stage2",
    "compile": True,
}


# ---------------------------------------------------------------------------
# TAL Box Capping Monkey-Patch
# ---------------------------------------------------------------------------
def _apply_box_capping_patch(max_boxes: int = 400) -> None:
    """Monkey-patch v8DetectionLoss.preprocess to cap GT boxes per image.

    When an image has > max_boxes ground truth boxes, randomly sample max_boxes.
    This caps TAL memory from O(batch * N_max * anchors) to a fixed budget.
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

        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        actual_max = int(counts.max().item())

        if actual_max <= max_boxes:
            # No capping needed — use original
            return _original_preprocess(self, targets, batch_size, scale_tensor)

        # Cap: randomly sample max_boxes per image that exceeds limit
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

        # Rebuild output tensor with capped counts
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
    logging.getLogger("phase4").info(
        "TAL box capping applied: max_boxes=%d (was unlimited)", max_boxes
    )


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
    log = logging.getLogger("phase4")

    log.info("torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info(
            "gpu=%s vram=%.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Apply box capping BEFORE importing YOLO (so loss class is patched)
    _apply_box_capping_patch(MAX_BOXES)

    from ultralytics import YOLO

    # ===== Stage 1: Frozen backbone =====
    log.info("=" * 60)
    log.info("STAGE 1: Freeze backbone (layers 0-9), train head only")
    log.info("=" * 60)
    model = YOLO(BASE_MODEL)
    log.info("Model: %s", BASE_MODEL)
    log.info("Data: %s", DATA)
    log.info(
        "Config: epochs=%d, lr0=%s, freeze=%s, batch=%d",
        STAGE1_ARGS["epochs"],
        STAGE1_ARGS["lr0"],
        STAGE1_ARGS["freeze"],
        STAGE1_ARGS["batch"],
    )

    t0 = time.time()
    model.train(**STAGE1_ARGS)
    stage1_time = time.time() - t0
    log.info("Stage 1 complete in %.1f hours", stage1_time / 3600)

    # Get Stage 1 best weights
    try:
        stage1_dir = Path(model.trainer.save_dir)
        stage1_best = stage1_dir / "weights" / "best.pt"
    except Exception:
        stage1_best = Path(PROJECT) / NAME / "weights" / "best.pt"

    log.info("Stage 1 best: %s (exists=%s)", stage1_best, stage1_best.exists())

    # ===== Stage 2: Full model, very low LR =====
    log.info("=" * 60)
    log.info("STAGE 2: Unfreeze all, lr0=0.0001")
    log.info("=" * 60)

    if stage1_best.exists():
        stage2_model = YOLO(str(stage1_best))
    else:
        log.warning("Stage 1 best not found, using last.pt")
        stage1_last = stage1_dir / "weights" / "last.pt"
        stage2_model = YOLO(str(stage1_last))

    log.info(
        "Config: epochs=%d, lr0=%s, freeze=%s",
        STAGE2_ARGS["epochs"],
        STAGE2_ARGS["lr0"],
        STAGE2_ARGS["freeze"],
    )

    t1 = time.time()
    stage2_model.train(**STAGE2_ARGS)
    stage2_time = time.time() - t1
    log.info("Stage 2 complete in %.1f hours", stage2_time / 3600)

    # Get Stage 2 best weights
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
        log.info("Running dual-protocol evaluation...")
        eval_model = YOLO(str(best_pt))

        # Deploy protocol (conf=0.25, iou=0.55)
        m1 = eval_model.val(
            data=DATA, imgsz=1280, conf=0.25, iou=0.55,
            device="0", verbose=False, plots=False,
        )
        log.info(
            "Cleaned v2 (deploy): mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f",
            float(m1.box.map50), float(m1.box.map),
            float(m1.box.mp), float(m1.box.mr),
        )

        # Strict protocol (conf=0.001, iou=0.7)
        m2 = eval_model.val(
            data=DATA, imgsz=1280, conf=0.001, iou=0.7,
            device="0", verbose=False, plots=False,
        )
        log.info(
            "Cleaned v2 (strict): mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f",
            float(m2.box.map50), float(m2.box.map),
            float(m2.box.mp), float(m2.box.mr),
        )

        # Gate check
        deploy_map50 = float(m1.box.map50)
        strict_map50 = float(m2.box.map50)
        log.info("=" * 60)
        log.info("GATE CHECK:")
        log.info("  Deploy mAP50: %.4f (baseline=0.7763, %s)",
                 deploy_map50, "PASS" if deploy_map50 > 0.7763 else "FAIL")
        log.info("  Strict mAP50: %.4f (baseline=0.7254, %s)",
                 strict_map50, "PASS" if strict_map50 > 0.7254 else "FAIL")
        log.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
