#!/usr/bin/env python3
"""Phase 9: CV Noteheads + Pseudo-Label Training (Phase 5 Teacher → OpenScore Relabeled)

Background:
  - Phase 6: fixed notehead bbox to 23×25px → model predicts 0 on OpenScore
  - Phase 7: universal bbox fix → 26/32 classes WORSE
  - Pseudo-label: Phase 5 predicts on OpenScore (large boxes) → shrink to
    TAL-learnable size → use as training labels

Dataset: yolo_harmony_v2_pseudo_v1
  - DoReMi labels: original (unchanged)
  - OpenScore labels: Phase 5 predictions, shrunk to 40-120px pseudo-labels
  - 32,555 train / 2,867 val images

Base model: Phase 5 Stage 2 best.pt
  - Already knows OpenScore visual features
  - cv2 (bbox regression) reset for new box sizes

Two-stage LP-FT (same as Phase 6, proven strategy):
  Stage 1: freeze=10, lr0=0.0003, 60 epochs — learn new bbox sizes
  Stage 2: freeze=0, lr0=0.0001, 120 epochs, patience=40 — full convergence

Success criteria:
  - mAP50 > 0.83 (Phase 6 was 0.816 with broken noteheads)
  - OpenScore notehead recall > 0.30 (Phase 6 was 0)
  - Other 30 classes: no significant regression from Phase 6
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_ROOT = Path("/home/thc1006/dev/music-app")
_TRAIN_ROOT = _ROOT / "training"

BASE_MODEL = str(
    _ROOT / "runs/detect/harmony_omr_v2_phase5/nostem_v1_stage2/weights/best.pt"
)
DATA = str(
    _TRAIN_ROOT / "datasets/yolo_harmony_v2_pseudo_v1/harmony_pseudo_v1.yaml"
)
PROJECT = str(_ROOT / "runs/detect/harmony_omr_v2_phase9")
NAME = "cv_noteheads_v1"

MAX_BOXES = 400
TAL_TOPK = 13

FOCUS_CLASSES = {
    0: "notehead_filled",
    1: "notehead_hollow",
    2: "beam",
    7: "tie",
    22: "barline",
    23: "barline_double",
    31: "ledger_line",
}

# Stage 1: Freeze backbone, train head + neck
STAGE1_ARGS = {
    "data": DATA,
    "imgsz": 1280,
    "batch": 8,
    "nbs": 64,
    "device": "0",
    "workers": 16,
    "cache": "ram",
    "amp": False,
    "optimizer": "AdamW",
    "lr0": 0.0003,
    "lrf": 0.1,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,
    "epochs": 60,
    "patience": 30,
    "freeze": 10,
    "cls": 1.0,
    "box": 7.5,       # Higher box loss for cv2 reset convergence
    "dfl": 1.5,
    "mosaic": 0.0,
    "copy_paste": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "scale": 0.2,
    "erasing": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.2,
    "close_mosaic": 0,
    "max_det": 1500,
    "save": True,
    "save_period": 10,
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
    "box": 5.0,
    "save_period": 10,
    "name": f"{NAME}_stage2",
}


# ---------------------------------------------------------------------------
# Logger (same as Phase 6)
# ---------------------------------------------------------------------------
_TW = timezone(timedelta(hours=8))


def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    class TaiwanFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=_TW)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    fmt = TaiwanFormatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_dir / f"{name}.log"), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class MetricsLogger:
    def __init__(self, log_dir: Path, name: str):
        self.path = log_dir / f"{name}_metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_epoch(self, stage: int, epoch: int, metrics: dict) -> None:
        record = {
            "ts": datetime.now(tz=_TW).isoformat(),
            "stage": stage,
            "epoch": epoch,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_eval(self, protocol: str, metrics: dict) -> None:
        record = {
            "ts": datetime.now(tz=_TW).isoformat(),
            "event": "eval",
            "protocol": protocol,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Surgical cv2 Reset
# ---------------------------------------------------------------------------

def surgical_reset_cv2(model, log: logging.Logger) -> int:
    detect = model.model.model[-1]
    reset_count = 0
    for i, seq in enumerate(detect.cv2):
        for m in seq.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                reset_count += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                reset_count += m.weight.numel() + m.bias.numel()
        seq[-1].bias.data[:] = 2.0

    total_params = sum(p.numel() for p in model.model.parameters())
    log.info(
        "cv2 surgical reset: %d params re-initialized (%.1f%% of %d total)",
        reset_count, reset_count / total_params * 100, total_params,
    )
    return reset_count


# ---------------------------------------------------------------------------
# Monkey-Patches
# ---------------------------------------------------------------------------

def _apply_box_capping_patch(max_boxes: int, log: logging.Logger) -> None:
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
    log.info("TAL box capping applied: max_boxes=%d", max_boxes)


def _apply_tal_topk_patch(topk: int, log: logging.Logger) -> None:
    from ultralytics.utils.loss import v8DetectionLoss

    _original_init = v8DetectionLoss.__init__

    def _patched_init(self, model, tal_topk: int = 10, tal_topk2=None):
        _original_init(self, model, tal_topk=topk, tal_topk2=tal_topk2)

    v8DetectionLoss.__init__ = _patched_init
    log.info("TAL topk patched: tal_topk=%d (was 10)", topk)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Hardware optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    log_dir = Path(PROJECT) / "logs"
    log = setup_logger("phase9_cv", log_dir)
    ml = MetricsLogger(log_dir, "phase9_cv")

    log.info("=" * 70)
    log.info("Phase 9: CV Noteheads + Pseudo-Label Training")
    log.info("=" * 70)
    log.info("Base model: %s", BASE_MODEL)
    log.info("Dataset:    %s", DATA)
    log.info("Project:    %s/%s", PROJECT, NAME)
    log.info("Hardware:   TF32=%s, cuDNN.benchmark=%s, amp=False",
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark)

    # Apply patches BEFORE importing YOLO
    _apply_box_capping_patch(MAX_BOXES, log)
    _apply_tal_topk_patch(TAL_TOPK, log)

    from ultralytics import YOLO

    # ── Stage 1: Frozen backbone ──────────────────────────────────────
    log.info("")
    log.info("=" * 40)
    log.info("STAGE 1: Freeze backbone (layers 0-9)")
    log.info("=" * 40)

    model = YOLO(BASE_MODEL)
    n_reset = surgical_reset_cv2(model, log)
    log.info("Starting Stage 1: %d epochs, lr0=%.4f, freeze=%s",
             STAGE1_ARGS["epochs"], STAGE1_ARGS["lr0"], STAGE1_ARGS["freeze"])

    t0 = time.time()
    results = model.train(**STAGE1_ARGS)
    elapsed_s1 = time.time() - t0
    log.info("Stage 1 done in %.1f hours", elapsed_s1 / 3600)

    # Save Stage 1 best for Stage 2
    s1_best = Path(PROJECT) / NAME / "weights" / "best.pt"
    if not s1_best.exists():
        s1_best = Path(PROJECT) / NAME / "weights" / "last.pt"
    log.info("Stage 1 best: %s", s1_best)

    # ── Stage 2: Full unfreeze ────────────────────────────────────────
    log.info("")
    log.info("=" * 40)
    log.info("STAGE 2: Full unfreeze, low LR")
    log.info("=" * 40)

    model2 = YOLO(str(s1_best))
    log.info("Starting Stage 2: %d epochs, lr0=%.5f, patience=%d",
             STAGE2_ARGS["epochs"], STAGE2_ARGS["lr0"], STAGE2_ARGS["patience"])

    t1 = time.time()
    results2 = model2.train(**STAGE2_ARGS)
    elapsed_s2 = time.time() - t1
    log.info("Stage 2 done in %.1f hours", elapsed_s2 / 3600)

    s2_best = Path(PROJECT) / STAGE2_ARGS["name"] / "weights" / "best.pt"
    log.info("Stage 2 best: %s", s2_best)
    log.info("Total training time: %.1f hours", (elapsed_s1 + elapsed_s2) / 3600)

    log.info("")
    log.info("Training complete. Run evaluation separately.")


if __name__ == "__main__":
    main()
