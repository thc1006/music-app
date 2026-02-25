#!/usr/bin/env python3
"""Train YOLO12s on cleaned Phase 8 dataset (v2) with recall-boosting hyperparameters.

Fine-tunes from Ultimate v5 Stable best model with aggressive recall-boosting:
- cls=1.5 (up from 0.5): Emphasize classification confidence
- box=7.5 (same as Ultimate v5): Maintain bbox regression quality
- copy_paste=0.15, mixup=0.1: Data augmentation for generalization
- lr0=0.0003 (down from 0.001): Fine-tuning from best model, not from scratch

Base model: Ultimate v5 Stable (mAP50=0.7059 with iou=0.55)
Dataset: Phase 8 Cleaned v2 (28,173 train / 3,125 val, lieder/ds2 removed)

Expected improvement:
1. Higher recall on barline_double (current: 0.172 mAP50)
2. Better small object detection (ledger_line, tie)
3. Reduced false negatives overall
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# --- Config ---
_ROOT = Path("/home/thc1006/dev/music-app")
BASE_MODEL = str(_ROOT / "training/harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt")
DATA = str(_ROOT / "training/datasets/yolo_harmony_v2_phase8_cleaned_v2/harmony_phase8_cleaned_v2.yaml")
PROJECT = "harmony_omr_v2_phase3"
NAME = "recall_boost_v1"

TRAIN_ARGS = {
    "data": DATA,
    "imgsz": 1280,
    "batch": 6,
    "nbs": 64,
    "device": "0",
    "workers": 12,
    "cache": False,
    "amp": False,  # Required: float16 causes inf loss on this dataset
    "optimizer": "AdamW",
    "lr0": 0.0003,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,
    "epochs": 150,
    "patience": 50,
    "cls": 1.5,
    "box": 7.5,
    "copy_paste": 0.15,
    "mixup": 0.1,
    "scale": 0.7,
    "mosaic": 1.0,
    "close_mosaic": 20,
    "save": True,
    "save_period": 10,
    "plots": True,
    "val": True,
    "verbose": True,
    "project": PROJECT,
    "name": NAME,
    "exist_ok": True,
    "deterministic": False,
}


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
    log = logging.getLogger("phase3_recall_boost")

    import torch
    log.info("torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("gpu=%s vram=%.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    from ultralytics import YOLO

    # Fine-tune from Ultimate v5 Stable best model
    model = YOLO(BASE_MODEL)
    log.info("Model: %s", BASE_MODEL)
    log.info("Data: %s", DATA)
    log.info("Config: epochs=%s, lr0=%s, batch=%s, imgsz=%s",
             TRAIN_ARGS["epochs"], TRAIN_ARGS["lr0"],
             TRAIN_ARGS["batch"], TRAIN_ARGS["imgsz"])
    log.info("Recall-boosting: cls=%.1f", TRAIN_ARGS["cls"])
    log.info("Augmentation: copy_paste=%.2f, mixup=%.2f",
             TRAIN_ARGS["copy_paste"], TRAIN_ARGS["mixup"])

    t0 = time.time()
    model.train(**TRAIN_ARGS)
    elapsed = time.time() - t0

    # Locate result
    try:
        save_dir = Path(model.trainer.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
    except Exception:
        best_pt = Path(PROJECT) / NAME / "weights" / "best.pt"

    log.info("Training complete in %.1f hours", elapsed / 3600)
    log.info("best_pt=%s exists=%s", best_pt, best_pt.exists())

    # Run evaluation on all relevant val sets
    if best_pt.exists():
        log.info("Running evaluation...")
        eval_model = YOLO(str(best_pt))

        # 1. Cleaned v2 val - deploy protocol (conf=0.25, iou=0.55)
        m1 = eval_model.val(data=DATA, imgsz=1280, conf=0.25, iou=0.55,
                           device="0", verbose=False, plots=False)
        log.info("Cleaned v2 val (deploy): mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f",
                 float(m1.box.map50), float(m1.box.map),
                 float(m1.box.mp), float(m1.box.mr))

        # 2. Cleaned v2 val - strict protocol (conf=0.001, iou=0.7)
        m2 = eval_model.val(data=DATA, imgsz=1280, conf=0.001, iou=0.7,
                           device="0", verbose=False, plots=False)
        log.info("Cleaned v2 val (strict): mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f",
                 float(m2.box.map50), float(m2.box.map),
                 float(m2.box.mp), float(m2.box.mr))

        # 3. Original val - deploy protocol (for comparison)
        orig_data = str(_ROOT / "training/datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml")
        m3 = eval_model.val(data=orig_data, imgsz=1280, conf=0.25, iou=0.55,
                           device="0", verbose=False, plots=False)
        log.info("Original val (deploy): mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f",
                 float(m3.box.map50), float(m3.box.map),
                 float(m3.box.mp), float(m3.box.mr))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
