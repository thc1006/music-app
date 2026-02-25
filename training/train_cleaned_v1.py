#!/usr/bin/env python3
"""Train YOLO12s on cleaned Phase 8 dataset (lieder/ds2 removed).

Uses the same Ultimate v5 config that achieved mAP50=0.7519 on original data,
but trained on cleaned data (28,173 train / 3,125 val).

Expected improvement: The original model already achieves mAP50=0.7763 on the
cleaned val set WITHOUT retraining. Retraining should further improve since the
model won't learn to suppress detections from the ~4,400 toxic training images.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# --- Config ---
_ROOT = Path("/home/thc1006/dev/music-app")
DATA = str(_ROOT / "training/datasets/yolo_harmony_v2_phase8_cleaned/harmony_phase8_cleaned.yaml")
PROJECT = "harmony_omr_v2_cleaned"
NAME = "yolo12s_cleaned_v1"

TRAIN_ARGS = {
    "data": DATA,
    "imgsz": 1280,
    "batch": 8,
    "device": "0",
    "workers": 16,
    "cache": False,
    "amp": False,  # Avoid inf loss issue
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "nbs": 64,
    "cls": 0.5,
    "box": 7.5,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "scale": 0.5,
    "copy_paste": 0.0,
    "mixup": 0.0,
    "epochs": 200,
    "patience": 30,
    "cos_lr": True,
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
    log = logging.getLogger("cleaned_v1")

    import torch
    log.info("torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("gpu=%s vram=%.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    from ultralytics import YOLO

    # Start from YOLO12s pretrained (same as Ultimate v5 starting point)
    model = YOLO("yolo12s.pt")
    log.info("Model: yolo12s.pt (pretrained)")
    log.info("Data: %s", DATA)
    log.info("Config: epochs=%s, lr0=%s, batch=%s, imgsz=%s",
             TRAIN_ARGS["epochs"], TRAIN_ARGS["lr0"],
             TRAIN_ARGS["batch"], TRAIN_ARGS["imgsz"])

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

    # Run eval on both cleaned and original val sets
    if best_pt.exists():
        log.info("Running evaluation...")
        eval_model = YOLO(str(best_pt))

        # Cleaned val
        m = eval_model.val(data=DATA, imgsz=1280, conf=0.25, iou=0.55,
                          device="0", verbose=False, plots=False)
        log.info("Cleaned val: mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f",
                 float(m.box.map50), float(m.box.map),
                 float(m.box.mp), float(m.box.mr))

        # Original val (for comparison)
        orig_data = str(_ROOT / "training/datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml")
        m2 = eval_model.val(data=orig_data, imgsz=1280, conf=0.25, iou=0.55,
                           device="0", verbose=False, plots=False)
        log.info("Original val: mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f",
                 float(m2.box.map50), float(m2.box.map),
                 float(m2.box.mp), float(m2.box.mr))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
