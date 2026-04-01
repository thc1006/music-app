#!/usr/bin/env python3
"""Resume Phase 3 Stage 2 from existing Stage 1 best.pt.

Stage 1 completed 30 epochs (best mAP50=0.6867 at epoch 21, strict protocol).
Stage 2 crashed due to a path resolution bug (now fixed in train_phase3.py).

This script directly launches Stage 2 (full fine-tune at ultra-low LR)
from the Stage 1 checkpoint without re-running Stage 1.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- Paths ---
_ROOT = Path("/home/thc1006/dev/music-app")
S1_BEST = (
    _ROOT / "runs/detect/harmony_omr_v2_phase3"
    / "yolo12s_two_stage_20260218_012553_s1/weights/best.pt"
)
DATA = str(_ROOT / "training/datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml")
PROJECT = "harmony_omr_v2_phase3"
S2_NAME = "yolo12s_two_stage_s2_v2"

# --- Stage 2 config ---
# Use original dataset (32K, not 92K balanced) — Stage 1 already trained on
# balanced data, Stage 2 ultra-low-LR refinement doesn't need oversampling.
# batch=6 to reduce TaskAlignedAssigner OOM on dense images.
S2_ARGS = {
    "data": DATA,
    "imgsz": 1280,
    "batch": 6,
    "device": "0",
    "workers": 16,
    "cache": False,
    "amp": False,
    "optimizer": "AdamW",
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "nbs": 64,  # effective batch = 64 (accumulate 64/6 ≈ 11 steps)
    "cls": 0.7,
    "box": 7.5,
    "mosaic": 0.5,
    "close_mosaic": 20,
    "scale": 0.5,
    "copy_paste": 0.0,
    "mixup": 0.0,
    "save": True,
    "save_period": 5,
    "plots": False,
    "val": True,
    "verbose": True,
    "project": PROJECT,
    "name": S2_NAME,
    "exist_ok": True,
    "deterministic": False,
    "resume": False,
    # Stage 2 specific
    "epochs": 150,
    "lr0": 0.0001,
    "lrf": 0.01,
    "warmup_epochs": 3.0,
    "patience": 40,
    "cos_lr": True,
    "freeze": 0,  # unfreeze everything
}


def main() -> int:
    log_dir = _ROOT / "training" / PROJECT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_dir / f"{S2_NAME}.log"), mode="a"),
        ],
    )
    log = logging.getLogger("phase3_s2")

    # Validate Stage 1 checkpoint
    if not S1_BEST.exists():
        log.error("Stage 1 best.pt not found: %s", S1_BEST)
        return 1
    log.info("Starting Stage 2 from: %s", S1_BEST)

    # Pre-flight checks
    import torch
    log.info("torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("gpu=%s vram=%.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    from ultralytics import YOLO

    model = YOLO(str(S1_BEST))
    log.info("Model loaded, starting Stage 2 training...")
    log.info("s2_args: epochs=%s lr0=%s patience=%s",
             S2_ARGS["epochs"], S2_ARGS["lr0"], S2_ARGS["patience"])

    model.train(**S2_ARGS)

    # Locate result
    try:
        save_dir = Path(model.trainer.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
    except Exception:
        best_pt = Path("runs/detect") / PROJECT / S2_NAME / "weights" / "best.pt"

    log.info("Stage 2 complete! best_pt=%s exists=%s", best_pt, best_pt.exists())

    # Run unified eval
    eval_script = Path(__file__).parent / "unified_eval.py"
    if eval_script.exists() and best_pt.exists():
        import subprocess
        log.info("Running unified evaluation...")
        subprocess.run(
            [sys.executable, str(eval_script),
             "--models", str(best_pt),
             "--data", DATA,
             "--imgsz", "1280",
             "--device", "0"],
            check=False,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
