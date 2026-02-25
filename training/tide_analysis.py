#!/usr/bin/env python3
"""Run TIDE error analysis on Ultimate v5 model.

Decomposes mAP loss into:
  - Cls: classification errors
  - Loc: localization errors
  - Both: cls + loc errors
  - Dupe: duplicate detections
  - Bkg: background false positives
  - Miss: missed ground truths (false negatives)

Usage:
    python tide_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

MODEL = Path(
    "/home/thc1006/dev/music-app/training/"
    "harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt"
)
DATA_YAML = "datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml"
DATA_ROOT = Path("datasets/yolo_harmony_v2_phase8_final")
IMGSZ = 1280
CONF = 0.001  # Low conf to get all predictions for TIDE analysis
IOU = 0.7


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_class_names(yaml_path: str) -> list[str]:
    """Load class names from YOLO data yaml."""
    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def build_gt_and_preds(model, data_yaml: str, imgsz: int, conf: float, iou: float):
    """Run model.val() and collect GT + predictions in TIDE-compatible format."""
    from tidecv import Data
    from ultralytics import YOLO
    from PIL import Image

    class_names = load_class_names(data_yaml)
    nc = len(class_names)

    # Build TIDE data objects
    gt_data = Data("GT")
    pred_data = Data("Predictions")

    # Register classes
    for i, name in enumerate(class_names):
        gt_data.add_class(i, name)
        pred_data.add_class(i, name)

    # Load val image paths
    val_img_dir = DATA_ROOT / "val" / "images"
    val_lbl_dir = DATA_ROOT / "val" / "labels"
    img_files = sorted(val_img_dir.glob("*.png")) + sorted(val_img_dir.glob("*.jpg"))

    if not img_files:
        print(f"ERROR: No images found in {val_img_dir}")
        return None, None

    print(f"Found {len(img_files)} validation images")

    # Run predictions
    print("Running model predictions...")
    t0 = time.time()
    results = model.predict(
        source=str(val_img_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device="0",
        verbose=False,
        save=False,
        stream=True,
    )

    pred_count = 0
    gt_count = 0

    for img_idx, result in enumerate(results):
        img_path = Path(result.path)
        img_id = img_idx  # Use index as image ID

        # Get image dimensions
        img_h, img_w = result.orig_shape

        # Add GT annotations
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    bbox = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
                    gt_data.add_ground_truth(img_id, cls_id, bbox)
                    gt_count += 1

        # Add predictions
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                score = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                pred_data.add_detection(img_id, cls_id, score, [x1, y1, x2, y2])
                pred_count += 1

        if (img_idx + 1) % 500 == 0:
            print(f"  Processed {img_idx + 1}/{len(img_files)} images...")

    elapsed = time.time() - t0
    print(f"Done: {gt_count} GT boxes, {pred_count} predictions in {elapsed:.0f}s")

    return gt_data, pred_data


def main() -> int:
    from ultralytics import YOLO
    from tidecv import TIDE

    if not MODEL.exists():
        print(f"ERROR: Model not found: {MODEL}")
        return 1

    model = YOLO(str(MODEL))
    print(f"Model: {MODEL.name}")
    print(f"Data:  {DATA_YAML}")
    print(f"Config: conf={CONF}, iou={IOU}, imgsz={IMGSZ}\n")

    gt_data, pred_data = build_gt_and_preds(model, DATA_YAML, IMGSZ, CONF, IOU)
    if gt_data is None:
        return 1

    # Run TIDE analysis
    print("\n" + "=" * 80)
    print("TIDE Error Analysis")
    print("=" * 80)

    tide = TIDE()
    tide.evaluate(gt_data, pred_data, mode=TIDE.BOX)
    tide.summarize()

    # Save plots
    out_dir = Path("reports/tide")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        tide.plot(out_dir=str(out_dir))
        print(f"\nPlots saved to {out_dir}/")
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")

    # Extract and save numeric summary
    errors = tide.get_all_errors()
    summary = {}
    for run_name, run_errors in errors.items():
        summary[run_name] = {}
        for err_type, err_value in run_errors.items():
            summary[run_name][str(err_type)] = round(float(err_value), 4)

    summary_path = out_dir / "tide_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
