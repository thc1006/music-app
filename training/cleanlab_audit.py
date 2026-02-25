#!/usr/bin/env python3
"""Run Cleanlab ObjectLab label audit on the validation set.

Uses Ultimate v5 predictions (out-of-sample on val set) to identify:
  - Overlooked: missing bounding boxes (annotations the model detects but aren't labeled)
  - Swapped: wrong class labels
  - Bad location: poorly placed bounding boxes

Usage:
    python cleanlab_audit.py [--split val|train] [--max-images N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MODEL = Path(
    "/home/thc1006/dev/music-app/training/"
    "harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt"
)
DATA_ROOT = Path("datasets/yolo_harmony_v2_phase8_final")
DATA_YAML = "datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml"
IMGSZ = 1280
NC = 33


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_class_names() -> list[str]:
    import yaml
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def build_labels_and_predictions(model, split: str, max_images: int | None = None):
    """Build Cleanlab-compatible labels and predictions."""
    from PIL import Image

    img_dir = DATA_ROOT / split / "images"
    lbl_dir = DATA_ROOT / split / "labels"

    img_files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if max_images:
        img_files = img_files[:max_images]

    print(f"Processing {len(img_files)} images from {split} split...", flush=True)

    # Use stream mode with directory source for efficiency
    print("Running model predictions (stream mode)...", flush=True)
    t0 = time.time()

    labels_list = []
    predictions_list = []

    # Build a mapping from filename to img_files index for label lookup
    img_name_to_path = {f.name: f for f in img_files}

    results_iter = model.predict(
        source=str(img_dir),
        imgsz=IMGSZ,
        conf=0.01,
        iou=0.7,
        device="0",
        verbose=False,
        save=False,
        batch=1,
        stream=True,
    )

    for img_idx, result in enumerate(results_iter):
        if max_images and img_idx >= max_images:
            break
        img_path = Path(result.path)
        img_h, img_w = result.orig_shape

        # Build GT label dict
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_bboxes = []
        gt_labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    bbox = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
                    gt_bboxes.append(bbox)
                    gt_labels.append(cls_id)

        labels_list.append({
            "bboxes": np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            "labels": np.array(gt_labels, dtype=np.int64),
            "image_name": img_path.name,
        })

        # Build predictions: list of K arrays, each (M, 5)
        pred_per_class = []
        for k in range(NC):
            pred_per_class.append(np.zeros((0, 5), dtype=np.float32))

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                score = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                if cls_id < NC:
                    pred_per_class[cls_id] = np.vstack([
                        pred_per_class[cls_id],
                        [x1, y1, x2, y2, score]
                    ])

        predictions_list.append(pred_per_class)

        if (img_idx + 1) % 500 == 0:
            print(f"  Processed {img_idx + 1}/{len(img_files)} images...", flush=True)

    elapsed = time.time() - t0
    total_gt = sum(len(l["labels"]) for l in labels_list)
    total_pred = sum(sum(len(p[k]) for k in range(NC)) for p in predictions_list)
    print(f"Done: {total_gt} GT boxes, {total_pred} predictions in {elapsed:.0f}s\n")

    return labels_list, predictions_list


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO
    from cleanlab.object_detection.rank import get_label_quality_scores
    from cleanlab.object_detection.filter import find_label_issues
    from cleanlab.object_detection.summary import visualize

    if not MODEL.exists():
        print(f"ERROR: Model not found: {MODEL}")
        return 1

    model = YOLO(str(MODEL))
    class_names = load_class_names()

    print(f"Model: {MODEL.name}")
    print(f"Split: {args.split}")
    print(f"Classes: {NC}\n")

    labels, predictions = build_labels_and_predictions(
        model, args.split, args.max_images
    )

    # Get label quality scores
    print("Computing label quality scores...")
    scores = get_label_quality_scores(labels, predictions)

    # Find label issues
    print("Finding label issues...")
    issue_indices = find_label_issues(
        labels, predictions, return_indices_ranked_by_score=True
    )

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Total images: {len(labels)}")
    print(f"Images with label issues: {len(issue_indices)} ({100*len(issue_indices)/len(labels):.1f}%)")
    print(f"Score distribution:")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  <0.5:   {(scores < 0.5).sum()} images")
    print(f"  <0.3:   {(scores < 0.3).sum()} images")
    print(f"  <0.1:   {(scores < 0.1).sum()} images")

    # Save results
    out_dir = Path("reports/cleanlab")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-image scores
    results = []
    for i in range(len(labels)):
        results.append({
            "image": labels[i].get("image_name", f"img_{i}"),
            "score": round(float(scores[i]), 5),
            "has_issue": bool(i in issue_indices),
            "num_gt_boxes": len(labels[i]["labels"]),
        })
    results.sort(key=lambda x: x["score"])

    results_path = out_dir / f"label_scores_{args.split}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPer-image scores saved to {results_path}")

    # Show worst 30 images
    print(f"\nWorst 30 images (most likely mislabeled):")
    print(f"{'Rank':<5} {'Score':<8} {'GT Boxes':<10} {'Image'}")
    print("-" * 80)
    for rank, idx in enumerate(issue_indices[:30], 1):
        img_name = labels[idx].get("image_name", f"img_{idx}")
        score = scores[idx]
        n_boxes = len(labels[idx]["labels"])
        print(f"{rank:<5} {score:<8.4f} {n_boxes:<10} {img_name}")

    # Per-class analysis of worst images
    print(f"\nClass distribution in worst 100 issue images:")
    worst_100 = issue_indices[:min(100, len(issue_indices))]
    class_counts = np.zeros(NC, dtype=int)
    for idx in worst_100:
        for cls_id in labels[idx]["labels"]:
            class_counts[cls_id] += 1

    # Sort by count
    sorted_classes = np.argsort(class_counts)[::-1]
    for cls_id in sorted_classes:
        if class_counts[cls_id] == 0:
            break
        print(f"  {class_names[cls_id]:<30} {class_counts[cls_id]}")

    # Save summary
    summary = {
        "split": args.split,
        "total_images": len(labels),
        "images_with_issues": len(issue_indices),
        "issue_percentage": round(100 * len(issue_indices) / len(labels), 2),
        "score_mean": round(float(scores.mean()), 5),
        "score_median": round(float(np.median(scores)), 5),
        "score_min": round(float(scores.min()), 5),
        "worst_30": [
            {
                "rank": rank,
                "image": labels[idx].get("image_name", f"img_{idx}"),
                "score": round(float(scores[idx]), 5),
                "num_gt_boxes": len(labels[idx]["labels"]),
            }
            for rank, idx in enumerate(issue_indices[:30], 1)
        ],
    }
    summary_path = out_dir / f"audit_summary_{args.split}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
