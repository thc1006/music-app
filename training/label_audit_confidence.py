"""
label_audit_confidence.py — Cross-validate model predictions against ground-truth labels
to surface annotation errors in the training set.

Issue types detected:
  - false_negative  : Model predicts with conf > 0.5, no GT box within IoU 0.3
  - phantom_label   : GT box exists but model never predicts anything nearby (IoU > 0.3)
  - position_misalign: GT/pred match class, IoU in [0.3, 0.5)
  - class_mismatch  : GT/pred overlap IoU > 0.5 but different class index

Output:
  reports/label_audit_confidence.csv   — one row per flagged instance, sorted by severity
  reports/label_audit_summary.json     — aggregate counts per issue type per class
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Class catalogue (index 0-32)
# --------------------------------------------------------------------------- #

CLASS_NAMES: list[str] = [
    "notehead_filled",      # 0
    "notehead_hollow",      # 1
    "stem",                 # 2
    "beam",                 # 3
    "flag_8th",             # 4
    "flag_16th",            # 5
    "flag_32nd",            # 6
    "augmentation_dot",     # 7
    "tie",                  # 8
    "clef_treble",          # 9
    "clef_bass",            # 10
    "clef_alto",            # 11
    "clef_tenor",           # 12
    "accidental_sharp",     # 13
    "accidental_flat",      # 14
    "accidental_natural",   # 15
    "accidental_double_sharp",  # 16
    "accidental_double_flat",   # 17
    "rest_whole",           # 18
    "rest_half",            # 19
    "rest_quarter",         # 20
    "rest_8th",             # 21
    "rest_16th",            # 22
    "barline",              # 23
    "barline_double",       # 24
    "barline_final",        # 25
    "barline_repeat",       # 26
    "time_signature",       # 27
    "key_signature",        # 28
    "fermata",              # 29
    "dynamic_soft",         # 30
    "dynamic_loud",         # 31
    "ledger_line",          # 32
]

NUM_CLASSES = len(CLASS_NAMES)

# Severity weights per issue type (used to sort the output CSV)
SEVERITY_WEIGHT: dict[str, float] = {
    "false_negative":    0.9,
    "phantom_label":     0.8,
    "class_mismatch":    0.7,
    "position_misalign": 0.4,
}

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

class Box(NamedTuple):
    """Axis-aligned bounding box in xyxy pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float


class GTBox(NamedTuple):
    class_id: int
    box: Box


class PredBox(NamedTuple):
    class_id: int
    conf: float
    box: Box


class Finding(NamedTuple):
    image_path: str
    issue_type: str
    severity: float          # 0-1 (higher = more urgent)
    gt_class: int            # -1 if no GT involved
    pred_class: int          # -1 if no prediction involved
    pred_conf: float         # -1 if no prediction involved
    iou: float               # -1 if not applicable
    gt_box: str              # "x1,y1,x2,y2" or empty
    pred_box: str            # "x1,y1,x2,y2" or empty


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _box_area(b: Box) -> float:
    return max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-Union for two xyxy boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / (_box_area(a) + _box_area(b) - inter)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int, img_h: int) -> Box:
    """Convert YOLO normalised (cx, cy, w, h) to absolute pixel xyxy."""
    half_w = w * img_w / 2.0
    half_h = h * img_h / 2.0
    px = cx * img_w
    py = cy * img_h
    return Box(px - half_w, py - half_h, px + half_w, py + half_h)


def box_to_str(b: Box) -> str:
    return f"{b.x1:.1f},{b.y1:.1f},{b.x2:.1f},{b.y2:.1f}"


# --------------------------------------------------------------------------- #
# Label file reader
# --------------------------------------------------------------------------- #

def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[GTBox]:
    """Parse a YOLO label file into absolute xyxy GTBox instances."""
    boxes: list[GTBox] = []
    if not label_path.exists():
        return boxes
    with label_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            box = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            boxes.append(GTBox(class_id=class_id, box=box))
    return boxes


# --------------------------------------------------------------------------- #
# Per-image audit logic
# --------------------------------------------------------------------------- #

def audit_image(
    image_path: str,
    gt_boxes: list[GTBox],
    pred_boxes: list[PredBox],
    *,
    conf_threshold_fn: float = 0.50,   # false-negative detection
    iou_match: float = 0.50,           # "proper match" threshold
    iou_loose: float = 0.30,           # "nearby" threshold
    iou_misalign_low: float = 0.30,    # position misalign lower bound
    iou_misalign_high: float = 0.50,   # position misalign upper bound
) -> list[Finding]:
    """
    Cross-validate predictions against GT for one image.

    Returns a list of Finding instances describing every detected annotation
    issue.
    """
    findings: list[Finding] = []

    # Build IoU matrix: rows = GT, cols = predictions
    # Shape: (n_gt, n_pred)
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    if n_gt == 0 and n_pred == 0:
        return findings

    iou_matrix: np.ndarray
    if n_gt > 0 and n_pred > 0:
        iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)
        for gi, gt in enumerate(gt_boxes):
            for pi, pred in enumerate(pred_boxes):
                iou_matrix[gi, pi] = iou(gt.box, pred.box)
    else:
        iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

    # ---------------------------------------------------------------------- #
    # 1. False Negatives
    #    Model predicts with conf > conf_threshold_fn but no GT box within
    #    IoU >= iou_loose.
    # ---------------------------------------------------------------------- #
    for pi, pred in enumerate(pred_boxes):
        if pred.conf <= conf_threshold_fn:
            continue
        if n_gt == 0:
            best_iou = 0.0
        else:
            best_iou = float(iou_matrix[:, pi].max())
        if best_iou < iou_loose:
            severity = SEVERITY_WEIGHT["false_negative"] * pred.conf
            findings.append(Finding(
                image_path=image_path,
                issue_type="false_negative",
                severity=round(severity, 4),
                gt_class=-1,
                pred_class=pred.class_id,
                pred_conf=round(pred.conf, 4),
                iou=round(best_iou, 4),
                gt_box="",
                pred_box=box_to_str(pred.box),
            ))

    # ---------------------------------------------------------------------- #
    # 2. Phantom Labels
    #    GT box exists but the model never predicts anything nearby
    #    (no prediction with IoU >= iou_loose for that GT).
    # ---------------------------------------------------------------------- #
    for gi, gt in enumerate(gt_boxes):
        if n_pred == 0:
            best_iou = 0.0
        else:
            best_iou = float(iou_matrix[gi, :].max())
        if best_iou < iou_loose:
            severity = SEVERITY_WEIGHT["phantom_label"]
            findings.append(Finding(
                image_path=image_path,
                issue_type="phantom_label",
                severity=round(severity, 4),
                gt_class=gt.class_id,
                pred_class=-1,
                pred_conf=-1.0,
                iou=round(best_iou, 4),
                gt_box=box_to_str(gt.box),
                pred_box="",
            ))

    # ---------------------------------------------------------------------- #
    # 3. Position Misalignment  &  4. Class Mismatch
    #    For all GT/pred pairs where at least one overlaps the other:
    #
    #    Position Misalignment: same class, iou in [iou_loose, iou_match)
    #    Class Mismatch:        different classes, iou >= iou_match
    # ---------------------------------------------------------------------- #
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            ov = iou_matrix[gi, pi]

            # Position misalignment: same class, mediocre overlap
            if (gt.class_id == pred.class_id
                    and iou_misalign_low <= ov < iou_misalign_high):
                severity = SEVERITY_WEIGHT["position_misalign"] * (1.0 - ov)
                findings.append(Finding(
                    image_path=image_path,
                    issue_type="position_misalign",
                    severity=round(severity, 4),
                    gt_class=gt.class_id,
                    pred_class=pred.class_id,
                    pred_conf=round(pred.conf, 4),
                    iou=round(float(ov), 4),
                    gt_box=box_to_str(gt.box),
                    pred_box=box_to_str(pred.box),
                ))

            # Class mismatch: different classes, good overlap
            if (gt.class_id != pred.class_id and ov >= iou_match):
                severity = SEVERITY_WEIGHT["class_mismatch"] * ov
                findings.append(Finding(
                    image_path=image_path,
                    issue_type="class_mismatch",
                    severity=round(severity, 4),
                    gt_class=gt.class_id,
                    pred_class=pred.class_id,
                    pred_conf=round(pred.conf, 4),
                    iou=round(float(ov), 4),
                    gt_box=box_to_str(gt.box),
                    pred_box=box_to_str(pred.box),
                ))

    return findings


# --------------------------------------------------------------------------- #
# CSV / JSON output helpers
# --------------------------------------------------------------------------- #

CSV_FIELDS = [
    "image_path",
    "issue_type",
    "severity",
    "gt_class_id",
    "gt_class_name",
    "pred_class_id",
    "pred_class_name",
    "pred_conf",
    "iou",
    "gt_box",
    "pred_box",
]


def _class_name(class_id: int) -> str:
    if class_id < 0:
        return ""
    if class_id < NUM_CLASSES:
        return CLASS_NAMES[class_id]
    return f"unknown_{class_id}"


def finding_to_row(f: Finding) -> dict:
    return {
        "image_path":     f.image_path,
        "issue_type":     f.issue_type,
        "severity":       f.severity,
        "gt_class_id":    f.gt_class if f.gt_class >= 0 else "",
        "gt_class_name":  _class_name(f.gt_class),
        "pred_class_id":  f.pred_class if f.pred_class >= 0 else "",
        "pred_class_name": _class_name(f.pred_class),
        "pred_conf":      f.pred_conf if f.pred_conf >= 0 else "",
        "iou":            f.iou if f.iou >= 0 else "",
        "gt_box":         f.gt_box,
        "pred_box":       f.pred_box,
    }


def build_summary(all_findings: list[Finding]) -> dict:
    """Aggregate finding counts by issue_type and class name."""
    from collections import defaultdict

    summary: dict[str, dict] = {
        "total": {
            "false_negative":    0,
            "phantom_label":     0,
            "position_misalign": 0,
            "class_mismatch":    0,
        },
        "by_class": defaultdict(lambda: {
            "false_negative":    0,
            "phantom_label":     0,
            "position_misalign": 0,
            "class_mismatch":    0,
        }),
    }

    for f in all_findings:
        summary["total"][f.issue_type] = summary["total"].get(f.issue_type, 0) + 1

        # Attribute to the GT class when available, else pred class
        ref_class = f.gt_class if f.gt_class >= 0 else f.pred_class
        name = _class_name(ref_class)
        summary["by_class"][name][f.issue_type] = (
            summary["by_class"][name].get(f.issue_type, 0) + 1
        )

    # Convert defaultdict to plain dict for JSON serialisation
    summary["by_class"] = dict(summary["by_class"])
    return summary


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validate YOLO model predictions against GT labels "
                    "to find annotation errors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO model weights (.pt).",
    )
    parser.add_argument(
        "--data-dir",
        default="datasets/yolo_harmony_v2_phase8_final",
        help="Dataset root directory (must contain train/images and train/labels).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for output CSV and JSON files.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for predictions.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.55,
        help="NMS IoU threshold for inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images to pass to the model per batch.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of training images to audit. 0 = all.",
    )
    parser.add_argument(
        "--fn-conf",
        type=float,
        default=0.50,
        help="Confidence threshold above which a prediction without a matching "
             "GT box is called a False Negative annotation candidate.",
    )
    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.50,
        help="IoU threshold for a proper GT-prediction match.",
    )
    parser.add_argument(
        "--iou-loose",
        type=float,
        default=0.30,
        help="IoU threshold for 'nearby' (used by phantom_label and "
             "false_negative detection).",
    )
    return parser.parse_args(argv)


def collect_image_paths(images_dir: Path, max_images: int) -> list[Path]:
    """Return sorted list of image paths, optionally capped."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in exts
    )
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def run_inference_batch(
    model,
    image_paths: list[Path],
    imgsz: int,
    conf: float,
    iou: float,
    verbose: bool = False,
) -> list[list[PredBox]]:
    """
    Run YOLO model on a batch of image paths.

    Returns a list (one entry per image) of lists of PredBox instances in
    absolute pixel xyxy coordinates matching the *original* image dimensions.
    """
    str_paths = [str(p) for p in image_paths]
    results = model.predict(
        source=str_paths,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=verbose,
        stream=False,
    )

    batch_preds: list[list[PredBox]] = []
    for result in results:
        preds: list[PredBox] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()    # (N, 4)
            confs      = result.boxes.conf.cpu().numpy()    # (N,)
            cls_ids    = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
            for i in range(len(cls_ids)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                preds.append(PredBox(
                    class_id=int(cls_ids[i]),
                    conf=float(confs[i]),
                    box=Box(float(x1), float(y1), float(x2), float(y2)),
                ))
        batch_preds.append(preds)
    return batch_preds


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ----- Resolve paths --------------------------------------------------- #
    # Support both absolute and relative paths (relative to cwd)
    model_path  = Path(args.model)
    data_dir    = Path(args.data_dir)
    output_dir  = Path(args.output_dir)
    images_dir  = data_dir / "train" / "images"
    labels_dir  = data_dir / "train" / "labels"

    if not model_path.exists():
        logger.error("Model weights not found: %s", model_path)
        return 1
    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        return 1
    if not labels_dir.exists():
        logger.error("Labels directory not found: %s", labels_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = output_dir / "label_audit_confidence.csv"
    json_path = output_dir / "label_audit_summary.json"

    # ----- Load model ------------------------------------------------------- #
    logger.info("Loading model from %s", model_path)
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        logger.error("ultralytics is not installed. Run: pip install ultralytics")
        return 1

    model = YOLO(str(model_path))

    # ----- Collect image paths ---------------------------------------------- #
    image_paths = collect_image_paths(images_dir, args.max_images)
    total_images = len(image_paths)
    logger.info(
        "Auditing %d images  (batch=%d, imgsz=%d, conf=%.2f, iou=%.2f)",
        total_images, args.batch_size, args.imgsz, args.conf, args.iou,
    )

    # ----- Open CSV for incremental writing --------------------------------- #
    all_findings: list[Finding] = []

    with csv_path.open("w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        # Iterate over batches
        batch_size = args.batch_size
        num_batches = (total_images + batch_size - 1) // batch_size

        with tqdm(total=total_images, desc="Auditing", unit="img") as pbar:
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end   = min(start + batch_size, total_images)
                batch_paths = image_paths[start:end]

                # ---- Inference --------------------------------------------- #
                try:
                    batch_preds = run_inference_batch(
                        model=model,
                        image_paths=batch_paths,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        verbose=False,
                    )
                except Exception:
                    logger.exception(
                        "Inference failed on batch %d; skipping.", batch_idx
                    )
                    pbar.update(len(batch_paths))
                    continue

                # ---- Per-image cross-validation ---------------------------- #
                for img_path, pred_boxes in zip(batch_paths, batch_preds):
                    # Determine image dimensions for coordinate conversion
                    try:
                        from PIL import Image as PilImage  # noqa: PLC0415
                        with PilImage.open(img_path) as im:
                            img_w, img_h = im.size
                    except Exception:
                        # Fall back to a fixed size if PIL is unavailable
                        img_w, img_h = args.imgsz, args.imgsz

                    # Load GT boxes
                    label_path = labels_dir / (img_path.stem + ".txt")
                    gt_boxes   = load_gt_boxes(label_path, img_w, img_h)

                    # Run audit
                    findings = audit_image(
                        image_path=str(img_path),
                        gt_boxes=gt_boxes,
                        pred_boxes=pred_boxes,
                        conf_threshold_fn=args.fn_conf,
                        iou_match=args.iou_match,
                        iou_loose=args.iou_loose,
                        iou_misalign_low=args.iou_loose,
                        iou_misalign_high=args.iou_match,
                    )

                    # Write to CSV immediately (incremental)
                    for f in findings:
                        writer.writerow(finding_to_row(f))
                    csv_fh.flush()

                    all_findings.extend(findings)
                    pbar.update(1)

    # ----- Sort CSV by severity (descending) -------------------------------- #
    logger.info("Sorting %d findings by severity …", len(all_findings))
    all_findings.sort(key=lambda f: f.severity, reverse=True)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for f in all_findings:
            writer.writerow(finding_to_row(f))

    logger.info("CSV written to %s", csv_path)

    # ----- Summary JSON ----------------------------------------------------- #
    summary = build_summary(all_findings)
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2, ensure_ascii=False)
    logger.info("Summary JSON written to %s", json_path)

    # ----- Print top-level summary ------------------------------------------ #
    total = summary["total"]
    logger.info(
        "\n"
        "  ┌──────────────────────────────────────────┐\n"
        "  │           AUDIT SUMMARY                  │\n"
        "  ├──────────────────────────────────────────┤\n"
        "  │  false_negative    : %6d              │\n"
        "  │  phantom_label     : %6d              │\n"
        "  │  class_mismatch    : %6d              │\n"
        "  │  position_misalign : %6d              │\n"
        "  │  total             : %6d              │\n"
        "  └──────────────────────────────────────────┘",
        total.get("false_negative",    0),
        total.get("phantom_label",     0),
        total.get("class_mismatch",    0),
        total.get("position_misalign", 0),
        len(all_findings),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
