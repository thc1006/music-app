#!/usr/bin/env python3
"""
Label Audit: Class Confusion Pattern Analyzer

Analyzes class confusion patterns from model predictions to identify
systematic annotation issues in the OMR dataset.

Outputs:
  - reports/label_audit_confusion_summary.csv   (top-20 confusion pairs)
  - reports/label_audit_confusion_examples.csv  (specific image examples)
  - Console summary

Usage:
    python label_audit_confusion.py \
        --model harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt \
        --data datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml \
        --output-dir reports/
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

CLASS_NAMES: list[str] = [
    "notehead_filled",        # 0
    "notehead_hollow",        # 1
    "stem",                   # 2
    "beam",                   # 3
    "flag_8th",               # 4
    "flag_16th",              # 5
    "flag_32nd",              # 6
    "augmentation_dot",       # 7
    "tie",                    # 8
    "clef_treble",            # 9
    "clef_bass",              # 10
    "clef_alto",              # 11
    "clef_tenor",             # 12
    "accidental_sharp",       # 13
    "accidental_flat",        # 14
    "accidental_natural",     # 15
    "accidental_double_sharp", # 16
    "accidental_double_flat",  # 17
    "rest_whole",             # 18
    "rest_half",              # 19
    "rest_quarter",           # 20
    "rest_8th",               # 21
    "rest_16th",              # 22
    "barline",                # 23
    "barline_double",         # 24
    "barline_final",          # 25
    "barline_repeat",         # 26
    "time_signature",         # 27
    "key_signature",          # 28
    "fermata",                # 29
    "dynamic_soft",           # 30
    "dynamic_loud",           # 31
    "ledger_line",            # 32
]

# Priority confusion pairs to highlight in the console report
PRIORITY_PAIRS: set[tuple[int, int]] = {
    (8, 8),    # tie predicted as tie (baseline sanity)
    (8, 13),   # tie confused with accidental_sharp
    (8, 14),   # tie confused with accidental_flat
    (8, 15),   # tie confused with accidental_natural
    (13, 14),  # accidental_sharp <-> flat
    (13, 15),  # accidental_sharp <-> natural
    (14, 15),  # accidental_flat <-> natural
    (16, 13),  # accidental_double_sharp <-> sharp
    (17, 14),  # accidental_double_flat <-> flat
    (24, 23),  # barline_double <-> barline
    (24, 25),  # barline_double <-> barline_final
}


class ConfusionEntry(NamedTuple):
    pred_class: int
    true_class: int
    count: int
    pred_name: str
    true_name: str


class ExampleEntry(NamedTuple):
    pred_class: int
    true_class: int
    pred_name: str
    true_name: str
    image_path: str
    pred_x1: float
    pred_y1: float
    pred_x2: float
    pred_y2: float
    gt_x1: float
    gt_y1: float
    gt_x2: float
    gt_y2: float
    iou: float
    conf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze class confusion patterns from model predictions."
    )
    parser.add_argument(
        "--model",
        default=(
            "harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt"
        ),
        help="Path to .pt model weights",
    )
    parser.add_argument(
        "--data",
        default="datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/",
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.55,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index or 'cpu'",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top confusion pairs to report",
    )
    parser.add_argument(
        "--examples-per-pair",
        type=int,
        default=10,
        help="Maximum example images per confusion pair",
    )
    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.3,
        help="IoU threshold for matching predicted box to a GT box",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=0,
        help="Limit number of val images for per-image analysis (0 = all)",
    )
    return parser.parse_args()


def class_name(idx: int) -> str:
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"class_{idx}"


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0.0 else 0.0


def load_gt_labels(
    label_path: Path,
    img_width: int,
    img_height: int,
) -> tuple[list[list[float]], list[int]]:
    """Load YOLO-format ground-truth label file and convert to xyxy pixel coords."""
    boxes: list[list[float]] = []
    classes: list[int] = []

    if not label_path.exists():
        return boxes, classes

    with label_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * img_width
            y1 = (cy - bh / 2) * img_height
            x2 = (cx + bw / 2) * img_width
            y2 = (cy + bh / 2) * img_height
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)

    return boxes, classes


def extract_confusion_matrix(
    model,
    data_yaml: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
):
    """Run model.val() and return the raw confusion matrix (numpy array).

    Ultralytics stores the confusion matrix as a (nc+1) x (nc+1) array where:
      - rows = predicted class (0..nc-1) + background row (nc)
      - cols = true class (0..nc-1) + background col (nc)
    The on-diagonal entries are true positives; off-diagonal are confusions.
    """
    print("Running model.val() to extract confusion matrix ...")
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        plots=False,
        save=False,
    )

    cm = results.confusion_matrix.matrix  # shape: (nc+1, nc+1)
    print(f"  Confusion matrix shape: {cm.shape}")
    print(f"  mAP50={results.box.map50:.4f}  mAP50-95={results.box.map:.4f}")
    print(f"  Precision={results.box.mp:.4f}  Recall={results.box.mr:.4f}")
    return cm


def get_top_confusion_pairs(
    cm,
    top_n: int,
    nc: int,
) -> list[ConfusionEntry]:
    """Extract the top-N off-diagonal confusion pairs from the confusion matrix.

    The Ultralytics confusion matrix convention (detection):
      cm[pred_idx][true_idx]  -- where pred_idx and true_idx are 0..nc-1,
      index nc represents background (FP/FN).

    We focus only on class-vs-class confusions (rows 0..nc-1, cols 0..nc-1,
    excluding the diagonal).
    """
    pairs: list[ConfusionEntry] = []

    for pred_idx in range(nc):
        for true_idx in range(nc):
            if pred_idx == true_idx:
                continue
            count = int(cm[pred_idx][true_idx])
            if count <= 0:
                continue
            pairs.append(
                ConfusionEntry(
                    pred_class=pred_idx,
                    true_class=true_idx,
                    count=count,
                    pred_name=class_name(pred_idx),
                    true_name=class_name(true_idx),
                )
            )

    pairs.sort(key=lambda e: e.count, reverse=True)
    return pairs[:top_n]


def find_confusion_examples(
    model,
    top_pairs: list[ConfusionEntry],
    val_images_dir: Path,
    val_labels_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    iou_match_threshold: float,
    examples_per_pair: int,
    max_val_images: int,
) -> list[ExampleEntry]:
    """Run model.predict on validation images and find examples of each confusion pair.

    Strategy: for each prediction that is class P, look for a GT box with class T
    that has IoU >= iou_match_threshold. If found, it is a confusion event (P, T).
    """
    # Build a lookup: {(pred_class, true_class): target example count}
    target_pairs: dict[tuple[int, int], int] = {
        (e.pred_class, e.true_class): examples_per_pair for e in top_pairs
    }
    # Storage: {(pred_class, true_class): [ExampleEntry, ...]}
    collected: dict[tuple[int, int], list[ExampleEntry]] = defaultdict(list)

    image_files = sorted(val_images_dir.glob("*.png")) + sorted(
        val_images_dir.glob("*.jpg")
    )
    if not image_files:
        image_files = sorted(val_images_dir.glob("*"))

    if max_val_images > 0:
        image_files = image_files[:max_val_images]

    total_images = len(image_files)
    print(f"\nRunning per-image prediction on {total_images} validation images ...")

    import cv2  # deferred to avoid import errors outside the venv

    still_needed = dict(target_pairs)

    for img_idx, img_path in enumerate(image_files):
        # Early exit if all pairs are saturated
        if all(v <= 0 for v in still_needed.values()):
            print(f"  All confusion pairs saturated after {img_idx} images.")
            break

        if img_idx % 500 == 0:
            print(f"  [{img_idx}/{total_images}] Processing {img_path.name} ...")

        # Read image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Load GT labels
        label_path = val_labels_dir / f"{img_path.stem}.txt"
        gt_boxes, gt_classes = load_gt_labels(label_path, img_w, img_h)
        if not gt_boxes:
            continue

        # Run prediction
        try:
            pred_results = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False,
                save=False,
            )
        except Exception as exc:
            print(f"  Warning: prediction failed for {img_path.name}: {exc}")
            continue

        if not pred_results or len(pred_results[0].boxes) == 0:
            continue

        pred_r = pred_results[0]
        pred_boxes_xyxy = pred_r.boxes.xyxy.cpu().numpy()   # (N, 4)
        pred_confs = pred_r.boxes.conf.cpu().numpy()        # (N,)
        pred_cls_ids = pred_r.boxes.cls.cpu().numpy().astype(int)  # (N,)

        # For each prediction, attempt to match against any GT box
        for p_idx in range(len(pred_boxes_xyxy)):
            pred_c = int(pred_cls_ids[p_idx])
            pred_box = pred_boxes_xyxy[p_idx].tolist()
            pred_conf = float(pred_confs[p_idx])

            # Find the GT box with maximum IoU regardless of class
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt_box in enumerate(gt_boxes):
                iou_val = box_iou(pred_box, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = g_idx

            if best_gt_idx < 0 or best_iou < iou_match_threshold:
                continue

            true_c = gt_classes[best_gt_idx]

            # Skip if this is a correct prediction
            if pred_c == true_c:
                continue

            pair_key = (pred_c, true_c)

            # Only record if this is a target pair and we still need examples
            if pair_key not in still_needed or still_needed[pair_key] <= 0:
                continue

            gt_box = gt_boxes[best_gt_idx]
            collected[pair_key].append(
                ExampleEntry(
                    pred_class=pred_c,
                    true_class=true_c,
                    pred_name=class_name(pred_c),
                    true_name=class_name(true_c),
                    image_path=str(img_path),
                    pred_x1=pred_box[0],
                    pred_y1=pred_box[1],
                    pred_x2=pred_box[2],
                    pred_y2=pred_box[3],
                    gt_x1=gt_box[0],
                    gt_y1=gt_box[1],
                    gt_x2=gt_box[2],
                    gt_y2=gt_box[3],
                    iou=best_iou,
                    conf=pred_conf,
                )
            )
            still_needed[pair_key] -= 1

    # Flatten into a single list ordered by pair occurrence in top_pairs
    examples: list[ExampleEntry] = []
    for entry in top_pairs:
        pair_key = (entry.pred_class, entry.true_class)
        examples.extend(collected.get(pair_key, []))

    return examples


def write_summary_csv(
    pairs: list[ConfusionEntry],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "rank",
                "pred_class_id",
                "pred_class_name",
                "true_class_id",
                "true_class_name",
                "count",
                "priority",
            ]
        )
        for rank, entry in enumerate(pairs, start=1):
            is_priority = (entry.pred_class, entry.true_class) in PRIORITY_PAIRS
            writer.writerow(
                [
                    rank,
                    entry.pred_class,
                    entry.pred_name,
                    entry.true_class,
                    entry.true_name,
                    entry.count,
                    "YES" if is_priority else "",
                ]
            )
    print(f"  Written: {output_path}")


def write_examples_csv(
    examples: list[ExampleEntry],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "pred_class_id",
                "pred_class_name",
                "true_class_id",
                "true_class_name",
                "image_path",
                "pred_x1",
                "pred_y1",
                "pred_x2",
                "pred_y2",
                "gt_x1",
                "gt_y1",
                "gt_x2",
                "gt_y2",
                "iou",
                "conf",
            ]
        )
        for ex in examples:
            writer.writerow(
                [
                    ex.pred_class,
                    ex.pred_name,
                    ex.true_class,
                    ex.true_name,
                    ex.image_path,
                    f"{ex.pred_x1:.2f}",
                    f"{ex.pred_y1:.2f}",
                    f"{ex.pred_x2:.2f}",
                    f"{ex.pred_y2:.2f}",
                    f"{ex.gt_x1:.2f}",
                    f"{ex.gt_y1:.2f}",
                    f"{ex.gt_x2:.2f}",
                    f"{ex.gt_y2:.2f}",
                    f"{ex.iou:.4f}",
                    f"{ex.conf:.4f}",
                ]
            )
    print(f"  Written: {output_path}")


def print_console_report(
    top_pairs: list[ConfusionEntry],
    examples: list[ExampleEntry],
    nc: int,
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("LABEL AUDIT - CLASS CONFUSION REPORT")
    print(sep)

    print(f"\nTop-{len(top_pairs)} confusion pairs (predicted -> true class):\n")
    print(f"  {'Rank':>4}  {'Predicted':25}  {'True':25}  {'Count':>7}  {'Priority'}")
    print(f"  {'-'*4}  {'-'*25}  {'-'*25}  {'-'*7}  {'-'*8}")

    for rank, entry in enumerate(top_pairs, start=1):
        is_priority = (entry.pred_class, entry.true_class) in PRIORITY_PAIRS
        flag = " <-- PRIORITY" if is_priority else ""
        print(
            f"  {rank:>4}  {entry.pred_name:25}  {entry.true_name:25}  "
            f"{entry.count:>7}{flag}"
        )

    # Per-pair example summary
    print(f"\n{sep}")
    print("EXAMPLE IMAGES PER CONFUSION PAIR")
    print(sep)

    examples_by_pair: dict[tuple[int, int], list[ExampleEntry]] = defaultdict(list)
    for ex in examples:
        examples_by_pair[(ex.pred_class, ex.true_class)].append(ex)

    for entry in top_pairs:
        pair_key = (entry.pred_class, entry.true_class)
        pair_examples = examples_by_pair.get(pair_key, [])
        print(
            f"\n  [{entry.pred_name}] predicted as [{entry.true_name}] "
            f"(confusion count={entry.count}, examples found={len(pair_examples)})"
        )
        for i, ex in enumerate(pair_examples[:5], start=1):
            img_name = Path(ex.image_path).name
            print(
                f"    {i}. {img_name}  "
                f"pred=[{ex.pred_x1:.0f},{ex.pred_y1:.0f},{ex.pred_x2:.0f},{ex.pred_y2:.0f}]  "
                f"gt=[{ex.gt_x1:.0f},{ex.gt_y1:.0f},{ex.gt_x2:.0f},{ex.gt_y2:.0f}]  "
                f"IoU={ex.iou:.3f}  conf={ex.conf:.3f}"
            )

    # Focus section: known problematic pairs
    print(f"\n{sep}")
    print("FOCUS: KNOWN PROBLEMATIC PAIRS")
    print(sep)

    # tie (8) confusions
    tie_pairs = [e for e in top_pairs if e.pred_class == 8 or e.true_class == 8]
    if tie_pairs:
        print("\n  tie (class 8) confusions:")
        for entry in tie_pairs:
            print(f"    {entry.pred_name} -> {entry.true_name}: {entry.count}")
    else:
        print("\n  tie (class 8): no confusions in top pairs")

    # accidental confusions (classes 13-17)
    accidental_ids = {13, 14, 15, 16, 17}
    acc_pairs = [
        e for e in top_pairs
        if e.pred_class in accidental_ids or e.true_class in accidental_ids
    ]
    if acc_pairs:
        print("\n  accidental (classes 13-17) confusions:")
        for entry in acc_pairs:
            print(f"    {entry.pred_name} -> {entry.true_name}: {entry.count}")
    else:
        print("\n  accidentals (classes 13-17): no confusions in top pairs")

    print(f"\n{sep}\n")


def resolve_val_dirs(data_yaml: str) -> tuple[Path, Path]:
    """Parse the dataset YAML and resolve val images/labels directories."""
    import yaml  # deferred to avoid import errors outside the venv

    yaml_path = Path(data_yaml)
    with yaml_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    dataset_root = Path(config.get("path", yaml_path.parent))
    val_images_rel = config.get("val", "val/images")
    val_images_dir = dataset_root / val_images_rel

    # Derive labels dir from images dir (standard YOLO convention)
    val_labels_dir = Path(str(val_images_dir).replace("/images", "/labels"))

    return val_images_dir, val_labels_dir


def main() -> int:
    args = parse_args()

    # Import here so the script can be imported without ultralytics installed
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ERROR: Required package not available: {exc}", file=sys.stderr)
        print("Install with: pip install ultralytics torch", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        return 1

    data_yaml = args.data
    if not Path(data_yaml).exists():
        print(f"ERROR: Dataset YAML not found: {data_yaml}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Label Audit - Class Confusion Analysis")
    print("=" * 70)
    print(f"  Model       : {model_path}")
    print(f"  Dataset     : {data_yaml}")
    print(f"  imgsz       : {args.imgsz}")
    print(f"  conf        : {args.conf}")
    print(f"  iou (NMS)   : {args.iou}")
    print(f"  iou_match   : {args.iou_match}")
    print(f"  top_n       : {args.top_n}")
    print(f"  examples/pair: {args.examples_per_pair}")
    print(f"  output_dir  : {output_dir}")
    print()

    model = YOLO(str(model_path))

    # --- Step 1: extract confusion matrix from model.val() ---
    cm = extract_confusion_matrix(
        model=model,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    nc = len(CLASS_NAMES)  # 33 classes

    # --- Step 2: identify top confusion pairs ---
    top_pairs = get_top_confusion_pairs(cm, top_n=args.top_n, nc=nc)

    if not top_pairs:
        print("No off-diagonal confusion entries found in the confusion matrix.")
        return 0

    print(f"\nFound {len(top_pairs)} confusion pairs (top-{args.top_n}).")

    # --- Step 3: find specific images for each confusion pair ---
    val_images_dir, val_labels_dir = resolve_val_dirs(data_yaml)

    if not val_images_dir.exists():
        print(
            f"WARNING: val images directory not found: {val_images_dir}\n"
            "Skipping per-image example collection.",
            file=sys.stderr,
        )
        examples: list[ExampleEntry] = []
    else:
        examples = find_confusion_examples(
            model=model,
            top_pairs=top_pairs,
            val_images_dir=val_images_dir,
            val_labels_dir=val_labels_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            iou_match_threshold=args.iou_match,
            examples_per_pair=args.examples_per_pair,
            max_val_images=args.max_val_images,
        )

    # --- Step 4: write outputs ---
    print("\nWriting output files ...")

    summary_csv = output_dir / "label_audit_confusion_summary.csv"
    examples_csv = output_dir / "label_audit_confusion_examples.csv"

    write_summary_csv(top_pairs, summary_csv)
    write_examples_csv(examples, examples_csv)

    # --- Step 5: console report ---
    print_console_report(top_pairs, examples, nc)

    # Release GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
