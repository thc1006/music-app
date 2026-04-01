"""
class_balanced_sampler.py
─────────────────────────
Creates a resampled training list to address extreme class imbalance in the
YOLO OMR dataset.  No image files are copied; only a text file of (possibly
repeated) image paths is produced.

Square-root resampling strategy
────────────────────────────────
  weight(class_i)  = 1 / sqrt(global_count_i)
  weight(image)    = max(weight(class) for each class that appears in the image)
  repeat_count     = round(weight(image) / min_weight)   capped at --oversample-cap

Usage
─────
  python class_balanced_sampler.py \
      --data-dir datasets/yolo_harmony_v2_phase8_final \
      --output-dir datasets/yolo_harmony_v2_phase8_cleaned \
      --oversample-cap 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Class metadata (33 classes, indices 0-32)
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES: dict[int, str] = {
    0: "notehead_filled",
    1: "notehead_hollow",
    2: "stem",
    3: "beam",
    4: "flag_8th",
    5: "flag_16th",
    6: "flag_32nd",
    7: "augmentation_dot",
    8: "tie",
    9: "clef_treble",
    10: "clef_bass",
    11: "clef_alto",
    12: "clef_tenor",
    13: "accidental_sharp",
    14: "accidental_flat",
    15: "accidental_natural",
    16: "accidental_double_sharp",
    17: "accidental_double_flat",
    18: "rest_whole",
    19: "rest_half",
    20: "rest_quarter",
    21: "rest_8th",
    22: "rest_16th",
    23: "barline",
    24: "barline_double",
    25: "barline_final",
    26: "barline_repeat",
    27: "time_signature",
    28: "key_signature",
    29: "fermata",
    30: "dynamic_soft",
    31: "dynamic_loud",
    32: "ledger_line",
}

NUM_CLASSES = 33


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
class ImageRecord(NamedTuple):
    image_path: str          # absolute path to image file
    label_path: str          # absolute path to label file
    class_set: frozenset[int]  # distinct class IDs present in this image


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – scan all label files
# ──────────────────────────────────────────────────────────────────────────────
def scan_labels(
    label_dir: Path, image_dir: Path
) -> tuple[list[ImageRecord], np.ndarray]:
    """Return list of ImageRecord and per-class global count array."""
    global_counts: np.ndarray = np.zeros(NUM_CLASSES, dtype=np.int64)
    records: list[ImageRecord] = []
    skipped = 0

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No .txt label files found under {label_dir}")

    print(f"Scanning {len(label_files):,} label files …", flush=True)

    for lbl_path in label_files:
        # Find the matching image file (support .png and .jpg)
        stem = lbl_path.stem
        img_path: Path | None = None
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            candidate = image_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        # Parse label file
        class_ids_in_image: set[int] = set()
        try:
            with lbl_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    if 0 <= class_id < NUM_CLASSES:
                        global_counts[class_id] += 1
                        class_ids_in_image.add(class_id)
        except (ValueError, IndexError) as exc:
            print(f"  [WARN] Failed to parse {lbl_path.name}: {exc}", file=sys.stderr)
            continue

        if not class_ids_in_image:
            # Background image; keep it with weight equal to minimum later
            pass

        records.append(
            ImageRecord(
                image_path=str(img_path.resolve()),
                label_path=str(lbl_path.resolve()),
                class_set=frozenset(class_ids_in_image),
            )
        )

    if skipped:
        print(
            f"  [WARN] {skipped} label files had no matching image and were skipped.",
            file=sys.stderr,
        )

    print(f"  Loaded {len(records):,} image-label pairs.", flush=True)
    return records, global_counts


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – compute per-image weights and repeat counts
# ──────────────────────────────────────────────────────────────────────────────
def compute_weights(
    records: list[ImageRecord],
    global_counts: np.ndarray,
    oversample_cap: int,
) -> list[int]:
    """Return a list of repeat counts (one per record)."""
    # Avoid division by zero for classes with 0 instances
    safe_counts = np.where(global_counts > 0, global_counts, np.inf)
    class_weights: np.ndarray = 1.0 / np.sqrt(safe_counts)  # shape (NUM_CLASSES,)

    image_weights: list[float] = []
    for rec in records:
        if rec.class_set:
            w = float(np.max(class_weights[list(rec.class_set)]))
        else:
            # Background image: assign minimum class weight
            w = float(np.min(class_weights[class_weights > 0]))
        image_weights.append(w)

    weights_arr = np.array(image_weights)
    min_w = float(weights_arr.min())

    # Repeat count = round(w / min_w), clamped to [1, oversample_cap]
    raw_repeats = np.round(weights_arr / min_w).astype(int)
    repeat_counts = np.clip(raw_repeats, 1, oversample_cap).tolist()

    return repeat_counts


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – generate balanced train.txt
# ──────────────────────────────────────────────────────────────────────────────
def write_train_txt(
    records: list[ImageRecord],
    repeat_counts: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for rec, count in zip(records, repeat_counts):
            for _ in range(count):
                fh.write(rec.image_path + "\n")
    total_lines = sum(repeat_counts)
    print(
        f"  Written {total_lines:,} lines to {output_path}  "
        f"(original: {len(records):,} images)",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – generate YAML
# ──────────────────────────────────────────────────────────────────────────────
def write_yaml(
    source_yaml: Path,
    output_dir: Path,
    balanced_txt: Path,
    original_data_path: Path,
) -> Path:
    """Write a new dataset YAML that points train to the balanced txt file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "harmony_phase8_balanced.yaml"

    # Build names block
    names_lines = "\n".join(
        f"  {idx}: {name}" for idx, name in sorted(CLASS_NAMES.items())
    )

    # val points to the original val/images directory (absolute)
    val_images = (original_data_path / "val" / "images").resolve()

    yaml_content = f"""# Harmony OMR – Phase 8 Balanced Dataset
# Generated by class_balanced_sampler.py
# Train list uses square-root resampling to reduce 329.5x class imbalance.

path: {output_dir.resolve()}
train: {balanced_txt.resolve()}   # text file with (repeated) absolute image paths
val: {val_images}

nc: {NUM_CLASSES}
names:
{names_lines}
"""
    yaml_path.write_text(yaml_content)
    print(f"  YAML written to {yaml_path}", flush=True)
    return yaml_path


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 – compute "effective samples" per class for reporting
# ──────────────────────────────────────────────────────────────────────────────
def compute_effective_counts(
    records: list[ImageRecord],
    repeat_counts: list[int],
    global_counts: np.ndarray,
) -> dict[int, int]:
    """
    Effective count for class c = sum of instances of c in each image,
    multiplied by that image's repeat count.
    """
    effective: np.ndarray = np.zeros(NUM_CLASSES, dtype=np.int64)
    for rec, cnt in zip(records, repeat_counts):
        if not rec.class_set:
            continue
        # Re-read the label to get per-class instance counts within this image.
        # We already have the label path stored, and this is a reporting pass
        # so memory is not a concern.
        pass  # handled below via a second dictionary

    # Build per-image per-class instance counts from records (class_set gives
    # distinct classes only; we need actual counts).  Since we need the actual
    # per-image per-class counts we build them during the scan phase and store
    # them here.  Because ImageRecord stores only frozenset we do a lightweight
    # re-scan of the label files – labels are small text files so this is fast.
    per_image_counts: list[dict[int, int]] = []
    for rec in records:
        counts: dict[int, int] = defaultdict(int)
        with open(rec.label_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts:
                    cid = int(parts[0])
                    if 0 <= cid < NUM_CLASSES:
                        counts[cid] += 1
        per_image_counts.append(dict(counts))

    effective = np.zeros(NUM_CLASSES, dtype=np.int64)
    for img_counts, cnt in zip(per_image_counts, repeat_counts):
        for cid, n in img_counts.items():
            effective[cid] += n * cnt

    return {i: int(effective[i]) for i in range(NUM_CLASSES)}


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 – reporting
# ──────────────────────────────────────────────────────────────────────────────
def print_report(
    global_counts: np.ndarray,
    effective_counts: dict[int, int],
    repeat_counts: list[int],
    oversample_cap: int,
) -> None:
    """Print a formatted table comparing original vs resampled distributions."""
    total_orig = int(global_counts.sum())
    total_eff = sum(effective_counts.values())

    imbalance_orig = int(global_counts.max()) / max(
        int(global_counts[global_counts > 0].min()), 1
    )
    eff_arr = np.array(
        [effective_counts[i] for i in range(NUM_CLASSES)], dtype=np.int64
    )
    present = eff_arr[eff_arr > 0]
    imbalance_eff = float(present.max()) / float(present.min()) if len(present) else 0.0

    col_w = [5, 26, 12, 12, 10, 8]
    header = (
        f"{'ID':>{col_w[0]}}  "
        f"{'Class Name':<{col_w[1]}}  "
        f"{'Original':>{col_w[2]}}  "
        f"{'Effective':>{col_w[3]}}  "
        f"{'Ratio':>{col_w[4]}}  "
        f"{'Cap?':>{col_w[5]}}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  Class Balance Report – Square-Root Resampling")
    print("=" * len(header))
    print(header)
    print(sep)

    # Sort by original count ascending to highlight tail classes first
    order = sorted(range(NUM_CLASSES), key=lambda i: global_counts[i])
    for cid in order:
        orig = int(global_counts[cid])
        eff = effective_counts.get(cid, 0)
        ratio = (eff / orig) if orig else 0.0
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        print(
            f"{cid:>{col_w[0]}}  "
            f"{name:<{col_w[1]}}  "
            f"{orig:>{col_w[2]},}  "
            f"{eff:>{col_w[3]},}  "
            f"{ratio:>{col_w[4]}.2f}x  "
            f"{'[cap]' if ratio >= oversample_cap - 0.1 else '':>{col_w[5]}}"
        )

    print(sep)
    print(
        f"{'TOTAL':>{col_w[0]}}  "
        f"{'(all classes)':<{col_w[1]}}  "
        f"{total_orig:>{col_w[2]},}  "
        f"{total_eff:>{col_w[3]},}  "
        f"{'':>{col_w[4]}}  "
    )
    print()
    print(f"  Imbalance ratio  – original : {imbalance_orig:>8.1f}x")
    print(f"  Imbalance ratio  – effective: {imbalance_eff:>8.1f}x")
    print(
        f"  Total training lines in balanced txt: {sum(repeat_counts):,}"
        f"  (original images: {len(repeat_counts):,})"
    )
    print("=" * len(header))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 – write JSON report
# ──────────────────────────────────────────────────────────────────────────────
def write_json_report(
    global_counts: np.ndarray,
    effective_counts: dict[int, int],
    repeat_counts: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eff_arr = np.array(
        [effective_counts[i] for i in range(NUM_CLASSES)], dtype=np.int64
    )
    present_orig = global_counts[global_counts > 0]
    present_eff = eff_arr[eff_arr > 0]

    per_class = {}
    for cid in range(NUM_CLASSES):
        orig = int(global_counts[cid])
        eff = effective_counts.get(cid, 0)
        per_class[CLASS_NAMES[cid]] = {
            "class_id": cid,
            "original_count": orig,
            "effective_count": eff,
            "boost_ratio": round(eff / orig, 4) if orig else None,
        }

    report = {
        "summary": {
            "total_images_original": len(repeat_counts),
            "total_lines_balanced_txt": int(sum(repeat_counts)),
            "imbalance_ratio_original": round(
                float(present_orig.max()) / float(present_orig.min()), 2
            )
            if len(present_orig)
            else None,
            "imbalance_ratio_effective": round(
                float(present_eff.max()) / float(present_eff.min()), 2
            )
            if len(present_eff)
            else None,
        },
        "per_class": per_class,
    }

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  JSON report written to {output_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a square-root resampled training list for YOLO OMR training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="datasets/yolo_harmony_v2_phase8_final",
        help="Root of the original YOLO dataset (contains train/ and val/ subdirs).",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/yolo_harmony_v2_phase8_cleaned",
        help="Directory where train_balanced.txt and the new YAML will be written.",
    )
    parser.add_argument(
        "--oversample-cap",
        type=int,
        default=5,
        help="Maximum number of times a single image may appear in the balanced list.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory for the JSON balance report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths relative to the script's parent directory so the script
    # can be run from any working directory.
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    report_dir = (script_dir / args.report_dir).resolve()

    label_dir = data_dir / "train" / "labels"
    image_dir = data_dir / "train" / "images"
    source_yaml = data_dir / "harmony_phase8_final.yaml"

    print()
    print("Class Balanced Sampler – Square-Root Resampling")
    print("=" * 55)
    print(f"  Dataset      : {data_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Oversample cap: {args.oversample_cap}x")
    print()

    # ── 1. Scan labels ────────────────────────────────────────────────────────
    print("[1/5] Scanning label files …")
    records, global_counts = scan_labels(label_dir, image_dir)

    # ── 2. Compute weights ────────────────────────────────────────────────────
    print("[2/5] Computing per-image weights …")
    repeat_counts = compute_weights(records, global_counts, args.oversample_cap)

    total_lines = sum(repeat_counts)
    print(
        f"  Repeat count distribution: min={min(repeat_counts)}, "
        f"max={max(repeat_counts)}, "
        f"mean={sum(repeat_counts)/len(repeat_counts):.2f}"
    )
    print(f"  Total training lines: {total_lines:,}")

    # ── 3. Write balanced train.txt ───────────────────────────────────────────
    print("[3/5] Writing balanced train.txt …")
    output_dir.mkdir(parents=True, exist_ok=True)
    balanced_txt = output_dir / "train_balanced.txt"
    write_train_txt(records, repeat_counts, balanced_txt)

    # ── 4. Write YAML ─────────────────────────────────────────────────────────
    print("[4/5] Writing balanced dataset YAML …")
    write_yaml(source_yaml, output_dir, balanced_txt, data_dir)

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print("[5/5] Computing effective class counts (second pass) …")
    effective_counts = compute_effective_counts(records, repeat_counts, global_counts)

    print_report(global_counts, effective_counts, repeat_counts, args.oversample_cap)

    json_report_path = report_dir / "class_balance_report.json"
    write_json_report(global_counts, effective_counts, repeat_counts, json_report_path)

    print("Done.")
    print()
    print("To use the balanced dataset in a YOLO training run:")
    print()
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolo26s.pt')")
    print("  model.train(")
    print(f"      data='{output_dir / 'harmony_phase8_balanced.yaml'}',")
    print("      epochs=100,")
    print("      imgsz=1280,")
    print("      batch=4,")
    print("  )")
    print()


if __name__ == "__main__":
    main()
