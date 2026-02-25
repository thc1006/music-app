#!/usr/bin/env python3
"""Create a cleaned dataset v2 with additional filtering steps.

Cleaning steps:
1. Remove lieder_* and phase7_phase6_base_ds2_* (same as v1)
2. Remove quartets_* images with ≤2 annotation lines
3. Filter out tiny boxes (width AND height < 0.003) from ALL label files

Strategy: Symlinked dataset (zero disk for images) with filtered labels.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path("/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase8_final")
DST = Path("/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase8_cleaned_v2")

# Prefixes to exclude entirely
BAD_PREFIXES = ("lieder_", "phase7_phase6_base_ds2_")

# Tiny box threshold (both width AND height must be < this to be removed)
TINY_THRESHOLD = 0.003


def is_bad_prefix(filename: str) -> bool:
    return any(filename.startswith(p) for p in BAD_PREFIXES)


def count_annotations(label_path: Path) -> int:
    """Count number of annotation lines in a label file."""
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text().strip().split('\n') if line.strip())


def is_sparse_quartets(filename: str, label_path: Path) -> bool:
    """Check if this is a quartets image with ≤2 annotations."""
    if not filename.startswith("quartets_"):
        return False
    return count_annotations(label_path) <= 2


def filter_tiny_boxes(label_content: str) -> tuple[str, int]:
    """Filter out tiny boxes from label content.

    Returns: (filtered_content, num_removed)
    """
    lines = label_content.strip().split('\n')
    filtered_lines = []
    removed = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            filtered_lines.append(line)  # Keep malformed lines (let YOLO handle)
            continue

        try:
            w = float(parts[3])
            h = float(parts[4])

            # Remove if BOTH dimensions are tiny
            if w < TINY_THRESHOLD and h < TINY_THRESHOLD:
                removed += 1
            else:
                filtered_lines.append(line)
        except (ValueError, IndexError):
            filtered_lines.append(line)  # Keep if parse fails

    return '\n'.join(filtered_lines) + '\n' if filtered_lines else '', removed


def process_split(split: str) -> dict:
    src_img = SRC / split / "images"
    src_lbl = SRC / split / "labels"
    dst_img = DST / split / "images"
    dst_lbl = DST / split / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': 0,
        'kept': 0,
        'removed_bad_prefix': 0,
        'removed_sparse_quartets': 0,
        'tiny_boxes_removed': 0
    }

    for img_path in sorted(src_img.iterdir()):
        if not img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            continue
        stats['total'] += 1

        lbl_path = src_lbl / (img_path.stem + ".txt")

        # Filter 1: Bad prefixes
        if is_bad_prefix(img_path.name):
            stats['removed_bad_prefix'] += 1
            continue

        # Filter 2: Sparse quartets
        if is_sparse_quartets(img_path.name, lbl_path):
            stats['removed_sparse_quartets'] += 1
            continue

        # Symlink image
        dst_img_path = dst_img / img_path.name
        if not dst_img_path.exists():
            dst_img_path.symlink_to(img_path.resolve())

        # Copy and filter label file
        dst_lbl_path = dst_lbl / lbl_path.name
        if lbl_path.exists() and not dst_lbl_path.exists():
            label_content = lbl_path.read_text()
            filtered_content, removed = filter_tiny_boxes(label_content)
            stats['tiny_boxes_removed'] += removed
            dst_lbl_path.write_text(filtered_content)

        stats['kept'] += 1

    return stats


def main() -> int:
    if DST.exists():
        print(f"Removing existing {DST}...")
        shutil.rmtree(DST)

    print(f"Source: {SRC}")
    print(f"Destination: {DST}")
    print(f"Excluding prefixes: {BAD_PREFIXES}")
    print(f"Removing quartets_* with ≤2 annotations")
    print(f"Removing tiny boxes (w AND h < {TINY_THRESHOLD})\n")

    # Create YAML
    yaml_content = f"""path: {DST.resolve()}
train: train/images
val: val/images
nc: 33
names:
  0: notehead_filled
  1: notehead_hollow
  2: stem
  3: beam
  4: flag_8th
  5: flag_16th
  6: flag_32nd
  7: augmentation_dot
  8: tie
  9: clef_treble
  10: clef_bass
  11: clef_alto
  12: clef_tenor
  13: accidental_sharp
  14: accidental_flat
  15: accidental_natural
  16: accidental_double_sharp
  17: accidental_double_flat
  18: rest_whole
  19: rest_half
  20: rest_quarter
  21: rest_8th
  22: rest_16th
  23: barline
  24: barline_double
  25: barline_final
  26: barline_repeat
  27: time_signature
  28: key_signature
  29: fermata
  30: dynamic_soft
  31: dynamic_loud
  32: ledger_line
"""

    total_stats = {
        'total': 0,
        'kept': 0,
        'removed_bad_prefix': 0,
        'removed_sparse_quartets': 0,
        'tiny_boxes_removed': 0
    }

    for split in ("train", "val"):
        stats = process_split(split)
        for k, v in stats.items():
            total_stats[k] += v

        removed = stats['total'] - stats['kept']
        print(f"{split}:")
        print(f"  Total: {stats['total']}")
        print(f"  Kept: {stats['kept']}")
        print(f"  Removed (bad prefix): {stats['removed_bad_prefix']}")
        print(f"  Removed (sparse quartets): {stats['removed_sparse_quartets']}")
        print(f"  Tiny boxes filtered: {stats['tiny_boxes_removed']}")
        print(f"  Total removed: {removed} ({100*removed/stats['total']:.1f}%)\n")

    # Write YAML
    yaml_path = DST / "harmony_phase8_cleaned_v2.yaml"
    yaml_path.write_text(yaml_content)
    print(f"YAML: {yaml_path}")

    # Final stats
    print(f"\n{'='*60}")
    print("TOTAL DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total images processed: {total_stats['total']}")
    print(f"Images kept: {total_stats['kept']}")
    print(f"Images removed (bad prefix): {total_stats['removed_bad_prefix']}")
    print(f"Images removed (sparse quartets): {total_stats['removed_sparse_quartets']}")
    print(f"Total images removed: {total_stats['total'] - total_stats['kept']} "
          f"({100*(total_stats['total'] - total_stats['kept'])/total_stats['total']:.1f}%)")
    print(f"Total tiny boxes removed: {total_stats['tiny_boxes_removed']}")

    # Label counts
    print(f"\n{'='*60}")
    print("FINAL DATASET")
    print(f"{'='*60}")
    for split in ("train", "val"):
        n_labels = sum(1 for _ in (DST / split / "labels").glob("*.txt"))
        n_images = sum(1 for _ in (DST / split / "images").iterdir())
        print(f"{split}: {n_images} images, {n_labels} labels")

    print("\nDone! Dataset ready for training/evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
