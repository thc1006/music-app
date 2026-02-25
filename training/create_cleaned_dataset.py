#!/usr/bin/env python3
"""Create a cleaned dataset by removing problematic data sources.

Identified issues:
1. lieder_openscore_* (4261 train, 482 val): Only 5 classes labeled (fermata,
   barline_double, barline, barline_final, barline_repeat) while images contain
   dozens of unlabeled symbol types. 17,935 tiny 0.02x0.02 boxes.
2. phase7_phase6_base_ds2_* (161 train): Only fermata labeled.

These partially-labeled images teach the model to suppress detections of
unlabeled symbols (noteheads, stems, beams, clefs, etc.), directly causing
low recall.

Strategy: Create symlinked dataset (zero disk for images) with filtered
train/val lists and label copies excluding bad sources.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path("datasets/yolo_harmony_v2_phase8_final")
DST = Path("datasets/yolo_harmony_v2_phase8_cleaned")

# Prefixes to exclude
BAD_PREFIXES = ("lieder_", "phase7_phase6_base_ds2_")


def is_bad(filename: str) -> bool:
    return any(filename.startswith(p) for p in BAD_PREFIXES)


def process_split(split: str) -> tuple[int, int]:
    src_img = SRC / split / "images"
    src_lbl = SRC / split / "labels"
    dst_img = DST / split / "images"
    dst_lbl = DST / split / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    for img_path in sorted(src_img.iterdir()):
        if not img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            continue
        total += 1

        if is_bad(img_path.name):
            continue

        # Symlink image (zero disk)
        dst_img_path = dst_img / img_path.name
        if not dst_img_path.exists():
            dst_img_path.symlink_to(img_path.resolve())

        # Copy label file (small txt)
        lbl_path = src_lbl / (img_path.stem + ".txt")
        dst_lbl_path = dst_lbl / lbl_path.name
        if lbl_path.exists() and not dst_lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl_path)

        kept += 1

    return total, kept


def main() -> int:
    if DST.exists():
        print(f"Removing existing {DST}...")
        shutil.rmtree(DST)

    print(f"Source: {SRC}")
    print(f"Destination: {DST}")
    print(f"Excluding prefixes: {BAD_PREFIXES}\n")

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

    for split in ("train", "val"):
        total, kept = process_split(split)
        removed = total - kept
        print(f"{split}: {total} → {kept} images (removed {removed}, {100*removed/total:.1f}%)")

    # Write YAML
    yaml_path = DST / "harmony_phase8_cleaned.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nYAML: {yaml_path}")

    # Stats on labels
    for split in ("train", "val"):
        n_labels = sum(1 for _ in (DST / split / "labels").glob("*.txt"))
        n_images = sum(1 for _ in (DST / split / "images").iterdir())
        print(f"{split}: {n_images} images, {n_labels} labels")

    print("\nDone! Dataset ready for evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
