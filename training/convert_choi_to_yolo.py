#!/usr/bin/env python3
"""
Convert Choi Accidentals Dataset to YOLO Format

This script converts the Choi Accidentals dataset (cropped 112x112 accidental images)
to YOLO format annotations for integration with our harmony OMR training.

Dataset: 2,955 images of accidentals (flats, naturals, sharps, junk)
Source: /home/thc1006/dev/music-app/training/datasets/external/omr_downloads/ChoiAccidentals/

Output:
- Images: /home/thc1006/dev/music-app/training/datasets/external_yolo/choi/images/
- Labels: /home/thc1006/dev/music-app/training/datasets/external_yolo/choi/labels/

Class Mapping (33-class system):
- flat -> class 14 (accidental_flat)
- sharp -> class 13 (accidental_sharp)
- natural -> class 15 (accidental_natural)
- junk -> ignored (not useful for training)

Author: Claude Code
Date: 2025-11-24
"""

import os
import shutil
from pathlib import Path
from PIL import Image
from collections import Counter
import re


# Configuration
INPUT_DIR = Path("/home/thc1006/dev/music-app/training/datasets/external/omr_downloads/ChoiAccidentals/choi_accidentals_dataset")
OUTPUT_DIR = Path("/home/thc1006/dev/music-app/training/datasets/external_yolo/choi")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

# Class mapping: Choi label -> YOLO class ID (33-class system)
CLASS_MAPPING = {
    "sharp": 13,    # accidental_sharp
    "flat": 14,     # accidental_flat
    "natural": 15,  # accidental_natural
}

# Labels to ignore
IGNORE_LABELS = {"junk"}


def parse_filename(filename: str) -> dict:
    """
    Parse Choi accidentals filename to extract metadata.

    Format: {source}-{page}_{label}_{crop_bbox}_{left_bbox}_{right_bbox}.jpg
    Example: wagner-27_flat_1196x4183x1308x4295_1252x4183x1285x4261_1288x4222x1329x4256.jpg

    Returns dict with:
        - source: e.g., "wagner-27"
        - label: e.g., "flat"
        - crop_bbox: (x1, y1, x2, y2) of crop region
        - left_bbox: (x1, y1, x2, y2) of left accidental
        - right_bbox: (x1, y1, x2, y2) of right accidental
    """
    base = filename.replace(".jpg", "").replace(".png", "")
    parts = base.split("_")

    if len(parts) < 5:
        return None

    source = parts[0]
    label = parts[1]

    def parse_bbox(s):
        coords = s.split("x")
        if len(coords) == 4:
            return tuple(int(c) for c in coords)
        return None

    crop_bbox = parse_bbox(parts[2])
    left_bbox = parse_bbox(parts[3])
    right_bbox = parse_bbox(parts[4])

    return {
        "source": source,
        "label": label,
        "crop_bbox": crop_bbox,
        "left_bbox": left_bbox,
        "right_bbox": right_bbox,
    }


def compute_relative_bbox(crop_bbox, symbol_bbox, image_size):
    """
    Compute the relative bounding box of the symbol within the cropped image.

    Args:
        crop_bbox: (x1, y1, x2, y2) of the crop region in original image
        symbol_bbox: (x1, y1, x2, y2) of the symbol in original image
        image_size: (width, height) of the cropped image

    Returns:
        YOLO format: (x_center, y_center, width, height) normalized to [0, 1]
    """
    if crop_bbox is None or symbol_bbox is None:
        return None

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    sym_x1, sym_y1, sym_x2, sym_y2 = symbol_bbox

    # Handle zero bounding boxes (junk class has 0x0x0x0)
    if sym_x1 == 0 and sym_y1 == 0 and sym_x2 == 0 and sym_y2 == 0:
        return None

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w <= 0 or crop_h <= 0:
        return None

    # Compute relative position within crop
    rel_x1 = (sym_x1 - crop_x1) / crop_w
    rel_y1 = (sym_y1 - crop_y1) / crop_h
    rel_x2 = (sym_x2 - crop_x1) / crop_w
    rel_y2 = (sym_y2 - crop_y1) / crop_h

    # Clamp to [0, 1]
    rel_x1 = max(0, min(1, rel_x1))
    rel_y1 = max(0, min(1, rel_y1))
    rel_x2 = max(0, min(1, rel_x2))
    rel_y2 = max(0, min(1, rel_y2))

    # Convert to YOLO format (x_center, y_center, width, height)
    x_center = (rel_x1 + rel_x2) / 2
    y_center = (rel_y1 + rel_y2) / 2
    width = rel_x2 - rel_x1
    height = rel_y2 - rel_y1

    # Skip if bounding box is too small or invalid
    if width < 0.05 or height < 0.05:
        return None

    return (x_center, y_center, width, height)


def convert_dataset():
    """Convert Choi Accidentals dataset to YOLO format."""

    print("=" * 60)
    print("Choi Accidentals to YOLO Converter")
    print("=" * 60)

    # Create output directories
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nInput:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Statistics
    stats = {
        "total_files": 0,
        "processed": 0,
        "skipped_junk": 0,
        "skipped_invalid": 0,
        "by_class": Counter(),
        "errors": [],
    }

    # Process all files
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if not filename.endswith((".jpg", ".png")):
                continue

            stats["total_files"] += 1
            filepath = Path(root) / filename

            # Parse filename
            metadata = parse_filename(filename)
            if metadata is None:
                stats["skipped_invalid"] += 1
                stats["errors"].append(f"Invalid filename: {filename}")
                continue

            label = metadata["label"]

            # Skip junk
            if label in IGNORE_LABELS:
                stats["skipped_junk"] += 1
                continue

            # Check if label is valid
            if label not in CLASS_MAPPING:
                stats["skipped_invalid"] += 1
                stats["errors"].append(f"Unknown label '{label}' in {filename}")
                continue

            class_id = CLASS_MAPPING[label]

            # Load image to get actual dimensions
            try:
                img = Image.open(filepath)
                img_w, img_h = img.size
            except Exception as e:
                stats["skipped_invalid"] += 1
                stats["errors"].append(f"Error loading {filename}: {e}")
                continue

            # Compute bounding box
            # Use the middle/left bbox as the primary symbol location
            bbox = compute_relative_bbox(
                metadata["crop_bbox"],
                metadata["left_bbox"],
                (img_w, img_h)
            )

            if bbox is None:
                # Fallback: use a centered bounding box covering most of the image
                # Since these are already cropped accidental images, the symbol
                # typically occupies the center ~70-80% of the image
                bbox = (0.5, 0.5, 0.6, 0.8)  # Centered, 60% width, 80% height

            x_center, y_center, width, height = bbox

            # Generate unique output filename
            source = metadata["source"]
            out_name = f"choi_{source}_{label}_{stats['processed']:05d}"

            # Copy image
            out_img_path = IMAGES_DIR / f"{out_name}.jpg"
            shutil.copy2(filepath, out_img_path)

            # Write YOLO annotation
            out_label_path = LABELS_DIR / f"{out_name}.txt"
            with open(out_label_path, "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            stats["processed"] += 1
            stats["by_class"][label] += 1

    # Print statistics
    print("\n" + "=" * 60)
    print("Conversion Statistics")
    print("=" * 60)
    print(f"Total files found:     {stats['total_files']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (junk):         {stats['skipped_junk']}")
    print(f"Skipped (invalid):      {stats['skipped_invalid']}")

    print("\nClass distribution:")
    for label, count in sorted(stats["by_class"].items()):
        class_id = CLASS_MAPPING[label]
        print(f"  {label} (class {class_id}): {count}")

    if stats["errors"]:
        print(f"\nFirst 10 errors:")
        for err in stats["errors"][:10]:
            print(f"  - {err}")

    print("\n" + "=" * 60)
    print("Output Summary")
    print("=" * 60)
    print(f"Images: {IMAGES_DIR}")
    print(f"Labels: {LABELS_DIR}")
    print(f"Total annotations: {stats['processed']}")

    # Verify output
    img_count = len(list(IMAGES_DIR.glob("*.jpg")))
    label_count = len(list(LABELS_DIR.glob("*.txt")))
    print(f"\nVerification:")
    print(f"  Image files: {img_count}")
    print(f"  Label files: {label_count}")

    if img_count == label_count:
        print("  Status: OK (matched)")
    else:
        print("  Status: WARNING (mismatch)")

    return stats


def verify_sample():
    """Verify a few sample annotations."""
    print("\n" + "=" * 60)
    print("Sample Verification")
    print("=" * 60)

    label_files = list(LABELS_DIR.glob("*.txt"))[:5]
    for label_file in label_files:
        img_file = IMAGES_DIR / f"{label_file.stem}.jpg"

        print(f"\n{label_file.name}:")
        with open(label_file) as f:
            content = f.read().strip()
            print(f"  Annotation: {content}")

        if img_file.exists():
            img = Image.open(img_file)
            print(f"  Image size: {img.size}")
        else:
            print(f"  WARNING: Image not found!")


if __name__ == "__main__":
    stats = convert_dataset()
    verify_sample()

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Merge with main dataset or use as separate training source")
    print("2. Update data.yaml to include this dataset")
    print("3. Run training with augmented accidental samples")
