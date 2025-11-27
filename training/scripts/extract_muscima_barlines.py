#!/usr/bin/env python3
"""
Extract Barline Annotations from MUSCIMA++ YOLO Dataset

This script extracts barline-related annotations from the already-converted
MUSCIMA++ YOLO dataset and creates a focused barline training dataset.

Source: /home/thc1006/dev/music-app/training/datasets/yolo_muscima_converted
Target: /home/thc1006/dev/music-app/training/datasets/muscima_barlines_yolo

Barline classes from MUSCIMA++:
  - barline: 3,330 samples → class 23
  - barlineHeavy/barlineDouble: 42 samples → class 24 (barline_double)
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# Configuration
SOURCE_DIR = Path("/home/thc1006/dev/music-app/training/datasets/yolo_muscima_converted")
OUTPUT_DIR = Path("/home/thc1006/dev/music-app/training/datasets/muscima_barlines_yolo")

# YOLO class IDs for barlines (based on harmony_phase5.yaml)
BARLINE_CLASSES = {
    23: "barline",
    24: "barline_double",
    25: "barline_final",
    26: "barline_repeat",
}


def parse_yolo_label(label_line: str) -> tuple:
    """Parse a YOLO format label line."""
    parts = label_line.strip().split()
    if len(parts) >= 5:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        return class_id, x_center, y_center, width, height
    return None


def is_barline_class(class_id: int) -> bool:
    """Check if class_id is a barline-related class."""
    return class_id in BARLINE_CLASSES


def extract_barlines():
    """Extract barline annotations from MUSCIMA++ converted dataset."""

    print("=" * 70)
    print("MUSCIMA++ Barline Extraction")
    print("=" * 70)

    # Check source
    if not SOURCE_DIR.exists():
        print(f"✗ Source directory not found: {SOURCE_DIR}")
        return

    source_images = SOURCE_DIR / "images"
    source_labels = SOURCE_DIR / "labels"

    if not source_images.exists() or not source_labels.exists():
        print(f"✗ Source images or labels directory not found")
        return

    # Create output directories
    (OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)

    print(f"\nSource: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Statistics
    stats = defaultdict(int)
    images_with_barlines = 0
    total_images = 0

    # Process all label files
    label_files = list(source_labels.glob("*.txt"))
    print(f"\nProcessing {len(label_files)} label files...")

    for label_file in tqdm(label_files, desc="Extracting barlines"):
        total_images += 1

        # Read labels
        barline_labels = []

        with open(label_file, 'r') as f:
            for line in f:
                parsed = parse_yolo_label(line)
                if parsed:
                    class_id, x, y, w, h = parsed
                    if is_barline_class(class_id):
                        barline_labels.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                        stats[BARLINE_CLASSES.get(class_id, f"class_{class_id}")] += 1

        # Skip if no barlines
        if not barline_labels:
            continue

        images_with_barlines += 1

        # Find corresponding image
        image_name = label_file.stem
        image_file = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            candidate = source_images / f"{image_name}{ext}"
            if candidate.exists():
                image_file = candidate
                break

        if not image_file:
            continue

        # Determine train/val split (90/10)
        is_train = (images_with_barlines % 10) != 0
        split = "train" if is_train else "val"

        # Copy image
        dest_image = OUTPUT_DIR / split / "images" / image_file.name
        shutil.copy2(image_file, dest_image)

        # Write labels
        dest_label = OUTPUT_DIR / split / "labels" / f"{image_name}.txt"
        with open(dest_label, 'w') as f:
            f.write('\n'.join(barline_labels))

    # Print statistics
    print("\n" + "=" * 70)
    print("Extraction Complete")
    print("=" * 70)
    print(f"Total images processed: {total_images}")
    print(f"Images with barlines: {images_with_barlines}")
    print(f"Percentage: {images_with_barlines/total_images*100:.1f}%")

    print("\nBarline annotations by class:")
    for class_name in sorted(stats.keys()):
        print(f"  {class_name}: {stats[class_name]}")

    print(f"\nTotal barline annotations: {sum(stats.values())}")

    # Count final dataset
    train_images = len(list((OUTPUT_DIR / "train" / "images").glob("*")))
    val_images = len(list((OUTPUT_DIR / "val" / "images").glob("*")))
    train_labels = len(list((OUTPUT_DIR / "train" / "labels").glob("*.txt")))
    val_labels = len(list((OUTPUT_DIR / "val" / "labels").glob("*.txt")))

    print("\nFinal dataset split:")
    print(f"  Train: {train_images} images, {train_labels} labels")
    print(f"  Val:   {val_images} images, {val_labels} labels")

    # Create dataset YAML
    create_dataset_yaml(stats)

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Review the extracted dataset")
    print("2. Generate synthetic barlines to augment:")
    print("   python /home/thc1006/dev/music-app/training/scripts/generate_synthetic_barlines.py")
    print("3. Merge with Phase 5 dataset")


def create_dataset_yaml(stats: dict):
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# MUSCIMA++ Barlines Dataset (YOLO Format)
# Extracted from MUSCIMA++ v2.1
# Source: Handwritten music notation

path: {OUTPUT_DIR}
train: train/images
val: val/images

nc: 4
names:
  0: barline
  1: barline_double
  2: barline_final
  3: barline_repeat

# Class mapping (to merge with harmony_phase5.yaml):
# 23: barline
# 24: barline_double
# 25: barline_final
# 26: barline_repeat

# Statistics from MUSCIMA++:
# barline: {stats.get('barline', 0)}
# barline_double: {stats.get('barline_double', 0)}
# barline_final: {stats.get('barline_final', 0)}
# barline_repeat: {stats.get('barline_repeat', 0)}
"""

    yaml_path = OUTPUT_DIR / "muscima_barlines.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Created dataset config: {yaml_path}")


if __name__ == "__main__":
    try:
        extract_barlines()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
