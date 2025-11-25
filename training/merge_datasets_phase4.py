#!/usr/bin/env python3
"""
Phase 4 Dataset Merger: Combine Phase 3 + MUSCIMA++ + Rebelo

This script merges all converted datasets for Phase 4 training with
enhanced fermata, natural, and barline samples.
"""

import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path("/home/thc1006/dev/music-app/training/datasets")
PHASE3_DIR = BASE_DIR / "yolo_harmony_v2_phase3"
MUSCIMA_DIR = BASE_DIR / "yolo_muscima_converted"
REBELO_DIR = BASE_DIR / "yolo_rebelo_converted"
OUTPUT_DIR = BASE_DIR / "yolo_harmony_v2_phase4"

# Validation split ratio
VAL_RATIO = 0.1

# Class names
CLASS_NAMES = [
    "notehead_filled", "notehead_hollow", "stem", "beam",
    "flag_8th", "flag_16th", "flag_32nd", "augmentation_dot",
    "tie", "clef_treble", "clef_bass", "clef_alto", "clef_tenor",
    "accidental_sharp", "accidental_flat", "accidental_natural",
    "accidental_double_sharp", "accidental_double_flat",
    "rest_whole", "rest_half", "rest_quarter", "rest_8th", "rest_16th",
    "barline", "barline_double", "barline_final", "barline_repeat",
    "time_signature", "key_signature", "fermata",
    "dynamic_soft", "dynamic_loud", "ledger_line"
]


def copy_dataset(src_images, src_labels, dst_images, dst_labels, prefix=""):
    """Copy images and labels from source to destination."""
    copied = 0
    class_counts = defaultdict(int)

    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(src_images.glob(ext))

    for img_path in image_files:
        # Find corresponding label
        label_name = img_path.stem + '.txt'
        label_path = src_labels / label_name

        if not label_path.exists():
            continue

        # Create new filename with prefix
        if prefix:
            new_name = f"{prefix}_{img_path.name}"
            new_label = f"{prefix}_{label_name}"
        else:
            new_name = img_path.name
            new_label = label_name

        # Copy image
        dst_img = dst_images / new_name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Copy label
        dst_lbl = dst_labels / new_label
        if not dst_lbl.exists():
            shutil.copy2(label_path, dst_lbl)

        # Count classes
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(CLASS_NAMES):
                        class_counts[CLASS_NAMES[class_id]] += 1

        copied += 1

    return copied, class_counts


def split_train_val(all_images_dir, all_labels_dir, train_images, train_labels,
                   val_images, val_labels, val_ratio=0.1):
    """Split dataset into train and validation sets."""
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(all_images_dir.glob(ext))

    # Shuffle
    random.seed(42)
    random.shuffle(image_files)

    # Split
    val_count = int(len(image_files) * val_ratio)
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]

    # Move to train
    for img_path in train_files:
        label_path = all_labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            shutil.move(str(img_path), str(train_images / img_path.name))
            shutil.move(str(label_path), str(train_labels / label_path.name))

    # Move to val
    for img_path in val_files:
        label_path = all_labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            shutil.move(str(img_path), str(val_images / img_path.name))
            shutil.move(str(label_path), str(val_labels / label_path.name))

    return len(train_files), len(val_files)


def count_class_distribution(labels_dir):
    """Count class distribution in a labels directory."""
    counts = defaultdict(int)

    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(CLASS_NAMES):
                        counts[CLASS_NAMES[class_id]] += 1

    return counts


def main():
    print("=" * 60)
    print("Phase 4 Dataset Merger")
    print("=" * 60)

    # Create output directories
    temp_images = OUTPUT_DIR / "temp" / "images"
    temp_labels = OUTPUT_DIR / "temp" / "labels"
    train_images = OUTPUT_DIR / "train" / "images"
    train_labels = OUTPUT_DIR / "train" / "labels"
    val_images = OUTPUT_DIR / "val" / "images"
    val_labels = OUTPUT_DIR / "val" / "labels"

    for d in [temp_images, temp_labels, train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    total_images = 0
    all_counts = defaultdict(int)

    # 1. Copy Phase 3 data (train + val)
    print("\n[1] Copying Phase 3 data...")
    for split in ['train', 'val']:
        src_img = PHASE3_DIR / split / "images"
        src_lbl = PHASE3_DIR / split / "labels"
        if src_img.exists() and src_lbl.exists():
            count, counts = copy_dataset(src_img, src_lbl, temp_images, temp_labels, prefix="p3")
            total_images += count
            for k, v in counts.items():
                all_counts[k] += v
            print(f"  Phase 3 {split}: {count} images")

    # 2. Copy MUSCIMA++ data
    print("\n[2] Copying MUSCIMA++ data...")
    src_img = MUSCIMA_DIR / "images"
    src_lbl = MUSCIMA_DIR / "labels"
    if src_img.exists() and src_lbl.exists():
        count, counts = copy_dataset(src_img, src_lbl, temp_images, temp_labels, prefix="muscima")
        total_images += count
        for k, v in counts.items():
            all_counts[k] += v
        print(f"  MUSCIMA++: {count} images")

    # 3. Copy Rebelo data
    print("\n[3] Copying Rebelo data...")
    src_img = REBELO_DIR / "images"
    src_lbl = REBELO_DIR / "labels"
    if src_img.exists() and src_lbl.exists():
        count, counts = copy_dataset(src_img, src_lbl, temp_images, temp_labels, prefix="rebelo")
        total_images += count
        for k, v in counts.items():
            all_counts[k] += v
        print(f"  Rebelo: {count} images")

    print(f"\n  Total merged: {total_images} images")

    # 4. Split into train/val
    print("\n[4] Splitting into train/val...")
    train_count, val_count = split_train_val(
        temp_images, temp_labels,
        train_images, train_labels,
        val_images, val_labels,
        val_ratio=VAL_RATIO
    )
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")

    # Clean up temp
    shutil.rmtree(OUTPUT_DIR / "temp")

    # 5. Create YAML config
    yaml_path = OUTPUT_DIR / "harmony_phase4.yaml"
    yaml_content = f"""# Phase 4: Enhanced Fermata/Natural/Barline Dataset
# Merged from: Phase3 + MUSCIMA++ + Rebelo
# Generated: merge_datasets_phase4.py

path: {OUTPUT_DIR}
train: train/images
val: val/images
nc: 33
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # 6. Print final statistics
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)

    # Count actual distribution
    train_counts = count_class_distribution(train_labels)
    val_counts = count_class_distribution(val_labels)

    print("\n--- TARGET CLASS IMPROVEMENTS ---")
    targets = [
        (15, 'accidental_natural'),
        (29, 'fermata'),
        (23, 'barline'),
        (24, 'barline_double')
    ]
    for class_id, class_name in targets:
        train_c = train_counts.get(class_name, 0)
        val_c = val_counts.get(class_name, 0)
        total_c = train_c + val_c
        print(f"  {class_name}: {total_c} total ({train_c} train, {val_c} val)")

    print("\n--- FULL CLASS DISTRIBUTION ---")
    total_counts = defaultdict(int)
    for k, v in train_counts.items():
        total_counts[k] += v
    for k, v in val_counts.items():
        total_counts[k] += v

    for class_name, count in sorted(total_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {class_name}: {count}")

    print(f"\nDataset saved to: {OUTPUT_DIR}")
    print(f"YAML config: {yaml_path}")

    return train_count, val_count


if __name__ == "__main__":
    main()
