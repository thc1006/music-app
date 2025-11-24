#!/usr/bin/env python3
"""
Phase 3 Dataset Merger
======================
Merges all external datasets with the original Phase 2 dataset for comprehensive training.

Datasets:
1. Original Phase 2: datasets/yolo_harmony_v2_phase2/ (2,954 train images)
2. Fornes: datasets/external_yolo/fornes/ (4,094 images, includes 497 double_sharp)
3. Choi: datasets/external_yolo/choi/ (1,987 images)
4. DoReMi: datasets/yolo_doremi_converted/ (4,435 train + 783 val images)

Output: datasets/yolo_harmony_v2_phase3/
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml

# Configuration
BASE_DIR = Path("/home/thc1006/dev/music-app/training")
OUTPUT_DIR = BASE_DIR / "datasets" / "yolo_harmony_v2_phase3"
VAL_RATIO = 0.10  # 10% for validation
RANDOM_SEED = 42

# Source datasets
DATASETS = {
    "phase2_train": {
        "images": BASE_DIR / "datasets" / "yolo_harmony_v2_phase2" / "train" / "images",
        "labels": BASE_DIR / "datasets" / "yolo_harmony_v2_phase2" / "train" / "labels",
        "prefix": "p2_"
    },
    "phase2_val": {
        "images": BASE_DIR / "datasets" / "yolo_harmony_v2_phase2" / "val" / "images",
        "labels": BASE_DIR / "datasets" / "yolo_harmony_v2_phase2" / "val" / "labels",
        "prefix": "p2v_"
    },
    "fornes": {
        "images": BASE_DIR / "datasets" / "external_yolo" / "fornes" / "images",
        "labels": BASE_DIR / "datasets" / "external_yolo" / "fornes" / "labels",
        "prefix": "fornes_"
    },
    "choi": {
        "images": BASE_DIR / "datasets" / "external_yolo" / "choi" / "images",
        "labels": BASE_DIR / "datasets" / "external_yolo" / "choi" / "labels",
        "prefix": "choi_"
    },
    "doremi_train": {
        "images": BASE_DIR / "datasets" / "yolo_doremi_converted" / "train" / "images",
        "labels": BASE_DIR / "datasets" / "yolo_doremi_converted" / "train" / "labels",
        "prefix": "doremi_"
    },
    "doremi_val": {
        "images": BASE_DIR / "datasets" / "yolo_doremi_converted" / "val" / "images",
        "labels": BASE_DIR / "datasets" / "yolo_doremi_converted" / "val" / "labels",
        "prefix": "doremiv_"
    },
}

# Class names (33 classes, consistent with Phase 2)
CLASS_NAMES = [
    "notehead_filled",       # 0
    "notehead_hollow",       # 1
    "stem",                  # 2
    "beam",                  # 3
    "flag_8th",              # 4
    "flag_16th",             # 5
    "flag_32nd",             # 6
    "augmentation_dot",      # 7
    "tie",                   # 8
    "clef_treble",           # 9
    "clef_bass",             # 10
    "clef_alto",             # 11
    "clef_tenor",            # 12
    "accidental_sharp",      # 13
    "accidental_flat",       # 14
    "accidental_natural",    # 15
    "accidental_double_sharp",  # 16
    "accidental_double_flat",   # 17
    "rest_whole",            # 18
    "rest_half",             # 19
    "rest_quarter",          # 20
    "rest_8th",              # 21
    "rest_16th",             # 22
    "barline",               # 23
    "barline_double",        # 24
    "barline_final",         # 25
    "barline_repeat",        # 26
    "time_signature",        # 27
    "key_signature",         # 28
    "fermata",               # 29
    "dynamic_soft",          # 30
    "dynamic_loud",          # 31
    "ledger_line",           # 32
]


def setup_output_dirs():
    """Create output directory structure."""
    print("\n[1/5] Setting up output directories...")

    # Remove existing output directory
    if OUTPUT_DIR.exists():
        print(f"  Removing existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # Create new structure
    (OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)

    print(f"  Created: {OUTPUT_DIR}")


def collect_all_samples():
    """Collect all image-label pairs from all datasets."""
    print("\n[2/5] Collecting samples from all datasets...")

    all_samples = []  # List of (image_path, label_path, prefix)

    for name, config in DATASETS.items():
        images_dir = config["images"]
        labels_dir = config["labels"]
        prefix = config["prefix"]

        if not images_dir.exists():
            print(f"  WARNING: {name} images directory not found: {images_dir}")
            continue

        # Get all image files
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))

        count = 0
        for img_path in image_files:
            # Find corresponding label file
            label_name = img_path.stem + ".txt"
            label_path = labels_dir / label_name

            if label_path.exists():
                all_samples.append((img_path, label_path, prefix))
                count += 1

        print(f"  {name}: {count} valid samples")

    print(f"  Total samples collected: {len(all_samples)}")
    return all_samples


def analyze_class_distribution(samples):
    """Analyze class distribution across all samples."""
    print("\n[3/5] Analyzing class distribution...")

    class_counts = defaultdict(int)
    samples_per_class = defaultdict(list)  # class_id -> list of sample indices

    for idx, (img_path, label_path, prefix) in enumerate(samples):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        classes_in_sample = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id < len(CLASS_NAMES):
                    class_counts[class_id] += 1
                    classes_in_sample.add(class_id)

        for class_id in classes_in_sample:
            samples_per_class[class_id].append(idx)

    print("\n  Class distribution:")
    print(f"  {'ID':<4} {'Class Name':<25} {'Annotations':<12} {'Images':<10}")
    print(f"  {'-'*4} {'-'*25} {'-'*12} {'-'*10}")

    for class_id in range(len(CLASS_NAMES)):
        name = CLASS_NAMES[class_id]
        count = class_counts.get(class_id, 0)
        img_count = len(samples_per_class.get(class_id, []))

        # Mark rare classes
        marker = "  [RARE]" if count < 500 else ""
        print(f"  {class_id:<4} {name:<25} {count:<12} {img_count:<10}{marker}")

    return class_counts, samples_per_class


def stratified_split(samples, samples_per_class, val_ratio=0.10):
    """
    Perform stratified split ensuring rare classes are represented in validation.
    """
    print(f"\n[4/5] Performing stratified train/val split ({int((1-val_ratio)*100)}/{int(val_ratio*100)})...")

    random.seed(RANDOM_SEED)

    # Identify rare classes (less than 100 images)
    rare_classes = [c for c in range(len(CLASS_NAMES))
                    if len(samples_per_class.get(c, [])) < 100 and len(samples_per_class.get(c, [])) > 0]

    print(f"  Rare classes identified: {len(rare_classes)}")

    # Start with samples that contain rare classes
    val_indices = set()
    train_indices = set()

    # Ensure each rare class has at least some representation in validation
    for class_id in rare_classes:
        class_samples = samples_per_class.get(class_id, [])
        if len(class_samples) > 0:
            # Put at least 1 sample (or 10% of rare class) in validation
            n_val = max(1, int(len(class_samples) * val_ratio))
            random.shuffle(class_samples)
            val_indices.update(class_samples[:n_val])

    # Now handle the rest of the samples
    all_indices = set(range(len(samples)))
    remaining_indices = list(all_indices - val_indices)
    random.shuffle(remaining_indices)

    # Calculate how many more we need in validation
    target_val_size = int(len(samples) * val_ratio)
    additional_val_needed = max(0, target_val_size - len(val_indices))

    # Add more samples to validation
    val_indices.update(remaining_indices[:additional_val_needed])

    # Rest goes to training
    train_indices = all_indices - val_indices

    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples: {len(val_indices)}")

    return list(train_indices), list(val_indices)


def copy_samples(samples, indices, split_name):
    """Copy samples to the appropriate split directory."""
    print(f"\n  Copying {len(indices)} samples to {split_name}...")

    images_dir = OUTPUT_DIR / split_name / "images"
    labels_dir = OUTPUT_DIR / split_name / "labels"

    copied = 0
    skipped = 0

    for idx in indices:
        img_path, label_path, prefix = samples[idx]

        # Create unique filename with prefix
        new_img_name = f"{prefix}{img_path.name}"
        new_label_name = f"{prefix}{img_path.stem}.txt"

        # Check for conflicts
        target_img = images_dir / new_img_name
        target_label = labels_dir / new_label_name

        if target_img.exists():
            # Add counter to make unique
            counter = 1
            while target_img.exists():
                new_img_name = f"{prefix}{counter}_{img_path.name}"
                new_label_name = f"{prefix}{counter}_{img_path.stem}.txt"
                target_img = images_dir / new_img_name
                target_label = labels_dir / new_label_name
                counter += 1

        try:
            shutil.copy2(img_path, target_img)
            shutil.copy2(label_path, target_label)
            copied += 1
        except Exception as e:
            print(f"    Error copying {img_path.name}: {e}")
            skipped += 1

    print(f"    Copied: {copied}, Skipped: {skipped}")
    return copied


def create_yaml_config():
    """Create the YAML configuration file."""
    print("\n[5/5] Creating YAML configuration...")

    config = {
        "path": str(OUTPUT_DIR),
        "train": "train/images",
        "val": "val/images",
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }

    yaml_path = OUTPUT_DIR / "harmony_phase3.yaml"
    with open(yaml_path, 'w') as f:
        # Write header comment
        f.write("# Phase 3: Combined External Datasets\n")
        f.write("# Merged from: Phase2 + Fornes + Choi + DoReMi\n")
        f.write(f"# Generated: {Path(__file__).name}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Created: {yaml_path}")
    return yaml_path


def verify_output():
    """Verify the output dataset."""
    print("\n" + "="*60)
    print("OUTPUT VERIFICATION")
    print("="*60)

    train_images = list((OUTPUT_DIR / "train" / "images").glob("*"))
    train_labels = list((OUTPUT_DIR / "train" / "labels").glob("*.txt"))
    val_images = list((OUTPUT_DIR / "val" / "images").glob("*"))
    val_labels = list((OUTPUT_DIR / "val" / "labels").glob("*.txt"))

    print(f"\nTrain set:")
    print(f"  Images: {len(train_images)}")
    print(f"  Labels: {len(train_labels)}")

    print(f"\nValidation set:")
    print(f"  Images: {len(val_images)}")
    print(f"  Labels: {len(val_labels)}")

    print(f"\nTotal:")
    print(f"  Images: {len(train_images) + len(val_images)}")
    print(f"  Labels: {len(train_labels) + len(val_labels)}")

    # Check class distribution in final dataset
    print("\n" + "-"*60)
    print("Final class distribution (train + val):")
    print("-"*60)

    class_counts = defaultdict(int)
    for label_dir in [OUTPUT_DIR / "train" / "labels", OUTPUT_DIR / "val" / "labels"]:
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id < len(CLASS_NAMES):
                            class_counts[class_id] += 1

    print(f"\n{'ID':<4} {'Class Name':<25} {'Total Annotations':<20}")
    print(f"{'-'*4} {'-'*25} {'-'*20}")

    for class_id in range(len(CLASS_NAMES)):
        name = CLASS_NAMES[class_id]
        count = class_counts.get(class_id, 0)
        marker = " [RARE]" if count < 500 else ""
        print(f"{class_id:<4} {name:<25} {count:<20}{marker}")

    return len(train_images), len(val_images)


def main():
    print("="*60)
    print("PHASE 3 DATASET MERGER")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Validation ratio: {VAL_RATIO}")
    print(f"Random seed: {RANDOM_SEED}")

    # Step 1: Setup directories
    setup_output_dirs()

    # Step 2: Collect all samples
    samples = collect_all_samples()

    if not samples:
        print("\nERROR: No samples collected! Please check dataset paths.")
        return

    # Step 3: Analyze class distribution
    class_counts, samples_per_class = analyze_class_distribution(samples)

    # Step 4: Stratified split
    train_indices, val_indices = stratified_split(samples, samples_per_class, VAL_RATIO)

    # Copy files
    print("\n" + "-"*60)
    print("Copying files...")
    print("-"*60)

    copy_samples(samples, train_indices, "train")
    copy_samples(samples, val_indices, "val")

    # Step 5: Create YAML config
    yaml_path = create_yaml_config()

    # Verify output
    train_count, val_count = verify_output()

    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Config: {yaml_path}")
    print(f"Train samples: {train_count}")
    print(f"Val samples: {val_count}")
    print("\nNext steps:")
    print("  1. Review the class distribution above")
    print("  2. Run Phase 3 training:")
    print("     python yolo12_train_phase3.py")


if __name__ == "__main__":
    main()
