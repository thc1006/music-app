#!/usr/bin/env python3
"""
Phase 5 Dataset Merger
======================
Merges all data sources into final Phase 5 dataset:
1. Cleaned Phase 4 dataset (24,566 images, deduplicated)
2. DeepScoresV2 converted fermata data (192 images, 2,244 fermatas)
3. Synthetic fermata data (5,000 images)
4. Synthetic barline data (10,000 images)

Output: yolo_harmony_v2_phase5/
Total Expected: ~39,000+ images

Author: Music OMR Training Pipeline
Date: 2025-11-26
"""

import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random
import yaml
from typing import Dict, List, Tuple
import sys


class Phase5DatasetMerger:
    """Merge all data sources into Phase 5 final training dataset."""

    # Class ID mapping for target classes
    CLASS_MAPPING = {
        'fermata': 29,
        'barline': 23,
        'barline_double': 24,
    }

    # Complete class names (33 classes)
    CLASS_NAMES = {
        0: 'notehead_filled',
        1: 'notehead_hollow',
        2: 'stem',
        3: 'beam',
        4: 'flag_8th',
        5: 'flag_16th',
        6: 'flag_32nd',
        7: 'augmentation_dot',
        8: 'tie',
        9: 'clef_treble',
        10: 'clef_bass',
        11: 'clef_alto',
        12: 'clef_tenor',
        13: 'accidental_sharp',
        14: 'accidental_flat',
        15: 'accidental_natural',
        16: 'accidental_double_sharp',
        17: 'accidental_double_flat',
        18: 'rest_whole',
        19: 'rest_half',
        20: 'rest_quarter',
        21: 'rest_8th',
        22: 'rest_16th',
        23: 'barline',
        24: 'barline_double',
        25: 'barline_final',
        26: 'barline_repeat',
        27: 'time_signature',
        28: 'key_signature',
        29: 'fermata',
        30: 'dynamic_soft',
        31: 'dynamic_loud',
        32: 'ledger_line',
    }

    def __init__(self, base_dir: Path):
        """Initialize merger with base training directory."""
        self.base_dir = base_dir
        self.datasets_dir = base_dir / "datasets"

        # Input dataset paths
        self.phase4_dir = self.datasets_dir / "yolo_harmony_v2_phase4"
        self.deepscores_dir = self.datasets_dir / "yolo_deepscores_converted"
        self.synth_fermata_dir = self.datasets_dir / "synthetic_fermatas_v2"
        self.synth_barline_dir = self.datasets_dir / "synthetic_barlines"

        # Output dataset path
        self.output_dir = self.datasets_dir / "yolo_harmony_v2_phase5"

        # Statistics tracking
        self.stats = {
            'phase4': {'images': 0, 'labels': 0, 'annotations': Counter()},
            'deepscores': {'images': 0, 'labels': 0, 'annotations': Counter()},
            'synth_fermata': {'images': 0, 'labels': 0, 'annotations': Counter()},
            'synth_barline': {'images': 0, 'labels': 0, 'annotations': Counter()},
            'phase5_train': {'images': 0, 'labels': 0, 'annotations': Counter()},
            'phase5_val': {'images': 0, 'labels': 0, 'annotations': Counter()},
        }

        # File tracking for conflict detection
        self.file_registry = set()

    def validate_inputs(self) -> bool:
        """Validate all input directories exist."""
        print("\n" + "="*80)
        print("PHASE 5 DATASET MERGER - INPUT VALIDATION")
        print("="*80)

        required_dirs = [
            ('Phase 4 Cleaned', self.phase4_dir),
            ('DeepScoresV2 Converted', self.deepscores_dir),
            ('Synthetic Fermatas', self.synth_fermata_dir),
            ('Synthetic Barlines', self.synth_barline_dir),
        ]

        all_valid = True
        for name, path in required_dirs:
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"{status} {name}: {path}")
            if not exists:
                all_valid = False

        if not all_valid:
            print("\n❌ ERROR: Some input directories are missing!")
            return False

        print("\n✓ All input directories found")
        return True

    def create_output_structure(self):
        """Create output directory structure."""
        print("\n" + "-"*80)
        print("Creating output directory structure...")
        print("-"*80)

        # Remove existing output directory
        if self.output_dir.exists():
            print(f"Removing existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)

        # Create new structure
        directories = [
            self.output_dir / "train" / "images",
            self.output_dir / "train" / "labels",
            self.output_dir / "val" / "images",
            self.output_dir / "val" / "labels",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {dir_path}")

    def get_unique_filename(self, original_name: str, prefix: str) -> str:
        """Generate unique filename with prefix to avoid conflicts."""
        stem = Path(original_name).stem
        ext = Path(original_name).suffix

        # Try with prefix first
        candidate = f"{prefix}{stem}{ext}"

        # If still conflicts, add counter
        counter = 1
        while candidate in self.file_registry:
            candidate = f"{prefix}{stem}_{counter}{ext}"
            counter += 1

        self.file_registry.add(candidate)
        return candidate

    def copy_dataset_split(self, source_split: Path, prefix: str,
                          split_name: str, stats_key: str):
        """Copy images and labels from a dataset split."""
        source_images = source_split / "images"
        source_labels = source_split / "labels"

        if not source_images.exists():
            print(f"⚠️  No images directory in {source_split}")
            return

        # Get all image files
        image_files = list(source_images.glob("*.png")) + \
                     list(source_images.glob("*.jpg")) + \
                     list(source_images.glob("*.jpeg"))

        print(f"\n  Processing {len(image_files)} images from {source_split.name}...")

        copied_count = 0
        skipped_count = 0

        for img_path in image_files:
            # Get corresponding label file
            label_path = source_labels / f"{img_path.stem}.txt"

            if not label_path.exists():
                print(f"  ⚠️  Skipping {img_path.name} - no label file")
                skipped_count += 1
                continue

            # Generate unique filenames
            new_img_name = self.get_unique_filename(img_path.name, prefix)
            new_label_name = f"{Path(new_img_name).stem}.txt"

            # Copy image
            dest_img = self.output_dir / split_name / "images" / new_img_name
            shutil.copy2(img_path, dest_img)

            # Copy label
            dest_label = self.output_dir / split_name / "labels" / new_label_name
            shutil.copy2(label_path, dest_label)

            # Update statistics
            self.stats[stats_key]['images'] += 1
            self.stats[stats_key]['labels'] += 1

            # Count class occurrences
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        self.stats[stats_key]['annotations'][class_id] += 1

            copied_count += 1

        print(f"  ✓ Copied {copied_count} images ({skipped_count} skipped)")

    def merge_phase4_dataset(self):
        """Merge Phase 4 cleaned dataset (base dataset)."""
        print("\n" + "="*80)
        print("STEP 1: Merging Phase 4 Cleaned Dataset (Base)")
        print("="*80)

        # Copy train split
        print("\n📁 Processing train split...")
        train_dir = self.phase4_dir / "train"
        if train_dir.exists():
            self.copy_dataset_split(train_dir, "p4_", "train", "phase4")

        # Copy val split
        print("\n📁 Processing val split...")
        val_dir = self.phase4_dir / "val"
        if val_dir.exists():
            self.copy_dataset_split(val_dir, "p4_", "val", "phase4")

        print(f"\n✓ Phase 4 Merge Complete:")
        print(f"  Train: {self.stats['phase4']['images']} images")
        print(f"  Total annotations: {sum(self.stats['phase4']['annotations'].values())}")

    def merge_deepscores_dataset(self):
        """Merge DeepScoresV2 converted fermata dataset."""
        print("\n" + "="*80)
        print("STEP 2: Merging DeepScoresV2 Fermata Dataset")
        print("="*80)

        # DeepScoresV2 has train/val splits already
        train_dir = self.deepscores_dir / "images" / "train"
        val_dir = self.deepscores_dir / "images" / "val"

        if train_dir.exists():
            print("\n📁 Processing train split...")
            # Need to map to correct directory structure
            temp_train = Path("/tmp/ds2_train")
            temp_train.mkdir(exist_ok=True)
            (temp_train / "images").mkdir(exist_ok=True)
            (temp_train / "labels").mkdir(exist_ok=True)

            # Copy images and labels
            for img in train_dir.glob("*.png"):
                shutil.copy2(img, temp_train / "images" / img.name)

            labels_dir = self.deepscores_dir / "labels" / "train"
            if labels_dir.exists():
                for lbl in labels_dir.glob("*.txt"):
                    shutil.copy2(lbl, temp_train / "labels" / lbl.name)

            self.copy_dataset_split(temp_train, "ds2_", "train", "deepscores")
            shutil.rmtree(temp_train)

        if val_dir.exists():
            print("\n📁 Processing val split...")
            temp_val = Path("/tmp/ds2_val")
            temp_val.mkdir(exist_ok=True)
            (temp_val / "images").mkdir(exist_ok=True)
            (temp_val / "labels").mkdir(exist_ok=True)

            for img in val_dir.glob("*.png"):
                shutil.copy2(img, temp_val / "images" / img.name)

            labels_dir = self.deepscores_dir / "labels" / "val"
            if labels_dir.exists():
                for lbl in labels_dir.glob("*.txt"):
                    shutil.copy2(lbl, temp_val / "labels" / lbl.name)

            self.copy_dataset_split(temp_val, "ds2_", "val", "deepscores")
            shutil.rmtree(temp_val)

        print(f"\n✓ DeepScoresV2 Merge Complete:")
        print(f"  Total: {self.stats['deepscores']['images']} images")
        print(f"  Fermata annotations: {self.stats['deepscores']['annotations'][29]}")

    def merge_synthetic_dataset(self, source_dir: Path, prefix: str,
                               stats_key: str, dataset_name: str):
        """Merge synthetic dataset with 90/10 train/val split."""
        print(f"\n" + "="*80)
        print(f"STEP: Merging {dataset_name}")
        print("="*80)

        # Create temporary split structure
        temp_dir = Path(f"/tmp/{prefix}split")
        temp_dir.mkdir(exist_ok=True)

        # Get all image files
        source_images = source_dir / "images"
        source_labels = source_dir / "labels"

        if not source_images.exists() or not source_labels.exists():
            print(f"⚠️  Missing images or labels directory in {source_dir}")
            return

        image_files = list(source_images.glob("*.png")) + \
                     list(source_images.glob("*.jpg"))

        print(f"Found {len(image_files)} images")

        # Shuffle and split 90/10
        random.shuffle(image_files)
        split_idx = int(len(image_files) * 0.9)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        print(f"Split: {len(train_files)} train, {len(val_files)} val")

        # Process train split
        print("\n📁 Processing train split...")
        train_temp = temp_dir / "train"
        (train_temp / "images").mkdir(parents=True, exist_ok=True)
        (train_temp / "labels").mkdir(parents=True, exist_ok=True)

        for img_path in train_files:
            label_path = source_labels / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(img_path, train_temp / "images" / img_path.name)
                shutil.copy2(label_path, train_temp / "labels" / f"{img_path.stem}.txt")

        self.copy_dataset_split(train_temp, prefix, "train", stats_key)

        # Process val split
        print("\n📁 Processing val split...")
        val_temp = temp_dir / "val"
        (val_temp / "images").mkdir(parents=True, exist_ok=True)
        (val_temp / "labels").mkdir(parents=True, exist_ok=True)

        for img_path in val_files:
            label_path = source_labels / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(img_path, val_temp / "images" / img_path.name)
                shutil.copy2(label_path, val_temp / "labels" / f"{img_path.stem}.txt")

        self.copy_dataset_split(val_temp, prefix, "val", stats_key)

        # Cleanup
        shutil.rmtree(temp_dir)

        print(f"\n✓ {dataset_name} Merge Complete:")
        print(f"  Total: {self.stats[stats_key]['images']} images")

    def calculate_final_statistics(self):
        """Calculate final dataset statistics by counting actual files."""
        print("\n" + "="*80)
        print("Calculating Final Statistics...")
        print("="*80)

        # Count train files
        train_images = list((self.output_dir / "train" / "images").glob("*"))
        train_labels = list((self.output_dir / "train" / "labels").glob("*.txt"))

        self.stats['phase5_train']['images'] = len(train_images)
        self.stats['phase5_train']['labels'] = len(train_labels)

        # Count train annotations by class
        for label_file in train_labels:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        self.stats['phase5_train']['annotations'][class_id] += 1

        # Count val files
        val_images = list((self.output_dir / "val" / "images").glob("*"))
        val_labels = list((self.output_dir / "val" / "labels").glob("*.txt"))

        self.stats['phase5_val']['images'] = len(val_images)
        self.stats['phase5_val']['labels'] = len(val_labels)

        # Count val annotations by class
        for label_file in val_labels:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        self.stats['phase5_val']['annotations'][class_id] += 1

    def generate_data_yaml(self):
        """Generate data.yaml configuration file."""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 33,
            'names': self.CLASS_NAMES
        }

        yaml_path = self.output_dir / "harmony_phase5.yaml"

        with open(yaml_path, 'w') as f:
            # Write header comments
            f.write("# Phase 5: Final Production Dataset\n")
            f.write("# Merged from: Phase4 + DeepScoresV2 + Synthetic Fermatas + Synthetic Barlines\n")
            f.write("# Generated: merge_datasets_phase5.py\n")
            f.write(f"# Total Images: {self.stats['phase5_train']['images'] + self.stats['phase5_val']['images']}\n\n")

            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        print(f"\n✓ Generated: {yaml_path}")

    def print_statistics_report(self):
        """Print comprehensive statistics report."""
        print("\n" + "="*80)
        print("PHASE 5 DATASET MERGE - FINAL REPORT")
        print("="*80)

        # Input sources summary
        print("\n📊 INPUT SOURCES:")
        print("-" * 80)

        sources = [
            ('Phase 4 (Base)', 'phase4'),
            ('DeepScoresV2', 'deepscores'),
            ('Synthetic Fermatas', 'synth_fermata'),
            ('Synthetic Barlines', 'synth_barline'),
        ]

        for name, key in sources:
            imgs = self.stats[key]['images']
            anns = sum(self.stats[key]['annotations'].values())
            print(f"{name:25s}: {imgs:6d} images, {anns:8d} annotations")

        # Phase 5 output summary
        print("\n📦 PHASE 5 OUTPUT:")
        print("-" * 80)

        train_imgs = self.stats['phase5_train']['images']
        train_anns = sum(self.stats['phase5_train']['annotations'].values())
        val_imgs = self.stats['phase5_val']['images']
        val_anns = sum(self.stats['phase5_val']['annotations'].values())
        total_imgs = train_imgs + val_imgs
        total_anns = train_anns + val_anns

        print(f"Train Split: {train_imgs:6d} images, {train_anns:8d} annotations")
        print(f"Val Split:   {val_imgs:6d} images, {val_anns:8d} annotations")
        print(f"{'Total:':13s} {total_imgs:6d} images, {total_anns:8d} annotations")

        # Target class improvements
        print("\n🎯 TARGET CLASS IMPROVEMENTS:")
        print("-" * 80)
        print(f"{'Class':<25s} {'Train':<10s} {'Val':<10s} {'Total':<10s}")
        print("-" * 80)

        target_classes = [
            (29, 'fermata'),
            (23, 'barline'),
            (24, 'barline_double'),
        ]

        for class_id, class_name in target_classes:
            train_count = self.stats['phase5_train']['annotations'][class_id]
            val_count = self.stats['phase5_val']['annotations'][class_id]
            total_count = train_count + val_count
            print(f"{class_name:<25s} {train_count:<10d} {val_count:<10d} {total_count:<10d}")

        # Top 10 classes by annotation count
        print("\n📈 TOP 10 CLASSES BY ANNOTATION COUNT:")
        print("-" * 80)
        print(f"{'Rank':<6s} {'Class ID':<10s} {'Class Name':<25s} {'Count':<10s}")
        print("-" * 80)

        # Combine train and val annotations
        combined_annotations = Counter()
        for class_id in range(33):
            combined_annotations[class_id] = \
                self.stats['phase5_train']['annotations'][class_id] + \
                self.stats['phase5_val']['annotations'][class_id]

        for rank, (class_id, count) in enumerate(combined_annotations.most_common(10), 1):
            class_name = self.CLASS_NAMES.get(class_id, f"unknown_{class_id}")
            print(f"{rank:<6d} {class_id:<10d} {class_name:<25s} {count:<10d}")

        # Classes with zero annotations (potential issues)
        zero_classes = [
            (class_id, self.CLASS_NAMES[class_id])
            for class_id in range(33)
            if combined_annotations[class_id] == 0
        ]

        if zero_classes:
            print("\n⚠️  CLASSES WITH ZERO ANNOTATIONS:")
            print("-" * 80)
            for class_id, class_name in zero_classes:
                print(f"  Class {class_id}: {class_name}")

        # Dataset quality metrics
        print("\n📋 DATASET QUALITY METRICS:")
        print("-" * 80)

        train_ratio = train_imgs / total_imgs * 100
        val_ratio = val_imgs / total_imgs * 100
        avg_anns_per_img = total_anns / total_imgs

        print(f"Train/Val Ratio: {train_ratio:.1f}% / {val_ratio:.1f}%")
        print(f"Avg Annotations per Image: {avg_anns_per_img:.1f}")
        print(f"Total Classes: 33")
        print(f"Classes with Data: {33 - len(zero_classes)}")

        # File integrity check
        print("\n✓ FILE INTEGRITY:")
        print("-" * 80)
        print(f"All images have corresponding labels: {train_imgs == self.stats['phase5_train']['labels'] and val_imgs == self.stats['phase5_val']['labels']}")

        print("\n" + "="*80)
        print("✓ Phase 5 Dataset Merge Complete!")
        print("="*80)
        print(f"\nOutput Location: {self.output_dir}")
        print(f"YAML Config: {self.output_dir / 'harmony_phase5.yaml'}")

    def save_statistics_file(self):
        """Save detailed statistics to file."""
        stats_file = self.output_dir / "merge_statistics.txt"

        with open(stats_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PHASE 5 DATASET MERGE - DETAILED STATISTICS\n")
            f.write("="*80 + "\n\n")

            # Source breakdown
            f.write("INPUT SOURCES:\n")
            f.write("-"*80 + "\n")

            sources = [
                ('Phase 4 (Base)', 'phase4'),
                ('DeepScoresV2', 'deepscores'),
                ('Synthetic Fermatas', 'synth_fermata'),
                ('Synthetic Barlines', 'synth_barline'),
            ]

            for name, key in sources:
                f.write(f"\n{name}:\n")
                f.write(f"  Images: {self.stats[key]['images']}\n")
                f.write(f"  Labels: {self.stats[key]['labels']}\n")
                f.write(f"  Total Annotations: {sum(self.stats[key]['annotations'].values())}\n")

                if self.stats[key]['annotations']:
                    f.write("  Top Classes:\n")
                    for class_id, count in self.stats[key]['annotations'].most_common(5):
                        class_name = self.CLASS_NAMES.get(class_id, f"unknown_{class_id}")
                        f.write(f"    {class_id} ({class_name}): {count}\n")

            # Final dataset
            f.write("\n" + "="*80 + "\n")
            f.write("PHASE 5 FINAL DATASET:\n")
            f.write("="*80 + "\n\n")

            f.write("Train Split:\n")
            f.write(f"  Images: {self.stats['phase5_train']['images']}\n")
            f.write(f"  Annotations: {sum(self.stats['phase5_train']['annotations'].values())}\n")

            f.write("\nVal Split:\n")
            f.write(f"  Images: {self.stats['phase5_val']['images']}\n")
            f.write(f"  Annotations: {sum(self.stats['phase5_val']['annotations'].values())}\n")

            # Complete class distribution
            f.write("\n" + "-"*80 + "\n")
            f.write("COMPLETE CLASS DISTRIBUTION:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'ID':<5s} {'Name':<30s} {'Train':<10s} {'Val':<10s} {'Total':<10s}\n")
            f.write("-"*80 + "\n")

            for class_id in range(33):
                class_name = self.CLASS_NAMES.get(class_id, f"unknown_{class_id}")
                train_count = self.stats['phase5_train']['annotations'][class_id]
                val_count = self.stats['phase5_val']['annotations'][class_id]
                total_count = train_count + val_count
                f.write(f"{class_id:<5d} {class_name:<30s} {train_count:<10d} {val_count:<10d} {total_count:<10d}\n")

        print(f"\n✓ Saved detailed statistics: {stats_file}")

    def run(self):
        """Execute complete Phase 5 merge pipeline."""
        print("\n" + "="*80)
        print("PHASE 5 DATASET MERGER")
        print("="*80)
        print("Merging all data sources into final training dataset...")

        # Set random seed for reproducibility
        random.seed(42)

        # Step 0: Validate inputs
        if not self.validate_inputs():
            return False

        # Step 1: Create output structure
        self.create_output_structure()

        # Step 2: Merge Phase 4 (base dataset)
        self.merge_phase4_dataset()

        # Step 3: Merge DeepScoresV2
        self.merge_deepscores_dataset()

        # Step 4: Merge Synthetic Fermatas
        self.merge_synthetic_dataset(
            self.synth_fermata_dir,
            "synth_f_",
            "synth_fermata",
            "Synthetic Fermatas"
        )

        # Step 5: Merge Synthetic Barlines
        self.merge_synthetic_dataset(
            self.synth_barline_dir,
            "synth_b_",
            "synth_barline",
            "Synthetic Barlines"
        )

        # Step 6: Calculate final statistics
        self.calculate_final_statistics()

        # Step 7: Generate data.yaml
        self.generate_data_yaml()

        # Step 8: Print report
        self.print_statistics_report()

        # Step 9: Save statistics file
        self.save_statistics_file()

        return True


def main():
    """Main entry point."""
    # Set base directory
    base_dir = Path("/home/thc1006/dev/music-app/training")

    if not base_dir.exists():
        print(f"❌ ERROR: Base directory not found: {base_dir}")
        return 1

    # Create merger and run
    merger = Phase5DatasetMerger(base_dir)

    try:
        success = merger.run()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
