#!/usr/bin/env python3
"""
Merge Synthetic Fermata Dataset into Phase 4 Dataset
Copies synthetic fermata images and updates labels to use class ID 29.

Usage:
  python merge_synthetic_fermatas.py \
    --synthetic datasets/synthetic_fermatas_v2 \
    --target datasets/yolo_harmony_v2_phase4 \
    --split train
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import random


def parse_yolo_label(label_file: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse YOLO label file."""
    annotations = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                annotations.append((class_id, x, y, w, h))
    return annotations


def write_yolo_label(label_file: Path, annotations: List[Tuple[int, float, float, float, float]]):
    """Write YOLO label file."""
    with open(label_file, 'w') as f:
        for class_id, x, y, w, h in annotations:
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def merge_synthetic_dataset(
    synthetic_dir: Path,
    target_dir: Path,
    split: str,
    target_class_id: int = 29
):
    """
    Merge synthetic dataset into target dataset.

    Args:
        synthetic_dir: Path to synthetic dataset (contains images/ and labels/)
        target_dir: Path to target dataset (Phase 4)
        split: 'train' or 'val'
        target_class_id: Class ID for fermata in target dataset (default: 29)
    """
    synthetic_images = synthetic_dir / 'images'
    synthetic_labels = synthetic_dir / 'labels'

    target_images = target_dir / split / 'images'
    target_labels = target_dir / split / 'labels'

    # Verify directories exist
    if not synthetic_images.exists():
        print(f"Error: Synthetic images directory not found: {synthetic_images}")
        return

    if not target_images.exists():
        print(f"Error: Target images directory not found: {target_images}")
        return

    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    # Get existing image count for naming
    existing_images = list(target_images.glob('*.png')) + list(target_images.glob('*.jpg'))
    start_index = len(existing_images)

    print(f"\n{'='*60}")
    print(f"Merging Synthetic Fermatas into {split} split")
    print(f"{'='*60}")
    print(f"Synthetic directory: {synthetic_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Existing images in target: {start_index}")
    print(f"Target fermata class ID: {target_class_id}")
    print()

    # Process each synthetic sample
    synthetic_files = sorted(synthetic_images.glob('*.png'))
    copied = 0
    skipped = 0

    for i, img_file in enumerate(synthetic_files):
        label_file = synthetic_labels / f"{img_file.stem}.txt"

        if not label_file.exists():
            print(f"Warning: Label not found for {img_file.name}, skipping")
            skipped += 1
            continue

        # Read and update labels
        try:
            annotations = parse_yolo_label(label_file)

            if not annotations:
                print(f"Warning: Empty label for {img_file.name}, skipping")
                skipped += 1
                continue

            # Update class IDs from 0 to target_class_id
            updated_annotations = [
                (target_class_id, x, y, w, h)
                for class_id, x, y, w, h in annotations
            ]

            # Generate new filename
            new_index = start_index + copied
            new_name = f"synthetic_fermata_{new_index:06d}"

            # Copy image
            target_img = target_images / f"{new_name}.png"
            shutil.copy2(img_file, target_img)

            # Write updated label
            target_label = target_labels / f"{new_name}.txt"
            write_yolo_label(target_label, updated_annotations)

            copied += 1

            if (copied) % 500 == 0:
                print(f"Progress: {copied}/{len(synthetic_files)} files copied")

        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Merge complete!")
    print(f"✓ Copied: {copied} images")
    print(f"✗ Skipped: {skipped} images")
    print(f"{'='*60}\n")

    # Print statistics
    print_dataset_stats(target_dir, split)


def print_dataset_stats(dataset_dir: Path, split: str):
    """Print dataset statistics."""
    images_dir = dataset_dir / split / 'images'
    labels_dir = dataset_dir / split / 'labels'

    total_images = len(list(images_dir.glob('*.png'))) + len(list(images_dir.glob('*.jpg')))
    total_labels = len(list(labels_dir.glob('*.txt')))

    # Count fermata annotations
    fermata_count = 0
    total_annotations = 0

    for label_file in labels_dir.glob('*.txt'):
        annotations = parse_yolo_label(label_file)
        total_annotations += len(annotations)
        fermata_count += sum(1 for ann in annotations if ann[0] == 29)

    print(f"Dataset Statistics ({split}):")
    print(f"  Total images: {total_images}")
    print(f"  Total labels: {total_labels}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Fermata annotations: {fermata_count}")
    print(f"  Fermata percentage: {100*fermata_count/total_annotations:.2f}%")


def split_synthetic_dataset(
    synthetic_dir: Path,
    train_ratio: float = 0.9
):
    """
    Split synthetic dataset into train/val.
    Creates train/ and val/ subdirectories.
    """
    images_dir = synthetic_dir / 'images'
    labels_dir = synthetic_dir / 'labels'

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return

    # Get all image files
    image_files = sorted(images_dir.glob('*.png'))

    # Shuffle
    random.shuffle(image_files)

    # Split
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"\nSplitting synthetic dataset:")
    print(f"  Total: {len(image_files)}")
    print(f"  Train: {len(train_files)} ({100*train_ratio:.0f}%)")
    print(f"  Val: {len(val_files)} ({100*(1-train_ratio):.0f}%)")

    # Create directories
    for split in ['train', 'val']:
        (synthetic_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (synthetic_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Copy files
    for img_file in train_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        shutil.copy2(img_file, synthetic_dir / 'train' / 'images' / img_file.name)
        if label_file.exists():
            shutil.copy2(label_file, synthetic_dir / 'train' / 'labels' / label_file.name)

    for img_file in val_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        shutil.copy2(img_file, synthetic_dir / 'val' / 'images' / img_file.name)
        if label_file.exists():
            shutil.copy2(label_file, synthetic_dir / 'val' / 'labels' / label_file.name)

    print("✓ Split complete")


def main():
    parser = argparse.ArgumentParser(
        description='Merge synthetic fermata dataset into Phase 4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--synthetic',
        type=str,
        required=True,
        help='Path to synthetic dataset'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Path to target dataset (Phase 4)'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'both'],
        default='train',
        help='Which split to merge into'
    )
    parser.add_argument(
        '--class-id',
        type=int,
        default=29,
        help='Target class ID for fermata'
    )
    parser.add_argument(
        '--split-synthetic',
        action='store_true',
        help='Split synthetic dataset into train/val before merging'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Train/val split ratio (only with --split-synthetic)'
    )

    args = parser.parse_args()

    synthetic_dir = Path(args.synthetic)
    target_dir = Path(args.target)

    if not synthetic_dir.exists():
        print(f"Error: Synthetic directory not found: {synthetic_dir}")
        return

    if not target_dir.exists():
        print(f"Error: Target directory not found: {target_dir}")
        return

    # Split synthetic dataset if requested
    if args.split_synthetic:
        split_synthetic_dataset(synthetic_dir, args.train_ratio)
        # Update paths to use split structure
        synthetic_has_splits = True
    else:
        # Check if synthetic dataset already has train/val splits
        synthetic_has_splits = (synthetic_dir / 'train').exists()

    # Merge datasets
    if args.split == 'both':
        if not synthetic_has_splits:
            print("Error: Cannot merge 'both' without train/val splits in synthetic dataset")
            print("Use --split-synthetic to create splits first")
            return

        for split in ['train', 'val']:
            split_synthetic = synthetic_dir / split if synthetic_has_splits else synthetic_dir
            merge_synthetic_dataset(split_synthetic, target_dir, split, args.class_id)
    else:
        split_synthetic = synthetic_dir / args.split if synthetic_has_splits else synthetic_dir
        merge_synthetic_dataset(split_synthetic, target_dir, args.split, args.class_id)

    print("\n✓ All operations complete!")
    print("\nRecommended next steps:")
    print("1. Verify merged dataset integrity")
    print("2. Update training script with increased fermata class weight")
    print("3. Train Phase 5 model with merged dataset")


if __name__ == '__main__':
    main()
