#!/usr/bin/env python3
"""
Phase 8 Dataset Merger
Merges Phase 7 Ultimate with new Phase 8 data:
- OpenScore Lieder (barlines, fermatas)
- OpenScore Quartets (fermatas)
- Synthetic data (double_flat, dynamic_loud)
"""

import os
import shutil
import random
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
RANDOM_SEED = 42
VAL_SPLIT = 0.1  # 10% for validation
NUM_WORKERS = 16

# Paths
BASE_DIR = Path('/home/thc1006/dev/music-app/training/datasets')
OUTPUT_DIR = BASE_DIR / 'yolo_harmony_v2_phase8_final'

# Source datasets
SOURCES = {
    'phase7': BASE_DIR / 'yolo_harmony_v2_phase7_ultimate',
    'lieder': BASE_DIR / 'yolo_openscore_lieder',
    'quartets': BASE_DIR / 'yolo_openscore_quartets',
    'synthetic': BASE_DIR / 'yolo_synthetic_phase8',
}

# Class names (33 classes)
CLASS_NAMES = [
    'notehead_filled', 'notehead_hollow', 'stem', 'beam',
    'flag_8th', 'flag_16th', 'flag_32nd', 'augmentation_dot',
    'tie', 'clef_treble', 'clef_bass', 'clef_alto',
    'clef_tenor', 'accidental_sharp', 'accidental_flat',
    'accidental_natural', 'accidental_double_sharp', 'accidental_double_flat',
    'rest_whole', 'rest_half', 'rest_quarter', 'rest_8th',
    'rest_16th', 'barline', 'barline_double', 'barline_final',
    'barline_repeat', 'time_signature', 'key_signature', 'fermata',
    'dynamic_soft', 'dynamic_loud', 'ledger_line'
]


def find_image_label_pairs(source_dir: Path, prefix: str) -> list:
    """Find all image-label pairs in a source directory."""
    pairs = []

    # Try different directory structures
    img_dirs = [
        source_dir / 'train' / 'images',
        source_dir / 'images' / 'train',
        source_dir / 'images',
    ]
    label_dirs = [
        source_dir / 'train' / 'labels',
        source_dir / 'labels' / 'train',
        source_dir / 'labels',
    ]

    img_dir = None
    label_dir = None

    for d in img_dirs:
        if d.exists():
            img_dir = d
            break

    for d in label_dirs:
        if d.exists():
            label_dir = d
            break

    if not img_dir or not label_dir:
        print(f"  ⚠️ Cannot find image/label dirs in {source_dir}")
        return pairs

    # Find pairs
    for img_path in img_dir.glob('*'):
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                # Skip empty label files
                if label_path.stat().st_size > 0:
                    pairs.append({
                        'img': img_path,
                        'label': label_path,
                        'prefix': prefix,
                        'stem': img_path.stem,
                    })

    return pairs


def copy_pair(pair: dict, output_img_dir: Path, output_label_dir: Path) -> dict:
    """Copy a single image-label pair to output directory."""
    try:
        new_name = f"{pair['prefix']}_{pair['stem']}"

        # Copy image
        img_ext = pair['img'].suffix
        dst_img = output_img_dir / f"{new_name}{img_ext}"
        shutil.copy2(pair['img'], dst_img)

        # Copy label
        dst_label = output_label_dir / f"{new_name}.txt"
        shutil.copy2(pair['label'], dst_label)

        # Count classes
        class_counts = Counter()
        with open(pair['label']) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1

        return {'success': True, 'classes': class_counts}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def main():
    random.seed(RANDOM_SEED)

    print("=" * 70)
    print("Phase 8 Dataset Merger")
    print("=" * 70)

    # Create output directories
    for split in ['train', 'val']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Collect all pairs from all sources
    all_pairs = []

    for source_name, source_path in SOURCES.items():
        print(f"\n📂 Processing {source_name}...")
        if not source_path.exists():
            print(f"  ❌ Directory not found: {source_path}")
            continue

        pairs = find_image_label_pairs(source_path, source_name)
        print(f"  Found {len(pairs):,} valid pairs")
        all_pairs.extend(pairs)

    print(f"\n📊 Total pairs collected: {len(all_pairs):,}")

    # Shuffle and split
    random.shuffle(all_pairs)

    val_size = int(len(all_pairs) * VAL_SPLIT)
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    print(f"  Training set: {len(train_pairs):,}")
    print(f"  Validation set: {len(val_pairs):,}")

    # Process each split
    total_class_counts = Counter()

    for split_name, pairs in [('train', train_pairs), ('val', val_pairs)]:
        print(f"\n🔄 Processing {split_name} split...")

        output_img_dir = OUTPUT_DIR / split_name / 'images'
        output_label_dir = OUTPUT_DIR / split_name / 'labels'

        success_count = 0
        error_count = 0
        split_class_counts = Counter()

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(copy_pair, pair, output_img_dir, output_label_dir): pair
                for pair in pairs
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {split_name}"):
                result = future.result()
                if result['success']:
                    success_count += 1
                    split_class_counts.update(result['classes'])
                else:
                    error_count += 1

        print(f"  ✅ Success: {success_count:,}")
        if error_count > 0:
            print(f"  ❌ Errors: {error_count}")

        if split_name == 'train':
            total_class_counts = split_class_counts

    # Generate YAML config
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: val/images
nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = OUTPUT_DIR / 'harmony_phase8_final.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Save merge report
    report = {
        'total_train': len(train_pairs),
        'total_val': len(val_pairs),
        'total_images': len(all_pairs),
        'class_counts': dict(total_class_counts),
        'sources': {name: str(path) for name, path in SOURCES.items()},
    }

    report_path = OUTPUT_DIR / 'merge_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print class distribution
    print("\n" + "=" * 70)
    print("Phase 8 Final Dataset - Class Distribution")
    print("=" * 70)

    for i, name in enumerate(CLASS_NAMES):
        count = total_class_counts.get(i, 0)
        bar = '█' * min(50, count // 10000)
        print(f"{i:2d}. {name:<28} {count:>10,} {bar}")

    total_bboxes = sum(total_class_counts.values())
    print(f"\n📊 Total bounding boxes: {total_bboxes:,}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📄 YAML config: {yaml_path}")

    print("\n✅ Merge complete!")


if __name__ == '__main__':
    main()
