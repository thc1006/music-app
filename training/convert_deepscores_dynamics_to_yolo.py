#!/usr/bin/env python3
"""
Convert DeepScores V2 Dynamics to YOLO Format

DeepScores V2 annotates dynamics at the LETTER level (F, M, P, S, Z, R).
This script intelligently groups these letters into complete dynamic markings
and classifies them as dynamic_soft (class 30) or dynamic_loud (class 31).

Key Mapping:
- dynamic_soft: p, pp, ppp, mp
- dynamic_loud: f, ff, fff, mf, sf, sfz

Process:
1. Load DeepScores V2 COCO annotations
2. Group dynamics letters by proximity (same line, close x-position)
3. Combine letters into complete markings (e.g., F+F = "ff")
4. Classify and create unified bounding boxes
5. Export to YOLO format

Author: Claude Code
Date: 2025-11-27
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

# ==================== CLASS MAPPING ====================

# DeepScores V2 dynamics letter categories (deepscores only, avoid muscima++ duplicates)
DYNAMICS_LETTER_IDS = {
    '94': 'p',   # dynamicP
    '95': 'm',   # dynamicM
    '96': 'f',   # dynamicF
    '97': 's',   # dynamicS
    '98': 'z',   # dynamicZ
    '99': 'r',   # dynamicR
}

# Mapping complete dynamics to Harmony classes
DYNAMICS_TO_CLASS = {
    # Soft dynamics (class 30)
    'p': 30,
    'pp': 30,
    'ppp': 30,
    'pppp': 30,
    'mp': 30,

    # Loud dynamics (class 31)
    'f': 31,
    'ff': 31,
    'fff': 31,
    'ffff': 31,
    'mf': 31,
    'sf': 31,
    'sfz': 31,
    'sfp': 30,  # sforzando piano - soft
    'fp': 30,   # forte-piano - soft
    'fz': 31,   # forzando - loud
    'rf': 31,   # rinforzando - loud
    'rfz': 31,  # rinforzando - loud
}

# 33-class names (matching Phase 3/4)
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

# ==================== HELPER FUNCTIONS ====================

def load_deepscores_annotations(json_path):
    """Load DeepScoresV2 COCO-format annotations."""
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    images_dict = data['images'] if isinstance(data['images'], dict) else {img['id']: img for img in data['images']}
    annotations_dict = data['annotations'] if isinstance(data['annotations'], dict) else {ann['id']: ann for ann in data['annotations']}
    categories_dict = data['categories'] if isinstance(data['categories'], dict) else {cat['id']: cat['name'] for cat in data['categories']}

    print(f"  Images: {len(images_dict)}")
    print(f"  Annotations: {len(annotations_dict)}")
    print(f"  Categories: {len(categories_dict)}")

    return images_dict, annotations_dict, categories_dict


def extract_dynamics_letters(image_id, image_data, annotations_dict):
    """
    Extract all dynamics letter annotations for an image.

    Returns:
        List of dicts with: {letter, bbox, center_x, center_y}
    """
    dynamics = []

    image_annotations = image_data.get('ann_ids', [])
    if not image_annotations:
        return dynamics

    for ann_id in image_annotations:
        if ann_id not in annotations_dict:
            continue

        annotation = annotations_dict[ann_id]
        cat_ids = annotation.get('cat_id', [])

        if isinstance(cat_ids, (int, str)):
            cat_ids = [str(cat_ids)]
        else:
            cat_ids = [str(cid) for cid in cat_ids]

        # Check if this is a dynamics letter
        for cat_id in cat_ids:
            if cat_id in DYNAMICS_LETTER_IDS:
                bbox = annotation.get('a_bbox', annotation.get('bbox', None))
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    dynamics.append({
                        'letter': DYNAMICS_LETTER_IDS[cat_id],
                        'bbox': bbox,
                        'center_x': x + w / 2,
                        'center_y': y + h / 2,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    })

    return dynamics


def group_dynamics_letters(dynamics, max_distance=50, max_y_diff=20):
    """
    Group dynamics letters that are close together into complete markings.

    Args:
        dynamics: List of dynamics letter dicts
        max_distance: Max horizontal distance to group (pixels)
        max_y_diff: Max vertical difference to be on same line (pixels)

    Returns:
        List of grouped dynamics: [{letters: str, bbox: [x,y,w,h], class_id: int}]
    """
    if not dynamics:
        return []

    # Sort by center_y first (to group by line), then by center_x (reading order)
    dynamics_sorted = sorted(dynamics, key=lambda d: (d['center_y'], d['center_x']))

    groups = []
    current_group = [dynamics_sorted[0]]

    for i in range(1, len(dynamics_sorted)):
        prev = dynamics_sorted[i-1]
        curr = dynamics_sorted[i]

        # Check if on same line and close horizontally
        y_diff = abs(curr['center_y'] - prev['center_y'])
        x_diff = curr['center_x'] - prev['center_x']

        if y_diff <= max_y_diff and 0 <= x_diff <= max_distance:
            # Part of same group
            current_group.append(curr)
        else:
            # Start new group
            groups.append(current_group)
            current_group = [curr]

    # Don't forget last group
    if current_group:
        groups.append(current_group)

    # Convert groups to complete dynamics
    complete_dynamics = []

    for group in groups:
        # Combine letters
        letters = ''.join([d['letter'] for d in group])

        # Skip if not a recognized dynamic
        if letters not in DYNAMICS_TO_CLASS:
            # Try some common corrections
            if letters == 'ppp':
                letters = 'ppp'
            elif letters == 'fff':
                letters = 'fff'
            elif letters.startswith('sf') and letters not in DYNAMICS_TO_CLASS:
                # Try sfz, sfp
                if 'z' in letters:
                    letters = 'sfz'
                elif 'p' in letters:
                    letters = 'sfp'

            if letters not in DYNAMICS_TO_CLASS:
                # Skip unrecognized patterns
                continue

        # Create unified bounding box
        min_x = min(d['x'] for d in group)
        min_y = min(d['y'] for d in group)
        max_x = max(d['x'] + d['w'] for d in group)
        max_y = max(d['y'] + d['h'] for d in group)

        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        complete_dynamics.append({
            'letters': letters,
            'bbox': bbox,
            'class_id': DYNAMICS_TO_CLASS[letters]
        })

    return complete_dynamics


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert absolute bbox to YOLO normalized format."""
    x, y, w, h = bbox

    # Ensure bbox is within image bounds
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    w = max(1, min(w, img_width - x))
    h = max(1, min(h, img_height - y))

    # Convert to YOLO format
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_width = max(0.0, min(1.0, norm_width))
    norm_height = max(0.0, min(1.0, norm_height))

    return x_center, y_center, norm_width, norm_height


def process_image(image_id, image_data, annotations_dict, images_dir,
                  output_images, output_labels, split_name, stats):
    """
    Process a single image: extract dynamics, group, and convert to YOLO.

    Returns:
        (success, num_dynamics)
    """
    filename = image_data['filename']
    img_width = image_data['width']
    img_height = image_data['height']

    # Check if source image exists
    src_img_path = images_dir / filename
    if not src_img_path.exists():
        return False, 0

    # Extract dynamics letters
    dynamics_letters = extract_dynamics_letters(image_id, image_data, annotations_dict)
    if not dynamics_letters:
        return False, 0

    # Group into complete dynamics
    complete_dynamics = group_dynamics_letters(dynamics_letters)
    if not complete_dynamics:
        return False, 0

    # Convert to YOLO format
    yolo_annotations = []

    for dyn in complete_dynamics:
        x_c, y_c, w, h = convert_bbox_to_yolo(dyn['bbox'], img_width, img_height)

        yolo_annotations.append({
            'class_id': dyn['class_id'],
            'x_center': x_c,
            'y_center': y_c,
            'width': w,
            'height': h,
            'letters': dyn['letters']
        })

        # Update statistics
        class_name = CLASS_NAMES[dyn['class_id']]
        stats[class_name] += 1
        stats[f"{class_name}_detail_{dyn['letters']}"] += 1

    # Copy image
    output_name = f"ds2_dyn_{split_name}_{image_id:06d}"
    dest_img = output_images / f"{output_name}.png"

    try:
        shutil.copy2(src_img_path, dest_img)
    except Exception as e:
        print(f"  Error copying image {filename}: {e}")
        return False, 0

    # Write YOLO label
    label_path = output_labels / f"{output_name}.txt"
    with open(label_path, 'w') as f:
        for ann in yolo_annotations:
            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                   f"{ann['width']:.6f} {ann['height']:.6f}\n")

    return True, len(yolo_annotations)


def convert_split(json_path, images_dir, output_images, output_labels,
                  split_name, stats):
    """Convert a single split (train or test) of DeepScoresV2."""
    print(f"\n{'='*60}")
    print(f"Converting {split_name.upper()} split")
    print(f"{'='*60}")

    # Load annotations
    images_dict, annotations_dict, categories_dict = load_deepscores_annotations(json_path)

    # Build image-annotation index
    print("Building image-annotation index...")
    for ann_id, ann in annotations_dict.items():
        img_id = ann.get('img_id', ann.get('image_id'))
        if img_id in images_dict:
            if 'ann_ids' not in images_dict[img_id]:
                images_dict[img_id]['ann_ids'] = []
            images_dict[img_id]['ann_ids'].append(ann_id)

    # Process images
    total_images = 0
    total_dynamics = 0

    print(f"\nProcessing images (extracting dynamics)...")

    for image_id, image_data in tqdm(images_dict.items(), desc=f"{split_name}"):
        success, num_dyns = process_image(
            image_id, image_data, annotations_dict, images_dir,
            output_images, output_labels, split_name, stats
        )

        if success:
            total_images += 1
            total_dynamics += num_dyns

    return total_images, total_dynamics


def create_yaml(output_dir, train_count, val_count, stats):
    """Create YOLO dataset YAML configuration."""

    # Calculate detailed statistics
    soft_total = stats.get('dynamic_soft', 0)
    loud_total = stats.get('dynamic_loud', 0)

    # Get detailed breakdown
    soft_details = []
    loud_details = []
    for key, count in stats.items():
        if key.startswith('dynamic_soft_detail_'):
            letters = key.replace('dynamic_soft_detail_', '')
            soft_details.append(f"    - {letters}: {count}")
        elif key.startswith('dynamic_loud_detail_'):
            letters = key.replace('dynamic_loud_detail_', '')
            loud_details.append(f"    - {letters}: {count}")

    yaml_content = f"""# DeepScores V2 → YOLO Dynamics Dataset
# Converted: 2025-11-27
# Focus: Dynamics (soft/loud) extracted from letter-level annotations
# Source: DeepScores V2 train/test splits

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes (33-class Harmony OMR V2 system)
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

# Dataset statistics
# Train images: {train_count}
# Val images: {val_count}
# Total images: {train_count + val_count}

# Dynamics breakdown:
# dynamic_soft (class 30): {soft_total} annotations
{''.join(soft_details) if soft_details else '  # (none)'}

# dynamic_loud (class 31): {loud_total} annotations
{''.join(loud_details) if loud_details else '  # (none)'}
"""

    yaml_path = output_dir / "deepscores_dynamics.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset YAML created: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepScores V2 dynamics to YOLO format"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/thc1006/dev/music-app/training/datasets/external/deepscores_v2/ds2_dense"),
        help="DeepScores V2 source directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/thc1006/dev/music-app/training/datasets/yolo_deepscores_dynamics"),
        help="Output directory for YOLO format"
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=50,
        help="Max horizontal distance to group letters (pixels)"
    )
    parser.add_argument(
        "--max-y-diff",
        type=int,
        default=20,
        help="Max vertical difference to be on same line (pixels)"
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 60)
    print("DeepScores V2 → YOLO Dynamics Converter")
    print("=" * 60)
    print(f"\nSource: {args.source}")
    print(f"Output: {args.output}")
    print(f"\nGrouping parameters:")
    print(f"  Max horizontal distance: {args.max_distance}px")
    print(f"  Max vertical difference: {args.max_y_diff}px")
    print(f"\nTarget classes:")
    print(f"  dynamic_soft (30): p, pp, ppp, mp, sfp, fp")
    print(f"  dynamic_loud (31): f, ff, fff, mf, sf, sfz, fz, rf, rfz")

    # Validate source
    if not args.source.exists():
        print(f"\nERROR: Source directory not found: {args.source}")
        return 1

    train_json = args.source / "deepscores_train.json"
    test_json = args.source / "deepscores_test.json"
    images_dir = args.source / "images"

    if not train_json.exists() or not test_json.exists() or not images_dir.exists():
        print(f"\nERROR: Missing required files in source directory")
        return 1

    # Create output directories
    output_images_train = args.output / "images" / "train"
    output_images_val = args.output / "images" / "val"
    output_labels_train = args.output / "labels" / "train"
    output_labels_val = args.output / "labels" / "val"

    for dir_path in [output_images_train, output_images_val,
                     output_labels_train, output_labels_val]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = defaultdict(int)

    # Convert splits
    train_images, train_dyns = convert_split(
        train_json, images_dir, output_images_train, output_labels_train,
        "train", stats
    )

    val_images, val_dyns = convert_split(
        test_json, images_dir, output_images_val, output_labels_val,
        "val", stats
    )

    # Print final statistics
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(f"\nTrain: {train_images} images, {train_dyns} dynamics")
    print(f"Val:   {val_images} images, {val_dyns} dynamics")
    print(f"Total: {train_images + val_images} images, {train_dyns + val_dyns} dynamics")

    print("\n--- DYNAMICS CLASS COUNTS ---")
    for class_id in [30, 31]:
        class_name = CLASS_NAMES[class_id]
        count = stats.get(class_name, 0)
        print(f"  {class_name:20s} (id={class_id:2d}): {count:6d} annotations")

    print("\n--- DETAILED BREAKDOWN ---")
    for key in sorted(stats.keys()):
        if '_detail_' in key:
            parts = key.split('_detail_')
            class_name = parts[0]
            letters = parts[1]
            count = stats[key]
            print(f"  {class_name:20s} [{letters:5s}]: {count:6d}")

    # Create YAML
    create_yaml(args.output, train_images, val_images, stats)

    # Save conversion report
    report_path = args.output / "conversion_report.txt"
    with open(report_path, 'w') as f:
        f.write("DeepScores V2 → YOLO Dynamics Conversion Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: 2025-11-27\n")
        f.write(f"Source: {args.source}\n")
        f.write(f"Output: {args.output}\n\n")
        f.write(f"Train images: {train_images}\n")
        f.write(f"Val images: {val_images}\n")
        f.write(f"Total images: {train_images + val_images}\n")
        f.write(f"Total dynamics: {train_dyns + val_dyns}\n\n")
        f.write("Dynamics Class Counts:\n")
        for class_id in [30, 31]:
            class_name = CLASS_NAMES[class_id]
            count = stats.get(class_name, 0)
            f.write(f"  {class_name:20s} (id={class_id:2d}): {count:6d}\n")
        f.write("\nDetailed Breakdown:\n")
        for key in sorted(stats.keys()):
            if '_detail_' in key:
                parts = key.split('_detail_')
                class_name = parts[0]
                letters = parts[1]
                count = stats[key]
                f.write(f"  {class_name:20s} [{letters:5s}]: {count:6d}\n")

    print(f"\nConversion report saved: {report_path}")
    print(f"\nOutput directory: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
