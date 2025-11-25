#!/usr/bin/env python3
"""
Convert DeepScoresV2 dataset to YOLO format - FERMATA FOCUSED

This script extracts ONLY fermata classes from DeepScoresV2 to avoid OOM issues
from the massive dataset (175K staff instances).

⭐ Key Discovery: DeepScoresV2 has 1,712 fermata annotations (856 unique)!
   - fermataAbove: 649 instances
   - fermataBelow: 207 instances

Note: DeepScoresV2 does NOT contain barline annotations. Barlines come from
      MUSCIMA++, DoReMi, and Rebelo datasets (already converted in Phase 3/4).

DeepScoresV2 format: COCO-style JSON with oriented bounding boxes
Output: YOLO format (class_id x_center y_center width height)

Author: Claude Code
Date: 2025-11-25
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

# ==================== CLASS MAPPING ====================

# DeepScoresV2 → Harmony 33-class mapping (FERMATA ONLY)
DEEPSCORES_TO_HARMONY_FILTERED = {
    # Fermata - KEY TARGET (discovered: 1,712 annotations!)
    '81': 29,   # fermataAbove (deepscores) → fermata
    '82': 29,   # fermataBelow (deepscores) → fermata
    '181': 29,  # fermataAbove (muscima++) → fermata
    '182': 29,  # fermataBelow (muscima++) → fermata
}

# NOTE: DeepScoresV2 categories are STRING keys, not integers!

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

# Target class names for reporting
TARGET_CLASSES = {
    29: "fermata"
}

# Expected fermata counts (for validation)
EXPECTED_FERMATA_COUNTS = {
    '81': 649,   # fermataAbove (deepscores)
    '82': 207,   # fermataBelow (deepscores)
    '181': 649,  # fermataAbove (muscima++) - duplicates
    '182': 207,  # fermataBelow (muscima++) - duplicates
}

# ==================== CONVERSION FUNCTIONS ====================

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert DeepScores axis-aligned bounding box to YOLO format.

    Args:
        bbox: [x, y, w, h] in absolute pixels
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    x, y, w, h = bbox

    # Ensure bbox is within image bounds
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    w = max(1, min(w, img_width - x))
    h = max(1, min(h, img_height - y))

    # Convert to YOLO format (normalized center coordinates)
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


def load_deepscores_annotations(json_path):
    """
    Load DeepScoresV2 COCO-format annotations.

    Returns:
        (images_dict, annotations_dict, categories_dict)
    """
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build lookup dictionaries
    # Note: DeepScoresV2 stores images and annotations as dicts with IDs as keys
    images_dict = data['images'] if isinstance(data['images'], dict) else {img['id']: img for img in data['images']}
    annotations_dict = data['annotations'] if isinstance(data['annotations'], dict) else {ann['id']: ann for ann in data['annotations']}
    categories_dict = data['categories'] if isinstance(data['categories'], dict) else {cat['id']: cat['name'] for cat in data['categories']}

    print(f"  Images: {len(images_dict)}")
    print(f"  Annotations: {len(annotations_dict)}")
    print(f"  Categories: {len(categories_dict)}")

    return images_dict, annotations_dict, categories_dict


def process_image(image_id, image_data, annotations_dict, images_dir,
                  output_images, output_labels, split_name, stats):
    """
    Process a single image: filter target annotations and convert to YOLO.

    Returns:
        (success, num_annotations)
    """
    filename = image_data['filename']
    img_width = image_data['width']
    img_height = image_data['height']

    # Check if source image exists
    src_img_path = images_dir / filename
    if not src_img_path.exists():
        return False, 0

    # Get annotations for this image
    image_annotations = image_data.get('ann_ids', [])
    if not image_annotations:
        return False, 0

    # Filter and convert annotations
    yolo_annotations = []

    for ann_id in image_annotations:
        if ann_id not in annotations_dict:
            continue

        annotation = annotations_dict[ann_id]

        # Get category IDs (DeepScores may have multiple categories per annotation)
        cat_ids = annotation.get('cat_id', [])
        if isinstance(cat_ids, int):
            cat_ids = [cat_ids]

        # Check if any category is in our target list
        for cat_id in cat_ids:
            if cat_id in DEEPSCORES_TO_HARMONY_FILTERED:
                harmony_class_id = DEEPSCORES_TO_HARMONY_FILTERED[cat_id]

                # Get bounding box (DeepScores uses 'a_bbox' for axis-aligned bbox)
                bbox = annotation.get('a_bbox', annotation.get('bbox', None))
                if bbox is None or len(bbox) != 4:
                    continue

                # Convert to YOLO format
                x_c, y_c, w, h = convert_bbox_to_yolo(bbox, img_width, img_height)

                yolo_annotations.append({
                    'class_id': harmony_class_id,
                    'x_center': x_c,
                    'y_center': y_c,
                    'width': w,
                    'height': h,
                    'deepscores_cat': cat_id
                })

                # Update statistics
                stats[CLASS_NAMES[harmony_class_id]] += 1

    # Skip images with no target annotations
    if not yolo_annotations:
        return False, 0

    # Copy image to output directory
    output_name = f"ds2_{split_name}_{image_id:06d}"
    dest_img = output_images / f"{output_name}.png"

    try:
        # Copy or convert image
        shutil.copy2(src_img_path, dest_img)
    except Exception as e:
        print(f"  Error copying image {filename}: {e}")
        return False, 0

    # Write YOLO label file
    label_path = output_labels / f"{output_name}.txt"
    with open(label_path, 'w') as f:
        for ann in yolo_annotations:
            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                   f"{ann['width']:.6f} {ann['height']:.6f}\n")

    return True, len(yolo_annotations)


def convert_split(json_path, images_dir, output_images, output_labels,
                  split_name, stats):
    """
    Convert a single split (train or test) of DeepScoresV2.

    Returns:
        (total_images, total_annotations)
    """
    print(f"\n{'='*60}")
    print(f"Converting {split_name.upper()} split")
    print(f"{'='*60}")

    # Load annotations
    images_dict, annotations_dict, categories_dict = load_deepscores_annotations(json_path)

    # Build image-to-annotations mapping
    print("Building image-annotation index...")
    for ann_id, ann in annotations_dict.items():
        img_id = ann.get('img_id', ann.get('image_id'))  # Try both keys
        if img_id in images_dict:
            if 'ann_ids' not in images_dict[img_id]:
                images_dict[img_id]['ann_ids'] = []
            images_dict[img_id]['ann_ids'].append(ann_id)

    # Process images
    total_images = 0
    total_annotations = 0

    print(f"\nProcessing images (filtering for {len(DEEPSCORES_TO_HARMONY_FILTERED)} target classes)...")

    for image_id, image_data in tqdm(images_dict.items(), desc=f"{split_name}"):
        success, num_anns = process_image(
            image_id, image_data, annotations_dict, images_dir,
            output_images, output_labels, split_name, stats
        )

        if success:
            total_images += 1
            total_annotations += num_anns

    return total_images, total_annotations


def create_yaml(output_dir, train_count, val_count):
    """Create YOLO dataset YAML configuration."""
    yaml_content = f"""# DeepScoresV2 → YOLO Fermata Dataset
# Converted: 2025-11-25
# Focus: Fermata class only (avoiding OOM from 175K staff instances)
# Source: DeepScoresV2 train/test splits with 1,712 fermata annotations

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
"""

    yaml_path = output_dir / "deepscores_fermata.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset YAML created: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepScoresV2 to YOLO format (fermata & barline only)"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/thc1006/dev/music-app/training/datasets/ds2_dense"),
        help="DeepScoresV2 source directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/thc1006/dev/music-app/training/datasets/yolo_deepscores_converted"),
        help="Output directory for YOLO format"
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default="deepscores_train.json",
        help="Training split JSON filename"
    )
    parser.add_argument(
        "--test-json",
        type=str,
        default="deepscores_test.json",
        help="Test split JSON filename (used as validation)"
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 60)
    print("DeepScoresV2 → YOLO Converter (Fermata Only)")
    print("=" * 60)
    print(f"\nSource: {args.source}")
    print(f"Output: {args.output}")
    print(f"\nTarget class: {list(TARGET_CLASSES.values())}")
    print(f"DeepScores categories: {list(DEEPSCORES_TO_HARMONY_FILTERED.keys())}")
    print(f"\nExpected fermata instances: ~856 (1,712 with duplicates)")

    # Validate source directory
    if not args.source.exists():
        print(f"\nERROR: Source directory not found: {args.source}")
        print("\nPlease download DeepScoresV2 first.")
        print("Dataset: https://zenodo.org/record/4012193")
        return 1

    train_json = args.source / args.train_json
    test_json = args.source / args.test_json
    images_dir = args.source / "images"

    if not train_json.exists():
        print(f"\nERROR: Training JSON not found: {train_json}")
        return 1

    if not test_json.exists():
        print(f"\nERROR: Test JSON not found: {test_json}")
        return 1

    if not images_dir.exists():
        print(f"\nERROR: Images directory not found: {images_dir}")
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

    # Convert training split
    train_images, train_anns = convert_split(
        train_json, images_dir, output_images_train, output_labels_train,
        "train", stats
    )

    # Convert test split (as validation)
    val_images, val_anns = convert_split(
        test_json, images_dir, output_images_val, output_labels_val,
        "val", stats
    )

    # Print final statistics
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(f"\nTrain: {train_images} images, {train_anns} annotations")
    print(f"Val:   {val_images} images, {val_anns} annotations")
    print(f"Total: {train_images + val_images} images, {train_anns + val_anns} annotations")

    print("\n--- TARGET CLASS COUNTS ---")
    for class_id, class_name in sorted(TARGET_CLASSES.items()):
        count = stats.get(class_name, 0)
        print(f"  {class_name:20s} (id={class_id:2d}): {count:6d} annotations")

    # Show all non-zero classes
    print("\n--- ALL CONVERTED CLASSES ---")
    for class_name, count in sorted(stats.items(), key=lambda x: -x[1]):
        class_id = CLASS_NAMES.index(class_name)
        print(f"  {class_name:20s} (id={class_id:2d}): {count:6d}")

    # Create YAML config
    create_yaml(args.output, train_images, val_images)

    # Save conversion report
    report_path = args.output / "conversion_report.txt"
    with open(report_path, 'w') as f:
        f.write("DeepScoresV2 → YOLO Conversion Report (Fermata Only)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: 2025-11-25\n")
        f.write(f"Source: {args.source}\n")
        f.write(f"Output: {args.output}\n\n")
        f.write(f"Train images: {train_images}\n")
        f.write(f"Val images: {val_images}\n")
        f.write(f"Total images: {train_images + val_images}\n")
        f.write(f"Total annotations: {train_anns + val_anns}\n\n")
        f.write("Target Class Counts:\n")
        for class_id, class_name in sorted(TARGET_CLASSES.items()):
            count = stats.get(class_name, 0)
            f.write(f"  {class_name:20s} (id={class_id:2d}): {count:6d}\n")
        f.write("\nAll Converted Classes:\n")
        for class_name, count in sorted(stats.items(), key=lambda x: -x[1]):
            class_id = CLASS_NAMES.index(class_name)
            f.write(f"  {class_name:20s} (id={class_id:2d}): {count:6d}\n")

    print(f"\nConversion report saved: {report_path}")
    print(f"\nOutput directory: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
