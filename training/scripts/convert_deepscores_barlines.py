#!/usr/bin/env python3
"""
Convert DeepScoresV2 Barline Annotations to YOLO Format

This script extracts barline-related annotations from the DeepScoresV2 dataset
and converts them to YOLO format for training.

Barline class mappings (based on harmony_phase5.yaml):
  - barLine → barline (class 23)
  - barLineDouble / barLineHeavyHeavy → barline_double (class 24)
  - barLineHeavyLight / barLineFinal → barline_final (class 25)
  - repeatDots related → barline_repeat (class 26)

DeepScoresV2 uses COCO-like JSON format with oriented bounding boxes.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


# Configuration
DEEPSCORES_DIR = Path("/home/thc1006/dev/music-app/training/datasets/external/deepscores_v2")
OUTPUT_DIR = Path("/home/thc1006/dev/music-app/training/datasets/deepscores_barlines_yolo")

# Class mappings from DeepScoresV2 to YOLO
# Based on SMuFL standard barline names commonly used in DeepScores
BARLINE_CLASS_MAPPING = {
    # Simple barlines
    "barline": 23,
    "barLine": 23,
    "thinBarline": 23,

    # Double barlines
    "barlineDouble": 24,
    "barLineDouble": 24,
    "doubleBarline": 24,
    "barLineHeavyHeavy": 24,

    # Final barlines (heavy-light)
    "barlineFinal": 25,
    "barLineFinal": 25,
    "finalBarline": 25,
    "barLineHeavyLight": 25,
    "reverseFinalBarline": 25,
    "barLineLightHeavy": 25,

    # Repeat barlines (with dots)
    "barlineRepeat": 26,
    "repeatBarline": 26,
    "repeatLeft": 26,
    "repeatRight": 26,
    "repeatBothSides": 26,
    "repeatDots": 26,
}

# Class names for reference
CLASS_NAMES = {
    23: "barline",
    24: "barline_double",
    25: "barline_final",
    26: "barline_repeat",
}


def find_annotations_file(base_dir: Path) -> Path:
    """Locate the annotations.json file in DeepScoresV2 dataset."""
    # Common locations in DeepScoresV2 structure
    possible_paths = [
        base_dir / "annotations.json",
        base_dir / "ds2_dense" / "annotations.json",
        base_dir / "deepscores" / "annotations.json",
    ]

    # Also search recursively
    json_files = list(base_dir.rglob("*.json"))

    # Filter for likely annotation files (large JSON files with "annotations" in name)
    candidates = [f for f in json_files if "annotation" in f.name.lower() and f.stat().st_size > 1_000_000]

    # Try explicit paths first
    for path in possible_paths:
        if path.exists():
            return path

    # Try candidates by size (largest first)
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find annotations.json in {base_dir}\n"
        f"Please ensure DeepScoresV2 dataset is extracted correctly."
    )


def find_images_dir(base_dir: Path) -> Path:
    """Locate the images directory in DeepScoresV2 dataset."""
    possible_paths = [
        base_dir / "images",
        base_dir / "ds2_dense" / "images",
        base_dir / "deepscores" / "images",
    ]

    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path

    # Search recursively
    image_dirs = [d for d in base_dir.rglob("images") if d.is_dir()]
    if image_dirs:
        return image_dirs[0]

    raise FileNotFoundError(f"Could not find images directory in {base_dir}")


def convert_obb_to_bbox(obb: List[float]) -> List[float]:
    """
    Convert oriented bounding box to axis-aligned bounding box.

    Args:
        obb: [x0, y0, x1, y1, x2, y2, x3, y3] - four corner coordinates

    Returns:
        [x_min, y_min, x_max, y_max]
    """
    if len(obb) != 8:
        raise ValueError(f"OBB must have 8 values, got {len(obb)}")

    x_coords = [obb[i] for i in range(0, 8, 2)]
    y_coords = [obb[i] for i in range(1, 8, 2)]

    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert absolute bbox [x_min, y_min, x_max, y_max] to YOLO format.

    Returns:
        (x_center, y_center, width, height) - normalized to [0, 1]
    """
    x_min, y_min, x_max, y_max = bbox

    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def load_annotations(annotations_path: Path) -> Dict:
    """Load and parse DeepScoresV2 annotations JSON."""
    print(f"Loading annotations from: {annotations_path}")
    print(f"File size: {annotations_path.stat().st_size / (1024*1024):.2f} MB")

    with open(annotations_path, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded annotations")
    print(f"  - Images: {len(data.get('images', []))}")
    print(f"  - Annotations: {len(data.get('annotations', []))}")
    print(f"  - Categories: {len(data.get('categories', []))}")

    return data


def analyze_barline_categories(data: Dict) -> Dict[str, int]:
    """
    Analyze categories to find all barline-related classes.

    Returns:
        Dict mapping category_id to YOLO class_id
    """
    categories = data.get('categories', [])

    print("\n" + "=" * 70)
    print("Analyzing DeepScoresV2 Categories for Barlines")
    print("=" * 70)

    barline_categories = {}
    all_category_names = []

    for cat in categories:
        cat_id = cat.get('id') or cat.get('cat_id')
        cat_name = cat.get('name')

        if cat_name:
            all_category_names.append(cat_name)

            # Check if this is a barline-related category
            if cat_name in BARLINE_CLASS_MAPPING:
                yolo_class = BARLINE_CLASS_MAPPING[cat_name]
                barline_categories[str(cat_id)] = yolo_class
                print(f"  ✓ Found: {cat_name} (DeepScores ID: {cat_id}) → YOLO class {yolo_class} ({CLASS_NAMES[yolo_class]})")

    # Also check for partial matches (case-insensitive)
    for cat in categories:
        cat_id = cat.get('id') or cat.get('cat_id')
        cat_name = cat.get('name')

        if cat_name and str(cat_id) not in barline_categories:
            cat_name_lower = cat_name.lower()
            if 'barline' in cat_name_lower or 'repeat' in cat_name_lower:
                print(f"  ? Potential match (not in mapping): {cat_name} (ID: {cat_id})")

    if not barline_categories:
        print("\n⚠ WARNING: No barline categories found in mapping!")
        print("\nAll categories containing 'bar' or 'repeat':")
        for name in sorted(all_category_names):
            if 'bar' in name.lower() or 'repeat' in name.lower():
                print(f"  - {name}")

    print(f"\n✓ Found {len(barline_categories)} barline categories")
    return barline_categories


def convert_dataset(data: Dict, barline_cat_map: Dict[str, int], images_dir: Path, output_dir: Path):
    """
    Convert DeepScoresV2 barline annotations to YOLO format.

    Args:
        data: Parsed annotations JSON
        barline_cat_map: Mapping from DeepScores category_id to YOLO class_id
        images_dir: Directory containing images
        output_dir: Output directory for YOLO dataset
    """
    # Create output directories
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Index images by ID
    images_by_id = {}
    for img in data.get('images', []):
        img_id = str(img.get('id') or img.get('img_id'))
        images_by_id[img_id] = img

    # Group annotations by image
    annotations_by_image = defaultdict(list)
    annotations_list = data.get('annotations', [])

    if isinstance(annotations_list, dict):
        # Handle case where annotations is a dict instead of list
        annotations_list = list(annotations_list.values())

    for ann in annotations_list:
        img_id = str(ann.get('img_id') or ann.get('image_id'))
        cat_id = str(ann.get('cat_id') or ann.get('category_id'))

        # Only keep barline annotations
        if cat_id in barline_cat_map:
            annotations_by_image[img_id].append(ann)

    print(f"\n✓ Found barline annotations in {len(annotations_by_image)} images")

    # Statistics
    stats = defaultdict(int)
    skipped_no_barlines = 0
    skipped_missing_image = 0
    converted_images = 0

    # Process each image with barline annotations
    print("\nConverting annotations to YOLO format...")

    for img_id, annotations in tqdm(annotations_by_image.items(), desc="Processing"):
        if img_id not in images_by_id:
            skipped_missing_image += 1
            continue

        img_info = images_by_id[img_id]
        img_filename = img_info.get('filename') or img_info.get('file_name')
        img_width = img_info.get('width')
        img_height = img_info.get('height')

        if not all([img_filename, img_width, img_height]):
            continue

        # Find source image
        source_image = images_dir / img_filename
        if not source_image.exists():
            # Try without subdirectory
            source_image = images_dir.parent / img_filename

        if not source_image.exists():
            skipped_missing_image += 1
            continue

        # Determine train/val split (90/10)
        is_train = (int(img_id) % 10) != 0
        split = "train" if is_train else "val"

        # Copy image
        dest_image = output_dir / split / "images" / img_filename
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, dest_image)

        # Convert annotations
        yolo_lines = []
        for ann in annotations:
            cat_id = str(ann.get('cat_id') or ann.get('category_id'))
            yolo_class = barline_cat_map[cat_id]

            # Get bounding box
            # Try oriented bbox first
            if 'o_bbox' in ann:
                obb = ann['o_bbox']
                bbox = convert_obb_to_bbox(obb)
            elif 'a_bbox' in ann:
                bbox = ann['a_bbox']
            elif 'bbox' in ann:
                bbox = ann['bbox']
                # COCO format: [x, y, width, height] → convert to [x_min, y_min, x_max, y_max]
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    bbox = [x, y, x + w, y + h]
            else:
                continue

            # Convert to YOLO format
            try:
                x_center, y_center, width, height = bbox_to_yolo(bbox, img_width, img_height)
                yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                stats[CLASS_NAMES[yolo_class]] += 1
            except Exception as e:
                print(f"Error converting bbox for image {img_filename}: {e}")
                continue

        # Write label file
        if yolo_lines:
            label_file = output_dir / split / "labels" / f"{Path(img_filename).stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            converted_images += 1

    # Print statistics
    print("\n" + "=" * 70)
    print("Conversion Statistics")
    print("=" * 70)
    print(f"Images converted: {converted_images}")
    print(f"Images skipped (no barlines): {skipped_no_barlines}")
    print(f"Images skipped (missing file): {skipped_missing_image}")

    print("\nAnnotations by class:")
    for class_name in sorted(stats.keys()):
        print(f"  {class_name}: {stats[class_name]}")

    print(f"\nTotal annotations: {sum(stats.values())}")

    # Count actual files
    train_images = len(list((output_dir / "train" / "images").glob("*")))
    val_images = len(list((output_dir / "val" / "images").glob("*")))
    train_labels = len(list((output_dir / "train" / "labels").glob("*.txt")))
    val_labels = len(list((output_dir / "val" / "labels").glob("*.txt")))

    print("\nDataset split:")
    print(f"  Train: {train_images} images, {train_labels} labels")
    print(f"  Val:   {val_images} images, {val_labels} labels")


def create_dataset_yaml(output_dir: Path, stats: Dict[str, int]):
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# DeepScoresV2 Barlines Dataset (YOLO Format)
# Converted from DeepScoresV2 Dense
# Source: https://zenodo.org/records/4012193

path: {output_dir}
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

# Statistics:
# barline: {stats.get('barline', 0)}
# barline_double: {stats.get('barline_double', 0)}
# barline_final: {stats.get('barline_final', 0)}
# barline_repeat: {stats.get('barline_repeat', 0)}
"""

    yaml_path = output_dir / "deepscores_barlines.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Created dataset config: {yaml_path}")


def main():
    """Main conversion workflow."""
    print("=" * 70)
    print("DeepScoresV2 Barline Annotation Converter")
    print("=" * 70)

    # Check if DeepScoresV2 is downloaded
    if not DEEPSCORES_DIR.exists():
        print(f"\n✗ DeepScoresV2 dataset not found at: {DEEPSCORES_DIR}")
        print("\nPlease run the download script first:")
        print("  python /home/thc1006/dev/music-app/training/scripts/download_deepscores.py")
        sys.exit(1)

    try:
        # Find annotations and images
        annotations_path = find_annotations_file(DEEPSCORES_DIR)
        images_dir = find_images_dir(DEEPSCORES_DIR)

        print(f"\n✓ Found annotations: {annotations_path}")
        print(f"✓ Found images: {images_dir}")

        # Load annotations
        data = load_annotations(annotations_path)

        # Analyze and map barline categories
        barline_cat_map = analyze_barline_categories(data)

        if not barline_cat_map:
            print("\n✗ No barline categories found. Cannot proceed.")
            print("\nPlease check the category names in the dataset.")
            sys.exit(1)

        # Convert dataset
        print("\n" + "=" * 70)
        print("Converting Dataset")
        print("=" * 70)
        convert_dataset(data, barline_cat_map, images_dir, OUTPUT_DIR)

        print("\n" + "=" * 70)
        print("✓ Conversion Complete!")
        print("=" * 70)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("  1. Review the converted dataset")
        print("  2. Merge with Phase 5 dataset:")
        print("     python /home/thc1006/dev/music-app/training/scripts/merge_with_phase5.py")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
