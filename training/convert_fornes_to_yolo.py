#!/usr/bin/env python3
"""
Fornes Dataset to YOLO Format Converter
=======================================
將 Fornes 手寫音樂符號數據集轉換為 YOLO 格式。

Fornes 數據集包含從 19 世紀樂譜中提取的譜號和變音記號。
每個圖片包含單一符號，因此 bounding box 覆蓋整個圖片（留小邊距）。

來源數據集引用：
- A.Fornés and J.Lladós and G. Sanchez, "Old Handwritten Musical Symbol
  Classification by a Dynamic Time Warping Based Method", Graphics Recognition, 2008.

類別映射（對應 33 類別系統）：
- ACCIDENTAL_DoubSharp → 16 (accidental_double_sharp)
- ACCIDENTAL_Flat → 14 (accidental_flat)
- ACCIDENTAL_Sharp → 13 (accidental_sharp)
- ACCIDENTAL_Natural → 15 (accidental_natural)
- CLEF_Alto → 11 (clef_alto)
- CLEF_Bass → 10 (clef_bass)
- CLEF_Trebble → 9 (clef_treble)
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# Configuration
# ============================================

# Source dataset path
FORNES_ROOT = Path("/home/thc1006/dev/music-app/training/datasets/external/omr_downloads/Fornes")

# Output YOLO dataset path
OUTPUT_ROOT = Path("/home/thc1006/dev/music-app/training/datasets/external_yolo/fornes")

# Class mapping: Fornes folder name → YOLO class ID (33-class system)
CLASS_MAPPING = {
    "ACCIDENTAL_DoubSharp": 16,  # accidental_double_sharp - 關鍵類別！
    "ACCIDENTAL_Flat": 14,       # accidental_flat
    "ACCIDENTAL_Sharp": 13,      # accidental_sharp
    "ACCIDENTAL_Natural": 15,    # accidental_natural
    "CLEF_Alto": 11,             # clef_alto
    "CLEF_Bass": 10,             # clef_bass
    "CLEF_Trebble": 9,           # clef_treble (注意原始拼寫)
}

# Bounding box margin (percentage of image size)
# 0.05 = 5% margin on each side
BBOX_MARGIN = 0.05


# ============================================
# Conversion Functions
# ============================================

def create_yolo_annotation(img_width: int, img_height: int, class_id: int, margin: float = BBOX_MARGIN) -> str:
    """
    Create YOLO format annotation for a single-symbol image.

    YOLO format: class_id x_center y_center width height
    All coordinates are normalized to [0, 1].

    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        class_id: YOLO class ID
        margin: Margin percentage (0.05 = 5%)

    Returns:
        YOLO annotation string
    """
    # Center is always 0.5, 0.5 for single-symbol images
    x_center = 0.5
    y_center = 0.5

    # Width and height with margin
    # If margin = 0.05, we leave 5% on each side, so bbox covers 90%
    bbox_width = 1.0 - (2 * margin)
    bbox_height = 1.0 - (2 * margin)

    # YOLO format: class x_center y_center width height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"


def convert_bmp_to_png(src_path: Path, dst_path: Path) -> tuple[int, int]:
    """
    Convert BMP image to PNG format.

    Args:
        src_path: Source BMP file path
        dst_path: Destination PNG file path

    Returns:
        Tuple of (width, height) of the image
    """
    with Image.open(src_path) as img:
        # Convert 1-bit images to RGB for better compatibility
        if img.mode == '1':
            img = img.convert('RGB')
        elif img.mode == 'L':
            img = img.convert('RGB')
        elif img.mode == 'P':
            img = img.convert('RGB')

        img.save(dst_path, 'PNG')
        return img.size


def process_category(category_name: str, class_id: int, stats: dict) -> int:
    """
    Process all images in a category folder.

    Args:
        category_name: Fornes category folder name
        class_id: YOLO class ID
        stats: Statistics dictionary to update

    Returns:
        Number of successfully processed images
    """
    src_dir = FORNES_ROOT / category_name

    if not src_dir.exists():
        logger.warning(f"Category folder not found: {src_dir}")
        return 0

    images_dir = OUTPUT_ROOT / "images"
    labels_dir = OUTPUT_ROOT / "labels"

    processed = 0
    errors = 0

    # Get all BMP files
    bmp_files = list(src_dir.glob("*.bmp"))

    logger.info(f"Processing {category_name}: {len(bmp_files)} files -> class {class_id}")

    for bmp_file in bmp_files:
        try:
            # Create unique filename with category prefix to avoid collisions
            base_name = bmp_file.stem
            new_name = f"fornes_{category_name}_{base_name}"

            png_path = images_dir / f"{new_name}.png"
            txt_path = labels_dir / f"{new_name}.txt"

            # Convert image
            width, height = convert_bmp_to_png(bmp_file, png_path)

            # Create annotation
            annotation = create_yolo_annotation(width, height, class_id)
            txt_path.write_text(annotation + "\n")

            processed += 1

        except Exception as e:
            logger.error(f"Error processing {bmp_file}: {e}")
            errors += 1

    stats[category_name] = {
        'class_id': class_id,
        'total': len(bmp_files),
        'processed': processed,
        'errors': errors
    }

    return processed


def main():
    """Main conversion function."""
    logger.info("=" * 60)
    logger.info("Fornes Dataset to YOLO Format Converter")
    logger.info("=" * 60)

    # Verify source exists
    if not FORNES_ROOT.exists():
        logger.error(f"Source dataset not found: {FORNES_ROOT}")
        return

    # Create output directories
    images_dir = OUTPUT_ROOT / "images"
    labels_dir = OUTPUT_ROOT / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Source: {FORNES_ROOT}")
    logger.info(f"Output: {OUTPUT_ROOT}")
    logger.info(f"Margin: {BBOX_MARGIN * 100:.1f}%")
    logger.info("-" * 60)

    # Process each category
    stats = {}
    total_processed = 0

    for category_name, class_id in CLASS_MAPPING.items():
        count = process_category(category_name, class_id, stats)
        total_processed += count

    # Print summary
    logger.info("=" * 60)
    logger.info("Conversion Summary")
    logger.info("=" * 60)

    print("\n### Category Statistics ###\n")
    print(f"{'Category':<25} {'Class ID':<10} {'Total':<8} {'Processed':<10} {'Errors':<8}")
    print("-" * 65)

    for category_name, data in stats.items():
        print(f"{category_name:<25} {data['class_id']:<10} {data['total']:<8} {data['processed']:<10} {data['errors']:<8}")

    print("-" * 65)
    print(f"{'TOTAL':<25} {'':<10} {sum(d['total'] for d in stats.values()):<8} {total_processed:<10}")

    # Class distribution for training
    print("\n### YOLO Class Distribution ###\n")

    class_names = {
        9: "clef_treble",
        10: "clef_bass",
        11: "clef_alto",
        13: "accidental_sharp",
        14: "accidental_flat",
        15: "accidental_natural",
        16: "accidental_double_sharp",
    }

    class_counts = {}
    for data in stats.values():
        class_id = data['class_id']
        class_counts[class_id] = class_counts.get(class_id, 0) + data['processed']

    print(f"{'Class ID':<10} {'Class Name':<25} {'Count':<8}")
    print("-" * 45)
    for class_id in sorted(class_counts.keys()):
        class_name = class_names.get(class_id, f"class_{class_id}")
        print(f"{class_id:<10} {class_name:<25} {class_counts[class_id]:<8}")

    # Key insight for double_sharp
    double_sharp_count = class_counts.get(16, 0)
    print("\n" + "=" * 60)
    print(f"CRITICAL: accidental_double_sharp (class 16) samples: {double_sharp_count}")
    print("This is the key bottleneck class with mAP=0!")
    print("=" * 60)

    # Output paths
    print(f"\nOutput locations:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"  Total files: {total_processed} images + {total_processed} labels")


if __name__ == "__main__":
    main()
