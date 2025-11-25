#!/usr/bin/env python3
"""
Convert MUSCIMA++ V2 annotations to YOLO format.
Extracts fermata, natural, barline and other relevant classes.

MUSCIMA++ has handwritten music score annotations with precise bounding boxes.
"""

import os
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path("/home/thc1006/dev/music-app/training")
MUSCIMA_DIR = BASE_DIR / "datasets" / "external" / "omr_downloads" / "MuscimaPlusPlus_V2" / "v2.0"
IMAGES_DIR = BASE_DIR / "datasets" / "external" / "omr_downloads" / "MuscimaPlusPlus_Images" / "fulls"
OUTPUT_DIR = BASE_DIR / "datasets" / "yolo_muscima_converted"

# MUSCIMA++ class to our 33-class system mapping
# Only map classes we care about
MUSCIMA_TO_HARMONY = {
    # Noteheads
    "noteheadFull": 0,           # notehead_filled
    "noteheadHalf": 1,           # notehead_hollow
    "noteheadWhole": 1,          # notehead_hollow

    # Stems and beams
    "stem": 2,                   # stem
    "beam": 3,                   # beam

    # Flags
    "flag8thUp": 4,              # flag_8th
    "flag8thDown": 4,            # flag_8th
    "flag16thUp": 5,             # flag_16th
    "flag16thDown": 5,           # flag_16th
    "flag32ndUp": 6,             # flag_32nd
    "flag32ndDown": 6,           # flag_32nd

    # Dots
    "augmentationDot": 7,        # augmentation_dot

    # Ties/slurs (optional)
    "tie": 8,                    # tie
    "slur": 8,                   # tie (approximate)

    # Clefs
    "gClef": 9,                  # clef_treble
    "fClef": 10,                 # clef_bass
    "cClef": 11,                 # clef_alto (also tenor)

    # Accidentals - KEY TARGETS
    "accidentalSharp": 13,       # accidental_sharp
    "accidentalFlat": 14,        # accidental_flat
    "accidentalNatural": 15,     # accidental_natural - TARGET
    "accidentalDoubleSharp": 16, # accidental_double_sharp
    "accidentalDoubleFlat": 17,  # accidental_double_flat

    # Rests
    "restWhole": 18,             # rest_whole
    "restHalf": 19,              # rest_half
    "restQuarter": 20,           # rest_quarter
    "rest8th": 21,               # rest_8th
    "rest16th": 22,              # rest_16th

    # Barlines - KEY TARGETS
    "barline": 23,               # barline - TARGET
    "barlineHeavy": 24,          # barline_double - TARGET
    "barlineDotted": 23,         # barline

    # Time signatures (as single unit)
    "timeSignature": 27,         # time_signature

    # Key signatures
    "keySignature": 28,          # key_signature

    # Fermata - KEY TARGET (most important!)
    "fermataAbove": 29,          # fermata - TARGET
    "fermataBelow": 29,          # fermata - TARGET

    # Dynamics
    "dynamicP": 30,              # dynamic_soft
    "dynamicPP": 30,             # dynamic_soft
    "dynamicPPP": 30,            # dynamic_soft
    "dynamicF": 31,              # dynamic_loud
    "dynamicFF": 31,             # dynamic_loud
    "dynamicFFF": 31,            # dynamic_loud

    # Ledger lines
    "legerLine": 32,             # ledger_line
}

# Our 33 class names
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


def parse_muscima_xml(xml_path):
    """Parse a MUSCIMA++ XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    for node in root.findall('.//Node'):
        class_name = node.find('ClassName').text

        # Skip if not in our mapping
        if class_name not in MUSCIMA_TO_HARMONY:
            continue

        # Get bounding box
        top = int(node.find('Top').text)
        left = int(node.find('Left').text)
        width = int(node.find('Width').text)
        height = int(node.find('Height').text)

        # Convert to our class ID
        class_id = MUSCIMA_TO_HARMONY[class_name]

        annotations.append({
            'class_name': class_name,
            'class_id': class_id,
            'top': top,
            'left': left,
            'width': width,
            'height': height
        })

    return annotations


def get_image_dimensions(image_path):
    """Get image dimensions without loading the full image."""
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def convert_to_yolo(annotations, img_width, img_height):
    """Convert annotations to YOLO format (normalized center x, y, w, h)."""
    yolo_annotations = []

    for ann in annotations:
        # Calculate center coordinates
        center_x = (ann['left'] + ann['width'] / 2) / img_width
        center_y = (ann['top'] + ann['height'] / 2) / img_height
        norm_width = ann['width'] / img_width
        norm_height = ann['height'] / img_height

        # Clamp to [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        norm_width = max(0, min(1, norm_width))
        norm_height = max(0, min(1, norm_height))

        yolo_annotations.append({
            'class_id': ann['class_id'],
            'center_x': center_x,
            'center_y': center_y,
            'width': norm_width,
            'height': norm_height,
            'original_class': ann['class_name']
        })

    return yolo_annotations


def main():
    print("=" * 60)
    print("MUSCIMA++ V2 to YOLO Converter")
    print("=" * 60)

    # Check paths
    if not MUSCIMA_DIR.exists():
        print(f"ERROR: MUSCIMA++ directory not found: {MUSCIMA_DIR}")
        sys.exit(1)

    if not IMAGES_DIR.exists():
        print(f"ERROR: Images directory not found: {IMAGES_DIR}")
        sys.exit(1)

    # Find annotation files
    annotations_dir = MUSCIMA_DIR / "data" / "annotations"
    if not annotations_dir.exists():
        # Try alternative path
        annotations_dir = MUSCIMA_DIR / "data"

    xml_files = list(annotations_dir.glob("**/*.xml"))
    print(f"\nFound {len(xml_files)} XML annotation files")

    # Find images
    image_files = list(IMAGES_DIR.glob("*.png"))
    print(f"Found {len(image_files)} images")

    # Create output directories
    output_images = OUTPUT_DIR / "images"
    output_labels = OUTPUT_DIR / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # Statistics
    class_counts = defaultdict(int)
    target_counts = {
        'fermataAbove': 0, 'fermataBelow': 0,
        'accidentalNatural': 0,
        'barline': 0, 'barlineHeavy': 0
    }
    total_images = 0
    total_annotations = 0

    # Build image lookup (document name -> image path)
    image_lookup = {}
    for img_path in image_files:
        # Extract document identifier from filename
        # Format: CVC-MUSCIMA_W-01_N-10_D-ideal.png
        name = img_path.stem
        image_lookup[name] = img_path

    print("\nProcessing annotations...")

    for xml_path in xml_files:
        try:
            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get document name from XML
            doc_name = root.get('document', '')

            if not doc_name:
                continue

            # Find corresponding image
            if doc_name not in image_lookup:
                # Try without extension
                found = False
                for key in image_lookup:
                    if doc_name in key or key in doc_name:
                        doc_name = key
                        found = True
                        break
                if not found:
                    continue

            img_path = image_lookup[doc_name]

            # Get image dimensions
            img_width, img_height = get_image_dimensions(img_path)

            # Parse annotations
            annotations = parse_muscima_xml(xml_path)

            if not annotations:
                continue

            # Convert to YOLO format
            yolo_anns = convert_to_yolo(annotations, img_width, img_height)

            # Count statistics
            for ann in yolo_anns:
                class_counts[CLASS_NAMES[ann['class_id']]] += 1
                if ann['original_class'] in target_counts:
                    target_counts[ann['original_class']] += 1

            # Write YOLO label file
            output_name = f"muscima_{doc_name}"
            label_path = output_labels / f"{output_name}.txt"

            with open(label_path, 'w') as f:
                for ann in yolo_anns:
                    f.write(f"{ann['class_id']} {ann['center_x']:.6f} {ann['center_y']:.6f} "
                           f"{ann['width']:.6f} {ann['height']:.6f}\n")

            # Copy image
            dest_img = output_images / f"{output_name}.png"
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            total_images += 1
            total_annotations += len(yolo_anns)

        except Exception as e:
            print(f"  Error processing {xml_path.name}: {e}")
            continue

    # Print results
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"\nTotal images converted: {total_images}")
    print(f"Total annotations: {total_annotations}")

    print("\n--- TARGET CLASS COUNTS (KEY RESULTS) ---")
    for class_name, count in target_counts.items():
        harmony_class = MUSCIMA_TO_HARMONY.get(class_name, -1)
        if harmony_class >= 0:
            print(f"  {class_name} -> {CLASS_NAMES[harmony_class]}: {count}")

    print("\n--- ALL CLASS DISTRIBUTION ---")
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count}")

    print(f"\nOutput saved to: {OUTPUT_DIR}")

    # Create dataset info file
    info_path = OUTPUT_DIR / "dataset_info.txt"
    with open(info_path, 'w') as f:
        f.write("MUSCIMA++ V2 to YOLO Conversion\n")
        f.write("=" * 40 + "\n")
        f.write(f"Images: {total_images}\n")
        f.write(f"Annotations: {total_annotations}\n")
        f.write("\nTarget Class Counts:\n")
        for class_name, count in target_counts.items():
            f.write(f"  {class_name}: {count}\n")
        f.write("\nClass Distribution:\n")
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {class_name}: {count}\n")

    return total_images, total_annotations, target_counts


if __name__ == "__main__":
    main()
