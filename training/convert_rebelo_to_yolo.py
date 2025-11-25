#!/usr/bin/env python3
"""
Convert Rebelo Symbol Datasets (1 & 2) to YOLO format.
These are symbol classification datasets - each image contains a single symbol.
We create synthetic training images by placing symbols on backgrounds.

Key targets: Fermata (102), Natural (1,095)
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from collections import defaultdict

# Paths
BASE_DIR = Path("/home/thc1006/dev/music-app/training")
REBELO1_DIR = BASE_DIR / "datasets" / "external" / "omr_downloads" / "Rebelo1"
REBELO2_DIR = BASE_DIR / "datasets" / "external" / "omr_downloads" / "Rebelo2"
OUTPUT_DIR = BASE_DIR / "datasets" / "yolo_rebelo_converted"

# Rebelo class to our 33-class mapping
REBELO_TO_HARMONY = {
    # Rebelo1 classes
    "Natural": 15,           # accidental_natural - TARGET
    "Sharp": 13,             # accidental_sharp
    "Flat": 14,              # accidental_flat
    "Barline": 23,           # barline - TARGET
    "Beam": 3,               # beam
    "C-Clef": 11,            # clef_alto
    "F-Clef": 10,            # clef_bass
    "G-Clef": 9,             # clef_treble
    "Dot": 7,                # augmentation_dot
    "Quarter-Rest": 20,      # rest_quarter
    "Eighth-Rest": 21,       # rest_8th
    "Sixteenth-Rest": 22,    # rest_16th
    "Tie-Slur": 8,           # tie
    "Accent": None,          # skip

    # Rebelo2 additional classes
    "Fermata": 29,           # fermata - KEY TARGET!
    "Double-Whole-Rest": 18, # rest_whole (approximate)
    "Breve": 1,              # notehead_hollow (approximate)
    "Thirty-Two-Rest": 22,   # rest_16th (approximate)
    "Staccatissimo": None,   # skip
    "Common-Time": 27,       # time_signature
    "Cut-Time": 27,          # time_signature
    "Chord": None,           # skip (complex)
    "Multiple-Eighth-Notes": None,  # skip
    "Multiple-Quarter-Notes": None, # skip
    "Multiple-Half-Notes": None,    # skip
    "Multiple-Sixteenth-Notes": None, # skip
}

# Note-related classes (we'll skip these for now as they're complex)
NOTE_CLASSES = {
    "Eighth-Note", "Quarter-Note", "Half-Note", "Whole-Note",
    "Sixteenth-Note", "Thirty-Two-Note", "Eighth-Grace-Note"
}

# Time signature classes
TIME_SIG_CLASSES = {
    "12-8-Time", "1-8-Time", "2-4-Time", "2-8-Time", "3-4-Time",
    "3-8-Time", "4-2-Time", "4-4-Time", "4-8-Time", "5-4-Time",
    "5-8-Time", "6-4-Time", "6-8-Time", "7-4-Time", "8-8-Time", "9-8-Time"
}

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


def create_staff_background(width=640, height=640):
    """Create a simple staff-line background."""
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    pixels = img.load()

    # Draw 5 staff lines
    line_spacing = 20
    start_y = height // 2 - 2 * line_spacing

    for i in range(5):
        y = start_y + i * line_spacing
        for x in range(width):
            if 0 <= y < height:
                pixels[x, y] = (0, 0, 0)

    return img


def load_symbol_image(path):
    """Load a symbol image and convert to proper format."""
    img = Image.open(path)

    # Convert to RGB if needed
    if img.mode == '1':
        img = img.convert('L')
    if img.mode == 'L':
        img = img.convert('RGB')
    if img.mode == 'RGBA':
        # White background for transparent areas
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    return img


def place_symbol_on_background(symbol_img, bg_img, position=None):
    """Place a symbol on a background image."""
    bg_width, bg_height = bg_img.size
    sym_width, sym_height = symbol_img.size

    # Random position if not specified
    if position is None:
        max_x = bg_width - sym_width
        max_y = bg_height - sym_height
        if max_x > 0 and max_y > 0:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        else:
            x, y = 0, 0
    else:
        x, y = position

    # Create composite
    result = bg_img.copy()
    result.paste(symbol_img, (x, y))

    # Return image and bbox (x_center, y_center, width, height) normalized
    x_center = (x + sym_width / 2) / bg_width
    y_center = (y + sym_height / 2) / bg_height
    norm_width = sym_width / bg_width
    norm_height = sym_height / bg_height

    return result, (x_center, y_center, norm_width, norm_height)


def process_rebelo_directory(rebelo_dir, output_images, output_labels, prefix, stats):
    """Process all classes in a Rebelo directory."""

    if not rebelo_dir.exists():
        print(f"  Directory not found: {rebelo_dir}")
        return

    for class_dir in rebelo_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # Skip note classes (too complex) and time signatures
        if class_name in NOTE_CLASSES:
            continue

        # Map time signatures
        if class_name in TIME_SIG_CLASSES:
            class_id = 27  # time_signature
        elif class_name in REBELO_TO_HARMONY:
            class_id = REBELO_TO_HARMONY[class_name]
        else:
            continue

        if class_id is None:
            continue

        # Get all symbol images
        symbol_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.bmp"))

        if not symbol_files:
            continue

        print(f"  Processing {class_name}: {len(symbol_files)} files -> class {class_id} ({CLASS_NAMES[class_id]})")

        for idx, symbol_path in enumerate(symbol_files):
            try:
                # Load symbol
                symbol_img = load_symbol_image(symbol_path)

                # Scale symbol if too large
                max_size = 150
                if symbol_img.width > max_size or symbol_img.height > max_size:
                    ratio = max_size / max(symbol_img.width, symbol_img.height)
                    new_size = (int(symbol_img.width * ratio), int(symbol_img.height * ratio))
                    symbol_img = symbol_img.resize(new_size, Image.LANCZOS)

                # Create background
                bg = create_staff_background(640, 640)

                # Place symbol
                result_img, bbox = place_symbol_on_background(symbol_img, bg)

                # Save image
                output_name = f"{prefix}_{class_name}_{idx:04d}"
                img_path = output_images / f"{output_name}.png"
                result_img.save(img_path)

                # Save label
                label_path = output_labels / f"{output_name}.txt"
                x_c, y_c, w, h = bbox
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

                stats[CLASS_NAMES[class_id]] += 1

            except Exception as e:
                print(f"    Error processing {symbol_path.name}: {e}")
                continue


def main():
    print("=" * 60)
    print("Rebelo Symbol Datasets to YOLO Converter")
    print("=" * 60)

    # Create output directories
    output_images = OUTPUT_DIR / "images"
    output_labels = OUTPUT_DIR / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)

    # Process Rebelo1
    print("\n[1] Processing Rebelo1...")
    process_rebelo_directory(REBELO1_DIR, output_images, output_labels, "rebelo1", stats)

    # Process Rebelo2
    print("\n[2] Processing Rebelo2...")
    process_rebelo_directory(REBELO2_DIR, output_images, output_labels, "rebelo2", stats)

    # Print results
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    total = sum(stats.values())
    print(f"\nTotal images created: {total}")

    print("\n--- TARGET CLASS COUNTS ---")
    targets = ['accidental_natural', 'fermata', 'barline']
    for target in targets:
        count = stats.get(target, 0)
        print(f"  {target}: {count}")

    print("\n--- ALL CLASS DISTRIBUTION ---")
    for class_name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count}")

    print(f"\nOutput saved to: {OUTPUT_DIR}")

    # Save info file
    info_path = OUTPUT_DIR / "dataset_info.txt"
    with open(info_path, 'w') as f:
        f.write("Rebelo Symbol Datasets to YOLO Conversion\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total images: {total}\n\n")
        f.write("Target Class Counts:\n")
        for target in targets:
            f.write(f"  {target}: {stats.get(target, 0)}\n")
        f.write("\nAll Classes:\n")
        for class_name, count in sorted(stats.items(), key=lambda x: -x[1]):
            f.write(f"  {class_name}: {count}\n")

    return total, stats


if __name__ == "__main__":
    main()
