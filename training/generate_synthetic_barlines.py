#!/usr/bin/env python3
"""
Synthetic Barline Generator for YOLO Training
Generates diverse barline samples using LilyPond with controlled variations.

Class IDs:
  23: barline (single)
  24: barline_double
  25: barline_final
  26: barline_repeat
"""

import argparse
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# Barline types with LilyPond syntax and class IDs
BARLINE_TYPES = {
    'single': ('|', 23),
    'double': ('||', 24),
    'final': ('|.', 25),
    'repeat_start': ('.|:', 26),
    'repeat_end': (':|.', 26),
}

# Generation distribution
BARLINE_DISTRIBUTION = {
    'single': 0.40,
    'double': 0.25,
    'final': 0.25,
    'repeat_start': 0.05,
    'repeat_end': 0.05,
}

# LilyPond staff sizes
STAFF_SIZES = list(range(16, 27, 2))  # [16, 18, 20, 22, 24, 26]

# Barline thickness variations
THICKNESS_VARIATIONS = [
    {'hair': 1.5, 'thick': 5.0},  # thin
    {'hair': 1.9, 'thick': 6.0},  # default
    {'hair': 2.3, 'thick': 7.0},  # thick
]

# Clef options
CLEFS = ['treble', 'bass', 'alto', 'tenor']

# Time signatures
TIME_SIGNATURES = ['4/4', '3/4', '2/4', '6/8', '3/8', '2/2']

# Note patterns for context
NOTE_PATTERNS = [
    "c'4 d'4 e'4 f'4",
    "g'8 a'8 b'8 c''8 d''8 e''8 f''8 g''8",
    "c'2 d'2",
    "e'16 f'16 g'16 a'16 b'16 c''16 d''16 e''16",
    "c'1",
    "r4 g'4 r4 e'4",
]


@dataclass
class BarlineConfig:
    """Configuration for a single barline generation."""
    barline_type: str
    staff_size: int
    thickness: dict
    clef: str
    time_sig: str
    num_measures: int
    note_pattern: str
    multi_staff: bool


def generate_lilypond_score(config: BarlineConfig) -> str:
    """Generate LilyPond source code for a barline sample."""

    barline_symbol, _ = BARLINE_TYPES[config.barline_type]

    # Header
    score = f"""\\version "2.24.0"
\\paper {{
  indent = 0\\mm
  line-width = 180\\mm
  oddHeaderMarkup = ""
  evenHeaderMarkup = ""
  oddFooterMarkup = ""
  evenFooterMarkup = ""
  top-margin = 5\\mm
  bottom-margin = 5\\mm
}}

\\layout {{
  \\context {{
    \\Score
    \\override StaffSymbol.staff-space = #(magstep {(config.staff_size - 20) / 4})
    \\override BarLine.hair-thickness = #{config.thickness['hair']}
    \\override BarLine.thick-thickness = #{config.thickness['thick']}
  }}
}}

"""

    # Generate measures with barlines
    measures = []
    for i in range(config.num_measures):
        measures.append(config.note_pattern)

        # Add barlines at different positions
        if i < config.num_measures - 1:
            # Internal barlines
            if random.random() < 0.3:
                measures.append(f"\\bar \"{barline_symbol}\"")
        else:
            # Final barline
            measures.append(f"\\bar \"{barline_symbol}\"")

    music_content = " ".join(measures)

    if config.multi_staff:
        # Piano grand staff
        score += f"""
\\new PianoStaff <<
  \\new Staff {{
    \\clef {config.clef}
    \\time {config.time_sig}
    {music_content}
  }}
  \\new Staff {{
    \\clef bass
    \\time {config.time_sig}
    {music_content.replace("'", "")}
  }}
>>
"""
    else:
        # Single staff
        score += f"""
{{
  \\clef {config.clef}
  \\time {config.time_sig}
  {music_content}
}}
"""

    return score


def detect_barlines_in_image(image_path: str, class_id: int) -> List[Tuple[float, float, float, float]]:
    """
    Detect barline positions in the generated image using computer vision.
    Returns list of bounding boxes in YOLO format (x_center, y_center, width, height).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    height, width = img.shape

    # Threshold to binary
    _, binary = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to isolate vertical lines
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 20))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)

    # Find contours
    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter by aspect ratio and size (vertical lines)
        aspect_ratio = h / (w + 1e-6)
        if aspect_ratio > 3 and h > height * 0.1:  # Must be tall and thin
            # Convert to YOLO format (normalized)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            bbox_width = w / width
            bbox_height = h / height

            # Expand width slightly to capture double/final barlines
            bbox_width = min(bbox_width * 2.5, 0.05)

            bboxes.append((x_center, y_center, bbox_width, bbox_height))

    # Remove duplicate/overlapping boxes
    bboxes = merge_overlapping_boxes(bboxes)

    return bboxes


def merge_overlapping_boxes(bboxes: List[Tuple[float, float, float, float]],
                            iou_threshold: float = 0.5) -> List[Tuple[float, float, float, float]]:
    """Merge overlapping bounding boxes."""
    if not bboxes:
        return []

    # Sort by x_center
    bboxes = sorted(bboxes, key=lambda b: b[0])

    merged = []
    current = list(bboxes[0])

    for bbox in bboxes[1:]:
        # Check overlap
        x1_min = current[0] - current[2] / 2
        x1_max = current[0] + current[2] / 2
        x2_min = bbox[0] - bbox[2] / 2
        x2_max = bbox[0] + bbox[2] / 2

        overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        union = (x1_max - x1_min) + (x2_max - x2_min) - overlap
        iou = overlap / (union + 1e-6)

        if iou > iou_threshold:
            # Merge
            current[0] = (current[0] + bbox[0]) / 2
            current[1] = (current[1] + bbox[1]) / 2
            current[2] = max(current[2], bbox[2])
            current[3] = max(current[3], bbox[3])
        else:
            merged.append(tuple(current))
            current = list(bbox)

    merged.append(tuple(current))
    return merged


def compile_lilypond(ly_content: str, output_png: str) -> bool:
    """Compile LilyPond source to PNG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ly_path = Path(tmpdir) / "score.ly"
        ly_path.write_text(ly_content)

        # Compile to PNG
        try:
            result = subprocess.run(
                ['lilypond', '--png', '-dresolution=300', '-o', tmpdir, str(ly_path)],
                capture_output=True,
                timeout=30,
                check=True
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"LilyPond compilation failed: {e}")
            return False

        # Find generated PNG
        png_files = list(Path(tmpdir).glob("*.png"))
        if not png_files:
            return False

        # Crop and resize to 640x640
        img = Image.open(png_files[0])

        # Crop white margins
        img_array = np.array(img.convert('L'))
        coords = cv2.findNonZero((255 - img_array).astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img.crop((x, y, x + w, y + h))

        # Resize to 640x640 with padding
        img.thumbnail((640, 640), Image.Resampling.LANCZOS)

        # Create white canvas
        canvas = Image.new('RGB', (640, 640), 'white')
        offset_x = (640 - img.width) // 2
        offset_y = (640 - img.height) // 2
        canvas.paste(img, (offset_x, offset_y))

        canvas.save(output_png)
        return True


def generate_random_config() -> BarlineConfig:
    """Generate a random barline configuration."""
    # Select barline type based on distribution
    barline_type = random.choices(
        list(BARLINE_DISTRIBUTION.keys()),
        weights=list(BARLINE_DISTRIBUTION.values()),
        k=1
    )[0]

    return BarlineConfig(
        barline_type=barline_type,
        staff_size=random.choice(STAFF_SIZES),
        thickness=random.choice(THICKNESS_VARIATIONS),
        clef=random.choice(CLEFS),
        time_sig=random.choice(TIME_SIGNATURES),
        num_measures=random.randint(2, 6),
        note_pattern=random.choice(NOTE_PATTERNS),
        multi_staff=random.random() < 0.3,  # 30% piano grand staff
    )


def generate_dataset(output_dir: Path, num_samples: int):
    """Generate synthetic barline dataset."""

    # Create output directories
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_samples} (success: {successful}, failed: {failed})")

        config = generate_random_config()

        # Generate LilyPond source
        ly_content = generate_lilypond_score(config)

        # Output paths
        image_path = images_dir / f"barline_{i:06d}.png"
        label_path = labels_dir / f"barline_{i:06d}.txt"

        # Compile to PNG
        if not compile_lilypond(ly_content, str(image_path)):
            failed += 1
            continue

        # Detect barlines and create labels
        class_id = BARLINE_TYPES[config.barline_type][1]
        bboxes = detect_barlines_in_image(str(image_path), class_id)

        if not bboxes:
            # No barlines detected, skip
            image_path.unlink()
            failed += 1
            continue

        # Write YOLO labels
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        successful += 1

    print(f"\nGeneration complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful / num_samples * 100:.1f}%")

    # Generate statistics
    print("\nBarline type distribution:")
    for barline_type, dist in BARLINE_DISTRIBUTION.items():
        expected = int(num_samples * dist)
        print(f"  {barline_type}: ~{expected} samples ({dist*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic barline dataset using LilyPond"
    )
    parser.add_argument(
        '--count',
        type=int,
        default=10000,
        help='Number of samples to generate (default: 10000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets/synthetic_barlines',
        help='Output directory (default: datasets/synthetic_barlines)'
    )
    parser.add_argument(
        '--check-lilypond',
        action='store_true',
        help='Check if LilyPond is installed and exit'
    )

    args = parser.parse_args()

    # Check LilyPond installation
    try:
        result = subprocess.run(
            ['lilypond', '--version'],
            capture_output=True,
            check=True,
            text=True
        )
        version = result.stdout.split('\n')[0]
        print(f"✓ {version}")

        if args.check_lilypond:
            return
    except FileNotFoundError:
        print("✗ LilyPond not found!")
        print("Please install LilyPond:")
        print("  Ubuntu/Debian: sudo apt-get install lilypond")
        print("  macOS: brew install lilypond")
        return

    output_dir = Path(args.output)
    print(f"\nGenerating {args.count} synthetic barline samples...")
    print(f"Output directory: {output_dir}")

    generate_dataset(output_dir, args.count)


if __name__ == '__main__':
    main()
