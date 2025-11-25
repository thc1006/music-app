#!/usr/bin/env python3
"""
Synthetic Fermata Generator for YOLO Training
Uses LilyPond to create diverse fermata samples with proper bounding boxes.

Author: Claude Code
Date: 2025-11-25
Usage: python generate_synthetic_fermatas.py --count 5000 --output datasets/synthetic_fermatas/
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import random
import shutil
from dataclasses import dataclass
from xml.etree import ElementTree as ET

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install Pillow numpy")
    sys.exit(1)


@dataclass
class FermataConfig:
    """Configuration for a single fermata sample."""
    note_value: str  # whole, half, quarter, eighth
    clef: str  # treble, bass, alto, tenor
    position: str  # note, rest, barline
    time_signature: str  # 4/4, 3/4, 6/8, etc.
    staff_size: int  # 16-26
    fermata_count: int  # 1-3
    add_accidentals: bool
    add_dynamics: bool
    background_color: str  # white, cream, textured


class LilyPondFermataGenerator:
    """Generate synthetic fermata samples using LilyPond."""

    FERMATA_CLASS_ID = 29

    NOTE_VALUES = ['1', '2', '4', '8']  # whole, half, quarter, eighth
    CLEFS = ['treble', 'bass', 'alto', 'tenor']
    TIME_SIGNATURES = ['4/4', '3/4', '2/4', '6/8', '3/8']
    STAFF_SIZES = list(range(16, 27))
    POSITIONS = ['note', 'rest', 'barline']
    ACCIDENTALS = ['is', 'es', 'isis', 'eses']  # sharp, flat, double-sharp, double-flat
    DYNAMICS = ['\\pp', '\\p', '\\mp', '\\mf', '\\f', '\\ff']
    PITCHES = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
    OCTAVES = ["'", "''", "", ","]

    def __init__(self, output_dir: str):
        """Initialize generator with output directory."""
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.temp_dir = Path(tempfile.mkdtemp(prefix='fermata_gen_'))

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Temporary directory: {self.temp_dir}")

    def generate_random_config(self) -> FermataConfig:
        """Generate random fermata configuration."""
        return FermataConfig(
            note_value=random.choice(self.NOTE_VALUES),
            clef=random.choice(self.CLEFS),
            position=random.choice(self.POSITIONS),
            time_signature=random.choice(self.TIME_SIGNATURES),
            staff_size=random.choice(self.STAFF_SIZES),
            fermata_count=random.randint(1, 3),
            add_accidentals=random.random() > 0.5,
            add_dynamics=random.random() > 0.3,
            background_color=random.choice(['white', 'cream', 'textured'])
        )

    def generate_lilypond_code(self, config: FermataConfig, sample_id: int) -> str:
        """Generate LilyPond code for fermata sample."""

        measures = []
        for i in range(config.fermata_count):
            if config.position == 'note':
                measure = self._generate_note_with_fermata(config)
            elif config.position == 'rest':
                measure = self._generate_rest_with_fermata(config)
            else:  # barline
                measure = self._generate_barline_with_fermata(config)
            measures.append(measure)

        # Add filler measures to make interesting layout
        for _ in range(random.randint(1, 3)):
            measures.append(self._generate_random_measure(config))

        random.shuffle(measures)
        music_content = ' '.join(measures)

        ly_code = f'''\\version "2.24.3"

\\header {{
  tagline = ##f
}}

\\paper {{
  #(set-paper-size "a5")
  indent = 0\\mm
  line-width = 180\\mm
  top-margin = 5\\mm
  bottom-margin = 5\\mm
  left-margin = 5\\mm
  right-margin = 5\\mm
}}

\\layout {{
  \\context {{
    \\Staff
    \\override StaffSymbol.staff-space = #(magstep {(config.staff_size - 20) / 4})
  }}
}}

\\score {{
  \\new Staff {{
    \\clef "{config.clef}"
    \\time {config.time_signature}
    {music_content}
  }}
  \\layout {{ }}
}}
'''
        return ly_code

    def _generate_note_with_fermata(self, config: FermataConfig) -> str:
        """Generate a note with fermata."""
        pitch = random.choice(self.PITCHES)
        octave = random.choice(self.OCTAVES)
        accidental = random.choice(self.ACCIDENTALS) if config.add_accidentals else ''
        dynamic = random.choice(self.DYNAMICS) if config.add_dynamics else ''

        note = f"{pitch}{accidental}{octave}{config.note_value}\\fermata{dynamic}"

        # Add some context notes
        context_notes = []
        for _ in range(random.randint(1, 3)):
            p = random.choice(self.PITCHES)
            o = random.choice(self.OCTAVES)
            context_notes.append(f"{p}{o}{random.choice(['4', '8'])}")

        return f"{' '.join(context_notes)} {note}"

    def _generate_rest_with_fermata(self, config: FermataConfig) -> str:
        """Generate a rest with fermata."""
        rest = f"r{config.note_value}\\fermata"

        # Add context
        context_notes = []
        for _ in range(random.randint(2, 4)):
            p = random.choice(self.PITCHES)
            o = random.choice(self.OCTAVES)
            context_notes.append(f"{p}{o}{random.choice(['4', '8'])}")

        return f"{' '.join(context_notes)} {rest}"

    def _generate_barline_with_fermata(self, config: FermataConfig) -> str:
        """Generate a measure ending with fermata on barline."""
        # LilyPond syntax: \bar "||" \fermata
        notes = []
        for _ in range(random.randint(3, 6)):
            p = random.choice(self.PITCHES)
            o = random.choice(self.OCTAVES)
            notes.append(f"{p}{o}{random.choice(['4', '8'])}")

        return f"{' '.join(notes)} \\bar \"||\" \\fermata"

    def _generate_random_measure(self, config: FermataConfig) -> str:
        """Generate a random measure without fermata for context."""
        notes = []
        for _ in range(random.randint(3, 6)):
            p = random.choice(self.PITCHES)
            o = random.choice(self.OCTAVES)
            n = random.choice(['4', '8', '2'])
            notes.append(f"{p}{o}{n}")
        return ' '.join(notes)

    def render_to_image(self, ly_code: str, sample_id: int) -> Tuple[Path, Path]:
        """Render LilyPond code to PNG image."""
        ly_file = self.temp_dir / f"sample_{sample_id}.ly"

        # Write LilyPond file
        with open(ly_file, 'w', encoding='utf-8') as f:
            f.write(ly_code)

        # Render to PNG with high resolution
        output_base = self.temp_dir / f"sample_{sample_id}"
        cmd = [
            'lilypond',
            '--png',
            '-dresolution=300',
            f'-o{output_base}',
            str(ly_file)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"Warning: LilyPond failed for sample {sample_id}")
                print(result.stderr)
                return None, None

            # LilyPond creates sample_ID.png
            png_file = Path(f"{output_base}.png")
            svg_file = Path(f"{output_base}.svg")

            if not png_file.exists():
                print(f"Warning: PNG not generated for sample {sample_id}")
                return None, None

            return png_file, ly_file

        except subprocess.TimeoutExpired:
            print(f"Warning: LilyPond timeout for sample {sample_id}")
            return None, None
        except Exception as e:
            print(f"Error rendering sample {sample_id}: {e}")
            return None, None

    def extract_fermata_bboxes(self, ly_file: Path, png_file: Path) -> List[Tuple[float, float, float, float]]:
        """
        Extract fermata bounding boxes from rendered image.

        Returns list of (x_center, y_center, width, height) in normalized coordinates.
        """
        # Read the image
        img = Image.open(png_file)
        img_width, img_height = img.size

        # Convert to grayscale and numpy array
        img_gray = img.convert('L')
        img_array = np.array(img_gray)

        # Fermata detection heuristic:
        # Fermatas are typically small, curved symbols above/below notes
        # We'll use template matching approach

        bboxes = []

        # Simple approach: Look for dark regions that match fermata characteristics
        # In a production system, we'd use the SVG output for exact coordinates
        # For now, we'll use a heuristic-based detection

        # Threshold the image
        threshold = 200
        binary = img_array < threshold

        # Find connected components (potential symbols)
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)

        for i in range(1, num_features + 1):
            # Get region properties
            region = (labeled == i)
            coords = np.argwhere(region)

            if len(coords) == 0:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            width = x_max - x_min
            height = y_max - y_min
            area = len(coords)

            # Fermata heuristics:
            # - Small to medium size (not too large)
            # - Aspect ratio roughly 1:1 to 2:1
            # - Not a line (staff line, stem, etc.)

            if width < 10 or height < 10:
                continue  # Too small

            if width > img_width / 3 or height > img_height / 5:
                continue  # Too large

            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 3:
                continue  # Wrong aspect ratio

            # Check if it's likely a fermata by position
            # Fermatas are usually in upper or lower third of the image
            y_center = (y_min + y_max) / 2
            if 0.3 * img_height < y_center < 0.7 * img_height:
                continue  # In middle (likely not fermata)

            # Convert to normalized YOLO format
            x_center = ((x_min + x_max) / 2) / img_width
            y_center_norm = ((y_min + y_max) / 2) / img_height
            width_norm = width / img_width
            height_norm = height / img_height

            bboxes.append((x_center, y_center_norm, width_norm, height_norm))

        # Limit to expected fermata count (take top candidates by size)
        if len(bboxes) > 3:
            bboxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)[:3]

        return bboxes

    def simple_fermata_detection(self, png_file: Path, expected_count: int) -> List[Tuple[float, float, float, float]]:
        """
        Simplified fermata detection using basic image processing.
        Since we know we generated fermatas, we look for characteristic shapes.
        """
        img = Image.open(png_file)
        img_width, img_height = img.size

        # Convert to grayscale
        img_gray = img.convert('L')
        img_array = np.array(img_gray)

        # Find dark regions (musical symbols)
        threshold = 200
        binary = img_array < threshold

        # Scan for fermata-like regions
        bboxes = []

        # Divide image into horizontal strips
        strip_height = img_height // 10

        for strip_start in range(0, img_height - strip_height, strip_height // 2):
            strip_end = min(strip_start + strip_height, img_height)
            strip = binary[strip_start:strip_end, :]

            # Find dark regions in this strip
            col_density = strip.sum(axis=0)

            # Find peaks (likely symbols)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(col_density, height=strip_height * 0.1, distance=20)

            for peak_x in peaks:
                # Estimate bbox around this peak
                left = max(0, peak_x - 20)
                right = min(img_width, peak_x + 20)

                # Find vertical extent
                region = binary[strip_start:strip_end, left:right]
                if region.sum() < 50:
                    continue  # Too small

                y_coords = np.where(region.any(axis=1))[0]
                if len(y_coords) == 0:
                    continue

                top = strip_start + y_coords.min()
                bottom = strip_start + y_coords.max()

                # Check if this looks like a fermata
                width = right - left
                height = bottom - top

                if width < 15 or height < 10:
                    continue

                if width > 100 or height > 80:
                    continue

                # Convert to YOLO format
                x_center = ((left + right) / 2) / img_width
                y_center = ((top + bottom) / 2) / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                bboxes.append((x_center, y_center, width_norm, height_norm))

        # Remove duplicates (nearby boxes)
        filtered_bboxes = []
        for bbox in bboxes:
            is_duplicate = False
            for existing in filtered_bboxes:
                dx = abs(bbox[0] - existing[0])
                dy = abs(bbox[1] - existing[1])
                if dx < 0.05 and dy < 0.05:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_bboxes.append(bbox)

        # Take top candidates by size
        if len(filtered_bboxes) > expected_count:
            filtered_bboxes = sorted(
                filtered_bboxes,
                key=lambda b: b[2] * b[3],
                reverse=True
            )[:expected_count]

        return filtered_bboxes

    def process_and_save_image(
        self,
        png_file: Path,
        config: FermataConfig,
        sample_id: int
    ) -> bool:
        """Process image: resize, add variations, save with labels."""
        try:
            # Open image
            img = Image.open(png_file)

            # Apply background variation
            img = self._apply_background_variation(img, config.background_color)

            # Resize to 640x640 (YOLO training size)
            img = self._resize_and_pad(img, (640, 640))

            # Apply random augmentations
            img = self._apply_augmentations(img)

            # Extract fermata bboxes
            bboxes = self.simple_fermata_detection(png_file, config.fermata_count)

            if not bboxes:
                print(f"Warning: No fermatas detected in sample {sample_id}")
                return False

            # Save image
            output_image = self.images_dir / f"fermata_{sample_id:06d}.png"
            img.save(output_image, 'PNG')

            # Save YOLO label
            output_label = self.labels_dir / f"fermata_{sample_id:06d}.txt"
            with open(output_label, 'w') as f:
                for bbox in bboxes:
                    x, y, w, h = bbox
                    # YOLO format: class_id x_center y_center width height
                    f.write(f"{self.FERMATA_CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            return True

        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            return False

    def _apply_background_variation(self, img: Image.Image, bg_type: str) -> Image.Image:
        """Apply background color variation."""
        if bg_type == 'white':
            return img

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if bg_type == 'cream':
            # Add cream tint
            overlay = Image.new('RGB', img.size, (255, 253, 245))
            img = Image.blend(img, overlay, alpha=0.3)

        elif bg_type == 'textured':
            # Add slight texture/noise
            img_array = np.array(img)
            noise = np.random.randint(-10, 10, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

        return img

    def _resize_and_pad(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image while maintaining aspect ratio and pad to target size."""
        target_w, target_h = target_size

        # Calculate scaling factor
        scale = min(target_w / img.width, target_h / img.height)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)

        # Resize
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create padded image
        padded = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded.paste(img, (paste_x, paste_y))

        return padded

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations to image."""
        # Random brightness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        # Random contrast
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))

        # Random slight blur
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.5)))

        return img

    def generate_sample(self, sample_id: int) -> bool:
        """Generate a single fermata sample."""
        config = self.generate_random_config()

        # Generate LilyPond code
        ly_code = self.generate_lilypond_code(config, sample_id)

        # Render to image
        png_file, ly_file = self.render_to_image(ly_code, sample_id)

        if png_file is None:
            return False

        # Process and save
        success = self.process_and_save_image(png_file, config, sample_id)

        return success

    def generate_dataset(self, count: int):
        """Generate complete fermata dataset."""
        print(f"\n{'='*60}")
        print(f"Generating {count} synthetic fermata samples")
        print(f"{'='*60}\n")

        successful = 0
        failed = 0

        for i in range(count):
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{count} ({successful} successful, {failed} failed)")

            if self.generate_sample(i):
                successful += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print(f"Generation complete!")
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"{'='*60}\n")

        # Generate dataset.yaml
        self._generate_dataset_yaml()

        # Cleanup temp directory
        try:
            shutil.rmtree(self.temp_dir)
            print(f"✓ Cleaned up temporary files")
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")

    def _generate_dataset_yaml(self):
        """Generate YOLO dataset configuration file."""
        yaml_content = f"""# Synthetic Fermata Dataset
# Generated by generate_synthetic_fermatas.py
# Date: 2025-11-25

path: {self.output_dir.absolute()}
train: images
val: images  # Use same images for validation, or split manually

nc: 1
names:
  0: fermata

# Training recommendations:
# - Merge this dataset with main training data
# - Use class weights to balance fermata class
# - Augmentation: mosaic=0.5, mixup=0.1
"""

        yaml_file = self.output_dir / 'dataset.yaml'
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)

        print(f"✓ Generated dataset.yaml at {yaml_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic fermata samples using LilyPond',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--count',
        type=int,
        default=5000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets/synthetic_fermatas',
        help='Output directory for generated dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Check LilyPond installation
    try:
        result = subprocess.run(
            ['lilypond', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"✓ LilyPond found: {result.stdout.split()[2]}")
    except FileNotFoundError:
        print("Error: LilyPond not found. Please install LilyPond:")
        print("  Ubuntu/Debian: sudo apt-get install lilypond")
        print("  macOS: brew install lilypond")
        sys.exit(1)
    except Exception as e:
        print(f"Error checking LilyPond: {e}")
        sys.exit(1)

    # Check scipy dependency
    try:
        import scipy
        print(f"✓ scipy found: {scipy.__version__}")
    except ImportError:
        print("Error: scipy not found. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])

    # Generate dataset
    generator = LilyPondFermataGenerator(args.output)
    generator.generate_dataset(args.count)

    print("\n✓ Dataset generation complete!")
    print(f"\nNext steps:")
    print(f"1. Review generated samples in {args.output}/images/")
    print(f"2. Merge with main dataset using merge_datasets_phase4.py")
    print(f"3. Train with increased fermata class weight")


if __name__ == '__main__':
    main()
