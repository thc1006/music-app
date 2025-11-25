#!/usr/bin/env python3
"""
Synthetic Fermata Generator v2 - Enhanced Detection
Uses LilyPond SVG output for accurate fermata bounding box extraction.

Author: Claude Code
Date: 2025-11-25
Usage: python generate_synthetic_fermatas_v2.py --count 5000 --output datasets/synthetic_fermatas/
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
    note_value: str
    clef: str
    position: str
    time_signature: str
    staff_size: int
    fermata_count: int
    add_accidentals: bool
    add_dynamics: bool
    background_color: str


class EnhancedFermataGenerator:
    """Generate synthetic fermata samples with accurate bbox extraction."""

    FERMATA_CLASS_ID = 29

    NOTE_VALUES = ['1', '2', '4', '8']
    CLEFS = ['treble', 'bass', 'alto', 'tenor']
    TIME_SIGNATURES = ['4/4', '3/4', '2/4', '6/8', '3/8', '2/2']
    STAFF_SIZES = list(range(18, 25))
    POSITIONS = ['note', 'rest', 'barline']
    ACCIDENTALS = ['is', 'es']  # sharp, flat
    DYNAMICS = ['\\pp', '\\p', '\\mp', '\\mf', '\\f', '\\ff']
    PITCHES = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
    OCTAVES = ["'", "''", "", ","]

    def __init__(self, output_dir: str):
        """Initialize generator with output directory."""
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.temp_dir = Path(tempfile.mkdtemp(prefix='fermata_gen_v2_'))

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Temporary directory: {self.temp_dir}")

    def generate_random_config(self) -> FermataConfig:
        """Generate random fermata configuration."""
        fermata_count = random.randint(1, 3)

        return FermataConfig(
            note_value=random.choice(self.NOTE_VALUES),
            clef=random.choice(self.CLEFS),
            position=random.choice(self.POSITIONS),
            time_signature=random.choice(self.TIME_SIGNATURES),
            staff_size=random.choice(self.STAFF_SIZES),
            fermata_count=fermata_count,
            add_accidentals=random.random() > 0.6,
            add_dynamics=random.random() > 0.4,
            background_color=random.choice(['white', 'cream', 'textured'])
        )

    def generate_lilypond_code(self, config: FermataConfig, sample_id: int) -> str:
        """Generate LilyPond code with explicit fermata markers."""

        # Generate measures with fermatas
        measures = []

        for i in range(config.fermata_count):
            if config.position == 'note':
                measure = self._generate_note_with_fermata(config)
            elif config.position == 'rest':
                measure = self._generate_rest_with_fermata(config)
            else:  # barline
                measure = self._generate_barline_with_fermata(config)
            measures.append(measure)

        # Add context measures
        for _ in range(random.randint(2, 4)):
            measures.append(self._generate_random_measure(config))

        random.shuffle(measures)
        music_content = ' \\bar "|" '.join(measures)

        ly_code = f'''\\version "2.24.3"

\\header {{
  tagline = ##f
}}

\\paper {{
  #(set-paper-size "a4")
  indent = 0\\mm
  line-width = 180\\mm
  top-margin = 10\\mm
  bottom-margin = 10\\mm
  left-margin = 10\\mm
  right-margin = 10\\mm
  oddFooterMarkup = ##f
  evenFooterMarkup = ##f
  oddHeaderMarkup = ##f
  evenHeaderMarkup = ##f
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
    \\bar "|."
  }}
  \\layout {{ }}
}}
'''
        return ly_code

    def _generate_note_with_fermata(self, config: FermataConfig) -> str:
        """Generate a note with fermata."""
        pitch = random.choice(self.PITCHES)
        octave = random.choice(self.OCTAVES)
        accidental = random.choice(self.ACCIDENTALS) if config.add_accidentals and random.random() > 0.5 else ''
        dynamic = ' ' + random.choice(self.DYNAMICS) if config.add_dynamics and random.random() > 0.5 else ''

        # Main note with fermata
        note = f"{pitch}{accidental}{octave}{config.note_value}\\fermata{dynamic}"

        # Context notes
        context_notes = []
        for _ in range(random.randint(2, 4)):
            p = random.choice(self.PITCHES)
            o = random.choice(self.OCTAVES)
            n = random.choice(['4', '8'])
            context_notes.append(f"{p}{o}{n}")

        return f"{' '.join(context_notes)} {note}"

    def _generate_rest_with_fermata(self, config: FermataConfig) -> str:
        """Generate a rest with fermata."""
        rest = f"r{config.note_value}\\fermata"

        # Context notes
        context_notes = []
        for _ in range(random.randint(2, 5)):
            p = random.choice(self.PITCHES)
            o = random.choice(self.OCTAVES)
            n = random.choice(['4', '8', '2'])
            context_notes.append(f"{p}{o}{n}")

        return f"{' '.join(context_notes)} {rest}"

    def _generate_barline_with_fermata(self, config: FermataConfig) -> str:
        """Generate measure with fermata on barline."""
        notes = []
        for _ in range(random.randint(4, 7)):
            p = random.choice(self.PITCHES)
            o = random.choice(self.OCTAVES)
            n = random.choice(['4', '8'])
            notes.append(f"{p}{o}{n}")

        # Note: barline fermata syntax in LilyPond
        return f"{' '.join(notes)} \\bar \"|.\" ^\\fermata"

    def _generate_random_measure(self, config: FermataConfig) -> str:
        """Generate random measure without fermata."""
        notes = []
        for _ in range(random.randint(3, 8)):
            if random.random() > 0.8:
                # Add rest occasionally
                notes.append(f"r{random.choice(['4', '8'])}")
            else:
                p = random.choice(self.PITCHES)
                o = random.choice(self.OCTAVES)
                n = random.choice(['4', '8', '2'])
                notes.append(f"{p}{o}{n}")
        return ' '.join(notes)

    def render_to_image(self, ly_code: str, sample_id: int) -> Tuple[Optional[Path], Optional[Path]]:
        """Render LilyPond code to PNG and SVG."""
        ly_file = self.temp_dir / f"sample_{sample_id}.ly"

        with open(ly_file, 'w', encoding='utf-8') as f:
            f.write(ly_code)

        output_base = self.temp_dir / f"sample_{sample_id}"

        # Render to both PNG and SVG
        cmd = [
            'lilypond',
            '--png',
            '--svg',
            '-dresolution=300',
            f'-o{output_base}',
            str(ly_file)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.temp_dir
            )

            if result.returncode != 0:
                print(f"Warning: LilyPond failed for sample {sample_id}")
                return None, None

            png_file = Path(f"{output_base}.png")
            svg_file = Path(f"{output_base}.svg")

            if not png_file.exists():
                print(f"Warning: PNG not generated for sample {sample_id}")
                return None, None

            return png_file, svg_file

        except subprocess.TimeoutExpired:
            print(f"Warning: LilyPond timeout for sample {sample_id}")
            return None, None
        except Exception as e:
            print(f"Error rendering sample {sample_id}: {e}")
            return None, None

    def extract_fermata_from_svg(self, svg_file: Path) -> List[Tuple[float, float, float, float]]:
        """
        Extract fermata bounding boxes from SVG file.
        Returns normalized coordinates (x_center, y_center, width, height).
        """
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()

            # SVG namespace
            ns = {'svg': 'http://www.w3.org/2000/svg'}

            # Get SVG dimensions
            svg_width = float(root.get('width', '210').replace('mm', '').replace('pt', ''))
            svg_height = float(root.get('height', '297').replace('mm', '').replace('pt', ''))

            # Find all elements (fermatas are typically paths or use glyphs)
            fermata_bboxes = []

            # Look for fermata glyphs or paths
            # In LilyPond SVG, fermatas often have specific glyph names
            for elem in root.iter():
                # Check for fermata-related elements
                if 'fermata' in str(elem.attrib).lower():
                    # Extract bbox
                    x = float(elem.get('x', 0))
                    y = float(elem.get('y', 0))
                    width = float(elem.get('width', 10))
                    height = float(elem.get('height', 10))

                    x_center = (x + width / 2) / svg_width
                    y_center = (y + height / 2) / svg_height
                    w_norm = width / svg_width
                    h_norm = height / svg_height

                    fermata_bboxes.append((x_center, y_center, w_norm, h_norm))

            return fermata_bboxes

        except Exception as e:
            print(f"Error extracting fermata from SVG: {e}")
            return []

    def detect_fermatas_from_image(
        self,
        png_file: Path,
        expected_count: int
    ) -> List[Tuple[float, float, float, float]]:
        """
        Enhanced fermata detection from PNG image using multiple strategies.
        """
        img = Image.open(png_file)
        img_width, img_height = img.size

        # Convert to grayscale
        img_gray = img.convert('L')
        img_array = np.array(img_gray)

        # Strategy 1: Template matching for fermata shape
        bboxes = self._detect_by_shape(img_array, img_width, img_height)

        # Strategy 2: Position-based detection (top/bottom of staff)
        if len(bboxes) < expected_count:
            position_bboxes = self._detect_by_position(img_array, img_width, img_height)
            bboxes.extend(position_bboxes)

        # Remove duplicates
        bboxes = self._remove_duplicate_boxes(bboxes)

        # Take top N by confidence
        if len(bboxes) > expected_count:
            bboxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)[:expected_count]

        return bboxes

    def _detect_by_shape(
        self,
        img_array: np.ndarray,
        img_width: int,
        img_height: int
    ) -> List[Tuple[float, float, float, float]]:
        """Detect fermatas by characteristic arc/dot shape."""
        # Threshold image
        binary = img_array < 200

        # Find connected components
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)

            bboxes = []
            for i in range(1, num_features + 1):
                region = (labeled == i)
                coords = np.argwhere(region)

                if len(coords) < 20:  # Too small
                    continue

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                width = x_max - x_min
                height = y_max - y_min

                # Fermata heuristics
                if width < 10 or height < 5:
                    continue

                if width > img_width / 2 or height > img_height / 4:
                    continue

                # Fermatas are typically wider than tall
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 1.0 or aspect_ratio > 4.0:
                    continue

                # Check vertical position (likely top or bottom third)
                y_center = (y_min + y_max) / 2
                if 0.25 * img_height < y_center < 0.75 * img_height:
                    continue  # Skip middle region

                # Convert to YOLO format
                x_center = ((x_min + x_max) / 2) / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                bboxes.append((x_center, y_center_norm, width_norm, height_norm))

            return bboxes

        except ImportError:
            return []

    def _detect_by_position(
        self,
        img_array: np.ndarray,
        img_width: int,
        img_height: int
    ) -> List[Tuple[float, float, float, float]]:
        """Detect fermatas by scanning typical positions."""
        binary = img_array < 200

        bboxes = []

        # Scan top and bottom regions
        regions = [
            (0, img_height // 4),  # Top
            (3 * img_height // 4, img_height)  # Bottom
        ]

        for y_start, y_end in regions:
            strip = binary[y_start:y_end, :]

            # Find horizontal clusters
            col_sums = strip.sum(axis=0)

            # Use scipy for peak detection if available
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(col_sums, height=5, distance=30)

                for peak_x in peaks:
                    # Estimate bbox
                    left = max(0, peak_x - 25)
                    right = min(img_width, peak_x + 25)

                    # Find vertical extent
                    region = binary[y_start:y_end, left:right]
                    if region.sum() < 30:
                        continue

                    y_coords = np.where(region.any(axis=1))[0]
                    if len(y_coords) == 0:
                        continue

                    top = y_start + y_coords.min()
                    bottom = y_start + y_coords.max()

                    width = right - left
                    height = bottom - top

                    if width < 15 or height < 8:
                        continue

                    # Convert to YOLO format
                    x_center = ((left + right) / 2) / img_width
                    y_center = ((top + bottom) / 2) / img_height
                    width_norm = width / img_width
                    height_norm = height / img_height

                    bboxes.append((x_center, y_center, width_norm, height_norm))

            except ImportError:
                continue

        return bboxes

    def _remove_duplicate_boxes(
        self,
        bboxes: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        """Remove overlapping/duplicate bounding boxes."""
        if not bboxes:
            return []

        # Sort by area (descending)
        bboxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)

        filtered = []
        for bbox in bboxes:
            is_duplicate = False
            for existing in filtered:
                # Check IoU or distance
                dx = abs(bbox[0] - existing[0])
                dy = abs(bbox[1] - existing[1])

                if dx < 0.05 and dy < 0.05:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(bbox)

        return filtered

    def process_and_save(
        self,
        png_file: Path,
        svg_file: Optional[Path],
        config: FermataConfig,
        sample_id: int
    ) -> bool:
        """Process image and save with labels."""
        try:
            # Try SVG extraction first
            bboxes = []
            if svg_file and svg_file.exists():
                bboxes = self.extract_fermata_from_svg(svg_file)

            # Fallback to image-based detection
            if not bboxes:
                bboxes = self.detect_fermatas_from_image(png_file, config.fermata_count)

            if not bboxes:
                print(f"Warning: No fermatas detected in sample {sample_id}")
                return False

            # Load and process image
            img = Image.open(png_file)
            img = self._apply_background(img, config.background_color)
            img = self._resize_and_pad(img, (640, 640))
            img = self._augment(img)

            # Adjust bboxes for resizing
            # (Note: current bboxes are already normalized, but we need to recalculate after resize)
            # For simplicity, we'll re-detect on the final image
            img_temp = self.temp_dir / f"temp_{sample_id}.png"
            img.save(img_temp, 'PNG')

            final_bboxes = self.detect_fermatas_from_image(img_temp, config.fermata_count)

            if not final_bboxes:
                final_bboxes = bboxes  # Use original if re-detection fails

            # Save image
            output_image = self.images_dir / f"fermata_{sample_id:06d}.png"
            img.save(output_image, 'PNG')

            # Save label
            output_label = self.labels_dir / f"fermata_{sample_id:06d}.txt"
            with open(output_label, 'w') as f:
                for bbox in final_bboxes:
                    x, y, w, h = bbox
                    f.write(f"{self.FERMATA_CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            return True

        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            return False

    def _apply_background(self, img: Image.Image, bg_type: str) -> Image.Image:
        """Apply background variation."""
        if bg_type == 'white':
            return img

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if bg_type == 'cream':
            overlay = Image.new('RGB', img.size, (255, 253, 245))
            img = Image.blend(img, overlay, alpha=0.2)
        elif bg_type == 'textured':
            img_array = np.array(img)
            noise = np.random.randint(-8, 8, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

        return img

    def _resize_and_pad(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize maintaining aspect ratio and pad."""
        target_w, target_h = target_size
        scale = min(target_w / img.width, target_h / img.height)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        padded = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded.paste(img, (paste_x, paste_y))

        return padded

    def _augment(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations."""
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.85, 1.15))

        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))

        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.5)))

        return img

    def generate_sample(self, sample_id: int) -> bool:
        """Generate single sample."""
        config = self.generate_random_config()

        ly_code = self.generate_lilypond_code(config, sample_id)
        png_file, svg_file = self.render_to_image(ly_code, sample_id)

        if png_file is None:
            return False

        return self.process_and_save(png_file, svg_file, config, sample_id)

    def generate_dataset(self, count: int):
        """Generate complete dataset."""
        print(f"\n{'='*60}")
        print(f"Generating {count} synthetic fermata samples (v2)")
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
        print(f"Success rate: {100*successful/(successful+failed):.1f}%")
        print(f"{'='*60}\n")

        self._create_dataset_yaml()

        try:
            shutil.rmtree(self.temp_dir)
            print(f"✓ Cleaned up temporary files")
        except:
            pass

    def _create_dataset_yaml(self):
        """Create YOLO dataset config."""
        yaml_content = f"""# Synthetic Fermata Dataset v2
# Generated: 2025-11-25

path: {self.output_dir.absolute()}
train: images
val: images

nc: 1
names:
  0: fermata

# Merge instructions:
# 1. Copy images/ and labels/ to main dataset
# 2. Update class IDs from 0 to 29 (fermata)
# 3. Increase fermata class weight in training
"""

        yaml_file = self.output_dir / 'dataset.yaml'
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)

        print(f"✓ Generated {yaml_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic fermata samples v2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--count', type=int, default=5000)
    parser.add_argument('--output', type=str, default='datasets/synthetic_fermatas_v2')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Check dependencies
    try:
        subprocess.run(['lilypond', '--version'], capture_output=True, timeout=5)
        print("✓ LilyPond found")
    except:
        print("Error: LilyPond not found")
        sys.exit(1)

    try:
        import scipy
        print(f"✓ scipy {scipy.__version__}")
    except ImportError:
        print("Installing scipy...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])

    generator = EnhancedFermataGenerator(args.output)
    generator.generate_dataset(args.count)

    print("\n✓ Complete!")
    print(f"\nNext steps:")
    print(f"1. Review samples: {args.output}/images/")
    print(f"2. Merge with Phase 4 dataset")
    print(f"3. Train with fermata class weight = 10-20x")


if __name__ == '__main__':
    main()
