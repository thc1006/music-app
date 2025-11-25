#!/usr/bin/env python3
"""
Render OpenScore Lieder MusicXML files to images and convert to YOLO format.

This script uses Verovio toolkit to:
1. Render MusicXML to high-resolution PNG images
2. Extract fermata and barline bounding boxes from MusicXML
3. Convert to YOLO format for training
4. Split into train/val sets

Dependencies:
    pip install verovio pillow lxml tqdm

Usage:
    python render_openscore_to_yolo.py --input openscore_lieder/ --output yolo_openscore/
"""

import os
import json
import argparse
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import shutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import verovio
    VEROVIO_AVAILABLE = True
except ImportError:
    VEROVIO_AVAILABLE = False
    print("WARNING: Verovio not installed. Run: pip install verovio")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: Pillow not installed. Run: pip install pillow")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("WARNING: tqdm not installed. Using simple progress. Run: pip install tqdm")
    def tqdm(iterable, **kwargs):
        return iterable


# YOLO class mapping (extend from existing harmony dataset)
YOLO_CLASS_MAPPING = {
    'fermata': 0,  # NEW class for Phase 4.5
    'barline': 23,  # Existing class
    'barline_double': 24,  # Existing class (mAP50=0 in Phase 3)
    'barline_final': 25,  # Could map to barline_double or separate
}


@dataclass
class BoundingBox:
    """Bounding box in YOLO format (normalized)."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_string(self) -> str:
        """Convert to YOLO annotation line."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


def extract_mxl(mxl_path: str) -> Optional[str]:
    """Extract MusicXML from compressed .mxl file."""
    try:
        with zipfile.ZipFile(mxl_path, 'r') as zip_ref:
            xml_files = [f for f in zip_ref.namelist()
                        if f.endswith('.xml') and not f.startswith('META-INF')]
            if not xml_files:
                return None

            xml_file = next((f for f in xml_files if 'score' in f.lower()), xml_files[0])

            with zip_ref.open(xml_file) as xml_content:
                return xml_content.read().decode('utf-8')
    except Exception as e:
        print(f"Error extracting {mxl_path}: {e}")
        return None


def render_with_verovio(musicxml_data: str, output_path: str, scale: int = 100) -> bool:
    """
    Render MusicXML to SVG/PNG using Verovio.

    Args:
        musicxml_data: MusicXML string content
        output_path: Output image path
        scale: Verovio scale factor (default 100 = ~300 DPI)

    Returns:
        True if successful
    """
    if not VEROVIO_AVAILABLE:
        print("ERROR: Verovio not available. Cannot render.")
        return False

    try:
        tk = verovio.toolkit()
        tk.setOptions({
            'scale': scale,
            'adjustPageHeight': True,
            'breaks': 'none',  # Single system for easier bbox extraction
            'pageWidth': 2100,
            'pageHeight': 2970,
        })

        # Load MusicXML
        if not tk.loadData(musicxml_data):
            print(f"ERROR: Verovio failed to load MusicXML")
            return False

        # Render to SVG
        svg_data = tk.renderToSVG(1)  # Page 1

        # Save SVG
        svg_path = output_path.replace('.png', '.svg')
        with open(svg_path, 'w') as f:
            f.write(svg_data)

        # Convert SVG to PNG (requires additional tool or Pillow with svg support)
        # For now, we keep SVG and note that PNG conversion is needed
        # TODO: Implement SVG → PNG conversion (cairosvg, wand, or inkscape)

        return True

    except Exception as e:
        print(f"ERROR rendering with Verovio: {e}")
        return False


def extract_fermata_positions(musicxml_data: str) -> List[Dict]:
    """
    Extract fermata positions from MusicXML.

    Returns:
        List of dictionaries with fermata metadata (without exact pixel coordinates yet)
    """
    fermatas = []

    try:
        root = ET.fromstring(musicxml_data)

        # Find all parts
        for part_idx, part in enumerate(root.findall('.//part')):
            # Find all measures
            for measure_idx, measure in enumerate(part.findall('.//measure')):
                measure_number = measure.get('number', str(measure_idx + 1))

                # Find all notes with fermatas
                for note in measure.findall('.//note'):
                    fermata_elem = note.find('.//fermata')
                    if fermata_elem is not None:
                        # Extract note position info
                        pitch = note.find('pitch')
                        step = pitch.find('step').text if pitch is not None and pitch.find('step') is not None else None
                        octave = pitch.find('octave').text if pitch is not None and pitch.find('octave') is not None else None

                        fermatas.append({
                            'measure': measure_number,
                            'part': part_idx,
                            'pitch': f"{step}{octave}" if step and octave else "rest",
                            'type': fermata_elem.get('type', 'upright'),
                        })

    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")

    return fermatas


def extract_barline_positions(musicxml_data: str) -> List[Dict]:
    """
    Extract barline positions and types from MusicXML.

    Returns:
        List of dictionaries with barline metadata
    """
    barlines = []

    try:
        root = ET.fromstring(musicxml_data)

        for part_idx, part in enumerate(root.findall('.//part')):
            for measure_idx, measure in enumerate(part.findall('.//measure')):
                measure_number = measure.get('number', str(measure_idx + 1))

                # Find all barlines in measure
                for barline in measure.findall('.//barline'):
                    location = barline.get('location', 'right')

                    # Get bar style
                    bar_style = barline.find('bar-style')
                    style = bar_style.text if bar_style is not None else 'regular'

                    # Map to YOLO class
                    if style in ['light-heavy', 'heavy-light']:
                        yolo_class = 'barline_double'
                    elif style == 'light-heavy':
                        yolo_class = 'barline_final'
                    else:
                        yolo_class = 'barline'

                    barlines.append({
                        'measure': measure_number,
                        'part': part_idx,
                        'location': location,
                        'style': style,
                        'yolo_class': yolo_class,
                    })

    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")

    return barlines


def process_mxl_file(mxl_path: Path, output_dir: Path, class_filter: List[str] = None) -> bool:
    """
    Process a single .mxl file: render and extract annotations.

    Args:
        mxl_path: Path to .mxl file
        output_dir: Output directory for images and labels
        class_filter: Only process files containing these classes (e.g., ['fermata'])

    Returns:
        True if processed successfully
    """
    # Extract MusicXML
    musicxml_data = extract_mxl(str(mxl_path))
    if not musicxml_data:
        return False

    # Extract annotations
    fermatas = extract_fermata_positions(musicxml_data)
    barlines = extract_barline_positions(musicxml_data)

    # Apply class filter (skip files without target classes)
    if class_filter:
        has_target = (('fermata' in class_filter and fermatas) or
                     ('barline' in class_filter and barlines))
        if not has_target:
            return False

    # Create output paths
    image_dir = output_dir / 'images'
    label_dir = output_dir / 'labels'
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename (use relative path to preserve structure)
    rel_path = mxl_path.stem  # Just use filename without extension
    output_image = image_dir / f"{rel_path}.png"
    output_label = label_dir / f"{rel_path}.txt"

    # Render image
    if not render_with_verovio(musicxml_data, str(output_image)):
        return False

    # TODO: Generate YOLO annotations
    # This requires knowing exact pixel coordinates from rendered image
    # For now, we just save metadata
    metadata = {
        'source_file': str(mxl_path),
        'fermata_count': len(fermatas),
        'barline_count': len(barlines),
        'fermatas': fermatas,
        'barlines': barlines,
    }

    metadata_path = output_dir / 'metadata' / f"{rel_path}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Processed: {mxl_path.name} ({len(fermatas)} fermatas, {len(barlines)} barlines)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Render OpenScore Lieder to YOLO format')
    parser.add_argument('--input', default='/home/thc1006/dev/music-app/training/datasets/external/openscore_lieder',
                        help='Path to OpenScore Lieder corpus')
    parser.add_argument('--output', default='/home/thc1006/dev/music-app/training/datasets/yolo_openscore_rendered',
                        help='Output directory for YOLO dataset')
    parser.add_argument('--filter', choices=['all', 'fermata', 'barline'], default='fermata',
                        help='Only process files containing specific classes')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to process (for testing)')
    args = parser.parse_args()

    if not VEROVIO_AVAILABLE:
        print("ERROR: Verovio not installed. Run: pip install verovio")
        return

    corpus_dir = Path(args.input)
    output_dir = Path(args.output)

    # Find all .mxl files
    mxl_files = list(corpus_dir.rglob('*.mxl'))
    print(f"Found {len(mxl_files)} MusicXML files")

    if args.limit:
        mxl_files = mxl_files[:args.limit]
        print(f"Processing limited set: {len(mxl_files)} files")

    # Set class filter
    class_filter = None if args.filter == 'all' else [args.filter]

    # Process files
    success_count = 0
    for mxl_path in tqdm(mxl_files, desc="Rendering"):
        if process_mxl_file(mxl_path, output_dir, class_filter):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"Rendering Complete")
    print(f"{'='*80}")
    print(f"Total files processed: {success_count}/{len(mxl_files)}")
    print(f"Output directory: {output_dir}")
    print(f"\n⚠️  NOTE: YOLO annotation generation is incomplete.")
    print(f"    Metadata saved to {output_dir}/metadata/")
    print(f"    Next step: Implement pixel coordinate extraction from Verovio")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
