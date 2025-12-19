#!/usr/bin/env python3
"""
OpenScore .mscx to YOLO Format Renderer

Converts MuseScore 3.x (.mscx) files from OpenScore datasets to YOLO format
with precise barline and fermata annotations.

Strategy:
1. Use music21 to read .mscx and extract musical elements
2. Use Verovio to render to SVG (for precise coordinates) and PNG
3. Parse SVG to extract barline positions
4. Generate YOLO format labels

Target classes:
- barline (class 23)
- barline_double (class 24)
- barline_final (class 25)
- barline_repeat (class 26)
- fermata (class 29)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import json

# Third-party imports
try:
    import verovio
    from music21 import converter, stream
    from PIL import Image
    import cairosvg  # For SVG to PNG conversion
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Install with: pip install verovio music21 cairosvg pillow")
    sys.exit(1)


@dataclass
class BoundingBox:
    """YOLO format bounding box"""
    class_id: int
    x_center: float  # normalized [0, 1]
    y_center: float  # normalized [0, 1]
    width: float     # normalized [0, 1]
    height: float    # normalized [0, 1]

    def to_yolo_string(self) -> str:
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class OpenScoreRenderer:
    """Renders OpenScore .mscx files to YOLO format"""

    # Class mappings
    CLASS_MAP = {
        'barline': 23,
        'barline_double': 24,
        'barline_final': 25,
        'barline_repeat': 26,
        'fermata': 29,
    }

    # Verovio rendering options
    RENDER_OPTIONS = {
        'scale': 40,  # Higher scale for better quality
        'pageWidth': 2100,
        'pageHeight': 2970,  # A4 aspect ratio
        'adjustPageHeight': True,
        'breaks': 'auto',
        'footer': 'none',
        'header': 'none',
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.labels_dir = output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Verovio toolkit
        self.tk = verovio.toolkit()

        # Statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'annotations': {class_name: 0 for class_name in self.CLASS_MAP.keys()}
        }

    def process_mscx_file(self, mscx_path: Path) -> bool:
        """
        Process a single .mscx file

        Returns:
            True if successful, False otherwise
        """
        try:
            self.stats['total_files'] += 1
            print(f"Processing: {mscx_path.name}")

            # Step 1: Convert .mscx to MusicXML using music21
            score = converter.parse(str(mscx_path))

            # Export to MusicXML (temporary)
            temp_xml = self.output_dir / f"temp_{mscx_path.stem}.musicxml"
            score.write('musicxml', fp=str(temp_xml))

            # Step 2: Load into Verovio and render
            with open(temp_xml, 'r') as f:
                musicxml_content = f.read()

            self.tk.setOptions(json.dumps(self.RENDER_OPTIONS))
            self.tk.loadData(musicxml_content)

            # Get SVG for coordinate extraction
            svg_content = self.tk.renderToSVG(1)  # Render page 1

            # Get page dimensions
            page_width = self.tk.getPageWidth() * self.RENDER_OPTIONS['scale']
            page_height = self.tk.getPageHeight() * self.RENDER_OPTIONS['scale']

            # Step 3: Parse SVG to extract barlines and fermatas
            bboxes = self._extract_annotations_from_svg(svg_content, page_width, page_height)

            # Step 4: Render to PNG
            output_stem = mscx_path.stem
            png_path = self.images_dir / f"{output_stem}.png"

            # Convert SVG to PNG using cairosvg
            cairosvg.svg2png(
                bytestring=svg_content.encode('utf-8'),
                write_to=str(png_path),
                output_width=int(page_width),
                output_height=int(page_height)
            )

            # Step 5: Save YOLO labels
            label_path = self.labels_dir / f"{output_stem}.txt"
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    f.write(bbox.to_yolo_string() + '\n')

            # Update statistics
            self.stats['successful'] += 1
            for bbox in bboxes:
                class_name = [k for k, v in self.CLASS_MAP.items() if v == bbox.class_id][0]
                self.stats['annotations'][class_name] += 1

            # Clean up temp file
            temp_xml.unlink()

            print(f"  ✓ Generated {len(bboxes)} annotations")
            return True

        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats['failed'] += 1
            return False

    def _extract_annotations_from_svg(
        self,
        svg_content: str,
        page_width: float,
        page_height: float
    ) -> List[BoundingBox]:
        """
        Extract barline and fermata bounding boxes from SVG

        Verovio SVG structure:
        - Barlines: <path class="barline" .../>
        - Double barlines: <g class="barLine"> with specific d attributes
        - Fermatas: <use class="fermata" xlink:href="#E510"/>

        Args:
            svg_content: SVG XML string
            page_width: Page width in pixels
            page_height: Page height in pixels

        Returns:
            List of bounding boxes in YOLO format
        """
        bboxes = []

        try:
            # Parse SVG
            root = ET.fromstring(svg_content)
            ns = {'svg': 'http://www.w3.org/2000/svg', 'xlink': 'http://www.w3.org/1999/xlink'}

            # Extract barlines
            # Verovio represents barlines as <path> elements with class containing "barLine"
            for path in root.findall('.//svg:path', ns):
                class_attr = path.get('class', '')

                if 'barLine' in class_attr:
                    # Parse path data to get coordinates
                    d_attr = path.get('d', '')
                    bbox = self._parse_barline_path(d_attr, page_width, page_height, class_attr)
                    if bbox:
                        bboxes.append(bbox)

            # Extract fermatas
            # Verovio represents fermatas as <use> elements referencing SMuFL glyphs
            for use_elem in root.findall('.//svg:use', ns):
                href = use_elem.get('{http://www.w3.org/1999/xlink}href', '')

                # SMuFL fermata codes: E510 (above), E511 (below)
                if 'E510' in href or 'E511' in href or 'fermata' in use_elem.get('class', '').lower():
                    x = float(use_elem.get('x', 0))
                    y = float(use_elem.get('y', 0))

                    # Fermata approximate size (SMuFL default)
                    fermata_width = 80  # pixels at scale 40
                    fermata_height = 60

                    bbox = BoundingBox(
                        class_id=self.CLASS_MAP['fermata'],
                        x_center=(x + fermata_width / 2) / page_width,
                        y_center=(y + fermata_height / 2) / page_height,
                        width=fermata_width / page_width,
                        height=fermata_height / page_height
                    )
                    bboxes.append(bbox)

        except Exception as e:
            print(f"Warning: SVG parsing error: {e}")

        return bboxes

    def _parse_barline_path(
        self,
        d_attr: str,
        page_width: float,
        page_height: float,
        class_attr: str
    ) -> Optional[BoundingBox]:
        """
        Parse SVG path data to extract barline bounding box

        Barline path format (example):
        "M 500 100 L 500 600"  -> vertical line from (500,100) to (500,600)

        Double barline:
        Multiple vertical lines close together

        Args:
            d_attr: SVG path 'd' attribute
            page_width: Page width in pixels
            page_height: Page height in pixels
            class_attr: CSS class attribute (may contain type hints)

        Returns:
            BoundingBox or None if parsing fails
        """
        try:
            # Parse path coordinates
            coords = []
            tokens = d_attr.replace(',', ' ').split()

            i = 0
            while i < len(tokens):
                if tokens[i] in ['M', 'L']:
                    if i + 2 < len(tokens):
                        x = float(tokens[i + 1])
                        y = float(tokens[i + 2])
                        coords.append((x, y))
                        i += 3
                    else:
                        i += 1
                else:
                    i += 1

            if len(coords) < 2:
                return None

            # Calculate bounding box
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Determine barline type
            class_id = self._classify_barline(x_max - x_min, class_attr)

            # Standard barline width (thin vertical line)
            if x_max - x_min < 5:  # Single line
                x_min -= 5
                x_max += 5

            # Convert to YOLO format
            x_center = (x_min + x_max) / 2 / page_width
            y_center = (y_min + y_max) / 2 / page_height
            width = (x_max - x_min) / page_width
            height = (y_max - y_min) / page_height

            return BoundingBox(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height
            )

        except Exception as e:
            print(f"Warning: Barline path parsing error: {e}")
            return None

    def _classify_barline(self, width: float, class_attr: str) -> int:
        """
        Classify barline type based on width and class attributes

        Args:
            width: Barline width in pixels
            class_attr: CSS class attribute

        Returns:
            YOLO class ID
        """
        class_lower = class_attr.lower()

        # Check for specific types in class name
        if 'repeat' in class_lower or 'repetition' in class_lower:
            return self.CLASS_MAP['barline_repeat']
        elif 'final' in class_lower or 'end' in class_lower:
            return self.CLASS_MAP['barline_final']
        elif 'double' in class_lower:
            return self.CLASS_MAP['barline_double']

        # Classify by width
        if width > 15:  # Thick barline
            return self.CLASS_MAP['barline_final']
        elif width > 5:  # Double barline
            return self.CLASS_MAP['barline_double']
        else:  # Regular barline
            return self.CLASS_MAP['barline']

    def process_directory(self, input_dir: Path, recursive: bool = True) -> None:
        """
        Process all .mscx files in a directory

        Args:
            input_dir: Directory containing .mscx files
            recursive: Whether to search subdirectories
        """
        pattern = "**/*.mscx" if recursive else "*.mscx"
        mscx_files = list(input_dir.glob(pattern))

        print(f"Found {len(mscx_files)} .mscx files in {input_dir}")
        print(f"Output directory: {self.output_dir}\n")

        for mscx_file in mscx_files:
            self.process_mscx_file(mscx_file)

        self._print_statistics()

    def _print_statistics(self) -> None:
        """Print processing statistics"""
        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"\nAnnotations by class:")
        for class_name, count in self.stats['annotations'].items():
            print(f"  {class_name:20s}: {count:6d}")
        print("="*60)

        # Save statistics to JSON
        stats_path = self.output_dir / "processing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert OpenScore .mscx files to YOLO format")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .mscx files"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for YOLO dataset"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Create renderer
    renderer = OpenScoreRenderer(args.output_dir)

    # Process directory
    renderer.process_directory(
        args.input_dir,
        recursive=not args.no_recursive
    )


if __name__ == "__main__":
    main()
