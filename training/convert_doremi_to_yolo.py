#!/usr/bin/env python3
"""
DoReMi Dataset to YOLO Format Converter
========================================

Converts DoReMi OMR annotations (XML format) to YOLO format for training.

DoReMi Dataset Info:
- Source: https://github.com/steinbergmedia/DoReMi
- Total: 5,218 images with corresponding XML annotations
- Original: 78 unique symbol classes
- Target: 33-class YOLO format

XML Format (Parsed_by_page_omr_xml):
- <Page pageIndex="N">: Page indicator
- <Nodes>: Container for all symbols
- <Node>: Individual symbol with:
  - <Id>: Unique identifier
  - <ClassName>: Symbol type (e.g., gClef, noteheadBlack)
  - <Top>, <Left>, <Width>, <Height>: Bounding box in pixels
  - <Mask>: Run-length encoded segmentation mask (optional)
  - <Inlinks>: Relationships to other symbols (optional)

Author: Claude Code
Date: 2025-11-24
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from PIL import Image
import shutil
from tqdm import tqdm
import argparse


# ==============================================================================
# CATEGORY MAPPING: DoReMi (78 classes) -> YOLO (33 classes)
# ==============================================================================

DOREMI_TO_YOLO_MAPPING: Dict[str, Optional[int]] = {
    # -------------------------------------------------------------------------
    # Noteheads -> Class 0, 1
    # -------------------------------------------------------------------------
    'noteheadBlack': 0,         # notehead_filled (135,583 samples)
    'noteheadHalf': 1,          # notehead_hollow (2,138 samples)
    'noteheadWhole': 1,         # notehead_hollow (493 samples)

    # -------------------------------------------------------------------------
    # Stems -> Class 2 (stem unified)
    # -------------------------------------------------------------------------
    'stem': 2,                  # stem (120,339 samples)

    # -------------------------------------------------------------------------
    # Beams and Flags -> Class 3, 4, 5, 6
    # -------------------------------------------------------------------------
    'beam': 3,                  # beam (28,788 samples)
    'flag8thUp': 4,             # flag_8th (1,817 samples)
    'flag8thDown': 4,           # flag_8th (1,010 samples)
    'flag16thUp': 5,            # flag_16th (17,931 samples)
    'flag16thDown': 5,          # flag_16th (113 samples)
    'flag32ndUp': 6,            # flag_32nd (7,244 samples)
    'flag32ndDown': 6,          # flag_32nd (12 samples)

    # -------------------------------------------------------------------------
    # Augmentation and Ties -> Class 7, 8
    # -------------------------------------------------------------------------
    'augmentationDot': 7,       # augmentation_dot (2,463 samples)
    'tie': 8,                   # tie (3,454 samples)

    # -------------------------------------------------------------------------
    # Clefs -> Class 9, 10, 11, 12
    # -------------------------------------------------------------------------
    'gClef': 9,                 # clef_treble (6,173 samples)
    'fClef': 10,                # clef_bass (927 samples)
    'cClef': 11,                # clef_alto (156 samples)
    # Note: clef_tenor (class 12) - not directly in DoReMi

    # -------------------------------------------------------------------------
    # Accidentals -> Class 13, 14, 15, 16, 17
    # -------------------------------------------------------------------------
    'accidentalSharp': 13,      # accidental_sharp (6,052 samples)
    'accidentalFlat': 14,       # accidental_flat (6,056 samples)
    'accidentalNatural': 15,    # accidental_natural (4,592 samples)
    'accidentalDoubleSharp': 16,  # accidental_double_sharp (308 samples)
    'accidentalDoubleFlat': 17,   # accidental_double_flat (240 samples)

    # Microtonal accidentals - map to closest standard accidental
    'accidentalQuarterToneSharpStein': 13,  # -> sharp (86 samples)
    'accidentalQuarterToneFlatStein': 14,   # -> flat (73 samples)
    'accidentalThreeQuarterTonesSharpStein': 13,  # -> sharp (26 samples)

    # -------------------------------------------------------------------------
    # Rests -> Class 18, 19, 20, 21, 22
    # -------------------------------------------------------------------------
    'restWhole': 18,            # rest_whole (336 samples)
    'restHalf': 19,             # rest_half (233 samples)
    'restQuarter': 20,          # rest_quarter (2,191 samples)
    'rest8th': 21,              # rest_8th (14,247 samples)
    'rest16th': 22,             # rest_16th (28,284 samples)
    'rest32nd': 22,             # rest_32nd -> rest_16th (10,001 samples) - merge

    # -------------------------------------------------------------------------
    # Barlines -> Class 23, 24, 25, 26
    # -------------------------------------------------------------------------
    'barline': 23,              # barline (16,467 samples)
    'systemicBarline': 23,      # barline (859 samples) - merge with single barline
    # Note: barline_double (24), barline_final (25), barline_repeat (26)
    # Not directly distinguished in DoReMi

    # -------------------------------------------------------------------------
    # Time Signatures -> Class 27
    # -------------------------------------------------------------------------
    'timeSig2': 27,             # time_signature (293 samples)
    'timeSig3': 27,             # time_signature (298 samples)
    'timeSig4': 27,             # time_signature (573 samples)
    'timeSig5': 27,             # time_signature (77 samples)
    'timeSig6': 27,             # time_signature (40 samples)
    'timeSig7': 27,             # time_signature (37 samples)
    'timeSig8': 27,             # time_signature (518 samples)
    'timeSig9': 27,             # time_signature (25 samples)
    'timeSigCommon': 27,        # time_signature (34 samples) - C
    'timeSigCutCommon': 27,     # time_signature (8 samples) - Cut C
    'timeSignatureComponent': 27,  # time_signature (115 samples)

    # -------------------------------------------------------------------------
    # Key Signature -> Class 28 (Not directly in DoReMi - accidentals form it)
    # -------------------------------------------------------------------------
    # Note: Key signatures are composed of accidentals in DoReMi

    # -------------------------------------------------------------------------
    # Fermata -> Class 29 (Not in DoReMi)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Dynamics -> Class 30, 31
    # -------------------------------------------------------------------------
    'dynamicPiano': 30,         # dynamic_soft (351 samples)
    'dynamicPP': 30,            # dynamic_soft (141 samples)
    'dynamicPPP': 30,           # dynamic_soft (14 samples)
    'dynamicMP': 30,            # dynamic_soft (45 samples)
    'dynamicMF': 30,            # dynamic_soft -> neutral, map to soft (96 samples)

    'dynamicForte': 31,         # dynamic_loud (298 samples)
    'dynamicFF': 31,            # dynamic_loud (63 samples)
    'dynamicFFF': 31,           # dynamic_loud (18 samples)
    'dynamicSforzato': 31,      # dynamic_loud (132 samples)
    'dynamicFortePiano': 31,    # dynamic_loud (28 samples)
    'dynamicForzando': 31,      # dynamic_loud (18 samples)
    'dynamicText': None,        # Exclude - text-based (50 samples)
    'gradualDynamic': None,     # Exclude - crescendo/decrescendo (875 samples)

    # -------------------------------------------------------------------------
    # Ledger Line -> Class 32 (Not directly in DoReMi)
    # -------------------------------------------------------------------------
    # Note: Ledger lines are typically part of noteheads in DoReMi

    # -------------------------------------------------------------------------
    # EXCLUDED: Classes not in our target system
    # -------------------------------------------------------------------------
    'kStaffLine': None,         # Staff lines (33,960 samples) - part of image background
    'slur': None,               # Slur (3,586 samples) - excluded per CLAUDE.md
    'tupletText': None,         # Tuplet numbers (1,015 samples)
    'tupletBracket': None,      # Tuplet brackets (752 samples)

    # Articulations - could be added in future versions
    'articStaccatoAbove': None,     # (1,339 samples)
    'articStaccatoBelow': None,     # (867 samples)
    'articStaccatissimoAbove': None,  # (452 samples)
    'articStaccatissimoBelow': None,  # (332 samples)
    'articAccentAbove': None,       # (316 samples)
    'articAccentBelow': None,       # (369 samples)
    'articTenutoAbove': None,       # (332 samples)
    'articTenutoBelow': None,       # (284 samples)
    'articMarcatoAbove': None,      # (191 samples)
    'articMarcatoBelow': None,      # (21 samples)

    # Ornaments
    'ornamentTrill': None,          # (34 samples)
}


# Target class names (33 classes)
YOLO_CLASS_NAMES = [
    'notehead_filled',          # 0
    'notehead_hollow',          # 1
    'stem',                     # 2
    'beam',                     # 3
    'flag_8th',                 # 4
    'flag_16th',                # 5
    'flag_32nd',                # 6
    'augmentation_dot',         # 7
    'tie',                      # 8
    'clef_treble',              # 9
    'clef_bass',                # 10
    'clef_alto',                # 11
    'clef_tenor',               # 12
    'accidental_sharp',         # 13
    'accidental_flat',          # 14
    'accidental_natural',       # 15
    'accidental_double_sharp',  # 16
    'accidental_double_flat',   # 17
    'rest_whole',               # 18
    'rest_half',                # 19
    'rest_quarter',             # 20
    'rest_8th',                 # 21
    'rest_16th',                # 22
    'barline',                  # 23
    'barline_double',           # 24
    'barline_final',            # 25
    'barline_repeat',           # 26
    'time_signature',           # 27
    'key_signature',            # 28
    'fermata',                  # 29
    'dynamic_soft',             # 30
    'dynamic_loud',             # 31
    'ledger_line',              # 32
]


class DoReMiConverter:
    """Converts DoReMi OMR annotations to YOLO format."""

    def __init__(self, doremi_root: str, output_root: str,
                 train_ratio: float = 0.85, seed: int = 42):
        """
        Initialize converter.

        Args:
            doremi_root: Path to DoReMi_v1 directory
            output_root: Path to output YOLO dataset directory
            train_ratio: Ratio of training data (default 0.85)
            seed: Random seed for reproducibility
        """
        self.doremi_root = Path(doremi_root)
        self.output_root = Path(output_root)
        self.train_ratio = train_ratio
        self.seed = seed

        # Paths
        self.xml_dir = self.doremi_root / 'Parsed_by_page_omr_xml'
        self.img_dir = self.doremi_root / 'Images'

        # Statistics
        self.stats = defaultdict(int)
        self.class_counts = defaultdict(int)
        self.unmapped_classes = defaultdict(int)

    def parse_xml_file(self, xml_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """
        Parse a DoReMi XML file and extract bounding boxes.

        Args:
            xml_path: Path to XML file

        Returns:
            List of (class_id, x_center, y_center, width, height) in YOLO format
        """
        annotations = []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for node in root.findall('.//Node'):
                class_name_elem = node.find('ClassName')
                top_elem = node.find('Top')
                left_elem = node.find('Left')
                width_elem = node.find('Width')
                height_elem = node.find('Height')

                if None in (class_name_elem, top_elem, left_elem, width_elem, height_elem):
                    continue

                class_name = class_name_elem.text
                top = float(top_elem.text)
                left = float(left_elem.text)
                width = float(width_elem.text)
                height = float(height_elem.text)

                # Get YOLO class ID
                yolo_class = DOREMI_TO_YOLO_MAPPING.get(class_name)

                if yolo_class is None:
                    self.unmapped_classes[class_name] += 1
                    continue

                # Store absolute pixel values (will be normalized later)
                annotations.append((yolo_class, left, top, width, height))
                self.class_counts[yolo_class] += 1

        except ET.ParseError as e:
            print(f"  Warning: Failed to parse {xml_path}: {e}")
            self.stats['parse_errors'] += 1

        return annotations

    def normalize_annotations(self, annotations: List[Tuple],
                             img_width: int, img_height: int) -> List[str]:
        """
        Convert absolute pixel coordinates to YOLO normalized format.

        Args:
            annotations: List of (class_id, x, y, w, h) in pixels
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            List of YOLO format strings: "class_id x_center y_center width height"
        """
        yolo_lines = []

        for class_id, x, y, w, h in annotations:
            # Convert to center coordinates
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height

            # Clamp values to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.001, min(1.0, norm_w))
            norm_h = max(0.001, min(1.0, norm_h))

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        return yolo_lines

    def build_file_mapping(self) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Build mapping between image files and XML annotation files.

        Returns:
            Dict mapping piece prefix to list of (image_path, xml_path) tuples
        """
        print("Building file mapping...")

        # Get all XML files
        xml_files = list(self.xml_dir.glob('*.xml'))
        print(f"  Found {len(xml_files)} XML files")

        # Parse XML filenames
        xml_mapping = {}
        for xml_path in xml_files:
            # Pattern: Parsed_<PieceName>-layout-0-muscima_Page_<N>.xml
            match = re.match(r'Parsed_(.+)-layout-0-muscima_Page_(\d+)\.xml', xml_path.name)
            if match:
                prefix = match.group(1)
                page_num = int(match.group(2))
                if prefix not in xml_mapping:
                    xml_mapping[prefix] = {}
                xml_mapping[prefix][page_num] = xml_path

        # Get all image files
        img_files = list(self.img_dir.glob('*.png'))
        print(f"  Found {len(img_files)} image files")

        # Parse image filenames
        img_mapping = {}
        for img_path in img_files:
            # Pattern: <PieceName>-<NNN>.png
            match = re.match(r'(.+)-(\d+)\.png', img_path.name)
            if match:
                prefix = match.group(1)
                page_num = int(match.group(2))
                if prefix not in img_mapping:
                    img_mapping[prefix] = {}
                img_mapping[prefix][page_num] = img_path

        # Build matched pairs
        file_pairs = {}
        matched_count = 0
        unmatched_count = 0

        for prefix in set(xml_mapping.keys()) | set(img_mapping.keys()):
            xml_pages = xml_mapping.get(prefix, {})
            img_pages = img_mapping.get(prefix, {})

            pairs = []
            for page_num in xml_pages:
                if page_num in img_pages:
                    pairs.append((img_pages[page_num], xml_pages[page_num]))
                    matched_count += 1
                else:
                    unmatched_count += 1

            if pairs:
                file_pairs[prefix] = pairs

        print(f"  Matched {matched_count} image-XML pairs from {len(file_pairs)} pieces")
        print(f"  Unmatched: {unmatched_count} XML files")

        return file_pairs

    def convert(self, dry_run: bool = False):
        """
        Run the conversion process.

        Args:
            dry_run: If True, only analyze without creating files
        """
        print("\n" + "=" * 70)
        print("  DoReMi to YOLO Format Converter")
        print("=" * 70)
        print(f"\nSource: {self.doremi_root}")
        print(f"Output: {self.output_root}")
        print(f"Train ratio: {self.train_ratio}")
        print(f"Dry run: {dry_run}")

        # Build file mapping
        file_pairs = self.build_file_mapping()

        # Flatten to list of pairs
        all_pairs = []
        for prefix, pairs in file_pairs.items():
            all_pairs.extend(pairs)

        print(f"\nTotal image-annotation pairs: {len(all_pairs)}")

        # Shuffle and split
        import random
        random.seed(self.seed)
        random.shuffle(all_pairs)

        split_idx = int(len(all_pairs) * self.train_ratio)
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]

        print(f"Train set: {len(train_pairs)} images")
        print(f"Val set: {len(val_pairs)} images")

        if dry_run:
            print("\n[DRY RUN] Analyzing without creating files...")
        else:
            # Create output directories
            train_img_dir = self.output_root / 'train' / 'images'
            train_lbl_dir = self.output_root / 'train' / 'labels'
            val_img_dir = self.output_root / 'val' / 'images'
            val_lbl_dir = self.output_root / 'val' / 'labels'

            for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
                d.mkdir(parents=True, exist_ok=True)

        # Process files
        print("\nProcessing files...")

        for split_name, pairs, img_dir, lbl_dir in [
            ('train', train_pairs,
             self.output_root / 'train' / 'images' if not dry_run else None,
             self.output_root / 'train' / 'labels' if not dry_run else None),
            ('val', val_pairs,
             self.output_root / 'val' / 'images' if not dry_run else None,
             self.output_root / 'val' / 'labels' if not dry_run else None),
        ]:
            print(f"\n  Processing {split_name} set ({len(pairs)} images)...")

            for img_path, xml_path in tqdm(pairs, desc=f"  {split_name}"):
                # Parse annotations
                annotations = self.parse_xml_file(xml_path)

                if not annotations:
                    self.stats['empty_annotations'] += 1
                    continue

                # Get image dimensions
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    print(f"  Warning: Failed to read image {img_path}: {e}")
                    self.stats['image_errors'] += 1
                    continue

                # Normalize annotations
                yolo_lines = self.normalize_annotations(annotations, img_width, img_height)

                if not dry_run:
                    # Generate unique filename
                    new_name = f"doremi_{img_path.stem}"

                    # Copy image
                    dst_img = img_dir / f"{new_name}.png"
                    shutil.copy2(img_path, dst_img)

                    # Write label file
                    dst_lbl = lbl_dir / f"{new_name}.txt"
                    with open(dst_lbl, 'w') as f:
                        f.write('\n'.join(yolo_lines))

                self.stats[f'{split_name}_images'] += 1
                self.stats[f'{split_name}_annotations'] += len(yolo_lines)

        # Create dataset YAML if not dry run
        if not dry_run:
            self._create_yaml_config()

        # Print statistics
        self._print_statistics()

    def _create_yaml_config(self):
        """Create YOLO dataset configuration YAML file."""
        yaml_path = self.output_root / 'doremi_harmony.yaml'

        yaml_content = f"""# DoReMi OMR Dataset - Converted to YOLO Format
# Generated by convert_doremi_to_yolo.py
# Date: 2025-11-24

path: {self.output_root}
train: train/images
val: val/images

nc: 33

names:
"""
        for i, name in enumerate(YOLO_CLASS_NAMES):
            yaml_content += f"  {i}: {name}\n"

        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"\nCreated dataset config: {yaml_path}")

    def _print_statistics(self):
        """Print conversion statistics."""
        print("\n" + "=" * 70)
        print("  Conversion Statistics")
        print("=" * 70)

        print(f"\nProcessed:")
        print(f"  Train images: {self.stats['train_images']}")
        print(f"  Train annotations: {self.stats['train_annotations']}")
        print(f"  Val images: {self.stats['val_images']}")
        print(f"  Val annotations: {self.stats['val_annotations']}")

        print(f"\nIssues:")
        print(f"  Empty annotations: {self.stats['empty_annotations']}")
        print(f"  Parse errors: {self.stats['parse_errors']}")
        print(f"  Image errors: {self.stats['image_errors']}")

        print(f"\nClass distribution (top 20):")
        sorted_counts = sorted(self.class_counts.items(), key=lambda x: -x[1])
        for class_id, count in sorted_counts[:20]:
            print(f"  {class_id:2d} ({YOLO_CLASS_NAMES[class_id]:25s}): {count:>8,}")

        if self.unmapped_classes:
            print(f"\nUnmapped/Excluded classes:")
            sorted_unmapped = sorted(self.unmapped_classes.items(), key=lambda x: -x[1])
            for class_name, count in sorted_unmapped[:15]:
                print(f"  {class_name:35s}: {count:>8,}")


def analyze_doremi_xml_format():
    """Analyze and document the DoReMi XML format."""
    print("\n" + "=" * 70)
    print("  DoReMi XML Format Analysis")
    print("=" * 70)

    print("""
XML Structure:
--------------
<?xml version="1.0" ?>
<Page pageIndex="N">
    <Nodes>
        <Node>
            <Id>unique_id</Id>
            <ClassName>symbol_type</ClassName>
            <Top>y_pixels</Top>
            <Left>x_pixels</Left>
            <Width>w_pixels</Width>
            <Height>h_pixels</Height>
            <Mask>run_length_encoded_mask</Mask>  <!-- Optional -->
            <Inlinks>related_node_ids</Inlinks>    <!-- Optional -->
        </Node>
        ...
    </Nodes>
</Page>

Bounding Box Format:
--------------------
- Origin: Top-left corner of image (0, 0)
- (Left, Top): Top-left corner of bounding box
- (Width, Height): Size of bounding box
- Coordinates are in absolute pixel values

File Naming Pattern:
--------------------
- Image: <PieceName>-<NNN>.png (e.g., "accidentals-001.png")
- XML: Parsed_<PieceName>-layout-0-muscima_Page_<N>.xml
- Mapping: Page number N corresponds to image number NNN

""")

    print("DoReMi Classes (78 unique classes, sorted by frequency):")
    print("-" * 50)

    # Print mapping info
    mapped_classes = [(k, v) for k, v in DOREMI_TO_YOLO_MAPPING.items() if v is not None]
    excluded_classes = [(k, v) for k, v in DOREMI_TO_YOLO_MAPPING.items() if v is None]

    print(f"\nMapped classes: {len(mapped_classes)}")
    print(f"Excluded classes: {len(excluded_classes)}")

    print("\nMapping Summary:")
    for doremi_class, yolo_class in sorted(mapped_classes, key=lambda x: x[1]):
        print(f"  {doremi_class:40s} -> {yolo_class:2d} ({YOLO_CLASS_NAMES[yolo_class]})")


def main():
    parser = argparse.ArgumentParser(
        description='Convert DoReMi OMR annotations to YOLO format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze XML format only
  python convert_doremi_to_yolo.py --analyze

  # Dry run (no files created)
  python convert_doremi_to_yolo.py --dry-run

  # Full conversion
  python convert_doremi_to_yolo.py

  # Custom output path
  python convert_doremi_to_yolo.py --output /path/to/output
        """
    )

    parser.add_argument(
        '--doremi-root',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external/omr_downloads/DoReMi/DoReMi_v1',
        help='Path to DoReMi_v1 directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/yolo_doremi_converted',
        help='Output directory for YOLO dataset'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.85,
        help='Ratio of data for training (default: 0.85)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze without creating output files'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Only print XML format analysis'
    )

    args = parser.parse_args()

    if args.analyze:
        analyze_doremi_xml_format()
        return

    # Check paths
    doremi_root = Path(args.doremi_root)
    if not doremi_root.exists():
        print(f"Error: DoReMi root not found: {doremi_root}")
        return

    # Run conversion
    converter = DoReMiConverter(
        doremi_root=args.doremi_root,
        output_root=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    converter.convert(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
