#!/usr/bin/env python3
"""
Extract Fermata annotations from OpenScore .mscx files

Strategy: Since we cannot easily render .mscx to images with precise coordinates,
we will instead:
1. Analyze .mscx files to extract fermata locations and other metadata
2. Create a mapping file for later processing with MuseScore (if/when available)
3. For now, focus on understanding the fermata distribution and structure

This script will:
- Parse all .mscx files
- Extract fermata positions and types
- Generate statistics and analysis
- Prepare for future rendering pipeline
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class FermataInfo:
    """Information about a fermata"""
    file_path: str
    measure_number: int
    staff_number: int
    subtype: str  # fermataAbove, fermataBelow
    time_stretch: Optional[float] = None
    offset_x: float = 0.0
    offset_y: float = 0.0
    visible: bool = True


@dataclass
class BarlineInfo:
    """Information about a barline"""
    file_path: str
    measure_number: int
    barline_type: str  # normal, double, final, repeat_start, repeat_end


class OpenScoreMscxAnalyzer:
    """Analyze OpenScore .mscx files for annotations"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fermatas: List[FermataInfo] = []
        self.barlines: List[BarlineInfo] = []

        self.stats = {
            'total_files': 0,
            'files_with_fermata': 0,
            'total_fermatas': 0,
            'total_barlines': 0,
            'fermata_subtypes': defaultdict(int),
            'barline_types': defaultdict(int),
            'errors': []
        }

    def process_mscx_file(self, mscx_path: Path) -> bool:
        """
        Process a single .mscx file

        Returns:
            True if successful
        """
        try:
            self.stats['total_files'] += 1

            # Parse XML
            tree = ET.parse(mscx_path)
            root = tree.getroot()

            # Extract fermatas and barlines
            fermatas_found = self._extract_fermatas(root, str(mscx_path))
            barlines_found = self._extract_barlines(root, str(mscx_path))

            if fermatas_found > 0:
                self.stats['files_with_fermata'] += 1

            self.stats['total_fermatas'] += fermatas_found
            self.stats['total_barlines'] += barlines_found

            if self.stats['total_files'] % 100 == 0:
                print(f"Processed {self.stats['total_files']} files, "
                      f"fermatas: {self.stats['total_fermatas']}, "
                      f"barlines: {self.stats['total_barlines']}")

            return True

        except Exception as e:
            error_msg = f"Error processing {mscx_path.name}: {e}"
            self.stats['errors'].append(error_msg)
            if len(self.stats['errors']) <= 10:  # Only print first 10 errors
                print(f"  ✗ {error_msg}")
            return False

    def _extract_fermatas(self, root: ET.Element, file_path: str) -> int:
        """
        Extract fermata elements from .mscx XML

        Fermata structure in .mscx:
        <Fermata>
            <subtype>fermataAbove</subtype>
            <timeStretch>1.3</timeStretch>  <!-- optional -->
            <offset x="0" y="-5"/>          <!-- optional -->
            <visible>0</visible>             <!-- optional, 0=invisible -->
        </Fermata>

        Args:
            root: XML root element
            file_path: File path for reference

        Returns:
            Number of fermatas found
        """
        count = 0
        measure_number = 0
        staff_number = 0

        # Navigate through the score structure
        for staff in root.findall('.//Staff'):
            staff_number += 1
            measure_number = 0

            for measure in staff.findall('.//Measure'):
                measure_number += 1

                # Find all fermatas in this measure
                for fermata in measure.findall('.//Fermata'):
                    # Extract fermata properties
                    subtype = fermata.findtext('subtype', 'fermataAbove')

                    # Time stretch (tempo modification)
                    time_stretch_elem = fermata.find('timeStretch')
                    time_stretch = float(time_stretch_elem.text) if time_stretch_elem is not None else None

                    # Offset (visual position adjustment)
                    offset = fermata.find('offset')
                    offset_x = float(offset.get('x', 0)) if offset is not None else 0.0
                    offset_y = float(offset.get('y', 0)) if offset is not None else 0.0

                    # Visibility
                    visible_elem = fermata.find('visible')
                    visible = visible_elem.text != '0' if visible_elem is not None else True

                    # Create FermataInfo
                    info = FermataInfo(
                        file_path=file_path,
                        measure_number=measure_number,
                        staff_number=staff_number,
                        subtype=subtype,
                        time_stretch=time_stretch,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        visible=visible
                    )

                    self.fermatas.append(info)
                    self.stats['fermata_subtypes'][subtype] += 1
                    count += 1

        return count

    def _extract_barlines(self, root: ET.Element, file_path: str) -> int:
        """
        Extract barline information from .mscx

        In .mscx, barlines are mostly implicit (measure boundaries).
        Explicit barlines appear in specific contexts:
        - Repeat marks
        - Final barlines (often in last measure)
        - Double barlines (section markers)

        We will count total measures (implicit barlines) and mark special types.

        Args:
            root: XML root element
            file_path: File path for reference

        Returns:
            Number of barlines found
        """
        count = 0

        # Get all measures (each has an implicit barline at the end)
        measures = root.findall('.//Measure')

        for i, measure in enumerate(measures):
            measure_number = i + 1

            # Default: normal barline
            barline_type = 'normal'

            # Check for special barline types
            # Repeat starts/ends
            start_repeat = measure.find('.//startRepeat')
            end_repeat = measure.find('.//endRepeat')

            if start_repeat is not None:
                barline_type = 'repeat_start'
            elif end_repeat is not None:
                barline_type = 'repeat_end'

            # Final barline (usually last measure)
            if measure_number == len(measures):
                # Check if explicitly marked as final
                if barline_type == 'normal':  # Don't override repeats
                    barline_type = 'final'

            # Add barline info
            info = BarlineInfo(
                file_path=file_path,
                measure_number=measure_number,
                barline_type=barline_type
            )

            self.barlines.append(info)
            self.stats['barline_types'][barline_type] += 1
            count += 1

        return count

    def process_directory(self, input_dir: Path, recursive: bool = True, limit: Optional[int] = None) -> None:
        """
        Process all .mscx files in a directory

        Args:
            input_dir: Directory containing .mscx files
            recursive: Whether to search subdirectories
            limit: Maximum number of files to process (for testing)
        """
        pattern = "**/*.mscx" if recursive else "*.mscx"
        mscx_files = list(input_dir.glob(pattern))

        if limit:
            mscx_files = mscx_files[:limit]

        print(f"Found {len(mscx_files)} .mscx files in {input_dir}")
        print(f"Output directory: {self.output_dir}\n")

        for mscx_file in mscx_files:
            self.process_mscx_file(mscx_file)

        self._save_results()
        self._print_statistics()

    def _save_results(self) -> None:
        """Save extracted annotations to JSON files"""

        # Save fermatas
        fermatas_path = self.output_dir / "fermatas.json"
        with open(fermatas_path, 'w') as f:
            json.dump([asdict(f) for f in self.fermatas], f, indent=2)
        print(f"\n✓ Saved {len(self.fermatas)} fermatas to {fermatas_path}")

        # Save barlines
        barlines_path = self.output_dir / "barlines.json"
        with open(barlines_path, 'w') as f:
            json.dump([asdict(b) for b in self.barlines], f, indent=2)
        print(f"✓ Saved {len(self.barlines)} barlines to {barlines_path}")

        # Save statistics
        stats_path = self.output_dir / "extraction_stats.json"

        # Convert defaultdict to regular dict for JSON serialization
        stats_for_json = {
            **self.stats,
            'fermata_subtypes': dict(self.stats['fermata_subtypes']),
            'barline_types': dict(self.stats['barline_types'])
        }

        with open(stats_path, 'w') as f:
            json.dump(stats_for_json, f, indent=2)
        print(f"✓ Saved statistics to {stats_path}")

    def _print_statistics(self) -> None:
        """Print extraction statistics"""
        print("\n" + "="*60)
        print("EXTRACTION STATISTICS")
        print("="*60)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Files with fermatas: {self.stats['files_with_fermata']} "
              f"({100 * self.stats['files_with_fermata'] / max(1, self.stats['total_files']):.1f}%)")
        print(f"\nTotal fermatas: {self.stats['total_fermatas']}")
        print(f"Total barlines: {self.stats['total_barlines']}")

        print(f"\nFermata subtypes:")
        for subtype, count in sorted(self.stats['fermata_subtypes'].items()):
            print(f"  {subtype:20s}: {count:6d}")

        print(f"\nBarline types:")
        for btype, count in sorted(self.stats['barline_types'].items()):
            print(f"  {btype:20s}: {count:6d}")

        if self.stats['errors']:
            print(f"\nErrors: {len(self.stats['errors'])}")
            if len(self.stats['errors']) <= 10:
                for error in self.stats['errors']:
                    print(f"  - {error}")
            else:
                print(f"  (showing first 10 of {len(self.stats['errors'])} errors)")
                for error in self.stats['errors'][:10]:
                    print(f"  - {error}")

        print("="*60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract annotations from OpenScore .mscx files"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .mscx files (e.g., OpenScoreLieder/scores)"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for JSON extraction results"
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

    # Create analyzer
    analyzer = OpenScoreMscxAnalyzer(args.output_dir)

    # Process directory
    analyzer.process_directory(
        args.input_dir,
        recursive=not args.no_recursive,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
