#!/usr/bin/env python3
"""
Analyze OpenScore Lieder corpus for fermata and barline content.

This script extracts statistics on:
- Fermata elements
- Barline types (double, final, light-heavy, etc.)
- Distribution across files
- Potential for synthetic training data generation
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
import zipfile
import argparse
from typing import Dict, List, Tuple


def extract_mxl(mxl_path: str) -> str:
    """Extract MusicXML from compressed .mxl file."""
    try:
        with zipfile.ZipFile(mxl_path, 'r') as zip_ref:
            # Find the main .xml file (usually named after the container or 'score.xml')
            xml_files = [f for f in zip_ref.namelist() if f.endswith('.xml') and not f.startswith('META-INF')]
            if not xml_files:
                return None

            # Prefer files with 'score' in name, otherwise take first
            xml_file = next((f for f in xml_files if 'score' in f.lower()), xml_files[0])

            with zip_ref.open(xml_file) as xml_content:
                return xml_content.read().decode('utf-8')
    except Exception as e:
        print(f"Error extracting {mxl_path}: {e}")
        return None


def analyze_musicxml(xml_content: str, filepath: str) -> Dict:
    """
    Analyze a MusicXML file for fermatas and barlines.

    Returns:
        Dictionary with counts and examples
    """
    if not xml_content:
        return None

    try:
        root = ET.fromstring(xml_content)

        # MusicXML uses namespaces sometimes, handle both cases
        namespaces = {'': 'http://www.musicxml.org/xsd/MusicXML'}

        # Find all fermatas
        fermatas = root.findall('.//fermata')
        fermatas_ns = root.findall('.//{http://www.musicxml.org/xsd/MusicXML}fermata')
        fermata_count = len(fermatas) + len(fermatas_ns)

        # Find all barlines
        barlines = root.findall('.//barline')
        barlines_ns = root.findall('.//{http://www.musicxml.org/xsd/MusicXML}barline')
        all_barlines = barlines + barlines_ns

        # Categorize barline types
        barline_types = Counter()
        for barline in all_barlines:
            # Look for bar-style element
            bar_style = barline.find('bar-style')
            if bar_style is None:
                bar_style = barline.find('{http://www.musicxml.org/xsd/MusicXML}bar-style')

            if bar_style is not None:
                style_text = bar_style.text or 'regular'
                barline_types[style_text] += 1
            else:
                # Check for repeat elements
                repeat = barline.find('repeat')
                if repeat is None:
                    repeat = barline.find('{http://www.musicxml.org/xsd/MusicXML}repeat')

                if repeat is not None:
                    direction = repeat.get('direction', 'unknown')
                    barline_types[f'repeat-{direction}'] += 1
                else:
                    barline_types['regular'] += 1

        # Count parts (for harmony context)
        parts = root.findall('.//part')
        parts_ns = root.findall('.//{http://www.musicxml.org/xsd/MusicXML}part')
        part_count = len(parts) + len(parts_ns)

        # Get title
        title_elem = root.find('.//work/work-title')
        if title_elem is None:
            title_elem = root.find('.//{http://www.musicxml.org/xsd/MusicXML}work/{http://www.musicxml.org/xsd/MusicXML}work-title')
        title = title_elem.text if title_elem is not None else Path(filepath).stem

        return {
            'filepath': filepath,
            'title': title,
            'fermata_count': fermata_count,
            'barline_types': dict(barline_types),
            'total_barlines': sum(barline_types.values()),
            'part_count': part_count,
            'has_fermata': fermata_count > 0,
            'has_double_barline': 'light-heavy' in barline_types or 'heavy-light' in barline_types,
            'has_final_barline': 'light-heavy' in barline_types,
        }

    except ET.ParseError as e:
        print(f"XML Parse error in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Analyze OpenScore Lieder for fermata/barline content')
    parser.add_argument('--corpus-dir', default='/home/thc1006/dev/music-app/training/datasets/external/openscore_lieder',
                        help='Path to OpenScore Lieder corpus')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Only analyze first N files (for testing)')
    parser.add_argument('--show-examples', type=int, default=10,
                        help='Number of example files to show')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)

    # Find all .mxl files
    mxl_files = list(corpus_dir.rglob('*.mxl'))
    print(f"Found {len(mxl_files)} MusicXML (.mxl) files")

    if args.sample_size:
        mxl_files = mxl_files[:args.sample_size]
        print(f"Analyzing sample of {len(mxl_files)} files...")
    else:
        print(f"Analyzing all {len(mxl_files)} files...")

    # Statistics
    total_files = 0
    files_with_fermata = []
    files_with_double_barline = []
    files_with_final_barline = []
    total_fermatas = 0
    total_barlines = 0
    barline_type_stats = Counter()
    part_count_distribution = Counter()
    errors = 0

    # Process each file
    for i, mxl_path in enumerate(mxl_files, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(mxl_files)} files...")

        # Extract and analyze
        xml_content = extract_mxl(str(mxl_path))
        result = analyze_musicxml(xml_content, str(mxl_path))

        if result is None:
            errors += 1
            continue

        total_files += 1

        # Accumulate statistics
        if result['has_fermata']:
            files_with_fermata.append(result)

        if result['has_double_barline']:
            files_with_double_barline.append(result)

        if result['has_final_barline']:
            files_with_final_barline.append(result)

        total_fermatas += result['fermata_count']
        total_barlines += result['total_barlines']

        for barline_type, count in result['barline_types'].items():
            barline_type_stats[barline_type] += count

        part_count_distribution[result['part_count']] += 1

    print(f"\n{'='*80}")
    print(f"OPENSCORE LIEDER ANALYSIS REPORT")
    print(f"{'='*80}\n")

    # Summary statistics
    print(f"📊 SUMMARY STATISTICS")
    print(f"{'─'*80}")
    print(f"Total files analyzed:          {total_files}")
    print(f"Errors encountered:            {errors}")
    print(f"\n🎵 FERMATA STATISTICS")
    print(f"{'─'*80}")
    print(f"Files with fermatas:           {len(files_with_fermata)} ({len(files_with_fermata)/total_files*100:.1f}%)")
    print(f"Total fermata elements:        {total_fermatas}")
    print(f"Avg fermatas per file:         {total_fermatas/total_files:.2f}")
    print(f"Avg fermatas (files w/ >0):    {total_fermatas/len(files_with_fermata) if files_with_fermata else 0:.2f}")

    print(f"\n📏 BARLINE STATISTICS")
    print(f"{'─'*80}")
    print(f"Total barline elements:        {total_barlines}")
    print(f"Files with double barlines:    {len(files_with_double_barline)} ({len(files_with_double_barline)/total_files*100:.1f}%)")
    print(f"Files with final barlines:     {len(files_with_final_barline)} ({len(files_with_final_barline)/total_files*100:.1f}%)")

    print(f"\n📋 BARLINE TYPE BREAKDOWN")
    print(f"{'─'*80}")
    for barline_type, count in barline_type_stats.most_common():
        percentage = count / total_barlines * 100 if total_barlines > 0 else 0
        print(f"  {barline_type:20s}: {count:6d} ({percentage:5.1f}%)")

    print(f"\n🎼 PART COUNT DISTRIBUTION")
    print(f"{'─'*80}")
    for part_count, file_count in sorted(part_count_distribution.items()):
        print(f"  {part_count} parts: {file_count} files ({file_count/total_files*100:.1f}%)")

    print(f"\n📈 TRAINING DATA POTENTIAL")
    print(f"{'─'*80}")
    print(f"Estimated fermata annotations:  {total_fermatas:,}")
    print(f"Estimated barline annotations:  {total_barlines:,}")
    print(f"Files suitable for rendering:   {total_files} (all valid MusicXML)")
    print(f"\n💡 RECOMMENDATION:")
    print(f"   - OpenScore Lieder contains {total_fermatas:,} fermata annotations")
    print(f"   - This can supplement MUSCIMA++'s 35 fermata annotations significantly")
    print(f"   - Rendering these files with MuseScore/Verovio can generate diverse fermata/barline images")
    print(f"   - Song format (vocal + piano) provides different visual context than orchestral scores")

    # Show examples
    if args.show_examples and files_with_fermata:
        print(f"\n📝 EXAMPLE FILES WITH FERMATAS (top {args.show_examples})")
        print(f"{'─'*80}")

        # Sort by fermata count
        top_fermata_files = sorted(files_with_fermata, key=lambda x: x['fermata_count'], reverse=True)[:args.show_examples]

        for i, file_info in enumerate(top_fermata_files, 1):
            rel_path = Path(file_info['filepath']).relative_to(corpus_dir)
            print(f"\n{i}. {file_info['title']}")
            print(f"   Path: {rel_path}")
            print(f"   Fermatas: {file_info['fermata_count']}")
            print(f"   Barlines: {file_info['total_barlines']} (types: {', '.join(file_info['barline_types'].keys())})")
            print(f"   Parts: {file_info['part_count']}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
