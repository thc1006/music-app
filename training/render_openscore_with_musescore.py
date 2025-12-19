#!/usr/bin/env python3
"""
OpenScore .mscx to YOLO format converter using MuseScore3
Renders .mscx files to images and extracts barline/fermata bounding boxes
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re

# YOLO class mapping
YOLO_CLASSES = {
    'barline': 23,
    'barline_double': 24,
    'barline_final': 25,
    'barline_repeat': 26,
    'fermata': 29,
}

# MuseScore barline type mapping
BARLINE_TYPE_MAP = {
    'normal': 'barline',
    'single': 'barline',
    'double': 'barline_double',
    'end': 'barline_final',
    'final': 'barline_final',
    'start-repeat': 'barline_repeat',
    'end-repeat': 'barline_repeat',
    'end-start-repeat': 'barline_repeat',
}


def find_musescore() -> Tuple[str, bool]:
    """Find MuseScore executable. Returns (path, needs_xvfb)"""
    script_dir = Path(__file__).parent

    # First priority: MuseScore 4 AppImage (required for OpenScore files)
    ms4_path = script_dir / "squashfs-root-ms4" / "bin" / "mscore4portable"
    if ms4_path.exists():
        return str(ms4_path), True  # Needs xvfb-run

    # Second: Check for extracted MuseScore 3 AppImage
    appimage_path = script_dir / "squashfs-root" / "AppRun"
    if appimage_path.exists():
        return str(appimage_path), True  # Needs xvfb-run

    # Fallback to system MuseScore
    for cmd in ['mscore4', 'musescore4', 'mscore3', 'musescore3', 'mscore', 'musescore']:
        result = subprocess.run(['which', cmd], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip(), False

    raise RuntimeError("MuseScore not found. Download MuseScore 4 AppImage from https://musescore.org/en/download")


def convert_mscx_to_musicxml(mscx_path: str, output_dir: str, musescore_cmd: str) -> Optional[str]:
    """Convert .mscx to MusicXML using MuseScore"""
    try:
        basename = Path(mscx_path).stem
        xml_path = os.path.join(output_dir, f"{basename}.musicxml")

        # Use MuseScore CLI to convert
        cmd = [musescore_cmd, '-o', xml_path, mscx_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
        )

        if os.path.exists(xml_path):
            return xml_path
        return None
    except Exception as e:
        return None


def convert_mscx_to_png(mscx_path: str, output_dir: str, musescore_cmd: str, use_xvfb: bool = False) -> List[str]:
    """Convert .mscx to PNG images using MuseScore"""
    try:
        basename = Path(mscx_path).stem
        png_path = os.path.join(output_dir, f"{basename}.png")

        # Use MuseScore CLI to export as PNG (creates multiple pages if needed)
        if use_xvfb:
            # Use xvfb-run with proper screen settings for MuseScore 4
            cmd = [
                'xvfb-run', '-a',
                '-s', '-screen 0 1280x1024x24 -ac +extension GLX +render -noreset',
                musescore_cmd, '-o', png_path, '-r', '150', mscx_path
            ]
            env = os.environ.copy()
        else:
            cmd = [musescore_cmd, '-o', png_path, '-r', '150', mscx_path]
            env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # Increased timeout for MuseScore 4
            env=env
        )

        # MuseScore creates files like basename-01.png, basename-02.png for multi-page
        png_files = []
        # Check for single page output
        if os.path.exists(png_path):
            png_files.append(png_path)

        # Check for multi-page output (MuseScore uses -01, -02 format)
        for i in range(1, 200):
            # Try both formats: -01 and -1
            page_path_padded = os.path.join(output_dir, f"{basename}-{i:02d}.png")
            page_path_simple = os.path.join(output_dir, f"{basename}-{i}.png")

            if os.path.exists(page_path_padded):
                png_files.append(page_path_padded)
            elif os.path.exists(page_path_simple):
                png_files.append(page_path_simple)
            else:
                break

        return png_files
    except Exception as e:
        return []


def parse_mscx_annotations(mscx_path: str) -> Dict:
    """Parse .mscx file to extract barline and fermata information"""
    annotations = {
        'barlines': [],
        'fermatas': [],
    }

    try:
        tree = ET.parse(mscx_path)
        root = tree.getroot()

        # Find all BarLine elements
        for barline in root.iter('BarLine'):
            barline_type = 'normal'
            subtype = barline.find('subtype')
            if subtype is not None and subtype.text:
                barline_type = subtype.text.lower()

            yolo_class = BARLINE_TYPE_MAP.get(barline_type, 'barline')
            annotations['barlines'].append({
                'type': yolo_class,
                'original_type': barline_type,
            })

        # Find all Fermata elements
        for fermata in root.iter('Fermata'):
            annotations['fermatas'].append({
                'type': 'fermata',
            })

        # Also check for fermata in Articulation
        for artic in root.iter('Articulation'):
            subtype = artic.find('subtype')
            if subtype is not None and subtype.text and 'fermata' in subtype.text.lower():
                annotations['fermatas'].append({
                    'type': 'fermata',
                })

    except Exception as e:
        pass

    return annotations


def estimate_bbox_positions(png_path: str, annotations: Dict, page_num: int = 0) -> List[Tuple[int, float, float, float, float]]:
    """
    Estimate bounding box positions based on image analysis.
    Returns list of (class_id, center_x, center_y, width, height) in normalized coords.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(png_path).convert('L')
        img_array = np.array(img)
        height, width = img_array.shape

        # Binarize the image
        threshold = 200
        binary = (img_array < threshold).astype(np.uint8)

        yolo_labels = []

        # Detect vertical lines (potential barlines)
        # Barlines are thin vertical lines
        barline_count = len(annotations['barlines'])
        if barline_count > 0:
            # Scan for vertical lines
            col_sums = np.sum(binary, axis=0)
            # Find peaks in column sums (indicating vertical lines)
            mean_sum = np.mean(col_sums)
            std_sum = np.std(col_sums)
            threshold_val = mean_sum + 1.5 * std_sum

            # Find positions where column sum exceeds threshold
            potential_barlines = []
            in_barline = False
            start_x = 0

            for x in range(width):
                if col_sums[x] > threshold_val:
                    if not in_barline:
                        in_barline = True
                        start_x = x
                else:
                    if in_barline:
                        in_barline = False
                        end_x = x
                        # Check if this is a thin vertical line (barline characteristic)
                        if end_x - start_x < 20:  # Barlines are thin
                            center_x = (start_x + end_x) / 2 / width
                            # Find vertical extent
                            col_data = binary[:, start_x:end_x]
                            rows_with_ink = np.any(col_data > 0, axis=1)
                            if np.any(rows_with_ink):
                                y_indices = np.where(rows_with_ink)[0]
                                y_start = y_indices[0]
                                y_end = y_indices[-1]
                                center_y = (y_start + y_end) / 2 / height
                                box_width = (end_x - start_x) / width
                                box_height = (y_end - y_start) / height
                                if box_height > 0.1:  # Barlines should be tall
                                    potential_barlines.append((center_x, center_y, box_width, box_height))

            # Assign barline types based on detected positions
            # Sort detected barlines by x position
            potential_barlines.sort(key=lambda x: x[0])

            # Use annotations to determine types
            barline_idx = 0
            for pos in potential_barlines:
                if barline_idx < len(annotations['barlines']):
                    barline_info = annotations['barlines'][barline_idx]
                    class_id = YOLO_CLASSES.get(barline_info['type'], YOLO_CLASSES['barline'])
                    yolo_labels.append((class_id, pos[0], pos[1], max(pos[2], 0.01), pos[3]))
                    barline_idx += 1

        # Detect fermatas (look for circular/dot-like shapes in upper portion)
        fermata_count = len(annotations['fermatas'])
        if fermata_count > 0:
            # Fermatas are usually in the upper portion of the staff
            upper_region = binary[:height//3, :]

            # Use connected component analysis
            from scipy import ndimage
            labeled, num_features = ndimage.label(upper_region)

            fermata_candidates = []
            for i in range(1, num_features + 1):
                component = (labeled == i)
                y_indices, x_indices = np.where(component)
                if len(y_indices) > 10:  # Minimum size
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    comp_width = x_max - x_min
                    comp_height = y_max - y_min
                    # Fermatas are roughly circular/arc shaped
                    aspect_ratio = comp_width / max(comp_height, 1)
                    if 0.5 < aspect_ratio < 3.0 and comp_width > 10:
                        center_x = (x_min + x_max) / 2 / width
                        center_y = (y_min + y_max) / 2 / height
                        box_width = comp_width / width
                        box_height = comp_height / height
                        fermata_candidates.append((center_x, center_y, box_width, box_height))

            # Sort by x position and take the first N matching annotation count
            fermata_candidates.sort(key=lambda x: x[0])
            for i, pos in enumerate(fermata_candidates[:fermata_count]):
                class_id = YOLO_CLASSES['fermata']
                yolo_labels.append((class_id, pos[0], pos[1], max(pos[2], 0.02), max(pos[3], 0.02)))

        return yolo_labels

    except Exception as e:
        return []


def process_single_file(args: Tuple[str, str, str, str, bool]) -> Dict:
    """Process a single .mscx file"""
    mscx_path, output_dir, musescore_cmd, file_id, use_xvfb = args

    result = {
        'file_id': file_id,
        'success': False,
        'images': 0,
        'barlines': 0,
        'fermatas': 0,
        'error': None,
    }

    try:
        # Create temp directory for this file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert to PNG
            png_files = convert_mscx_to_png(mscx_path, temp_dir, musescore_cmd, use_xvfb)

            if not png_files:
                result['error'] = "Failed to convert to PNG"
                return result

            # Parse annotations from mscx
            annotations = parse_mscx_annotations(mscx_path)

            # Process each page
            images_dir = os.path.join(output_dir, 'images', 'train')
            labels_dir = os.path.join(output_dir, 'labels', 'train')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            for page_idx, png_path in enumerate(png_files):
                # Copy image
                output_image = os.path.join(images_dir, f"{file_id}_p{page_idx}.png")
                shutil.copy2(png_path, output_image)

                # Generate YOLO labels
                labels = estimate_bbox_positions(png_path, annotations, page_idx)

                # Write labels
                output_label = os.path.join(labels_dir, f"{file_id}_p{page_idx}.txt")
                with open(output_label, 'w') as f:
                    for label in labels:
                        class_id, cx, cy, w, h = label
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                result['images'] += 1
                result['barlines'] += sum(1 for l in labels if l[0] in [23, 24, 25, 26])
                result['fermatas'] += sum(1 for l in labels if l[0] == 29)

            result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert OpenScore .mscx to YOLO format')
    parser.add_argument('--input', type=str, required=True, help='Input directory with .mscx files')
    parser.add_argument('--output', type=str, required=True, help='Output directory for YOLO dataset')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process')
    args = parser.parse_args()

    # Find MuseScore
    musescore_cmd, use_xvfb = find_musescore()
    print(f"Using MuseScore: {musescore_cmd}")
    print(f"Using xvfb-run: {use_xvfb}")

    # Find all .mscx files
    input_path = Path(args.input)
    mscx_files = list(input_path.rglob('*.mscx'))

    if args.limit:
        mscx_files = mscx_files[:args.limit]

    print(f"Found {len(mscx_files)} .mscx files")

    # Prepare output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare processing tasks
    tasks = []
    for i, mscx_file in enumerate(mscx_files):
        file_id = f"openscore_{i:05d}"
        tasks.append((str(mscx_file), str(output_path), musescore_cmd, file_id, use_xvfb))

    # Process files
    results = []
    total_images = 0
    total_barlines = 0
    total_fermatas = 0
    errors = 0

    print(f"Processing with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Converting") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result['success']:
                    total_images += result['images']
                    total_barlines += result['barlines']
                    total_fermatas += result['fermatas']
                else:
                    errors += 1

                pbar.update(1)
                pbar.set_postfix({
                    'images': total_images,
                    'barlines': total_barlines,
                    'fermatas': total_fermatas,
                    'errors': errors
                })

    # Create YAML config
    yaml_content = f"""# OpenScore YOLO Dataset
path: {output_path.absolute()}
train: images/train
val: images/train

nc: 33
names:
  0: notehead_filled
  1: notehead_hollow
  2: stem
  3: beam
  4: flag_8th
  5: flag_16th
  6: flag_32nd
  7: augmentation_dot
  8: tie
  9: clef_treble
  10: clef_bass
  11: clef_alto
  12: clef_tenor
  13: accidental_sharp
  14: accidental_flat
  15: accidental_natural
  16: accidental_double_sharp
  17: accidental_double_flat
  18: rest_whole
  19: rest_half
  20: rest_quarter
  21: rest_8th
  22: rest_16th
  23: barline
  24: barline_double
  25: barline_final
  26: barline_repeat
  27: time_signature
  28: key_signature
  29: fermata
  30: dynamic_soft
  31: dynamic_loud
  32: ledger_line
"""

    yaml_path = output_path / 'openscore.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Save stats
    stats = {
        'total_files': len(mscx_files),
        'successful': len(mscx_files) - errors,
        'errors': errors,
        'total_images': total_images,
        'total_barlines': total_barlines,
        'total_fermatas': total_fermatas,
    }

    stats_path = output_path / 'conversion_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print("Conversion Complete!")
    print(f"{'='*50}")
    print(f"Total files processed: {len(mscx_files)}")
    print(f"Successful: {len(mscx_files) - errors}")
    print(f"Errors: {errors}")
    print(f"Total images: {total_images}")
    print(f"Total barlines: {total_barlines}")
    print(f"Total fermatas: {total_fermatas}")
    print(f"\nOutput: {output_path}")
    print(f"YAML config: {yaml_path}")


if __name__ == '__main__':
    main()
