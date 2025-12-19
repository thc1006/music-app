#!/usr/bin/env python3
"""Test if Verovio can directly load .mscx files"""

import verovio
from pathlib import Path

# Find a sample .mscx file
sample_file = list(Path("/home/thc1006/dev/music-app/training/datasets/external/omr_downloads/OpenScoreLieder/Lieder-main/scores/").glob("**/*.mscx"))[0]

print(f"Testing with: {sample_file}")

# Try loading with Verovio
tk = verovio.toolkit()

with open(sample_file, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File size: {len(content)} bytes")

# Try to load
try:
    result = tk.loadData(content)
    print(f"✓ Loaded successfully: {result}")

    # Get page count
    page_count = tk.getPageCount()
    print(f"Pages: {page_count}")

    # Try rendering
    tk.setOptions('{"scale": 40}')
    svg = tk.renderToSVG(1)
    print(f"✓ Rendered SVG: {len(svg)} bytes")

    # Check for barlines in SVG
    barline_count = svg.count('barLine')
    print(f"Barlines found in SVG: {barline_count}")

except Exception as e:
    print(f"✗ Error: {e}")
