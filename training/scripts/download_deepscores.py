#!/usr/bin/env python3
"""
Download DeepScoresV2 Dense dataset from Zenodo.

This script downloads the DeepScoresV2 Dense dataset (741.8 MB) which contains
1714 diverse images with annotations for 135 music symbol classes.

Official source: https://zenodo.org/records/4012193
"""

import os
import sys
import hashlib
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm


# Configuration
DOWNLOAD_URL = "https://zenodo.org/api/records/4012193/files/ds2_dense.tar.gz/content"
MD5_CHECKSUM = "7237318e381e6e0848ec30eb82decb83"
OUTPUT_DIR = Path("/home/thc1006/dev/music-app/training/datasets/external/deepscores_v2")
ARCHIVE_NAME = "ds2_dense.tar.gz"


def calculate_md5(filepath, chunk_size=8192):
    """Calculate MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download_file(url, output_path):
    """Download a file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

    print(f"\nDownload complete: {output_path}")


def verify_checksum(filepath, expected_md5):
    """Verify file integrity using MD5 checksum."""
    print(f"\nVerifying checksum...")
    actual_md5 = calculate_md5(filepath)

    if actual_md5 == expected_md5:
        print(f"✓ Checksum verified: {actual_md5}")
        return True
    else:
        print(f"✗ Checksum mismatch!")
        print(f"  Expected: {expected_md5}")
        print(f"  Actual:   {actual_md5}")
        return False


def extract_archive(archive_path, extract_dir):
    """Extract tar.gz archive."""
    print(f"\nExtracting archive to: {extract_dir}")

    with tarfile.open(archive_path, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                tar.extract(member, path=extract_dir)
                pbar.update(1)

    print(f"✓ Extraction complete")


def main():
    """Main download and extraction workflow."""
    print("=" * 70)
    print("DeepScoresV2 Dense Dataset Download")
    print("=" * 70)
    print(f"Dataset info:")
    print(f"  - Size: 741.8 MB")
    print(f"  - Images: 1714 high-quality annotated images")
    print(f"  - Classes: 135 music symbol classes")
    print(f"  - Source: https://zenodo.org/records/4012193")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / ARCHIVE_NAME

    # Check if already downloaded
    if archive_path.exists():
        print(f"\nArchive already exists: {archive_path}")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download. Proceeding to extraction...")
        else:
            print("Re-downloading...")
            download_file(DOWNLOAD_URL, archive_path)
    else:
        download_file(DOWNLOAD_URL, archive_path)

    # Verify checksum
    if not verify_checksum(archive_path, MD5_CHECKSUM):
        print("\n✗ Checksum verification failed. The downloaded file may be corrupted.")
        response = input("Continue with extraction anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborting.")
            sys.exit(1)

    # Extract archive
    extract_archive(archive_path, OUTPUT_DIR)

    # Check extracted contents
    print("\n" + "=" * 70)
    print("Dataset Structure:")
    print("=" * 70)

    # List top-level contents
    contents = sorted(OUTPUT_DIR.iterdir())
    for item in contents:
        if item.is_dir():
            num_files = len(list(item.rglob('*')))
            print(f"  📁 {item.name}/ ({num_files} items)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.2f} MB)")

    # Look for annotations file
    annotations_file = None
    for pattern in ['*.json', '**/annotations.json', '**/deepscores*.json']:
        found = list(OUTPUT_DIR.rglob(pattern))
        if found:
            annotations_file = found[0]
            break

    if annotations_file:
        print(f"\n✓ Annotations file found: {annotations_file}")
    else:
        print(f"\n⚠ Warning: Could not locate annotations.json file")

    print("\n" + "=" * 70)
    print("✓ Download and extraction complete!")
    print(f"Dataset location: {OUTPUT_DIR}")
    print("=" * 70)

    print("\nNext steps:")
    print("  Run the conversion script to extract barline annotations:")
    print("  python /home/thc1006/dev/music-app/training/scripts/convert_deepscores_barlines.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
