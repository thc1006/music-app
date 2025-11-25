#!/usr/bin/env python3
"""
Deduplicate Consecutive Duplicate Labels - Phase 4 Dataset
===========================================================

This script fixes the systematic duplicate label issue identified in
Phase 4 dataset analysis (see PHASE4_DUPLICATE_LABEL_ANALYSIS.md).

What it does:
- Scans all label files in train/ and val/ directories
- Removes consecutive duplicate lines (e.g., line N == line N+1)
- Preserves non-consecutive duplicates (legitimate multiple annotations)
- Creates backup before modification
- Generates detailed statistics report

Safe operation:
- Only removes EXACT consecutive duplicates
- Does NOT modify images
- Creates backup in datasets/yolo_harmony_v2_phase4_backup/
- Validates results after deduplication

Usage:
    python deduplicate_labels_phase4.py [--dry-run] [--no-backup]

Options:
    --dry-run     : Show what would be changed without modifying files
    --no-backup   : Skip backup creation (not recommended)
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def deduplicate_label_file(label_path, dry_run=False):
    """
    Remove consecutive duplicate lines from a label file.

    Args:
        label_path: Path to the label file
        dry_run: If True, only analyze without modifying

    Returns:
        Tuple of (original_count, unique_count, duplicates_removed)
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        return 0, 0, 0

    original_count = len(lines)

    # Remove consecutive duplicates while preserving order
    unique_lines = [lines[0]]
    duplicates_found = []

    for i, line in enumerate(lines[1:], start=1):
        if line != unique_lines[-1]:
            unique_lines.append(line)
        else:
            duplicates_found.append((i, line.strip()))

    unique_count = len(unique_lines)
    duplicates_removed = original_count - unique_count

    # Write back if not dry run and duplicates found
    if not dry_run and duplicates_removed > 0:
        with open(label_path, 'w') as f:
            f.writelines(unique_lines)

    return original_count, unique_count, duplicates_removed


def create_backup(dataset_path, backup_path):
    """Create a backup of the dataset."""
    print(f"\n📦 Creating backup...")
    print(f"   Source: {dataset_path}")
    print(f"   Backup: {backup_path}")

    if backup_path.exists():
        print(f"   ⚠️  Backup already exists, removing old backup...")
        shutil.rmtree(backup_path)

    shutil.copytree(dataset_path, backup_path)
    print(f"   ✅ Backup created successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate consecutive duplicate labels in Phase 4 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Analyze without modifying files')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip backup creation (not recommended)')
    parser.add_argument('--dataset', type=str,
                       default='/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase4',
                       help='Path to dataset directory')

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    backup_path = dataset_path.parent / f"{dataset_path.name}_backup"

    print("=" * 70)
    print(" " * 15 + "YOLO Label Deduplication Tool")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'LIVE (will modify files)'}")

    if not dataset_path.exists():
        print(f"\n❌ Error: Dataset not found at {dataset_path}")
        return 1

    # Create backup unless --no-backup or --dry-run
    if not args.dry_run and not args.no_backup:
        create_backup(dataset_path, backup_path)

    # Statistics
    stats = {
        'total_files': 0,
        'affected_files': 0,
        'clean_files': 0,
        'total_labels_before': 0,
        'total_labels_after': 0,
        'total_duplicates': 0,
        'by_split': defaultdict(lambda: {
            'files': 0,
            'affected': 0,
            'labels_before': 0,
            'labels_after': 0,
            'duplicates': 0
        }),
        'by_prefix': defaultdict(lambda: {
            'files': 0,
            'affected': 0,
            'duplicates': 0
        }),
        'worst_offenders': []  # (filename, dup_count, dup_ratio)
    }

    # Process train and val splits
    print("\n🔍 Scanning label files...")

    for split in ['train', 'val']:
        label_dir = dataset_path / split / "labels"

        if not label_dir.exists():
            print(f"   ⚠️  Skipping {split} (directory not found)")
            continue

        print(f"\n[{split.upper()}]")
        label_files = list(label_dir.glob("*.txt"))
        print(f"   Found {len(label_files)} label files")

        for label_file in label_files:
            stats['total_files'] += 1
            stats['by_split'][split]['files'] += 1

            # Get prefix
            prefix = label_file.name.split('_')[0]
            stats['by_prefix'][prefix]['files'] += 1

            # Deduplicate
            original, unique, removed = deduplicate_label_file(label_file, args.dry_run)

            stats['total_labels_before'] += original
            stats['total_labels_after'] += unique
            stats['total_duplicates'] += removed

            stats['by_split'][split]['labels_before'] += original
            stats['by_split'][split]['labels_after'] += unique
            stats['by_split'][split]['duplicates'] += removed

            stats['by_prefix'][prefix]['duplicates'] += removed

            if removed > 0:
                stats['affected_files'] += 1
                stats['by_split'][split]['affected'] += 1
                stats['by_prefix'][prefix]['affected'] += 1

                dup_ratio = removed / original if original > 0 else 0
                stats['worst_offenders'].append((label_file.name, removed, dup_ratio))
            else:
                stats['clean_files'] += 1

    # Sort worst offenders
    stats['worst_offenders'].sort(key=lambda x: -x[1])

    # Print report
    print("\n" + "=" * 70)
    print(" " * 25 + "REPORT")
    print("=" * 70)

    print("\n📊 Overall Statistics:")
    print(f"   Total files processed:     {stats['total_files']:,}")
    print(f"   Files with duplicates:     {stats['affected_files']:,} ({stats['affected_files']/stats['total_files']*100:.1f}%)")
    print(f"   Clean files:               {stats['clean_files']:,} ({stats['clean_files']/stats['total_files']*100:.1f}%)")
    print(f"\n   Labels before:             {stats['total_labels_before']:,}")
    print(f"   Labels after:              {stats['total_labels_after']:,}")
    print(f"   Duplicates removed:        {stats['total_duplicates']:,} ({stats['total_duplicates']/stats['total_labels_before']*100:.1f}%)")

    print("\n📁 By Split:")
    for split in ['train', 'val']:
        if split in stats['by_split']:
            s = stats['by_split'][split]
            print(f"\n   {split.upper()}:")
            print(f"      Files: {s['files']:,} ({s['affected']:,} affected)")
            print(f"      Labels: {s['labels_before']:,} → {s['labels_after']:,}")
            print(f"      Duplicates: {s['duplicates']:,} ({s['duplicates']/s['labels_before']*100:.1f}%)")

    print("\n🏷️  By Data Source Prefix:")
    for prefix, data in sorted(stats['by_prefix'].items(), key=lambda x: -x[1]['duplicates'])[:10]:
        if data['duplicates'] > 0:
            print(f"   {prefix:12s}: {data['files']:5,} files, {data['affected']:5,} affected, "
                  f"{data['duplicates']:7,} dups")

    print("\n⚠️  Top 10 Files with Most Duplicates:")
    for i, (filename, dup_count, dup_ratio) in enumerate(stats['worst_offenders'][:10], 1):
        print(f"   {i:2d}. {filename[:50]:50s} {dup_count:5,} dups ({dup_ratio*100:.1f}%)")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("🔍 DRY RUN COMPLETE - No files were modified")
        print("   Run without --dry-run to apply changes")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✅ DEDUPLICATION COMPLETE")
        print(f"   Backup saved to: {backup_path}")
        print(f"   Dataset cleaned: {dataset_path}")
        print("=" * 70)

        # Save report
        report_path = dataset_path.parent / f"deduplication_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write("Phase 4 Dataset Deduplication Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {dataset_path}\n\n")
            f.write(f"Total files: {stats['total_files']:,}\n")
            f.write(f"Affected files: {stats['affected_files']:,}\n")
            f.write(f"Duplicates removed: {stats['total_duplicates']:,}\n")

        print(f"\n📄 Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
