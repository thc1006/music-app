#!/usr/bin/env python3
"""Verify Phase 5 dataset integrity after merge."""

from pathlib import Path
from collections import Counter

def verify_integrity(dataset_dir: Path):
    """Verify dataset integrity."""
    print("="*80)
    print("PHASE 5 DATASET INTEGRITY CHECK")
    print("="*80)
    
    # Check train split
    train_images = list((dataset_dir / "train" / "images").glob("*"))
    train_labels = list((dataset_dir / "train" / "labels").glob("*.txt"))
    
    print(f"\n📁 Train Split:")
    print(f"  Images: {len(train_images)}")
    print(f"  Labels: {len(train_labels)}")
    
    # Count prefixes
    prefix_count = Counter()
    for img in train_images:
        if img.name.startswith("p4_"):
            prefix_count["Phase 4"] += 1
        elif img.name.startswith("ds2_"):
            prefix_count["DeepScoresV2"] += 1
        elif img.name.startswith("synth_f_"):
            prefix_count["Synthetic Fermata"] += 1
        elif img.name.startswith("synth_b_"):
            prefix_count["Synthetic Barline"] += 1
        else:
            prefix_count["Unknown"] += 1
    
    print(f"\n  Source Breakdown:")
    for source, count in prefix_count.most_common():
        print(f"    {source}: {count}")
    
    # Check val split
    val_images = list((dataset_dir / "val" / "images").glob("*"))
    val_labels = list((dataset_dir / "val" / "labels").glob("*.txt"))
    
    print(f"\n📁 Val Split:")
    print(f"  Images: {len(val_images)}")
    print(f"  Labels: {len(val_labels)}")
    
    # Count prefixes
    prefix_count_val = Counter()
    for img in val_images:
        if img.name.startswith("p4_"):
            prefix_count_val["Phase 4"] += 1
        elif img.name.startswith("ds2_"):
            prefix_count_val["DeepScoresV2"] += 1
        elif img.name.startswith("synth_f_"):
            prefix_count_val["Synthetic Fermata"] += 1
        elif img.name.startswith("synth_b_"):
            prefix_count_val["Synthetic Barline"] += 1
        else:
            prefix_count_val["Unknown"] += 1
    
    print(f"\n  Source Breakdown:")
    for source, count in prefix_count_val.most_common():
        print(f"    {source}: {count}")
    
    # Check for orphaned files
    train_img_stems = {img.stem for img in train_images}
    train_lbl_stems = {lbl.stem for lbl in train_labels}
    
    orphaned_images = train_img_stems - train_lbl_stems
    orphaned_labels = train_lbl_stems - train_img_stems
    
    print(f"\n🔍 Integrity Check:")
    print(f"  Orphaned images (no label): {len(orphaned_images)}")
    print(f"  Orphaned labels (no image): {len(orphaned_labels)}")
    
    if orphaned_images:
        print(f"  ⚠️  Sample orphaned images: {list(orphaned_images)[:5]}")
    if orphaned_labels:
        print(f"  ⚠️  Sample orphaned labels: {list(orphaned_labels)[:5]}")
    
    # Verify label format
    print(f"\n📝 Label Format Check:")
    sample_labels = list(train_labels)[:10]
    
    for lbl in sample_labels:
        with open(lbl, 'r') as f:
            lines = f.readlines()
            for line in lines[:1]:  # Check first line only
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if 0 <= class_id <= 32:
                        print(f"  ✓ {lbl.name}: Class {class_id} - Valid")
                    else:
                        print(f"  ✗ {lbl.name}: Class {class_id} - INVALID (out of range)")
                else:
                    print(f"  ✗ {lbl.name}: Invalid format (expected 5 values)")
                break
    
    print(f"\n✅ Integrity check complete!")
    
    return len(orphaned_images) == 0 and len(orphaned_labels) == 0

if __name__ == "__main__":
    dataset_dir = Path("/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase5")
    success = verify_integrity(dataset_dir)
    exit(0 if success else 1)
