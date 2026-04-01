#!/usr/bin/env python3
"""Create Phase 5 dataset: remove stem annotations, remap class IDs.

Strategy: "dirty train, clean val"
  - Train: original 32,555 images (all sources, noisy labels OK)
  - Val: cleaned_v2 2,867 images (high quality)
  - Both: stem (class 2) annotations removed, class IDs remapped

Stems account for 24% of annotations (606K / 2.53M) but are only 2.6px wide
— borderline undetectable. Removing them:
  1. Reduces TAL pressure by ~24%
  2. Eliminates 113K tiny-box warnings
  3. Lets the model focus on detectable symbols
  4. Stems recovered deterministically in SymbolAssembler.kt

Output: training/datasets/yolo_harmony_v2_phase5_nostem/
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path("/home/thc1006/dev/music-app/training/datasets")

# Sources
ORIGINAL = ROOT / "yolo_harmony_v2_phase8_final"
CLEANED_V2 = ROOT / "yolo_harmony_v2_phase8_cleaned_v2"

# Output
OUTPUT = ROOT / "yolo_harmony_v2_phase5_nostem"

STEM_CLASS_ID = 2  # class to remove

# Original 33 classes → 32 classes after removing stem
ORIGINAL_NAMES = [
    "notehead_filled",      # 0 → 0
    "notehead_hollow",      # 1 → 1
    "stem",                 # 2 → REMOVED
    "beam",                 # 3 → 2
    "flag_8th",             # 4 → 3
    "flag_16th",            # 5 → 4
    "flag_32nd",            # 6 → 5
    "augmentation_dot",     # 7 → 6
    "tie",                  # 8 → 7
    "clef_treble",          # 9 → 8
    "clef_bass",            # 10 → 9
    "clef_alto",            # 11 → 10
    "clef_tenor",           # 12 → 11
    "accidental_sharp",     # 13 → 12
    "accidental_flat",      # 14 → 13
    "accidental_natural",   # 15 → 14
    "accidental_double_sharp",  # 16 → 15
    "accidental_double_flat",   # 17 → 16
    "rest_whole",           # 18 → 17
    "rest_half",            # 19 → 18
    "rest_quarter",         # 20 → 19
    "rest_8th",             # 21 → 20
    "rest_16th",            # 22 → 21
    "barline",              # 23 → 22
    "barline_double",       # 24 → 23
    "barline_final",        # 25 → 24
    "barline_repeat",       # 26 → 25
    "time_signature",       # 27 → 26
    "key_signature",        # 28 → 27
    "fermata",              # 29 → 28
    "dynamic_soft",         # 30 → 29
    "dynamic_loud",         # 31 → 30
    "ledger_line",          # 32 → 31
]

# Build remap table: old_id → new_id (or -1 for removed)
REMAP = {}
new_id = 0
for old_id, name in enumerate(ORIGINAL_NAMES):
    if old_id == STEM_CLASS_ID:
        REMAP[old_id] = -1  # removed
    else:
        REMAP[old_id] = new_id
        new_id += 1

NEW_NAMES = [n for i, n in enumerate(ORIGINAL_NAMES) if i != STEM_CLASS_ID]
assert len(NEW_NAMES) == 32


def process_labels(src_dir: Path, dst_dir: Path) -> dict[str, int]:
    """Remove stem annotations and remap class IDs.

    Returns stats dict with counts.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "lines_in": 0, "lines_out": 0, "stems_removed": 0, "empty_after": 0}

    label_files = sorted(src_dir.glob("*.txt"))
    for lf in label_files:
        stats["files"] += 1
        lines = lf.read_text().strip().split("\n") if lf.stat().st_size > 0 else []
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            stats["lines_in"] += 1
            parts = line.strip().split()
            old_cls = int(parts[0])
            if old_cls == STEM_CLASS_ID:
                stats["stems_removed"] += 1
                continue
            new_cls = REMAP.get(old_cls, old_cls)
            if new_cls == -1:
                continue
            new_lines.append(f"{new_cls} {' '.join(parts[1:])}")
            stats["lines_out"] += 1

        out_path = dst_dir / lf.name
        if new_lines:
            out_path.write_text("\n".join(new_lines) + "\n")
        else:
            # Keep empty file so Ultralytics knows image has no objects
            out_path.write_text("")
            stats["empty_after"] += 1

    return stats


def symlink_images(src_dir: Path, dst_dir: Path) -> int:
    """Create symlinks for images (zero disk cost)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src_dir.iterdir()):
        if img.suffix.lower() in (".png", ".jpg", ".jpeg"):
            target = dst_dir / img.name
            if target.exists() or target.is_symlink():
                target.unlink()
            # Resolve through any existing symlinks to get absolute path
            real_path = img.resolve()
            target.symlink_to(real_path)
            count += 1
    return count


def write_yaml(output_dir: Path) -> None:
    """Write dataset YAML with 32 classes."""
    lines = [
        f"path: {output_dir}",
        "train: train/images",
        "val: val/images",
        f"nc: {len(NEW_NAMES)}",
        "names:",
    ]
    for i, name in enumerate(NEW_NAMES):
        lines.append(f"  {i}: {name}")
    (output_dir / "harmony_phase5_nostem.yaml").write_text("\n".join(lines) + "\n")


def main() -> int:
    print("=" * 60)
    print("Phase 5 Dataset: Remove Stems + Dirty Train / Clean Val")
    print("=" * 60)

    if OUTPUT.exists():
        print(f"\nOutput directory already exists: {OUTPUT}")
        print("Removing and recreating...")
        shutil.rmtree(OUTPUT)

    OUTPUT.mkdir(parents=True)

    # --- Train: original 32K images ---
    print("\n--- Train Set (original 32K, all sources) ---")
    train_img_src = ORIGINAL / "train" / "images"
    train_lbl_src = ORIGINAL / "train" / "labels"
    train_img_dst = OUTPUT / "train" / "images"
    train_lbl_dst = OUTPUT / "train" / "labels"

    n_img = symlink_images(train_img_src, train_img_dst)
    print(f"  Images symlinked: {n_img}")

    train_stats = process_labels(train_lbl_src, train_lbl_dst)
    print(f"  Label files: {train_stats['files']}")
    print(f"  Annotations in:  {train_stats['lines_in']:,}")
    print(f"  Stems removed:   {train_stats['stems_removed']:,}")
    print(f"  Annotations out: {train_stats['lines_out']:,}")
    print(f"  Empty after:     {train_stats['empty_after']}")

    # --- Val: cleaned_v2 2,867 images ---
    print("\n--- Val Set (cleaned_v2, high quality) ---")
    val_img_src = CLEANED_V2 / "val" / "images"
    val_lbl_src = CLEANED_V2 / "val" / "labels"
    val_img_dst = OUTPUT / "val" / "images"
    val_lbl_dst = OUTPUT / "val" / "labels"

    n_img = symlink_images(val_img_src, val_img_dst)
    print(f"  Images symlinked: {n_img}")

    val_stats = process_labels(val_lbl_src, val_lbl_dst)
    print(f"  Label files: {val_stats['files']}")
    print(f"  Annotations in:  {val_stats['lines_in']:,}")
    print(f"  Stems removed:   {val_stats['stems_removed']:,}")
    print(f"  Annotations out: {val_stats['lines_out']:,}")
    print(f"  Empty after:     {val_stats['empty_after']}")

    # --- Write YAML ---
    write_yaml(OUTPUT)
    print(f"\nYAML: {OUTPUT / 'harmony_phase5_nostem.yaml'}")
    print(f"Classes: {len(NEW_NAMES)} (was 33, removed stem)")

    # --- Summary ---
    total_removed = train_stats["stems_removed"] + val_stats["stems_removed"]
    total_in = train_stats["lines_in"] + val_stats["lines_in"]
    print(f"\n{'='*60}")
    print(f"TOTAL: removed {total_removed:,} stem annotations ({total_removed/total_in*100:.1f}%)")
    print(f"  Train: {train_stats['lines_out']:,} annotations, {train_stats['files']} images")
    print(f"  Val:   {val_stats['lines_out']:,} annotations, {val_stats['files']} images")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
