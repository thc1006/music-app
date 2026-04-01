#!/usr/bin/env python3
"""Create Phase 6 dataset: fix OpenScore notehead bboxes + remove stems.

Root cause (ADR-001): OpenScore (lg-*) notehead bboxes are LilyPond glyph-group
bounding boxes (~380×335px) that include stem+beam, not individual noteheads (~23×25px).
This causes NMS to suppress 59% of notehead detections (IoU up to 0.986).

Fix strategy:
  1. For OpenScore images: use Phase 8 stem annotations to determine stem direction,
     then place notehead bbox at the correct edge of the original bbox.
  2. For DoReMi/MUSCIMA images: keep original bbox (already correct).
  3. Remove stem annotations and remap class IDs (33 → 32).

Validation: NMS@0.7 recall ceiling: 0.526 → 0.990 after fix.

Output: training/datasets/yolo_harmony_v2_phase6_fixed/
"""

from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path("/home/thc1006/dev/music-app/training/datasets")

# Sources (Phase 8 original with stems for stem-direction detection)
ORIGINAL = ROOT / "yolo_harmony_v2_phase8_final"
CLEANED_V2 = ROOT / "yolo_harmony_v2_phase8_cleaned_v2"

# Output
OUTPUT = ROOT / "yolo_harmony_v2_phase6_fixed"

STEM_CLASS_ID = 2   # class to remove
NOTEHEAD_FILLED = 0  # class to fix
NOTEHEAD_HOLLOW = 1  # class to fix

# Estimated true notehead size (normalized), derived from chord-spacing analysis
# See: training/reports/notehead_recall_analysis.md
NOTEHEAD_W = 0.01180  # ~23px at 1960 width
NOTEHEAD_H = 0.00908  # ~25px at 2772 height

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

REMAP: dict[int, int] = {}
new_id = 0
for old_id in range(len(ORIGINAL_NAMES)):
    if old_id == STEM_CLASS_ID:
        REMAP[old_id] = -1
    else:
        REMAP[old_id] = new_id
        new_id += 1

NEW_NAMES = [n for i, n in enumerate(ORIGINAL_NAMES) if i != STEM_CLASS_ID]
assert len(NEW_NAMES) == 32


def is_openscore(filename: str) -> bool:
    """Check if a label file is from OpenScore (has giant bboxes)."""
    return "lg-" in filename


def fix_openscore_noteheads(
    noteheads: list[tuple[int, float, float, float, float]],
    stems: list[tuple[float, float, float, float]],
) -> list[tuple[int, float, float, float, float]]:
    """Fix OpenScore notehead bboxes: place at top edge of original bbox.

    OpenScore LilyPond glyph-group bboxes always extend from the notehead
    position downward to the stem bottom. Therefore the TRUE notehead y
    position is at the TOP edge of the original bbox, regardless of stem
    direction. No stem matching needed.

    Verified:
      - Top-edge spacings cluster at 3-5px (semitone) and 8-12px (whole tone)
      - NMS@0.7 ceiling with "always TOP": 100.00% (70,592/70,594)
      - Previous stem-matching approach left 43.6% at wrong center position

    Args:
        noteheads: list of (line_idx, cx, cy, w, h) for notehead annotations
        stems: unused (kept for API compatibility)

    Returns:
        Fixed noteheads with corrected bbox at top edge.
    """
    if not noteheads:
        return noteheads

    result = []
    for idx, cx, cy, w, h in noteheads:
        # Notehead is at the TOP edge of the glyph-group bbox
        new_cy = (cy - h / 2) + NOTEHEAD_H / 2
        # Clamp to [0, 1]
        new_cy = max(NOTEHEAD_H / 2, min(1.0 - NOTEHEAD_H / 2, new_cy))
        result.append((idx, cx, new_cy, NOTEHEAD_W, NOTEHEAD_H))

    return result


def process_labels(
    src_dir: Path,
    dst_dir: Path,
    fix_openscore: bool = True,
) -> dict[str, int]:
    """Process labels: fix OpenScore noteheads, remove stems, remap IDs."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "files": 0,
        "lines_in": 0,
        "lines_out": 0,
        "stems_removed": 0,
        "noteheads_fixed": 0,
        "noteheads_kept": 0,
        "empty_after": 0,
    }

    label_files = sorted(src_dir.glob("*.txt"))
    for lf in label_files:
        stats["files"] += 1
        lines = lf.read_text().strip().split("\n") if lf.stat().st_size > 0 else []

        # Parse all annotations
        all_annots = []  # (line_idx, cls, cx, cy, w, h)
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            stats["lines_in"] += 1
            parts = line.strip().split()
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            all_annots.append((i, cls, cx, cy, w, h))

        # Separate noteheads and stems
        noteheads = [(i, cx, cy, w, h) for i, cls, cx, cy, w, h in all_annots
                     if cls in (NOTEHEAD_FILLED, NOTEHEAD_HOLLOW)]
        stems = [(cx, cy, w, h) for i, cls, cx, cy, w, h in all_annots
                 if cls == STEM_CLASS_ID]

        # Fix OpenScore notehead bboxes
        fixed_nh = {}  # line_idx → (cx, cy, w, h)
        if fix_openscore and is_openscore(lf.stem) and noteheads:
            fixed_list = fix_openscore_noteheads(noteheads, stems)
            for idx, cx, cy, w, h in fixed_list:
                fixed_nh[idx] = (cx, cy, w, h)
                stats["noteheads_fixed"] += 1
        else:
            stats["noteheads_kept"] += len(noteheads)

        # Build output lines
        new_lines = []
        for i, cls, cx, cy, w, h in all_annots:
            if cls == STEM_CLASS_ID:
                stats["stems_removed"] += 1
                continue

            new_cls = REMAP.get(cls, cls)
            if new_cls == -1:
                continue

            if i in fixed_nh:
                cx, cy, w, h = fixed_nh[i]

            new_lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["lines_out"] += 1

        out_path = dst_dir / lf.name
        if new_lines:
            out_path.write_text("\n".join(new_lines) + "\n")
        else:
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
            target.symlink_to(img.resolve())
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
    (output_dir / "harmony_phase6_fixed.yaml").write_text("\n".join(lines) + "\n")


def main() -> int:
    print("=" * 60)
    print("Phase 6 Dataset: Fix OpenScore Notehead Bboxes + Remove Stems")
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

    train_stats = process_labels(train_lbl_src, train_lbl_dst, fix_openscore=True)
    print(f"  Label files:       {train_stats['files']}")
    print(f"  Annotations in:    {train_stats['lines_in']:,}")
    print(f"  Stems removed:     {train_stats['stems_removed']:,}")
    print(f"  Noteheads fixed:   {train_stats['noteheads_fixed']:,}")
    print(f"  Noteheads kept:    {train_stats['noteheads_kept']:,}")
    print(f"  Annotations out:   {train_stats['lines_out']:,}")
    print(f"  Empty after:       {train_stats['empty_after']}")

    # --- Val: cleaned_v2 2,867 images ---
    print("\n--- Val Set (cleaned_v2, high quality) ---")
    val_img_src = CLEANED_V2 / "val" / "images"
    val_lbl_src = CLEANED_V2 / "val" / "labels"
    val_img_dst = OUTPUT / "val" / "images"
    val_lbl_dst = OUTPUT / "val" / "labels"

    n_img = symlink_images(val_img_src, val_img_dst)
    print(f"  Images symlinked: {n_img}")

    val_stats = process_labels(val_lbl_src, val_lbl_dst, fix_openscore=True)
    print(f"  Label files:       {val_stats['files']}")
    print(f"  Annotations in:    {val_stats['lines_in']:,}")
    print(f"  Stems removed:     {val_stats['stems_removed']:,}")
    print(f"  Noteheads fixed:   {val_stats['noteheads_fixed']:,}")
    print(f"  Noteheads kept:    {val_stats['noteheads_kept']:,}")
    print(f"  Annotations out:   {val_stats['lines_out']:,}")
    print(f"  Empty after:       {val_stats['empty_after']}")

    # --- Write YAML ---
    write_yaml(OUTPUT)
    print(f"\nYAML: {OUTPUT / 'harmony_phase6_fixed.yaml'}")
    print(f"Classes: {len(NEW_NAMES)} (was 33, removed stem)")

    # --- Summary ---
    total_fixed = train_stats["noteheads_fixed"] + val_stats["noteheads_fixed"]
    total_removed = train_stats["stems_removed"] + val_stats["stems_removed"]
    total_in = train_stats["lines_in"] + val_stats["lines_in"]
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Stems removed:    {total_removed:,} ({total_removed/total_in*100:.1f}%)")
    print(f"  Noteheads fixed:  {total_fixed:,}")
    print(f"  Train: {train_stats['lines_out']:,} annotations, {train_stats['files']} images")
    print(f"  Val:   {val_stats['lines_out']:,} annotations, {val_stats['files']} images")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
