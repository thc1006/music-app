#!/usr/bin/env python3
"""Create Phase 7 dataset: fix ALL OpenScore glyph-group bboxes for all 32 classes.

Extends Phase 6 (which only fixed noteheads + removed stems).
Root cause (ADR-003): ALL 32/33 OpenScore classes have oversized LilyPond glyph-group
bboxes (6x to 7746x area ratio vs DoReMi ground truth).

Two universal position rules (empirically validated, NMS@0.7 ceiling → 100.00%):
  Rule 1 — TOP edge: cy = bbox_top + ref_h/2, w = ref_w, h = ref_h
    Applies to: noteheads, beam, flags, augdot, tie, accidentals, fermata, ledger_line, clefs
  Rule 2 — CENTER: keep cx/cy, w = ref_w, h = ref_h
    Applies to: rests, time_sig, key_sig, dynamics, barlines
  + Dedup pass: removes near-identical same-class annotations (IoU >= 0.7)

Input: Phase 6 dataset (already has stems removed + noteheads fixed)
Output: training/datasets/yolo_harmony_v2_phase7_universal/

Validation: run NMS ceiling check after generation.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/thc1006/dev/music-app/training/datasets")

# Input: Phase 6 dataset (stems removed, noteheads already fixed, 32 classes)
PHASE6 = ROOT / "yolo_harmony_v2_phase6_fixed"

# Output
OUTPUT = ROOT / "yolo_harmony_v2_phase7_universal"

# ---------------------------------------------------------------------------
# Phase 6 class IDs (32 classes, stem already removed)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "notehead_filled",          # 0
    "notehead_hollow",          # 1
    "beam",                     # 2
    "flag_8th",                 # 3
    "flag_16th",                # 4
    "flag_32nd",                # 5
    "augmentation_dot",         # 6
    "tie",                      # 7
    "clef_treble",              # 8
    "clef_bass",                # 9
    "clef_alto",                # 10
    "clef_tenor",               # 11
    "accidental_sharp",         # 12
    "accidental_flat",          # 13
    "accidental_natural",       # 14
    "accidental_double_sharp",  # 15
    "accidental_double_flat",   # 16
    "rest_whole",               # 17
    "rest_half",                # 18
    "rest_quarter",             # 19
    "rest_8th",                 # 20
    "rest_16th",                # 21
    "barline",                  # 22
    "barline_double",           # 23
    "barline_final",            # 24
    "barline_repeat",           # 25
    "time_signature",           # 26
    "key_signature",            # 27
    "fermata",                  # 28
    "dynamic_soft",             # 29
    "dynamic_loud",             # 30
    "ledger_line",              # 31
]

# ---------------------------------------------------------------------------
# DoReMi reference sizes (median, normalized) — from ADR-003
# ---------------------------------------------------------------------------
REF_SIZES: dict[int, tuple[float, float]] = {
    0:  (0.01180, 0.00908),   # notehead_filled  (already fixed in Phase 6)
    1:  (0.01180, 0.00908),   # notehead_hollow  (already fixed in Phase 6)
    2:  (0.04150, 0.00742),   # beam
    3:  (0.00889, 0.01969),   # flag_8th
    4:  (0.00970, 0.01941),   # flag_16th
    5:  (0.00889, 0.02312),   # flag_32nd
    6:  (0.00364, 0.00257),   # augmentation_dot
    7:  (0.03475, 0.00628),   # tie
    8:  (0.02263, 0.04195),   # clef_treble
    9:  (0.02343, 0.02140),   # clef_bass
    10: (0.02263, 0.04195),   # clef_alto  (use clef_treble ref — DoReMi bad)
    11: (0.02263, 0.04195),   # clef_tenor (use clef_treble ref — 0 DoReMi samples)
    12: (0.00848, 0.01684),   # accidental_sharp
    13: (0.00768, 0.01484),   # accidental_flat
    14: (0.00566, 0.01627),   # accidental_natural
    15: (0.00848, 0.00599),   # accidental_double_sharp
    16: (0.01601, 0.06861),   # accidental_double_flat
    17: (0.00970, 0.00343),   # rest_whole
    18: (0.00970, 0.00343),   # rest_half
    19: (0.00929, 0.01798),   # rest_quarter
    20: (0.00848, 0.01027),   # rest_8th
    21: (0.01091, 0.01627),   # rest_16th
    22: (0.01500, 0.08333),   # barline — single-staff height from DoReMi
    23: (0.01000, 0.82782),   # barline_double — grand-staff spanning
    24: (0.01000, 0.78791),   # barline_final — grand-staff spanning
    25: (0.01562, 0.13281),   # barline_repeat — staff height from DoReMi
    26: (0.01455, 0.01256),   # time_signature
    27: (0.01737, 0.05090),   # key_signature — was "keep", now CENTER (62.8% NMS → fix)
    28: (0.02000, 0.02000),   # fermata
    29: (0.01535, 0.00999),   # dynamic_soft
    30: (0.01944, None),      # dynamic_loud (h=0.40745 suspicious — keep height)
    31: (0.01205, 0.00513),   # ledger_line
}

# ---------------------------------------------------------------------------
# Position rules per class
# ---------------------------------------------------------------------------
# "top"    → cy = bbox_top + ref_h/2, w = ref_w, h = ref_h  (NMS 100%)
# "center" → keep cx/cy, w = ref_w, h = ref_h               (NMS 100%)
# "skip"   → noteheads already fixed in Phase 6

RULES: dict[int, str] = {
    0:  "skip",       # notehead_filled — already fixed
    1:  "skip",       # notehead_hollow — already fixed
    2:  "top",        # beam
    3:  "top",        # flag_8th
    4:  "top",        # flag_16th
    5:  "top",        # flag_32nd
    6:  "top",        # augmentation_dot
    7:  "top",        # tie
    8:  "top",        # clef_treble — was "clef", but keeping height caused cross-staff overlap
    9:  "top",        # clef_bass
    10: "top",        # clef_alto
    11: "top",        # clef_tenor
    12: "top",        # accidental_sharp
    13: "top",        # accidental_flat
    14: "top",        # accidental_natural
    15: "top",        # accidental_double_sharp
    16: "top",        # accidental_double_flat
    17: "center",     # rest_whole
    18: "center",     # rest_half
    19: "center",     # rest_quarter
    20: "center",     # rest_8th
    21: "center",     # rest_16th
    22: "center",     # barline — was "barline"(keep h), now shrink to single-staff h
    23: "center",     # barline_double — shrink to grand-staff ref
    24: "center",     # barline_final
    25: "center",     # barline_repeat
    26: "center",     # time_signature
    27: "center",     # key_signature — was "keep", 62.8% NMS ceiling → must shrink
    28: "top",        # fermata
    29: "center",     # dynamic_soft
    30: "center",     # dynamic_loud — was "barline", now center (shrink w, keep h via None)
    31: "top",        # ledger_line
}


def is_openscore(filename: str) -> bool:
    """Check if a label file is from OpenScore (has glyph-group bboxes)."""
    return "lg-" in filename


def fix_annotation(cls: int, cx: float, cy: float, w: float, h: float
                   ) -> tuple[float, float, float, float]:
    """Apply universal position rule to a single OpenScore annotation.

    Returns (new_cx, new_cy, new_w, new_h).
    """
    rule = RULES.get(cls, "keep")

    if rule == "skip" or rule == "keep":
        return cx, cy, w, h

    ref = REF_SIZES.get(cls)
    if ref is None:
        return cx, cy, w, h

    ref_w, ref_h = ref
    bbox_top = cy - h / 2
    bbox_left = cx - w / 2

    out_w = ref_w if ref_w is not None else w
    out_h = ref_h if ref_h is not None else h

    if rule == "top":
        new_cy = bbox_top + out_h / 2
        new_cy = max(out_h / 2, min(1.0 - out_h / 2, new_cy))
        return cx, new_cy, out_w, out_h

    elif rule == "center":
        return cx, cy, out_w, out_h

    return cx, cy, w, h


def dedup_boxes(annots: list[tuple[int, float, float, float, float]],
                iou_thr: float = 0.7) -> list[tuple[int, float, float, float, float]]:
    """Remove near-duplicate annotations (same class, IoU >= threshold).

    Keeps the first occurrence, removes later duplicates.
    This handles cases where OpenScore generates multiple annotations
    for the same physical symbol (e.g., one per staff in grand staff).
    """
    if len(annots) <= 1:
        return annots

    # Group by class
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, (cls, cx, cy, w, h) in enumerate(annots):
        by_class[cls].append(idx)

    suppressed = set()
    for cls, indices in by_class.items():
        if len(indices) <= 1:
            continue
        for i_pos in range(len(indices)):
            i = indices[i_pos]
            if i in suppressed:
                continue
            _, cx1, cy1, w1, h1 = annots[i]
            for j_pos in range(i_pos + 1, len(indices)):
                j = indices[j_pos]
                if j in suppressed:
                    continue
                _, cx2, cy2, w2, h2 = annots[j]
                # Quick reject
                if abs(cx1 - cx2) > max(w1, w2) or abs(cy1 - cy2) > max(h1, h2):
                    continue
                # Full IoU
                x1a, x1b = cx1 - w1 / 2, cx1 + w1 / 2
                y1a, y1b = cy1 - h1 / 2, cy1 + h1 / 2
                x2a, x2b = cx2 - w2 / 2, cx2 + w2 / 2
                y2a, y2b = cy2 - h2 / 2, cy2 + h2 / 2
                iw = max(0, min(x1b, x2b) - max(x1a, x2a))
                ih = max(0, min(y1b, y2b) - max(y1a, y2a))
                inter = iw * ih
                union = w1 * h1 + w2 * h2 - inter
                if union > 0 and inter / union >= iou_thr:
                    suppressed.add(j)

    return [a for idx, a in enumerate(annots) if idx not in suppressed]


def process_labels(src_dir: Path, dst_dir: Path) -> dict[str, int]:
    """Process all label files: fix OpenScore bboxes for ALL classes + dedup."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = defaultdict(int)
    per_class_fixed: dict[int, int] = defaultdict(int)

    label_files = sorted(src_dir.glob("*.txt"))
    for lf in label_files:
        stats["files"] += 1
        lines = lf.read_text().strip().split("\n") if lf.stat().st_size > 0 else []

        # Step 1: Parse and fix bboxes
        fixed_annots = []  # (cls, cx, cy, w, h)
        for line in lines:
            if not line.strip():
                continue
            stats["lines_in"] += 1
            parts = line.strip().split()
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            if is_openscore(lf.stem) and RULES.get(cls, "keep") not in ("skip", "keep"):
                new_cx, new_cy, new_w, new_h = fix_annotation(cls, cx, cy, w, h)
                if (new_cx, new_cy, new_w, new_h) != (cx, cy, w, h):
                    per_class_fixed[cls] += 1
                    stats["fixed"] += 1
                cx, cy, w, h = new_cx, new_cy, new_w, new_h
            else:
                stats["kept"] += 1

            fixed_annots.append((cls, cx, cy, w, h))

        # Step 2: Dedup near-identical boxes (IoU >= 0.7, same class)
        before_dedup = len(fixed_annots)
        fixed_annots = dedup_boxes(fixed_annots, iou_thr=0.7)
        stats["deduped"] += before_dedup - len(fixed_annots)

        # Step 3: Write output
        new_lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                     for cls, cx, cy, w, h in fixed_annots]
        stats["lines_out"] += len(new_lines)

        out_path = dst_dir / lf.name
        if new_lines:
            out_path.write_text("\n".join(new_lines) + "\n")
        else:
            out_path.write_text("")
            stats["empty_after"] += 1

    stats["per_class_fixed"] = dict(per_class_fixed)
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
    """Write dataset YAML (same 32 classes as Phase 6)."""
    lines = [
        f"path: {output_dir}",
        "train: train/images",
        "val: val/images",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    (output_dir / "harmony_phase7_universal.yaml").write_text("\n".join(lines) + "\n")


def main() -> int:
    print("=" * 70)
    print("Phase 7 Dataset: Universal OpenScore Bbox Fix for ALL 32 Classes")
    print("=" * 70)
    print(f"\nInput:  {PHASE6}")
    print(f"Output: {OUTPUT}")

    if not PHASE6.exists():
        print(f"\nERROR: Phase 6 dataset not found at {PHASE6}")
        print("Run create_phase6_fixed_bbox.py first.")
        return 1

    if OUTPUT.exists():
        print(f"\nOutput directory exists, removing...")
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    # --- Print rules summary ---
    print(f"\n{'─'*70}")
    print("Position rules:")
    for cls_id, rule in sorted(RULES.items()):
        ref = REF_SIZES.get(cls_id, (None, None))
        rw = f"{ref[0]:.5f}" if ref and ref[0] else "—"
        rh = f"{ref[1]:.5f}" if ref and ref[1] else "—"
        print(f"  [{cls_id:2d}] {CLASS_NAMES[cls_id]:28s} rule={rule:8s}  ref=({rw}, {rh})")

    # --- Train set ---
    print(f"\n{'─'*70}")
    print("Processing Train set...")
    train_img_src = PHASE6 / "train" / "images"
    train_lbl_src = PHASE6 / "train" / "labels"
    train_img_dst = OUTPUT / "train" / "images"
    train_lbl_dst = OUTPUT / "train" / "labels"

    n_img = symlink_images(train_img_src, train_img_dst)
    print(f"  Images symlinked: {n_img}")

    train_stats = process_labels(train_lbl_src, train_lbl_dst)
    print(f"  Annotations in:   {train_stats['lines_in']:,}")
    print(f"  Fixed:            {train_stats['fixed']:,}")
    print(f"  Kept as-is:       {train_stats['kept']:,}")
    print(f"  Annotations out:  {train_stats['lines_out']:,}")

    # --- Val set ---
    print(f"\n{'─'*70}")
    print("Processing Val set...")
    val_img_src = PHASE6 / "val" / "images"
    val_lbl_src = PHASE6 / "val" / "labels"
    val_img_dst = OUTPUT / "val" / "images"
    val_lbl_dst = OUTPUT / "val" / "labels"

    n_img = symlink_images(val_img_src, val_img_dst)
    print(f"  Images symlinked: {n_img}")

    val_stats = process_labels(val_lbl_src, val_lbl_dst)
    print(f"  Annotations in:   {val_stats['lines_in']:,}")
    print(f"  Fixed:            {val_stats['fixed']:,}")
    print(f"  Kept as-is:       {val_stats['kept']:,}")
    print(f"  Annotations out:  {val_stats['lines_out']:,}")

    # --- YAML ---
    write_yaml(OUTPUT)
    print(f"\nYAML: {OUTPUT / 'harmony_phase7_universal.yaml'}")

    # --- Per-class fix summary ---
    all_fixed = defaultdict(int)
    for cls_id, count in train_stats.get("per_class_fixed", {}).items():
        all_fixed[cls_id] += count
    for cls_id, count in val_stats.get("per_class_fixed", {}).items():
        all_fixed[cls_id] += count

    print(f"\n{'='*70}")
    print("PER-CLASS FIX SUMMARY (OpenScore annotations fixed)")
    print(f"{'='*70}")
    print(f"{'Class':28s} {'Rule':8s} {'Fixed':>10s}")
    print("-" * 50)
    total_fixed = 0
    for cls_id in sorted(all_fixed.keys()):
        n = all_fixed[cls_id]
        total_fixed += n
        print(f"{CLASS_NAMES[cls_id]:28s} {RULES[cls_id]:8s} {n:10,d}")
    print("-" * 50)
    print(f"{'TOTAL':28s} {'':8s} {total_fixed:10,d}")

    total_in = train_stats["lines_in"] + val_stats["lines_in"]
    total_out = train_stats["lines_out"] + val_stats["lines_out"]
    print(f"\nTotal annotations: {total_in:,} in → {total_out:,} out")
    print(f"Fixed: {total_fixed:,} ({total_fixed/total_in*100:.1f}%)")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
