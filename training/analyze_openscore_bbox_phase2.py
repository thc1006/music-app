#!/usr/bin/env python3
"""
Phase 2 analysis: deeper investigation of OpenScore glyph-group bboxes.

Key questions:
  A. The dy_top_median ≈ +0.5 for ALL oversized classes means the glyph-group
     bbox CENTER equals the notehead CENTER (not top edge). Why?
     → Because the glyph-group bbox IS the notehead bbox (same cx, cy, w, h).
     → We need to verify: are the oversized bbox dimensions concentrated around
       one large standard size, suggesting they all come from ONE glyph-group bbox
       that is shared across ALL annotations in that chord column?

  B. For classes with glyph-group sharing (flag, clef, accidental, tie):
     How many distinct bbox shapes are there? Is there a dominant glyph-group size?

  C. Barline: width is CORRECT (0.01), only HEIGHT is inflated (0.74 vs 0.02).
     Verify this is the staff-height envelope.

  D. Key_signature, barline_double, barline_final, barline_repeat, ledger_line:
     No DoReMi reference. Use MUSCIMA or Rebelo as reference instead.

  E. For each oversized class: what is the ACTUAL symbol size relative to the
     glyph-group bbox?
     → If glyph-group bbox = notehead glyph-group, then symbol center is at
       glyph-group center, but TRUE size must come from a clean source.

  F. For accidentals (high glyph-group sharing): they share bbox with each other
     not with noteheads. So their glyph-group is the ACCIDENTAL group, not the
     note group.
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

LABEL_DIR = Path(
    "/home/thc1006/dev/music-app/training/datasets/"
    "yolo_harmony_v2_phase8_final/val/labels"
)
CLASS_NAMES = [
    "notehead_filled",        # 0
    "notehead_hollow",        # 1
    "stem",                   # 2
    "beam",                   # 3
    "flag_8th",               # 4
    "flag_16th",              # 5
    "flag_32nd",              # 6
    "augmentation_dot",       # 7
    "tie",                    # 8
    "clef_treble",            # 9
    "clef_bass",              # 10
    "clef_alto",              # 11
    "clef_tenor",             # 12
    "accidental_sharp",       # 13
    "accidental_flat",        # 14
    "accidental_natural",     # 15
    "accidental_double_sharp",# 16
    "accidental_double_flat", # 17
    "rest_whole",             # 18
    "rest_half",              # 19
    "rest_quarter",           # 20
    "rest_8th",               # 21
    "rest_16th",              # 22
    "barline",                # 23
    "barline_double",         # 24
    "barline_final",          # 25
    "barline_repeat",         # 26
    "time_signature",         # 27
    "key_signature",          # 28
    "fermata",                # 29
    "dynamic_soft",           # 30
    "dynamic_loud",           # 31
    "ledger_line",            # 32
]
NUM_CLASSES = len(CLASS_NAMES)
SAME_TOL = 0.005

def classify_source(filename):
    n = filename.lower()
    if "lg-" in n:              return "openscore_lg"
    if "lieder_openscore" in n: return "openscore_lieder"
    if "muscima" in n or "cvc" in n: return "muscima"
    if "_ds2_" in n or "ds2_train" in n or "ds2_val" in n: return "ds2"
    if "doremi" in n or "doremiv" in n: return "doremi"
    if "rebelo" in n:           return "rebelo"
    if "fornes" in n:           return "fornes"
    return "other"

def is_openscore(src): return src in ("openscore_lg", "openscore_lieder")

def load_all(label_dir):
    per_source = defaultdict(list)
    per_file   = defaultdict(list)
    sources    = {}
    for fpath in sorted(label_dir.glob("*.txt")):
        src = classify_source(fpath.name)
        sources[fpath.name] = src
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                row = (int(parts[0]), float(parts[1]), float(parts[2]),
                       float(parts[3]), float(parts[4]))
                per_source[src].append(row)
                per_file[fpath.name].append(row)
    return per_source, per_file, sources


def main():
    print("=" * 80)
    print("Phase 2: Deep Glyph-Group Structure Analysis")
    print("=" * 80)

    per_source, per_file, sources = load_all(LABEL_DIR)
    os_files = [fn for fn, s in sources.items() if is_openscore(s)]

    # -----------------------------------------------------------------------
    # A. Glyph-group bbox SIZE distribution for oversized classes
    # -----------------------------------------------------------------------
    print("\n[A] GLYPH-GROUP BBOX SIZE DISTRIBUTION")
    print("    Are OpenScore oversized bboxes all the same large size,")
    print("    or do they vary (suggesting variable glyph-group boundaries)?")
    print()

    oversized = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,27,30,31]
    os_annots = per_source["openscore_lg"] + per_source["openscore_lieder"]

    for cid in oversized:
        ws = [a[3] for a in os_annots if a[0] == cid]
        hs = [a[4] for a in os_annots if a[0] == cid]
        if not ws: continue
        ws, hs = np.array(ws), np.array(hs)
        # How many unique bbox sizes? (rounded to 3 decimal places)
        sizes = set(zip(np.round(ws, 3), np.round(hs, 3)))
        cv_w = np.std(ws) / np.mean(ws)
        cv_h = np.std(hs) / np.mean(hs)
        print(f"  [{cid:2d}] {CLASS_NAMES[cid]:<26}  n={len(ws):5d}  "
              f"CV_w={cv_w:.3f}  CV_h={cv_h:.3f}  "
              f"unique_sizes={len(sizes):4d}  "
              f"w∈[{ws.min():.3f},{ws.max():.3f}]  h∈[{hs.min():.3f},{hs.max():.3f}]")

    # -----------------------------------------------------------------------
    # B. For key "glyph-group sharing" classes: which classes share with which?
    # -----------------------------------------------------------------------
    print()
    print("[B] WHICH CLASSES SHARE BBOXES WITH WHICH? (OpenScore files)")
    print("    For classes with high sharing rate, show co-occurrence matrix.")
    print()

    FOCUS = {
        8:  "tie",
        9:  "clef_treble",
        11: "clef_alto",
        12: "clef_tenor",
        13: "accidental_sharp",
        14: "accidental_flat",
        25: "barline_final",
        26: "barline_repeat",
        28: "key_signature",
    }

    def round_bbox(cx, cy, w, h):
        r = SAME_TOL
        return (round(cx/r)*r, round(cy/r)*r, round(w/r)*r, round(h/r)*r)

    for focus_cid, focus_name in FOCUS.items():
        cooccur = defaultdict(int)
        total   = 0
        for fn in os_files:
            annots = per_file[fn]
            bbox_to_classes = defaultdict(set)
            for (cid, cx, cy, w, h) in annots:
                bbox_to_classes[round_bbox(cx,cy,w,h)].add(cid)

            for (cid, cx, cy, w, h) in annots:
                if cid != focus_cid: continue
                total += 1
                key = round_bbox(cx, cy, w, h)
                for other_cid in bbox_to_classes[key]:
                    if other_cid != focus_cid:
                        cooccur[other_cid] += 1

        if total == 0: continue
        print(f"  [{focus_cid}] {focus_name} (total OS annots: {total})")
        for other_cid, cnt in sorted(cooccur.items(), key=lambda x:-x[1])[:8]:
            pct = 100.0 * cnt / total
            print(f"      shares with [{other_cid:2d}] {CLASS_NAMES[other_cid]:<26} "
                  f": {cnt:5d} ({pct:5.1f}%)")
        print()

    # -----------------------------------------------------------------------
    # C. Barline: width is OK but height is inflated → staff-height envelope
    # -----------------------------------------------------------------------
    print()
    print("[C] BARLINE ANALYSIS: width vs height for OpenScore vs DoReMi")
    print()

    for src in ("openscore_lg", "openscore_lieder", "doremi", "muscima", "rebelo"):
        annots = per_source[src]
        bl = [(a[3], a[4]) for a in annots if a[0] == 23]
        if not bl: continue
        ws = np.array([b[0] for b in bl])
        hs = np.array([b[1] for b in bl])
        print(f"  {src:<22}: n={len(bl):4d}  "
              f"w={np.median(ws):.5f} [{np.percentile(ws,5):.4f}–{np.percentile(ws,95):.4f}]  "
              f"h={np.median(hs):.5f} [{np.percentile(hs,5):.4f}–{np.percentile(hs,95):.4f}]")

    # Are barlines in OpenScore spanning the full staff height?
    print("\n  OpenScore barline height histogram (n bins=10):")
    bl_h = np.array([a[4] for a in per_source["openscore_lg"] + per_source["openscore_lieder"] if a[0] == 23])
    if len(bl_h):
        counts, edges = np.histogram(bl_h, bins=10)
        for i, c in enumerate(counts):
            print(f"    [{edges[i]:.3f}–{edges[i+1]:.3f}]: {'#'*min(c,60)} ({c})")

    # -----------------------------------------------------------------------
    # D. Classes with no DoReMi reference: use MUSCIMA + Rebelo + Fornes
    # -----------------------------------------------------------------------
    print()
    print("[D] CLASSES WITH NO DOREMI REFERENCE — MUSCIMA / REBELO / FORNES STATS")
    no_dr_classes = [12, 24, 25, 26, 28, 29, 32]  # ledger_line has no DR
    alt_srcs = ["muscima", "rebelo", "fornes", "other"]

    for cid in no_dr_classes:
        rows = []
        for src in alt_srcs:
            data = [(a[3], a[4]) for a in per_source[src] if a[0] == cid]
            if data:
                ws = np.array([d[0] for d in data])
                hs = np.array([d[1] for d in data])
                rows.append(f"{src}(n={len(data)}: "
                            f"w={np.median(ws):.5f}, h={np.median(hs):.5f})")
        os_data = [(a[3], a[4]) for a in os_annots if a[0] == cid]
        os_str = ""
        if os_data:
            ws = np.array([d[0] for d in os_data])
            hs = np.array([d[1] for d in os_data])
            os_str = f"OS(n={len(os_data)}: w={np.median(ws):.5f}, h={np.median(hs):.5f})"
        print(f"  [{cid:2d}] {CLASS_NAMES[cid]:<26}: {os_str}")
        for r in rows:
            print(f"         {r}")

    # -----------------------------------------------------------------------
    # E. CRUCIAL: Are glyph-group bboxes IDENTICAL to notehead bboxes?
    #    For each OS file, compare class X bbox vs class 0 bbox at same position.
    # -----------------------------------------------------------------------
    print()
    print("[E] ARE OVERSIZED BBOXES IDENTICAL TO THE NOTEHEAD GLYPH-GROUP BOX?")
    print("    For each OS file: fraction of class X annotations where bbox == nearest NH bbox")
    print()

    def bbox_iou(a, b):
        """Simple IoU between two (cx, cy, w, h) bboxes."""
        ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
        bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
        ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw*ih
        ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/ua if ua>0 else 0

    iou_high_thresh = 0.90  # "identical" bbox

    for cid in [0,1,2,3,4,7,8,9,13,14,15,18,20,27,30]:
        match_any_nh  = 0
        match_exact   = 0  # IoU > 0.90 with nearest NH
        total_annots  = 0
        for fn in os_files:
            annots = per_file[fn]
            noteheads = [(a[1],a[2],a[3],a[4]) for a in annots if a[0] in (0,1)]
            targets   = [(a[1],a[2],a[3],a[4]) for a in annots if a[0] == cid]
            if not noteheads or not targets: continue
            for t in targets:
                total_annots += 1
                # Find nearest notehead by center distance
                dists = [((t[0]-n[0])**2+(t[1]-n[1])**2)**0.5 for n in noteheads]
                best_nh = noteheads[int(np.argmin(dists))]
                best_dist = min(dists)
                iou = bbox_iou(t, best_nh)
                if iou > iou_high_thresh:
                    match_exact += 1
                if best_dist < 0.05:  # within 5% normalized dist
                    match_any_nh += 1

        if total_annots == 0: continue
        p_exact = 100.0 * match_exact / total_annots
        p_near  = 100.0 * match_any_nh / total_annots
        print(f"  [{cid:2d}] {CLASS_NAMES[cid]:<26}: total={total_annots:5d}  "
              f"IoU>0.90 with NH={p_exact:5.1f}%  "
              f"center within 5%={p_near:5.1f}%")

    # -----------------------------------------------------------------------
    # F. For accidentals: what IS the glyph-group they share?
    #    They have high "shared_with_any" but 0% with noteheads.
    #    → They share with OTHER accidentals (key signature column)
    # -----------------------------------------------------------------------
    print()
    print("[F] ACCIDENTAL GLYPH-GROUP: which classes appear in the SAME bbox?")
    print()

    accidental_classes = [13, 14, 15, 16, 17]
    for focus_cid in accidental_classes:
        same_cls_bbox  = 0  # different annotation, same class, same bbox
        diff_cls_bbox  = 0  # different class, same bbox
        total = 0
        detail = defaultdict(int)
        for fn in os_files:
            annots = per_file[fn]
            bbox_map = defaultdict(list)
            for (cid, cx, cy, w, h) in annots:
                bbox_map[round_bbox(cx,cy,w,h)].append(cid)
            for (cid, cx, cy, w, h) in annots:
                if cid != focus_cid: continue
                total += 1
                key = round_bbox(cx,cy,w,h)
                others = [c for c in bbox_map[key] if c != focus_cid]
                for o in others:
                    detail[o] += 1
                if any(c == focus_cid for c in bbox_map[key] if c == focus_cid
                       and bbox_map[key].count(focus_cid) > 1):
                    same_cls_bbox += 1
                if others:
                    diff_cls_bbox += 1

        print(f"  [{focus_cid}] {CLASS_NAMES[focus_cid]}: total={total}")
        for o, cnt in sorted(detail.items(), key=lambda x:-x[1])[:6]:
            print(f"      co-bbox [{o:2d}] {CLASS_NAMES[o]}: {cnt} ({100*cnt/total:.1f}%)")
        print()

    # -----------------------------------------------------------------------
    # G. Summary: bbox structure hypothesis
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SYNTHESIS: LilyPond Glyph-Group Bbox Structure")
    print("=" * 80)
    print("""
FINDING 1: ALL oversized classes have bbox CENTER ≈ notehead CENTER
  (dy_top_median ≈ 0.5 in all cases).
  This is because the glyph-group bbox IS the same large column bbox for all
  symbols in that LilyPond column — the center of the column bbox is ~equal
  to the center of the note column, which is roughly at the notehead.

FINDING 2: Most note-related classes (stem, beam, flags, aug_dot, rests) have
  ZERO glyph-group sharing with noteheads.
  → They have their OWN glyph-group bbox, not shared with the notehead bbox.
  → Each class has its own glyph-group bbox, all of which happen to be roughly
    the same large column size (w~0.30, h~0.26) because they all cover the
    same musical column.

FINDING 3: Clefs (9,11,12) share bboxes with OTHER classes 50-100%.
  Accidentals (13,14) share bboxes with each other ~60-70%.
  → Clef glyph-group = full measure start column.
  → Accidental glyph-group = accidental column (may contain multiple accidentals).

FINDING 4: The barline has w=0.01 (CORRECT) but h=0.74 (30x too tall).
  → LilyPond barline height = full staff system height.
  → Width is already correct; only height needs clamping.

FINDING 5: The glyph-group bbox size is VARIABLE (CV 0.3-0.5), not fixed.
  → Cannot use a fixed pixel-offset approach.
  → Must use true symbol size from DoReMi as target, and center at glyph-group center.

IMPLICATION: The universal fix strategy is:
  For EACH annotation in an OpenScore file:
    new_cx = old_cx  (glyph-group center ≈ symbol center for most classes)
    new_cy = old_cy  (same reason)
    new_w  = DoReMi_median_w  (replace with true symbol width)
    new_h  = DoReMi_median_h  (replace with true symbol height)

  EXCEPT:
    barline: keep cx,cy,w (correct); replace h = DoReMi_median_h (0.025)
    clef:    cx,cy from glyph-group, but offset cx LEFT by ~0.7*notehead_w
             to get actual clef position (clef is at start of measure, not at note col)
    accidentals with key_signature sharing: need special handling
""")


if __name__ == "__main__":
    main()
