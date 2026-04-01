#!/usr/bin/env python3
"""
Comprehensive OpenScore bbox analysis for ALL classes.

Compares OpenScore (lg-* in filename) vs DoReMi annotation sizes,
finds classes with oversized bboxes, and analyzes glyph-group co-location
to determine which classes can be fixed with a simple position rule.
"""

import os
import sys
import numpy as np
import collections
from pathlib import Path
from typing import Dict, List, Tuple, Optional

LABEL_DIR = Path("/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase8_final/val/labels")

CLASS_NAMES = {
    0: "notehead_filled",
    1: "notehead_hollow",
    2: "stem",
    3: "beam",
    4: "flag_8th",
    5: "flag_16th",
    6: "flag_32nd",
    7: "augmentation_dot",
    8: "tie",
    9: "clef_treble",
    10: "clef_bass",
    11: "clef_alto",
    12: "clef_tenor",
    13: "accidental_sharp",
    14: "accidental_flat",
    15: "accidental_natural",
    16: "accidental_double_sharp",
    17: "accidental_double_flat",
    18: "rest_whole",
    19: "rest_half",
    20: "rest_quarter",
    21: "rest_8th",
    22: "rest_16th",
    23: "barline",
    24: "barline_double",
    25: "barline_final",
    26: "barline_repeat",
    27: "time_signature",
    28: "key_signature",
    29: "fermata",
    30: "dynamic_soft",
    31: "dynamic_loud",
    32: "ledger_line",
}
N_CLASSES = 33


# ---------------------------------------------------------------------------
# File categorisation helpers
# ---------------------------------------------------------------------------

def categorise_file(fname: str) -> str:
    """Return 'openscore', 'doremi', 'muscima', or 'other'."""
    if "lg-" in fname:
        return "openscore"
    if "doremi" in fname:
        return "doremi"
    if "muscima" in fname:
        return "muscima"
    return "other"


def load_labels(path: Path) -> np.ndarray:
    """
    Load YOLO label file.
    Returns array shape (N,5): cls cx cy w h — all float.
    Returns empty (0,5) if file is empty or unreadable.
    """
    try:
        data = np.loadtxt(path, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.size == 0:
            return np.zeros((0, 5), dtype=float)
        return data
    except Exception:
        return np.zeros((0, 5), dtype=float)


# ---------------------------------------------------------------------------
# Phase 1: aggregate per-class width/height statistics by source
# ---------------------------------------------------------------------------

def collect_stats():
    openscore_wh  = collections.defaultdict(list)   # cls -> [(w,h)]
    doremi_wh     = collections.defaultdict(list)
    openscore_files = {}    # fname -> np.ndarray labels

    all_files = sorted(LABEL_DIR.glob("*.txt"))
    n_os = n_dr = n_other = 0

    for fpath in all_files:
        cat = categorise_file(fpath.name)
        labels = load_labels(fpath)
        if labels.shape[0] == 0:
            continue

        if cat == "openscore":
            n_os += 1
            openscore_files[fpath.name] = labels
            for row in labels:
                cls = int(row[0])
                openscore_wh[cls].append((row[3], row[4]))
        elif cat == "doremi":
            n_dr += 1
            for row in labels:
                cls = int(row[0])
                doremi_wh[cls].append((row[3], row[4]))
        else:
            n_other += 1

    print(f"Files loaded  —  OpenScore: {n_os}  |  DoReMi: {n_dr}  |  Other: {n_other}")
    return openscore_wh, doremi_wh, openscore_files


# ---------------------------------------------------------------------------
# Phase 2: per-class comparison table
# ---------------------------------------------------------------------------

def bbox_stats(wh_list):
    if not wh_list:
        return None
    ws = np.array([x[0] for x in wh_list])
    hs = np.array([x[1] for x in wh_list])
    return {
        "n": len(wh_list),
        "med_w": float(np.median(ws)),
        "med_h": float(np.median(hs)),
    }


def print_comparison_table(openscore_wh, doremi_wh):
    print("\n" + "=" * 115)
    print(f"{'Cls':>3}  {'Class Name':<28} {'OS_n':>6} {'OS_medW':>8} {'OS_medH':>8}"
          f" {'DR_n':>6} {'DR_medW':>8} {'DR_medH':>8}"
          f" {'R_W':>6} {'R_H':>6} {'FLAG':<10}")
    print("-" * 115)

    oversized = []
    results = {}

    for cls in range(N_CLASSES):
        os_s = bbox_stats(openscore_wh.get(cls, []))
        dr_s = bbox_stats(doremi_wh.get(cls, []))
        name = CLASS_NAMES[cls]

        os_n = os_s["n"] if os_s else 0
        dr_n = dr_s["n"] if dr_s else 0

        if os_s and dr_s:
            rw = os_s["med_w"] / dr_s["med_w"] if dr_s["med_w"] > 0 else float("nan")
            rh = os_s["med_h"] / dr_s["med_h"] if dr_s["med_h"] > 0 else float("nan")
            flag = ""
            if rw > 2.0 or rh > 2.0:
                flag = "*** OVERSIZED"
                oversized.append(cls)
            elif rw > 1.5 or rh > 1.5:
                flag = "slightly"
            print(f"{cls:>3}  {name:<28} {os_n:>6} {os_s['med_w']:>8.5f} {os_s['med_h']:>8.5f}"
                  f" {dr_n:>6} {dr_s['med_w']:>8.5f} {dr_s['med_h']:>8.5f}"
                  f" {rw:>6.2f} {rh:>6.2f}  {flag}")
            results[cls] = dict(os=os_s, dr=dr_s, rw=rw, rh=rh, flag=flag)
        elif os_s:
            print(f"{cls:>3}  {name:<28} {os_n:>6} {os_s['med_w']:>8.5f} {os_s['med_h']:>8.5f}"
                  f" {'—':>6} {'—':>8} {'—':>8}  {'—':>6} {'—':>6}  (no DoReMi ref)")
            results[cls] = dict(os=os_s, dr=None, rw=None, rh=None, flag="no_ref")
        elif dr_s:
            print(f"{cls:>3}  {name:<28} {'—':>6} {'—':>8} {'—':>8}"
                  f" {dr_n:>6} {dr_s['med_w']:>8.5f} {dr_s['med_h']:>8.5f}  {'—':>6} {'—':>6}  (no OS data)")
            results[cls] = dict(os=None, dr=dr_s, rw=None, rh=None, flag="no_os")
        else:
            print(f"{cls:>3}  {name:<28} {'—':>6} {'—':>8} {'—':>8}"
                  f" {'—':>6} {'—':>8} {'—':>8}  {'—':>6} {'—':>6}  (no data)")
            results[cls] = dict(os=None, dr=None, rw=None, rh=None, flag="no_data")

    print("=" * 115)
    return oversized, results


# ---------------------------------------------------------------------------
# Phase 3: glyph-group co-location analysis
# ---------------------------------------------------------------------------

IOU_THRESH = 0.85
NOTEHEAD_CLASSES = {0, 1}


def iou_bbox(a, b):
    ax1 = a[0] - a[2]/2;  ax2 = a[0] + a[2]/2
    ay1 = a[1] - a[3]/2;  ay2 = a[1] + a[3]/2
    bx1 = b[0] - b[2]/2;  bx2 = b[0] + b[2]/2
    by1 = b[1] - b[3]/2;  by2 = b[1] + b[3]/2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def analyse_glyph_groups(openscore_files, oversized_classes, results):
    print("\n" + "=" * 100)
    print("PHASE 3: GLYPH-GROUP CO-LOCATION ANALYSIS  (IoU >= 0.85 = same bbox)")
    print("=" * 100)

    oversized_set = set(oversized_classes)

    # counters per class
    total       = collections.defaultdict(int)
    coloc_nh    = collections.defaultdict(int)   # co-located with a notehead
    coloc_os    = collections.defaultdict(int)   # co-located with another oversized
    larger_nh   = collections.defaultdict(int)   # annotation area >= notehead area

    for fname, labels in openscore_files.items():
        if labels.shape[0] == 0:
            continue
        nh_rows  = labels[np.isin(labels[:,0].astype(int), list(NOTEHEAD_CLASSES))]
        ovs_rows = labels[np.isin(labels[:,0].astype(int), list(oversized_set))]

        for row in labels:
            cls = int(row[0])
            if cls not in oversized_set:
                continue
            box = row[1:5]
            total[cls] += 1

            # co-location with notehead
            matched_nh = False
            for nh in nh_rows:
                if iou_bbox(box, nh[1:5]) >= IOU_THRESH:
                    matched_nh = True
                    if box[2]*box[3] >= nh[1:5][2]*nh[1:5][3]:
                        larger_nh[cls] += 1
                    break
            if matched_nh:
                coloc_nh[cls] += 1

            # co-location with any other oversized class
            for other in ovs_rows:
                if int(other[0]) == cls:
                    continue
                if iou_bbox(box, other[1:5]) >= IOU_THRESH:
                    coloc_os[cls] += 1
                    break

    print(f"\n{'Cls':>3}  {'Class Name':<28} {'Total':>7} "
          f"{'ColocNH':>8} {'%NH':>6} {'ColocOtherOS':>13} "
          f"{'Larger?':>8}  Strategy")
    print("-" * 110)

    fix_recs = {}

    for cls in oversized_classes:
        name = CLASS_NAMES[cls]
        tot  = total[cls]
        cn   = coloc_nh[cls]
        co   = coloc_os[cls]
        lnh  = larger_nh[cls]
        pct  = 100.0 * cn / tot if tot > 0 else 0.0

        rw = results[cls]["rw"] or 0
        rh = results[cls]["rh"] or 0

        if pct >= 70:
            if lnh > cn * 0.5:
                strat = "shrink → top-edge anchor"
            else:
                strat = "shrink → centre anchor"
        elif pct >= 30:
            strat = "partial: needs per-image detection"
        else:
            strat = "independent — check manually"

        fix_recs[cls] = strat
        print(f"{cls:>3}  {name:<28} {tot:>7} "
              f"{cn:>8} {pct:>5.1f}% {co:>13} "
              f"{lnh:>8}  {strat}")

    return fix_recs


# ---------------------------------------------------------------------------
# Phase 4: positional deep-dive for top oversized classes
# ---------------------------------------------------------------------------

def deep_dive_position(openscore_files, oversized_classes, results, top_n=10):
    ranked = sorted(
        [c for c in oversized_classes if results[c]["rw"] is not None],
        key=lambda c: max(results[c]["rw"] or 0, results[c]["rh"] or 0),
        reverse=True,
    )[:top_n]

    print("\n" + "=" * 100)
    print("PHASE 4: POSITIONAL DEEP-DIVE  (relative to co-located notehead bbox)")
    print("  rel_y=0 → annotation centre at TOP of notehead bbox")
    print("  rel_y=1 → annotation centre at BOTTOM of notehead bbox")
    print("  size_W/H = annotation size / notehead bbox size")
    print("=" * 100)

    for cls in ranked:
        name = CLASS_NAMES[cls]
        rel_y, rel_x, sw, sh = [], [], [], []

        for fname, labels in openscore_files.items():
            if labels.shape[0] == 0:
                continue
            nh_rows = labels[np.isin(labels[:,0].astype(int), list(NOTEHEAD_CLASSES))]
            for row in labels:
                if int(row[0]) != cls:
                    continue
                box = row[1:5]
                for nh in nh_rows:
                    nhb = nh[1:5]
                    if iou_bbox(box, nhb) >= IOU_THRESH:
                        nh_top = nhb[1] - nhb[3]/2
                        if nhb[3] > 0:
                            rel_y.append((box[1] - nh_top) / nhb[3])
                        if nhb[2] > 0:
                            nh_left = nhb[0] - nhb[2]/2
                            rel_x.append((box[0] - nh_left) / nhb[2])
                        if nhb[2] > 0:
                            sw.append(box[2] / nhb[2])
                        if nhb[3] > 0:
                            sh.append(box[3] / nhb[3])
                        break

        if not rel_y:
            print(f"\n  {cls:>2} {name:<28}  (no co-located noteheads)")
            continue

        ry = np.array(rel_y)
        rx = np.array(rel_x)
        sw_a = np.array(sw)
        sh_a = np.array(sh)
        rw = results[cls]["rw"]
        rh = results[cls]["rh"]

        print(f"\n  Cls {cls:>2}  {name}  [OS/DR: W={rw:.2f}x H={rh:.2f}x  n_colocated={len(ry)}]")
        print(f"    rel_Y  p5={np.percentile(ry,5):.3f}  p25={np.percentile(ry,25):.3f}  "
              f"med={np.median(ry):.3f}  p75={np.percentile(ry,75):.3f}  p95={np.percentile(ry,95):.3f}")
        print(f"    rel_X  p5={np.percentile(rx,5):.3f}  p25={np.percentile(rx,25):.3f}  "
              f"med={np.median(rx):.3f}  p75={np.percentile(rx,75):.3f}  p95={np.percentile(rx,95):.3f}")
        print(f"    size_W p5={np.percentile(sw_a,5):.3f}  med={np.median(sw_a):.3f}  p95={np.percentile(sw_a,95):.3f}  "
              f"(1.0 = same width as notehead bbox)")
        print(f"    size_H p5={np.percentile(sh_a,5):.3f}  med={np.median(sh_a):.3f}  p95={np.percentile(sh_a,95):.3f}")


# ---------------------------------------------------------------------------
# Phase 5: cross-class co-location matrix
# ---------------------------------------------------------------------------

def cross_class_matrix(openscore_files, oversized_classes):
    cls_list = oversized_classes
    n = len(cls_list)
    idx = {c: i for i, c in enumerate(cls_list)}
    matrix = np.zeros((n, n), dtype=int)

    for fname, labels in openscore_files.items():
        if labels.shape[0] == 0:
            continue
        by_cls = {c: [] for c in cls_list}
        for row in labels:
            c = int(row[0])
            if c in idx:
                by_cls[c].append(row[1:5])

        for i, c1 in enumerate(cls_list):
            for j, c2 in enumerate(cls_list):
                if j <= i:
                    continue
                for b1 in by_cls[c1]:
                    matched = False
                    for b2 in by_cls[c2]:
                        if iou_bbox(b1, b2) >= IOU_THRESH:
                            matrix[i, j] += 1
                            matrix[j, i] += 1
                            matched = True
                            break
                    if matched:
                        break

    print("\n" + "=" * 100)
    print("PHASE 5: CROSS-CLASS CO-LOCATION  (pairs sharing same bbox in same file)")
    print("=" * 100)
    # short names for header
    short = [CLASS_NAMES[c][:11] for c in cls_list]
    hdr = f"{'':>14}" + "".join(f"{s:>13}" for s in short)
    print(hdr)
    for i, c in enumerate(cls_list):
        row = f"{CLASS_NAMES[c][:14]:>14}" + "".join(
            f"{'—':>13}" if i == j else f"{matrix[i,j]:>13}" for j in range(n)
        )
        print(row)


# ---------------------------------------------------------------------------
# Phase 6: final recommendations
# ---------------------------------------------------------------------------

def print_recommendations(oversized_classes, results, fix_recs):
    print("\n" + "=" * 100)
    print("PHASE 6: FIX RECOMMENDATIONS SUMMARY")
    print("=" * 100)
    print(f"\n{'Cls':>3}  {'Class Name':<30} {'W ratio':>8} {'H ratio':>8}  Strategy")
    print("-" * 90)

    by_strat = collections.defaultdict(list)
    ranked = sorted(oversized_classes, key=lambda c: -max(results[c]["rw"] or 0, results[c]["rh"] or 0))
    for cls in ranked:
        name = CLASS_NAMES[cls]
        rw = results[cls]["rw"] or 0
        rh = results[cls]["rh"] or 0
        strat = fix_recs.get(cls, "unknown")
        by_strat[strat].append(cls)
        print(f"{cls:>3}  {name:<30} {rw:>8.2f} {rh:>8.2f}  {strat}")

    print("\n--- ACTIONABLE GROUPS ---")
    for strat, cls_list in sorted(by_strat.items(), key=lambda x: -len(x[1])):
        names = [CLASS_NAMES[c] for c in cls_list]
        print(f"\n  [{strat}]  ({len(cls_list)} classes)")
        for c, nm in zip(cls_list, names):
            rw = results[c]["rw"] or 0
            rh = results[c]["rh"] or 0
            print(f"      {c:>2} {nm:<28}  W×{rw:.2f} H×{rh:.2f}")

    print("""
LEGEND:
  shrink → top-edge anchor : place symbol bbox at the TOP of the LilyPond glyph-group box
                              (notehead, dot, accidental — all at top of column glyph)
  shrink → centre anchor   : place symbol bbox centred on glyph-group box
  partial / independent    : need case-by-case inspection or are already correct
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("OpenScore bbox Annotation Analysis — ALL 33 Classes  (Phase 8 val set)")
    print("=" * 100)

    openscore_wh, doremi_wh, openscore_files = collect_stats()
    os_total = sum(len(v) for v in openscore_wh.values())
    dr_total = sum(len(v) for v in doremi_wh.values())
    print(f"Total annotations  —  OpenScore: {os_total:,}  |  DoReMi: {dr_total:,}")

    print("\nPHASE 2: SIZE COMPARISON TABLE")
    oversized, results = print_comparison_table(openscore_wh, doremi_wh)
    print(f"\nOversized classes (ratio > 2x): {len(oversized)}")
    for c in sorted(oversized, key=lambda x: -max(results[x]["rw"] or 0, results[x]["rh"] or 0)):
        print(f"  {c:>2} {CLASS_NAMES[c]:<28}  W×{results[c]['rw']:.2f}  H×{results[c]['rh']:.2f}")

    fix_recs = analyse_glyph_groups(openscore_files, oversized, results)
    deep_dive_position(openscore_files, oversized, results, top_n=10)
    cross_class_matrix(openscore_files, oversized)
    print_recommendations(oversized, results, fix_recs)
    print("\nDone.")


if __name__ == "__main__":
    main()
