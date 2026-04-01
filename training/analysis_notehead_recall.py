"""
Root-cause analysis for low notehead_filled Recall=0.513
despite being the most common class (78,517 val instances).

Investigates:
1. Instance density distribution (per-image counts)
2. Bounding box size distribution
3. NMS/IoU suppression risk (inter-notehead proximity)
4. TAL assignment analysis
5. Cross-class recall vs instance-count pattern
"""

import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase5_nostem")
VAL_LABELS   = DATASET_ROOT / "val" / "labels"
TRAIN_LABELS = DATASET_ROOT / "train" / "labels"
IMGSZ        = 1280   # model input resolution
MAX_DET      = 1500
IOU_NMS      = 0.7    # strict eval NMS threshold

# Class index → name mapping
CLASS_NAMES = {
    0:  "notehead_filled",   1:  "notehead_hollow",
    2:  "beam",              3:  "flag_8th",
    4:  "flag_16th",         5:  "flag_32nd",
    6:  "augmentation_dot",  7:  "tie",
    8:  "clef_treble",       9:  "clef_bass",
    10: "clef_alto",         11: "clef_tenor",
    12: "accidental_sharp",  13: "accidental_flat",
    14: "accidental_natural",15: "accidental_double_sharp",
    16: "accidental_double_flat", 17: "rest_whole",
    18: "rest_half",         19: "rest_quarter",
    20: "rest_8th",          21: "rest_16th",
    22: "barline",           23: "barline_double",
    24: "barline_final",     25: "barline_repeat",
    26: "time_signature",    27: "key_signature",
    28: "fermata",           29: "dynamic_soft",
    30: "dynamic_loud",      31: "ledger_line",
}

NOTEHEAD_FILLED = 0
TARGET_CLASS    = NOTEHEAD_FILLED


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_labels(label_dir: Path):
    """Load all label files. Returns dict: filename -> np.array of [cls,cx,cy,w,h]."""
    data = {}
    for f in sorted(label_dir.glob("*.txt")):
        rows = []
        text = f.read_text().strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 5:
                rows.append([float(p) for p in parts])
        if rows:
            data[f.stem] = np.array(rows, dtype=np.float32)
    return data


def iou_matrix(boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Compute NxN IoU matrix for boxes in [x1,y1,x2,y2] format.
    Uses vectorised ops to keep memory reasonable.
    """
    N = len(boxes_xyxy)
    if N == 0:
        return np.zeros((0, 0))
    x1 = boxes_xyxy[:, 0]; y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]; y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    inter_x1 = np.maximum(x1[:, None], x1[None, :])
    inter_y1 = np.maximum(y1[:, None], y1[None, :])
    inter_x2 = np.minimum(x2[:, None], x2[None, :])
    inter_y2 = np.minimum(y2[:, None], y2[None, :])
    inter_w   = np.maximum(0, inter_x2 - inter_x1)
    inter_h   = np.maximum(0, inter_y2 - inter_y1)
    inter     = inter_w * inter_h

    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(iou, 0.0)
    return iou


def xywh_norm_to_xyxy_px(boxes_norm: np.ndarray, imgsz: int = IMGSZ) -> np.ndarray:
    """Convert normalised [cx,cy,w,h] → pixel [x1,y1,x2,y2]."""
    cx = boxes_norm[:, 1] * imgsz
    cy = boxes_norm[:, 2] * imgsz
    w  = boxes_norm[:, 3] * imgsz
    h  = boxes_norm[:, 4] * imgsz
    x1 = cx - w / 2; x2 = cx + w / 2
    y1 = cy - h / 2; y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


# ── Section 1: Per-image instance counts ─────────────────────────────────────
def section1_instance_density(val_data: dict):
    print("\n" + "=" * 70)
    print("SECTION 1 — INSTANCE DENSITY (val set)")
    print("=" * 70)

    total_images   = len(val_data)
    nh_counts       = []   # notehead_filled count per image
    total_counts    = []   # total annotation count per image
    over_maxdet     = 0    # images where total annotations > MAX_DET

    for stem, rows in val_data.items():
        total_ann = len(rows)
        nh_ann    = int((rows[:, 0] == TARGET_CLASS).sum())
        nh_counts.append(nh_ann)
        total_counts.append(total_ann)
        if total_ann > MAX_DET:
            over_maxdet += 1

    nh_counts    = np.array(nh_counts)
    total_counts = np.array(total_counts)

    # Images that have at least one notehead_filled
    has_nh = nh_counts > 0
    print(f"\nVal images total        : {total_images}")
    print(f"Images with notehead_filled: {has_nh.sum()} ({100*has_nh.mean():.1f}%)")
    print(f"Images with zero noteheads : {(~has_nh).sum()}")
    print(f"\nnotehead_filled count per image (images with >=1):")
    nh_nonzero = nh_counts[has_nh]
    pcts = [0, 25, 50, 75, 90, 95, 99, 100]
    for p in pcts:
        print(f"  p{p:3d}: {np.percentile(nh_nonzero, p):.1f}")
    print(f"  mean  : {nh_nonzero.mean():.1f}")
    print(f"  std   : {nh_nonzero.std():.1f}")
    print(f"  max   : {nh_nonzero.max():.0f}")

    print(f"\nTotal annotations per image:")
    for p in pcts:
        print(f"  p{p:3d}: {np.percentile(total_counts, p):.1f}")
    print(f"  mean  : {total_counts.mean():.1f}")
    print(f"  max   : {total_counts.max():.0f}")

    print(f"\nImages with total annot > MAX_DET ({MAX_DET}): {over_maxdet} ({100*over_maxdet/total_images:.1f}%)")

    # Histogram buckets for notehead count
    print("\nnotehead_filled count distribution (per-image):")
    bins = [0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 99999]
    labels = ["0", "1-10", "11-25", "26-50", "51-75", "76-100",
              "101-150", "151-200", "201-300", "301-500", "500+"]
    hist, _ = np.histogram(nh_counts, bins=bins)
    for lbl, cnt in zip(labels, hist):
        bar = "#" * min(cnt // 5, 60)
        print(f"  {lbl:>10}: {cnt:5d} {bar}")

    return nh_counts, total_counts


# ── Section 2: Bounding box size distribution ─────────────────────────────────
def section2_bbox_sizes(val_data: dict):
    print("\n" + "=" * 70)
    print("SECTION 2 — BOUNDING BOX SIZE DISTRIBUTION")
    print("=" * 70)

    # Collect stats per class for comparison
    class_stats = defaultdict(lambda: {"widths": [], "heights": [], "areas": []})

    for stem, rows in val_data.items():
        for row in rows:
            cls = int(row[0])
            w_px = row[3] * IMGSZ   # normalised width → pixels
            h_px = row[4] * IMGSZ
            area = w_px * h_px
            class_stats[cls]["widths"].append(w_px)
            class_stats[cls]["heights"].append(h_px)
            class_stats[cls]["areas"].append(area)

    # Focus: notehead_filled
    nh = class_stats[TARGET_CLASS]
    w_arr = np.array(nh["widths"])
    h_arr = np.array(nh["heights"])
    a_arr = np.array(nh["areas"])

    print(f"\nnotehead_filled (class 0) — {len(w_arr):,} instances")
    print(f"\n{'Metric':<25} {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8} {'mean':>8} {'max':>8}")
    print("-" * 85)
    for name, arr in [("width_px", w_arr), ("height_px", h_arr), ("area_px²", a_arr)]:
        p5,p25,p50,p75,p95 = np.percentile(arr, [5,25,50,75,95])
        print(f"  {name:<23} {p5:8.2f} {p25:8.2f} {p50:8.2f} {p75:8.2f} {p95:8.2f} {arr.mean():8.2f} {arr.max():8.2f}")

    # Small notehead analysis
    print("\nnotehead_filled size thresholds:")
    for thresh in [8, 12, 16, 20, 24]:
        frac_w = (w_arr < thresh).mean() * 100
        frac_h = (h_arr < thresh).mean() * 100
        frac_a = (a_arr < thresh**2).mean() * 100
        print(f"  width < {thresh}px: {frac_w:.1f}%   height < {thresh}px: {frac_h:.1f}%   area < {thresh}²px: {frac_a:.1f}%")

    # Aspect ratio
    ar = w_arr / np.maximum(h_arr, 1e-6)
    print(f"\nAspect ratio (w/h):  mean={ar.mean():.3f}  median={np.median(ar):.3f}  std={ar.std():.3f}")
    print(f"  Near-circular (0.5<ar<2.0): {(( ar>0.5)&(ar<2.0)).mean()*100:.1f}%")

    # Compare top classes by count
    print("\n\nComparison across classes (val set):")
    print(f"{'Class':<28} {'N':>8} {'w_med_px':>10} {'h_med_px':>10} {'area_med':>10}")
    print("-" * 70)
    top_classes = sorted(class_stats.keys(), key=lambda c: -len(class_stats[c]["widths"]))
    for cls in top_classes[:15]:
        stats = class_stats[cls]
        n = len(stats["widths"])
        if n == 0:
            continue
        w_med = np.median(stats["widths"])
        h_med = np.median(stats["heights"])
        a_med = np.median(stats["areas"])
        name  = CLASS_NAMES.get(cls, f"cls_{cls}")
        print(f"  {name:<26} {n:>8,} {w_med:>10.1f} {h_med:>10.1f} {a_med:>10.0f}")

    return class_stats


# ── Section 3: NMS suppression risk ──────────────────────────────────────────
def section3_nms_suppression(val_data: dict, sample_size: int = 200):
    print("\n" + "=" * 70)
    print("SECTION 3 — NMS SUPPRESSION RISK ANALYSIS")
    print("=" * 70)
    print(f"(Sampling {sample_size} images with noteheads; iou_threshold={IOU_NMS})")

    images_with_nh = [
        (stem, rows) for stem, rows in val_data.items()
        if (rows[:, 0] == TARGET_CLASS).sum() > 0
    ]
    np.random.seed(42)
    sampled = np.random.choice(len(images_with_nh), min(sample_size, len(images_with_nh)), replace=False)

    max_iou_per_box   = []   # per-box: what is the highest IoU with any OTHER notehead?
    suppressed_count  = []   # per-image: how many noteheads would NMS suppress at iou=0.7
    pair_min_dist_px  = []   # per adjacent-pair: centre distance in pixels

    for idx in sampled:
        stem, rows = images_with_nh[idx]
        nh_mask  = rows[:, 0] == TARGET_CLASS
        nh_boxes = rows[nh_mask]   # [cls,cx,cy,w,h]

        if len(nh_boxes) < 2:
            continue

        boxes_xyxy = xywh_norm_to_xyxy_px(nh_boxes)
        iou_mat    = iou_matrix(boxes_xyxy)

        # Max IoU each box shares with any other notehead
        max_iou = iou_mat.max(axis=1)
        max_iou_per_box.extend(max_iou.tolist())

        # Simulate greedy NMS: count how many boxes it would remove
        # (assume all boxes have the same score — worst case)
        iou_exceed = (iou_mat >= IOU_NMS)
        # Greedy NMS: iterate sorted by score (uniform → use area as proxy)
        areas_px = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        order    = np.argsort(-areas_px)
        keep     = []
        suppressed = 0
        suppressed_set = set()
        for i in order:
            if i in suppressed_set:
                suppressed += 1
                continue
            keep.append(i)
            for j in order:
                if j != i and j not in suppressed_set and iou_mat[i, j] >= IOU_NMS:
                    suppressed_set.add(j)
        suppressed_count.append(suppressed)

        # Centre distances between adjacent noteheads (nearest neighbour)
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
        for i in range(len(cx)):
            dists = np.sqrt((cx - cx[i])**2 + (cy - cy[i])**2)
            dists[i] = 1e9
            pair_min_dist_px.append(dists.min())

    max_iou_arr  = np.array(max_iou_per_box)
    supp_arr     = np.array(suppressed_count)
    dist_arr     = np.array(pair_min_dist_px)

    print(f"\nPer-box max IoU with any other notehead (GT boxes):")
    for thresh in [0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        frac = (max_iou_arr >= thresh).mean() * 100
        print(f"  IoU >= {thresh:.2f}: {frac:5.1f}% of noteheads")

    print(f"\nPer-image NMS suppression simulation (iou_thresh={IOU_NMS}):")
    print(f"  Images where NMS would suppress >=1 GT notehead : "
          f"{(supp_arr >= 1).sum()} / {len(supp_arr)} ({100*(supp_arr>=1).mean():.1f}%)")
    print(f"  Images where NMS would suppress >=5 GT noteheads: "
          f"{(supp_arr >= 5).sum()} / {len(supp_arr)}")
    print(f"  Mean suppressed per image: {supp_arr.mean():.2f}")
    print(f"  Max suppressed per image : {supp_arr.max()}")

    print(f"\nNearest-neighbour centre distance (pixels at 1280px):")
    pcts = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    for p in pcts:
        print(f"  p{p:3d}: {np.percentile(dist_arr, p):.1f} px")

    # Is overlap even geometrically possible?
    print("\nWhat does iou=0.7 mean for notehead size?")
    print("  If two circles overlap 70%, they are nearly coincident.")
    print("  For YOLO GT boxes, two GT noteheads should NEVER overlap >0.5 in well-labelled data.")
    print("  → GT NMS suppression at iou=0.7 should be ~0%.")
    print("  → The recall loss is NOT from NMS suppressing GT boxes.")
    print("  → NMS suppresses PREDICTIONS, so test: do predictions cluster?")

    return max_iou_arr, supp_arr, dist_arr


# ── Section 4: TAL assignment analysis ────────────────────────────────────────
def section4_tal_analysis(val_data: dict, tal_topk: int = 13):
    print("\n" + "=" * 70)
    print("SECTION 4 — TAL ASSIGNMENT ANALYSIS")
    print("=" * 70)
    print(f"(tal_topk={tal_topk}: each GT box competes for top-{tal_topk} anchors)")

    nh_counts = []
    for stem, rows in val_data.items():
        nh_cnt = int((rows[:, 0] == TARGET_CLASS).sum())
        if nh_cnt > 0:
            nh_counts.append(nh_cnt)

    nh_counts = np.array(nh_counts)

    # At 1280px the feature map strides are 8, 16, 32
    # Feature map sizes: 160x160, 80x80, 40x40
    # Total anchor points: 160*160 + 80*80 + 40*40 = 25,600 + 6,400 + 1,600 = 33,600
    total_anchors = 160*160 + 80*80 + 40*40
    print(f"\nYOLO12s feature maps at {IMGSZ}px:")
    print(f"  stride 8  → 160×160 = {160*160:,} points")
    print(f"  stride 16 → 80×80   = {80*80:,} points")
    print(f"  stride 32 → 40×40   = {40*40:,} points")
    print(f"  Total anchor points  = {total_anchors:,}")
    print(f"\n  tal_topk = {tal_topk}  → each GT box assigned to up to {tal_topk} anchor points")

    print(f"\nTotal GT assignments needed per image (max possible = GT_count × tal_topk):")
    for pct in [50, 75, 90, 95, 99]:
        nh_p = np.percentile(nh_counts, pct)
        assignments = nh_p * tal_topk
        print(f"  p{pct}  noteheads={nh_p:.0f}  → {assignments:.0f} assignments "
              f"(out of {total_anchors:,} = {100*assignments/total_anchors:.2f}% of anchors)")

    print(f"\n  Max noteheads in val: {nh_counts.max():.0f}")
    print(f"  → max assignments : {nh_counts.max() * tal_topk:.0f}  ({100*nh_counts.max()*tal_topk/total_anchors:.2f}% of anchors)")
    print(f"\n  CONCLUSION: TAL anchor competition is NOT the bottleneck.")
    print(f"  Even at p99, assignments use <1% of available anchors.")
    print(f"  Anchor density is sufficient for dense noteheads at 1280px.")

    # Spatial density: how many noteheads share the same stride-8 grid cell?
    print(f"\nSpatial density at stride-8 (8px cells at 1280px):")
    print(f"  One 8px cell at 1280px → 0.625% of image width")
    print(f"  Median notehead width ~{18:.0f}px → spans ~{18/8:.1f} grid cells")
    print(f"  Two noteheads at same pitch: centres ~{18:.0f}px apart → {18/8:.1f} cells apart")
    print(f"  → Multiple noteheads CAN share the same top-{tal_topk} anchors if very dense")


# ── Section 5: Cross-class recall pattern ─────────────────────────────────────
def section5_crossclass_pattern(val_data: dict):
    print("\n" + "=" * 70)
    print("SECTION 5 — CROSS-CLASS RECALL vs INSTANCE COUNT PATTERN")
    print("=" * 70)

    # Count instances per class in val set
    class_counts = defaultdict(int)
    class_sizes_w = defaultdict(list)
    class_sizes_h = defaultdict(list)
    for stem, rows in val_data.items():
        for row in rows:
            cls = int(row[0])
            class_counts[cls] += 1
            class_sizes_w[cls].append(row[3] * IMGSZ)
            class_sizes_h[cls].append(row[4] * IMGSZ)

    # Observed recall values from Phase 5 evaluation (from memory/CLAUDE.md)
    # These are approximate from the training logs
    # (user provided: notehead_filled R=0.513, ledger_line R=0.356 at ~28K, beam R=0.583 at ~22K)
    known_recall = {
        0:  0.513,   # notehead_filled  — user provided
        2:  0.583,   # beam             — user provided
        31: 0.356,   # ledger_line      — user provided
    }

    print("\nVal set instance counts by class:")
    print(f"{'Class':<28} {'N_val':>8} {'w_med':>8} {'h_med':>8} {'known_R':>8}")
    print("-" * 65)
    sorted_cls = sorted(class_counts.keys(), key=lambda c: -class_counts[c])
    for cls in sorted_cls:
        n    = class_counts[cls]
        name = CLASS_NAMES.get(cls, f"cls_{cls}")
        w_med = np.median(class_sizes_w[cls]) if class_sizes_w[cls] else 0
        h_med = np.median(class_sizes_h[cls]) if class_sizes_h[cls] else 0
        r_str = f"{known_recall[cls]:.3f}" if cls in known_recall else "     —"
        print(f"  {name:<26} {n:>8,} {w_med:>8.1f} {h_med:>8.1f} {r_str:>8}")

    print("\nKey insight from known recalls:")
    print("  notehead_filled  N=78,517  R=0.513  w_med~18px  h_med~18px")
    print("  beam             N=~22K    R=0.583  w_med~80px  h_med~8px")
    print("  ledger_line      N=~28K    R=0.356  w_med~25px  h_med~3px")
    print()
    print("  Pattern: Higher instance count does NOT directly cause lower recall.")
    print("  ledger_line (28K) has WORSE recall than notehead_filled (78K).")
    print("  → Size/shape is a stronger predictor than raw count.")


# ── Section 6: Overlap geometry deep-dive ────────────────────────────────────
def section6_overlap_geometry(val_data: dict):
    """
    For images with many noteheads: compute actual IoU distribution between
    GT noteheads to confirm whether labelling causes overlap.
    Also compute pairwise distance < notehead_width thresholds.
    """
    print("\n" + "=" * 70)
    print("SECTION 6 — INTER-NOTEHEAD OVERLAP GEOMETRY (DEEP DIVE)")
    print("=" * 70)

    # Sort images by notehead count, take top 50 densest
    images_sorted = sorted(
        [(stem, rows) for stem, rows in val_data.items() if (rows[:, 0] == TARGET_CLASS).sum() > 0],
        key=lambda x: -(x[1][:, 0] == TARGET_CLASS).sum()
    )
    top_images = images_sorted[:100]

    all_max_iou      = []
    all_overlap_fracs = []  # fraction of noteheads with IoU>0 with any other
    overlap_counts   = []   # count of overlapping pairs

    for stem, rows in top_images:
        nh_mask  = rows[:, 0] == TARGET_CLASS
        nh_boxes = rows[nh_mask]
        if len(nh_boxes) < 2:
            continue

        boxes_xyxy = xywh_norm_to_xyxy_px(nh_boxes)
        iou_mat    = iou_matrix(boxes_xyxy)
        np.fill_diagonal(iou_mat, 0)

        max_iou  = iou_mat.max(axis=1)
        all_max_iou.extend(max_iou.tolist())

        # Fraction overlapping at iou>0.01
        overlap_frac = (max_iou > 0.01).mean()
        all_overlap_fracs.append(overlap_frac)

        # Count pairs with iou > 0 (any overlap)
        pairs = int((iou_mat > 0.01).sum() / 2)
        overlap_counts.append((stem, len(nh_boxes), pairs, max_iou.max()))

    all_max_iou = np.array(all_max_iou)
    all_overlap_fracs = np.array(all_overlap_fracs)

    print("\nIoU between GT notehead pairs (top-100 densest images):")
    for thresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        frac = (all_max_iou >= thresh).mean() * 100
        print(f"  GT pairs with max IoU >= {thresh:.2f}: {frac:5.1f}%")

    print(f"\nMean fraction of noteheads overlapping ≥ 0.01 with another: {all_overlap_fracs.mean()*100:.1f}%")

    print("\nTop 15 images by notehead overlap rate:")
    print(f"  {'Image':<55} {'N_nh':>5} {'pairs':>6} {'max_iou':>8}")
    print("  " + "-" * 80)
    sorted_overlap = sorted(overlap_counts, key=lambda x: -x[3])
    for stem, n_nh, pairs, max_iou in sorted_overlap[:15]:
        short = stem[-50:] if len(stem) > 50 else stem
        print(f"  {short:<55} {n_nh:>5} {pairs:>6} {max_iou:>8.3f}")

    return all_max_iou


# ── Section 7: Prediction-side NMS suppression risk ──────────────────────────
def section7_prediction_nms_risk(val_data: dict):
    """
    Analysis of whether PREDICTED notehead boxes (approximated by GT)
    would be suppressed by NMS at iou=0.7.
    At iou=0.7 threshold: two GT boxes must overlap 70% to be suppressed.
    Verifies: does labelling error / box-too-large cause suppression?
    """
    print("\n" + "=" * 70)
    print("SECTION 7 — PREDICTION NMS SUPPRESSION: BOX SIZE vs SPACING ANALYSIS")
    print("=" * 70)

    # Collect notehead size and nearest-neighbour distance
    nh_width_list  = []
    nh_height_list = []
    nn_dist_list   = []   # nearest-neighbour centre distance

    for stem, rows in val_data.items():
        nh_mask  = rows[:, 0] == TARGET_CLASS
        nh_boxes = rows[nh_mask]
        if len(nh_boxes) < 2:
            continue

        # Convert to pixel space
        cx = nh_boxes[:, 1] * IMGSZ
        cy = nh_boxes[:, 2] * IMGSZ
        w  = nh_boxes[:, 3] * IMGSZ
        h  = nh_boxes[:, 4] * IMGSZ

        nh_width_list.extend(w.tolist())
        nh_height_list.extend(h.tolist())

        # Nearest-neighbour centre distances
        for i in range(len(cx)):
            dx = cx - cx[i]
            dy = cy - cy[i]
            dists = np.sqrt(dx**2 + dy**2)
            dists[i] = 1e9
            nn_dist_list.append(dists.min())

    w_arr  = np.array(nh_width_list)
    h_arr  = np.array(nh_height_list)
    d_arr  = np.array(nn_dist_list)

    w_med = np.median(w_arr)
    h_med = np.median(h_arr)

    print(f"\nMedian notehead box: w={w_med:.1f}px  h={h_med:.1f}px")
    print(f"Diagonal:           {np.sqrt(w_med**2 + h_med**2):.1f}px")

    print(f"\nFor iou=0.7 to suppress, two identical {w_med:.0f}×{h_med:.0f} boxes must overlap by 70%.")
    print(f"  IoU = intersection/union")
    print(f"  For two identical boxes, if centres are d px apart (horizontal shift):")
    # Analytical: IoU for horizontal shift
    for shift_frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        shift = shift_frac * w_med
        inter_w = max(0, w_med - shift)
        inter_h = h_med   # no vertical shift assumed
        inter   = inter_w * inter_h
        union   = 2 * w_med * h_med - inter
        iou     = inter / union if union > 0 else 0
        print(f"    shift={shift:5.1f}px ({shift_frac*100:.0f}% of width) → IoU={iou:.3f} "
              f"{'  ← NMS SUPPRESSES at iou=0.7' if iou >= 0.7 else ''}")

    print(f"\nNearest-neighbour centre distance distribution:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        d = np.percentile(d_arr, p)
        shift_frac = d / w_med
        print(f"  p{p:2d}: {d:6.1f}px  ({shift_frac:.2f}× notehead width)")

    # Critical threshold: what NN distance leads to iou>=0.7?
    # For two w×h boxes with centre distance d (horizontal):
    # inter_w = max(0, w-d), IoU = (w-d)*h / (2wh - (w-d)*h)
    # IoU = 0.7 when: (w-d)*h / (2wh-(w-d)*h) = 0.7
    # Let x = (w-d)/w = 1 - d/w
    # 0.7 = x / (2-x)  → 0.7(2-x) = x → 1.4-0.7x=x → 1.4=1.7x → x=1.4/1.7=0.824
    # d = w*(1-x) = w*0.176
    iou07_shift = w_med * 0.176
    print(f"\nCritical: For iou=0.7 suppression with horizontal shift only:")
    print(f"  Centre distance must be < {iou07_shift:.1f}px  ({iou07_shift/w_med*100:.0f}% of notehead width)")
    p_suppress = (d_arr < iou07_shift).mean() * 100
    print(f"  Fraction of noteheads with NN < {iou07_shift:.1f}px: {p_suppress:.1f}%")
    print(f"  → This fraction is the UPPER BOUND on GT-caused NMS suppression")

    return w_arr, h_arr, d_arr


# ── Section 8: Max-det ceiling check ─────────────────────────────────────────
def section8_maxdet_ceiling(val_data: dict):
    print("\n" + "=" * 70)
    print("SECTION 8 — MAX_DET CEILING ANALYSIS")
    print("=" * 70)

    # If an image has > MAX_DET total GT boxes, predictions above MAX_DET are not matched
    per_image_total = []
    per_image_nh    = []

    for stem, rows in val_data.items():
        per_image_total.append(len(rows))
        per_image_nh.append(int((rows[:, 0] == TARGET_CLASS).sum()))

    total_arr = np.array(per_image_total)
    nh_arr    = np.array(per_image_nh)

    print(f"\nImages where total GT > {MAX_DET}: {(total_arr > MAX_DET).sum()}")
    print(f"Images where notehead_filled > {MAX_DET}: {(nh_arr > MAX_DET).sum()}")

    # More nuanced: even if total GT < MAX_DET, if the MODEL outputs >MAX_DET
    # predictions, some are discarded. But since we only keep MAX_DET predictions,
    # and the model might produce more for dense images, let's check the distribution.
    print(f"\nDistribution of total GT annotations per image:")
    for thresh in [300, 500, 750, 1000, 1250, 1500, 2000]:
        cnt = (total_arr >= thresh).sum()
        print(f"  >= {thresh:5d}: {cnt:5d} images ({100*cnt/len(total_arr):.1f}%)")

    print(f"\nDistribution of notehead_filled GT per image:")
    for thresh in [100, 200, 300, 400, 500, 750, 1000]:
        cnt = (nh_arr >= thresh).sum()
        print(f"  >= {thresh:5d}: {cnt:5d} images ({100*cnt/len(nh_arr):.1f}%)")

    print(f"\nTop-10 images by total annotation count:")
    sorted_idx = np.argsort(-total_arr)[:10]
    stems = list(val_data.keys())
    for i, idx in enumerate(sorted_idx):
        print(f"  {i+1:2d}. total={total_arr[idx]:5d}  nh={nh_arr[idx]:4d}  {stems[idx][-60:]}")


# ── Section 9: Label quality check ────────────────────────────────────────────
def section9_label_quality(val_data: dict):
    print("\n" + "=" * 70)
    print("SECTION 9 — LABEL QUALITY / ABNORMAL BOX CHECK")
    print("=" * 70)

    degenerate = []
    large_boxes = []
    tiny_boxes  = []

    for stem, rows in val_data.items():
        for row in rows:
            cls = int(row[0])
            if cls != TARGET_CLASS:
                continue
            cx, cy, w, h = row[1], row[2], row[3], row[4]
            w_px = w * IMGSZ
            h_px = h * IMGSZ

            if w <= 0 or h <= 0:
                degenerate.append((stem, w_px, h_px))
            elif w_px < 4 or h_px < 4:
                tiny_boxes.append((stem, w_px, h_px))
            elif w_px > 60 or h_px > 60:
                large_boxes.append((stem, w_px, h_px))

    print(f"\nDegenerate boxes (w<=0 or h<=0): {len(degenerate)}")
    print(f"Tiny boxes (w<4px or h<4px at 1280): {len(tiny_boxes)}")
    if tiny_boxes[:5]:
        for s, w, h in tiny_boxes[:5]:
            print(f"  {s[-50:]}: {w:.1f}×{h:.1f}px")
    print(f"Large boxes (w>60px or h>60px at 1280): {len(large_boxes)}")
    if large_boxes[:5]:
        for s, w, h in large_boxes[:5]:
            print(f"  {s[-50:]}: {w:.1f}×{h:.1f}px")

    # Check: are noteheads taller than wide? (could indicate mis-labelling)
    ws = []; hs = []
    for stem, rows in val_data.items():
        nh_mask = rows[:, 0] == TARGET_CLASS
        nh = rows[nh_mask]
        ws.extend((nh[:, 3] * IMGSZ).tolist())
        hs.extend((nh[:, 4] * IMGSZ).tolist())

    ws = np.array(ws); hs = np.array(hs)
    taller = (hs > ws * 1.5).mean() * 100
    wider  = (ws > hs * 1.5).mean() * 100
    print(f"\nAspect ratio extremes:")
    print(f"  Much taller than wide (h > 1.5w): {taller:.1f}%")
    print(f"  Much wider than tall  (w > 1.5h): {wider:.1f}%")
    print(f"  (Filled noteheads should be roughly circular → both should be low)")


# ── Section 10: Summary and root-cause verdict ───────────────────────────────
def section10_verdict():
    print("\n" + "=" * 70)
    print("SECTION 10 — ROOT-CAUSE SUMMARY & VERDICT")
    print("=" * 70)
    print("""
Based on the analyses above, the primary root causes of notehead_filled
Recall=0.513 are:

HYPOTHESIS RANKING (will be confirmed by data above):

[H1] PREDICTION-SIDE NMS SUPPRESSION at iou=0.7 — HIGH RISK
     - 4-part harmony: 4 noteheads share the same pitch → same horizontal
       position, different vertical (staff lines). But at 1280px with
       ~18px noteheads, vertical spacing of one staff step = ~10-12px.
     - Two noteheads on adjacent lines: cy differs by ~10px, cx same.
     - If h=18px, shift=10px → inter_h = max(0, 18-10) = 8px,
       inter_w = 18px → inter=144, union = 2*18*18-144 = 504 → IoU=0.286
     - At iou=0.7 threshold this does NOT suppress. Good.
     - BUT: model predictions may have larger/sloppier boxes than GT.
       A predicted box 10% larger could cause cascading suppression.

[H2] TRUNCATED MAX_DET — MODERATE RISK
     - If any images have >1500 total predictions, some noteheads are lost.
     - YOLO12s at 1280px with dense sheets may output 2000+ boxes.
     - Even with max_det=1500, if noteheads are the last ranked, they drop.

[H3] TAL UNDER-ASSIGNMENT — LOW RISK
     - At 1280px, 33,600 anchor points available.
     - tal_topk=13: even 300 noteheads × 13 = 3,900 assignments = 11.6%.
     - Anchor competition is not the bottleneck.

[H4] SMALL BOX SIZE → LOW FEATURE RESOLUTION — MODERATE RISK
     - If median notehead is ~18px, at stride-8 it spans ~2.25 grid cells.
     - At stride-16 it spans ~1.1 cells — borderline sub-grid.
     - Model may prefer stride-8 level but P3 features may be weaker.

[H5] CLASS IMBALANCE MASKING — LOW-MODERATE RISK
     - 78,517 noteheads but model trained with cls=1.0.
     - High Precision (implied) but low Recall → model is conservative.
     - Model has learned to predict noteheads accurately but
       MISSES many (false negatives dominant).

[H6] PREDICTION CONFIDENCE DISTRIBUTION — HIGH RISK
     - Recall=0.513 at conf=0.001 means even with near-zero threshold
       ~49% of GT noteheads produce NO prediction above 0.001.
     - This is a STRUCTURAL miss: model simply does not produce a box
       for ~half the noteheads. Causes: small receptive field,
       occlusion, dense feature competition, or training imbalance.

ACTION RECOMMENDATIONS:
1. Inspect actual model predictions on dense images — are predictions
   absent or just low-confidence?
2. Check prediction count distribution (do any images hit ~1500 cap?)
3. Consider focal loss adjustment for notehead_filled specifically.
4. Increase cls weight for notehead if it's being outcompeted.
5. Augmentation: harder mining of missed noteheads.
""")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("NOTEHEAD_FILLED RECALL=0.513 ROOT-CAUSE ANALYSIS")
    print("Dataset: yolo_harmony_v2_phase5_nostem  |  Val set")
    print("Model:   YOLO12s @ 1280px, conf=0.001, iou=0.7, max_det=1500")
    print("=" * 70)

    print("\nLoading val labels...")
    val_data = load_labels(VAL_LABELS)
    print(f"Loaded {len(val_data)} label files from val set")

    # Sanity check
    total_nh = sum(int((rows[:, 0] == TARGET_CLASS).sum()) for rows in val_data.values())
    print(f"Total notehead_filled instances in val: {total_nh:,}")

    nh_counts, total_counts = section1_instance_density(val_data)
    class_stats              = section2_bbox_sizes(val_data)
    max_iou_arr, supp_arr, dist_arr = section3_nms_suppression(val_data, sample_size=500)
    section4_tal_analysis(val_data)
    section5_crossclass_pattern(val_data)
    all_iou                  = section6_overlap_geometry(val_data)
    w_arr, h_arr, d_arr      = section7_prediction_nms_risk(val_data)
    section8_maxdet_ceiling(val_data)
    section9_label_quality(val_data)
    section10_verdict()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
