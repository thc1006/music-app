"""
DEEP DIVE: notehead_filled bbox anomaly investigation.

Findings so far:
- Median notehead width = 318px, height = 253px at 1280px image
- These are NOT individual noteheads (should be ~12-25px)
- 90.8% of GT noteheads in dense images have IoU>=0.7 with another notehead
- 80,000 large boxes (w>60px) flagged

This script:
1. Examines actual label files from dense images to understand what the boxes represent
2. Computes the FULL bimodal distribution of notehead sizes
3. Checks if the "large" noteheads are from specific image sources (augmented vs original)
4. Identifies whether this is a labelling strategy (group boxes) or a bug
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path("/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase5_nostem")
VAL_LABELS   = DATASET_ROOT / "val" / "labels"
VAL_IMAGES   = DATASET_ROOT / "val" / "images"
IMGSZ        = 1280
TARGET_CLASS = 0   # notehead_filled

CLASS_NAMES = {
    0: "notehead_filled", 1: "notehead_hollow", 2: "beam",
    6: "augmentation_dot", 12: "accidental_sharp", 13: "accidental_flat",
    14: "accidental_natural", 22: "barline", 31: "ledger_line",
}


def load_labels(label_dir: Path):
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


def section_a_bimodal_analysis(val_data: dict):
    """Check if notehead sizes are bimodal (small genuine vs large anomalous)."""
    print("=" * 70)
    print("SECTION A — BIMODAL SIZE ANALYSIS")
    print("=" * 70)

    all_w = []
    all_h = []
    source_by_size = defaultdict(list)  # 'small' or 'large' -> list of sources

    for stem, rows in val_data.items():
        nh_mask = rows[:, 0] == TARGET_CLASS
        nh = rows[nh_mask]
        for row in nh:
            w_px = row[3] * IMGSZ
            h_px = row[4] * IMGSZ
            all_w.append(w_px)
            all_h.append(h_px)

            # Classify by source prefix
            if 'muscima' in stem.lower():
                src = 'muscima'
            elif 'doremi' in stem.lower():
                src = 'doremi'
            elif 'beethoven' in stem.lower():
                src = 'beethoven'
            elif 'emmentaler' in stem.lower():
                src = 'emmentaler'
            elif 'lilyjazz' in stem.lower():
                src = 'lilyjazz'
            elif 'gutenberg' in stem.lower():
                src = 'gutenberg'
            else:
                src = 'other'

            if w_px < 50:
                source_by_size['small'].append(src)
            else:
                source_by_size['large'].append(src)

    all_w = np.array(all_w)
    all_h = np.array(all_h)

    print(f"\nFull width distribution (px at 1280):")
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 650]
    counts, _ = np.histogram(all_w, bins=bins)
    for i in range(len(bins)-1):
        bar = '#' * min(counts[i] // 200, 60)
        print(f"  {bins[i]:>4}-{bins[i+1]:>4}px: {counts[i]:6,}  {bar}")

    print(f"\nTotal notehead_filled: {len(all_w):,}")
    print(f"Width < 50px (likely genuine individual noteheads): {(all_w < 50).sum():,} ({100*(all_w<50).mean():.1f}%)")
    print(f"Width >= 50px (anomalously large): {(all_w >= 50).sum():,} ({100*(all_w>=50).mean():.1f}%)")
    print(f"Width < 30px: {(all_w < 30).sum():,} ({100*(all_w<30).mean():.1f}%)")

    # Source breakdown of small vs large
    from collections import Counter
    print(f"\nSource breakdown of SMALL noteheads (w<50px):")
    sc = Counter(source_by_size['small'])
    total_small = sum(sc.values())
    for src, cnt in sc.most_common():
        print(f"  {src:<20}: {cnt:6,} ({100*cnt/total_small:.1f}%)")

    print(f"\nSource breakdown of LARGE noteheads (w>=50px):")
    lc = Counter(source_by_size['large'])
    total_large = sum(lc.values())
    for src, cnt in lc.most_common():
        print(f"  {src:<20}: {cnt:6,} ({100*cnt/total_large:.1f}%)")

    return all_w, all_h, source_by_size


def section_b_inspect_actual_labels(val_data: dict):
    """Print actual label contents for 3 image types: muscima, beethoven, doremi."""
    print("\n" + "=" * 70)
    print("SECTION B — ACTUAL LABEL INSPECTION (sample files)")
    print("=" * 70)

    # Find one muscima, one beethoven, one doremi file with noteheads
    samples = {'muscima': None, 'beethoven': None, 'doremi': None}
    for stem, rows in val_data.items():
        nh_cnt = (rows[:, 0] == TARGET_CLASS).sum()
        if nh_cnt >= 5:
            for key in samples:
                if samples[key] is None and key in stem.lower():
                    samples[key] = (stem, rows)

    for source, item in samples.items():
        if item is None:
            print(f"\n  [No {source} sample found]")
            continue
        stem, rows = item
        nh_mask = rows[:, 0] == TARGET_CLASS
        nh = rows[nh_mask]
        print(f"\n--- {source.upper()} sample: {stem[-70:]} ---")
        print(f"  Total annotations: {len(rows)},  noteheads: {len(nh)}")
        print(f"  {'cx':>8} {'cy':>8} {'w_n':>8} {'h_n':>8} {'w_px':>8} {'h_px':>8}")
        for row in nh[:10]:
            print(f"  {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f} {row[4]:8.4f} "
                  f"  {row[3]*IMGSZ:7.1f} {row[4]*IMGSZ:7.1f}")
        if len(nh) > 10:
            print(f"  ... ({len(nh)-10} more rows)")


def section_c_small_notehead_analysis(val_data: dict):
    """Analyse only the small (genuine) noteheads separately."""
    print("\n" + "=" * 70)
    print("SECTION C — GENUINE SMALL NOTEHEAD ANALYSIS (w<50px, h<50px)")
    print("=" * 70)

    small_w = []; small_h = []
    large_w = []; large_h = []
    small_nn_dist = []
    small_per_image = {}

    for stem, rows in val_data.items():
        nh_mask = rows[:, 0] == TARGET_CLASS
        nh = rows[nh_mask]
        if len(nh) == 0:
            continue

        w_px = nh[:, 3] * IMGSZ
        h_px = nh[:, 4] * IMGSZ

        small_mask = (w_px < 50) & (h_px < 50)
        large_mask = ~small_mask

        small_w.extend(w_px[small_mask].tolist())
        small_h.extend(h_px[small_mask].tolist())
        large_w.extend(w_px[large_mask].tolist())
        large_h.extend(h_px[large_mask].tolist())

        cnt_small = small_mask.sum()
        cnt_large = large_mask.sum()
        small_per_image[stem] = (cnt_small, cnt_large, len(nh))

        # NN distance for small noteheads only
        if cnt_small >= 2:
            small_nh = nh[small_mask]
            cx = small_nh[:, 1] * IMGSZ
            cy = small_nh[:, 2] * IMGSZ
            for i in range(len(cx)):
                dx = cx - cx[i]; dy = cy - cy[i]
                dists = np.sqrt(dx**2 + dy**2)
                dists[i] = 1e9
                small_nn_dist.append(dists.min())

    small_w = np.array(small_w); small_h = np.array(small_h)
    large_w = np.array(large_w); large_h = np.array(large_h)

    print(f"\nSmall noteheads (w<50px, h<50px): {len(small_w):,}")
    if len(small_w) > 0:
        print(f"  width:  mean={small_w.mean():.1f}  median={np.median(small_w):.1f}  p5={np.percentile(small_w,5):.1f}  p95={np.percentile(small_w,95):.1f}")
        print(f"  height: mean={small_h.mean():.1f}  median={np.median(small_h):.1f}  p5={np.percentile(small_h,5):.1f}  p95={np.percentile(small_h,95):.1f}")

    print(f"\nLarge noteheads (w>=50px or h>=50px): {len(large_w):,}")
    if len(large_w) > 0:
        print(f"  width:  mean={large_w.mean():.1f}  median={np.median(large_w):.1f}  p5={np.percentile(large_w,5):.1f}  p95={np.percentile(large_w,95):.1f}")
        print(f"  height: mean={large_h.mean():.1f}  median={np.median(large_h):.1f}  p5={np.percentile(large_h,5):.1f}  p95={np.percentile(large_h,95):.1f}")

    # Which images have ONLY large noteheads?
    only_large = {s: v for s, v in small_per_image.items() if v[1] > 0 and v[0] == 0}
    mixed      = {s: v for s, v in small_per_image.items() if v[0] > 0 and v[1] > 0}
    only_small = {s: v for s, v in small_per_image.items() if v[0] > 0 and v[1] == 0}

    print(f"\nImages with noteheads only:")
    print(f"  Only small noteheads (w<50): {len(only_small)}")
    print(f"  Only large noteheads (w>=50): {len(only_large)}")
    print(f"  Mixed (both small and large): {len(mixed)}")

    # Example of mixed — is one type from oversample?
    if mixed:
        sample_mixed = list(mixed.items())[:3]
        print(f"\nSample mixed images:")
        for stem, (cs, cl, total) in sample_mixed:
            print(f"  {stem[-65:]}: {cs} small, {cl} large, {total} total")

    # For pure SMALL noteheads: NMS risk
    if small_nn_dist:
        d_arr = np.array(small_nn_dist)
        w_med_small = np.median(small_w) if len(small_w) > 0 else 20
        print(f"\nFor SMALL noteheads (w<50px):")
        print(f"  Median width: {w_med_small:.1f}px")
        print(f"  Nearest-neighbour centre distances:")
        for p in [5, 10, 25, 50, 75, 90]:
            d = np.percentile(d_arr, p)
            print(f"    p{p:2d}: {d:.1f}px  ({d/w_med_small:.2f}× width)")

        # IoU=0.7 suppression threshold for small noteheads
        # For horizontal shift with w=w_med, h=h_med:
        h_med_small = np.median(small_h) if len(small_h) > 0 else 20
        # analytically: IoU = max(0, w-d)*max(0, h-dy) / (2wh - max(0,w-d)*max(0,h-dy))
        # worst case: dy=0 → IoU ≥ 0.7 when d < w*(1-0.7/(2-0.7)) = w*(1-0.538) = w*0.462?
        # Let me recalc:  IoU = (w-d)/w * h / (2h - (w-d)/w * h) ← assuming h overlap = h
        # = (1-d/w) / (2-(1-d/w)) = x/(2-x) where x=1-d/w
        # 0.7 = x/(2-x) → 0.7(2-x)=x → 1.4=1.7x → x=0.824 → d=w*(1-0.824)=0.176w
        iou07_dist = w_med_small * 0.176
        p_at_risk = (d_arr < iou07_dist).mean() * 100
        print(f"\n  For IoU>=0.7 suppression: need NN dist < {iou07_dist:.1f}px")
        print(f"  Fraction of small noteheads at risk: {p_at_risk:.1f}%")

    return small_w, small_h, large_w, large_h


def section_d_oversample_analysis(val_data: dict):
    """
    Check if large-box noteheads come from oversampled/augmented images.
    Also check if the SAME box is duplicated across oversampled versions.
    """
    print("\n" + "=" * 70)
    print("SECTION D — OVERSAMPLE ARTIFACT ANALYSIS")
    print("=" * 70)

    # Images with 'oversample' in name
    oversample_images = {s: rows for s, rows in val_data.items() if 'oversample' in s}
    original_images   = {s: rows for s, rows in val_data.items() if 'oversample' not in s}

    print(f"\nImages with 'oversample' in name: {len(oversample_images)}")
    print(f"Images without 'oversample': {len(original_images)}")

    def notehead_stats(img_dict):
        ws = []; hs = []; counts = []
        for stem, rows in img_dict.items():
            nh_mask = rows[:, 0] == TARGET_CLASS
            nh = rows[nh_mask]
            if len(nh) == 0:
                continue
            ws.extend((nh[:, 3] * IMGSZ).tolist())
            hs.extend((nh[:, 4] * IMGSZ).tolist())
            counts.append(len(nh))
        return np.array(ws), np.array(hs), np.array(counts)

    w_o, h_o, c_o = notehead_stats(oversample_images)
    w_r, h_r, c_r = notehead_stats(original_images)

    print(f"\nNotehead widths — OVERSAMPLED images:")
    if len(w_o) > 0:
        for p in [5, 25, 50, 75, 95]:
            print(f"  p{p:2d}: {np.percentile(w_o, p):.1f}px")
        print(f"  Large (>=50px): {(w_o>=50).mean()*100:.1f}%")

    print(f"\nNotehead widths — ORIGINAL images:")
    if len(w_r) > 0:
        for p in [5, 25, 50, 75, 95]:
            print(f"  p{p:2d}: {np.percentile(w_r, p):.1f}px")
        print(f"  Large (>=50px): {(w_r>=50).mean()*100:.1f}%")

    # Look at the top-dense beethoven image in detail
    beethoven_samples = [(s, rows) for s, rows in val_data.items() if 'beethoven' in s and 'oversample' in s]
    if beethoven_samples:
        stem, rows = beethoven_samples[0]
        print(f"\nDetailed inspection: {stem[-70:]}")
        nh_mask = rows[:, 0] == TARGET_CLASS
        nh = rows[nh_mask]
        w_px = nh[:, 3] * IMGSZ
        h_px = nh[:, 4] * IMGSZ
        print(f"  Noteheads: {len(nh)}")
        print(f"  Width range: {w_px.min():.1f} - {w_px.max():.1f}px")
        print(f"  Height range: {h_px.min():.1f} - {h_px.max():.1f}px")
        print(f"  Small (w<50): {(w_px<50).sum()}")
        print(f"  Large (w>=50): {(w_px>=50).sum()}")

        # Print first 5 large and 5 small
        large_mask = w_px >= 50
        small_mask = ~large_mask
        print(f"\n  First 5 LARGE noteheads:")
        print(f"    {'cx':>8} {'cy':>8} {'w_px':>8} {'h_px':>8}")
        for row, w, h in zip(nh[large_mask][:5], w_px[large_mask][:5], h_px[large_mask][:5]):
            print(f"    {row[1]:8.4f} {row[2]:8.4f} {w:8.1f} {h:8.1f}")
        print(f"\n  First 5 SMALL noteheads:")
        print(f"    {'cx':>8} {'cy':>8} {'w_px':>8} {'h_px':>8}")
        if small_mask.sum() > 0:
            for row, w, h in zip(nh[small_mask][:5], w_px[small_mask][:5], h_px[small_mask][:5]):
                print(f"    {row[1]:8.4f} {row[2]:8.4f} {w:8.1f} {h:8.1f}")
        else:
            print(f"    [none]")

    return w_o, h_o, w_r, h_r


def section_e_nms_with_real_sizes(val_data: dict):
    """
    Re-run NMS simulation using ONLY the large boxes (w>=50px)
    to understand the actual suppression risk from ground truth.
    """
    print("\n" + "=" * 70)
    print("SECTION E — NMS SIMULATION WITH LARGE BOXES ONLY")
    print("=" * 70)
    print("(The large boxes are what the model tries to predict AND what evaluator uses)")
    print("(iou=0.7 NMS on predictions that resemble these large GT boxes)")

    def iou_matrix(boxes):
        N = len(boxes)
        if N < 2:
            return np.zeros((N, N))
        x1=boxes[:,0]; y1=boxes[:,1]; x2=boxes[:,2]; y2=boxes[:,3]
        areas = (x2-x1)*(y2-y1)
        ix1 = np.maximum(x1[:,None], x1[None,:])
        iy1 = np.maximum(y1[:,None], y1[None,:])
        ix2 = np.minimum(x2[:,None], x2[None,:])
        iy2 = np.minimum(y2[:,None], y2[None,:])
        iw = np.maximum(0, ix2-ix1)
        ih = np.maximum(0, iy2-iy1)
        inter = iw*ih
        union = areas[:,None] + areas[None,:] - inter
        iou = np.where(union>0, inter/union, 0.0)
        np.fill_diagonal(iou, 0)
        return iou

    # Find images with large noteheads, sample top 50 by count
    large_images = []
    for stem, rows in val_data.items():
        nh_mask = rows[:, 0] == TARGET_CLASS
        nh = rows[nh_mask]
        if len(nh) == 0:
            continue
        w_px = nh[:, 3] * IMGSZ
        large = nh[w_px >= 50]
        if len(large) >= 5:
            large_images.append((stem, large))
    large_images.sort(key=lambda x: -len(x[1]))
    large_images = large_images[:100]

    print(f"\nImages with >= 5 large noteheads: {len(large_images)}")

    suppressed_per_image = []
    iou_vals = []

    for stem, large_nh in large_images:
        cx = large_nh[:, 1] * IMGSZ
        cy = large_nh[:, 2] * IMGSZ
        w  = large_nh[:, 3] * IMGSZ
        h  = large_nh[:, 4] * IMGSZ
        x1 = cx - w/2; x2 = cx + w/2
        y1 = cy - h/2; y2 = cy + h/2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        iou_mat = iou_matrix(boxes_xyxy)
        max_iou = iou_mat.max(axis=1)
        iou_vals.extend(max_iou.tolist())

        # Greedy NMS simulation
        areas = (x2-x1)*(y2-y1)
        order = np.argsort(-areas)
        suppressed_set = set()
        suppressed = 0
        for i in order:
            if i in suppressed_set:
                suppressed += 1
                continue
            for j in order:
                if j != i and j not in suppressed_set and iou_mat[i,j] >= 0.7:
                    suppressed_set.add(j)
        suppressed_per_image.append(suppressed)

    iou_arr = np.array(iou_vals)
    supp_arr = np.array(suppressed_per_image)

    print(f"\nGT large-box max-IoU distribution (these are the boxes model must detect):")
    for t in [0.3, 0.5, 0.65, 0.7, 0.75, 0.8, 0.9]:
        pct = (iou_arr >= t).mean() * 100
        print(f"  IoU >= {t:.2f}: {pct:.1f}% of large noteheads")

    print(f"\nNMS suppression (iou=0.7) among large GT boxes:")
    print(f"  Images with suppression >= 1: {(supp_arr>=1).sum()}/{len(supp_arr)} ({100*(supp_arr>=1).mean():.1f}%)")
    print(f"  Mean suppressed per image: {supp_arr.mean():.1f}")
    print(f"  Max suppressed per image:  {supp_arr.max()}")

    print(f"""
  INTERPRETATION:
  If the GT boxes themselves would be suppressed by NMS at iou=0.7,
  then it is IMPOSSIBLE for the model to achieve high recall at that
  IoU threshold, because:
  1. GT box A and GT box B both exist in the dataset.
  2. Model predicts P_A matching GT box A.
  3. NMS removes P_A because it overlaps with P_B at IoU >= 0.7.
  4. GT box A remains unmatched → counted as False Negative.
  5. This gives recall credit for neither A nor B.

  This is the MAXIMUM ACHIEVABLE RECALL under NMS iou=0.7 with
  overlapping GT boxes.
""")


def section_f_what_are_large_boxes(val_data: dict):
    """
    Try to understand: what DO the large notehead boxes represent?
    Check if they are note clusters, or if the image is very low-res
    causing a zoom effect.
    Also look at the image actual dimensions vs expected 1280px.
    """
    print("\n" + "=" * 70)
    print("SECTION F — WHAT ARE THE LARGE NOTEHEAD BOXES?")
    print("=" * 70)

    # Check actual image dimensions
    import subprocess
    img_files = list(VAL_IMAGES.glob("*.png"))[:5]
    print("\nSample image dimensions (first 5 .png files):")
    for f in img_files:
        result = subprocess.run(
            ["python3", "-c",
             f"from PIL import Image; img=Image.open('{f}'); print(img.size)"],
            capture_output=True, text=True
        )
        print(f"  {f.name[-60:]}: {result.stdout.strip()}")

    # Take a specific large-box image and inspect ALL its labels
    # to understand context
    beethoven_dense = None
    for stem, rows in val_data.items():
        if 'beethoven' in stem and 'oversample' in stem:
            nh = rows[rows[:, 0] == TARGET_CLASS]
            if len(nh) >= 100:
                beethoven_dense = (stem, rows)
                break

    if beethoven_dense:
        stem, rows = beethoven_dense
        print(f"\nAll annotations for: {stem[-70:]}")
        print(f"Total rows: {len(rows)}")

        # Group by class
        class_counts = defaultdict(int)
        class_sample = defaultdict(list)
        for row in rows:
            cls = int(row[0])
            class_counts[cls] += 1
            if len(class_sample[cls]) < 3:
                class_sample[cls].append(row)

        print(f"\n{'Class':<28} {'N':>5}  sample_w_px  sample_h_px")
        for cls in sorted(class_counts.keys()):
            name = CLASS_NAMES.get(cls, f"cls_{cls}")
            n = class_counts[cls]
            samples = class_sample[cls]
            sizes = [f"{r[3]*IMGSZ:.0f}×{r[4]*IMGSZ:.0f}" for r in samples]
            print(f"  {name:<26} {n:>5}  {', '.join(sizes)}")

    # Find a muscima image with noteheads (should have correct small-box labels)
    muscima_sample = None
    for stem, rows in val_data.items():
        if 'muscima' in stem:
            nh = rows[rows[:, 0] == TARGET_CLASS]
            if len(nh) >= 10:
                muscima_sample = (stem, rows)
                break

    if muscima_sample:
        stem, rows = muscima_sample
        print(f"\nMUSCIMA sample: {stem[-70:]}")
        nh = rows[rows[:, 0] == TARGET_CLASS]
        w_px = nh[:, 3] * IMGSZ
        h_px = nh[:, 4] * IMGSZ
        print(f"  Noteheads: {len(nh)}")
        print(f"  Width range: {w_px.min():.1f} - {w_px.max():.1f}px")
        print(f"  Height range: {h_px.min():.1f} - {h_px.max():.1f}px")
        print(f"  First 5 boxes (cx,cy,w_px,h_px):")
        for row in nh[:5]:
            print(f"    {row[1]:.4f} {row[2]:.4f}  {row[3]*IMGSZ:.1f}px × {row[4]*IMGSZ:.1f}px")


def section_g_theoretical_max_recall(val_data: dict):
    """
    Compute the theoretical maximum recall achievable given overlapping GT boxes
    and NMS at iou=0.7.
    Key insight: if GT box A and GT box B overlap at IoU >= 0.7, then ANY
    prediction matching A will also match B (and vice versa), so NMS must
    remove one. The remaining matched prediction can only cover one GT box.
    """
    print("\n" + "=" * 70)
    print("SECTION G — THEORETICAL MAX RECALL UNDER NMS iou=0.7")
    print("=" * 70)

    def iou_single(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter/union if union > 0 else 0

    total_gt = 0
    total_matchable = 0  # GT boxes that could be matched without NMS conflict
    images_with_nh = [(s, rows) for s, rows in val_data.items()
                      if (rows[:, 0] == TARGET_CLASS).sum() > 0]

    # For each image: run greedy NMS on GT boxes.
    # The number of "keep" boxes = theoretical max predictions we can have.
    # Recall = keep / total_gt  (upper bound)
    supp_counts = []
    for stem, rows in images_with_nh:
        nh_mask = rows[:, 0] == TARGET_CLASS
        nh = rows[nh_mask]
        n = len(nh)
        total_gt += n

        if n == 0:
            continue

        cx = nh[:, 1] * IMGSZ; cy = nh[:, 2] * IMGSZ
        w  = nh[:, 3] * IMGSZ; h  = nh[:, 4] * IMGSZ
        x1 = cx - w/2; x2 = cx + w/2
        y1 = cy - h/2; y2 = cy + h/2

        # Build IoU matrix
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        N = len(boxes)
        iou_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                iou = iou_single(boxes[i], boxes[j])
                iou_mat[i, j] = iou_mat[j, i] = iou

        # Greedy NMS: suppress all boxes with IoU >= 0.7 with a kept box
        areas = w * h
        order = np.argsort(-areas)
        suppressed = set()
        n_keep = 0
        for idx in order:
            if idx in suppressed:
                continue
            n_keep += 1
            for jdx in range(N):
                if jdx != idx and jdx not in suppressed and iou_mat[idx, jdx] >= 0.7:
                    suppressed.add(jdx)

        total_matchable += n_keep
        supp_counts.append(n - n_keep)

    max_recall = total_matchable / total_gt if total_gt > 0 else 0

    supp_arr = np.array(supp_counts)
    print(f"\nTotal GT notehead_filled (val): {total_gt:,}")
    print(f"GT boxes that CAN be matched (after NMS deconflict): {total_matchable:,}")
    print(f"GT boxes that CANNOT be matched (suppression victims): {total_gt - total_matchable:,}")
    print(f"\n{'='*50}")
    print(f"THEORETICAL MAXIMUM RECALL at NMS iou=0.7: {max_recall:.4f}  ({max_recall*100:.2f}%)")
    print(f"{'='*50}")
    print(f"\nActual observed Recall: 0.513")
    print(f"Gap from theoretical max: {max_recall - 0.513:.4f}  ({(max_recall-0.513)*100:.2f}pp)")

    print(f"\nPer-image GT suppression:")
    for thresh in [0, 1, 5, 10, 20, 50]:
        cnt = (supp_arr >= thresh).sum()
        print(f"  Images with >= {thresh:3d} suppressed GT: {cnt:4d} ({100*cnt/len(supp_arr):.1f}%)")

    print(f"  Mean suppressed per image: {supp_arr.mean():.2f}")
    print(f"  Max suppressed per image:  {supp_arr.max()}")

    return max_recall, total_gt, total_matchable


def main():
    print("=" * 70)
    print("DEEP DIVE: notehead_filled RECALL ROOT CAUSE")
    print("=" * 70)

    print("\nLoading val labels...")
    val_data = load_labels(VAL_LABELS)
    print(f"Loaded {len(val_data)} files")

    all_w, all_h, source_by_size = section_a_bimodal_analysis(val_data)
    section_b_inspect_actual_labels(val_data)
    small_w, small_h, large_w, large_h = section_c_small_notehead_analysis(val_data)
    section_d_oversample_analysis(val_data)
    section_e_nms_with_real_sizes(val_data)
    section_f_what_are_large_boxes(val_data)
    max_recall, total_gt, total_matchable = section_g_theoretical_max_recall(val_data)

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print(f"""
Key numbers:
  Total GT noteheads (val): {total_gt:,}
  Theoretical max recall (NMS iou=0.7): {max_recall:.4f}
  Observed recall:                        0.5130
  Gap:                                    {max_recall - 0.513:.4f}

If max_recall << 1.0:
  → The PRIMARY cause of low recall is GT box overlap exceeding NMS threshold.
  → This is a DATA LABELLING or EVALUATION PROTOCOL problem, not a model problem.
  → The model CANNOT achieve higher recall at iou=0.7 if GT boxes overlap ≥ 0.7.
  → FIX: Lower NMS iou threshold to 0.5 or use non-maximum-suppression-free eval.

If max_recall ≈ 1.0 but observed recall = 0.513:
  → The model is genuinely missing ~49% of noteheads.
  → Fix: training adjustments (cls weight, focal, data augmentation).
""")


if __name__ == "__main__":
    main()
