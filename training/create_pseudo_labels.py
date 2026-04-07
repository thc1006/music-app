"""
Phase 5 Teacher Pseudo-Label Pipeline

Uses Phase 5 model (trained on glyph-group bboxes) to predict on OpenScore images,
then shrinks the large predictions to TAL-learnable sizes for retraining.

Key design decisions (from Phase 6/7 failure analysis):
  - Box size: 3-8× DoReMi reference, NOT as tiny as Phase 6 (0.012 → TAL failure)
  - Box jitter: ±30% random variation (Phase 6 had all 63K boxes identical)
  - Notehead position: top-biased within glyph-group (validated in Phase 6)
  - Boundary clamping: all boxes clipped to [0, 1]

Usage:
    python create_pseudo_labels.py                    # full pipeline
    python create_pseudo_labels.py --predict-only     # only generate predictions
    python create_pseudo_labels.py --assemble-only    # only assemble dataset
"""
import random
import os
import shutil
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
TRAINING_DIR = PROJECT_ROOT / "training"

PHASE5_WEIGHTS = PROJECT_ROOT / "runs/detect/harmony_omr_v2_phase5/nostem_v1_stage2/weights/best.pt"
PHASE6_DATASET = TRAINING_DIR / "datasets/yolo_harmony_v2_phase6_fixed"
PSEUDO_DATASET = TRAINING_DIR / "datasets/yolo_harmony_v2_pseudo_v1"

NUM_CLASSES = 32
NAMES = [
    'notehead_filled', 'notehead_hollow', 'beam', 'flag_8th', 'flag_16th',
    'flag_32nd', 'augmentation_dot', 'tie', 'clef_treble', 'clef_bass',
    'clef_alto', 'clef_tenor', 'accidental_sharp', 'accidental_flat',
    'accidental_natural', 'accidental_double_sharp', 'accidental_double_flat',
    'rest_whole', 'rest_half', 'rest_quarter', 'rest_8th', 'rest_16th',
    'barline', 'barline_double', 'barline_final', 'barline_repeat',
    'time_signature', 'key_signature', 'dynamic_loud', 'dynamic_soft',
    'fermata', 'ledger_line',
]

# Target pseudo-label sizes: (target_w, target_h) in normalized coords.
# Derived from DoReMi reference × scale factor, ensuring >NOTEHEAD_MIN and <NOTEHEAD_MAX.
# For classes with reliable DoReMi reference: use 4-6× DoReMi size.
# For classes with unreliable reference: use median of reliable classes or class-specific estimate.
#
# DoReMi reference → target (with scale):
#   notehead: 0.011×0.007 → ~0.05×0.04  (4.5×, 5.7×)
#   beam:     0.036×0.007 → ~0.07×0.03  (1.9×, 4.3×) — wide+thin
#   barline:  0.015×0.085 → ~0.03×0.10  (2.0×, 1.2×) — narrow+tall
#   clef:     0.023×0.042 → ~0.05×0.08  (2.2×, 1.9×)
TARGET_SIZES = {
    0:  (0.050, 0.040),   # notehead_filled
    1:  (0.050, 0.040),   # notehead_hollow
    2:  (0.070, 0.030),   # beam (wide, thin)
    3:  (0.035, 0.055),   # flag_8th (narrow, tall)
    4:  (0.035, 0.055),   # flag_16th
    5:  (0.035, 0.060),   # flag_32nd
    6:  (0.025, 0.025),   # augmentation_dot (small, square)
    7:  (0.070, 0.035),   # tie (wide arc)
    8:  (0.050, 0.080),   # clef_treble
    9:  (0.050, 0.065),   # clef_bass
    10: (0.050, 0.080),   # clef_alto (unreliable ref, use treble as proxy)
    11: (0.050, 0.080),   # clef_tenor (unreliable ref, use treble as proxy)
    12: (0.035, 0.050),   # accidental_sharp
    13: (0.030, 0.045),   # accidental_flat
    14: (0.025, 0.050),   # accidental_natural
    15: (0.035, 0.035),   # accidental_double_sharp (unreliable ref, use sharp as proxy)
    16: (0.035, 0.050),   # accidental_double_flat (unreliable ref)
    17: (0.040, 0.025),   # rest_whole
    18: (0.040, 0.030),   # rest_half
    19: (0.035, 0.055),   # rest_quarter
    20: (0.035, 0.040),   # rest_8th
    21: (0.040, 0.050),   # rest_16th
    22: (0.030, 0.100),   # barline (narrow, tall)
    23: (0.030, 0.100),   # barline_double (unreliable ref)
    24: (0.030, 0.100),   # barline_final (unreliable ref)
    25: (0.040, 0.110),   # barline_repeat
    26: (0.040, 0.040),   # time_signature
    27: (0.040, 0.070),   # key_signature
    28: (0.045, 0.045),   # dynamic_loud
    29: (0.040, 0.035),   # dynamic_soft
    30: (0.045, 0.060),   # fermata (unreliable ref)
    31: (0.040, 0.025),   # ledger_line (wide, thin)
}

# Position bias within glyph-group prediction box.
# 'top': notehead sits at top edge of glyph-group (validated Phase 6)
# 'center': default, keep at prediction center
# 'bottom': symbol at bottom of glyph-group
POSITION_BIAS = {
    0: 'top',      # notehead_filled — validated in Phase 6
    1: 'top',      # notehead_hollow — validated in Phase 6
    6: 'center',   # augmentation_dot
    31: 'center',  # ledger_line
}
# All other classes default to 'center'

JITTER_RANGE = 0.30  # ±30% random variation on target size


# ──────────────────────────────────────────────────────────────────────
# Core: shrink_box()
# ──────────────────────────────────────────────────────────────────────

def shrink_box(cls_id: int, cx: float, cy: float, w: float, h: float) -> tuple:
    """Shrink a glyph-group prediction box to TAL-learnable pseudo-label size.

    Args:
        cls_id: YOLO class index (0-31)
        cx, cy, w, h: normalized prediction box (center-x, center-y, width, height)

    Returns:
        (new_cx, new_cy, new_w, new_h) — normalized, clamped to [0, 1]
    """
    target_w, target_h = TARGET_SIZES.get(cls_id, (0.045, 0.045))

    # Apply jitter: uniform ±JITTER_RANGE
    jitter_w = 1.0 + random.uniform(-JITTER_RANGE, JITTER_RANGE)
    jitter_h = 1.0 + random.uniform(-JITTER_RANGE, JITTER_RANGE)
    new_w = target_w * jitter_w
    new_h = target_h * jitter_h

    # Position bias
    bias = POSITION_BIAS.get(cls_id, 'center')
    if bias == 'top':
        # Move center toward top of prediction box (lower cy in YOLO coords)
        # Place new center at top quarter of original box
        new_cy = cy - h * 0.25
        new_cx = cx
    elif bias == 'bottom':
        new_cy = cy + h * 0.25
        new_cx = cx
    else:
        new_cx = cx
        new_cy = cy

    # Clamp box to stay fully inside [0, 1]
    half_w = new_w / 2
    half_h = new_h / 2
    new_cx = max(half_w, min(1.0 - half_w, new_cx))
    new_cy = max(half_h, min(1.0 - half_h, new_cy))

    return (new_cx, new_cy, new_w, new_h)


# ──────────────────────────────────────────────────────────────────────
# Pipeline: generate_pseudo_labels()
# ──────────────────────────────────────────────────────────────────────

def generate_pseudo_labels_v2(
    phase5_weights: str = str(PHASE5_WEIGHTS),
    source_dataset: str = str(PHASE6_DATASET),
    output_dataset: str = str(PSEUDO_DATASET),
    conf: float = 0.10,
    iou: float = 0.55,
    imgsz: int = 1280,
    max_det: int = 1500,
    seed: int = 42,
    use_cv_noteheads: bool = True,
):
    """V2 pipeline: CV noteheads + Phase 5 other classes.

    Key improvement over v1:
      - Noteheads: detected by CV (notehead_detector_cv) — precise position,
        no stem-direction dependency, no rule-based guessing
      - Other classes: Phase 5 predictions + shrink_box (same as v1)

    Args:
        use_cv_noteheads: if True, use CV detector for noteheads (cls 0, 1).
            Phase 5 predictions for noteheads are discarded.
    """
    from ultralytics import YOLO
    from notehead_detector_cv import detect_noteheads

    random.seed(seed)
    src = Path(source_dataset)
    out = Path(output_dataset)

    print(f"Phase 5 weights: {phase5_weights}")
    print(f"Source dataset:   {source_dataset}")
    print(f"Output dataset:   {output_dataset}")
    print(f"Config: conf={conf}, iou={iou}, imgsz={imgsz}, max_det={max_det}")
    print(f"CV noteheads: {use_cv_noteheads}")

    model = YOLO(phase5_weights)

    if out.exists():
        shutil.rmtree(out)

    stats = defaultdict(int)

    for split in ["train", "val"]:
        img_src = src / split / "images"
        lbl_src = src / split / "labels"
        img_dst = out / split / "images"
        lbl_dst = out / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        all_images = sorted(img_src.glob("*.png"))
        print(f"\n{split}: {len(all_images)} images")

        for img_path in all_images:
            stem = img_path.stem
            lbl_path = lbl_src / f"{stem}.txt"

            # Symlink image
            dst_img = img_dst / img_path.name
            if not dst_img.exists():
                real_img = img_path.resolve()
                dst_img.symlink_to(real_img)

            is_openscore = "lg-" in stem

            if not is_openscore:
                # DoReMi: copy labels unchanged
                dst_lbl = lbl_dst / f"{stem}.txt"
                if lbl_path.exists():
                    shutil.copy2(str(lbl_path), str(dst_lbl))
                else:
                    dst_lbl.write_text("")
                stats[f"{split}_doremi"] += 1
            else:
                lines = []

                # === Noteheads: CV detector (precise position) ===
                if use_cv_noteheads:
                    cv_dets = detect_noteheads(str(img_path))
                    for cls, cx, cy, w, h in cv_dets:
                        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    stats[f"{split}_cv_noteheads"] += len(cv_dets)

                # === Other classes: Phase 5 predictions + shrink ===
                results = model.predict(
                    str(img_path), imgsz=imgsz, conf=conf,
                    iou=iou, max_det=max_det, verbose=False,
                )
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])

                    # Skip noteheads from Phase 5 if using CV
                    if use_cv_noteheads and cls in (0, 1):
                        continue

                    xywhn = boxes.xywhn[i]
                    pred_cx = float(xywhn[0])
                    pred_cy = float(xywhn[1])
                    pred_w = float(xywhn[2])
                    pred_h = float(xywhn[3])

                    new_cx, new_cy, new_w, new_h = shrink_box(
                        cls, pred_cx, pred_cy, pred_w, pred_h
                    )
                    lines.append(
                        f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                    )
                    stats[f"{split}_phase5_other"] += 1

                dst_lbl = lbl_dst / f"{stem}.txt"
                dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
                stats[f"{split}_openscore"] += 1

        print(f"  DoReMi: {stats[f'{split}_doremi']}")
        print(f"  OpenScore: {stats[f'{split}_openscore']}")
        print(f"    CV noteheads: {stats[f'{split}_cv_noteheads']}")
        print(f"    Phase 5 other: {stats[f'{split}_phase5_other']}")

    # Write YAML config
    yaml_content = f"""# Pseudo-label dataset v2
# DoReMi labels: original (unchanged)
# OpenScore noteheads: CV detector (notehead_detector_cv.py)
# OpenScore other classes: Phase 5 predictions, shrunk

path: {out.resolve()}
train: train/images
val: val/images

names:
"""
    for i, name in enumerate(NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = out / "harmony_pseudo_v1.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nYAML: {yaml_path}")
    print("Done.")


# Keep v1 for backwards compatibility
def generate_pseudo_labels(**kwargs):
    """V1 pipeline (deprecated). Use generate_pseudo_labels_v2 instead."""
    return generate_pseudo_labels_v2(use_cv_noteheads=False, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate pseudo-label dataset v2")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cv", action="store_true", help="Disable CV noteheads (v1 mode)")
    args = parser.parse_args()

    generate_pseudo_labels_v2(
        conf=args.conf, iou=args.iou, seed=args.seed,
        use_cv_noteheads=not args.no_cv,
    )
