"""
Pitch Accuracy Harness — Phase C foundation.

Measures how accurate our Phase 9 + downstream pitch estimation is on real
sheet music images, by comparing against a small hand-curated ground truth
of (image, notehead position, expected MIDI) tuples.

Workflow:
  1. Use save_notehead_crops() to dump small thumbnails of detected noteheads
  2. Visually inspect the small crops (no large-image dimension issues)
  3. Add expected MIDI to ground_truth.json by hand
  4. Run measure_pitch_accuracy() to get a score
  5. Iterate on pitch_estimator until accuracy ≥80%

Ground truth file: training/pitch_ground_truth.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GROUND_TRUTH_PATH = Path(__file__).parent / "pitch_ground_truth.json"
CROP_OUTPUT_DIR = Path("/tmp/pitch_crops")


# ──────────────────────────────────────────────────────────────────────
# Ground truth I/O
# ──────────────────────────────────────────────────────────────────────

def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> dict[str, list[dict]]:
    """Load ground truth from JSON file. Returns empty dict if missing.

    Keys starting with `_` are treated as metadata/comments and filtered out.

    Format:
        {
            "image_basename.png": [
                {"idx": 0, "cx": 100, "cy": 100, "midi": 60, "letter": "C4"},
                ...
            ]
        }
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        # Filter out metadata keys (starting with _)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (json.JSONDecodeError, OSError):
        return {}


def save_ground_truth(gt: dict[str, list[dict]], path: Path = GROUND_TRUTH_PATH):
    """Save ground truth dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────
# Crop generation (small images, no dimension limit issues)
# ──────────────────────────────────────────────────────────────────────

def save_notehead_crops(
    image_path: str,
    output_dir: Path = CROP_OUTPUT_DIR,
    crop_size: int = 80,
    max_crops: int = 30,
) -> list[dict]:
    """Save small crops centered on each Phase 9-detected notehead.

    Each crop is `crop_size` × `crop_size` pixels — small enough to never
    hit dimension limits in Read/image-loading tools.

    Returns list of {idx, cx, cy, w, h, crop_path}.
    """
    from downstream_eval import run_phase9_detection

    output_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    H, W = img.shape[:2]

    detections = run_phase9_detection(image_path)
    noteheads = [d for d in detections if d.get("class_id") in (0, 1)]
    noteheads.sort(key=lambda n: (n["cy"], n["cx"]))

    image_stem = Path(image_path).stem
    half = crop_size // 2

    results = []
    for i, nh in enumerate(noteheads[:max_crops]):
        cx, cy = int(nh["cx"]), int(nh["cy"])
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(W, cx + half)
        y2 = min(H, cy + half)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Mark the center with a small red cross for clarity
        h_crop, w_crop = crop.shape[:2]
        center_x, center_y = w_crop // 2, h_crop // 2
        crop_marked = crop.copy()
        cv2.line(crop_marked, (center_x - 5, center_y), (center_x + 5, center_y), (0, 0, 255), 1)
        cv2.line(crop_marked, (center_x, center_y - 5), (center_x, center_y + 5), (0, 0, 255), 1)

        crop_path = output_dir / f"{image_stem}_nh{i:03d}.png"
        cv2.imwrite(str(crop_path), crop_marked)

        results.append({
            "idx": i,
            "cx": float(nh["cx"]),
            "cy": float(nh["cy"]),
            "w": float(nh["w"]),
            "h": float(nh["h"]),
            "crop_path": str(crop_path),
            "class_id": nh.get("class_id", 0),
        })

    return results


# ──────────────────────────────────────────────────────────────────────
# Accuracy measurement
# ──────────────────────────────────────────────────────────────────────

def _match_predictions_to_gt(
    predicted: list[dict],
    ground_truth: list[dict],
    max_match_dist: float = 30.0,
) -> list[tuple[dict, dict | None]]:
    """For each ground truth entry, find the nearest prediction within max_match_dist.

    Returns list of (gt_entry, matched_pred_or_None) pairs.
    """
    matches = []
    for gt in ground_truth:
        best = None
        best_dist = float("inf")
        for pred in predicted:
            dx = pred["cx"] - gt["cx"]
            dy = pred["cy"] - gt["cy"]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = pred
        if best is not None and best_dist <= max_match_dist:
            matches.append((gt, best))
        else:
            matches.append((gt, None))
    return matches


def measure_pitch_accuracy(
    predicted: list[dict],
    ground_truth: list[dict],
    tolerance_semitones: int = 0,
    max_match_dist: float = 30.0,
) -> dict[str, Any]:
    """Compare predicted pitches to ground truth.

    Args:
        predicted: list of {cx, cy, midi}
        ground_truth: list of {cx, cy, midi}
        tolerance_semitones: max acceptable error for "lenient" accuracy
        max_match_dist: max pixel distance to consider a position match

    Returns:
        {
            "exact_accuracy": float,    # exact MIDI match
            "lenient_accuracy": float,  # within tolerance_semitones
            "matched_count": int,
            "total_gt": int,
            "errors": list[dict],
        }
    """
    matches = _match_predictions_to_gt(predicted, ground_truth, max_match_dist)
    total = len(ground_truth)
    if total == 0:
        return {
            "exact_accuracy": 0.0,
            "lenient_accuracy": 0.0,
            "matched_count": 0,
            "total_gt": 0,
            "errors": [],
        }

    exact = 0
    lenient = 0
    errors = []
    for gt, pred in matches:
        if pred is None:
            errors.append({"gt": gt, "pred": None, "delta": None})
            continue
        delta = pred["midi"] - gt["midi"]
        if delta == 0:
            exact += 1
            lenient += 1
        elif abs(delta) <= tolerance_semitones:
            lenient += 1
            errors.append({"gt": gt, "pred": pred, "delta": delta})
        else:
            errors.append({"gt": gt, "pred": pred, "delta": delta})

    return {
        "exact_accuracy": exact / total,
        "lenient_accuracy": lenient / total,
        "matched_count": exact,
        "total_gt": total,
        "errors": errors,
    }


# ──────────────────────────────────────────────────────────────────────
# End-to-end baseline runner
# ──────────────────────────────────────────────────────────────────────

def run_baseline(image_path: str) -> dict[str, Any]:
    """Run full pipeline on an image and compare to ground truth (if available).

    Returns:
        {
            "predictions": list of {cx, cy, midi},
            "ground_truth_count": int,
            "accuracy": dict (only if ground truth exists for this image),
        }
    """
    from downstream_eval import run_phase9_detection, build_chord_snapshots

    image_basename = Path(image_path).name
    detections = run_phase9_detection(image_path)
    chords = build_chord_snapshots(detections, image_path=image_path)

    # Extract (cx, cy, midi) from chord snapshots
    predictions: list[dict] = []
    # Build map from voice notehead → original detection cx/cy
    # (chord snapshot only has midi + voice; we need to back-trace position)
    # Easier: re-run pitch estimation directly on noteheads with staves
    from staff_detector import detect_staves
    from clef_detector import assign_clefs_to_staves
    from pitch_estimator import estimate_pitch

    staves = detect_staves(image_path)
    if staves:
        # C2: per-stave clef detection
        clef_classes = {8, 9, 10, 11}
        clefs = [d for d in detections if d.get("class_id") in clef_classes]
        staff_clefs = assign_clefs_to_staves(clefs, staves)

        for d in detections:
            if d.get("class_id") not in (0, 1):
                continue
            # Find nearest staff
            nh_cy = d["cy"]
            staff_idx = min(
                range(len(staves)),
                key=lambda i: abs(nh_cy - (staves[i].line_ys[0] + staves[i].line_ys[-1]) / 2),
            )
            staff = staves[staff_idx]
            clef = staff_clefs[staff_idx]
            midi = estimate_pitch(nh_cy, staff, clef)
            predictions.append({
                "cx": d["cx"],
                "cy": d["cy"],
                "midi": midi,
            })

    gt_all = load_ground_truth()
    gt_for_image = gt_all.get(image_basename, [])

    result: dict[str, Any] = {
        "image": image_basename,
        "predictions": predictions,
        "ground_truth_count": len(gt_for_image),
    }

    if gt_for_image:
        result["accuracy"] = measure_pitch_accuracy(predictions, gt_for_image)

    return result


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_crop = sub.add_parser("crop", help="Save small crops for inspection")
    p_crop.add_argument("image")
    p_crop.add_argument("--max", type=int, default=20)

    p_run = sub.add_parser("baseline", help="Run baseline measurement")
    p_run.add_argument("image")

    args = parser.parse_args()

    if args.cmd == "crop":
        results = save_notehead_crops(args.image, max_crops=args.max)
        print(f"Saved {len(results)} crops to {CROP_OUTPUT_DIR}")
        for r in results:
            print(f"  idx={r['idx']:3d}  cy={r['cy']:6.0f}  cls={r['class_id']}  {r['crop_path']}")
    elif args.cmd == "baseline":
        result = run_baseline(args.image)
        print(f"Image: {result['image']}")
        print(f"Predictions: {len(result['predictions'])}")
        print(f"Ground truth entries: {result['ground_truth_count']}")
        if "accuracy" in result:
            acc = result["accuracy"]
            print(f"Exact accuracy:   {acc['exact_accuracy']:.1%}")
            print(f"Lenient (±1):     {acc['lenient_accuracy']:.1%}")
            print(f"Matched: {acc['matched_count']}/{acc['total_gt']}")
        else:
            print("(no ground truth for this image yet)")
