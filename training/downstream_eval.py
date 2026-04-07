"""
Phase 9 → harmony_rules.py Downstream Pipeline

The real evaluation metric: can Phase 9 + SymbolAssembler + harmony_rules
correctly identify harmony errors on real sheet music?

If this pipeline works on a handful of real images, Phase 9 is deployable
and we can skip Phase 10 retraining entirely.

Pipeline stages:
  1. YOLO detection (Phase 9 best.pt)
  2. Per-class name mapping
  3. Heuristic symbol assembly: notehead bboxes → note events → chord snapshots
  4. harmony_rules.HarmonyAnalyzer → violations

Note: The symbol assembler here is intentionally simple. A production-grade
assembler (voice tracking, staff binding, pitch estimation from vertical
position relative to staff lines, etc.) belongs in Kotlin (Android) or in
a dedicated module. This file proves the pipeline is wireable end-to-end.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harmony_rules import (
    HarmonyAnalyzer,
    ChordSnapshot,
    NoteEvent,
    KeySignature,
    RuleViolation,
)

PHASE9_WEIGHTS = PROJECT_ROOT / (
    "runs/detect/harmony_omr_v2_phase9/cv_noteheads_v1_stage2/weights/best.pt"
)

# Class ID → name mapping (aligned with training dataset yaml)
CLASS_NAMES = [
    "notehead_filled", "notehead_hollow", "beam", "flag_8th", "flag_16th",
    "flag_32nd", "augmentation_dot", "tie", "clef_treble", "clef_bass",
    "clef_alto", "clef_tenor", "accidental_sharp", "accidental_flat",
    "accidental_natural", "accidental_double_sharp", "accidental_double_flat",
    "rest_whole", "rest_half", "rest_quarter", "rest_8th", "rest_16th",
    "barline", "barline_double", "barline_final", "barline_repeat",
    "time_signature", "key_signature", "dynamic_loud", "dynamic_soft",
    "fermata", "ledger_line",
]


# ──────────────────────────────────────────────────────────────────────
# Detection layer
# ──────────────────────────────────────────────────────────────────────

def run_phase9_detection(
    image_path: str,
    weights: str = str(PHASE9_WEIGHTS),
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.55,
    max_det: int = 1500,
) -> list[dict]:
    """Run Phase 9 model on an image; return list of detection dicts.

    Returns:
        List of {class_id, class_name, cx, cy, w, h, confidence} dicts
        in absolute pixel coordinates.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )
    if not results:
        return []

    boxes = results[0].boxes
    detections = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        xyxy = boxes.xyxy[i].tolist()
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        confidence = float(boxes.conf[i])
        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
            "cx": float(cx),
            "cy": float(cy),
            "w": float(w),
            "h": float(h),
            "confidence": confidence,
        })
    return detections


# ──────────────────────────────────────────────────────────────────────
# Symbol assembly layer
# ──────────────────────────────────────────────────────────────────────

def _find_accidental_for_notehead(
    notehead: dict,
    detections: list[dict],
    max_dist_left: float = 40.0,
) -> str | None:
    """Find an accidental immediately left of this notehead (same y ± tolerance)."""
    acc_map = {
        12: "sharp", 13: "flat", 14: "natural",
        15: "double_sharp", 16: "double_flat",
    }
    nh_cx = notehead["cx"]
    nh_cy = notehead["cy"]
    best = None
    best_dist = float("inf")
    for d in detections:
        if d.get("class_id") not in acc_map:
            continue
        # Accidentals sit to the LEFT of the notehead
        dx = nh_cx - d["cx"]
        dy = abs(nh_cy - d["cy"])
        if 0 < dx < max_dist_left and dy < 15:
            if dx < best_dist:
                best_dist = dx
                best = d
    return acc_map[best["class_id"]] if best else None


def build_chord_snapshots(
    detections: list[dict],
    image_path: str | None = None,
) -> list[ChordSnapshot]:
    """Assemble detections into ChordSnapshot using the full B1-B4 + C2 pipeline.

    Pipeline:
      1. staff_detector: find staff lines
      2. clef_detector: assign one clef per staff (C2)
      3. measure_detector: segment by barlines
      4. voice_binder: group noteheads into SATB chords (C4 layout-aware)
      5. accidental_resolver: in-measure accidental persistence (C5)
      6. pitch_estimator: compute MIDI from staff position + clef + accidentals
    """
    from staff_detector import detect_staves
    from clef_detector import assign_clefs_to_staves
    from voice_binder import bind_voices
    from measure_detector import segment_measures
    from pitch_estimator import estimate_pitch
    from accidental_resolver import resolve_measure_accidentals

    if image_path is None:
        return []

    staves = detect_staves(image_path)
    if not staves:
        return []

    # C2: assign clef per staff using dedicated module
    clef_classes = {8, 9, 10, 11}
    clef_detections = [d for d in detections if d.get("class_id") in clef_classes]
    staff_clefs = assign_clefs_to_staves(clef_detections, staves)

    # Segment into measures
    measures = segment_measures(detections)
    if not measures:
        return []

    chords: list[ChordSnapshot] = []
    chord_index = 0

    def _staff_centerline(s):
        return (s.line_ys[0] + s.line_ys[-1]) / 2

    for measure in measures:
        measure_nh = measure["noteheads"]
        if not measure_nh:
            continue

        # C5: pre-annotate noteheads with staff_idx, step, explicit_accidental
        # so accidental_resolver can track in-measure state
        for nh in measure_nh:
            nh_cy = nh["cy"]
            staff_idx = min(
                range(len(staves)),
                key=lambda i: abs(nh_cy - _staff_centerline(staves[i])),
            )
            nh["staff_idx"] = staff_idx
            nh["step"] = staves[staff_idx].y_to_step(nh_cy)
            nh["explicit_accidental"] = _find_accidental_for_notehead(nh, detections)

        # C5: resolve in-measure accidental persistence
        resolve_measure_accidentals(measure_nh)

        # Bind voices within this measure
        voice_chords = bind_voices(measure_nh, staves)

        for chord_map in voice_chords:
            notes = {}
            for voice_name in ("S", "A", "T", "B"):
                nh = chord_map.get(voice_name)
                if nh is None:
                    continue

                staff_idx = nh["staff_idx"]
                staff = staves[staff_idx]
                clef = staff_clefs[staff_idx]
                # C5: use resolved accidental (not raw explicit detection)
                accidental = nh.get("resolved_accidental")

                # Estimate pitch
                midi = estimate_pitch(nh["cy"], staff, clef, accidental)

                notes[voice_name] = NoteEvent(
                    voice=voice_name,
                    midi=midi,
                    measure=measure["number"],
                    beat=nh.get("beat_in_measure", 0.0),
                )

            if len(notes) == 4:
                chords.append(ChordSnapshot(
                    index=chord_index,
                    measure=measure["number"],
                    beat=float(list(notes.values())[0].beat),
                    notes=notes,
                ))
                chord_index += 1

    return chords


# ──────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.55,
) -> dict[str, Any]:
    """Run the complete Phase 9 → harmony rules pipeline.

    Args:
        image_path: path to sheet music image
        imgsz: YOLO inference size. Defaults match Phase 6 fixed val (1280).
            For high-DPI Bach chorale renders (e.g. 3300×2550 from LilyPond
            300 DPI), imgsz=1600 + conf=0.10 gives ~3× more noteheads —
            see `training/datasets/chorale_gt/README.md`.
        conf: detection confidence threshold
        iou: NMS IoU threshold

    Returns:
        {
            "num_detections": int,
            "num_noteheads": int,
            "num_chords": int,
            "violations": list[RuleViolation],
        }
    """
    detections = run_phase9_detection(image_path, imgsz=imgsz, conf=conf, iou=iou)
    noteheads = [d for d in detections if d["class_id"] in (0, 1)]
    chords = build_chord_snapshots(detections, image_path=image_path)

    violations: list[RuleViolation] = []
    if chords:
        analyzer = HarmonyAnalyzer(chords)
        violations = analyzer.analyze()

    return {
        "num_detections": len(detections),
        "num_noteheads": len(noteheads),
        "num_chords": len(chords),
        "violations": violations,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to sheet music image")
    args = parser.parse_args()

    result = run_pipeline(args.image)
    print(f"Detections: {result['num_detections']}")
    print(f"Noteheads:  {result['num_noteheads']}")
    print(f"Chords:     {result['num_chords']}")
    print(f"Violations: {len(result['violations'])}")
    for v in result["violations"][:5]:
        print(f"  [{v.rule_id}] {v.message_zh}")
