"""
Measure Detector — segment noteheads into measures using barline x-positions.

Input: detection dicts (from downstream_eval.run_phase9_detection)
Output: list of measure dicts, each containing:
  {
    "number": int,  # 1-indexed measure number
    "x_start": float,  # left boundary
    "x_end": float,  # right boundary
    "noteheads": list[dict],  # noteheads in this measure, sorted by x
  }
"""
from __future__ import annotations

BARLINE_CLASS_IDS = {22, 23, 24, 25}  # barline, double, final, repeat
NOTEHEAD_CLASS_IDS = {0, 1}  # filled, hollow


def segment_measures(detections: list[dict]) -> list[dict]:
    """Segment noteheads into measures using barline positions.

    Args:
        detections: list of detection dicts from run_phase9_detection

    Returns:
        List of measure dicts sorted by measure number.
    """
    if not detections:
        return []

    barlines = [d for d in detections if d.get("class_id") in BARLINE_CLASS_IDS]
    noteheads = [d for d in detections if d.get("class_id") in NOTEHEAD_CLASS_IDS]

    if not noteheads:
        return []

    barlines.sort(key=lambda b: b["cx"])
    noteheads.sort(key=lambda n: n["cx"])

    # Create measure boundaries from barline x-coords
    # A measure is the region BEFORE each barline (and the final trailing region)
    x_min = min(n["cx"] for n in noteheads) - 10
    x_max = max(n["cx"] for n in noteheads) + 10

    boundaries = [x_min]
    for b in barlines:
        if b["cx"] > boundaries[-1]:
            boundaries.append(b["cx"])
    if boundaries[-1] < x_max:
        boundaries.append(x_max)

    # Build measures from consecutive boundary pairs
    measures: list[dict] = []
    for i in range(len(boundaries) - 1):
        x_start = boundaries[i]
        x_end = boundaries[i + 1]
        in_measure = [
            n for n in noteheads
            if x_start <= n["cx"] < x_end
        ]
        if not in_measure and len(measures) > 0:
            # Skip empty measures at the end (e.g., after final barline)
            continue

        # Assign beat-in-measure: beat = index of chord onset within measure
        # Group noteheads by x (simultaneous = same beat)
        beat_groups: list[list[dict]] = []
        current_group: list[dict] = []
        current_x: float = -1.0
        x_tol = 20.0
        for n in sorted(in_measure, key=lambda nh: nh["cx"]):
            if not current_group or abs(n["cx"] - current_x) <= x_tol:
                current_group.append(n)
                if current_group:
                    current_x = sum(m["cx"] for m in current_group) / len(current_group)
            else:
                beat_groups.append(current_group)
                current_group = [n]
                current_x = n["cx"]
        if current_group:
            beat_groups.append(current_group)

        for beat_idx, group in enumerate(beat_groups):
            for n in group:
                n["beat_in_measure"] = float(beat_idx)

        measures.append({
            "number": len(measures) + 1,
            "x_start": x_start,
            "x_end": x_end,
            "noteheads": sorted(in_measure, key=lambda n: n["cx"]),
        })

    return measures
