"""
Clef Detector — assign one clef per staff.

Phase C C2: dedicated, well-tested module that replaces the simple
clef-detection heuristic that lived inside downstream_eval.py.

Strategy:
  1. Filter clef detections by spatial validity:
     - Clef cx must be in the LEFT zone of the staff
       (within `CLEF_LEFT_ZONE_FRACTION` × staff width from x_min)
     - Clef must be in the staff's y range, accepted via either:
       a) Bbox y-overlap with the staff lines (with padding
          `CLEF_BBOX_PADDING_FACTOR` × spacing on each side), OR
       b) Centerline-to-centerline distance ≤
          `CLEF_CENTERLINE_TOLERANCE_FACTOR` × spacing (fallback for
          clefs whose bbox doesn't overlap due to detection quirks).
  2. Greedy 1-to-1 matching: each clef detection is assigned to at most
     one staff (its nearest unassigned staff). Each staff gets at most
     one clef. (Implemented in C2-Fix5; see assign_clefs_to_staves.)
  3. If no clef found for a staff, default to DEFAULT_CLEF.

Output: list[str] of length len(staves), each entry one of:
  "treble", "bass", "alto", "tenor"
"""
from __future__ import annotations

from staff_detector import Staff


CLEF_CLASS_TO_NAME: dict[int, str] = {
    8: "treble",
    9: "bass",
    10: "alto",
    11: "tenor",
}

# ──────────────────────────────────────────────────────────────────
# Tunable thresholds (extracted from previously hardcoded values)
# ──────────────────────────────────────────────────────────────────
# Clef cx must be within this fraction of the staff width (from x_min).
# 0.15 = left 15% — covers typical clef position with safety margin.
CLEF_LEFT_ZONE_FRACTION: float = 0.15

# When checking bbox overlap, expand the staff y range by this many
# spacings on each side. Clefs are usually drawn slightly above/below
# the actual staff lines.
CLEF_BBOX_PADDING_FACTOR: float = 1.5

# Fallback: if bbox doesn't overlap, accept clefs whose centerline is
# within this many spacings of the staff centerline. Phase 9 detections
# can have inaccurate bbox heights, so we need a centerline fallback.
CLEF_CENTERLINE_TOLERANCE_FACTOR: float = 5.0

# Default clef when no detection is associated with a staff.
DEFAULT_CLEF: str = "treble"


def _is_clef_at_staff_left(clef: dict, staff: Staff) -> bool:
    """Clefs sit at the left of the staff. Reject if cx is too far right."""
    staff_width = staff.x_max - staff.x_min
    if staff_width <= 0:
        return False
    left_zone_max = staff.x_min + staff_width * CLEF_LEFT_ZONE_FRACTION
    return clef["cx"] <= left_zone_max


def _clef_distance_to_staff(clef: dict, staff: Staff) -> float:
    """Distance from clef centerline to staff centerline (smaller = better match)."""
    staff_mid = (staff.line_ys[0] + staff.line_ys[-1]) / 2
    return abs(clef["cy"] - staff_mid)


def _clef_bbox_overlaps_staff(clef: dict, staff: Staff) -> bool:
    """Check if the clef's vertical bbox overlaps with the staff's y range.

    Clefs are visually anchored at the staff but their bbox can extend above
    or below the staff lines. Padding controlled by CLEF_BBOX_PADDING_FACTOR.
    """
    clef_top = clef["cy"] - clef["h"] / 2
    clef_bot = clef["cy"] + clef["h"] / 2
    pad = staff.spacing * CLEF_BBOX_PADDING_FACTOR
    staff_top = staff.line_ys[0] - pad
    staff_bot = staff.line_ys[-1] + pad
    return clef_top <= staff_bot and clef_bot >= staff_top


def _is_clef_in_staff_range(clef: dict, staff: Staff) -> bool:
    """Combined check: bbox overlap OR centerline within tolerance.

    Bbox overlap is primary; centerline distance is fallback for clefs
    where Phase 9's bbox h is unreliable.
    """
    if _clef_bbox_overlaps_staff(clef, staff):
        return True
    distance = _clef_distance_to_staff(clef, staff)
    return distance <= staff.spacing * CLEF_CENTERLINE_TOLERANCE_FACTOR


def assign_clefs_to_staves(
    clef_detections: list[dict],
    staves: list[Staff],
) -> list[str]:
    """Assign one clef name to each staff using greedy 1-to-1 matching.

    Algorithm:
      1. Build all valid (clef, staff) candidate pairs with their distances.
      2. Sort by distance (smallest first).
      3. Greedy: for each candidate, if neither the clef nor the staff is
         already used, assign it. Otherwise skip.
      4. Staves with no assignment fall back to DEFAULT_CLEF.

    This guarantees:
      - Each clef detection is assigned to AT MOST one staff
      - Each staff has AT MOST one clef
      - The closest clef-staff pair always wins

    Args:
        clef_detections: list of detection dicts (class_id 8-11)
        staves: list of Staff objects from staff_detector

    Returns:
        list[str] of length len(staves), each one of
        {"treble","bass","alto","tenor"}
    """
    if not staves:
        return []

    # Filter to clefs whose class_id is recognized
    valid_clefs = [
        c for c in clef_detections
        if c.get("class_id") in CLEF_CLASS_TO_NAME
    ]

    # Build candidate (distance, clef_idx, staff_idx) tuples
    candidates: list[tuple[float, int, int]] = []
    for ci, clef in enumerate(valid_clefs):
        for si, staff in enumerate(staves):
            if not _is_clef_at_staff_left(clef, staff):
                continue
            if not _is_clef_in_staff_range(clef, staff):
                continue
            dist = _clef_distance_to_staff(clef, staff)
            candidates.append((dist, ci, si))

    # Sort by distance (closest first)
    candidates.sort()

    # Greedy 1-to-1 assignment
    assigned_staves: dict[int, int] = {}  # staff_idx → clef_idx
    used_clefs: set[int] = set()
    for dist, ci, si in candidates:
        if ci in used_clefs:
            continue
        if si in assigned_staves:
            continue
        assigned_staves[si] = ci
        used_clefs.add(ci)

    # Build result list
    result: list[str] = []
    for si in range(len(staves)):
        if si in assigned_staves:
            ci = assigned_staves[si]
            result.append(CLEF_CLASS_TO_NAME[valid_clefs[ci]["class_id"]])
        else:
            result.append(DEFAULT_CLEF)

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from downstream_eval import run_phase9_detection
    from staff_detector import detect_staves

    if len(sys.argv) < 2:
        print("Usage: python clef_detector.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    staves = detect_staves(img_path)
    detections = run_phase9_detection(img_path)
    clefs = [d for d in detections if d.get("class_id") in CLEF_CLASS_TO_NAME]

    print(f"Image: {img_path}")
    print(f"Staves: {len(staves)}, Clef detections: {len(clefs)}")
    assignments = assign_clefs_to_staves(clefs, staves)
    for i, (staff, clef_name) in enumerate(zip(staves, assignments)):
        print(f"  Staff {i:2d} (y={int(staff.line_ys[0])}): {clef_name}")
