"""
Voice Binder — group notehead detections into SATB chord snapshots.

Input:
  - noteheads: list of detection dicts with cx, cy (pixel coords)
  - staves: list of Staff objects

Output:
  - list of chord dicts: {"S": notehead, "A": notehead, "T": notehead, "B": notehead}

Strategy:
  1. Sort noteheads by cx (horizontal position = time)
  2. Cluster into "simultaneous" groups (x-tolerance = half staff spacing)
  3. For each group, split by staff membership
  4. Multi-staff: top staff → S+A, bottom staff → T+B
  5. Single-staff: top 4 by y → S/A/T/B
"""
from __future__ import annotations

from staff_detector import Staff


def _notehead_to_staff(notehead: dict, staves: list[Staff]) -> int:
    """Return index of the staff whose y-range most likely contains the notehead.

    Uses distance to staff centerline; ties broken by closest edge.
    """
    if not staves:
        return -1
    cy = notehead["cy"]
    best_idx = 0
    best_dist = float("inf")
    for i, staff in enumerate(staves):
        staff_mid = (staff.line_ys[0] + staff.line_ys[-1]) / 2
        dist = abs(cy - staff_mid)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _cluster_by_x(
    noteheads: list[dict],
    x_tolerance: float,
) -> list[list[dict]]:
    """Group noteheads with similar x into chords."""
    if not noteheads:
        return []
    sorted_nh = sorted(noteheads, key=lambda n: n["cx"])
    clusters: list[list[dict]] = []
    current: list[dict] = [sorted_nh[0]]
    current_x = sorted_nh[0]["cx"]
    for nh in sorted_nh[1:]:
        if abs(nh["cx"] - current_x) <= x_tolerance:
            current.append(nh)
            current_x = sum(n["cx"] for n in current) / len(current)
        else:
            clusters.append(current)
            current = [nh]
            current_x = nh["cx"]
    if current:
        clusters.append(current)
    return clusters


def bind_voices(
    noteheads: list[dict],
    staves: list[Staff],
    x_tolerance: float = 20.0,
) -> list[dict[str, dict]]:
    """Group notehead detections into SATB chords.

    Args:
        noteheads: detection dicts (from downstream_eval.run_phase9_detection)
        staves: list of Staff objects from staff_detector
        x_tolerance: max horizontal distance to consider as simultaneous

    Returns:
        List of chord dicts. Each chord has keys "S", "A", "T", "B" mapping
        to the corresponding notehead detection dict.
        Chords with fewer than 4 noteheads are skipped.
        List is sorted by average x-coordinate.
    """
    if not noteheads:
        return []

    # Filter to notehead classes only (0 = filled, 1 = hollow)
    noteheads = [n for n in noteheads if n.get("class_id") in (0, 1)]
    if not noteheads:
        return []

    # Cluster simultaneously-sounding noteheads
    clusters = _cluster_by_x(noteheads, x_tolerance)

    chords: list[dict[str, dict]] = []
    for cluster in clusters:
        if len(cluster) < 4:
            continue  # not a full 4-voice chord

        if len(staves) >= 2:
            # Multi-staff: assign each notehead to its staff
            # Top staff (smallest y) takes S, A
            # Bottom staff takes T, B
            staves_sorted = sorted(staves, key=lambda s: s.line_ys[0])
            top_staff = staves_sorted[0]
            bot_staff = staves_sorted[-1]

            top_mid = (top_staff.line_ys[0] + top_staff.line_ys[-1]) / 2
            bot_mid = (bot_staff.line_ys[0] + bot_staff.line_ys[-1]) / 2
            boundary = (top_mid + bot_mid) / 2

            on_top = sorted(
                [n for n in cluster if n["cy"] < boundary],
                key=lambda n: n["cy"],
            )
            on_bot = sorted(
                [n for n in cluster if n["cy"] >= boundary],
                key=lambda n: n["cy"],
            )

            if len(on_top) >= 2 and len(on_bot) >= 2:
                chord = {
                    "S": on_top[0],
                    "A": on_top[1],
                    "T": on_bot[0],
                    "B": on_bot[1],
                }
                chords.append(chord)
            elif len(cluster) >= 4:
                # Fallback: pure y ordering
                sorted_by_y = sorted(cluster, key=lambda n: n["cy"])[:4]
                chord = {
                    "S": sorted_by_y[0],
                    "A": sorted_by_y[1],
                    "T": sorted_by_y[2],
                    "B": sorted_by_y[3],
                }
                chords.append(chord)
        else:
            # Single-staff: top 4 by y = S, A, T, B
            sorted_by_y = sorted(cluster, key=lambda n: n["cy"])[:4]
            chord = {
                "S": sorted_by_y[0],
                "A": sorted_by_y[1],
                "T": sorted_by_y[2],
                "B": sorted_by_y[3],
            }
            chords.append(chord)

    return chords
