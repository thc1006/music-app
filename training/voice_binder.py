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


def _staff_centerline(staff: Staff) -> float:
    return (staff.line_ys[0] + staff.line_ys[-1]) / 2


def _bind_grand_staff(cluster: list[dict], staves: list[Staff]) -> dict | None:
    """2-stave grand staff: top staff = S/A, bottom = T/B.

    Returns None if cluster doesn't have 2 noteheads on each staff.
    """
    staves_sorted = sorted(staves, key=lambda s: s.line_ys[0])
    top_staff, bot_staff = staves_sorted[0], staves_sorted[1]
    boundary = (_staff_centerline(top_staff) + _staff_centerline(bot_staff)) / 2

    on_top = sorted([n for n in cluster if n["cy"] < boundary], key=lambda n: n["cy"])
    on_bot = sorted([n for n in cluster if n["cy"] >= boundary], key=lambda n: n["cy"])

    if len(on_top) >= 2 and len(on_bot) >= 2:
        return {
            "S": on_top[0],
            "A": on_top[1],
            "T": on_bot[0],
            "B": on_bot[1],
        }
    return None


def _bind_quartet(cluster: list[dict], staves: list[Staff]) -> dict | None:
    """4-stave quartet: each staff = one voice (S, A, T, B from top).

    For each staff, find the closest notehead in the cluster and assign it
    to that voice. Returns None if any staff has no notehead in this cluster.
    """
    staves_sorted = sorted(staves, key=lambda s: s.line_ys[0])
    voice_names = ["S", "A", "T", "B"]

    # For each notehead in cluster, find which staff it belongs to (closest centerline)
    nh_to_staff: dict[int, list[dict]] = {i: [] for i in range(4)}
    for nh in cluster:
        staff_idx = min(
            range(4),
            key=lambda i: abs(nh["cy"] - _staff_centerline(staves_sorted[i])),
        )
        nh_to_staff[staff_idx].append(nh)

    # Each staff must have at least one notehead in this cluster
    if any(len(nh_to_staff[i]) == 0 for i in range(4)):
        return None

    # Take the topmost notehead per staff (most likely the actual voice)
    chord = {}
    for i, voice in enumerate(voice_names):
        # If multiple noteheads on same staff at this beat, take the highest
        chord[voice] = sorted(nh_to_staff[i], key=lambda n: n["cy"])[0]
    return chord


def bind_voices(
    noteheads: list[dict],
    staves: list[Staff],
    x_tolerance: float = 20.0,
) -> list[dict[str, dict]]:
    """Group notehead detections into SATB chords with layout-aware dispatch.

    Layout dispatch (Phase C4):
      - 2 staves → grand staff (piano/keyboard chorale): top=S/A, bottom=T/B
      - 4 staves → quartet (string quartet, hymn): each staff = one voice
      - Other counts → return [] (not 4-part harmony content; rule engine
        does not apply meaningfully)

    Args:
        noteheads: detection dicts (from downstream_eval.run_phase9_detection)
        staves: list of Staff objects from staff_detector
        x_tolerance: max horizontal distance to consider as simultaneous

    Returns:
        List of chord dicts (S/A/T/B → notehead). Empty list if layout
        unsupported or no full chord can be assembled.
    """
    if not noteheads or not staves:
        return []

    # C4: layout-aware dispatch
    if len(staves) not in (2, 4):
        # Not a 4-part harmony layout — skip silently
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

        if len(staves) == 2:
            chord = _bind_grand_staff(cluster, staves)
        else:  # len(staves) == 4
            chord = _bind_quartet(cluster, staves)

        if chord is not None:
            chords.append(chord)

    return chords
