"""
Accidental Resolver — Phase C5

Resolves which accidental applies to each notehead within a measure,
following the music theory rule:

  An accidental in a measure applies to ALL subsequent notes of the SAME
  pitch class (same letter + same octave) in that measure. The accidental
  resets at the next barline.

Scope (C5-Lite): in-measure persistence only.
  - Does NOT interpret key signature (notes default to natural unless
    explicitly marked or carried from a prior accidental in same measure).
  - Does NOT track cross-measure ties (a tied note keeps its accidental
    across barlines, but we don't model ties yet).

Input format: each notehead dict should have:
  - cx, cy: position
  - staff_idx: which staff this notehead belongs to (from voice binding)
  - step: diatonic step from staff top line (from staff.y_to_step)
  - explicit_accidental (optional): "sharp", "flat", "natural",
    "double_sharp", "double_flat" — from accidental detection upstream

Output: same noteheads, but each annotated with:
  - resolved_accidental: the effective accidental considering measure state
"""
from __future__ import annotations


def resolve_measure_accidentals(noteheads: list[dict]) -> list[dict]:
    """Resolve accidentals for all noteheads in a single measure.

    Processes left-to-right (sorted by cx). For each notehead:
      1. If it has an explicit_accidental → update measure state, mark
         resolved_accidental to that value.
      2. Else if measure state has an entry for this (staff_idx, step) →
         apply that state.
      3. Else → resolved_accidental = None (diatonic, no accidental).

    Returns the same noteheads in input order with the new field added.
    """
    if not noteheads:
        return []

    # Process in x-order, but preserve original order in output
    sorted_indices = sorted(range(len(noteheads)), key=lambda i: noteheads[i].get("cx", 0))

    # state: (staff_idx, step) → effective accidental
    state: dict[tuple, str] = {}

    # Annotate in processing order
    for i in sorted_indices:
        nh = noteheads[i]
        staff_idx = nh.get("staff_idx")
        step = nh.get("step")
        explicit = nh.get("explicit_accidental")

        # Notes without staff/step info just pass through
        if staff_idx is None or step is None:
            nh["resolved_accidental"] = explicit
            continue

        key = (staff_idx, step)

        if explicit is not None:
            # Explicit accidental updates state
            state[key] = explicit
            nh["resolved_accidental"] = explicit
        elif key in state:
            # Inherit from earlier in measure
            nh["resolved_accidental"] = state[key]
        else:
            # Diatonic
            nh["resolved_accidental"] = None

    return noteheads


if __name__ == "__main__":
    # Quick smoke test
    test_nhs = [
        {"cx": 100, "cy": 130, "staff_idx": 0, "step": 6, "explicit_accidental": "sharp"},
        {"cx": 200, "cy": 130, "staff_idx": 0, "step": 6},
        {"cx": 300, "cy": 130, "staff_idx": 0, "step": 6, "explicit_accidental": "natural"},
        {"cx": 400, "cy": 130, "staff_idx": 0, "step": 6},
    ]
    resolved = resolve_measure_accidentals(test_nhs)
    print("Smoke test (G♯ → G → G♮ → G):")
    for nh in resolved:
        print(f"  cx={nh['cx']}: explicit={nh.get('explicit_accidental')} "
              f"resolved={nh['resolved_accidental']}")
