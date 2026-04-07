"""
Pitch Estimator — convert staff position to MIDI pitch.

Algorithm: walk diatonic steps from the clef's reference line.
"""
from __future__ import annotations

from staff_detector import Staff


# Top-line MIDI pitch for each clef (no accidentals)
CLEF_TOP_LINE_MIDI: dict[str, int] = {
    "treble": 77,  # F5
    "bass":   57,  # A3
    "alto":   67,  # G4 (actually alto has C4 on middle line, so top = G4)
    "tenor":  64,  # E4 (tenor has C4 on 4th line from bottom, top = E4)
}

CLEF_TOP_LINE_LETTER: dict[str, str] = {
    "treble": "F",
    "bass":   "A",
    "alto":   "G",
    "tenor":  "E",
}

# Semitone interval to move DOWN one diatonic step
# from current letter name (e.g., F → E = 1 semitone down)
DOWN_INTERVAL: dict[str, int] = {
    "C": 1,  # C → B (1 semitone down)
    "D": 2,  # D → C
    "E": 2,  # E → D
    "F": 1,  # F → E
    "G": 2,  # G → F
    "A": 2,  # A → G
    "B": 2,  # B → A
}

LETTER_DOWN: dict[str, str] = {
    "C": "B", "D": "C", "E": "D", "F": "E",
    "G": "F", "A": "G", "B": "A",
}

LETTER_UP: dict[str, str] = {v: k for k, v in LETTER_DOWN.items()}
# Semitone interval to move UP one diatonic step
UP_INTERVAL: dict[str, int] = {
    "B": 1,  # B → C (1 semitone up)
    "A": 2,  # A → B
    "G": 2,  # G → A
    "F": 2,  # F → G
    "E": 1,  # E → F
    "D": 2,  # D → E
    "C": 2,  # C → D
}

ACCIDENTAL_SEMITONES: dict[str | None, int] = {
    None: 0,
    "natural": 0,
    "sharp": 1,
    "flat": -1,
    "double_sharp": 2,
    "double_flat": -2,
}


def _walk_diatonic(start_midi: int, start_letter: str, steps: int) -> int:
    """Walk `steps` diatonic positions from (start_midi, start_letter).

    Positive steps = go DOWN (to lower pitch, bigger step index on staff).
    Negative steps = go UP (to higher pitch, smaller step index on staff).
    """
    midi = start_midi
    letter = start_letter
    if steps >= 0:
        for _ in range(steps):
            midi -= DOWN_INTERVAL[letter]
            letter = LETTER_DOWN[letter]
    else:
        for _ in range(-steps):
            midi += UP_INTERVAL[letter]
            letter = LETTER_UP[letter]
    return midi


def estimate_pitch(
    notehead_y: float,
    staff: Staff,
    clef: str = "treble",
    accidental: str | None = None,
) -> int:
    """Estimate MIDI pitch from notehead position on staff."""
    if clef not in CLEF_TOP_LINE_MIDI:
        clef = "treble"

    step = staff.y_to_step(notehead_y)
    base_midi = _walk_diatonic(
        start_midi=CLEF_TOP_LINE_MIDI[clef],
        start_letter=CLEF_TOP_LINE_LETTER[clef],
        steps=step,
    )

    offset = ACCIDENTAL_SEMITONES.get(accidental, 0)
    midi = base_midi + offset
    midi = max(21, min(108, midi))
    return midi


if __name__ == "__main__":
    staff = Staff(line_ys=[100, 110, 120, 130, 140], spacing=10, x_min=0, x_max=500)
    print("Treble:")
    print(f"  y=100 step=0 → {estimate_pitch(100, staff, 'treble')} (want 77 F5)")
    print(f"  y=120 step=4 → {estimate_pitch(120, staff, 'treble')} (want 71 B4)")
    print(f"  y=140 step=8 → {estimate_pitch(140, staff, 'treble')} (want 64 E4)")
    print(f"  y=150 step=10 → {estimate_pitch(150, staff, 'treble')} (want 60 C4)")
    print(f"  y=80 step=-2 → {estimate_pitch(80, staff, 'treble')} (want 81 A5)")
    print(f"  Sharp on B4: {estimate_pitch(120, staff, 'treble', 'sharp')} (want 72)")
