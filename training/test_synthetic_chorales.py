"""
TDD: Synthetic chorale validation of harmony_rules.py

Build ChordSnapshot lists with KNOWN correct/incorrect progressions and
verify HarmonyAnalyzer detects exactly the expected violations.

KNOWN RULE IDs (from harmony_rules.py):
  M1 — Melodic interval issues (leaps > octave, unresolved 7ths, etc.)
  V1 — Voice crossing/overlap
  P1 — Parallel P5/P8 (CURRENT IMPL IS LOOSE: flags P5→P8 as well)
  P2 — Hidden intervals
  D1 — Triad doubling issues
  L1 — Leading tone resolution

AMBIGUITY NOTE (2026-04-07): harmony_rules_zh.md says "前一和弦為八度／五度,
...下一和弦仍為八度／五度". The word "仍" could mean "still the same" (strict)
or "still some P5 or P8" (loose). Current implementation is loose.
Stricter interpretation would check interval1 == interval2 (mod 12).
"""
import pytest
import sys
sys.path.insert(0, "/home/thc1006/dev/music-app")
from harmony_rules import (
    HarmonyAnalyzer, ChordSnapshot, NoteEvent, KeySignature,
)


def _chord(index: int, s: int, a: int, t: int, b: int,
           measure: int = 1, beat: float = 0.0) -> ChordSnapshot:
    """Build a ChordSnapshot from MIDI pitches."""
    return ChordSnapshot(
        index=index,
        measure=measure,
        beat=beat,
        notes={
            "S": NoteEvent(voice="S", midi=s, measure=measure, beat=beat),
            "A": NoteEvent(voice="A", midi=a, measure=measure, beat=beat),
            "T": NoteEvent(voice="T", midi=t, measure=measure, beat=beat),
            "B": NoteEvent(voice="B", midi=b, measure=measure, beat=beat),
        },
    )


class TestStaticAndSingleVoice:
    """Sanity: static or single-voice moves should never trigger voice-leading rules."""

    def test_static_chord_no_violations(self):
        """Two identical chords — no voice movement — should have no parallel violations."""
        chords = [
            _chord(0, s=72, a=67, t=64, b=48),
            _chord(1, s=72, a=67, t=64, b=48),
        ]
        violations = HarmonyAnalyzer(chords).analyze()
        p1 = [v for v in violations if v.rule_id == "P1"]
        assert len(p1) == 0

    def test_single_bass_move_no_parallel(self):
        """Only bass moves — other voices static — no parallels possible."""
        chords = [
            _chord(0, s=72, a=67, t=64, b=48),
            _chord(1, s=72, a=67, t=64, b=50),
        ]
        violations = HarmonyAnalyzer(chords).analyze()
        p1 = [v for v in violations if v.rule_id == "P1"]
        assert len(p1) == 0, f"Got: {[v.detail for v in p1]}"


class TestParallelOctaveDetection:
    """P1 should fire when both voices move same direction, both intervals = P8."""

    def test_parallel_octave_S_B_same_direction(self):
        """S and B both move C→D in parallel octaves.

        Chord 1: S=C5(72), B=C3(48), interval = 24 (2 octaves)
        Chord 2: S=D5(74), B=D3(50), interval = 24 (2 octaves)
        Both voices moved up by 2. → parallel octaves.
        """
        chords = [
            _chord(0, s=72, a=67, t=64, b=48),
            _chord(1, s=74, a=67, t=64, b=50),
        ]
        violations = HarmonyAnalyzer(chords).analyze()
        p1_sb = [v for v in violations if v.rule_id == "P1"
                 and v.location.get("voices") == "S-B"]
        assert len(p1_sb) >= 1, f"Expected S-B parallel octave, got {[v.location for v in violations]}"


class TestParallelFifthDetection:
    """P1 should fire for parallel P5s as well."""

    def test_parallel_fifth_S_A_same_direction(self):
        """S and A move with a perfect fifth between them.

        Chord 1: S=C5(72), A=F4(65) → interval 7 (P5)
        Chord 2: S=D5(74), A=G4(67) → interval 7 (P5)
        Both moved up 2 semitones. Parallel fifth.
        """
        chords = [
            _chord(0, s=72, a=65, t=60, b=48),
            _chord(1, s=74, a=67, t=60, b=48),
        ]
        violations = HarmonyAnalyzer(chords).analyze()
        p1 = [v for v in violations if v.rule_id == "P1"
              and v.location.get("voices") == "S-A"]
        assert len(p1) >= 1, (
            f"Expected S-A parallel fifth, got {[v.location for v in violations]}"
        )


class TestVoiceCrossing:
    """V1 should fire when a voice is above a higher voice (e.g., A above S)."""

    def test_alto_above_soprano_triggers_V1(self):
        chords = [
            _chord(0, s=67, a=72, t=60, b=48),  # A > S — invalid
        ]
        violations = HarmonyAnalyzer(chords).analyze()
        v1 = [v for v in violations if v.rule_id == "V1"]
        assert len(v1) >= 1, (
            f"Expected voice crossing, got {[v.rule_id for v in violations]}"
        )

    def test_normal_order_no_V1(self):
        """Normal SATB ordering: S > A > T > B."""
        chords = [
            _chord(0, s=72, a=67, t=60, b=48),
        ]
        violations = HarmonyAnalyzer(chords).analyze()
        v1 = [v for v in violations if v.rule_id == "V1"]
        assert len(v1) == 0, f"Unexpected V1: {[v.detail for v in v1]}"


class TestMelodicInterval:
    """M1 fires when a voice leaps > octave or has other melodic issues."""

    def test_soprano_jumps_beyond_octave_triggers_M1(self):
        """S leaps from C4 (60) to D5 (74) = 14 semitones > octave."""
        chords = [
            _chord(0, s=60, a=55, t=52, b=48),
            _chord(1, s=74, a=55, t=52, b=48),
        ]
        violations = HarmonyAnalyzer(chords).analyze()
        m1 = [v for v in violations if v.rule_id == "M1"]
        assert len(m1) >= 1, (
            f"Expected M1 for octave+ leap, got {[v.rule_id for v in violations]}"
        )


class TestAnalyzerSmoke:

    def test_analyze_empty_no_crash(self):
        assert HarmonyAnalyzer([]).analyze() == []

    def test_analyze_single_chord_no_progression_rules(self):
        chords = [_chord(0, s=72, a=67, t=64, b=48)]
        violations = HarmonyAnalyzer(chords).analyze()
        # Single chord: no parallel or melodic violations possible
        assert not any(v.rule_id in ("P1", "P2", "M1") for v in violations)

    def test_rule_ids_are_known(self):
        """Any violation's rule_id should be one of the documented IDs."""
        chords = [
            _chord(0, s=72, a=67, t=64, b=48),
            _chord(1, s=74, a=69, t=66, b=50),  # all move up — will trigger parallels
        ]
        violations = HarmonyAnalyzer(chords).analyze()
        known_ids = {"M1", "V1", "P1", "P2", "D1", "L1"}
        for v in violations:
            assert v.rule_id in known_ids, f"Unknown rule_id: {v.rule_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
