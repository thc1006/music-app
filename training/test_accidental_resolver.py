"""
TDD Red: Phase C5 — measure-scoped accidental resolver tests.

Music theory rule: an accidental in a measure applies to ALL subsequent
notes of the SAME pitch class (same letter name + same octave) in that measure.

Example: in measure 5, if there's a G♯ at beat 2, then any G at beat 3, 4
also becomes G♯ — even without an explicit ♯ symbol on those notes.

Cross-measure: accidental does NOT carry over to the next measure (resets
at barline). Key signature accidentals are persistent (handled separately).

Scope (C5-Lite): in-measure persistence only. Key signature interpretation
is out of scope.
"""
import pytest


def _make_notehead_with_step(cx, cy, staff_idx, step, explicit_acc=None):
    """Build an annotated notehead dict.

    Includes the staff/step info that the resolver needs to track which
    pitch class is which.
    """
    nh = {
        "class_id": 0,
        "class_name": "notehead_filled",
        "cx": float(cx),
        "cy": float(cy),
        "w": 20.0,
        "h": 25.0,
        "confidence": 0.9,
        "staff_idx": staff_idx,
        "step": step,
    }
    if explicit_acc is not None:
        nh["explicit_accidental"] = explicit_acc
    return nh


# ──────────────────────────────────────────────────────────────────────
# Module imports
# ──────────────────────────────────────────────────────────────────────

class TestModuleImport:

    def test_module_importable(self):
        import importlib.util
        spec = importlib.util.find_spec("accidental_resolver")
        assert spec is not None, "Create training/accidental_resolver.py"

    def test_resolve_function_exists(self):
        from accidental_resolver import resolve_measure_accidentals
        assert callable(resolve_measure_accidentals)


# ──────────────────────────────────────────────────────────────────────
# Single accidental persistence within a measure
# ──────────────────────────────────────────────────────────────────────

class TestSingleAccidentalPersistence:

    def test_first_sharp_persists_to_subsequent_same_pitch(self):
        """G♯ at beat 1 → all G's in same measure should also be G♯."""
        from accidental_resolver import resolve_measure_accidentals
        # 3 noteheads in same measure, all on staff 0 step 6 (G in treble)
        noteheads = [
            _make_notehead_with_step(cx=100, cy=130, staff_idx=0, step=6,
                                     explicit_acc="sharp"),
            _make_notehead_with_step(cx=200, cy=130, staff_idx=0, step=6),
            _make_notehead_with_step(cx=300, cy=130, staff_idx=0, step=6),
        ]
        resolved = resolve_measure_accidentals(noteheads)
        assert resolved[0]["resolved_accidental"] == "sharp"
        assert resolved[1]["resolved_accidental"] == "sharp"
        assert resolved[2]["resolved_accidental"] == "sharp"

    def test_flat_persists(self):
        from accidental_resolver import resolve_measure_accidentals
        noteheads = [
            _make_notehead_with_step(cx=100, cy=140, staff_idx=0, step=4,
                                     explicit_acc="flat"),
            _make_notehead_with_step(cx=200, cy=140, staff_idx=0, step=4),
        ]
        resolved = resolve_measure_accidentals(noteheads)
        assert resolved[0]["resolved_accidental"] == "flat"
        assert resolved[1]["resolved_accidental"] == "flat"

    def test_no_accidental_means_natural(self):
        """A notehead with no explicit accidental and no prior state → no accidental."""
        from accidental_resolver import resolve_measure_accidentals
        noteheads = [
            _make_notehead_with_step(cx=100, cy=130, staff_idx=0, step=6),
            _make_notehead_with_step(cx=200, cy=130, staff_idx=0, step=6),
        ]
        resolved = resolve_measure_accidentals(noteheads)
        assert resolved[0]["resolved_accidental"] is None
        assert resolved[1]["resolved_accidental"] is None


# ──────────────────────────────────────────────────────────────────────
# Pitch class scoping (different staff/step → different state)
# ──────────────────────────────────────────────────────────────────────

class TestPitchClassScoping:

    def test_different_step_does_not_inherit(self):
        """G♯ does not affect F or A on the same staff."""
        from accidental_resolver import resolve_measure_accidentals
        noteheads = [
            _make_notehead_with_step(cx=100, cy=130, staff_idx=0, step=6,
                                     explicit_acc="sharp"),  # G♯
            _make_notehead_with_step(cx=200, cy=125, staff_idx=0, step=5),  # A
            _make_notehead_with_step(cx=300, cy=135, staff_idx=0, step=7),  # F
        ]
        resolved = resolve_measure_accidentals(noteheads)
        assert resolved[0]["resolved_accidental"] == "sharp"
        assert resolved[1]["resolved_accidental"] is None
        assert resolved[2]["resolved_accidental"] is None

    def test_different_staff_does_not_inherit(self):
        """G♯ on staff 0 does not affect G on staff 1.

        (Music theory note: this is technically a simplification — in some
        rule sets, accidentals carry across staves of a single instrument.
        For 4-part chorales each voice is independent, so per-staff is fine.)
        """
        from accidental_resolver import resolve_measure_accidentals
        noteheads = [
            _make_notehead_with_step(cx=100, cy=130, staff_idx=0, step=6,
                                     explicit_acc="sharp"),
            _make_notehead_with_step(cx=200, cy=330, staff_idx=1, step=6),
        ]
        resolved = resolve_measure_accidentals(noteheads)
        assert resolved[0]["resolved_accidental"] == "sharp"
        assert resolved[1]["resolved_accidental"] is None


# ──────────────────────────────────────────────────────────────────────
# Natural cancellation
# ──────────────────────────────────────────────────────────────────────

class TestNaturalCancellation:

    def test_natural_cancels_prior_sharp(self):
        from accidental_resolver import resolve_measure_accidentals
        noteheads = [
            _make_notehead_with_step(cx=100, cy=130, staff_idx=0, step=6,
                                     explicit_acc="sharp"),
            _make_notehead_with_step(cx=200, cy=130, staff_idx=0, step=6,
                                     explicit_acc="natural"),
            _make_notehead_with_step(cx=300, cy=130, staff_idx=0, step=6),
        ]
        resolved = resolve_measure_accidentals(noteheads)
        assert resolved[0]["resolved_accidental"] == "sharp"
        assert resolved[1]["resolved_accidental"] == "natural"
        assert resolved[2]["resolved_accidental"] == "natural", (
            "After natural, subsequent same-pitch should be natural too"
        )


# ──────────────────────────────────────────────────────────────────────
# Order matters
# ──────────────────────────────────────────────────────────────────────

class TestProcessingOrder:

    def test_noteheads_processed_in_x_order(self):
        """Even if input is unsorted, processing should be left-to-right."""
        from accidental_resolver import resolve_measure_accidentals
        noteheads = [
            # Out of order: x=300 first, then 100, then 200
            _make_notehead_with_step(cx=300, cy=130, staff_idx=0, step=6),
            _make_notehead_with_step(cx=100, cy=130, staff_idx=0, step=6,
                                     explicit_acc="sharp"),
            _make_notehead_with_step(cx=200, cy=130, staff_idx=0, step=6),
        ]
        resolved = resolve_measure_accidentals(noteheads)
        # Result should be in input order (resolver doesn't reorder),
        # but the accidental at x=100 affects x=200 and x=300
        # Find by original cx
        result_by_cx = {n["cx"]: n["resolved_accidental"] for n in resolved}
        assert result_by_cx[100] == "sharp"
        assert result_by_cx[200] == "sharp"
        assert result_by_cx[300] == "sharp"


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_noteheads_returns_empty(self):
        from accidental_resolver import resolve_measure_accidentals
        assert resolve_measure_accidentals([]) == []

    def test_noteheads_without_step_are_passed_through(self):
        """If a notehead lacks staff_idx/step (e.g., from old code), don't crash."""
        from accidental_resolver import resolve_measure_accidentals
        noteheads = [
            {"cx": 100, "cy": 130, "class_id": 0, "w": 20, "h": 25,
             "confidence": 0.9},
        ]
        resolved = resolve_measure_accidentals(noteheads)
        assert len(resolved) == 1
        # Without step info, treated as no accidental (or pass-through)
        assert resolved[0].get("resolved_accidental") is None


# ──────────────────────────────────────────────────────────────────────
# Integration: cross-measure reset
# ──────────────────────────────────────────────────────────────────────

class TestCrossMeasureReset:
    """When called per-measure (which is the usage), state resets between calls."""

    def test_separate_calls_dont_share_state(self):
        from accidental_resolver import resolve_measure_accidentals

        # Measure 1
        m1_nhs = [
            _make_notehead_with_step(cx=100, cy=130, staff_idx=0, step=6,
                                     explicit_acc="sharp"),
            _make_notehead_with_step(cx=200, cy=130, staff_idx=0, step=6),
        ]
        m1_resolved = resolve_measure_accidentals(m1_nhs)
        assert m1_resolved[1]["resolved_accidental"] == "sharp"

        # Measure 2 (separate call → no shared state)
        m2_nhs = [
            _make_notehead_with_step(cx=300, cy=130, staff_idx=0, step=6),
        ]
        m2_resolved = resolve_measure_accidentals(m2_nhs)
        assert m2_resolved[0]["resolved_accidental"] is None, (
            "Accidental should NOT carry across measure boundary"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
