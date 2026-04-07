"""
TDD Red: Phase C4 — voice_binder layout-aware tests.

Phase B's voice_binder treats all multi-staff cases the same way (top staff
boundary + bottom staff boundary). This is correct for 2-stave grand staff
but WRONG for:
  - 4-stave string quartet (each staff = one voice)
  - 8+ stave orchestral (not 4-part harmony at all)

C4-Lite goal:
  - 2 staves → grand staff: top=S/A (treble), bottom=T/B (bass)
  - 4 staves → each staff = one voice (S, A, T, B from top to bottom)
  - Other counts → return [] (not applicable to 4-part rule engine)
"""
import pytest


def _make_staff(top_y, spacing=16, x_min=0, x_max=1800):
    from staff_detector import Staff
    return Staff(
        line_ys=[top_y + i * spacing for i in range(5)],
        spacing=spacing,
        x_min=x_min,
        x_max=x_max,
    )


def _make_notehead(x, y, w=20, h=25):
    return {
        "class_id": 0,
        "class_name": "notehead_filled",
        "cx": float(x),
        "cy": float(y),
        "w": float(w),
        "h": float(h),
        "confidence": 0.9,
    }


# ──────────────────────────────────────────────────────────────────────
# 2-stave grand staff (existing behavior, should still work)
# ──────────────────────────────────────────────────────────────────────

class TestTwoStaveGrandStaff:
    """2 staves close together → grand staff: top staff = S/A, bottom = T/B."""

    def test_two_staves_split_satb(self):
        from voice_binder import bind_voices
        top = _make_staff(top_y=100, spacing=16)  # y 100-164
        bot = _make_staff(top_y=300, spacing=16)  # y 300-364

        noteheads = [
            _make_notehead(x=300, y=110),  # top staff upper
            _make_notehead(x=300, y=130),  # top staff lower
            _make_notehead(x=300, y=310),  # bot staff upper
            _make_notehead(x=300, y=330),  # bot staff lower
        ]
        chords = bind_voices(noteheads, [top, bot])
        assert len(chords) == 1
        assert chords[0]["S"]["cy"] == 110
        assert chords[0]["A"]["cy"] == 130
        assert chords[0]["T"]["cy"] == 310
        assert chords[0]["B"]["cy"] == 330


# ──────────────────────────────────────────────────────────────────────
# 4-stave: each staff = one voice (NEW behavior in C4)
# ──────────────────────────────────────────────────────────────────────

class TestFourStaveQuartet:
    """4 staves (string quartet) → S=Vln1, A=Vln2, T=Vla, B=Vc.
    Each notehead is assigned to its OWN staff, not to a top/bottom group."""

    def test_four_staves_each_one_voice(self):
        from voice_binder import bind_voices
        s1 = _make_staff(top_y=100, spacing=16)  # Vln 1 (S)
        s2 = _make_staff(top_y=300, spacing=16)  # Vln 2 (A)
        s3 = _make_staff(top_y=500, spacing=16)  # Vla   (T)
        s4 = _make_staff(top_y=700, spacing=16)  # Vc    (B)

        noteheads = [
            _make_notehead(x=300, y=130),  # in s1 → S
            _make_notehead(x=300, y=330),  # in s2 → A
            _make_notehead(x=300, y=530),  # in s3 → T
            _make_notehead(x=300, y=730),  # in s4 → B
        ]
        chords = bind_voices(noteheads, [s1, s2, s3, s4])
        assert len(chords) == 1
        assert chords[0]["S"]["cy"] == 130, "S should be the notehead in staff 1"
        assert chords[0]["A"]["cy"] == 330, "A should be the notehead in staff 2"
        assert chords[0]["T"]["cy"] == 530, "T should be the notehead in staff 3"
        assert chords[0]["B"]["cy"] == 730, "B should be the notehead in staff 4"

    def test_four_staves_chord_at_each_beat(self):
        """Multiple beats: each beat produces one chord."""
        from voice_binder import bind_voices
        staves = [
            _make_staff(top_y=100, spacing=16),
            _make_staff(top_y=300, spacing=16),
            _make_staff(top_y=500, spacing=16),
            _make_staff(top_y=700, spacing=16),
        ]
        noteheads = []
        for x in [200, 400, 600]:
            for staff_top in [100, 300, 500, 700]:
                noteheads.append(_make_notehead(x=x, y=staff_top + 30))
        chords = bind_voices(noteheads, staves)
        assert len(chords) == 3, f"Expected 3 chords, got {len(chords)}"

    def test_four_staves_skip_incomplete_beats(self):
        """If one staff is missing a notehead at a beat, that beat is not a full chord."""
        from voice_binder import bind_voices
        staves = [
            _make_staff(top_y=100, spacing=16),
            _make_staff(top_y=300, spacing=16),
            _make_staff(top_y=500, spacing=16),
            _make_staff(top_y=700, spacing=16),
        ]
        # Beat 1: only 3 staves have notes (s4 missing)
        noteheads = [
            _make_notehead(x=200, y=130),
            _make_notehead(x=200, y=330),
            _make_notehead(x=200, y=530),
            # No s4 notehead at x=200
        ]
        chords = bind_voices(noteheads, staves)
        assert len(chords) == 0, "Incomplete chord should be skipped"


# ──────────────────────────────────────────────────────────────────────
# Other layouts: skip (return [])
# ──────────────────────────────────────────────────────────────────────

class TestUnsupportedLayouts:
    """Layouts other than 2 or 4 staves are not 4-part harmony content."""

    def test_three_staves_returns_empty(self):
        """3 staves (lieder voice + piano grand staff) → not 4-part harmony."""
        from voice_binder import bind_voices
        staves = [
            _make_staff(top_y=100, spacing=16),
            _make_staff(top_y=300, spacing=16),
            _make_staff(top_y=500, spacing=16),
        ]
        noteheads = [
            _make_notehead(x=300, y=130),
            _make_notehead(x=300, y=330),
            _make_notehead(x=300, y=530),
            _make_notehead(x=300, y=550),  # extra
        ]
        chords = bind_voices(noteheads, staves)
        assert chords == [], (
            f"3-stave layout should return [], got {len(chords)} chords"
        )

    def test_orchestral_12_staves_returns_empty(self):
        """12-stave orchestral score is NOT 4-part harmony."""
        from voice_binder import bind_voices
        staves = [_make_staff(top_y=100 + i*200, spacing=16) for i in range(12)]
        noteheads = [
            _make_notehead(x=300, y=130 + i*200) for i in range(12)
        ]
        chords = bind_voices(noteheads, staves)
        assert chords == [], (
            f"Orchestral 12-stave should return [], got {len(chords)} chords"
        )

    def test_eight_staves_returns_empty(self):
        from voice_binder import bind_voices
        staves = [_make_staff(top_y=100 + i*200, spacing=16) for i in range(8)]
        noteheads = []
        for i in range(8):
            for x in [200, 400]:
                noteheads.append(_make_notehead(x=x, y=130 + i*200))
        chords = bind_voices(noteheads, staves)
        assert chords == []

    def test_one_staff_returns_empty_or_singlestaff(self):
        """1-staff: cannot have 4 voices on one staff in standard chorale layout.
        Implementation choice: skip these too."""
        from voice_binder import bind_voices
        staff = _make_staff(top_y=100, spacing=16)
        noteheads = [
            _make_notehead(x=200, y=110),
            _make_notehead(x=200, y=125),
            _make_notehead(x=200, y=140),
            _make_notehead(x=200, y=155),
        ]
        chords = bind_voices(noteheads, [staff])
        # We accept either: skip (preferred) OR keep existing single-staff behavior
        # This test documents the choice; implementation picks one.
        assert isinstance(chords, list)


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────

class TestLayoutEdgeCases:

    def test_no_staves_returns_empty(self):
        from voice_binder import bind_voices
        chords = bind_voices([], [])
        assert chords == []

    def test_two_staves_grand_staff_correctly_handled(self):
        """Specifically verify 2-stave doesn't accidentally fall into 'unsupported'."""
        from voice_binder import bind_voices
        s1 = _make_staff(top_y=100, spacing=16)
        s2 = _make_staff(top_y=300, spacing=16)
        nhs = [
            _make_notehead(x=200, y=110),
            _make_notehead(x=200, y=130),
            _make_notehead(x=200, y=310),
            _make_notehead(x=200, y=330),
        ]
        chords = bind_voices(nhs, [s1, s2])
        assert len(chords) >= 1, "2-stave grand staff should produce chord"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
