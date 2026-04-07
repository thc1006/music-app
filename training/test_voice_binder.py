"""
TDD Red: Voice binder tests

Binds notehead detections to SATB voices based on:
- Horizontal alignment (same x = simultaneous)
- Staff membership (top staff = S/A, bottom = T/B)
- Vertical order within staff (upper = S or T, lower = A or B)
"""
import pytest


class TestModuleImport:
    def test_module_importable(self):
        import importlib.util
        spec = importlib.util.find_spec("voice_binder")
        assert spec is not None

    def test_bind_voices_exists(self):
        from voice_binder import bind_voices
        assert callable(bind_voices)


def _make_staff(top_y, spacing=10, x_min=0, x_max=1000):
    from staff_detector import Staff
    return Staff(
        line_ys=[top_y + i * spacing for i in range(5)],
        spacing=spacing,
        x_min=x_min,
        x_max=x_max,
    )


def _make_notehead(x, y, w=20, h=25):
    """Return a notehead detection dict matching downstream_eval format."""
    return {
        "class_id": 0,
        "class_name": "notehead_filled",
        "cx": float(x),
        "cy": float(y),
        "w": float(w),
        "h": float(h),
        "confidence": 0.9,
    }


class TestSingleStaffBinding:
    """Single staff with 4 simultaneous noteheads."""

    def test_four_noteheads_at_same_x_bind_to_SATB(self):
        """Top = S, next = A, next = T, bottom = B within one staff."""
        from voice_binder import bind_voices
        staff = _make_staff(top_y=100)
        noteheads = [
            _make_notehead(x=200, y=95),   # highest
            _make_notehead(x=200, y=110),
            _make_notehead(x=200, y=125),
            _make_notehead(x=200, y=140),  # lowest
        ]
        chords = bind_voices(noteheads, [staff])
        assert len(chords) == 1, f"Expected 1 chord, got {len(chords)}"
        chord = chords[0]
        assert "S" in chord and "A" in chord and "T" in chord and "B" in chord
        # S should have the smallest y (highest pitch)
        assert chord["S"]["cy"] < chord["A"]["cy"] < chord["T"]["cy"] < chord["B"]["cy"]


class TestMultiStaffBinding:
    """Two staves: top staff S+A, bottom staff T+B."""

    def test_two_staves_split_satb(self):
        from voice_binder import bind_voices
        top_staff = _make_staff(top_y=100, spacing=10)
        bot_staff = _make_staff(top_y=200, spacing=10)
        noteheads = [
            _make_notehead(x=300, y=110),  # top staff, upper
            _make_notehead(x=300, y=130),  # top staff, lower
            _make_notehead(x=300, y=210),  # bottom staff, upper
            _make_notehead(x=300, y=230),  # bottom staff, lower
        ]
        chords = bind_voices(noteheads, [top_staff, bot_staff])
        assert len(chords) == 1
        chord = chords[0]
        # Top staff upper = S (y=110), lower = A (y=130)
        assert chord["S"]["cy"] == 110
        assert chord["A"]["cy"] == 130
        # Bottom staff upper = T (y=210), lower = B (y=230)
        assert chord["T"]["cy"] == 210
        assert chord["B"]["cy"] == 230


class TestInsufficientNoteheads:
    """Fewer than 4 noteheads at same x → chord is skipped."""

    def test_three_noteheads_skipped(self):
        from voice_binder import bind_voices
        staff = _make_staff(top_y=100)
        noteheads = [
            _make_notehead(x=200, y=95),
            _make_notehead(x=200, y=115),
            _make_notehead(x=200, y=135),
        ]
        chords = bind_voices(noteheads, [staff])
        assert len(chords) == 0

    def test_no_noteheads_returns_empty(self):
        from voice_binder import bind_voices
        staff = _make_staff(top_y=100)
        chords = bind_voices([], [staff])
        assert chords == []


class TestHorizontalClustering:
    """Noteheads within x-tolerance are treated as simultaneous."""

    def test_noteheads_within_tolerance_are_chord(self):
        from voice_binder import bind_voices
        staff = _make_staff(top_y=100)
        noteheads = [
            _make_notehead(x=200, y=95),
            _make_notehead(x=205, y=110),  # slightly different x
            _make_notehead(x=195, y=125),
            _make_notehead(x=202, y=140),
        ]
        chords = bind_voices(noteheads, [staff])
        assert len(chords) == 1

    def test_noteheads_at_different_beats_are_separate_chords(self):
        from voice_binder import bind_voices
        staff = _make_staff(top_y=100)
        # Chord 1 at x=200
        # Chord 2 at x=400
        noteheads = [
            _make_notehead(x=200, y=95),
            _make_notehead(x=200, y=110),
            _make_notehead(x=200, y=125),
            _make_notehead(x=200, y=140),
            _make_notehead(x=400, y=95),
            _make_notehead(x=400, y=110),
            _make_notehead(x=400, y=125),
            _make_notehead(x=400, y=140),
        ]
        chords = bind_voices(noteheads, [staff])
        assert len(chords) == 2


class TestChordOrdering:
    """Chords should be ordered by x (left to right)."""

    def test_chords_sorted_by_x(self):
        from voice_binder import bind_voices
        staff = _make_staff(top_y=100)
        # Intentionally out-of-order input
        noteheads = []
        for x in [400, 200, 600]:
            for dy in [0, 15, 30, 45]:
                noteheads.append(_make_notehead(x=x, y=95 + dy))
        chords = bind_voices(noteheads, [staff])
        assert len(chords) == 3
        # Chords should be in ascending x order
        # Each chord's average x should be ascending
        avg_xs = [
            sum(c[v]["cx"] for v in ("S", "A", "T", "B")) / 4
            for c in chords
        ]
        assert avg_xs == sorted(avg_xs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
