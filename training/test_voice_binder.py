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
    """Single staff: NOT supported by C4-Lite (not 4-part harmony content).

    The C4 layout dispatcher returns [] for 1-stave layouts because they
    are not 4-part harmony. This test documents the intentional behavior.
    """

    def test_single_staff_returns_empty(self):
        """1-stave should return [] (not 4-part harmony)."""
        from voice_binder import bind_voices
        staff = _make_staff(top_y=100)
        noteheads = [
            _make_notehead(x=200, y=95),
            _make_notehead(x=200, y=110),
            _make_notehead(x=200, y=125),
            _make_notehead(x=200, y=140),
        ]
        chords = bind_voices(noteheads, [staff])
        assert chords == [], "1-stave should return [] in C4-Lite layout dispatcher"


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
        staves = [
            _make_staff(top_y=100, spacing=10),
            _make_staff(top_y=200, spacing=10),
        ]
        noteheads = [
            _make_notehead(x=200, y=95),
            _make_notehead(x=200, y=115),
            _make_notehead(x=200, y=135),
        ]
        chords = bind_voices(noteheads, staves)
        assert len(chords) == 0

    def test_no_noteheads_returns_empty(self):
        from voice_binder import bind_voices
        staves = [
            _make_staff(top_y=100, spacing=10),
            _make_staff(top_y=200, spacing=10),
        ]
        chords = bind_voices([], staves)
        assert chords == []


class TestHorizontalClustering:
    """Noteheads within x-tolerance are treated as simultaneous.

    Uses 2-stave grand staff layout (supported by C4-Lite).
    """

    def test_noteheads_within_tolerance_are_chord(self):
        from voice_binder import bind_voices
        top = _make_staff(top_y=100, spacing=10)
        bot = _make_staff(top_y=200, spacing=10)
        noteheads = [
            _make_notehead(x=300, y=110),  # top staff
            _make_notehead(x=305, y=130),  # top staff (slightly different x)
            _make_notehead(x=295, y=210),  # bot staff
            _make_notehead(x=302, y=230),  # bot staff
        ]
        chords = bind_voices(noteheads, [top, bot])
        assert len(chords) == 1

    def test_noteheads_at_different_beats_are_separate_chords(self):
        from voice_binder import bind_voices
        top = _make_staff(top_y=100, spacing=10)
        bot = _make_staff(top_y=200, spacing=10)
        # Chord 1 at x=200, chord 2 at x=400
        noteheads = []
        for x in [200, 400]:
            for y in [110, 130, 210, 230]:
                noteheads.append(_make_notehead(x=x, y=y))
        chords = bind_voices(noteheads, [top, bot])
        assert len(chords) == 2


class TestChordOrdering:
    """Chords should be ordered by x (left to right)."""

    def test_chords_sorted_by_x(self):
        from voice_binder import bind_voices
        top = _make_staff(top_y=100, spacing=10)
        bot = _make_staff(top_y=200, spacing=10)
        noteheads = []
        for x in [400, 200, 600]:
            for y in [110, 130, 210, 230]:
                noteheads.append(_make_notehead(x=x, y=y))
        chords = bind_voices(noteheads, [top, bot])
        assert len(chords) == 3
        avg_xs = [
            sum(c[v]["cx"] for v in ("S", "A", "T", "B")) / 4
            for c in chords
        ]
        assert avg_xs == sorted(avg_xs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
