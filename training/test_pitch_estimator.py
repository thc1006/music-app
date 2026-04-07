"""
TDD Red: Pitch estimator tests.

Converts (notehead y-position, staff, clef, accidental) → MIDI pitch.

Music theory reference:
  Treble clef G4 on line 2 (from bottom, = step 6 from top)
  Treble clef F5 on line 5 (top, = step 0 from top)
  Treble clef E4 on line 1 (bottom, = step 8 from top)

  Bass clef F3 on line 4 (from bottom, = step 2 from top)
  Bass clef D3 on middle line (line 3, = step 4 from top)

  Diatonic scale (C major): C=0, D=2, E=4, F=5, G=7, A=9, B=11 (semitones)

  MIDI: C4 = 60, C#4 = 61, D4 = 62, ..., B4 = 71, C5 = 72
"""
import pytest
from pathlib import Path

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")


class TestModuleImport:
    def test_module_importable(self):
        import importlib.util
        spec = importlib.util.find_spec("pitch_estimator")
        assert spec is not None, "Create training/pitch_estimator.py"

    def test_estimate_pitch_exists(self):
        from pitch_estimator import estimate_pitch
        assert callable(estimate_pitch)


class TestTrebleClef:
    """High Clef: G4 is on line 2 from bottom (2nd line).

    Using step-from-top-line convention:
      step 0 = top line = F5 (MIDI 77)
      step 1 = top space = E5 (76)
      step 2 = 2nd line = D5 (74)
      step 3 = 2nd space = C5 (72)
      step 4 = middle line = B4 (71)
      step 5 = 2nd space from bottom = A4 (69)
      step 6 = 2nd line from bottom = G4 (67)
      step 7 = bottom space = F4 (65)
      step 8 = bottom line = E4 (64)
    """

    def _make_staff(self, top_y=100, spacing=10):
        from staff_detector import Staff
        line_ys = [top_y + i * spacing for i in range(5)]
        return Staff(line_ys=line_ys, spacing=spacing, x_min=0, x_max=500)

    def test_treble_top_line_is_F5(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff(top_y=100, spacing=10)
        midi = estimate_pitch(notehead_y=100, staff=staff, clef="treble")
        assert midi == 77, f"Top line should be F5 (77), got {midi}"

    def test_treble_middle_line_is_B4(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff(top_y=100, spacing=10)
        # Middle line (3rd from top) at y = 100 + 2*10 = 120
        midi = estimate_pitch(notehead_y=120, staff=staff, clef="treble")
        assert midi == 71, f"Middle line should be B4 (71), got {midi}"

    def test_treble_bottom_line_is_E4(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff(top_y=100, spacing=10)
        # Bottom line at y = 100 + 4*10 = 140
        midi = estimate_pitch(notehead_y=140, staff=staff, clef="treble")
        assert midi == 64, f"Bottom line should be E4 (64), got {midi}"

    def test_treble_second_line_from_bottom_is_G4(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff(top_y=100, spacing=10)
        # 2nd line from bottom = 4th line from top (step 6) at y = 100 + 3*10 = 130
        midi = estimate_pitch(notehead_y=130, staff=staff, clef="treble")
        assert midi == 67, f"G4 line should be 67, got {midi}"


class TestBassClef:
    """Bass clef: F3 on line 2 from top (step 2), D3 on middle line (step 4).

    Steps:
      step 0 = top line    = A3 (MIDI 57)
      step 2 = 2nd line    = F3 (53)
      step 4 = middle line = D3 (50)
      step 6 = 4th line    = B2 (47)
      step 8 = bottom line = G2 (43)
    """

    def _make_staff(self, top_y=200, spacing=10):
        from staff_detector import Staff
        line_ys = [top_y + i * spacing for i in range(5)]
        return Staff(line_ys=line_ys, spacing=spacing, x_min=0, x_max=500)

    def test_bass_top_line_is_A3(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff()
        midi = estimate_pitch(notehead_y=200, staff=staff, clef="bass")
        assert midi == 57, f"Bass top line should be A3 (57), got {midi}"

    def test_bass_middle_line_is_D3(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff()
        midi = estimate_pitch(notehead_y=220, staff=staff, clef="bass")
        assert midi == 50, f"Bass middle line should be D3 (50), got {midi}"

    def test_bass_bottom_line_is_G2(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff()
        midi = estimate_pitch(notehead_y=240, staff=staff, clef="bass")
        assert midi == 43, f"Bass bottom line should be G2 (43), got {midi}"


class TestAccidentals:

    def _make_staff(self, top_y=100, spacing=10):
        from staff_detector import Staff
        line_ys = [top_y + i * spacing for i in range(5)]
        return Staff(line_ys=line_ys, spacing=spacing, x_min=0, x_max=500)

    def test_sharp_raises_semitone(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff()
        # Middle line in treble = B4 = 71. B# = 72
        midi = estimate_pitch(notehead_y=120, staff=staff, clef="treble",
                              accidental="sharp")
        assert midi == 72, f"B# should be 72, got {midi}"

    def test_flat_lowers_semitone(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff()
        midi = estimate_pitch(notehead_y=120, staff=staff, clef="treble",
                              accidental="flat")
        assert midi == 70, f"Bb should be 70, got {midi}"

    def test_natural_cancels_accidental(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff()
        midi = estimate_pitch(notehead_y=120, staff=staff, clef="treble",
                              accidental="natural")
        assert midi == 71, f"B natural should be 71, got {midi}"

    def test_double_sharp_raises_two_semitones(self):
        from pitch_estimator import estimate_pitch
        staff = self._make_staff()
        midi = estimate_pitch(notehead_y=120, staff=staff, clef="treble",
                              accidental="double_sharp")
        assert midi == 73, f"Bx should be 73, got {midi}"


class TestLedgerLines:
    """Notes above/below staff (ledger lines)."""

    def _make_staff(self, top_y=100, spacing=10):
        from staff_detector import Staff
        line_ys = [top_y + i * spacing for i in range(5)]
        return Staff(line_ys=line_ys, spacing=spacing, x_min=0, x_max=500)

    def test_one_ledger_line_above_staff_treble(self):
        """First ledger line above staff = A5 (MIDI 81).

        At spacing=10, top line is at y=100.
        step -1 = top space = y=95 = G5
        step -2 = first ledger line above = y=90 = A5
        """
        from pitch_estimator import estimate_pitch
        staff = self._make_staff(top_y=100, spacing=10)
        midi = estimate_pitch(notehead_y=90, staff=staff, clef="treble")
        assert midi == 81, f"A5 ledger line should be 81, got {midi}"

    def test_middle_c_below_treble_staff(self):
        """Middle C (C4) is one ledger line below treble staff."""
        from pitch_estimator import estimate_pitch
        staff = self._make_staff(top_y=100, spacing=10)
        # Bottom line = E4 (step 8) at y=140
        # Below bottom line: step 9 = D4 (space), step 10 = C4 (ledger line)
        # y = 140 + spacing (= 10) = 150 → step 10
        midi = estimate_pitch(notehead_y=150, staff=staff, clef="treble")
        assert midi == 60, f"Middle C (C4) should be 60, got {midi}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
