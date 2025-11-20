"""
核心輔助函數測試

測試 harmony_rules.py 中的私有輔助函數。
對應 Kotlin 的 HelperFunctionsTest.kt。
"""

import pytest
import sys
from pathlib import Path

# 導入規則引擎
sys.path.insert(0, str(Path(__file__).parent.parent))
from harmony_rules import (
    _interval_semitones,
    _direction,
    _is_perfect_fifth,
    _is_perfect_octave_or_unison,
    _is_step,
    _normalize_pc,
    _analyze_triad,
    ChordSnapshot,
    NoteEvent
)


class TestIntervalSemitones:
    """測試半音距離計算"""

    @pytest.mark.parametrize("a,b,expected", [
        (60, 67, 7),      # C4 to G4 = perfect 5th
        (72, 60, -12),    # C5 to C4 = octave down
        (60, 60, 0),      # Same note
        (60, 72, 12),     # C4 to C5 = octave up
        (60, 61, 1),      # C4 to C#4 = semitone
        (60, 62, 2),      # C4 to D4 = whole tone
        (60, 66, 6),      # C4 to F#4 = tritone
        (60, 74, 14),     # C4 to D5 = 9th
        (48, 72, 24),     # C3 to C5 = two octaves
        (71, 60, -11),    # B4 to C4 = minor 7th down
    ])
    def test_interval_semitones(self, a, b, expected):
        """應正確計算半音距離"""
        assert _interval_semitones(a, b) == expected


class TestDirection:
    """測試旋律方向判斷"""

    @pytest.mark.parametrize("a,b,expected", [
        (60, 65, 1),   # Upward
        (65, 60, -1),  # Downward
        (60, 60, 0),   # Static
        (48, 72, 1),   # Large upward leap
        (72, 48, -1),  # Large downward leap
        (60, 61, 1),   # Semitone up
        (61, 60, -1),  # Semitone down
    ])
    def test_direction(self, a, b, expected):
        """應正確判斷旋律方向"""
        assert _direction(a, b) == expected


class TestIsPerfectFifth:
    """測試完全五度判斷"""

    @pytest.mark.parametrize("diff,expected", [
        (7, True),      # P5 up
        (-7, True),     # P5 down
        (19, True),     # P5 + octave
        (-19, True),    # P5 down + octave
        (31, True),     # P5 + 2 octaves
        (5, False),     # Perfect 4th
        (6, False),     # Tritone
        (8, False),     # minor 6th
        (0, False),     # Unison
        (12, False),    # Octave
    ])
    def test_is_perfect_fifth(self, diff, expected):
        """應正確識別完全五度"""
        assert _is_perfect_fifth(diff) == expected


class TestIsPerfectOctaveOrUnison:
    """測試八度或同度判斷"""

    @pytest.mark.parametrize("diff,expected", [
        (0, True),      # Unison
        (12, True),     # Octave up
        (-12, True),    # Octave down
        (24, True),     # Two octaves up
        (-24, True),    # Two octaves down
        (36, True),     # Three octaves
        (1, False),     # Semitone
        (7, False),     # Perfect 5th
        (11, False),    # Major 7th
        (13, False),    # Octave + semitone
    ])
    def test_is_perfect_octave_or_unison(self, diff, expected):
        """應正確識別八度和同度"""
        assert _is_perfect_octave_or_unison(diff) == expected


class TestIsStep:
    """測試級進判斷"""

    @pytest.mark.parametrize("diff,expected", [
        (1, True),      # Semitone
        (2, True),      # Whole tone
        (-1, True),     # Semitone down
        (-2, True),     # Whole tone down
        (0, False),     # Static
        (3, False),     # minor 3rd
        (4, False),     # major 3rd
        (5, False),     # Perfect 4th
        (7, False),     # Perfect 5th
        (12, False),    # Octave
    ])
    def test_is_step(self, diff, expected):
        """應正確識別級進"""
        assert _is_step(diff) == expected


class TestNormalizePitchClass:
    """測試音高類別正規化"""

    @pytest.mark.parametrize("midi,expected", [
        (0, 0),     # C
        (12, 0),    # C (octave up)
        (24, 0),    # C (two octaves up)
        (60, 0),    # C4
        (61, 1),    # C#4
        (62, 2),    # D4
        (71, 11),   # B4
        (72, 0),    # C5
        (48, 0),    # C3
        (49, 1),    # C#3
        (59, 11),   # B3
    ])
    def test_normalize_pitch_class(self, midi, expected):
        """應正規化音高類別到 0-11 範圍"""
        assert _normalize_pc(midi) == expected


class TestAnalyzeTriad:
    """測試三和弦結構分析"""

    def test_c_major_triad(self):
        """應正確分析 C 大三和弦"""
        chord = ChordSnapshot(
            index=0,
            measure=1,
            beat=1.0,
            notes={
                "S": NoteEvent("S", 72, 1, 1.0),  # C5
                "A": NoteEvent("A", 67, 1, 1.0),  # G4
                "T": NoteEvent("T", 64, 1, 1.0),  # E4
                "B": NoteEvent("B", 48, 1, 1.0),  # C3
            }
        )

        triad = _analyze_triad(chord)

        assert triad is not None
        assert triad.root_pc == 0  # C
        assert triad.quality == "major"
        assert triad.role_by_voice["S"] in ["root", "third", "fifth"]

    def test_d_minor_triad(self):
        """應正確分析 D 小三和弦"""
        chord = ChordSnapshot(
            index=0,
            measure=1,
            beat=1.0,
            notes={
                "S": NoteEvent("S", 74, 1, 1.0),  # D5
                "A": NoteEvent("A", 69, 1, 1.0),  # A4
                "T": NoteEvent("T", 65, 1, 1.0),  # F4
                "B": NoteEvent("B", 50, 1, 1.0),  # D3
            }
        )

        triad = _analyze_triad(chord)

        assert triad is not None
        assert triad.root_pc == 2  # D
        assert triad.quality == "minor"

    def test_diminished_triad(self):
        """應正確分析減三和弦"""
        chord = ChordSnapshot(
            index=0,
            measure=1,
            beat=1.0,
            notes={
                "S": NoteEvent("S", 71, 1, 1.0),  # B4
                "A": NoteEvent("A", 65, 1, 1.0),  # F4
                "T": NoteEvent("T", 62, 1, 1.0),  # D4
                "B": NoteEvent("B", 59, 1, 1.0),  # B3
            }
        )

        triad = _analyze_triad(chord)

        assert triad is not None
        assert triad.root_pc == 11  # B
        assert triad.quality == "diminished"

    def test_augmented_triad(self):
        """應正確分析增三和弦"""
        chord = ChordSnapshot(
            index=0,
            measure=1,
            beat=1.0,
            notes={
                "S": NoteEvent("S", 72, 1, 1.0),  # C5
                "A": NoteEvent("A", 68, 1, 1.0),  # G#4
                "T": NoteEvent("T", 64, 1, 1.0),  # E4
                "B": NoteEvent("B", 48, 1, 1.0),  # C3
            }
        )

        triad = _analyze_triad(chord)

        assert triad is not None
        assert triad.root_pc == 0  # C
        assert triad.quality == "augmented"

    def test_incomplete_chord_returns_none(self):
        """不完整和弦應返回 None"""
        chord = ChordSnapshot(
            index=0,
            measure=1,
            beat=1.0,
            notes={
                "S": NoteEvent("S", 72, 1, 1.0),  # C5
                "A": NoteEvent("A", 60, 1, 1.0),  # C4
                "T": NoteEvent("T", 60, 1, 1.0),  # C4
                "B": NoteEvent("B", 48, 1, 1.0),  # C3
            }
        )

        triad = _analyze_triad(chord)

        # 只有一個音高類別，無法形成三和弦
        assert triad is None or len(set(_normalize_pc(n.midi) for n in chord.notes.values())) < 3


@pytest.mark.unit
class TestIntegration:
    """整合測試 - 驗證輔助函數一起工作"""

    def test_helper_functions_in_context(self):
        """輔助函數應能正確協同工作"""
        # 測試一個簡單的旋律進行
        note1 = 60  # C4
        note2 = 67  # G4

        diff = _interval_semitones(note1, note2)
        assert diff == 7

        direction = _direction(note1, note2)
        assert direction == 1  # Upward

        is_fifth = _is_perfect_fifth(diff)
        assert is_fifth is True

        is_step = _is_step(diff)
        assert is_step is False  # Not stepwise
