"""
Harmony Rules 規則檢查測試

使用 test-data/ 中的 JSON 測試案例進行參數化測試。
對應 Kotlin 的 HarmonyRuleEngineTest.kt。
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from harmony_rules import HarmonyAnalyzer, KeySignature
from tests.test_utils import (
    load_test_directory,
    parse_test_input,
    violations_match_expected
)


TEST_DATA_ROOT = Path(__file__).parent.parent / "test-data"


# ========== Fixtures ==========

@pytest.fixture(params=list((TEST_DATA_ROOT / "violations/M1_melodic_leap").glob("*.json")))
def m1_test_case(request):
    """M1 測試案例"""
    from tests.test_utils import load_test_case
    return load_test_case(request.param)


@pytest.fixture(params=list((TEST_DATA_ROOT / "violations/V1_voice_crossing").glob("*.json")))
def v1_test_case(request):
    """V1 測試案例"""
    from tests.test_utils import load_test_case
    return load_test_case(request.param)


@pytest.fixture(params=list((TEST_DATA_ROOT / "violations/P1_parallel_fifths").glob("*.json")))
def p1_test_case(request):
    """P1 測試案例"""
    from tests.test_utils import load_test_case
    return load_test_case(request.param)


@pytest.fixture(params=list((TEST_DATA_ROOT / "violations/P2_hidden_intervals").glob("*.json")))
def p2_test_case(request):
    """P2 測試案例"""
    from tests.test_utils import load_test_case
    return load_test_case(request.param)


@pytest.fixture(params=list((TEST_DATA_ROOT / "violations/D1_triad_doubling").glob("*.json")))
def d1_test_case(request):
    """D1 測試案例"""
    from tests.test_utils import load_test_case
    return load_test_case(request.param)


@pytest.fixture(params=list((TEST_DATA_ROOT / "violations/L1_leading_tone").glob("*.json")))
def l1_test_case(request):
    """L1 測試案例"""
    from tests.test_utils import load_test_case
    return load_test_case(request.param)


@pytest.fixture(params=list((TEST_DATA_ROOT / "valid").glob("*.json")))
def valid_test_case(request):
    """正確案例"""
    from tests.test_utils import load_test_case
    return load_test_case(request.param)


@pytest.fixture(params=list((TEST_DATA_ROOT / "integration").glob("*.json")))
def integration_test_case(request):
    """整合測試案例"""
    from tests.test_utils import load_test_case
    return load_test_case(request.param)


# ========== M1: Melodic Intervals ==========

@pytest.mark.rule
class TestM1MelodicIntervals:
    """M1: 旋律跳進限制測試"""

    def test_m1_violations(self, m1_test_case):
        """M1 規則應符合預期違規"""
        chords, key_sig = parse_test_input(m1_test_case)
        analyzer = HarmonyAnalyzer(chords, key_sig)

        violations = analyzer._check_melodic_intervals()

        m1_violations = [v for v in violations if v.rule_id == "M1"]
        expected = m1_test_case["expected_output"]

        assert len(m1_violations) == expected["violation_count"], \
            f"{m1_test_case['test_id']}: Expected {expected['violation_count']} violations, got {len(m1_violations)}"

        if expected["violation_count"] > 0:
            assert violations_match_expected(m1_violations, expected["violations"])


# ========== V1: Voice Crossing & Overlap ==========

@pytest.mark.rule
class TestV1VoiceCrossing:
    """V1: 聲部交錯與超越測試"""

    def test_v1_violations(self, v1_test_case):
        """V1 規則應符合預期違規"""
        chords, key_sig = parse_test_input(v1_test_case)
        analyzer = HarmonyAnalyzer(chords, key_sig)

        violations = analyzer._check_voice_crossing_and_overlap()

        v1_violations = [v for v in violations if v.rule_id == "V1"]
        expected = v1_test_case["expected_output"]

        assert len(v1_violations) == expected["violation_count"], \
            f"{v1_test_case['test_id']}: Expected {expected['violation_count']} violations, got {len(v1_violations)}"


# ========== P1: Parallel Fifths & Octaves ==========

@pytest.mark.rule
class TestP1ParallelIntervals:
    """P1: 平行八度與平行五度測試"""

    def test_p1_violations(self, p1_test_case):
        """P1 規則應符合預期違規"""
        chords, key_sig = parse_test_input(p1_test_case)
        analyzer = HarmonyAnalyzer(chords, key_sig)

        violations = analyzer._check_parallel_intervals()

        p1_violations = [v for v in violations if v.rule_id == "P1"]
        expected = p1_test_case["expected_output"]

        assert len(p1_violations) == expected["violation_count"], \
            f"{p1_test_case['test_id']}: Expected {expected['violation_count']} violations, got {len(p1_violations)}"


# ========== P2: Hidden Fifths & Octaves ==========

@pytest.mark.rule
class TestP2HiddenIntervals:
    """P2: 隱伏八度與隱伏五度測試"""

    def test_p2_violations(self, p2_test_case):
        """P2 規則應符合預期違規"""
        chords, key_sig = parse_test_input(p2_test_case)
        analyzer = HarmonyAnalyzer(chords, key_sig)

        violations = analyzer._check_hidden_intervals()

        p2_violations = [v for v in violations if v.rule_id == "P2"]
        expected = p2_test_case["expected_output"]

        assert len(p2_violations) == expected["violation_count"], \
            f"{p2_test_case['test_id']}: Expected {expected['violation_count']} violations, got {len(p2_violations)}"


# ========== D1: Triad Doubling ==========

@pytest.mark.rule
class TestD1TriadDoubling:
    """D1: 三和弦重複音與省略音測試"""

    def test_d1_violations(self, d1_test_case):
        """D1 規則應符合預期違規"""
        chords, key_sig = parse_test_input(d1_test_case)
        analyzer = HarmonyAnalyzer(chords, key_sig)

        violations = analyzer._check_triad_doubling()

        d1_violations = [v for v in violations if v.rule_id == "D1"]
        expected = d1_test_case["expected_output"]

        assert len(d1_violations) == expected["violation_count"], \
            f"{d1_test_case['test_id']}: Expected {expected['violation_count']} violations, got {len(d1_violations)}"


# ========== L1: Leading Tone Resolution ==========

@pytest.mark.rule
class TestL1LeadingTone:
    """L1: 導音解決測試"""

    def test_l1_violations(self, l1_test_case):
        """L1 規則應符合預期違規"""
        chords, key_sig = parse_test_input(l1_test_case)
        analyzer = HarmonyAnalyzer(chords, key_sig)

        violations = analyzer._check_leading_tone_resolution()

        l1_violations = [v for v in violations if v.rule_id == "L1"]
        expected = l1_test_case["expected_output"]

        assert len(l1_violations) == expected["violation_count"], \
            f"{l1_test_case['test_id']}: Expected {expected['violation_count']} violations, got {len(l1_violations)}"


# ========== Valid Cases ==========

@pytest.mark.rule
class TestValidProgressions:
    """正確進行測試（無違規）"""

    def test_valid_no_violations(self, valid_test_case):
        """正確進行應無違規"""
        chords, key_sig = parse_test_input(valid_test_case)
        analyzer = HarmonyAnalyzer(chords, key_sig)

        violations = analyzer.analyze()

        expected = valid_test_case["expected_output"]
        assert len(violations) == expected["violation_count"], \
            f"{valid_test_case['test_id']}: Expected no violations, got {len(violations)}: {[v.rule_id for v in violations]}"


# ========== Integration Tests ==========

@pytest.mark.integration
class TestIntegration:
    """整合測試（混合違規）"""

    def test_integration_mixed_violations(self, integration_test_case):
        """整合測試應檢測所有預期違規"""
        chords, key_sig = parse_test_input(integration_test_case)
        analyzer = HarmonyAnalyzer(chords, key_sig)

        violations = analyzer.analyze()

        expected = integration_test_case["expected_output"]
        assert len(violations) == expected["violation_count"], \
            f"{integration_test_case['test_id']}: Expected {expected['violation_count']} violations, got {len(violations)}"

        # 驗證每個預期違規都被找到
        assert violations_match_expected(violations, expected["violations"])


# ========== Specific Tests ==========

class TestSpecificCases:
    """特定測試案例"""

    def test_p1_001_parallel_fifth(self):
        """P1_001: S-A 平行五度"""
        from tests.test_utils import load_test_case
        test_case = load_test_case(TEST_DATA_ROOT / "violations/P1_parallel_fifths/P1_001_sa_parallel_fifth.json")
        chords, key_sig = parse_test_input(test_case)

        analyzer = HarmonyAnalyzer(chords, key_sig)
        violations = analyzer.analyze()

        assert len(violations) == 1
        assert violations[0].rule_id == "P1"
        assert violations[0].severity == "error"
        assert "平行" in violations[0].message_zh

    def test_valid_001_no_violations(self):
        """VALID_001: 簡單 I-V-I 進行（無錯誤）"""
        from tests.test_utils import load_test_case
        test_case = load_test_case(TEST_DATA_ROOT / "valid/VALID_001_simple_progression.json")
        chords, key_sig = parse_test_input(test_case)

        analyzer = HarmonyAnalyzer(chords, key_sig)
        violations = analyzer.analyze()

        assert len(violations) == 0

    def test_int_001_mixed_violations(self):
        """INT_001: 混合違規"""
        from tests.test_utils import load_test_case
        test_case = load_test_case(TEST_DATA_ROOT / "integration/INT_001_mixed_violations.json")
        chords, key_sig = parse_test_input(test_case)

        analyzer = HarmonyAnalyzer(chords, key_sig)
        violations = analyzer.analyze()

        assert len(violations) == 2
        assert any(v.rule_id == "M1" for v in violations)
        assert any(v.rule_id == "P1" for v in violations)


# ========== Cadence Tests ==========

class TestCadenceClassification:
    """終止式分類測試"""

    def test_pac_classification(self):
        """應正確分類完美正格終止"""
        from tests.test_utils import load_test_case
        test_case = load_test_case(TEST_DATA_ROOT / "valid/VALID_005_authentic_cadence.json")
        chords, key_sig = parse_test_input(test_case)

        analyzer = HarmonyAnalyzer(chords, key_sig)
        cadence = analyzer.classify_cadence()

        assert cadence == "PAC"

    def test_no_cadence(self):
        """非終止式進行應返回 None"""
        from tests.test_utils import load_test_case
        test_case = load_test_case(TEST_DATA_ROOT / "violations/P1_parallel_fifths/P1_001_sa_parallel_fifth.json")
        chords, key_sig = parse_test_input(test_case)

        analyzer = HarmonyAnalyzer(chords, key_sig)
        cadence = analyzer.classify_cadence()

        assert cadence is None


# ========== Edge Cases ==========

class TestEdgeCases:
    """邊界條件測試"""

    def test_empty_chord_list(self):
        """空和弦列表應無錯誤"""
        analyzer = HarmonyAnalyzer([], None)
        violations = analyzer.analyze()

        assert len(violations) == 0

    def test_single_chord(self):
        """單一和弦應只檢查 D1"""
        from harmony_rules import ChordSnapshot, NoteEvent

        chord = ChordSnapshot(
            index=0,
            measure=1,
            beat=1.0,
            notes={
                "S": NoteEvent("S", 72, 1, 1.0),
                "A": NoteEvent("A", 67, 1, 1.0),
                "T": NoteEvent("T", 64, 1, 1.0),
                "B": NoteEvent("B", 48, 1, 1.0),
            }
        )

        analyzer = HarmonyAnalyzer([chord], None)
        violations = analyzer.analyze()

        # 單一和弦只能觸發 D1 規則
        assert all(v.rule_id == "D1" for v in violations) or len(violations) == 0
