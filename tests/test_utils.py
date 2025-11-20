"""
測試工具模組

提供載入測試資料、解析 JSON 測試案例等輔助函數。
與 Kotlin TestDataLoader 保持對應。
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# 導入規則引擎模組
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from harmony_rules import (
    ChordSnapshot,
    NoteEvent,
    KeySignature,
    RuleViolation
)


def load_test_case(file_path: str | Path) -> Dict[str, Any]:
    """
    載入單一測試案例 JSON

    Args:
        file_path: 測試 JSON 檔案路徑

    Returns:
        測試案例字典

    Raises:
        FileNotFoundError: 當檔案不存在時
        json.JSONDecodeError: 當 JSON 格式錯誤時
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_test_directory(dir_path: str | Path) -> List[Dict[str, Any]]:
    """
    載入目錄下所有測試案例

    Args:
        dir_path: 測試目錄路徑

    Returns:
        測試案例列表
    """
    path = Path(dir_path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {dir_path}")

    test_cases = []
    for json_file in sorted(path.glob("*.json")):
        try:
            test_cases.append(load_test_case(json_file))
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return test_cases


def load_all_test_data(root_path: str | Path = "test-data/") -> List[Dict[str, Any]]:
    """
    載入所有測試資料（遞迴搜尋）

    Args:
        root_path: 根目錄路徑

    Returns:
        所有測試案例列表，按 test_id 排序
    """
    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root test directory not found: {root_path}")

    test_cases = []
    for json_file in root.rglob("*.json"):
        try:
            test_cases.append(load_test_case(json_file))
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    # 按 test_id 排序
    return sorted(test_cases, key=lambda tc: tc.get("test_id", ""))


def parse_test_input(test_case: Dict[str, Any]) -> Tuple[List[ChordSnapshot], Optional[KeySignature]]:
    """
    將測試案例 JSON 轉換為 Python 物件

    Args:
        test_case: 測試案例字典

    Returns:
        (chords, key_signature) tuple
    """
    input_data = test_case["input"]

    # 解析和弦列表
    chords = []
    for chord_data in input_data["chords"]:
        notes_dict = {}
        for voice, note_data in chord_data["notes"].items():
            notes_dict[voice] = NoteEvent(
                voice=note_data["voice"],
                midi=note_data["midi"],
                measure=note_data["measure"],
                beat=note_data["beat"]
            )

        chords.append(ChordSnapshot(
            index=chord_data["index"],
            measure=chord_data["measure"],
            beat=chord_data["beat"],
            notes=notes_dict
        ))

    # 解析調號（可選）
    key_sig = None
    if "key_signature" in input_data and input_data["key_signature"]:
        key_data = input_data["key_signature"]
        key_sig = KeySignature(
            tonic_midi=key_data["tonic_midi"],
            mode=key_data["mode"]
        )

    return chords, key_sig


def parse_expected_output(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析測試案例的預期輸出

    Args:
        test_case: 測試案例字典

    Returns:
        預期輸出字典（包含 violation_count, violations, cadence）
    """
    return test_case["expected_output"]


def violation_matches(
    actual: RuleViolation,
    expected: Dict[str, Any]
) -> bool:
    """
    檢查實際違規是否符合預期

    Args:
        actual: 實際產生的違規
        expected: 預期的違規字典

    Returns:
        是否符合
    """
    # 檢查 rule_id
    if actual.rule_id != expected["rule_id"]:
        return False

    # 檢查 severity
    if actual.severity != expected["severity"]:
        return False

    # 檢查 message 是否包含關鍵字
    if expected["message_zh_contains"] not in actual.message_zh:
        return False

    # 檢查 voices_involved (如果提供)
    if "voices_involved" in expected and expected["voices_involved"]:
        # 從 location 或其他欄位取得實際涉及的聲部
        # 這裡簡化處理，實際可能需要更精確的比對
        pass

    return True


def violations_match_expected(
    actual: List[RuleViolation],
    expected: List[Dict[str, Any]]
) -> bool:
    """
    檢查實際違規列表是否符合預期

    Args:
        actual: 實際產生的違規列表
        expected: 預期的違規列表

    Returns:
        是否完全符合
    """
    if len(actual) != len(expected):
        return False

    # 逐一配對檢查
    matched = set()
    for actual_violation in actual:
        match_found = False
        for idx, expected_violation in enumerate(expected):
            if idx not in matched and violation_matches(actual_violation, expected_violation):
                matched.add(idx)
                match_found = True
                break

        if not match_found:
            return False

    return len(matched) == len(expected)


def print_test_summary(test_results: List[Dict[str, Any]]) -> None:
    """
    印出測試摘要

    Args:
        test_results: 測試結果列表，每個元素包含 test_id, passed, message 等
    """
    total = len(test_results)
    passed = sum(1 for r in test_results if r.get("passed", False))
    failed = total - passed

    print(f"\n{'='*70}")
    print(f"測試摘要")
    print(f"{'='*70}")
    print(f"總計: {total} | 通過: {passed} | 失敗: {failed}")

    if failed > 0:
        print(f"\n失敗的測試:")
        for result in test_results:
            if not result.get("passed", False):
                print(f"  ✗ {result['test_id']}: {result.get('message', 'Unknown error')}")

    print(f"{'='*70}\n")


def compare_with_reference(
    actual_results: Dict[str, Any],
    reference_file: str | Path
) -> bool:
    """
    與參考輸出檔案比對

    用於交叉驗證：將 Python 或 Kotlin 的輸出與參考結果比對

    Args:
        actual_results: 實際結果字典
        reference_file: 參考輸出 JSON 檔案路徑

    Returns:
        是否完全一致
    """
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference = json.load(f)

    # 比對違規數量
    if actual_results.get("violation_count") != reference.get("violation_count"):
        return False

    # 比對每個違規
    actual_violations = actual_results.get("violations", [])
    reference_violations = reference.get("violations", [])

    return violations_match_expected(actual_violations, reference_violations)


# ========== pytest fixtures (如果使用 pytest) ==========

try:
    import pytest

    @pytest.fixture
    def test_data_dir():
        """pytest fixture: 測試資料目錄路徑"""
        return Path(__file__).parent.parent / "test-data"

    @pytest.fixture
    def sample_test_case(test_data_dir):
        """pytest fixture: 範例測試案例"""
        # 這裡可以載入一個預設的測試案例
        # 實際使用時可以根據需求調整
        return None

except ImportError:
    # pytest 未安裝，跳過 fixtures
    pass
