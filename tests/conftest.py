"""
pytest 設定檔

提供全域 fixtures 與測試設定。
"""

import pytest
from pathlib import Path


# ========== 路徑相關 Fixtures ==========

@pytest.fixture
def project_root():
    """專案根目錄"""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root):
    """測試資料目錄"""
    return project_root / "test-data"


@pytest.fixture
def valid_test_dir(test_data_dir):
    """正確案例目錄"""
    return test_data_dir / "valid"


@pytest.fixture
def violations_test_dir(test_data_dir):
    """違規案例目錄"""
    return test_data_dir / "violations"


@pytest.fixture
def integration_test_dir(test_data_dir):
    """整合測試目錄"""
    return test_data_dir / "integration"


# ========== 測試資料 Fixtures ==========

@pytest.fixture
def sample_chord_c_major():
    """範例：C 大三和弦 (SATB)"""
    from harmony_rules import ChordSnapshot, NoteEvent

    return ChordSnapshot(
        index=0,
        measure=1,
        beat=1.0,
        notes={
            "S": NoteEvent(voice="S", midi=72, measure=1, beat=1.0),  # C5
            "A": NoteEvent(voice="A", midi=64, measure=1, beat=1.0),  # E4
            "T": NoteEvent(voice="T", midi=60, measure=1, beat=1.0),  # C4
            "B": NoteEvent(voice="B", midi=48, measure=1, beat=1.0),  # C3
        }
    )


@pytest.fixture
def sample_key_c_major():
    """範例：C 大調"""
    from harmony_rules import KeySignature
    return KeySignature(tonic_midi=60, mode="major")


# ========== pytest 配置 ==========

def pytest_configure(config):
    """pytest 初始化設定"""
    config.addinivalue_line(
        "markers",
        "unit: 單元測試（核心輔助函數）"
    )
    config.addinivalue_line(
        "markers",
        "rule: 規則檢查測試"
    )
    config.addinivalue_line(
        "markers",
        "integration: 整合測試"
    )
    config.addinivalue_line(
        "markers",
        "cross_validation: 交叉驗證測試（Kotlin vs Python）"
    )
    config.addinivalue_line(
        "markers",
        "slow: 執行較慢的測試"
    )
