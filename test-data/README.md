# 四部和聲規則引擎測試資料

本目錄包含用於驗證 Kotlin 與 Python 規則引擎行為一致性的共享測試資料。

## 目錄結構

```
test-data/
├── README.md                    # 本說明文件
├── valid/                       # 正確案例（無錯誤）
│   └── *.json
├── violations/                  # 違規案例（按規則分類）
│   ├── M1_melodic_leap/        # 旋律跳進限制
│   ├── V1_voice_crossing/      # 聲部交錯與超越
│   ├── P1_parallel_fifths/     # 平行八度與平行五度
│   ├── P2_hidden_intervals/    # 隱伏八度與隱伏五度
│   ├── D1_triad_doubling/      # 三和弦重複音與省略音
│   └── L1_leading_tone/        # 導音解決
└── integration/                 # 綜合測試（多種錯誤）
    └── *.json
```

## 測試案例格式

每個測試案例為一個 JSON 檔案，包含以下欄位：

### 基本結構

```jsonc
{
  "test_id": "規則ID_編號",
  "description": "測試案例說明（中文）",
  "category": "規則類別",
  "severity_expected": "error | warning",

  "input": {
    "key_signature": {         // 可選，調性資訊
      "tonic_midi": 60,        // 主音 MIDI 音高
      "mode": "major"          // "major" 或 "minor"
    },
    "chords": [                // 和弦序列
      {
        "index": 0,            // 和弦索引
        "measure": 1,          // 小節編號
        "beat": 1.0,           // 拍位置
        "notes": {             // 四個聲部的音符
          "S": {               // Soprano
            "voice": "S",
            "midi": 72,
            "measure": 1,
            "beat": 1.0
          },
          "A": { "voice": "A", "midi": 65, "measure": 1, "beat": 1.0 },
          "T": { "voice": "T", "midi": 60, "measure": 1, "beat": 1.0 },
          "B": { "voice": "B", "midi": 48, "measure": 1, "beat": 1.0 }
        }
      }
      // ... 更多和弦
    ]
  },

  "expected_output": {
    "violation_count": 1,      // 預期違規數量
    "violations": [
      {
        "rule_id": "P1",       // 規則 ID
        "severity": "error",   // "error" 或 "warning"
        "voices_involved": ["S", "A"],  // 涉及的聲部
        "location": {          // 錯誤位置
          "from_index": 0,
          "to_index": 1
        },
        "message_zh_contains": "平行完全五度"  // 中文訊息應包含的關鍵字
      }
    ],
    "cadence": null            // 預期終止式類型 (PAC/IAC/HC/PC/DC/null)
  },

  "notes": "補充說明或參考來源"
}
```

### 音高表示法

- **MIDI 音高**：C4 (Middle C) = 60
- **音高類別 (Pitch Class)**：0=C, 1=C#/Db, ..., 11=B

### 聲部代號

- **S**: Soprano (女高音)
- **A**: Alto (女低音)
- **T**: Tenor (男高音)
- **B**: Bass (男低音)

## 測試案例命名規則

檔名格式：`{rule_id}_{sequence}_{description}.json`

範例：
- `P1_001_sa_parallel_fifth.json` - S-A 平行五度
- `M1_002_octave_jump_error.json` - 超過八度跳進錯誤
- `D1_003_omit_root_error.json` - 省略根音錯誤

## 使用方式

### Kotlin 測試

```kotlin
import com.example.harmonychecker.TestDataLoader

val testCase = TestDataLoader.loadTestCase("test-data/violations/P1_parallel_fifths/P1_001_sa_parallel_fifth.json")
val result = HarmonyRuleEngine.analyze(testCase.input.chords, testCase.input.keySignature)

// 驗證結果
result.size shouldBe testCase.expectedOutput.violationCount
```

### Python 測試

```python
from tests.test_utils import load_test_case, parse_test_input

test_case = load_test_case("test-data/violations/P1_parallel_fifths/P1_001_sa_parallel_fifth.json")
chords, key_sig = parse_test_input(test_case)

analyzer = HarmonyAnalyzer(chords, key_sig)
result = analyzer.analyze()

# 驗證結果
assert len(result) == test_case["expected_output"]["violation_count"]
```

## 測試覆蓋目標

| 規則 ID | 目標案例數 | 當前進度 |
|---------|-----------|---------|
| M1 - 旋律跳進限制 | 8 | 0 |
| V1 - 聲部交錯與超越 | 6 | 0 |
| P1 - 平行八度與平行五度 | 8 | 0 |
| P2 - 隱伏八度與隱伏五度 | 6 | 0 |
| D1 - 三和弦重複音與省略音 | 10 | 0 |
| L1 - 導音解決 | 6 | 0 |
| 正確案例 | 5 | 0 |
| 整合測試 | 5 | 0 |
| **總計** | **54** | **0** |

## 新增測試案例指南

1. **選擇適當目錄**：根據規則類別放入對應資料夾
2. **遵循命名規則**：使用描述性檔名
3. **完整性檢查**：確保所有必要欄位都存在
4. **驗證 JSON 格式**：使用 JSON validator 檢查語法
5. **更新進度表**：在本 README 中更新測試覆蓋進度
6. **測試驗證**：在 Kotlin 與 Python 中都執行一次確保可讀取

## 測試資料來源

- 教科書範例：《和聲學總論》等經典教材
- 真實作業：音樂系學生作業（已匿名化）
- 人工構造：針對特定規則的極端案例

## 維護注意事項

- **不要修改已存在的測試案例**，除非發現明確的錯誤
- **新增案例時同步更新 Python 和 Kotlin 測試**
- **定期執行交叉驗證**，確保兩邊行為一致
- **保持測試資料與規則文件同步**（harmony_rules_zh.md）

## 相關文件

- [實作規劃文件](../docs/implementation_plan.md)
- [和聲規則說明](../harmony_rules_zh.md)
- [Python 規則引擎](../harmony_rules.py)
- [Kotlin 規則引擎](../android-app/app/src/main/java/com/example/harmonychecker/core/harmony/HarmonyModels.kt)
