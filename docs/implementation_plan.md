# 深度實作規劃：單元測試 & OMR Client

本文件提供 **單元測試驗證** 與 **OMR Client JSON 處理** 的完整實作規劃。

---

## Part A：單元測試策略（Kotlin ↔ Python 行為一致性驗證）

### A.1 目標與原則

**主要目標：**
1. 驗證 Kotlin 實作與 Python 實作在**相同輸入下產生相同輸出**
2. 確保所有規則檢查邏輯 100% 對等
3. 建立可重複使用的測試資料庫，供未來擴充規則使用

**測試原則（遵循 CLAUDE.md）：**
- ✅ 測試必須有實質意義（測試行為，非實作細節）
- ✅ 使用**共享測試資料**（JSON 格式），雙語言讀取
- ✅ 先實作核心測試，再補充邊界條件
- ✅ 避免過度 mock（使用真實資料流）

---

### A.2 測試框架選擇

#### A.2.1 Kotlin 測試棧
```kotlin
// build.gradle.kts (app module)
dependencies {
    // JUnit 5 (Jupiter) - 現代化測試框架
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.1")
    testRuntimeOnly("org.junit.platform.runner:junit-platform-runner:1.10.1")

    // Kotest - Kotlin 原生斷言與匹配器
    testImplementation("io.kotest:kotest-assertions-core:5.8.0")

    // kotlinx.serialization - 讀取測試 JSON
    testImplementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
}

tasks.withType<Test> {
    useJUnitPlatform()
}
```

**為什麼選擇 JUnit 5 + Kotest？**
- JUnit 5：Android 標準，支援參數化測試、動態測試
- Kotest：提供 Kotlin 風格的 `shouldBe`、`shouldContain` 等斷言，程式碼更易讀

#### A.2.2 Python 測試棧
```python
# requirements-dev.txt
pytest>=7.4.0
pytest-json-report>=1.5.0  # 產生 JSON 格式測試報告
```

**為什麼選擇 pytest？**
- Python 社群標準
- 支援參數化測試（`@pytest.mark.parametrize`）
- 可輸出 JSON 格式報告，方便比對

---

### A.3 測試資料架構

#### A.3.1 目錄結構
```
music-app/
├── test-data/                          # 共享測試資料（Git 追蹤）
│   ├── README.md                       # 測試資料說明文件
│   ├── valid/                          # 正確案例（無錯誤）
│   │   ├── simple_cadence_pac.json
│   │   ├── four_voices_stepwise.json
│   │   └── ...
│   ├── violations/                     # 違規案例（分類）
│   │   ├── M1_melodic_leap/
│   │   │   ├── octave_jump_error.json
│   │   │   ├── seventh_leap_warning.json
│   │   │   └── tritone_error.json
│   │   ├── V1_voice_crossing/
│   │   │   ├── sa_crossing.json
│   │   │   └── overlap_consecutive.json
│   │   ├── P1_parallel_fifths/
│   │   │   ├── sa_parallel_fifth.json
│   │   │   └── tb_parallel_octave.json
│   │   ├── P2_hidden_intervals/
│   │   │   ├── hidden_fifth_no_step.json
│   │   │   └── hidden_octave_outer.json
│   │   ├── D1_triad_doubling/
│   │   │   ├── omit_root_error.json
│   │   │   ├── leading_tone_doubled.json
│   │   │   └── omit_fifth_insufficient_root.json
│   │   └── L1_leading_tone/
│   │       ├── lt_not_resolved.json
│   │       └── lt_downward_resolution.json
│   └── integration/                    # 綜合測試（多種錯誤）
│       ├── mixed_violations_1.json
│       └── real_world_exercise_1.json
├── android-app/
│   └── app/src/
│       ├── test/                       # Kotlin 單元測試
│       │   └── java/com/example/harmonychecker/
│       │       ├── core/harmony/
│       │       │   ├── HarmonyRuleEngineTest.kt
│       │       │   ├── HelperFunctionsTest.kt
│       │       │   └── CrossValidationTest.kt
│       │       └── TestDataLoader.kt   # 讀取 test-data/ 的工具
│       └── androidTest/                # Android 整合測試（之後）
└── tests/                              # Python 測試
    ├── test_harmony_rules.py           # Python 版規則測試
    ├── test_cross_validation.py        # 交叉驗證
    └── conftest.py                     # pytest 設定
```

#### A.3.2 測試資料格式（JSON Schema）

**範例：`test-data/violations/P1_parallel_fifths/sa_parallel_fifth.json`**
```jsonc
{
  "test_id": "P1_001",
  "description": "S-A 聲部連續平行完全五度",
  "category": "P1_parallel_intervals",
  "severity_expected": "error",

  "input": {
    "key_signature": {
      "tonic_midi": 60,    // C4
      "mode": "major"
    },
    "chords": [
      {
        "index": 0,
        "measure": 1,
        "beat": 1.0,
        "notes": {
          "S": { "voice": "S", "midi": 72, "measure": 1, "beat": 1.0 },  // C5
          "A": { "voice": "A", "midi": 65, "measure": 1, "beat": 1.0 },  // F4
          "T": { "voice": "T", "midi": 60, "measure": 1, "beat": 1.0 },  // C4
          "B": { "voice": "B", "midi": 48, "measure": 1, "beat": 1.0 }   // C3
        }
      },
      {
        "index": 1,
        "measure": 1,
        "beat": 2.0,
        "notes": {
          "S": { "voice": "S", "midi": 74, "measure": 1, "beat": 2.0 },  // D5
          "A": { "voice": "A", "midi": 67, "measure": 1, "beat": 2.0 },  // G4 (S-A = P5)
          "T": { "voice": "T", "midi": 62, "measure": 1, "beat": 2.0 },  // D4
          "B": { "voice": "B", "midi": 50, "measure": 1, "beat": 2.0 }   // D3
        }
      }
    ]
  },

  "expected_output": {
    "violation_count": 1,
    "violations": [
      {
        "rule_id": "P1",
        "severity": "error",
        "voices_involved": ["S", "A"],
        "location": {
          "from_index": 0,
          "to_index": 1
        },
        "message_zh_contains": "平行完全五度"
      }
    ],
    "cadence": null
  },

  "notes": "教科書經典錯誤範例：外聲部 S-A 連續五度"
}
```

**資料格式設計理由：**
1. **雙語言可讀**：純 JSON，Python 和 Kotlin 都能直接解析
2. **完整性**：包含輸入、預期輸出、metadata
3. **可維護性**：人類可讀，便於新增或修改測試案例
4. **可擴充性**：未來可加入 `audio_file`、`score_image` 等欄位

---

### A.4 測試分類與覆蓋率計畫

#### A.4.1 Phase 1：核心輔助函數單元測試

**目標檔案：** `HelperFunctionsTest.kt` / `test_helper_functions.py`

| 函數 | 測試案例數 | 範例 |
|------|-----------|------|
| `intervalSemitones` | 5 | (60, 67) → 7, (72, 60) → -12 |
| `direction` | 5 | (60, 65) → 1, (65, 60) → -1, (60, 60) → 0 |
| `isPerfectFifth` | 6 | 7 → true, -7 → true, 19 → true, 8 → false |
| `isPerfectOctaveOrUnison` | 6 | 0 → true, 12 → true, 7 → false |
| `isStep` | 6 | 1 → true, 2 → true, 3 → false |
| `normalizePitchClass` | 6 | 60 → 0, 61 → 1, 72 → 0 |
| `analyzeTriad` | 12 | major/minor/dim/aug, 重複音, 省略音 |

**測試方法：**
```kotlin
// Kotlin 範例
@Test
fun `intervalSemitones should calculate semitone distance`() {
    intervalSemitones(60, 67) shouldBe 7
    intervalSemitones(72, 60) shouldBe -12
    intervalSemitones(60, 60) shouldBe 0
}
```

```python
# Python 範例
def test_interval_semitones():
    assert _interval_semitones(60, 67) == 7
    assert _interval_semitones(72, 60) == -12
    assert _interval_semitones(60, 60) == 0
```

#### A.4.2 Phase 2：個別規則檢查測試

**目標檔案：** `HarmonyRuleEngineTest.kt` / `test_harmony_rules.py`

使用 **參數化測試** 讀取 `test-data/violations/` 中的 JSON 檔案：

```kotlin
// Kotlin 參數化測試（JUnit 5 + Kotest）
class M1MelodicIntervalsTest {
    companion object {
        @JvmStatic
        fun loadM1TestCases(): List<Arguments> {
            return loadTestDataFromDirectory("test-data/violations/M1_melodic_leap/")
                .map { Arguments.of(it) }
        }
    }

    @ParameterizedTest
    @MethodSource("loadM1TestCases")
    fun `M1 melodic intervals should match expected violations`(testCase: TestCase) {
        val result = HarmonyRuleEngine.checkMelodicIntervals(testCase.input.chords)

        result.size shouldBe testCase.expected.violationCount
        result.first().ruleId shouldBe "M1"
        result.first().messageZh should include(testCase.expected.violations[0].messageZhContains)
    }
}
```

```python
# Python 參數化測試
import pytest
from pathlib import Path

@pytest.mark.parametrize("test_file", Path("test-data/violations/M1_melodic_leap/").glob("*.json"))
def test_m1_melodic_intervals(test_file):
    test_case = load_test_case(test_file)
    analyzer = HarmonyAnalyzer(test_case["input"]["chords"], test_case["input"].get("key_signature"))

    result = analyzer._check_melodic_intervals()

    assert len(result) == test_case["expected_output"]["violation_count"]
    assert result[0].rule_id == "M1"
    assert test_case["expected_output"]["violations"][0]["message_zh_contains"] in result[0].message_zh
```

**覆蓋計畫（每個規則至少 5 個測試案例）：**

| 規則 ID | 測試案例數量 | 涵蓋情境 |
|---------|-------------|---------|
| M1 | 8 | 超過八度、七度跳進、增四度、內聲部限制、正確案例 |
| V1 | 6 | S-A 交錯、A-T 交錯、overlap、正確案例 |
| P1 | 8 | S-A/S-T/S-B/A-T/A-B/T-B 各一組平行五/八度 |
| P2 | 6 | 外聲部隱伏五/八度、例外（S級進）、正確案例 |
| D1 | 10 | 省略根音、省略三音、省略五音、導音重複、正確重複 |
| L1 | 6 | 導音未解決、導音下行、導音在內聲部、正確解決 |

**總計：44 個基礎測試案例**

#### A.4.3 Phase 3：交叉驗證測試（Cross-Validation）

**目標：** 確保 Kotlin 與 Python **完全相同的輸出**

**實作方式：**
1. Python 先執行所有測試案例，產生 JSON 報告（`python_results.json`）
2. Kotlin 執行相同測試案例，產生 JSON 報告（`kotlin_results.json`）
3. 比對兩個 JSON 檔案：
   - 違規數量
   - 違規 rule_id
   - 違規位置（from_index, to_index）
   - 嚴重性（error/warning）

**Python 產生參考輸出：**
```bash
# 執行 Python 測試並輸出 JSON
pytest tests/ --json-report --json-report-file=python_results.json
```

**Kotlin 讀取參考輸出並驗證：**
```kotlin
@Test
fun `cross validation - Kotlin should match Python output`() {
    val pythonResults = loadJsonReport("python_results.json")
    val testCases = loadAllTestData("test-data/")

    testCases.forEach { testCase ->
        val kotlinResult = HarmonyRuleEngine.analyze(testCase.input.chords, testCase.input.keySignature)
        val pythonResult = pythonResults.findResult(testCase.testId)

        // 比對違規數量
        kotlinResult.size shouldBe pythonResult.size

        // 比對每個違規的細節
        kotlinResult.zip(pythonResult).forEach { (k, p) ->
            k.ruleId shouldBe p.ruleId
            k.severity shouldBe p.severity
            k.location shouldBe p.location
        }
    }
}
```

#### A.4.4 Phase 4：整合測試與邊界條件

**整合測試案例（`test-data/integration/`）：**
1. `mixed_violations_1.json` - 同時違反 M1, P1, D1 三條規則
2. `real_world_exercise_1.json` - 真實學生作業（8 小節，20 個和弦）
3. `edge_case_empty.json` - 空輸入
4. `edge_case_single_chord.json` - 單一和弦（無法檢查連續性規則）
5. `edge_case_missing_voice.json` - 缺少某個聲部

**邊界條件測試：**
```kotlin
@Test
fun `should handle empty chord list`() {
    val result = HarmonyRuleEngine.analyze(emptyList())
    result shouldBe emptyList()
}

@Test
fun `should handle single chord`() {
    val result = HarmonyRuleEngine.analyze(listOf(singleChord))
    // 只能檢查 D1（三和弦重複），無法檢查 M1/V1/P1/P2/L1
    result.all { it.ruleId == "D1" } shouldBe true
}
```

---

### A.5 測試工具實作

#### A.5.1 TestDataLoader.kt
```kotlin
package com.example.harmonychecker

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
data class TestCase(
    val testId: String,
    val description: String,
    val category: String,
    val input: TestInput,
    val expectedOutput: TestExpectedOutput
)

@Serializable
data class TestInput(
    val keySignature: KeySignature? = null,
    val chords: List<ChordSnapshot>
)

@Serializable
data class TestExpectedOutput(
    val violationCount: Int,
    val violations: List<ExpectedViolation>,
    val cadence: String? = null
)

@Serializable
data class ExpectedViolation(
    val ruleId: String,
    val severity: String,
    val voicesInvolved: List<String>? = null,
    val location: Map<String, Int>,
    val messageZhContains: String
)

object TestDataLoader {
    private val json = Json {
        ignoreUnknownKeys = true
        prettyPrint = true
    }

    fun loadTestCase(filePath: String): TestCase {
        val content = File(filePath).readText()
        return json.decodeFromString<TestCase>(content)
    }

    fun loadTestDataFromDirectory(dirPath: String): List<TestCase> {
        return File(dirPath)
            .listFiles { file -> file.extension == "json" }
            ?.map { loadTestCase(it.absolutePath) }
            ?: emptyList()
    }
}
```

#### A.5.2 Python test_utils.py
```python
import json
from pathlib import Path
from typing import List, Dict, Any

def load_test_case(file_path: Path) -> Dict[str, Any]:
    """載入單一測試案例 JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_test_directory(dir_path: str) -> List[Dict[str, Any]]:
    """載入目錄下所有測試案例"""
    return [
        load_test_case(p)
        for p in Path(dir_path).glob("*.json")
    ]

def parse_test_input(test_case: Dict) -> Tuple[List[ChordSnapshot], Optional[KeySignature]]:
    """將測試案例 JSON 轉成 Python 物件"""
    input_data = test_case["input"]

    chords = [
        ChordSnapshot(
            index=c["index"],
            measure=c["measure"],
            beat=c["beat"],
            notes={
                voice: NoteEvent(**note_data)
                for voice, note_data in c["notes"].items()
            }
        )
        for c in input_data["chords"]
    ]

    key_sig = None
    if "key_signature" in input_data and input_data["key_signature"]:
        key_sig = KeySignature(**input_data["key_signature"])

    return chords, key_sig
```

---

### A.6 實作階段規劃

#### Phase 1：基礎設施建置（1 天）
- [ ] 建立 `test-data/` 目錄結構
- [ ] 撰寫 `TestDataLoader.kt` 與 `test_utils.py`
- [ ] 設定 Gradle 測試依賴（JUnit 5 + Kotest）
- [ ] 設定 Python `pytest` 與 `requirements-dev.txt`
- [ ] 建立 5 個基礎測試案例 JSON（每個規則類別一個）

#### Phase 2：核心函數測試（1 天）
- [ ] 實作 `HelperFunctionsTest.kt`（40+ 測試）
- [ ] 實作 `test_helper_functions.py`（40+ 測試）
- [ ] 確保兩邊測試全部通過

#### Phase 3：規則檢查測試（2-3 天）
- [ ] 建立 44 個分類測試案例 JSON（M1/V1/P1/P2/D1/L1）
- [ ] 實作 `HarmonyRuleEngineTest.kt`（參數化測試）
- [ ] 實作 `test_harmony_rules.py`（參數化測試）
- [ ] 修正發現的 bug（Kotlin 或 Python）

#### Phase 4：交叉驗證（1 天）
- [ ] 產生 Python 參考輸出 JSON
- [ ] 實作 `CrossValidationTest.kt`
- [ ] 確保 100% 一致性

#### Phase 5：整合測試與文件（1 天）
- [ ] 建立 5 個整合測試案例
- [ ] 邊界條件測試
- [ ] 撰寫 `test-data/README.md`
- [ ] 更新專案主 README

**總預估時間：6-7 天**

---

## Part B：OMR Client JSON 序列化/反序列化

### B.1 目標與架構

**主要目標：**
1. 實作 Android 端 OMR API 請求/回應的完整 JSON 處理
2. 將 API 回應的 JSON 轉換為 `List<ChordSnapshot>` 供規則引擎使用
3. 支援錯誤處理與警告訊息

**架構設計（三層）：**

```
┌─────────────────────────────────────────────────────────────┐
│                      UI Layer (Composable)                  │
│  - MainScreen, CameraScreen, ResultScreen                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Domain Layer (Use Cases)                   │
│  - RecognizeScoreUseCase                                    │
│  - AnalyzeHarmonyUseCase                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer (本次重點)                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  OmrApi (Retrofit Interface)                           │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  OmrRequestDto / OmrResponseDto                        │ │
│  │  (kotlinx.serialization @Serializable)                │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  OmrResponseMapper                                     │ │
│  │  (DTO → List<ChordSnapshot>)                          │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  HttpOmrClient (Implementation of OmrClient)           │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

### B.2 依賴項與設定

#### B.2.1 Gradle 依賴（已存在，需確認版本）

```kotlin
// android-app/app/build.gradle.kts
dependencies {
    // Retrofit
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.jakewharton.retrofit:retrofit2-kotlinx-serialization-converter:1.0.0")

    // kotlinx.serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")

    // OkHttp (Logging Interceptor for debug)
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

#### B.2.2 序列化插件（已存在，需確認）

```kotlin
// android-app/build.gradle.kts
plugins {
    id("org.jetbrains.kotlin.plugin.serialization") version "1.9.20" apply false
}

// android-app/app/build.gradle.kts
plugins {
    id("org.jetbrains.kotlin.plugin.serialization")
}
```

---

### B.3 資料模型設計（DTO Layer）

#### B.3.1 Request DTO

**檔案位置：** `android-app/app/src/main/java/com/example/harmonychecker/data/omr/dto/OmrRequestDto.kt`

```kotlin
package com.example.harmonychecker.data.omr.dto

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class OmrRequestDto(
    @SerialName("image_base64")
    val imageBase64: String,

    @SerialName("filename")
    val filename: String? = null,

    @SerialName("options")
    val options: OmrRequestOptions? = null
)

@Serializable
data class OmrRequestOptions(
    @SerialName("staff_layout")
    val staffLayout: StaffLayout = StaffLayout.UNKNOWN,

    @SerialName("expected_voices")
    val expectedVoices: List<String> = listOf("S", "A", "T", "B"),

    @SerialName("language_hint")
    val languageHint: String = "zh-TW"
)

@Serializable
enum class StaffLayout {
    @SerialName("SATB_GRAND_STAFF")
    SATB_GRAND_STAFF,

    @SerialName("PIANO_REDUCTION")
    PIANO_REDUCTION,

    @SerialName("UNKNOWN")
    UNKNOWN
}
```

**設計考量：**
- 使用 `@SerialName` 對應 API 規格的 snake_case
- `filename` 與 `options` 為可選（符合 API 設計）
- 預設值符合最常見使用情境

#### B.3.2 Response DTO

**檔案位置：** `android-app/app/src/main/java/com/example/harmonychecker/data/omr/dto/OmrResponseDto.kt`

```kotlin
package com.example.harmonychecker.data.omr.dto

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class OmrResponseDto(
    @SerialName("measures")
    val measures: List<MeasureDto>,

    @SerialName("raw_model_output")
    val rawModelOutput: String? = null,

    @SerialName("warnings")
    val warnings: List<String> = emptyList(),

    @SerialName("error")
    val error: OmrErrorDto? = null
)

@Serializable
data class MeasureDto(
    @SerialName("index")
    val index: Int,

    @SerialName("time_signature")
    val timeSignature: String? = null,

    @SerialName("key_signature")
    val keySignature: String? = null,

    @SerialName("chords")
    val chords: List<ChordDto>
)

@Serializable
data class ChordDto(
    @SerialName("beat")
    val beat: Double,

    @SerialName("notes")
    val notes: List<NoteDto>
)

@Serializable
data class NoteDto(
    @SerialName("voice")
    val voice: String,

    @SerialName("pitch")
    val pitch: String,  // e.g., "C4", "F#3"

    @SerialName("duration")
    val duration: Double,

    @SerialName("tie")
    val tie: TieType = TieType.NONE
)

@Serializable
enum class TieType {
    @SerialName("NONE")
    NONE,

    @SerialName("START")
    START,

    @SerialName("STOP")
    STOP,

    @SerialName("CONTINUE")
    CONTINUE
}

@Serializable
data class OmrErrorDto(
    @SerialName("code")
    val code: String,

    @SerialName("message")
    val message: String
)
```

**設計考量：**
- 完整對應 `docs/omr_proxy_api.md` 的 Response Schema
- 所有可選欄位都有預設值
- 使用 `enum class` 管理固定值（TieType）

---

### B.4 DTO → Domain 轉換層（Mapper）

#### B.4.1 OmrResponseMapper

**檔案位置：** `android-app/app/src/main/java/com/example/harmonychecker/data/omr/mapper/OmrResponseMapper.kt`

```kotlin
package com.example.harmonychecker.data.omr.mapper

import com.example.harmonychecker.core.harmony.*
import com.example.harmonychecker.data.omr.dto.*

/**
 * 將 OMR API Response DTO 轉換為 Domain 層的 ChordSnapshot 列表。
 *
 * 轉換規則：
 * 1. 展平所有小節的 chords 成單一列表
 * 2. 將 pitch string (e.g., "C4") 轉換為 MIDI 數字
 * 3. 將 notes 從 List 轉為 Map<Voice, NoteEvent>
 * 4. 過濾掉不完整的 chord（缺少 S/A/T/B 任一聲部）
 */
object OmrResponseMapper {

    /**
     * 主要轉換函數
     */
    fun toDomain(response: OmrResponseDto): OmrMappingResult {
        val warnings = mutableListOf<String>()
        warnings.addAll(response.warnings)

        if (response.error != null) {
            return OmrMappingResult(
                chords = emptyList(),
                keySignature = null,
                warnings = listOf("API Error: ${response.error.message}"),
                rawJson = response.rawModelOutput
            )
        }

        val chords = mutableListOf<ChordSnapshot>()
        var globalChordIndex = 0
        var keySignature: KeySignature? = null

        for (measureDto in response.measures) {
            // 提取調號（取第一個非空調號）
            if (keySignature == null && measureDto.keySignature != null) {
                keySignature = parseKeySignature(measureDto.keySignature)
            }

            for (chordDto in measureDto.chords) {
                val chordResult = toChordSnapshot(
                    chordDto = chordDto,
                    measureIndex = measureDto.index,
                    globalIndex = globalChordIndex
                )

                when (chordResult) {
                    is ChordMappingResult.Success -> {
                        chords.add(chordResult.chord)
                        globalChordIndex++
                    }
                    is ChordMappingResult.Warning -> {
                        warnings.add(chordResult.message)
                    }
                }
            }
        }

        return OmrMappingResult(
            chords = chords,
            keySignature = keySignature,
            warnings = warnings,
            rawJson = response.rawModelOutput
        )
    }

    /**
     * 將單一 ChordDto 轉為 ChordSnapshot
     */
    private fun toChordSnapshot(
        chordDto: ChordDto,
        measureIndex: Int,
        globalIndex: Int
    ): ChordMappingResult {
        val notesByVoice = mutableMapOf<Voice, NoteEvent>()

        for (noteDto in chordDto.notes) {
            val voice = try {
                Voice.fromString(noteDto.voice)
            } catch (e: IllegalArgumentException) {
                return ChordMappingResult.Warning(
                    "Unknown voice '${noteDto.voice}' at measure $measureIndex, beat ${chordDto.beat}"
                )
            }

            val midi = try {
                pitchToMidi(noteDto.pitch)
            } catch (e: IllegalArgumentException) {
                return ChordMappingResult.Warning(
                    "Invalid pitch '${noteDto.pitch}' for voice $voice at measure $measureIndex"
                )
            }

            notesByVoice[voice] = NoteEvent(
                voice = voice,
                midi = midi,
                measure = measureIndex,
                beat = chordDto.beat
            )
        }

        // 檢查是否四個聲部齊全
        val missingVoices = Voice.values().filter { it !in notesByVoice.keys }
        if (missingVoices.isNotEmpty()) {
            return ChordMappingResult.Warning(
                "Incomplete chord at measure $measureIndex, beat ${chordDto.beat}: missing ${missingVoices.joinToString()}"
            )
        }

        return ChordMappingResult.Success(
            ChordSnapshot(
                index = globalIndex,
                measure = measureIndex,
                beat = chordDto.beat,
                notes = notesByVoice
            )
        )
    }

    /**
     * 將音高字串（如 "C4", "F#3"）轉為 MIDI 數字
     *
     * 規則：C4 = 60 (middle C)
     * 八度從 C 開始（C0, C1, ..., C8）
     */
    fun pitchToMidi(pitch: String): Int {
        if (pitch.length < 2) {
            throw IllegalArgumentException("Invalid pitch format: $pitch")
        }

        val noteNamePart = pitch.dropLast(1)  // "C#"
        val octavePart = pitch.last()          // '4'

        val octave = octavePart.digitToIntOrNull()
            ?: throw IllegalArgumentException("Invalid octave in pitch: $pitch")

        val pitchClass = when (noteNamePart) {
            "C" -> 0
            "C#", "Db" -> 1
            "D" -> 2
            "D#", "Eb" -> 3
            "E" -> 4
            "F" -> 5
            "F#", "Gb" -> 6
            "G" -> 7
            "G#", "Ab" -> 8
            "A" -> 9
            "A#", "Bb" -> 10
            "B" -> 11
            else -> throw IllegalArgumentException("Unknown note name: $noteNamePart")
        }

        return (octave + 1) * 12 + pitchClass  // C4 = (4+1)*12 + 0 = 60
    }

    /**
     * 解析調號字串（如 "C major", "a minor"）
     */
    fun parseKeySignature(keyString: String): KeySignature? {
        val parts = keyString.trim().split(" ")
        if (parts.size != 2) return null

        val (tonicStr, modeStr) = parts

        val tonicPc = when (tonicStr.uppercase()) {
            "C" -> 0
            "C#", "DB" -> 1
            "D" -> 2
            "D#", "EB" -> 3
            "E" -> 4
            "F" -> 5
            "F#", "GB" -> 6
            "G" -> 7
            "G#", "AB" -> 8
            "A" -> 9
            "A#", "BB" -> 10
            "B" -> 11
            else -> return null
        }

        val mode = when (modeStr.lowercase()) {
            "major" -> KeyMode.MAJOR
            "minor" -> KeyMode.MINOR
            else -> return null
        }

        // 計算 tonic MIDI（預設 C4 = 60 為基準）
        val tonicMidi = 60 + tonicPc

        return KeySignature(tonicMidi = tonicMidi, mode = mode)
    }
}

/**
 * Mapper 內部結果型別
 */
private sealed class ChordMappingResult {
    data class Success(val chord: ChordSnapshot) : ChordMappingResult()
    data class Warning(val message: String) : ChordMappingResult()
}

/**
 * Mapper 最終輸出結果
 */
data class OmrMappingResult(
    val chords: List<ChordSnapshot>,
    val keySignature: KeySignature?,
    val warnings: List<String>,
    val rawJson: String?
)
```

**設計亮點：**
1. **防禦性解析**：每個可能失敗的轉換都有錯誤處理
2. **警告收集**：不完整的 chord 不會導致整體失敗，而是記錄警告
3. **音高轉換**：支援升降記號（C#/Db 等）
4. **全域索引**：展平所有小節後重新編號 `index`

---

### B.5 Retrofit 介面定義

#### B.5.1 OmrApi Interface

**檔案位置：** `android-app/app/src/main/java/com/example/harmonychecker/data/omr/api/OmrApi.kt`

```kotlin
package com.example.harmonychecker.data.omr.api

import com.example.harmonychecker.data.omr.dto.OmrRequestDto
import com.example.harmonychecker.data.omr.dto.OmrResponseDto
import retrofit2.http.Body
import retrofit2.http.POST

/**
 * OMR 雲端 API Retrofit 介面
 *
 * 對應 docs/omr_proxy_api.md 的端點設計
 */
interface OmrApi {

    @POST("/api/omr/score")
    suspend fun recognizeScore(
        @Body request: OmrRequestDto
    ): OmrResponseDto
}
```

#### B.5.2 Retrofit Client 建立

**檔案位置：** `android-app/app/src/main/java/com/example/harmonychecker/data/omr/api/OmrApiClient.kt`

```kotlin
package com.example.harmonychecker.data.omr.api

import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit

/**
 * OMR API Retrofit Client Factory
 */
object OmrApiClient {

    private val json = Json {
        ignoreUnknownKeys = true  // 容錯：忽略 API 新增的未知欄位
        isLenient = true
        encodeDefaults = true
    }

    /**
     * 建立 OmrApi 實例
     *
     * @param baseUrl API 基礎 URL（例如：https://your-cloud-function.com）
     * @param apiKey 可選的 API key（若後端需要）
     * @param enableLogging 是否啟用 HTTP 請求日誌（Debug 用）
     */
    fun create(
        baseUrl: String,
        apiKey: String? = null,
        enableLogging: Boolean = false
    ): OmrApi {
        val okHttpClient = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)  // OMR 可能耗時較長
            .writeTimeout(30, TimeUnit.SECONDS)
            .apply {
                // API Key Interceptor
                if (apiKey != null) {
                    addInterceptor { chain ->
                        val request = chain.request().newBuilder()
                            .addHeader("X-API-Key", apiKey)
                            .build()
                        chain.proceed(request)
                    }
                }

                // Logging Interceptor (Debug only)
                if (enableLogging) {
                    val loggingInterceptor = HttpLoggingInterceptor().apply {
                        level = HttpLoggingInterceptor.Level.BODY
                    }
                    addInterceptor(loggingInterceptor)
                }
            }
            .build()

        val contentType = "application/json".toMediaType()

        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(json.asConverterFactory(contentType))
            .build()
            .create(OmrApi::class.java)
    }
}
```

---

### B.6 OmrClient 完整實作

#### B.6.1 更新 OmrClient.kt

**檔案位置：** `android-app/app/src/main/java/com/example/harmonychecker/core/omr/OmrClient.kt`

```kotlin
package com.example.harmonychecker.core.omr

import android.util.Base64
import com.example.harmonychecker.core.harmony.ChordSnapshot
import com.example.harmonychecker.core.harmony.KeySignature
import com.example.harmonychecker.data.omr.api.OmrApi
import com.example.harmonychecker.data.omr.api.OmrApiClient
import com.example.harmonychecker.data.omr.dto.OmrRequestDto
import com.example.harmonychecker.data.omr.dto.OmrRequestOptions
import com.example.harmonychecker.data.omr.dto.StaffLayout
import com.example.harmonychecker.data.omr.mapper.OmrResponseMapper

/**
 * OMR API 回傳結果
 */
data class OmrResult(
    val chords: List<ChordSnapshot>,
    val keySignature: KeySignature? = null,
    val rawJson: String? = null,
    val warnings: List<String> = emptyList()
)

/**
 * OMR Client 介面
 */
interface OmrClient {
    /**
     * 將樂譜圖片送至雲端 OMR API，回傳和弦快照列表
     *
     * @param imageBytes 圖片位元組陣列（JPEG/PNG）
     * @param filename 可選檔名
     * @return OmrResult 包含和弦列表與警告訊息
     * @throws OmrException 當 API 呼叫失敗或解析錯誤時
     */
    suspend fun recognizeScore(
        imageBytes: ByteArray,
        filename: String? = null
    ): OmrResult
}

/**
 * 基於 Retrofit 的 OMR Client 實作
 */
class RetrofitOmrClient(
    private val api: OmrApi,
    private val staffLayout: StaffLayout = StaffLayout.SATB_GRAND_STAFF,
    private val languageHint: String = "zh-TW"
) : OmrClient {

    constructor(
        baseUrl: String,
        apiKey: String? = null,
        enableLogging: Boolean = false,
        staffLayout: StaffLayout = StaffLayout.SATB_GRAND_STAFF
    ) : this(
        api = OmrApiClient.create(baseUrl, apiKey, enableLogging),
        staffLayout = staffLayout
    )

    override suspend fun recognizeScore(
        imageBytes: ByteArray,
        filename: String?
    ): OmrResult {
        try {
            // 1. 將圖片編碼為 Base64
            val imageBase64 = Base64.encodeToString(imageBytes, Base64.NO_WRAP)

            // 2. 建立請求
            val request = OmrRequestDto(
                imageBase64 = imageBase64,
                filename = filename,
                options = OmrRequestOptions(
                    staffLayout = staffLayout,
                    expectedVoices = listOf("S", "A", "T", "B"),
                    languageHint = languageHint
                )
            )

            // 3. 呼叫 API
            val response = api.recognizeScore(request)

            // 4. 檢查錯誤
            if (response.error != null) {
                throw OmrException(
                    code = response.error.code,
                    message = response.error.message
                )
            }

            // 5. 轉換 DTO → Domain
            val mappingResult = OmrResponseMapper.toDomain(response)

            return OmrResult(
                chords = mappingResult.chords,
                keySignature = mappingResult.keySignature,
                rawJson = mappingResult.rawJson,
                warnings = mappingResult.warnings
            )

        } catch (e: OmrException) {
            throw e
        } catch (e: Exception) {
            throw OmrException(
                code = "CLIENT_ERROR",
                message = "OMR Client 錯誤: ${e.message}",
                cause = e
            )
        }
    }
}

/**
 * OMR 相關例外
 */
class OmrException(
    val code: String,
    message: String,
    cause: Throwable? = null
) : Exception(message, cause)

/**
 * 舊版 HttpOmrClient - 已棄用，保留向後相容
 */
@Deprecated(
    message = "Use RetrofitOmrClient instead",
    replaceWith = ReplaceWith("RetrofitOmrClient")
)
class HttpOmrClient(
    private val baseUrl: String,
    private val httpPost: suspend (url: String, body: ByteArray, contentType: String) -> String
) : OmrClient {

    override suspend fun recognizeScore(imageBytes: ByteArray, filename: String?): OmrResult {
        return OmrResult(
            chords = emptyList(),
            warnings = listOf("HttpOmrClient 已棄用，請改用 RetrofitOmrClient")
        )
    }
}
```

**改進重點：**
1. 使用 Retrofit 取代原本的 `httpPost` lambda
2. 完整的錯誤處理（`OmrException`）
3. 自動處理 Base64 編碼
4. 舊版 API 標記為 `@Deprecated` 以保持向後相容

---

### B.7 使用範例與整合

#### B.7.1 在 ViewModel 中使用

```kotlin
// android-app/app/src/main/java/com/example/harmonychecker/ui/viewmodels/MainViewModel.kt
class MainViewModel : ViewModel() {

    private val omrClient = RetrofitOmrClient(
        baseUrl = "https://your-cloud-function.com",  // 需從 BuildConfig 或設定讀取
        apiKey = "YOUR_API_KEY",                       // 需從安全儲存讀取
        enableLogging = BuildConfig.DEBUG
    )

    fun recognizeAndAnalyzeScore(imageBytes: ByteArray) {
        viewModelScope.launch {
            try {
                _uiState.value = UiState.Loading

                // 1. OMR 識別
                val omrResult = omrClient.recognizeScore(imageBytes)

                // 2. 規則檢查
                val issues = HarmonyRuleEngine.analyze(
                    chords = omrResult.chords,
                    key = omrResult.keySignature
                )

                // 3. 更新 UI
                _uiState.value = UiState.Success(
                    chords = omrResult.chords,
                    issues = issues,
                    warnings = omrResult.warnings
                )

            } catch (e: OmrException) {
                _uiState.value = UiState.Error("OMR 錯誤: ${e.message}")
            } catch (e: Exception) {
                _uiState.value = UiState.Error("未知錯誤: ${e.message}")
            }
        }
    }
}
```

#### B.7.2 單元測試範例

```kotlin
// android-app/app/src/test/java/com/example/harmonychecker/data/omr/mapper/OmrResponseMapperTest.kt
class OmrResponseMapperTest {

    @Test
    fun `pitchToMidi should convert pitch strings correctly`() {
        OmrResponseMapper.pitchToMidi("C4") shouldBe 60
        OmrResponseMapper.pitchToMidi("C#4") shouldBe 61
        OmrResponseMapper.pitchToMidi("A4") shouldBe 69
        OmrResponseMapper.pitchToMidi("C5") shouldBe 72
    }

    @Test
    fun `parseKeySignature should parse key strings correctly`() {
        val cMajor = OmrResponseMapper.parseKeySignature("C major")
        cMajor?.tonicMidi shouldBe 60
        cMajor?.mode shouldBe KeyMode.MAJOR

        val aMinor = OmrResponseMapper.parseKeySignature("a minor")
        aMinor?.tonicMidi shouldBe 69
        aMinor?.mode shouldBe KeyMode.MINOR
    }

    @Test
    fun `toDomain should convert response DTO correctly`() {
        val responseJson = """
        {
          "measures": [
            {
              "index": 1,
              "key_signature": "C major",
              "chords": [
                {
                  "beat": 1.0,
                  "notes": [
                    {"voice": "S", "pitch": "C5", "duration": 1.0},
                    {"voice": "A", "pitch": "G4", "duration": 1.0},
                    {"voice": "T", "pitch": "E4", "duration": 1.0},
                    {"voice": "B", "pitch": "C3", "duration": 1.0}
                  ]
                }
              ]
            }
          ]
        }
        """.trimIndent()

        val response = Json.decodeFromString<OmrResponseDto>(responseJson)
        val result = OmrResponseMapper.toDomain(response)

        result.chords.size shouldBe 1
        result.chords[0].notes[Voice.S]?.midi shouldBe 72  // C5
        result.keySignature?.tonicMidi shouldBe 60         // C
    }
}
```

---

### B.8 實作階段規劃

#### Phase 1：DTO 與 Mapper 實作（1 天）
- [ ] 建立 `data/omr/dto/` 套件與所有 DTO 類別
- [ ] 實作 `OmrResponseMapper`
- [ ] 單元測試：`pitchToMidi`, `parseKeySignature`, `toDomain`

#### Phase 2：Retrofit 整合（0.5 天）
- [ ] 建立 `OmrApi` interface
- [ ] 實作 `OmrApiClient` factory
- [ ] 設定 OkHttp interceptors（API key, logging）

#### Phase 3：OmrClient 實作（0.5 天）
- [ ] 更新 `OmrClient.kt`
- [ ] 實作 `RetrofitOmrClient`
- [ ] 標記舊版 `HttpOmrClient` 為 deprecated

#### Phase 4：錯誤處理與邊界測試（0.5 天）
- [ ] 測試缺少聲部的 chord
- [ ] 測試無效的 pitch string
- [ ] 測試 API 錯誤回應
- [ ] 測試空 response

#### Phase 5：整合測試與文件（0.5 天）
- [ ] ViewModel 整合測試
- [ ] 建立 mock API server 用於測試
- [ ] 更新 `android-app/README.md`
- [ ] 撰寫 API 使用說明文件

**總預估時間：3 天**

---

## Part C：實作優先順序建議

### C.1 建議執行順序

根據 CLAUDE.md 的「先規劃再動手」原則，建議以下執行順序：

**Week 1：測試基礎設施**
- Day 1-2: 建立 `test-data/` 結構與 TestDataLoader
- Day 3: 核心輔助函數測試（Phase 2）

**Week 2：規則測試完成**
- Day 4-6: 建立所有測試案例 JSON 與規則檢查測試（Phase 3）
- Day 7: 交叉驗證（Phase 4）

**Week 3：OMR Client 實作**
- Day 8: DTO 與 Mapper（B.8 Phase 1）
- Day 9: Retrofit 整合（B.8 Phase 2-3）
- Day 10: 錯誤處理與整合（B.8 Phase 4-5）

### C.2 為什麼先測試後 OMR？

1. **測試驗證規則引擎正確性** - 這是系統的核心價值
2. **OMR Client 依賴測試資料** - 可以用測試 JSON 模擬 API 回應
3. **測試提供安全網** - 實作 OMR 時不怕破壞現有邏輯

---

## Part D：風險評估與緩解

### D.1 測試相關風險

| 風險 | 機率 | 影響 | 緩解措施 |
|------|------|------|---------|
| Kotlin 與 Python 行為不一致 | 中 | 高 | 交叉驗證測試強制一致性 |
| 測試資料不夠全面 | 中 | 中 | 從真實作業收集案例補充 |
| 測試執行太慢 | 低 | 低 | 使用 JUnit 5 並行執行 |

### D.2 OMR Client 相關風險

| 風險 | 機率 | 影響 | 緩解措施 |
|------|------|------|---------|
| API 回應格式變更 | 中 | 中 | `ignoreUnknownKeys = true` + 版本協商 |
| 網路逾時或失敗 | 高 | 中 | Retry 機制 + 離線模式 |
| Base64 編碼記憶體溢位 | 低 | 高 | 圖片壓縮 + 分段上傳（未來） |
| pitch string 解析錯誤 | 中 | 中 | 防禦性解析 + 詳細錯誤訊息 |

---

## Part E：成功標準

### E.1 單元測試成功標準

- [ ] Kotlin 與 Python 測試套件各自 100% 通過
- [ ] 交叉驗證測試顯示兩者輸出 100% 一致
- [ ] 至少 50 個測試案例（含基礎、違規、整合、邊界）
- [ ] 測試覆蓋率 > 90%（函數覆蓋率）
- [ ] 所有測試執行時間 < 10 秒

### E.2 OMR Client 成功標準

- [ ] 能成功解析符合 `omr_proxy_api.md` 規格的 JSON 回應
- [ ] `pitchToMidi` 支援 C0-C8 範圍與升降記號
- [ ] 錯誤處理完善，不會因單一欄位錯誤導致整體失敗
- [ ] 單元測試覆蓋率 > 85%
- [ ] 與 mock API server 整合測試通過

---

## Part F：後續擴充方向

完成本次實作後，未來可考慮：

1. **測試資料視覺化工具**：將測試 JSON 渲染成五線譜 PNG
2. **效能測試**：大量和弦（1000+ chords）的規則檢查效能
3. **OMR Client 快取機制**：離線儲存已識別的樂譜
4. **多模態 LLM Prompt 優化**：根據實際 API 回應品質調整 prompt
5. **Android Instrumented Test**：真實裝置上的端到端測試

---

## 總結

本規劃文件提供了：
- **A 部分**：完整的單元測試策略，包含 44+ 測試案例與交叉驗證機制
- **B 部分**：OMR Client 的三層架構（DTO → Mapper → Client）完整實作
- **C-F 部分**：執行順序、風險評估、成功標準與未來方向

**預估總工作量：9-10 天**（單人全職）

下一步：請確認此規劃是否符合需求，我將開始實作。
