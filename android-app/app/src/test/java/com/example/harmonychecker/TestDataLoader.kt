package com.example.harmonychecker

import com.example.harmonychecker.core.harmony.*
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

/**
 * 測試資料載入器
 *
 * 用於讀取 test-data/ 目錄中的 JSON 測試案例。
 * 支援 Kotlin 與 Python 的共享測試資料格式。
 */

// ========== 測試資料結構定義 ==========

@Serializable
data class TestCase(
    @SerialName("test_id")
    val testId: String,

    @SerialName("description")
    val description: String,

    @SerialName("category")
    val category: String,

    @SerialName("severity_expected")
    val severityExpected: String? = null,

    @SerialName("input")
    val input: TestInput,

    @SerialName("expected_output")
    val expectedOutput: TestExpectedOutput,

    @SerialName("notes")
    val notes: String? = null
)

@Serializable
data class TestInput(
    @SerialName("key_signature")
    val keySignature: TestKeySignature? = null,

    @SerialName("chords")
    val chords: List<TestChordSnapshot>
)

@Serializable
data class TestKeySignature(
    @SerialName("tonic_midi")
    val tonicMidi: Int,

    @SerialName("mode")
    val mode: String  // "major" or "minor"
)

@Serializable
data class TestChordSnapshot(
    @SerialName("index")
    val index: Int,

    @SerialName("measure")
    val measure: Int,

    @SerialName("beat")
    val beat: Double,

    @SerialName("notes")
    val notes: Map<String, TestNoteEvent>
)

@Serializable
data class TestNoteEvent(
    @SerialName("voice")
    val voice: String,

    @SerialName("midi")
    val midi: Int,

    @SerialName("measure")
    val measure: Int,

    @SerialName("beat")
    val beat: Double,

    @SerialName("duration")
    val duration: Double? = null
)

@Serializable
data class TestExpectedOutput(
    @SerialName("violation_count")
    val violationCount: Int,

    @SerialName("violations")
    val violations: List<TestExpectedViolation>,

    @SerialName("cadence")
    val cadence: String? = null
)

@Serializable
data class TestExpectedViolation(
    @SerialName("rule_id")
    val ruleId: String,

    @SerialName("severity")
    val severity: String,

    @SerialName("voices_involved")
    val voicesInvolved: List<String>? = null,

    @SerialName("location")
    val location: Map<String, Int>,

    @SerialName("message_zh_contains")
    val messageZhContains: String
)

// ========== TestDataLoader Object ==========

object TestDataLoader {

    /**
     * JSON 解析器設定
     */
    private val json = Json {
        ignoreUnknownKeys = true  // 容錯：忽略未知欄位
        isLenient = true           // 寬鬆模式
        prettyPrint = true
    }

    /**
     * 載入單一測試案例
     *
     * @param filePath 測試 JSON 檔案的相對或絕對路徑
     * @return TestCase 測試案例物件
     * @throws IllegalArgumentException 當檔案不存在或格式錯誤時
     */
    fun loadTestCase(filePath: String): TestCase {
        val file = File(filePath)
        if (!file.exists()) {
            throw IllegalArgumentException("Test file not found: $filePath")
        }

        val content = file.readText(Charsets.UTF_8)
        return try {
            json.decodeFromString<TestCase>(content)
        } catch (e: Exception) {
            throw IllegalArgumentException("Failed to parse test file: $filePath. Error: ${e.message}", e)
        }
    }

    /**
     * 載入目錄下的所有測試案例
     *
     * @param dirPath 測試目錄的相對或絕對路徑
     * @return List<TestCase> 測試案例列表
     */
    fun loadTestDataFromDirectory(dirPath: String): List<TestCase> {
        val dir = File(dirPath)
        if (!dir.exists() || !dir.isDirectory) {
            throw IllegalArgumentException("Test directory not found: $dirPath")
        }

        return dir.listFiles { file -> file.extension == "json" }
            ?.sortedBy { it.name }
            ?.map { loadTestCase(it.absolutePath) }
            ?: emptyList()
    }

    /**
     * 載入所有測試資料（遞迴搜尋）
     *
     * @param rootPath 根目錄路徑（預設為 test-data/）
     * @return List<TestCase> 所有測試案例列表
     */
    fun loadAllTestData(rootPath: String = "test-data/"): List<TestCase> {
        val root = File(rootPath)
        if (!root.exists() || !root.isDirectory) {
            throw IllegalArgumentException("Root test directory not found: $rootPath")
        }

        val testCases = mutableListOf<TestCase>()

        root.walkTopDown()
            .filter { it.isFile && it.extension == "json" }
            .forEach { file ->
                try {
                    testCases.add(loadTestCase(file.absolutePath))
                } catch (e: Exception) {
                    println("Warning: Failed to load ${file.absolutePath}: ${e.message}")
                }
            }

        return testCases.sortedBy { it.testId }
    }

    // ========== 轉換函數：TestCase → Domain Objects ==========

    /**
     * 將測試資料中的 KeySignature 轉換為 Domain 物件
     */
    fun TestKeySignature.toDomain(): KeySignature {
        val mode = when (this.mode.lowercase()) {
            "major" -> Mode.MAJOR
            "minor" -> Mode.MINOR
            else -> throw IllegalArgumentException("Unknown mode: ${this.mode}")
        }
        return KeySignature(tonicMidi = this.tonicMidi, mode = mode)
    }

    /**
     * 將測試資料中的 ChordSnapshot 轉換為 Domain 物件
     */
    fun TestChordSnapshot.toDomain(): ChordSnapshot {
        val notesByVoice = mutableMapOf<Voice, NoteEvent>()

        for ((voiceStr, noteData) in this.notes) {
            val voice = try {
                Voice.fromString(voiceStr)
            } catch (e: Exception) {
                throw IllegalArgumentException("Invalid voice '$voiceStr' in chord at index $index", e)
            }

            notesByVoice[voice] = NoteEvent(
                voice = voice,
                midi = noteData.midi,
                measure = noteData.measure,
                beat = noteData.beat,
                duration = noteData.duration
            )
        }

        return ChordSnapshot(
            index = this.index,
            measure = this.measure,
            beat = this.beat,
            notes = notesByVoice
        )
    }

    /**
     * 將測試資料中的 Input 轉換為可用於規則引擎的格式
     */
    fun TestInput.toDomain(): Pair<List<ChordSnapshot>, KeySignature?> {
        val chords = this.chords.map { it.toDomain() }
        val keySignature = this.keySignature?.toDomain()
        return Pair(chords, keySignature)
    }

    /**
     * 便利方法：直接從 TestCase 取得可執行的 Domain 物件
     */
    fun TestCase.toDomainInput(): Pair<List<ChordSnapshot>, KeySignature?> {
        return this.input.toDomain()
    }
}

// ========== 擴充函數：方便測試使用 ==========

/**
 * 載入測試案例的便利擴充函數
 */
fun String.asTestCase(): TestCase {
    return TestDataLoader.loadTestCase(this)
}

/**
 * 檢查 HarmonyIssue 是否符合預期違規
 */
fun HarmonyIssue.matches(expected: TestExpectedViolation): Boolean {
    // 檢查 rule_id
    if (this.ruleId != expected.ruleId) return false

    // 檢查 severity
    val expectedSeverity = when (expected.severity.lowercase()) {
        "error" -> Severity.ERROR
        "warning" -> Severity.WARNING
        else -> return false
    }
    if (this.severity != expectedSeverity) return false

    // 檢查 message 是否包含關鍵字
    if (!this.message.contains(expected.messageZhContains)) return false

    // 檢查 voices_involved (如果提供)
    if (expected.voicesInvolved != null) {
        val expectedVoices = expected.voicesInvolved.map { Voice.fromString(it) }.toSet()
        if (this.voices.toSet() != expectedVoices) return false
    }

    return true
}

/**
 * 檢查 HarmonyIssue 列表是否符合所有預期違規
 */
fun List<HarmonyIssue>.matchesExpected(expected: List<TestExpectedViolation>): Boolean {
    if (this.size != expected.size) return false

    // 逐一配對檢查
    val matched = mutableSetOf<Int>()
    for (issue in this) {
        val matchIndex = expected.indices.find { idx ->
            idx !in matched && issue.matches(expected[idx])
        }
        if (matchIndex == null) return false
        matched.add(matchIndex)
    }

    return matched.size == expected.size
}
