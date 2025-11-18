package com.example.harmonychecker.core.harmony

/**
 * 資料模型與規則引擎進入點骨架。
 * 目標：對齊 Python harmony_rules.py 的語意與結構。
 */

enum class Voice {
    SOPRANO, ALTO, TENOR, BASS
}

enum class Mode {
    MAJOR, MINOR
}

enum class Severity {
    ERROR, WARNING
}

data class KeySignature(
    val tonicMidi: Int,   // 主音 MIDI 數值，例如 C4 = 60
    val mode: Mode
)

data class NoteEvent(
    val voice: Voice,
    val midi: Int,
    val measure: Int,
    val beat: Double,
    val duration: Double? = null
)

data class ChordSnapshot(
    val measure: Int,
    val beat: Double,
    val notes: List<NoteEvent>
)

data class HarmonyIssue(
    val ruleId: String,
    val message: String,
    val measure: Int? = null,
    val beat: Double? = null,
    val voices: List<Voice> = emptyList(),
    val severity: Severity = Severity.ERROR
)

object HarmonyRuleEngine {

    /**
     * Kotlin 版規則引擎的進入點。
     *
     * TODO:
     *  - 將 Python harmony_rules.py 中的規則逐一移植到此處或相關 helper。
     *  - 確保行為與 Python 版本一致（可用相同測試資料比對）。
     */
    fun analyze(
        chords: List<ChordSnapshot>,
        key: KeySignature? = null
    ): List<HarmonyIssue> {
        val issues = mutableListOf<HarmonyIssue>()

        // TODO: 呼叫各種規則檢查函式，例如：
        // issues += checkParallelFifths(chords)
        // issues += checkParallelOctaves(chords)
        // issues += checkLeadingToneResolution(chords, key)
        // ...

        return issues
    }
}
