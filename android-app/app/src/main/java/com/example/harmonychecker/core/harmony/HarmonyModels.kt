package com.example.harmonychecker.core.harmony

import kotlin.math.abs

/**
 * 資料模型與規則引擎進入點。
 * 對齊 Python harmony_rules.py 的語意與結構。
 *
 * 對應 Python 檔案：harmony_rules.py
 */

enum class Voice {
    SOPRANO, ALTO, TENOR, BASS;

    companion object {
        fun fromString(s: String): Voice = when (s) {
            "S" -> SOPRANO
            "A" -> ALTO
            "T" -> TENOR
            "B" -> BASS
            else -> throw IllegalArgumentException("Unknown voice: $s")
        }
    }
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

/**
 * 和弦快照：四部和聲的垂直切片。
 * 對應 Python: ChordSnapshot
 *
 * 注意：使用 Map 而非 List 以便直接按聲部存取。
 */
data class ChordSnapshot(
    val index: Int,       // 和弦索引（用於錯誤定位）
    val measure: Int,
    val beat: Double,
    val notes: Map<Voice, NoteEvent>
) {
    // 便利方法：取得特定聲部的音高
    fun getMidi(voice: Voice): Int = notes[voice]?.midi
        ?: throw IllegalStateException("Voice $voice not found in chord")
}

data class HarmonyIssue(
    val ruleId: String,
    val message: String,
    val detail: String = "",
    val measure: Int? = null,
    val beat: Double? = null,
    val voices: List<Voice> = emptyList(),
    val severity: Severity = Severity.ERROR
)

/**
 * 三和弦分析結果。
 * 對應 Python: TriadInfo
 */
data class TriadInfo(
    val rootPc: Int,  // 根音音高類別 (0-11)
    val quality: TriadQuality,
    val roleByVoice: Map<Voice, TriadRole>  // 每個聲部在和弦中的角色
)

enum class TriadQuality {
    MAJOR, MINOR, DIMINISHED, AUGMENTED
}

enum class TriadRole {
    ROOT, THIRD, FIFTH
}

// ========== 核心辅助函数 ==========
// 对应 Python 的 private 函数 (_interval_semitones, _direction, etc.)

/**
 * 計算半音距離：b - a
 * 對應 Python: _interval_semitones
 */
private fun intervalSemitones(a: Int, b: Int): Int = b - a

/**
 * 計算旋律方向：1=上行, -1=下行, 0=持續
 * 對應 Python: _direction
 */
private fun direction(a: Int, b: Int): Int {
    val diff = b - a
    return when {
        diff > 0 -> 1
        diff < 0 -> -1
        else -> 0
    }
}

/**
 * 判斷是否為完全五度（±7 半音，mod 12）
 * 對應 Python: _is_perfect_fifth
 */
private fun isPerfectFifth(diff: Int): Boolean = abs(diff) % 12 == 7

/**
 * 判斷是否為八度或同度（±0 半音，mod 12）
 * 對應 Python: _is_perfect_octave_or_unison
 */
private fun isPerfectOctaveOrUnison(diff: Int): Boolean = abs(diff) % 12 == 0

/**
 * 判斷是否為級進（半音或全音）
 * 對應 Python: _is_step
 */
private fun isStep(diff: Int): Boolean = abs(diff) in listOf(1, 2)

/**
 * 正規化音高類別（mod 12）
 * 對應 Python: _normalize_pc
 */
private fun normalizePitchClass(midi: Int): Int = midi % 12

/**
 * 分析三和弦結構。
 * 對應 Python: _analyze_triad
 *
 * @return TriadInfo 若能識別為標準三和弦；否則 null
 */
private fun analyzeTriad(chord: ChordSnapshot): TriadInfo? {
    val pcsByVoice = chord.notes.mapValues { (_, note) -> normalizePitchClass(note.midi) }
    val uniquePcs = pcsByVoice.values.toSet().sorted()

    if (uniquePcs.size < 3) return null

    for (rootPc in uniquePcs) {
        val intervals = uniquePcs
            .filter { it != rootPc }
            .map { (it - rootPc + 12) % 12 }
            .sorted()

        val quality: TriadQuality? = when (intervals) {
            listOf(3, 7) -> TriadQuality.MINOR
            listOf(4, 7) -> TriadQuality.MAJOR
            listOf(3, 6) -> TriadQuality.DIMINISHED
            listOf(4, 8) -> TriadQuality.AUGMENTED
            else -> null
        }

        if (quality == null) continue

        val roleByVoice = mutableMapOf<Voice, TriadRole>()
        for ((voice, pc) in pcsByVoice) {
            val d = (pc - rootPc + 12) % 12
            val role = when {
                d == 0 -> TriadRole.ROOT
                d in listOf(3, 4) -> TriadRole.THIRD
                d in listOf(6, 7, 8) -> TriadRole.FIFTH
                else -> return null  // 無法識別為標準三和弦
            }
            roleByVoice[voice] = role
        }

        return TriadInfo(rootPc, quality, roleByVoice)
    }

    return null
}

object HarmonyRuleEngine {

    /**
     * Kotlin 版規則引擎的進入點。
     * 對應 Python: HarmonyAnalyzer.analyze()
     *
     * 執行所有已實作的和聲規則檢查。
     */
    fun analyze(
        chords: List<ChordSnapshot>,
        key: KeySignature? = null
    ): List<HarmonyIssue> {
        if (chords.isEmpty()) return emptyList()

        val issues = mutableListOf<HarmonyIssue>()
        issues += checkMelodicIntervals(chords)
        issues += checkVoiceCrossingAndOverlap(chords)
        issues += checkParallelIntervals(chords)
        issues += checkHiddenIntervals(chords)
        issues += checkTriadDoubling(chords, key)
        issues += checkLeadingToneResolution(chords, key)
        return issues
    }

    /**
     * 終止式分類（簡易版）。
     * 對應 Python: HarmonyAnalyzer.classify_cadence()
     */
    fun classifyCadence(chords: List<ChordSnapshot>, key: KeySignature?): String? {
        if (key == null || chords.size < 2) return null

        val last = chords.last()
        val prev = chords[chords.lastIndex - 1]

        val triadLast = analyzeTriad(last) ?: return null
        val triadPrev = analyzeTriad(prev) ?: return null

        val degLast = scaleDegree(triadLast.rootPc, key) ?: return null
        val degPrev = scaleDegree(triadPrev.rootPc, key) ?: return null

        val sopranoLast = last.notes[Voice.SOPRANO]?.midi ?: return null
        val tonicPc = normalizePitchClass(key.tonicMidi)

        val isLastRootPosition = normalizePitchClass(last.notes[Voice.BASS]?.midi ?: return null) == triadLast.rootPc
        val isPrevRootPosition = normalizePitchClass(prev.notes[Voice.BASS]?.midi ?: return null) == triadPrev.rootPc
        val sopranoIsTonic = normalizePitchClass(sopranoLast) == tonicPc

        return when {
            degPrev == 5 && degLast == 1 -> {
                if (isLastRootPosition && isPrevRootPosition && sopranoIsTonic) "PAC" else "IAC"
            }
            degPrev == 4 && degLast == 1 -> "PC"
            degPrev == 5 && degLast == 6 -> "DC"
            degLast == 5 -> "HC"
            else -> null
        }
    }

    // ========== 規則檢查實作 ==========

    /**
     * M1: 旋律跳進限制檢查。
     * 對應 Python: _check_melodic_intervals
     */
    private fun checkMelodicIntervals(chords: List<ChordSnapshot>): List<HarmonyIssue> {
        val issues = mutableListOf<HarmonyIssue>()

        for (voice in Voice.values()) {
            var prevNote: NoteEvent? = null
            for (chord in chords) {
                val note = chord.notes[voice] ?: continue

                if (prevNote != null) {
                    val diff = intervalSemitones(prevNote.midi, note.midi)
                    val adiff = abs(diff)

                    // 超過八度
                    if (adiff > 12) {
                        issues += HarmonyIssue(
                            ruleId = "M1",
                            message = "${voice.name} 聲部旋律跳進超過八度，違反聲部進行原則。",
                            detail = "from ${prevNote.midi} to ${note.midi}, diff=$adiff semitones.",
                            measure = note.measure,
                            beat = note.beat,
                            voices = listOf(voice),
                            severity = Severity.ERROR
                        )
                    }
                    // 七度跳進
                    else if (adiff in listOf(10, 11)) {
                        issues += HarmonyIssue(
                            ruleId = "M1",
                            message = "${voice.name} 聲部出現七度跳進，需確認後續是否有適當解決。",
                            detail = "from ${prevNote.midi} to ${note.midi}, diff=$adiff semitones.",
                            measure = note.measure,
                            beat = note.beat,
                            voices = listOf(voice),
                            severity = Severity.WARNING
                        )
                    }
                    // 增四度/減五度
                    else if (adiff == 6) {
                        issues += HarmonyIssue(
                            ruleId = "M1",
                            message = "${voice.name} 聲部出現增四度／減五度的跳進，一般視為不佳旋律。",
                            detail = "from ${prevNote.midi} to ${note.midi}, diff=6 semitones.",
                            measure = note.measure,
                            beat = note.beat,
                            voices = listOf(voice),
                            severity = Severity.ERROR
                        )
                    }

                    // 內聲部限制（不得超過完全四度）
                    if (voice in listOf(Voice.ALTO, Voice.TENOR) && adiff > 5) {
                        issues += HarmonyIssue(
                            ruleId = "M1",
                            message = "${voice.name} 聲部跳進超過完全四度，違反內聲部平穩進行原則。",
                            detail = "from ${prevNote.midi} to ${note.midi}, diff=$adiff semitones.",
                            measure = note.measure,
                            beat = note.beat,
                            voices = listOf(voice),
                            severity = Severity.ERROR
                        )
                    }
                }
                prevNote = note
            }
        }

        return issues
    }

    /**
     * V1: 聲部交錯與超越檢查。
     * 對應 Python: _check_voice_crossing_and_overlap
     */
    private fun checkVoiceCrossingAndOverlap(chords: List<ChordSnapshot>): List<HarmonyIssue> {
        val issues = mutableListOf<HarmonyIssue>()

        // 檢查每個和弦的交錯
        for (chord in chords) {
            val s = chord.getMidi(Voice.SOPRANO)
            val a = chord.getMidi(Voice.ALTO)
            val t = chord.getMidi(Voice.TENOR)
            val b = chord.getMidi(Voice.BASS)

            if (s < a) {
                issues += HarmonyIssue(
                    ruleId = "V1",
                    message = "Soprano 低於 Alto，發生聲部交錯（Crossing）。",
                    detail = "S=$s, A=$a",
                    measure = chord.measure,
                    beat = chord.beat,
                    severity = Severity.ERROR
                )
            }
            if (a < t) {
                issues += HarmonyIssue(
                    ruleId = "V1",
                    message = "Alto 低於 Tenor，發生聲部交錯（Crossing）。",
                    detail = "A=$a, T=$t",
                    measure = chord.measure,
                    beat = chord.beat,
                    severity = Severity.ERROR
                )
            }
            if (t < b) {
                issues += HarmonyIssue(
                    ruleId = "V1",
                    message = "Tenor 低於 Bass，發生聲部交錯（Crossing）。",
                    detail = "T=$t, B=$b",
                    measure = chord.measure,
                    beat = chord.beat,
                    severity = Severity.ERROR
                )
            }
        }

        // 檢查相鄰和弦的超越
        for (i in 0 until chords.size - 1) {
            val c1 = chords[i]
            val c2 = chords[i + 1]

            val voicePairs = listOf(
                Voice.SOPRANO to Voice.ALTO,
                Voice.ALTO to Voice.TENOR,
                Voice.TENOR to Voice.BASS
            )

            for ((upper, lower) in voicePairs) {
                val upperPrev = c1.getMidi(upper)
                val lowerPrev = c1.getMidi(lower)
                val upperNext = c2.getMidi(upper)
                val lowerNext = c2.getMidi(lower)

                if (lowerNext > upperPrev) {
                    issues += HarmonyIssue(
                        ruleId = "V1",
                        message = "$lower/$upper 聲部之間發生超越（Overlap）。",
                        detail = "prev $upper=$upperPrev, $lower=$lowerPrev; next $upper=$upperNext, $lower=$lowerNext",
                        measure = c2.measure,
                        beat = c2.beat,
                        severity = Severity.WARNING
                    )
                }
            }
        }

        return issues
    }

    /**
     * P1: 平行八度與平行五度檢查。
     * 對應 Python: _check_parallel_intervals
     */
    private fun checkParallelIntervals(chords: List<ChordSnapshot>): List<HarmonyIssue> {
        val issues = mutableListOf<HarmonyIssue>()

        val voicePairs = listOf(
            Voice.SOPRANO to Voice.ALTO,
            Voice.SOPRANO to Voice.TENOR,
            Voice.SOPRANO to Voice.BASS,
            Voice.ALTO to Voice.TENOR,
            Voice.ALTO to Voice.BASS,
            Voice.TENOR to Voice.BASS
        )

        for ((v1, v2) in voicePairs) {
            for (i in 0 until chords.size - 1) {
                val c1 = chords[i]
                val c2 = chords[i + 1]

                val n1a = c1.notes[v1] ?: continue
                val n1b = c1.notes[v2] ?: continue
                val n2a = c2.notes[v1] ?: continue
                val n2b = c2.notes[v2] ?: continue

                val interval1 = intervalSemitones(n1b.midi, n1a.midi)
                val interval2 = intervalSemitones(n2b.midi, n2a.midi)

                val dir1a = direction(n1a.midi, n2a.midi)
                val dir1b = direction(n1b.midi, n2b.midi)

                // 兩聲部必須同向移動，且非持續音
                if (dir1a == 0 || dir1b == 0) continue
                if (dir1a != dir1b) continue

                // 兩個音程都是完全五度或八度
                if ((isPerfectFifth(interval1) || isPerfectOctaveOrUnison(interval1)) &&
                    (isPerfectFifth(interval2) || isPerfectOctaveOrUnison(interval2))
                ) {
                    issues += HarmonyIssue(
                        ruleId = "P1",
                        message = "$v1-$v2 聲部之間出現平行八度或平行五度。",
                        detail = "chord ${c1.index}->${c2.index}, interval1=$interval1, interval2=$interval2",
                        measure = c2.measure,
                        beat = c2.beat,
                        voices = listOf(v1, v2),
                        severity = Severity.ERROR
                    )
                }
            }
        }

        return issues
    }

    /**
     * P2: 隱伏八度與隱伏五度檢查（僅外聲部）。
     * 對應 Python: _check_hidden_intervals
     */
    private fun checkHiddenIntervals(chords: List<ChordSnapshot>): List<HarmonyIssue> {
        val issues = mutableListOf<HarmonyIssue>()

        for (i in 0 until chords.size - 1) {
            val c1 = chords[i]
            val c2 = chords[i + 1]

            val s1 = c1.notes[Voice.SOPRANO]?.midi ?: continue
            val b1 = c1.notes[Voice.BASS]?.midi ?: continue
            val s2 = c2.notes[Voice.SOPRANO]?.midi ?: continue
            val b2 = c2.notes[Voice.BASS]?.midi ?: continue

            val interval1 = intervalSemitones(b1, s1)
            val interval2 = intervalSemitones(b2, s2)

            val dirS = direction(s1, s2)
            val dirB = direction(b1, b2)

            // 必須同向移動
            if (dirS == 0 || dirB == 0 || dirS != dirB) continue

            // 前一個音程不是完全音程，但後一個是
            if ((!isPerfectFifth(interval1) && !isPerfectOctaveOrUnison(interval1)) &&
                (isPerfectFifth(interval2) || isPerfectOctaveOrUnison(interval2))
            ) {
                val sStep = isStep(intervalSemitones(s1, s2))
                val bLeap = abs(intervalSemitones(b1, b2)) > 2

                val (severity, msg) = if (sStep && bLeap) {
                    Severity.WARNING to "外聲部出現隱伏八/五度，但屬於 S 級進、B 跳進的例外情形。"
                } else {
                    Severity.ERROR to "外聲部出現隱伏八度或隱伏五度，需避免。"
                }

                issues += HarmonyIssue(
                    ruleId = "P2",
                    message = msg,
                    detail = "chord ${c1.index}->${c2.index}, interval1=$interval1, interval2=$interval2, dir_s=$dirS, dir_b=$dirB",
                    measure = c2.measure,
                    beat = c2.beat,
                    voices = listOf(Voice.SOPRANO, Voice.BASS),
                    severity = severity
                )
            }
        }

        return issues
    }

    /**
     * D1: 三和弦重複音與省略音檢查。
     * 對應 Python: _check_triad_doubling
     */
    private fun checkTriadDoubling(chords: List<ChordSnapshot>, key: KeySignature?): List<HarmonyIssue> {
        val issues = mutableListOf<HarmonyIssue>()

        for (chord in chords) {
            val triad = analyzeTriad(chord) ?: continue

            val pcs = chord.notes.values.map { normalizePitchClass(it.midi) }
            val rootCount = pcs.count { it == triad.rootPc }

            // 計算三音與五音的音高類別
            val thirdPc = when (triad.quality) {
                TriadQuality.MAJOR, TriadQuality.AUGMENTED -> (triad.rootPc + 4) % 12
                else -> (triad.rootPc + 3) % 12
            }

            val fifthPc = when (triad.quality) {
                TriadQuality.MAJOR, TriadQuality.MINOR -> (triad.rootPc + 7) % 12
                TriadQuality.DIMINISHED -> (triad.rootPc + 6) % 12
                TriadQuality.AUGMENTED -> (triad.rootPc + 8) % 12
            }

            val hasRoot = triad.rootPc in pcs
            val hasThird = thirdPc in pcs
            val hasFifth = fifthPc in pcs

            // 檢查是否省略了根音或三音
            if (!hasRoot || !hasThird) {
                issues += HarmonyIssue(
                    ruleId = "D1",
                    message = "三和弦省略了根音或三音，違反基本配置原則。",
                    detail = "root_pc=${triad.rootPc}, has_root=$hasRoot, has_third=$hasThird, chord_index=${chord.index}",
                    measure = chord.measure,
                    beat = chord.beat,
                    severity = Severity.ERROR
                )
            }

            // 省略五音時，應有三個根音
            if (!hasFifth && pcs.toSet().size <= 3) {
                if (rootCount < 3) {
                    issues += HarmonyIssue(
                        ruleId = "D1",
                        message = "省略五音時，四部和聲應為三根一三，根音數量不足。",
                        detail = "root_pc=${triad.rootPc}, root_count=$rootCount, has_fifth=$hasFifth, chord_index=${chord.index}",
                        measure = chord.measure,
                        beat = chord.beat,
                        severity = Severity.WARNING
                    )
                }
            }

            // 檢查導音是否被重複（需要調性信息）
            if (key != null) {
                val tonicPc = normalizePitchClass(key.tonicMidi)
                val leadingPc = (tonicPc - 1 + 12) % 12

                if (triad.rootPc == leadingPc && rootCount > 1) {
                    issues += HarmonyIssue(
                        ruleId = "D1",
                        message = "導音根音被重複，違反七級和弦常見配置原則。",
                        detail = "leading_pc=$leadingPc, root_count=$rootCount, chord_index=${chord.index}",
                        measure = chord.measure,
                        beat = chord.beat,
                        severity = Severity.ERROR
                    )
                }
            }
        }

        return issues
    }

    /**
     * L1: 導音解決檢查（僅檢查外聲部）。
     * 對應 Python: _check_leading_tone_resolution
     */
    private fun checkLeadingToneResolution(chords: List<ChordSnapshot>, key: KeySignature?): List<HarmonyIssue> {
        if (key == null) return emptyList()

        val issues = mutableListOf<HarmonyIssue>()
        val tonicPc = normalizePitchClass(key.tonicMidi)
        val leadingPc = (tonicPc - 1 + 12) % 12

        for (i in 0 until chords.size - 1) {
            val c1 = chords[i]
            val c2 = chords[i + 1]

            // 僅檢查外聲部（S 和 B）
            for (voice in listOf(Voice.SOPRANO, Voice.BASS)) {
                val n1 = c1.notes[voice] ?: continue
                val n2 = c2.notes[voice] ?: continue

                // 若當前音是導音
                if (normalizePitchClass(n1.midi) == leadingPc) {
                    // 檢查是否向上解決到主音
                    val resolvedCorrectly = normalizePitchClass(n2.midi) == tonicPc &&
                            intervalSemitones(n1.midi, n2.midi) > 0

                    if (!resolvedCorrectly) {
                        issues += HarmonyIssue(
                            ruleId = "L1",
                            message = "$voice 聲部的導音未向上解決到主音。",
                            detail = "leading tone MIDI=${n1.midi}, next MIDI=${n2.midi}, expected tonic_pc=$tonicPc",
                            measure = n2.measure,
                            beat = n2.beat,
                            voices = listOf(voice),
                            severity = Severity.ERROR
                        )
                    }
                }
            }
        }

        return issues
    }

    /**
     * 將音高類別映射到音級（1-7）。
     * 對應 Python: _scale_degree
     */
    private fun scaleDegree(pc: Int, key: KeySignature): Int? {
        val tonicPc = normalizePitchClass(key.tonicMidi)
        val diff = (pc - tonicPc + 12) % 12

        val mappingMajor = mapOf(
            0 to 1, 2 to 2, 4 to 3, 5 to 4, 7 to 5, 9 to 6, 11 to 7
        )
        val mappingMinor = mapOf(
            0 to 1, 2 to 2, 3 to 3, 5 to 4, 7 to 5, 8 to 6, 11 to 7
        )

        return when (key.mode) {
            Mode.MAJOR -> mappingMajor[diff]
            Mode.MINOR -> mappingMinor[diff]
        }
    }
}
