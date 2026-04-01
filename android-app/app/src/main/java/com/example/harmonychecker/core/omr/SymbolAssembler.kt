package com.example.harmonychecker.core.omr

import android.graphics.RectF
import android.util.Log
import com.example.harmonychecker.core.harmony.*

/**
 * 符號組裝器
 *
 * 功能:
 * 將 YOLO12 檢測到的符號 bounding boxes 組裝成結構化的音樂資料
 * - 空間位置排序（上到下、左到右）
 * - 五線譜分組
 * - 音符與符幹配對
 * - 產生 ChordSnapshot 列表
 *
 * 作者: thc1006 + Claude
 * 日期: 2025-11-20
 */
class SymbolAssembler {

    companion object {
        private const val TAG = "SymbolAssembler"

        // 五線譜間距估計（像素）
        private const val STAFF_LINE_SPACING = 20f

        // 音高計算常數
        private const val MIDI_C4 = 60  // Middle C
    }

    // 組裝過程中的暫存資訊
    var detectedKeySignature: KeySignature? = null
        private set

    var detectedTimeSignature: TimeSignature? = null
        private set

    /**
     * 主要組裝方法
     */
    fun assemble(
        detections: List<Detection>,
        imageWidth: Int,
        imageHeight: Int
    ): List<ChordSnapshot> {
        if (detections.isEmpty()) {
            Log.w(TAG, "沒有檢測結果，無法組裝")
            return emptyList()
        }

        Log.d(TAG, "開始組裝 ${detections.size} 個符號")

        // 1. 分類符號
        val noteheads = detections.filter { it.className.startsWith("notehead") }
        val stafflines = detections.filter { it.className == "staffline" }
        val clefs = detections.filter { it.className.startsWith("clef") }
        val barlines = detections.filter { it.className == "barline" }
        val accidentals = detections.filter { it.className.startsWith("accidental") }

        Log.d(TAG, "符號分類: 音符頭=${noteheads.size}, 五線譜=${stafflines.size}, " +
                "譜號=${clefs.size}, 小節線=${barlines.size}, 升降號=${accidentals.size}")

        // 2. 識別五線譜區域
        val staves = identifyStaves(stafflines)
        Log.d(TAG, "識別到 ${staves.size} 個五線譜")

        // 3. 解析全域資訊（調號、拍號）
        parseGlobalInfo(detections)

        // 4. 將音符分配到五線譜
        val notesByStaff = assignNotesToStaves(noteheads, staves)

        // 5. 建立 ChordSnapshot 列表
        val chords = buildChordSnapshots(notesByStaff, barlines, accidentals)

        Log.d(TAG, "組裝完成，產生 ${chords.size} 個 ChordSnapshot")

        return chords
    }

    /**
     * 識別五線譜區域
     */
    private fun identifyStaves(stafflines: List<Detection>): List<StaffRegion> {
        if (stafflines.isEmpty()) {
            Log.w(TAG, "未檢測到五線譜線，使用預設區域")
            return listOf(StaffRegion(0, 0f, 1000f))  // 預設單一五線譜
        }

        // 根據 Y 座標分組（假設每個五線譜有 5 條線）
        val sortedLines = stafflines.sortedBy { it.bbox.centerY() }
        val staves = mutableListOf<StaffRegion>()
        val linesPerStaff = 5
        var currentLines = mutableListOf<Detection>()

        for (line in sortedLines) {
            if (currentLines.isEmpty()) {
                currentLines.add(line)
                continue
            }

            val lastY = currentLines.last().bbox.centerY()
            val currentY = line.bbox.centerY()

            // 如果距離太遠，開始新的五線譜
            if (currentY - lastY > STAFF_LINE_SPACING * 3) {
                if (currentLines.size >= linesPerStaff) {
                    staves.add(createStaffRegion(staves.size, currentLines))
                }
                currentLines = mutableListOf(line)
            } else {
                currentLines.add(line)
            }
        }

        // 處理最後一組
        if (currentLines.size >= linesPerStaff) {
            staves.add(createStaffRegion(staves.size, currentLines))
        }

        return staves
    }

    /**
     * 建立五線譜區域
     */
    private fun createStaffRegion(index: Int, lines: List<Detection>): StaffRegion {
        val topY = lines.first().bbox.centerY()
        val bottomY = lines.last().bbox.centerY()

        return StaffRegion(
            index = index,
            topY = topY,
            bottomY = bottomY
        )
    }

    /**
     * 解析全域資訊（調號、拍號）
     */
    private fun parseGlobalInfo(detections: List<Detection>) {
        val keySignatureSymbols = detections.filter { it.className == "key_signature" }
        val timeSignatureSymbols = detections.filter { it.className == "time_signature" }

        // 目前模型類別只輸出 key_signature / time_signature，未拆出具體值，先保守給預設值。
        detectedKeySignature = KeySignature(tonicMidi = MIDI_C4, mode = Mode.MAJOR)
        detectedTimeSignature = TimeSignature(numerator = 4, denominator = 4)

        Log.d(
            TAG,
            "解析全域資訊: Key=C major, Time=4/4 " +
                "(key_signature=${keySignatureSymbols.size}, time_signature=${timeSignatureSymbols.size})"
        )
    }

    /**
     * 將音符分配到五線譜
     */
    private fun assignNotesToStaves(
        noteheads: List<Detection>,
        staves: List<StaffRegion>
    ): Map<Int, List<Detection>> {
        return noteheads.groupBy { notehead ->
            val centerY = notehead.bbox.centerY()

            // 找最接近的五線譜
            staves.minByOrNull { staff ->
                val staffCenterY = (staff.topY + staff.bottomY) / 2
                kotlin.math.abs(centerY - staffCenterY)
            }?.index ?: 0
        }
    }

    /**
     * 建立 ChordSnapshot 列表
     */
    private fun buildChordSnapshots(
        notesByStaff: Map<Int, List<Detection>>,
        barlines: List<Detection>,
        accidentals: List<Detection>
    ): List<ChordSnapshot> {
        val chords = mutableListOf<ChordSnapshot>()
        val allNotes = notesByStaff.values.flatten().sortedBy { it.bbox.centerX() }
        if (allNotes.isEmpty()) {
            return emptyList()
        }
        if (accidentals.isNotEmpty()) {
            Log.d(TAG, "偵測到 ${accidentals.size} 個升降記號，暫未套用到音高推斷")
        }

        val barlineXs = barlines.map { it.bbox.centerX() }.sorted()
        val groupedNotes = allNotes.groupBy { detection ->
            val x = detection.bbox.centerX()
            val measure = barlineXs.count { it < x } + 1
            val beat = estimateBeat(x, measure, barlineXs)
            measure to beat
        }

        var chordIndex = 0
        groupedNotes
            .toList()
            .sortedWith(compareBy({ it.first.first }, { it.first.second }))
            .forEach { (position, detections) ->
                if (detections.size < 4) {
                    return@forEach
                }
                val (measure, beat) = position
                val sortedByY = detections.sortedBy { it.bbox.centerY() }.take(4)
                val voiceOrder = listOf(Voice.SOPRANO, Voice.ALTO, Voice.TENOR, Voice.BASS)
                val notes = voiceOrder.mapIndexed { index, voice ->
                    val detection = sortedByY[index]
                    voice to NoteEvent(
                        voice = voice,
                        midi = calculateMidiPitch(detection.bbox.centerY()),
                        measure = measure,
                        beat = beat
                    )
                }.toMap()

                chords.add(
                    ChordSnapshot(
                        index = chordIndex++,
                        measure = measure,
                        beat = beat,
                        notes = notes
                    )
                )
            )
        return chords
    }

    private fun estimateBeat(
        x: Float,
        measure: Int,
        barlineXs: List<Float>
    ): Double {
        if (barlineXs.isEmpty()) {
            return (1 + (x / 200f).toInt().coerceIn(0, 3)).toDouble()
        }

        val left = if (measure <= 1) 0f else barlineXs.getOrElse(measure - 2) { 0f }
        val right = barlineXs.getOrElse(measure - 1) { left + 800f }
        val width = (right - left).takeIf { it > 1f } ?: 800f
        val relative = ((x - left) / width).coerceIn(0f, 0.999f)
        return (1 + (relative * 4).toInt()).toDouble()
    }

    /**
     * 計算 MIDI 音高（基於 Y 座標）
     */
    private fun calculateMidiPitch(y: Float): Int {
        val semitoneStep = ((500f - y) / (STAFF_LINE_SPACING / 2f)).toInt()
        return (MIDI_C4 + semitoneStep).coerceIn(36, 84)
    }
}

/**
 * 五線譜區域
 */
data class StaffRegion(
    val index: Int,
    val topY: Float,
    val bottomY: Float
)
