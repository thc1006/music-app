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
        // 簡化實作：假設 C major, 4/4
        // TODO: 實際解析調號與拍號符號
        detectedKeySignature = KeySignature(tonicMidi = MIDI_C4, mode = Mode.MAJOR)
        detectedTimeSignature = TimeSignature(numerator = 4, denominator = 4)

        Log.d(TAG, "解析全域資訊: Key=C major, Time=4/4")
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

        // 簡化實作：將所有音符放入單一 chord
        // TODO: 根據時間位置（X 座標）分組

        val allNotes = notesByStaff.values.flatten().sortedBy { it.bbox.centerX() }

        if (allNotes.isEmpty()) {
            return emptyList()
        }

        // 建立測試用 ChordSnapshot
        val noteEvents = allNotes.take(4).mapIndexed { index, detection ->
            val voice = when (index) {
                0 -> Voice.SOPRANO
                1 -> Voice.ALTO
                2 -> Voice.TENOR
                else -> Voice.BASS
            }

            // 簡化音高計算：基於 Y 座標
            val midi = calculateMidiPitch(detection.bbox.centerY())

            NoteEvent(
                voice = voice,
                midi = midi,
                measure = 1,
                beat = 1.0
            )
        }

        if (noteEvents.size == 4) {
            val notes = mapOf(
                Voice.SOPRANO to noteEvents[0],
                Voice.ALTO to noteEvents[1],
                Voice.TENOR to noteEvents[2],
                Voice.BASS to noteEvents[3]
            )

            chords.add(
                ChordSnapshot(
                    index = 0,
                    measure = 1,
                    beat = 1.0,
                    notes = notes
                )
            )
        }

        return chords
    }

    /**
     * 計算 MIDI 音高（基於 Y 座標）
     */
    private fun calculateMidiPitch(y: Float): Int {
        // 簡化實作：線性映射
        // TODO: 實際根據五線譜位置計算
        val normalized = (y / 1000f).coerceIn(0f, 1f)
        return MIDI_C4 + ((1 - normalized) * 24).toInt()  // C4 上下兩個八度
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
