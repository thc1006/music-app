package com.example.harmonychecker.core.omr

import com.example.harmonychecker.core.harmony.ChordSnapshot
import com.example.harmonychecker.core.harmony.NoteEvent
import com.example.harmonychecker.core.harmony.Voice
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.util.Base64

/**
 * OMR / 多模態 LLM 代理 API 介面與 HTTP client 骨架。
 *
 * 實際實作時，請依照 docs/omr_proxy_api.md 中定義的 JSON 結構進行序列化 / 反序列化。
 */

data class OmrResult(
    val chords: List<ChordSnapshot>,
    val rawJson: String? = null,
    val warnings: List<String> = emptyList()
)

interface OmrClient {

    /**
     * 將樂譜圖片（壓縮後的位元組陣列）送至雲端 OMR 代理 API，
     * 並回傳對應的 ChordSnapshot 陣列與原始 JSON。
     */
    suspend fun recognizeScore(imageBytes: ByteArray): OmrResult
}

/**
 * HttpOmrClient 是一個平台無關的 HTTP client 包裝。
 *
 * 實作時可以選擇 Retrofit、Ktor client 等，
 * 但請盡量將第三方套件依賴集中在 app 層或 data 層，
 * 核心邏輯保持乾淨與可測試。
 */
class HttpOmrClient(
    private val baseUrl: String,
    // 這裡不直接綁定特定 HTTP library，交由實作端決定。
    private val httpPost: suspend (url: String, body: ByteArray, contentType: String) -> String,
    private val json: Json = Json { ignoreUnknownKeys = true }
) : OmrClient {

    override suspend fun recognizeScore(imageBytes: ByteArray): OmrResult {
        val url = "$baseUrl/api/omr/score"
        val requestBody = json.encodeToString(
            OmrRequest(
                imageBase64 = Base64.getEncoder().encodeToString(imageBytes),
                options = OmrOptions(languageHint = "zh-TW")
            )
        ).toByteArray(Charsets.UTF_8)

        val responseText = httpPost(url, requestBody, "application/json; charset=utf-8")
        val response = json.decodeFromString<OmrResponse>(responseText)
        val warnings = response.warnings + if (response.measures.isEmpty()) {
            listOf("OMR 回應未包含任何 measures")
        } else {
            emptyList()
        }

        return OmrResult(
            chords = response.toChordSnapshots(),
            rawJson = responseText,
            warnings = warnings
        )
    }
}

@Serializable
private data class OmrRequest(
    @SerialName("image_base64")
    val imageBase64: String,
    val filename: String? = null,
    val options: OmrOptions? = null
)

@Serializable
private data class OmrOptions(
    @SerialName("staff_layout")
    val staffLayout: String = "SATB_GRAND_STAFF",
    @SerialName("expected_voices")
    val expectedVoices: List<String> = listOf("S", "A", "T", "B"),
    @SerialName("language_hint")
    val languageHint: String = "zh-TW"
)

@Serializable
private data class OmrResponse(
    val measures: List<MeasureDto> = emptyList(),
    @SerialName("raw_model_output")
    val rawModelOutput: String? = null,
    val warnings: List<String> = emptyList()
)

@Serializable
private data class MeasureDto(
    val index: Int,
    val chords: List<ChordDto> = emptyList()
)

@Serializable
private data class ChordDto(
    val beat: Double,
    val notes: List<NoteDto> = emptyList()
)

@Serializable
private data class NoteDto(
    val voice: String,
    val pitch: String,
    val duration: Double
)

private fun OmrResponse.toChordSnapshots(): List<ChordSnapshot> {
    var chordIndex = 0
    return measures
        .sortedBy { it.index }
        .flatMap { measure ->
            measure.chords.map { chord ->
                val notes = chord.notes.associate { note ->
                    val voice = Voice.fromString(note.voice)
                    voice to NoteEvent(
                        voice = voice,
                        midi = pitchToMidi(note.pitch),
                        measure = measure.index,
                        beat = chord.beat,
                        duration = note.duration
                    )
                }
                ChordSnapshot(
                    index = chordIndex++,
                    measure = measure.index,
                    beat = chord.beat,
                    notes = notes
                )
            }
        }
}

private val PITCH_REGEX = Regex("^([A-Ga-g])([#b]?)(-?\\d+)$")

private fun pitchToMidi(pitch: String): Int {
    val match = PITCH_REGEX.matchEntire(pitch)
        ?: throw IllegalArgumentException("無法解析音高格式: $pitch")
    val note = match.groupValues[1].uppercase()
    val accidental = match.groupValues[2]
    val octave = match.groupValues[3].toInt()

    val basePc = when (note) {
        "C" -> 0
        "D" -> 2
        "E" -> 4
        "F" -> 5
        "G" -> 7
        "A" -> 9
        "B" -> 11
        else -> throw IllegalArgumentException("不支援的音名: $note")
    }
    val adjustedPc = when (accidental) {
        "#" -> basePc + 1
        "b" -> basePc - 1
        else -> basePc
    }
    return (octave + 1) * 12 + adjustedPc
}
