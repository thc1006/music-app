package com.example.harmonychecker.core.omr

import com.example.harmonychecker.core.harmony.ChordSnapshot

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
    private val httpPost: suspend (url: String, body: ByteArray, contentType: String) -> String
) : OmrClient {

    override suspend fun recognizeScore(imageBytes: ByteArray): OmrResult {
        // TODO: 序列化成符合 docs/omr_proxy_api.md 要求的 JSON，
        // 例如：{"image_base64": "...", "options": {...}}
        val url = "$baseUrl/api/omr/score"

        // 這裡僅示範呼叫，實際上應該使用 JSON 字串作為 body。
        val requestBody = imageBytes // placeholder

        val responseText = httpPost(url, requestBody, "application/json")

        // TODO: 將 responseText parse 成 OmrResult
        // 目前先回傳空結構。
        return OmrResult(
            chords = emptyList(),
            rawJson = responseText,
            warnings = listOf("HttpOmrClient.recognizeScore 尚未實作 JSON 解析")
        )
    }
}
