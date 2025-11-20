package com.example.harmonychecker.core.omr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import com.example.harmonychecker.core.harmony.ChordSnapshot
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * YOLO12 TFLite 端側 OMR 客戶端
 *
 * 功能:
 * 1. 載入 INT8 量化 YOLO12 模型
 * 2. 執行樂譜符號檢測推論
 * 3. NMS 後處理
 * 4. 輸出檢測結果供 SymbolAssembler 使用
 *
 * @param context Android Context
 * @param modelVariant 模型變體（YOLO12S 或 YOLO12N）
 * @param useGpuDelegate 是否使用 GPU 加速
 */
class Yolo12OmrClient(
    private val context: Context,
    private val modelVariant: ModelVariant = ModelVariant.YOLO12S,
    private val useGpuDelegate: Boolean = true
) : OmrClient {

    companion object {
        private const val TAG = "Yolo12OmrClient"

        // 模型配置
        private const val INPUT_SIZE = 640
        private const val NUM_CLASSES = 20
        private const val CONFIDENCE_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f

        // 類別名稱（對應 omr_harmony.yaml）
        private val CLASS_NAMES = listOf(
            "notehead_filled", "notehead_hollow",
            "stem_up", "stem_down", "beam", "flag",
            "clef_treble", "clef_bass", "clef_alto", "clef_tenor",
            "accidental_sharp", "accidental_flat", "accidental_natural",
            "rest_quarter", "rest_half", "rest_whole",
            "barline", "time_signature", "key_signature", "staffline"
        )
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var isInitialized = false

    /**
     * 初始化模型
     */
    fun initialize() {
        if (isInitialized) return

        val modelPath = when (modelVariant) {
            ModelVariant.YOLO12S -> "models/yolo12s_int8.tflite"
            ModelVariant.YOLO12N -> "models/yolo12n_int8.tflite"
        }

        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)

                if (useGpuDelegate) {
                    try {
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate)
                        Log.i(TAG, "GPU delegate 初始化成功")
                    } catch (e: Exception) {
                        Log.w(TAG, "GPU delegate 初始化失敗，使用 CPU: ${e.message}")
                    }
                }
            }

            val modelBuffer = FileUtil.loadMappedFile(context, modelPath)
            interpreter = Interpreter(modelBuffer, options)
            isInitialized = true

            Log.i(TAG, "YOLO12 模型載入成功: $modelPath")

        } catch (e: Exception) {
            Log.e(TAG, "模型載入失敗", e)
            throw RuntimeException("無法載入 YOLO12 模型: ${e.message}")
        }
    }

    /**
     * OmrClient 介面實作
     * 注意：這裡的參數是 ByteArray，但我們需要 Bitmap
     * 實際使用時需要轉換
     */
    override suspend fun recognizeScore(imageBytes: ByteArray): OmrResult {
        // TODO: 將 ByteArray 轉換為 Bitmap
        // 這裡暫時回傳空結果
        Log.w(TAG, "recognizeScore(ByteArray) 尚未完整實作")

        return OmrResult(
            chords = emptyList(),
            rawJson = null,
            warnings = listOf("Yolo12OmrClient 需要 Bitmap 輸入")
        )
    }

    /**
     * 主要識別方法（使用 Bitmap）
     */
    suspend fun recognizeScore(bitmap: Bitmap): OmrResult {
        if (!isInitialized) {
            initialize()
        }

        val detections = detect(bitmap)

        // TODO: 使用 SymbolAssembler 組裝成 ChordSnapshot
        // 目前先回傳空的 chords

        return OmrResult(
            chords = emptyList(),
            rawJson = "Detected ${detections.size} symbols",
            warnings = if (detections.isEmpty()) {
                listOf("未檢測到任何符號")
            } else {
                emptyList()
            }
        )
    }

    /**
     * YOLO12 推論
     */
    private fun detect(bitmap: Bitmap): List<Detection> {
        val startTime = System.currentTimeMillis()

        // 1. 預處理
        val inputBuffer = preprocessImage(bitmap)

        // 2. 推論
        val outputBuffer = runInference(inputBuffer)

        // 3. 後處理（NMS）
        val detections = postprocess(outputBuffer, bitmap.width, bitmap.height)

        val elapsed = System.currentTimeMillis() - startTime
        Log.d(TAG, "YOLO12 推論完成: ${detections.size} 個檢測, 耗時 ${elapsed}ms")

        return detections
    }

    /**
     * 圖像預處理
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        val byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // 轉換為浮點數並正規化 [0, 1]
        for (pixelValue in intValues) {
            val r = ((pixelValue shr 16) and 0xFF) / 255.0f
            val g = ((pixelValue shr 8) and 0xFF) / 255.0f
            val b = (pixelValue and 0xFF) / 255.0f

            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        return byteBuffer
    }

    /**
     * 執行推論
     */
    private fun runInference(input: ByteBuffer): Array<FloatArray> {
        val outputArray = Array(1) { FloatArray(8400 * (NUM_CLASSES + 4)) }

        interpreter?.run(input, outputArray)

        return Array(8400) { i ->
            FloatArray(NUM_CLASSES + 4) { j ->
                outputArray[0][i * (NUM_CLASSES + 4) + j]
            }
        }
    }

    /**
     * 後處理與 NMS
     */
    private fun postprocess(
        output: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()

        // 解析 YOLO 輸出
        for (row in output) {
            val centerX = row[0]
            val centerY = row[1]
            val width = row[2]
            val height = row[3]

            val classScores = row.sliceArray(4 until row.size)
            val maxScore = classScores.maxOrNull() ?: 0f
            val classId = classScores.indexOf(maxScore)

            if (maxScore < CONFIDENCE_THRESHOLD) continue

            // 座標還原
            val scaleX = originalWidth.toFloat() / INPUT_SIZE
            val scaleY = originalHeight.toFloat() / INPUT_SIZE

            val bbox = RectF(
                (centerX - width / 2) * scaleX,
                (centerY - height / 2) * scaleY,
                (centerX + width / 2) * scaleX,
                (centerY + height / 2) * scaleY
            )

            detections.add(
                Detection(
                    bbox = bbox,
                    classId = classId,
                    className = CLASS_NAMES.getOrElse(classId) { "unknown" },
                    confidence = maxScore
                )
            )
        }

        // NMS
        return nms(detections, IOU_THRESHOLD)
    }

    /**
     * Non-Maximum Suppression
     */
    private fun nms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        val suppressed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (suppressed[i]) continue

            selected.add(sorted[i])

            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (calculateIoU(sorted[i].bbox, sorted[j].bbox) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }

        return selected
    }

    /**
     * 計算 IoU
     */
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersection = RectF(box1)
        if (!intersection.intersect(box2)) return 0f

        val intersectionArea = intersection.width() * intersection.height()
        val box1Area = box1.width() * box1.height()
        val box2Area = box2.width() * box2.height()
        val unionArea = box1Area + box2Area - intersectionArea

        return intersectionArea / unionArea
    }

    /**
     * 清理資源
     */
    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        isInitialized = false
        Log.i(TAG, "YOLO12 資源已釋放")
    }

    /**
     * 模型變體
     */
    enum class ModelVariant {
        YOLO12S,  // 高準確度，~10MB
        YOLO12N   // 輕量級，~3MB
    }
}

/**
 * 檢測結果
 */
data class Detection(
    val bbox: RectF,
    val classId: Int,
    val className: String,
    val confidence: Float
)
