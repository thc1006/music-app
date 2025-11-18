# Android 子專案說明（草案）

本資料夾將承載 Android APP 的實作，目標裝置為：  
Android 10 以上、中階手機（4GB RAM 以上）為主。

## 目標功能

1. 拍照或選取樂譜圖片。  
2. 呼叫雲端 OMR / LLM Proxy API 將圖片轉為和聲結構 JSON。  
3. 使用 Kotlin 規則引擎檢查四部和聲錯誤。  
4. 在畫面上標示錯誤位置，並顯示中文說明。

## 建議技術

- 語言：Kotlin  
- UI：Jetpack Compose 或傳統 View（以 Compose 優先）  
- 網路：Retrofit / Ktor client（任選其一，請保持整體一致性）  
- JSON：kotlinx.serialization 或 Moshi

## 目前檔案

- core/harmony/HarmonyModels.kt  
  - 資料模型與規則引擎進入點 skeleton。

- core/omr/OmrClient.kt  
  - OMR 雲端 API 介面與 HTTP client skeleton。

後續可依照實際需要新增 ViewModel、UI composable 等檔案。
