# 四部和聲助教 Android App

這是「四部和聲助教」的 Android 應用程式，使用 Kotlin + Jetpack Compose 開發。

## 專案概述

本應用允許音樂班與音樂系學生：
1. 拍攝或上傳四部和聲作業的樂譜照片
2. 透過雲端 OMR / 多模態 LLM 將圖片轉為結構化資料
3. 在手機端使用規則引擎檢查和聲錯誤
4. 視覺化顯示錯誤位置與中文說明

## 技術棧

- **語言**: Kotlin 1.9.20
- **最低 SDK**: Android 8.0 (API 26)
- **目標 SDK**: Android 14 (API 34)
- **UI 框架**: Jetpack Compose (Material 3)
- **網路**: Retrofit 2.9.0 + OkHttp 4.12.0
- **序列化**: kotlinx.serialization 1.6.0
- **相機**: CameraX 1.3.0
- **圖片載入**: Coil 2.5.0
- **導航**: Navigation Compose 2.7.5

## 專案結構

```
android-app/
├── app/
│   ├── src/
│   │   └── main/
│   │       ├── AndroidManifest.xml
│   │       ├── java/com/example/harmonychecker/
│   │       │   ├── MainActivity.kt                    # 主 Activity
│   │       │   ├── core/                              # 核心業務邏輯
│   │       │   │   ├── harmony/                       # 和聲規則引擎
│   │       │   │   │   └── HarmonyModels.kt          # 資料模型與規則引擎
│   │       │   │   └── omr/                           # OMR 客戶端
│   │       │   │       └── OmrClient.kt              # OMR API 介面
│   │       │   └── ui/                                # UI 層
│   │       │       ├── HarmonyApp.kt                 # 主應用導航
│   │       │       ├── screens/                       # 各畫面
│   │       │       │   └── MainScreen.kt             # 主畫面
│   │       │       └── theme/                         # 主題設定
│   │       │           ├── Theme.kt
│   │       │           └── Type.kt
│   │       └── res/                                   # 資源檔案
│   │           ├── values/
│   │           │   ├── strings.xml                   # 字串資源（中文）
│   │           │   ├── colors.xml                    # 顏色定義
│   │           │   └── themes.xml                    # 主題
│   │           └── xml/
│   │               ├── backup_rules.xml
│   │               ├── data_extraction_rules.xml
│   │               └── file_paths.xml                # FileProvider 路徑
│   ├── build.gradle.kts                              # App 模組建置設定
│   └── proguard-rules.pro                            # ProGuard 規則
├── build.gradle.kts                                  # 專案級建置設定
├── settings.gradle.kts                               # 專案設定
├── gradle.properties                                 # Gradle 屬性
└── .gitignore                                        # Git 忽略規則
```

## 建置與執行

### 前置需求

1. Android Studio Hedgehog (2023.1.1) 或更新版本
2. JDK 17 或更新版本
3. Android SDK (API 26-34)

### 建置步驟

1. 使用 Android Studio 開啟 `android-app` 資料夾
2. 等待 Gradle sync 完成
3. 連接 Android 裝置或啟動模擬器
4. 點擊 Run 按鈕（或按 Shift+F10）

### 或使用命令列：

```bash
cd android-app
./gradlew assembleDebug        # 建置 Debug APK
./gradlew installDebug          # 安裝到已連接的裝置
```

## 開發狀態

### ✅ 已完成

- [x] 完整的 Gradle 建置設定
- [x] AndroidManifest 與權限宣告
- [x] Material 3 主題與顏色系統
- [x] 主畫面 UI（拍照/選圖按鈕）
- [x] Navigation 導航骨架
- [x] 核心資料模型定義
- [x] OMR Client 介面定義

### 🚧 進行中

- [ ] Kotlin 規則引擎實作（移植自 Python）
- [ ] 相機拍照功能
- [ ] 照片選擇功能
- [ ] OMR API HTTP 客戶端實作
- [ ] 結果顯示畫面
- [ ] 錯誤標記 Overlay

### 📋 待辦事項

- [ ] 樂譜圖片預處理
- [ ] 離線快取機制
- [ ] 單元測試
- [ ] UI/UX 優化
- [ ] 效能優化

## 與 Python 規則引擎的對應

Kotlin 版規則引擎位於 `core/harmony/HarmonyModels.kt`，設計目標是與專案根目錄的 `harmony_rules.py` 保持邏輯一致。

主要對應關係：

| Python | Kotlin |
|--------|--------|
| `NoteEvent` | `NoteEvent` |
| `ChordSnapshot` | `ChordSnapshot` |
| `KeySignature` | `KeySignature` |
| `RuleViolation` | `HarmonyIssue` |
| `HarmonyAnalyzer` | `HarmonyRuleEngine` |

## API 端點設定

OMR 雲端 API 端點設定位於 `OmrClient.kt`。實際部署時需提供：

- 基礎 URL (例如: `https://your-cloud-function.com`)
- API 端點: `/api/omr/score`
- 認證機制（API key 或 JWT）

詳細 API 規格請參考 `docs/omr_proxy_api.md`。

## 權限說明

本應用需要以下權限：

- **CAMERA**: 拍攝樂譜照片
- **INTERNET**: 呼叫雲端 OMR API
- **READ_MEDIA_IMAGES**: 從相簿選擇照片

所有權限都會在執行時請求（Runtime Permissions）。

## 授權

本專案採用 Apache License 2.0 授權，詳見專案根目錄的 LICENSE 檔案。

## 相關文件

- [專案概述](../專案概述.md)
- [和聲規則說明](../harmony_rules_zh.md)
- [OMR API 規格](../docs/omr_proxy_api.md)
- [開發指南](../CLAUDE.md)
