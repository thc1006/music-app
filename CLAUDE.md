# CLAUDE.md — 四部和聲助教（Android + 規則引擎）

你是這個專案的長期協作夥伴與工程師，負責協助我完成：  
「從樂譜照片，到四部和聲錯誤標示與文字說明」的完整系統。

---
## 0. 專案一句話

開發一個給音樂班／音樂系學生使用的 Android APP：
拍照或上傳四部和聲作業 → **端側 YOLO12 深度學習模型解讀樂譜** → 手機端規則引擎檢查 → 在樂譜上標出錯誤並給中文解釋。

**🔥 重要架構決策（2025-11-20）**：採用 **完全端側運算** 架構
- OMR 辨識：使用 YOLO12s/n + TensorFlow Lite INT8 量化，完全在手機上運行
- 無雲端依賴：無需後端伺服器，完全離線運作
- 訓練資源：使用 RTX 5060 GPU 訓練自定義 YOLO12 模型
- 目標裝置：支援 2025 年所有等級 Android 手機（含平價機）

---
## 1. 目前重要檔案

根目錄關鍵檔案：

- README.md  
  專案簡介與 rule engine 的定位。

- harmony_rules.py  
  Python 版四部和聲規則引擎，是「邏輯與行為」的權威實作來源。

- harmony_rules_zh.md  
  每一條規則的中文說明與 rationale。新增或修改規則時，必須與程式同步更新。

- 專案概述.md  
  產品定位、目標使用者、使用情境與功能說明。

- CLAUDE.md（本檔）  
  給 Claude Code 的「憲法」：專案目標、限制、推薦工作流程與你可以做的事。

docs 資料夾：

- docs/yolo12_omr_implementation.md ⭐ **NEW**
  YOLO12 端側 OMR 完整實作規劃：訓練、量化、Android 部署、多裝置適配策略。

- docs/omr_proxy_api.md ⚠️ **DEPRECATED**
  （已棄用）原雲端 API 設計，保留作為參考。

- docs/omr_prompt_gemini.md ⚠️ **DEPRECATED**
  （已棄用）原 LLM prompt 設計，保留作為參考。

訓練資料與腳本：

- training/
  - yolo12_train.py ⭐ **NEW** - YOLO12 訓練主腳本（RTX 5060）
  - omr_harmony.yaml ⭐ **NEW** - 資料集配置
  - export_models.py ⭐ **NEW** - 模型匯出與量化腳本
  - requirements-train.txt ⭐ **NEW** - 訓練環境依賴

Android 核心：

- android-app/README.md
  Android 子專案結構說明與建置方向。

- android-app/core/harmony/HarmonyModels.kt
  Kotlin 版資料模型與規則引擎（已完成 2000+ 行）。

- android-app/core/omr/OmrClient.kt
  OMR 介面定義（端側推論）。

- android-app/core/omr/Yolo12OmrClient.kt ⭐ **NEW**
  YOLO12 TFLite 推論實作。

- android-app/core/omr/SymbolAssembler.kt ⭐ **NEW**
  符號檢測結果組裝成 ChordSnapshot 的邏輯。

---
## 2. 目標架構與流程

### 2.1 資料流（端側運算架構）

1. 使用者在 Android APP：拍照或選擇一張樂譜圖片（四部和聲作業）。
2. **圖像預處理**（手機端）：
   - 調整解析度至 640x640
   - 灰階化與對比增強
   - 透視矯正（可選）
3. **YOLO12 推論**（手機端 TFLite）：
   - 使用 INT8 量化模型進行符號檢測
   - 偵測音符頭、譜號、調號、拍號、升降記號等
   - 輸出 bounding boxes + 類別 + 信心度
4. **符號組裝**（手機端）：
   - 根據檢測結果的空間位置排序
   - 組合成 ChordSnapshot 列表（小節、拍點、SATB 音高）
   - 處理特殊情況（連結線、延音線等）
5. **規則引擎檢查**（手機端）：
   - Kotlin HarmonyRuleEngine 執行所有和聲規則
   - 產生 HarmonyIssue 列表
6. **UI 顯示**：
   - 錯誤位置 overlay 在原始樂譜上
   - 顯示中文錯誤說明與建議

**關鍵優勢**：完全離線、無網路延遲、保護隱私、無雲端成本。

### 2.2 設計原則（更新）

- **完全端側運算**：所有運算（OMR + 規則檢查）在手機上完成，無雲端依賴。
- **多裝置適配**：
  - 使用 INT8 量化確保低階手機可運行
  - 準備 YOLO12n（極輕量）和 YOLO12s（高準確度）雙模型
  - 根據裝置效能動態選擇模型
- **無敏感資訊**：所有資料留在本地，無 API key 或網路傳輸。
- **音樂規則權威性**：規則以 `harmony_rules.py` + `harmony_rules_zh.md` 為準。
- **模型可更新性**：TFLite 模型可透過 App 更新機制升級，無需重裝 App。

---
## 3. 預期目錄結構

完整專案預期結構（目前已部分建立）：

- /README.md  
- /harmony_rules.py  
- /harmony_rules_zh.md  
- /專案概述.md  
- /CLAUDE.md  
- /docs/  
  - omr_proxy_api.md  
  - omr_prompt_gemini.md  
- /android-app/  
  - README.md  
  - core/  
    - harmony/  
      - HarmonyModels.kt  
    - omr/  
      - OmrClient.kt  
  - 其他 Android 專案檔案（之後逐步補齊）

你在新增檔案或資料夾時，若改變高層結構，請盡量同步更新此說明。

---
## 4. 行為準則（Behavior Constraints）

1. 先規劃再動手：大型變更前先用條列步驟說明計畫，取得簡短確認後再實作。  
2. 不擅自更改音樂理論本質：若規則定義有疑慮，標記為「需要作者確認」，不要自行改寫理論。  
3. 修改規則時必須同步：
   - 更新 harmony_rules.py
   - 更新 harmony_rules_zh.md
   - 若 Kotlin 版已有對應實作，也要同步更新。  
4. 不在程式碼中放入私密資訊（API key、密碼、token）。  
5. 優先採用清楚小檔案與模組化結構，避免單一超大檔案。

---
## 5. 你可以執行的技能（Skills）

### Skill A：維護與擴充和聲規則

- 依據 harmony_rules_zh.md 的敘述，修改或新增 harmony_rules.py 規則。
- 為每條規則設計簡單測試資料（正確 / 錯誤案例），可在之後轉為自動化測試。
- 幫忙整理規則分類（旋律、聲部間關係、和弦結構、終止式判定等）。

### Skill B：將 Python 規則翻成 Kotlin

- 在 android-app/core/harmony/ 中：
  - 宣告與 Python 對應的資料結構（NoteEvent、ChordSnapshot、HarmonyIssue 等）。
  - 實作與 Python 邏輯等價的規則檢查骨架或完整實作。  
- Kotlin 端註解中標明對應的 Python 函式名稱或規則編號。

### Skill C：設計與實作 OMR / LLM Proxy 介面

- 在 android-app/core/omr/：
  - 定義 OmrClient 介面（suspend 函式）。
  - 實作 HttpOmrClient，封裝與後端 serverless OMR 代理 API 的溝通。  
- 遵守 docs/omr_proxy_api.md 所定義的 JSON Schema。

### Skill D：Android UI / UX 協助

- 設計並實作：
  - 拍照 / 選圖畫面  
  - 辨識中 loading 狀態  
  - 顯示錯誤標記的樂譜檢視畫面  
  - 錯誤列表與文字說明（中文為主，可附英文）。

### Skill E：工具腳本與測試資料產生

- 撰寫 Python 或 Kotlin 工具：
  - 將 MusicXML 轉成 ChordSnapshot 陣列。  
  - 批次產生測試資料集，用於驗證規則引擎與 OMR 輸出的一致性。

---
## 6. MCP 與外部工具（若已設定）

若在 Claude Code 中有設定以下 MCP server，可以使用：

- filesystem / git 類 MCP：  
  - 瀏覽、修改專案檔案與查看 Git 歷史。

- OCR / PDF 類 MCP：  
  - 將老師提供的 PDF 題庫轉成文字資料。

- HTTP / fetch 類 MCP：  
  - 在開發階段直接呼叫雲端 OMR proxy API 測試。

不要假設 MCP 一定存在；若不可用，退回使用本地檔案與 shell 指令。

---
## 7. 推薦開發流程

每次被要求執行一項任務時，請遵循：

1. 閱讀相關檔案（至少：README.md、harmony_rules.py、harmony_rules_zh.md、專案概述.md）。  
2. 用條列方式提出計畫（檔案會改哪幾個、預計步驟）。  
3. 在使用者簡短確認後，分步實作並說明變更內容。  
4. 若有測試框架，協助撰寫與執行測試。  
5. 重要行為變更時，更新 README.md 與本 CLAUDE.md。

---
## 8. Roadmap（更新為端側 YOLO12 架構）

### Phase 1: YOLO12 訓練與基礎整合（Week 1-3）✅ **當前階段**

1. **資料準備** (Week 1, Day 1-2):
   - 下載 MUSCIMA++, DeepScoresV2 資料集
   - 準備標註格式轉換（YOLO format）
   - 建立訓練/驗證/測試集分割

2. **模型訓練** (Week 1, Day 3-7):
   - RTX 5060 訓練 YOLO12s (200-250 epochs)
   - 同步訓練 YOLO12n 作為備援 (150-200 epochs)
   - 驗證準確度與調參

3. **模型匯出與量化** (Week 2, Day 1-2):
   - 匯出 TFLite INT8 量化模型（YOLO12s, YOLO12n）
   - 驗證量化後準確度損失 < 2%
   - 測試模型大小與推論速度

4. **Android TFLite 整合** (Week 2, Day 3-7):
   - 建立 Yolo12OmrClient.kt
   - 整合 TensorFlow Lite Interpreter
   - 實作推論 pipeline（前處理 + 推論 + 後處理）

5. **符號組裝邏輯** (Week 3, Day 1-5):
   - 實作 SymbolAssembler.kt
   - 空間位置排序與五線譜解析
   - 生成 ChordSnapshot 列表

6. **UI 整合與測試** (Week 3, Day 6-7):
   - 串接 CameraX + YOLO12 + HarmonyRuleEngine
   - 初步多裝置測試

### Phase 2: 多裝置優化與降級策略（Week 4-5）

1. **裝置效能分析**:
   - 在低階（SD 6 Gen 1）、中階（SD 7 Gen 3）、高階手機上實測
   - 收集推論時間、記憶體使用、準確度數據

2. **動態模型選擇**:
   - 實作裝置檢測與效能評分
   - 低階機自動降級到 YOLO12n
   - 中高階機使用 YOLO12s

3. **準確度提升**:
   - 根據實測結果 fine-tuning 模型
   - 收集錯誤案例重新訓練
   - 提升符號組裝邏輯健壯性

### Phase 3: 規則覆蓋與教材整合（Week 6+）

1. **規則引擎擴充**:
   - 補齊剩餘和聲規則
   - 處理更多音樂記號（表情、力度等）
   - 實際作業測試集驗證

2. **使用者體驗優化**:
   - 錯誤標記 UI 精緻化
   - 中文說明文字優化
   - 互動式教學功能

3. **模型持續改進**:
   - 建立使用者反饋機制
   - 定期更新模型（透過 App 更新）
   - 擴充訓練資料集

---

**當前進度**：Phase 1 啟動，正在建立訓練腳本與資料集配置。

若此檔案與實際專案結構不一致，以使用者指示為準，並在後續修改中更新本檔內容。
