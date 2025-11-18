# CLAUDE.md — 四部和聲助教（Android + 規則引擎）

你是這個專案的長期協作夥伴與工程師，負責協助我完成：  
「從樂譜照片，到四部和聲錯誤標示與文字說明」的完整系統。

---
## 0. 專案一句話

開發一個給音樂班／音樂系學生使用的 Android APP：  
拍照或上傳四部和聲作業 → 雲端 OMR / 多模態模型解讀樂譜 → 手機端規則引擎檢查 → 在樂譜上標出錯誤並給中文解釋。

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

- docs/omr_proxy_api.md  
  雲端 OMR / Vision 代理 API 的 JSON 結構與端點設計。

- docs/omr_prompt_gemini.md  
  針對多模態 LLM（例如 Gemini）設計的 prompt 草稿，讓模型輸出結構化 JSON。

Android skeleton：

- android-app/README.md  
  Android 子專案結構說明與建置方向。

- android-app/core/harmony/HarmonyModels.kt  
  Kotlin 版資料模型與規則引擎進入點骨架。

- android-app/core/omr/OmrClient.kt  
  OMR 雲端 API 介面與 HTTP client 骨架。

---
## 2. 目標架構與流程

### 2.1 資料流

1. 使用者在 Android APP：拍照或選擇一張樂譜圖片（四部和聲作業）。
2. APP 將圖片壓縮後打到「雲端 OMR / Vision 代理 API」（serverless endpoint）：
   - 由後端呼叫多模態 LLM（例如 Gemini）或其他 OMR 服務。
   - 產生「調號、拍號、每一拍的 S/A/T/B 音高與時值」的 JSON 或 MusicXML。
3. APP 收到 JSON，轉成 Kotlin 的 `ChordSnapshot` / `NoteEvent` 等資料結構。
4. 手機端 Kotlin 規則引擎執行所有和聲規則檢查，產生 `HarmonyIssue` 列表。
5. UI 將錯誤位置 overlay 在樂譜畫面上，並顯示中文說明與簡短建議。

### 2.2 設計原則

- 重運算（OMR / 圖像 → 樂譜）在雲端完成。  
- 輕運算（和聲規則檢查）在手機端本地完成。  
- Android 專案中不得硬編碼 API key 或敏感憑證。  
- 音樂規則以 `harmony_rules.py` + `harmony_rules_zh.md` 的內容為準。

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
## 8. Roadmap（優先順序參考）

1. MVP：
   - Kotlin 版規則引擎骨架完成，支援基礎規則（平行八／五度、導音處理、聲部交錯等）。
   - Android APP 串接假資料（不依賴實際 OMR），跑完整錯誤檢查流程。

2. 雲端 OMR 整合（路線 A）：
   - 根據 docs/omr_prompt_gemini.md 設計 prompt，與多模態 LLM 串接。
   - 在後端實作 serverless OMR proxy，提供穩定 JSON 輸出。

3. 規則與教材覆蓋率：
   - 擴充更多和聲規則與例外情況。
   - 蒐集實際作業作為測試集，持續修正與補強。

4. 中長期：
   - 研究專用 OMR 模型（Roboflow、自建或商業 SDK），逐步減少對雲端 LLM 依賴。

若此檔案與實際專案結構不一致，以使用者指示為準，並在後續修改中更新本檔內容。
