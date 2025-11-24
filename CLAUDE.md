# CLAUDE.md — 四部和聲助教（Android + 規則引擎）

你是這個專案的長期協作夥伴與工程師，負責協助我完成：
「從樂譜照片，到四部和聲錯誤標示與文字說明」的完整系統。

> 密碼：Kinoshita15@

## 📍 即時狀態（Last Updated: 2025-11-24 17:30 UTC+8）

### 當前工作
- **階段**：Phase 2 完成 ✅ → **準備 Phase 3（外部數據整合）**
- **分支**：main
- **外部數據集**：已下載 2.7GB (8 個數據集)

### ✅ Phase 2 訓練結果（2025-11-24 完成）

| 指標 | Phase 1 | Phase 2 | 說明 |
|------|---------|---------|------|
| **Epochs** | 300/300 | 131/150 | Phase 2 早停 |
| **mAP50** | 0.452 | **0.509** | +12.6% 提升 |
| **mAP50-95** | 0.408 | 0.46+ | 良好 |
| **最佳模型** | - | epoch 111 | mAP50=0.50935 |

### 📁 當前最佳模型
```
training/harmony_omr_v2_phase2/balanced_training/weights/best.pt
```

### 📁 Phase 1 模型位置
```
training/harmony_omr_v2_optimized/train_phase1/weights/
├── best.pt   (18.9 MB) ← 最佳模型，用於 Phase 2
└── last.pt   (18.9 MB)
```

### 📊 各類別 mAP50 表現（Phase 1 最終）
| 類別 | mAP50 | 狀態 |
|------|-------|------|
| notehead_filled | 0.695 | ✅ 良好 |
| notehead_hollow | 0.713 | ✅ 良好 |
| stem | 0.691 | ✅ 良好 |
| beam | 0.580 | ⚠️ 可改進 |
| flag_8th | 0.342 | ⚠️ 需加強 |
| flag_16th | 0.156 | ❌ 需 Phase 2 |
| flag_32nd | 0.287 | ⚠️ 樣本太少 |
| augmentation_dot | 0.361 | ⚠️ 可改進 |

### ⚠️ 瓶頸類別分析（Phase 3 目標）

| 類別 | 當前 mAP50 | 外部數據解決方案 | 狀態 |
|------|-----------|-----------------|------|
| Class 16 (double_sharp) | 0 | **Fornes: +497 樣本** | 🟢 可解決 |
| Class 17 (double_flat) | 0 | 需 LilyPond 合成 | 🟡 待合成 |
| Class 24 (barline_double) | 0 | AudioLabs v2 | 🟡 間接幫助 |
| Class 15 (natural) | 0.187 | Choi + Fornes: +1,500+ | 🟢 可解決 |
| Class 12 (clef_tenor) | 0.273 | Fornes Alto: +759 | 🟢 可解決 |

---

## 🔄 恢復訓練指南（Phase 2）

當 GPU 可用時，執行以下步驟恢復訓練：

### Step 1: 確認 GPU 狀態
```bash
nvidia-smi
# 確認 GPU 閒置 (memory < 500MB)
```

### Step 2: 啟動 Phase 2 訓練
```bash
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate
python yolo12_train_phase2_balanced.py
```

### Phase 2 主要改進
- 類別加權損失函數（稀有類別權重 50x）
- 過採樣稀有類別圖片
- 從 Phase 1 best.pt 繼續訓練
- 預期 mAP50 提升至 0.60-0.65

---

## 📋 已完成的優化工作

### 1. 數據集優化 ✅
- `optimize_dataset_phase1.py` 已執行
- stem_down → 合併到 stem
- slur → 排除
- 驗證集：205 → 273 張
- 類別數：35 → 33

### 2. 訓練配置優化 ✅
- Batch: 24 → 16（解決 OOM）
- LR: 0.01 → 0.005（減少震盪）
- Mosaic: 1.0 → 0.5（穩定性）
- Copy-paste: 關閉（不適合樂譜）

### 3. 長期戰略規劃 ✅
- `PERFECT_MODEL_STRATEGY.md` - 6-Phase 路線圖
- `synthetic_data_generator.py` - 合成數據工具
- `yolo12_train_phase2_balanced.py` - Phase 2 腳本

---

## 🎯 六階段路線圖進度

| Phase | 名稱 | 目標 mAP50 | 狀態 |
|-------|------|-----------|------|
| 1 | 基礎訓練 | 0.45-0.50 | ✅ 完成 (0.452) |
| 2 | 類別平衡 | 0.50-0.55 | ✅ 完成 (0.509) |
| 3 | 外部數據整合 | 0.65-0.70 | 🔄 **進行中** |
| 4 | LilyPond 合成 | 0.75-0.80 | ⏳ 待執行 |
| 5 | 高解析度訓練 | 0.85-0.90 | ⏳ 待執行 |
| 6 | 生產優化 | 0.90+ | ⏳ 待執行 |

---

## 📦 外部數據集資源（2025-11-24 更新）

### 已下載數據集（總計 ~2.7GB）

位置：`training/datasets/external/`

| 數據集 | 大小 | 授權 | 內容 | 用途 |
|--------|------|------|------|------|
| **DoReMi** | 672MB | Research | 5,218 圖片 + OMR XML (94類) | 完整物件檢測訓練 |
| **Fornes** | 25MB | Research | **497 double_sharp** + 臨時記號 | **解決 mAP=0 瓶頸** |
| **MUSCIMA++** | 103MB | CC-BY-NC-SA | 91,255 手寫符號 | 手寫泛化能力 |
| **Choi Accidentals** | 13MB | Research | 2,955 臨時記號圖片 | 解決 natural/sharp 混淆 |
| **Rebelo 1 & 2** | 62MB | CC-BY-SA 4.0 | ~15,000 符號 | 補充基礎符號 |
| **AudioLabs v2** | 298MB | CC-BY-NC-SA | 940 圖, 85,980 bbox | 小節線/五線譜檢測 |
| **OpenScore String Quartets** | 1.1GB | **CC-0** | 106 弦樂四重奏 MusicXML | **四部和聲直接相關** |
| **OpenScore Lieder** | 499MB | **CC-0** | 1,356 藝術歌曲 MusicXML | LilyPond 合成來源 |

### 關鍵發現：解決 mAP=0 瓶頸

| 瓶頸類別 | 當前樣本 | 外部數據源 | 新增樣本 |
|----------|---------|-----------|----------|
| `double_sharp` (16) | 338 | **Fornes** | **+497** |
| `double_flat` (17) | 12 | 需 LilyPond 合成 | - |
| `barline_double` (24) | 1,620 | AudioLabs (間接) | - |
| `accidental_natural` (15) | - | Choi + Fornes + Rebelo | **+1,500+** |
| `clef_tenor` (12) | - | Fornes Alto (759) | +759 |

### 各數據集使用方法

#### 1. Fornes (最高優先級)
```bash
# 直接解決 double_sharp mAP=0 問題
cd training/datasets/external/omr_downloads/Fornes/
ls ACCIDENTAL_DoubSharp/  # 497 個 BMP 樣本
# 需要轉換為 YOLO 格式並整合到訓練集
```

#### 2. DoReMi (完整物件檢測)
```bash
# 包含完整 bounding box 標註
cd training/datasets/external/omr_downloads/DoReMi/DoReMi_v1/
ls Images/   # PNG 圖片
ls OMR_XML/  # XML 標註 (需解析轉換)
```

#### 3. OpenScore String Quartets (四部和聲)
```bash
# 弦樂四重奏 = SATB 四部和聲等價物
cd training/datasets/external/omr_downloads/OpenScoreStringQuartets/
# 可用 LilyPond 渲染生成訓練圖片
```

### 待下載數據集（可選）

| 數據集 | 大小 | 授權 | 價值 | 下載命令 |
|--------|------|------|------|----------|
| **DeepScores V2** | ~7GB | **CC BY 4.0** | 商業可用！255K 圖片 | `OmrDataset.DeepScores_V2_Dense` |
| HOMUS | - | Research | 在線手寫符號 | `OmrDataset.Homus_V2` |

### 商業授權重要提示

✅ **可商業使用**：
- DeepScores V2 (CC BY 4.0)
- Rebelo (CC-BY-SA 4.0)
- OpenScore 系列 (CC-0)
- MSMD (CC-BY-SA 4.0)

⚠️ **僅限研究/訓練**（模型權重不受限）：
- MUSCIMA++ (NC)
- AudioLabs (NC)
- Choi, Fornes, DoReMi (需確認)

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



# CLAUDE.md

## AI Patch Guardrails (for Claude Code)

You are Claude Code working on this repository.  
Your main responsibilities are:
- Help implement small, well-scoped changes.
- Respect existing architecture, tests, and maintainer feedback.
- Avoid over-engineering and premature abstraction.

**IMPORTANT: You MUST follow all rules in this section whenever you propose patches or edit files.**

---

### 0. General workflow

1. **Explore & understand before coding**
   - ALWAYS read the relevant files and existing tests first.
   - Summarize your understanding and planned changes before editing.
   - If anything is ambiguous, ask for clarification instead of guessing.

2. **Plan → Implement → Verify**
   - Make a short plan (“think hard”) before you start editing.
   - Keep changes minimal and focused on the requested task.
   - Always run the relevant tests or at least explain precisely how to run them.

3. **Respect project-local rules**
   - The rules below (imports, logging, Dockerfile, tests, etc.) come from real code review feedback.
   - Treat them as authoritative for this repository.

---

### 1. Function abstraction & structure

**IMPORTANT: DO NOT introduce premature abstractions.**

1. **No trivial wrapper functions**
   - If a function only:
     - has 1–2 lines, AND
     - just calls another function (e.g., `return compose_text_message(...)`),
     - and is used only 1–2 times,
   - THEN: DO NOT create a separate helper function for it.
   - Example: DO NOT create `create_error_message(lang_code: str)` that only wraps `compose_text_message(get_response(...))`.

2. **Rule of Three (YAGNI)**
   - 1st occurrence: write the code inline.
   - 2nd occurrence: copy-paste is acceptable.
   - 3rd occurrence: you MAY propose a helper.
   - 4th occurrence: you SHOULD refactor into a shared abstraction.
   - Any refactor MUST clearly improve readability and reduce real duplication, not just “cosmetic” wrapping.

3. **Handler vs implementation**
   - For public handlers, follow this pattern:
     - `handler()`:
       - Handles `try/except`.
       - Logs exceptions with `logger.exception(...)`.
       - Returns a standard error message.
     - `_handler_impl()`:
       - Contains business logic only.
   - DO NOT move complex business logic into the handler.

---

### 2. Python imports

**IMPORTANT: All imports MUST follow PEP 8 and be at module top-level.**

1. **Placement**
   - Place imports at the top of the file, after module comments/docstring.
   - DO NOT add imports inside functions or methods unless explicitly documented as an exception.

2. **Order**
   - Group imports as:
     1. Standard library
     2. Third-party libraries
     3. Local modules
   - Separate each group with a blank line.

3. **Example**

```python
# 1. Standard library
from typing import Dict, Optional

# 2. Third-party
from linebot.v3.messaging import TextMessage

# 3. Local modules
from src.modules.qna.constants import RESPONSE_DATA_PATH
from src.modules.utils import compose_text_message, get_response
```

---

### 3. Logging & error handling

1. **Use `logger.exception` in `except` blocks**
   - When catching unexpected errors in handlers, prefer:
     ```python
     except Exception as e:
         logger.exception(f"Error in qna_handler: {e}")
         return compose_text_message(
             get_response(RESPONSE_DATA_PATH, "error_message", lang_code)
         )
     ```
   - This captures the full stack trace at ERROR level.

2. **Separation of concerns**
   - Handlers:
     - Validate input.
     - Call `_impl`.
     - Catch and log unexpected errors.
   - `_impl` functions:
     - Contain business logic and can be unit-tested directly.

---

### 4. Dockerfile changes

**IMPORTANT: Keep runtime images slim and focused on runtime dependencies.**

1. **Base image**
   - Prefer minimal base images similar to:
     ```Dockerfile
     FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
     ```

2. **Dependency installation**
   - Copy only `pyproject.toml` and lockfiles before running the install command.
   - Install ONLY runtime dependencies inside the final image.
   - DO NOT install tools that are only required for:
     - type checking (e.g. pyright),
     - linters,
     - local development.
   - If such tools are needed, suggest:
     - a dev-only image, or
     - a separate `dev` target in the Dockerfile,
     - but DO NOT add them silently.

---

### 5. Code smell & refactoring

When you notice repetition:

1. **Do NOT refactor automatically just because you see repetition.**
   - First, check:
     - Is this “incidental” repetition (similar text but different semantics)?
     - Or “essential” repetition (same logic, same semantics)?

2. **Avoid shotgun surgery**
   - If a change requires modifying many different files and call sites for a small benefit, you are probably introducing a bad abstraction.
   - In that case:
     - Explain the tradeoffs.
     - Ask the user before proceeding with a large refactor.

---

### 6. Tests & TDD

**IMPORTANT: Tests must be meaningful, not just “green”.**

1. **Correct TDD order**
   - DO NOT follow:
     - “write tests → accept whatever output you get”.
   - Instead:
     - Read the existing implementation first.
     - Understand whether the feature is implemented or still TODO.
     - Design tests that match the intended behavior.
     - Then update implementation to satisfy those tests.

2. **Detect unimplemented features**
   - If you see any of the following:
     - `// TODO: implement this`
     - returning an **empty struct** (e.g., `Tracing: &SomeType{}`)
     - variables assigned but only used as `_ = variable`
     - golden files containing empty objects like `tracing: {}`
   - THEN:
     - Treat the feature as “NOT YET IMPLEMENTED”.
     - DO NOT write tests that pretend the feature is fully working.
     - Instead, you may:
       - Add clearly labeled placeholder tests, OR
       - Create a GitHub issue describing the missing implementation.

3. **Test naming**
   - Use precise names:
     - `valid-X` → tests the successful path.
     - `invalid-X` → tests error handling and validation failures.
     - `placeholder-X` → feature not yet fully implemented, placeholder coverage only.
   - DO NOT name a test `invalid-tracing` if it does not actually test invalid behavior.

4. **No skipped tests in new code**
   - DO NOT add tests with `t.Skip()` unless explicitly requested and clearly documented as a temporary measure.
   - All new tests you add SHOULD run and pass on CI.

5. **Avoid redundant tests**
   - Before adding a new test file:
     - Check existing E2E / integration tests.
     - If existing tests already cover the behavior, DO NOT add redundant tests.
   - Example: For minimal RBAC changes, prefer relying on existing E2E tests rather than adding new tests that just verify Kubernetes basics.

6. **Use standard library & project helpers**
   - In Go tests:
     - Prefer `strings.Contains` over custom substring checks.
     - Use existing helper packages (e.g. `ktesting/setup.go`) instead of building ad-hoc loggers or setups.

---

### 7. File selection & change scope

**IMPORTANT: Keep diffs minimal and focused.**

1. **Verify file usage before editing**
   - Before modifying a file:
     - Check if it is still used in the build/runtime.
     - For suspicious files (e.g., old generators like `kubebuilder-gen.go`):
       - Use `git grep` or build commands to confirm usage.
   - If a maintainer comment says “this file is not used anymore, better to delete it”:
     - DO NOT update the file.
     - Suggest deleting it instead, if appropriate for this PR.

2. **Minimal patch principle**
   - For tasks like “minimal RBAC fix”:
     - Focus only on the specific RBAC manifests mentioned by the issue or reviewer.
     - Avoid:
       - editing unrelated manifests,
       - adding new test suites,
       - touching generator files unless required.

3. **Respect project conventions**
   - Follow existing patterns in the codebase:
     - Same logging style.
     - Same error handling style.
     - Same file layout and naming conventions.

---

### 8. Human review & maintainer feedback

1. **Maintainer comments are authoritative**
   - When a reviewer (e.g. project maintainer) gives feedback like:
     - “These tests are unnecessary.”
     - “This file is unused; delete it instead of updating it.”
   - You MUST:
     - Treat this feedback as the source of truth for future edits.
     - Reflect these rules in your subsequent patches.

2. **Document learnings**
   - When you discover a new project-specific rule through review:
     - Propose an update to `CLAUDE.md` (or ask the user to add it).
     - Follow the updated rule consistently in future changes.

---

### 9. How to work with tests & golden files in this repo

1. **Golden files**
   - When adding or updating golden files (YAML, JSON, etc.):
     - Ensure they contain meaningful, non-empty configuration.
     - If the implementation is a placeholder, clearly mark the golden file as such with comments.
     - Question suspicious emptiness (e.g., `tracing: {}`) and check whether the feature is really implemented.

2. **Creating follow-up issues**
   - If you identify missing behavior (e.g., tracing translation not fully implemented):
     - Propose creating a GitHub issue with:
       - Title, e.g.: `"Implement tracing translation in AgentgatewayPolicy frontend"`.
       - Links to the relevant PR / tests / files.
       - A plan for implementation and test updates.

---

### 10. Claude Code behavior summary (TL;DR)

When generating patches in this repo, you MUST:

- **Understand before coding**: read implementation & tests first.
- **Keep changes minimal**: avoid editing unused files or adding redundant tests.
- **Avoid premature abstraction**: no one-line wrappers unless used ≥3 times AND more readable.
- **Follow local style**: imports at top, logging via `logger.exception`, handler + `_impl` split, slim Dockerfiles.
- **Design meaningful tests**: no fake “invalid” tests, no `t.Skip()` tests, no empty golden files unless clearly marked as placeholders.
- **Respect maintainers**: treat review comments as project rules and adjust your behavior accordingly.

If you are unsure which rule applies, you MUST stop, summarize the options, and ask the user for guidance before making large-scale or irreversible changes.
