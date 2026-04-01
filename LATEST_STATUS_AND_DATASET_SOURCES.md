# 專案最新狀態與資料集來源總覽（已含第二輪交叉驗證）

更新時間：2026-02-13  
文件目的：用「可直接執行與追蹤」的方式，整理目前模型狀態、資料集來源與已知風險。

---

## 一、先回答你的問題

### 1) 這份檔案先前有沒有包含「第二輪大規模交叉驗證」？
沒有。先前版本只有第一輪整理內容，**未完整寫入第二輪交叉驗證結論**。  
本次已全面補上，並改寫為台灣繁體中文、易讀版本。

### 2) 目前是否已更新？
已更新。你現在看到的這一版，已納入第二輪交叉驗證重點與補充說明。

---

## 二、最新模型進度（已交叉驗證）

### A. 目前可用的生產基準（最佳實務基線）
- 模型權重：
  - `training/harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt`
- 推論建議參數：
  - `imgsz=1280, conf=0.25, iou=0.55`
- 專案紀錄中的代表成果：
  - `mAP50 = 0.7059`（以推論參數調整後達成，見 `CLAUDE.md`）

### B. YOLO26s（batch4）目前狀態
- 執行目錄：
  - `training/harmony_omr_v2_yolo26/yolo26s_finetune_batch4/`
- `results.csv` 顯示：
  - 最佳點：**epoch 72，mAP50 = 0.642**
- 失敗訊號（log）：
  - `Loss NaN/Inf detected`
  - `RuntimeError: ... last.pt is corrupted with NaN/Inf weights`

### C. 第二輪交叉驗證補充（重要）
為避免誤讀，這點務必注意：
- epoch 73~75 的 **loss 欄位是 NaN**（train/val box/cls/dfl）。
- 但 `metrics/mAP50(B)` 在這幾列是 **0**（不是 NaN）。
- 結論不變：該次 YOLO26 batch4 訓練已發散並中止，`last.pt` 已不可靠。

---

## 三、已下載資料集路徑（本機）

主要外部資料集根目錄：
- `training/datasets/external/`

其中核心來源：
- `training/datasets/external/omr_downloads/ChoiAccidentals/`
- `training/datasets/external/omr_downloads/DoReMi/`
- `training/datasets/external/omr_downloads/Fornes/`
- `training/datasets/external/omr_downloads/MuscimaPlusPlus_Images/`
- `training/datasets/external/omr_downloads/MuscimaPlusPlus_V2/`
- `training/datasets/external/omr_downloads/OpenScoreLieder/`
- `training/datasets/external/omr_downloads/OpenScoreStringQuartets/`
- `training/datasets/external/omr_downloads/Rebelo1/`
- `training/datasets/external/omr_downloads/Rebelo2/`

其他外部資料夾：
- `training/datasets/external/deepscores_v2/`
- `training/datasets/external/AudioLabs_v2/`
- `training/datasets/external/openscore_lieder/`

---

## 四、資料集來源對照（路徑 -> 上游）

- DoReMi  
  `.../omr_downloads/DoReMi/` -> https://github.com/steinbergmedia/DoReMi/
- Choi Accidentals  
  `.../omr_downloads/ChoiAccidentals/` -> https://www-intuidoc.irisa.fr/en/choi_accidentals/
- Fornes  
  `.../omr_downloads/Fornes/` -> http://www.cvc.uab.es/~afornes/
- MUSCIMA++  
  `.../omr_downloads/MuscimaPlusPlus_V2/` -> https://ufal.mff.cuni.cz/muscima
- CVC-MUSCIMA（影像）  
  `.../omr_downloads/MuscimaPlusPlus_Images/` -> http://www.cvc.uab.es/cvcmuscima/index_database.html
- OpenScore Lieder  
  `.../omr_downloads/OpenScoreLieder/` -> https://github.com/OpenScore/Lieder
- OpenScore String Quartets  
  `.../omr_downloads/OpenScoreStringQuartets/` -> https://github.com/OpenScore/StringQuartets
- Rebelo 1 / 2  
  `.../omr_downloads/Rebelo1/`, `.../omr_downloads/Rebelo2/` -> http://www.inescporto.pt/~arebelo/index.php
- DeepScores V2  
  `.../deepscores_v2/` -> https://zenodo.org/records/4012193
- AudioLabs v2  
  `.../AudioLabs_v2/` -> https://www.audiolabs-erlangen.de/resources/MIR/2019-ISMIR-LBD-Measures

來源索引檔：
- `OMR_Datasets_Comprehensive_List.csv`
- `training/datasets/external/DATASETS_SUMMARY.md`
- `training/EXTERNAL_DATASETS_INVENTORY.md`

---

## 五、已下載壓縮檔（training/）

- `training/DoReMi_v1.zip`
- `training/MUSCIMA-pp_v2.0.zip`
- `training/CVC_MUSCIMA_PP_Annotated-Images.zip`
- `training/OpenScore-Lieder-Snapshot-2023-10-30.zip`
- `training/OpenScore-StringQuartets-Snapshot-2023-10-30.zip`
- `training/Rebelo-Music-Symbol-Dataset1.zip`
- `training/Rebelo-Music-Symbol-Dataset2.zip`
- `training/choi_accidentals_dataset.zip`
- `training/Music_Symbols.zip`

---

## 六、第二輪大規模交叉驗證摘要（重點）

以下項目已再次獨立重算並比對：
- `omr_downloads` 子資料集數：**9 套**（確認）
- `training/*.zip` 壓縮檔：**9 份**（確認）
- 最佳訓練結果仍為：
  - `stable_1280_resumed/results.csv`（best mAP50 = 0.69788）
- YOLO26 batch4：
  - best mAP50 = 0.642 @ epoch 72（確認）
  - epoch 73~75 loss 為 NaN，且發生 checkpoint 損壞錯誤（確認）
- Python 測試基線（當前）：
  - **29 failed / 96 passed**（確認）

---

## 七、當前建議（務實版）

1. **短期上線**：仍以 Ultimate v5 stable（1280 + iou=0.55）作為部署基線。  
2. **YOLO26 後續**：不要直接沿用目前 batch4 的 `last.pt`；建議重新啟訓或改穩定化策略（如 AMP/學習率/資料檢查）。  
3. **文件一致性**：建議同步更新 `CLAUDE.md` 中「YOLO26 訓練進行中」措辭，避免與實際失敗狀態不一致。

---

## 八、2026-02 外部最佳實踐調研（已對照本專案）

> 本段為最新一輪「上網深度調研」的摘要，重點是可直接落地到本專案的做法。

### 1) 偵測訓練與 YOLO26（Ultralytics 官方）
- YOLO26 重點：NMS-free、強化小物件、支援 TFLite/ONNX/TensorRT 匯出。
- 訓練穩定建議：先用預訓練權重、調整 batch/learning rate/warmup、必要時關閉 AMP、使用早停機制。
- 調參建議：可用 `model.tune()` 做系統化超參數搜索，但官方也提醒**短時間小規模 tuning 結果不一定可泛化**。

### 2) NaN/混合精度（PyTorch AMP）
- AMP 官方建議是只包 forward+loss，並搭配 GradScaler；遇到數值不穩可改回 FP32 或縮小學習率與梯度幅度。
- 這和你目前 YOLO26 batch4 的 NaN 現象一致，代表下一輪應先做「穩定化實驗」再追高 mAP。

### 3) 小物件與長尾（學術證據）
- 2024-2025 小物件偵測調研指出：多尺度訓練、資料增強、知識蒸餾、輕量化部署是高性價比方向。
- Equalized Focal Loss（EFL）指出一階偵測器在長尾資料容易受正負樣本不平衡影響，需做類別不平衡補償。
- 這與你目前弱類別（barline_double、tie、double_sharp 等）問題高度吻合。

### 4) OMR 評估方法（2025 新基準）
- SMB（Sheet Music Benchmark）提出 OMR-NED，可更細粒度分析錯誤（音頭、beam、pitch、accidental 等）。
- 對本專案意義：目前主要看 mAP，後續可增加符號層級錯誤分析，避免「mAP 有進步但規則引擎輸入品質沒提升」。

### 5) Android 端側部署（Google 官方）
- Android 架構建議：分層（UI/Data/可選 Domain）、UDF、ViewModel + StateFlow、Repository 單一資料來源。
- LiteRT/TFLite 效能建議：先 profile 再優化，評估 CPU/GPU/NNAPI delegate，避免不必要記憶體拷貝，必要時 full INT8。
- 本專案現況：Android OMR pipeline 還有多個 TODO，代表下一步應同步補齊工程架構，不只訓練模型。

### 6) 資料/實驗治理（DVC + MLflow）
- DVC：適合做資料版本化與可回溯（資料集快照、切分、來源追蹤）。
- MLflow：適合統一記錄參數、指標、artifact，做實驗比較與最佳模型挑選。
- 本專案目前實驗多但分散，導入輕量治理可大幅降低重複試錯。

---

## 九、依你的目標，下一步執行路線（建議版）

> 目標：從「樂譜照片 -> 端側辨識 -> 和聲規則檢查」走到可穩定交付。

### 路線 A：先穩定可部署（建議優先）

#### A-1. 鎖定可用基線
- 以 Ultimate v5 stable + `imgsz=1280, conf=0.25, iou=0.55` 作為當前生產基線。
- 建立基線驗收集（固定圖片集 + 固定評估腳本）作為後續所有實驗比較基準。

#### A-2. YOLO26 先做「穩定化」而非直接追高
- 先做小規模 smoke test（例如 `fraction=0.1~0.2`）檢查是否再出現 NaN。
- 第一輪穩定化實驗建議：
  1. `amp=False`
  2. `lr0` 降低（例：0.001 -> 0.0005 或更低）
  3. `warmup_epochs` 增加（例：3 -> 5）
  4. 降低增強強度（先減少高風險增強再逐步加回）
- 決策門檻：若 30 epoch 內仍出現 NaN，先停止 YOLO26 路線，避免浪費 GPU 時間。

#### A-3. 弱類別修復（以資料品質優先）
- 先做資料修正與針對性增強，再做 loss 權重調整。
- 優先順序：
  1. 標註品質（尤其 barline_double 邊框）
  2. 弱類別 oversampling
  3. 類別不平衡 loss（可參考 EFL 思路）
  4. 再做超參數 tuning

---

### 路線 B：同步補齊 Android 端側工程（避免模型好但產品流程斷裂）

#### B-1. 補齊 OMR 流程 TODO
- `Yolo12OmrClient`: ByteArray -> Bitmap、檢測結果轉結構化輸出。
- `SymbolAssembler`: 真正的時序分組、調號拍號解析、音高推斷穩定化。
- `HttpOmrClient`: JSON 序列化/反序列化正式實作（若保留雲端 fallback）。

#### B-2. 架構調整（對齊 Android 官方建議）
- 導入清楚分層（UI / Data / optional Domain）。
- ViewModel + StateFlow + UDF，減少畫面邏輯耦合。
- OMR 與規則引擎資料流明確化（單一資料來源）。

#### B-3. 端側效能驗證
- 固定同一組測試圖，分別測 CPU / GPU / NNAPI。
- 統一紀錄：延遲、記憶體、耗電、穩定性（連跑）。

---

### 路線 C：實驗治理與檔案整理（提升可維護性）

#### C-1. 實驗追蹤標準化
- 新實驗統一產出：`config + data snapshot + metrics + confusion + checkpoint`。
- 建議導入 MLflow（先本機）統一查詢最佳實驗。

#### C-2. 資料版本化
- 建議導入 DVC 管理：
  - Phase8_final / Phase9 / Phase10_1 等資料快照
  - 外部資料來源合併紀錄
  - 可回溯每次訓練使用的資料版本

#### C-3. training 目錄分層（先不破壞既有流程）
- 先採「增量式重構」：
  - `training/scripts/train/`
  - `training/scripts/data/`
  - `training/scripts/eval/`
  - `training/configs/`
  - `training/reports/`
- 舊檔先保留，透過 README 映射逐步遷移。

---

## 十、你接下來要拍板的 4 個決策

1. **YOLO26 是否先進入穩定化模式（而不是直接追精度）？**  
   建議：是。
2. **是否把 Ultimate v5 當作當前唯一可部署基線？**  
   建議：是。
3. **是否啟動 MLflow + DVC 的輕量導入？**  
   建議：是（先小範圍導入，不一次改全專案）。
4. **是否同時排程 Android TODO 補齊（與模型訓練並行）？**  
   建議：是，否則會卡在「模型有結果但 App 流程不完整」。

---

## 十一、A+B+C 已落地的第一批實作（2026-02-13）

### A 路線（訓練與基線）
- 新增 `training/validate_production_baseline.py`  
  - 固定以部署參數驗證目前基線模型，輸出 `reports/baseline_validation.json`。
- 新增 `training/yolo26s_stability_smoketest.py`  
  - 先跑小比例資料的穩定性煙霧測試，內建 NaN/Inf loss 檢查，避免直接長訓練踩雷。

### B 路線（Android OMR MVP 串接）
- 已補 `HttpOmrClient`：
  - request JSON 序列化（含 `image_base64`）
  - response JSON 解析為 `List<ChordSnapshot>`。
- 已補 `Yolo12OmrClient`：
  - `ByteArray -> Bitmap` 解碼
  - 偵測結果透過 `SymbolAssembler` 組裝 chord。
- 已補 `SymbolAssembler`：
  - 由 X 座標 + 小節線推估拍點分組（measure/beat）
  - 取 4 聲部建立 `ChordSnapshot`（S/A/T/B）。

### C 路線（實驗治理）
- 新增 `training/log_ultralytics_run_to_mlflow.py`  
  - 可把 Ultralytics `results.csv` 匯入 MLflow，統一管理 best/last 指標。
- 新增 `training/bootstrap_dvc.sh`  
  - 一鍵初始化 DVC 與 local remote（可改路徑），作為資料版本化入口。

---

## 十二、A+B+C 第二批執行結果（2026-02-13）

### 1) Gradle / gradlew 已補齊
- 已新增：
  - `android-app/gradlew`
  - `android-app/gradlew.bat`
  - `android-app/gradle/wrapper/gradle-wrapper.jar`
  - `android-app/gradle/wrapper/gradle-wrapper.properties`
- 已建立本機 JDK17 路徑：`.local/jdk17_pkg/usr/lib/jvm/java-17-openjdk-amd64`
- `gradlew` 已加上自動偵測本機 JDK17（未設定 JAVA_HOME 時可直接用）。

### 2) 路線 A 實跑結果
- **Production baseline 驗證完成**  
  - 檔案：`training/reports/baseline_validation.json`  
  - 結果：`map50=0.7519`, `recall=0.5598`, `passed=true`
- **YOLO26 stability smoke test 完成（縮短版）**  
  - run：`training/harmony_omr_v2_yolo26/yolo26s_stability_smoketest_e5w0/`  
  - 設定：`epochs=5, fraction=0.1, workers=0`  
  - 結論：**未偵測到 NaN/Inf loss**（腳本正常結束）

### 3) 路線 C 實跑結果
- **MLflow**：已建立與寫入
  - `yolo26_batch4_postmortem`
  - `yolo26_smoketest_e5w0`
- **DVC**：已初始化並設定 default remote
  - remote 名稱：`localstorage`
  - 路徑：`/home/thc1006/dev/music-app/.dvc/local-remote`

---

## 十三、2026-02 最新外部調研後的「下一步」建議（優先序）

### P0（先做，直接影響交付）
1. **把 YOLO26 smoke test 從 5 epoch 擴到 30 epoch（同穩定化設定）**  
   - 依據：Ultralytics train/val 文件 + PyTorch AMP 文件，先確保穩定再追 mAP。  
   - 建議沿用：`amp=False`, `lr0=0.0005`, `warmup_epochs=5`, `workers=0/低值`。  
   - Gate：若 30 epoch 內出現 NaN/Inf，立即停並回退 Ultimate v5 方案。

2. **維持 Ultimate v5 作為當前部署主線，並固定驗收門檻**  
   - 已達成基線：`map50=0.7519`, `recall=0.5598`。  
   - 每次模型更新都跑 `validate_production_baseline.py`，避免回歸。

### P1（接著做，提升產品可用性）
3. **Android 端做 LiteRT delegate 基準測試（CPU/GPU/NNAPI）**  
   - 依據：Google LiteRT delegates / performance best practices。  
   - 產出：同一批測試圖下 latency、記憶體、穩定性報表，作為 Pixel 7 預設 delegate 決策依據。

4. **導入 Baseline Profile / Startup Profile**  
   - 依據：Android 官方 Baseline Profiles 指南。  
   - 目標：改善冷啟動與首輪互動延遲，避免「模型可用但 App 首次體感卡頓」。

### P2（治理與中期成長）
5. **MLflow backend 從 file store 升級為 SQLite（先本機）**  
   - 原因：MLflow 官方提示 filesystem backend 即將淘汰。  
   - 目標：提高查詢穩定性與後續擴充性。

6. **DVC 正式追蹤 Phase8/主要訓練資料快照**  
   - 依據：DVC data versioning 官方流程。  
   - 建議先從 `training/datasets/yolo_harmony_v2_phase8_final` 開始，完成 `dvc add -> git commit -> dvc push`。

### P3（研究升級）
7. **評估 OMR-NED（SMB）作為輔助指標**  
   - 依據：SMB 2025 基準。  
   - 目的：補足僅看 mAP 的盲點，直接觀察 notehead/beam/accidental 等符號層級誤差。

8. **長尾弱類別策略：優先資料品質與重採樣，再做 loss 調參**  
   - 依據：EFL/YOLO LwF 等長尾與持續學習研究脈絡。  
   - 實務順序：標註品質 -> oversampling -> 類別不平衡 loss -> 最後才做大規模 tuning。
