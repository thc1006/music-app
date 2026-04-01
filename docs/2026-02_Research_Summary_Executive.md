# 執行摘要：YOLO26 OMR 訓練深度調研（2026-02-13）

## 一、調研背景

- **專案**: Music App 五線譜光學辨識 (OMR)
- **當前狀態**: 
  - 完成 20-epoch smoke test
  - 最佳 mAP50: 0.28241 (epoch 15)
  - 最終 mAP50: 0.26036
  - 最終 Recall: 0.22723
  - 訓練過程有 1 次 CUDA OOM 警告（TaskAlignedAssigner fallback CPU）
  - 數據有 duplicate labels 警告（已自動移除）

- **調研範圍**:
  1. Ultralytics/YOLO26 小物件與長尾數據訓練策略
  2. CUDA OOM 對結果與速度的影響
  3. OMR 任務的最佳實務（數據不平衡、標註噪音）
  4. 短期實驗 vs 長期訓練的決策

## 二、核心發現（Top 5）

### 1. **YOLO26 架構適合 OMR 任務**
- NMS-free 端到端推論，優化邊緣部署
- 強化小物件檢測能力（符號通常佔圖片 < 5%）
- 支援 TFLite 導出，適合 Android 部署
- **來源**: Ultralytics YOLO11/26 Documentation

### 2. **當前配置偏保守，有提升空間**
- `batch=4` 在 RTX 5090 (32GB) 下偏低，可提升至 8-12
- `workers=20` 過高，可能導致 CPU 競爭，建議 6-8
- `amp=false` 降低訓練速度，可嘗試啟用（需監控 NaN）
- `cache=false` 導致每 epoch 重新讀取，建議改為 `ram` 或 `disk`
- **來源**: Ultralytics Training Args + PyTorch Best Practices

### 3. **CUDA OOM 警告影響有限**
- TaskAlignedAssigner fallback CPU 僅影響速度（該 batch 變慢）
- 不影響模型正確性（標籤分配結果相同）
- 1 次/20 epochs 可接受，若 > 5 次/100 epochs 需降低 batch
- **來源**: Ultralytics GitHub Issues + PyTorch 內部實作

### 4. **數據不平衡是主要瓶頸**
- 專案 logs 顯示 duplicate labels（已自動移除）
- 弱類別（如 barline_double, tie）可能樣本不足
- 建議處理順序：標註品質 > Oversampling > Loss 權重 > 超參數
- **來源**: Academic Research + Phase 6 Config 經驗

### 5. **短期實驗結果不可泛化**
- Ultralytics 官方警告：30 epochs / fraction=0.1 的最佳參數很少適用於完整訓練
- 當前 20-epoch 結果僅用於穩定性驗證，不應據此做長期決策
- **來源**: Ultralytics Hyperparameter Tuning Guide

## 三、推薦方案（單一路線）

### **方案：平衡進取型（100 Epochs）**

#### 核心超參數
```yaml
epochs: 100
batch: -1          # 自動調整至 60% GPU 記憶體
imgsz: 1280
amp: true          # 啟用混合精度（需監控）
lr0: 0.0005        # 保持當前值
optimizer: AdamW
workers: 8         # 從 20 降至 8
cache: ram         # 若數據集 < 10GB
mosaic: 1.0
close_mosaic: 10
copy_paste: 0.3    # 補償弱類別
degrees: 0.0       # OMR 特殊需求（不旋轉）
perspective: 0.0
```

#### 預期成果
- **目標 mAP50**: 0.50-0.60
- **訓練時間**: 12-16 小時 @ RTX 5090
- **風險等級**: 中（AMP 可能觸發 NaN，但機率 < 10%）

#### 關鍵監控點
1. **Epoch 10**: 檢查 loss 是否穩定（無 NaN/Inf）
2. **Epoch 30**: 檢查 mAP 是否 ≥ 0.40
3. **Epoch 50**: 檢查 mAP 是否 ≥ 0.50
4. **Epoch 80**: 若連續 10 epochs 提升 < 1%，提早停止

## 四、兩階段執行計劃

### **階段一：驗證（2-3 天）**

#### Task 1: 數據品質健檢（4h）
- [ ] 檢查 duplicate labels（已知存在）
- [ ] 分析類別分佈，找出弱類別（< 1% 樣本）
- [ ] 視覺化檢查標註品質（抽樣 100 張）

#### Task 2: 30-Epoch 穩定性測試（4h 訓練 + 1h 分析）
- [ ] 使用 `amp=true` 測試數值穩定性
- [ ] 驗收標準：無 NaN, mAP ≥ 0.35, Loss 平滑
- [ ] 若通過，進入階段二；若失敗，關閉 AMP 重測

#### Task 3: 弱類別專項測試（可選，6h）
- [ ] 針對低 mAP 類別增強 `copy_paste=0.6`
- [ ] 比較前後 per-class mAP 差異

### **階段二：正式訓練（5-7 天）**

#### Task 1: 100-Epoch 訓練（12-16h）
- [ ] 執行「平衡進取型」方案
- [ ] 實時監控 loss curve 和 mAP 進展
- [ ] 記錄到 MLflow 便於後續比較

#### Task 2: 模型驗證與分析（4-6h）
- [ ] 在測試集評估 mAP, Recall, Precision
- [ ] 分析 confusion matrix，找出高混淆類別對
- [ ] 錯誤案例視覺化（抽樣 20 張）

#### Task 3: 超參數微調（可選，2-3 天）
- [ ] 若 mAP < 0.50，使用 `model.tune()` 自動調參
- [ ] 重點調整 `lr0`, `cls`, `copy_paste`

#### Task 4: 部署準備（2-3 天）
- [ ] 導出 TFLite 模型（FP16 先行，視延遲再決定 INT8）
- [ ] 集成至 Android app
- [ ] 實機測試：延遲、mAP、記憶體使用

## 五、風險管理

### 高風險因素
| 風險 | 機率 | 影響 | 緩解措施 |
|------|------|------|---------|
| AMP 導致 NaN | 15% | 高（訓練失敗） | 前 20 epochs 密切監控，出現即關閉 AMP |
| Batch=-1 觸發 OOM | 20% | 中（需重啟） | 首次用 batch=8 測試，穩定後改 -1 |
| mAP < 0.40 @ 50 epochs | 25% | 中（目標未達） | 延長至 150 epochs 或檢查數據品質 |

### 緊急應變
```bash
# 情況 1: Loss 出現 NaN
# 處置: 立即停止，關閉 AMP，降低 lr0 至 0.0003

# 情況 2: CUDA OOM 頻繁
# 處置: batch=-1 改為 6，workers 8 改為 4

# 情況 3: mAP 停滯
# 處置: 完成當前訓練觀察完整曲線，再調整策略
```

## 六、成功標準

### 階段一（驗證通過）
- [ ] 30-epoch 測試無 NaN/Inf
- [ ] mAP50 ≥ 0.35
- [ ] 數據品質問題已修正

### 階段二（生產就緒）
- [ ] 100-epoch mAP50 ≥ 0.50（優秀：≥ 0.60）
- [ ] Per-class 分析完成，弱類別有改進計畫
- [ ] TFLite 模型在 Pixel 7 延遲 < 500ms

### 最終交付
- [ ] 生產級 YOLO26s 模型
- [ ] 完整訓練報告（含所有實驗對比）
- [ ] Android 集成與效能基準測試
- [ ] 後續優化路線圖

## 七、關鍵資源

### 文檔
- **完整技術報告**: `docs/2026-02_YOLO_OMR_Training_Deep_Research_Report.md`
- **快速決策指南**: `docs/QUICK_DECISION_GUIDE.md`
- **專案現況**: `LATEST_STATUS_AND_DATASET_SOURCES.md`

### 參考來源
1. Ultralytics Documentation (Training, Hyperparameter Tuning, YOLO11)
2. PyTorch AMP & Performance Tuning Guide
3. Google AI Edge LiteRT Documentation
4. Academic Research on OMR & Small Object Detection
5. 專案內部 Phase 6-8 Config & Logs

### 工具與腳本
- **訓練啟動**: `run_balanced_training.sh`（見快速指南）
- **監控腳本**: `watch` + `grep` 組合指令
- **分析工具**: Pandas + Matplotlib（內建於報告）

## 八、時間線預估

| 里程碑 | 預計完成 | 累計時間 |
|--------|---------|---------|
| 數據健檢完成 | Day 1 | 4h |
| 穩定性測試通過 | Day 1-2 | 9h |
| 100-epoch 訓練完成 | Day 3-4 | 25h |
| 模型驗證與分析 | Day 4-5 | 31h |
| TFLite 集成測試 | Day 6-7 | 35h |
| **總計** | **7 天** | **35 工時** |

> **並行優化**: 訓練期間可同步進行文檔整理、Android 架構準備、數據擴充計畫。

## 九、後續優化路線圖

若 100-epoch 訓練後 mAP 仍 < 0.60：

1. **數據層**:
   - 擴充弱類別樣本（合成或外部數據）
   - 修正標註錯誤（人工復核 + 主動學習）
   - 資料增強強化（針對性 copy-paste）

2. **模型層**:
   - 嘗試 yolo26m（更大模型）
   - 超參數自動調優（model.tune 50-100 iterations）
   - 集成學習（多模型融合）

3. **架構層**:
   - 評估 OMR 專用模型（如 Transformer-based）
   - 考慮兩階段檢測（粗檢 + 精檢）

## 十、立即行動項

**今日（2026-02-13）**:
1. [ ] 閱讀完整技術報告（30 min）
2. [ ] 執行數據品質檢查腳本（2h）
3. [ ] 準備 30-epoch 穩定性測試配置（30 min）

**明日（2026-02-14）**:
4. [ ] 啟動 30-epoch 測試（4h 訓練）
5. [ ] 分析結果，決定 GO/NO-GO（1h）

**本週（2026-02-15 ~ 20）**:
6. [ ] 若通過，啟動 100-epoch 正式訓練
7. [ ] 並行準備 Android 集成框架

---

**報告產出時間**: 2 小時（深度調研 + 交叉驗證 + 方案設計）  
**預期 ROI**: 若方案成功，mAP 從 0.28 → 0.55，提升 96%，可支持生產部署  
**決策建議**: **立即執行階段一驗證**，48 小時內決定是否進入正式訓練

---

**聯絡**: 若遇到問題或需要調整策略，隨時回報當前 epoch、mAP、loss 狀況  
**最後更新**: 2026-02-13 by 資料科學家調研
