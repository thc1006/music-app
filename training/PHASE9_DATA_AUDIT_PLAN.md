# Phase 9 數據集審查計劃

## 目標
找出 OpenScore/DeepScores 數據導致性能下降的原因

## 背景
- **Phase 8**: 32,555 圖片, mAP50 = 0.6444 ✅
- **Phase 9**: 41,281 圖片 (+26.8%), mAP50 = 0.5723 ❌ (-11.2%)
- **關鍵發現**: 即使使用 Phase 8 完整配置訓練 150 epochs，Phase 9 仍比 Phase 8 差

## 審查步驟

### Step 1: 數據集分離測試 (2-3 天)

**目標**: 找出哪個數據源導致問題

```bash
# 測試 1: Phase 8 + DeepScores only
# - 數據集: Phase 8 (32,555) + DeepScores (847) = 33,402 圖
# - 訓練: 150 epochs, Phase 8 配置

# 測試 2: Phase 8 + OpenScore only
# - 數據集: Phase 8 (32,555) + OpenScore (5,238) = 37,793 圖
# - 訓練: 150 epochs, Phase 8 配置

# 測試 3: Phase 8 + OpenScore (抽樣 50%)
# - 數據集: Phase 8 (32,555) + OpenScore 50% (2,619) = 35,174 圖
# - 訓練: 150 epochs, Phase 8 配置
```

**預期結果**:
- 如果 DeepScores 單獨測試表現好 → 問題在 OpenScore
- 如果兩者都不好 → 需要深入檢查標註質量

### Step 2: 抽樣質量檢查 (4 小時)

#### OpenScore Lieder 審查
- [ ] 隨機抽樣 100 張圖片
- [ ] 檢查項目:
  - **標註精確度**: bbox 是否準確框住符號
  - **渲染品質**: 是否有模糊、雜訊、失真
  - **座標轉換**: Verovio SVG 座標轉 YOLO 是否正確
  - **標註一致性**: 同類符號標註是否統一

#### DeepScores 審查
- [ ] 隨機抽樣 50 張圖片
- [ ] 檢查項目:
  - **合成 vs 真實差異**: 合成圖是否與真實樂譜差異太大
  - **標註密度**: 是否有過多/過少標註
  - **類別映射**: 原始類別轉 YOLO 類別是否正確

### Step 3: 統計分析

#### Loss 分佈分析
```python
# 分析各數據來源的訓練 loss
- 計算 Phase 8、OpenScore、DeepScores 各自的平均 loss
- 找出 loss 異常高的數據子集
```

#### 錯誤案例分析
```python
# 在驗證集上找出錯誤預測
- 統計錯誤案例的數據來源
- 找出最容易錯誤的類別
- 分析錯誤模式（漏檢 vs 誤檢）
```

#### 類別分佈差異
```python
# 比較各數據源的類別分佈
- Phase 8 vs OpenScore vs DeepScores
- 找出分佈差異大的類別
- 檢查是否有「域偏移」(Domain Shift)
```

## 執行優先級

### 🔴 高優先級（立即執行）
1. **Step 2**: 抽樣質量檢查（4 小時）
   - 最快找出明顯問題
   - 無需 GPU 訓練

### 🟡 中優先級（1 週內）
2. **Step 1**: 數據集分離測試（2-3 天）
   - 需要 GPU 訓練
   - 能定量確認問題來源

### 🟢 低優先級（後續）
3. **Step 3**: 統計分析（需要先有測試結果）

## 預期結果

### 最佳情況
- 找出具體問題數據源
- 移除後 Phase 9.1 能達到 mAP50 > 0.60

### 最壞情況
- OpenScore/DeepScores 都無法使用
- 回退到 Phase 8，尋找其他數據源

### 中間情況
- 只保留高質量子集（例如 OpenScore fermata 數據）
- 調整數據權重/採樣策略

## 時間估計
- **Step 2 審查**: 4 小時
- **Step 1 測試訓練**: 2-3 天（3 次訓練 @ 8-10h each）
- **Step 3 分析**: 1 天
- **總計**: 3-5 天

## 下一步
選擇以下方案之一：

**A. 快速審查（推薦）**
```bash
# 立即執行 Step 2，4 小時內找出明顯問題
python training/audit_phase9_data_quality.py --sample 100
```

**B. 分離測試**
```bash
# 先訓練 Phase 8 + DeepScores only
python training/yolo12_train_phase9_deepscores_only.py
```

**C. 直接使用 Phase 8**
```bash
# 放棄 Phase 9，專注優化 Phase 8 或規劃 Phase 10
cp training/harmony_omr_v2_phase8/phase8_training/weights/best.pt \
   models/production/harmony_omr_best_v8.pt
```
