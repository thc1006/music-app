# Phase 9 深度評估報告

**日期**: 2025-11-28
**結論**: 原 Phase 9 清理策略需要修正

---

## 🔴 重大發現摘要

### 1. Tiny Bbox 清理策略問題

**皮爾遜相關係數**: -0.143 (非常弱)

| 發現 | 說明 |
|------|------|
| flag_16th | 84.3% tiny bbox → 0.731 mAP (高 tiny 但高 mAP!) |
| fermata | 59.0% tiny bbox → 0.644 mAP (高 tiny 但還可以) |
| barline_double | **0.2% tiny bbox** → **0.231 mAP** (低 tiny 但最差!) |
| accidental_double_sharp | 5.3% tiny bbox → 0.461 mAP (低 tiny 但差) |

**結論**: 清理 tiny bbox 對改善 mAP 效果非常有限！barline_double 的問題不是 tiny bbox。

### 2. 未使用數據資源 (8,726 張圖片)

| 數據集 | 圖片數 | 關鍵標註 |
|--------|--------|----------|
| **OpenScore Lieder** | 5,238 | barline_double: **+3,724**, fermata: +4,182 |
| DeepScores Dynamics | 700 | dynamic_soft: +3,125 |
| OpenScore Quartets | 2,529 | fermata: +1,026 |
| DeepScores Fermata | 147 | fermata: +1,712 |
| OpenScore Quartets New | 112 | fermata: +113 |

### 3. 瓶頸類別改進潛力

| 類別 | 當前 mAP | Phase 8 樣本 | 可新增 | 增幅 |
|------|----------|--------------|--------|------|
| barline_double | 0.231 | ~3,600 | **+3,724** | +103% |
| dynamic_soft | 0.572 | ~3,500 | **+3,125** | +89% |
| fermata | 0.644 | ~5,300 | **+7,033** | +133% |

---

## 📊 原 Phase 9 計劃 vs 修正計劃

### 原計劃問題

```
原計劃: 清理 tiny bbox → 訓練
問題:
1. tiny bbox 與 mAP 相關性僅 -0.143
2. 會移除大量有效數據 (如 flag_16th 84% 數據)
3. barline_double 問題不在 tiny bbox (只有 0.2% tiny)
4. 忽略了 8,726 張未使用圖片
```

### 修正計劃

```
修正計劃:
1. 合併所有未使用數據
2. 跳過或最小化 tiny bbox 清理
3. 訓練合併後的完整數據集
```

---

## ✅ 修正後的 Phase 9 執行計劃

### Step 1: 合併未使用數據 (新增)

```bash
# 需要合併的數據集:
datasets/yolo_deepscores_dynamics/      # 700 imgs, dynamics
datasets/yolo_deepscores_converted/     # 147 imgs, fermata
datasets/yolo_openscore_lieder/         # 5238 imgs, barline_double + fermata
datasets/yolo_openscore_quartets/       # 2529 imgs, fermata
datasets/yolo_openscore_quartets_new/   # 112 imgs, fermata
```

### Step 2: 創建 Phase 9 合併數據集

```
Phase 8 Final: 32,555 訓練 + 3,617 驗證
+ 未使用數據: ~8,726 圖片
= Phase 9 Merged: ~41,281 圖片
```

### Step 3: 訓練 (不做 tiny bbox 清理)

使用合併後的完整數據集訓練。

---

## 🎯 預期改善

| 指標 | Phase 8 | Phase 9 預期 | 改善 |
|------|---------|--------------|------|
| mAP50 | 0.644 | **0.72+** | +12% |
| barline_double | 0.231 | **0.50+** | +117% |
| dynamic_soft | 0.572 | **0.68+** | +19% |
| fermata | 0.644 | **0.75+** | +16% |

---

## 🔄 需要執行的操作

1. **創建合併腳本** `scripts/merge_phase9_datasets.py`
2. **更新訓練腳本** 指向合併後的數據集
3. **刪除或禁用 tiny bbox 清理**
4. **執行訓練**

---

*報告生成時間: 2025-11-28*
