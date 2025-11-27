# Phase 6 Custom Training System

完整的 Hard Example Mining (HEM) 系統和自定義損失函數，專門用於提升 barline 類別的檢測性能。

## 問題背景

Phase 5 訓練後的 barline 檢測性能極差：

| 類別 | mAP50 | 召回率 | 問題 |
|------|-------|--------|------|
| barline (23) | 0.201 | 9% | **91% 漏檢率** |
| barline_double (24) | 0.140 | 13.3% | **86.7% 漏檢率** |
| barline_final (25) | 0.708 | 52.5% | 精確率虛高 (框過大) |
| barline_repeat (26) | 0.879 | 83% | 良好 |

**根本原因**：
1. 極細線標註問題 (34% barline 寬度 < 0.005)
2. 標註框膨脹 (95% barline_final 面積 > 0.1)
3. 樣本稀少 (barline_double 只有 1,883 個)
4. 類別混淆 (barline vs stem)

## 解決方案

Phase 6 採用多階段訓練策略：

### Stage 1: 加權損失 + 加權採樣 (150 epochs)
- **Per-class weighted loss**: barline_double 8x, barline 4x
- **Weighted sampling**: 增加包含 barline 的圖片採樣概率
- **Focal loss**: gamma=2.0, 聚焦難檢測樣本
- **預期**: mAP50 0.58 → 0.62

### Stage 2: Hard Example 微調 (50 epochs)
- **Hard Example Mining**: 識別 False Negatives, 低置信度預測
- **專注數據集**: 只包含難檢測案例
- **更高學習率**: 快速適應
- **預期**: barline mAP 0.20 → 0.50, barline_double 0.14 → 0.40

## 目錄結構

```
custom_training/
├── __init__.py                    # Package initialization
├── barline_focused_loss.py        # 自定義損失函數
├── hard_example_miner.py          # HEM 系統
├── weighted_sampler.py            # 加權數據採樣
├── train_phase6.py                # 主訓練腳本
├── configs/
│   └── phase6_config.yaml         # 訓練配置
└── README.md                      # 本文件
```

## 快速開始

### 1. 環境準備

```bash
# 確保已安裝 Ultralytics
pip install ultralytics

# 進入訓練目錄
cd /home/thc1006/dev/music-app/training
```

### 2. 運行完整 Pipeline

```bash
# 自動運行 Stage 1 → HEM → Stage 2
python custom_training/train_phase6.py
```

這將：
1. 從 Phase 5 best.pt 開始
2. Stage 1 訓練 (150 epochs, ~4-6 小時)
3. 自動挖掘 hard examples (~30-60 分鐘)
4. Stage 2 微調 (50 epochs, ~1-2 小時)
5. 最終評估並生成報告

**總計時間**: 6-9 小時 (RTX 5090)

### 3. 單獨運行各組件

#### 測試 Barline-Focused Loss

```bash
python custom_training/barline_focused_loss.py
```

輸出：
- 類別權重配置
- 動態權重計算示例
- 不同策略的權重比較

#### 測試 Hard Example Mining

```bash
python custom_training/hard_example_miner.py \
  /path/to/phase5/best.pt \
  /path/to/harmony_phase5.yaml
```

輸出：
- `hard_examples_analysis/` - 分析結果
- `hard_examples_barline.txt` - barline 難例圖片列表
- `hard_examples_stats.json` - 統計數據
- `hard_examples_dataset/` - 難例數據集

#### 測試 Weighted Sampler

```bash
python custom_training/weighted_sampler.py \
  /path/to/labels/train
```

輸出：
- 類別分佈分析
- 最優權重計算 (多種策略)
- 採樣測試

## 配置文件說明

`configs/phase6_config.yaml` 包含完整配置：

### Stage 1 關鍵參數

```yaml
stage1:
  epochs: 150
  lr0: 0.001
  copy_paste: 0.4        # 增加稀有類別增強
  cls: 2.5               # 提高分類權重

  class_weights:
    23: 4.0              # barline
    24: 8.0              # barline_double (最難)
    25: 2.0              # barline_final
    26: 1.0              # barline_repeat

  sampling_weights:
    23: 5.0              # 5x 採樣機率
    24: 8.0              # 8x 採樣機率
    25: 2.0
    26: 1.5
```

### HEM 參數

```yaml
hem:
  conf_threshold: 0.5      # 預測置信度閾值
  iou_threshold: 0.5       # IoU 匹配閾值
  low_conf_threshold: 0.3  # 低置信度閾值
  min_difficulty: 1.5      # 最小難度分數
```

### Stage 2 關鍵參數

```yaml
stage2:
  epochs: 50
  lr0: 0.0005            # 較低學習率 (微調)
  copy_paste: 0.5        # 最大增強
  cls: 3.0               # 更高分類權重
  box: 10.0              # 強調 bbox 準確度
```

## 組件詳解

### 1. Barline-Focused Loss

**功能**：
- Per-class weighted focal loss
- 小物體強化 (針對極細 barline)
- IoU-based bbox weighting

**類別權重**：
```python
class_weights = {
    23: 4.0,   # barline - 召回率 9%
    24: 8.0,   # barline_double - 最差 (mAP 0.140)
    25: 2.0,   # barline_final
    26: 1.0,   # barline_repeat - 已經很好
}
```

**Focal Loss**：
```python
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
# gamma=2.0: 聚焦難例
# alpha=0.25: 平衡正負樣本
```

**小物體檢測**：
- 面積 < 0.01 的物體 bbox 損失權重 2x
- 針對極細 barline (34% 寬度 < 0.005)

### 2. Hard Example Miner

**識別標準**：

1. **False Negatives (FN)**:
   - Ground truth 存在但模型完全漏檢
   - Difficulty score: 2.0 (最高)

2. **Low Confidence Predictions**:
   - 檢測到但置信度 < 0.3
   - Difficulty score: 1.0 + (0.3 - conf)

3. **Misclassifications**:
   - IoU > 0.5 但類別錯誤
   - Difficulty score: 1.5

**輸出**：
- 每個類別的難例圖片列表
- 詳細 JSON 報告 (包含 bbox, 置信度, IoU)
- 可選的可視化 (annotated images)

### 3. Weighted Sampler

**策略**：

1. **Per-image weighting**:
   - 圖片權重 = max(class_weight for all classes in image)
   - 包含 barline_double 的圖片: 8x 機率
   - 包含 barline 的圖片: 5x 機率

2. **Progressive weighting** (可選):
   - 前 50 epochs: 逐步增加權重 (避免過擬合)
   - 後續 epochs: 完整權重

3. **與 YOLO 整合**:
   - 通過 weighted sampler 替換預設 DataLoader
   - 保持 YOLO 其他功能不變

## 預期改進

### 目標性能

| 類別 | Phase 5 | Phase 6 目標 | 提升 |
|------|---------|-------------|------|
| barline | 0.201 | 0.50-0.60 | **+150-200%** |
| barline_double | 0.140 | 0.40-0.50 | **+185-260%** |
| barline_final | 0.708 | 0.70-0.75 | +0-6% |
| barline_repeat | 0.879 | 0.87-0.89 | -1-1% |
| **Overall mAP50** | 0.580 | **0.65-0.68** | **+12-17%** |

### 關鍵成功因素

1. **數據品質優先**:
   - 修復極細線標註 (寬度 < 0.005)
   - 緊縮過大標註框 (barline_final, barline_double)

2. **訓練策略**:
   - 加權損失 + 加權採樣雙重作用
   - Hard example 專注訓練
   - 兩階段漸進優化

3. **評估指標**:
   - 不只看 mAP50，更要看召回率
   - Per-class 詳細分析
   - Confusion matrix 識別混淆

## 監控訓練

### Stage 1 監控

```bash
# 查看訓練日誌
tail -f harmony_omr_v2_phase6/stage1_weighted_loss/train.log

# TensorBoard (如果啟用)
tensorboard --logdir harmony_omr_v2_phase6/stage1_weighted_loss
```

**關鍵指標**：
- `cls_loss` 應逐步下降 (當前 1.25 → 目標 0.8-1.0)
- `box_loss` 穩定在 0.4 左右
- `mAP50` 應從 0.58 提升至 0.62+

### HEM 監控

```bash
# 查看挖掘結果
cat hard_examples_analysis/hard_examples_stats.json

# 難例數量
wc -l hard_examples_analysis/hard_examples_barline.txt
```

**預期**：
- barline FN: ~2,000-2,500 (91% 漏檢)
- barline_double FN: ~150-170 (86.7% 漏檢)
- Low conf: ~500-1,000

### Stage 2 監控

```bash
# 查看微調日誌
tail -f harmony_omr_v2_phase6/stage2_hard_examples/train.log
```

**關鍵指標**：
- 前 10 epochs loss 應快速下降
- barline 類別 precision/recall 應顯著提升
- 避免過擬合到難例 (監控 val loss)

## 故障排除

### 問題 1: OOM (Out of Memory)

**症狀**: CUDA out of memory error

**解決**：
```yaml
# 減少 batch size
batch: 16 → 12 或 8

# 關閉 cache
cache: false

# 降低 workers
workers: 8 → 4
```

### 問題 2: Stage 1 mAP 提升不明顯

**可能原因**:
1. 類別權重不夠高
2. 採樣權重未生效
3. 數據品質問題

**解決**：
```yaml
# 增加 barline 權重
class_weights:
  23: 6.0  # 4.0 → 6.0
  24: 10.0 # 8.0 → 10.0

# 提高分類損失權重
cls: 2.5 → 3.0

# 增加 copy_paste
copy_paste: 0.4 → 0.5
```

### 問題 3: Hard Examples 過少

**症狀**: HEM 只找到少量難例

**解決**：
```yaml
# 降低難度閾值
min_difficulty: 1.5 → 1.0

# 提高低置信度閾值
low_conf_threshold: 0.3 → 0.4

# 分析更多圖片
max_images: null  # 使用全部驗證集
```

### 問題 4: Stage 2 過擬合

**症狀**: Train loss 下降但 val loss 上升

**解決**：
```yaml
# 降低學習率
lr0: 0.0005 → 0.0003

# 減少 epochs
epochs: 50 → 30

# 增加 patience
patience: 15 → 10

# 更早關閉 mosaic
close_mosaic: 10 → 5
```

## 輸出結果

訓練完成後會生成：

```
harmony_omr_v2_phase6/
├── stage1_weighted_loss/
│   ├── weights/
│   │   ├── best.pt              # Stage 1 最佳權重
│   │   └── last.pt
│   ├── results.csv              # 訓練曲線
│   ├── confusion_matrix.png
│   └── PR_curve.png
│
├── hard_example_mining/
│   ├── hard_examples_barline.txt
│   ├── hard_examples_barline_double.txt
│   ├── hard_examples_stats.json  # 詳細統計
│   └── hard_examples_detailed.json
│
├── hard_examples_dataset/
│   ├── images/train/            # 難例圖片
│   ├── labels/train/            # 難例標註
│   └── hard_examples.yaml       # 數據集配置
│
├── stage2_hard_examples/
│   ├── weights/
│   │   ├── best.pt              # 最終模型 ⭐
│   │   └── last.pt
│   ├── results.csv
│   └── confusion_matrix.png
│
└── phase6_summary.json          # 完整訓練摘要
```

## 下一步

訓練完成後：

1. **評估改進**:
   ```bash
   # 與 Phase 5 比較
   python compare_phase5_vs_phase6.py
   ```

2. **錯誤分析**:
   - 查看 confusion matrix
   - 分析剩餘的 False Negatives
   - 識別新的問題模式

3. **如果未達目標**:
   - Phase 6.1: 修復數據標註問題
   - Phase 6.2: 更激進的權重 (barline_double 15x)
   - Phase 6.3: 合成數據增強 (Abjad + LilyPond)

4. **如果達到目標**:
   - 匯出 TFLite 模型
   - 整合到 Android App
   - 進行實際樂譜測試

## 參考文獻

- **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
- **Hard Example Mining**: Shrivastava et al. "Training Region-based Object Detectors with Online Hard Example Mining" (CVPR 2016)
- **Class-Balanced Loss**: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
- **YOLO**: Ultralytics YOLOv8/v12 Documentation

## 聯繫

如有問題或建議，請聯繫 Harmony OMR Team。

---

**最後更新**: 2025-11-26
**版本**: 1.0.0
**狀態**: Ready for deployment
