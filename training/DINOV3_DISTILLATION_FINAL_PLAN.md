# DINOv3 知識蒸餾最終實施計劃

## 創建日期: 2025-12-20
## 目標: 使用 DINOv3 蒸餾提升 OMR 模型至 mAP50 > 0.70

---

## 1. 當前狀態總結

### 1.1 最佳模型
| 模型 | mAP50 | mAP50-95 | 資料集 |
|------|-------|----------|--------|
| **Phase 8** | **0.6444** | 0.5809 | 32,555 訓練圖 |
| Phase 10.1 v2 | 0.6186 | 0.5569 | 33,410 訓練圖 |
| Phase 9 | 0.5841 | 0.5213 | 41,281 訓練圖 |

**結論**: Phase 8 是當前最佳，Phase 9 因資料品質問題退步

### 1.2 可用資料集

| 資料集 | 訓練圖 | 驗證圖 | 狀態 |
|--------|--------|--------|------|
| **yolo_harmony_v2_phase8_final** | 32,555 | 3,617 | ✅ 最佳品質 |
| yolo_harmony_v2_phase10_1 | 33,410 | 3,617 | ✅ 可用 |
| external/AudioLabs_v2 | - | - | 可整合 |
| external/openscore_lieder | - | - | 可整合 |
| external/deepscores_v2 | - | - | 可整合 |

### 1.3 已驗證的 DINOv3 環境

```
✅ timm 1.0.22 - 11 個 DINOv3 模型可用
✅ LightlyTrain 0.13.1 - 蒸餾 API 可用
✅ PyTorch 2.9.1+cu128
✅ RTX 5090 32GB - batch_size=16 可行
✅ vit_small_patch16_dinov3 可加載 (21.6M 參數)
✅ 640x640 輸入測試通過
```

---

## 2. 技術方案選擇

### 2.1 為何選擇 DINOv3 而非 DINOv2

| 特性 | DINOv3 | DINOv2 | 選擇 |
|------|--------|--------|------|
| **640x640 輸入** | ✅ 原生支援 | ❌ 需要 518x518 | DINOv3 |
| **訓練數據** | 1.7B 圖片 | 142M 圖片 | DINOv3 12x |
| **Patch 密度** | 40x40 @640px | 37x37 @518px | DINOv3 |
| **特徵維度** | 384 | 384 | 相同 |

### 2.2 蒸餾策略

採用 **特徵蒸餾 + 目標檢測微調** 兩階段方案：

```
階段 1: 特徵蒸餾預訓練 (50 epochs)
  └─ DINOv3 教師 → YOLO12s 學生
  └─ 在無標註圖片上學習特徵表示

階段 2: 目標檢測微調 (100 epochs)
  └─ 使用蒸餾後的 backbone
  └─ 在 Phase 8 資料集上微調
```

---

## 3. 實施計劃

### Phase 0: 環境驗證 [已完成 ✅]

```bash
# 已驗證項目
- timm 1.0.22 ✅
- LightlyTrain 0.13.1 ✅
- vit_small_patch16_dinov3 可加載 ✅
- 640x640 輸入測試通過 ✅
```

### Phase 1: DINOv3 蒸餾訓練

**目標**: mAP50 > 0.68 (+5.6%)

**執行腳本**: `training/yolo12_dinov3_distillation_v2.py`

**配置**:
```python
教師模型: vit_small_patch16_dinov3 (21.6M, 384dim)
學生模型: YOLO12s (從 Phase 8 初始化)
資料集: yolo_harmony_v2_phase8_final (32,555 訓練圖)
輸入尺寸: 640x640
Batch Size: 16
蒸餾溫度: 4.0
蒸餾權重: α=0.5 (蒸餾 50% + 任務 50%)
```

### Phase 2: 高解析度微調 (可選)

**目標**: 進一步提升小物件檢測

**配置**:
```python
輸入尺寸: 1280x1280
Batch Size: 4-8
Epochs: 50
```

---

## 4. 預期結果

| 指標 | Phase 8 | 蒸餾後目標 | 提升 |
|------|---------|-----------|------|
| mAP50 | 0.6444 | > 0.68 | +5.5% |
| mAP50-95 | 0.5809 | > 0.62 | +6.7% |
| 小物件 (flag, dot) | ~0.70 | > 0.75 | +7% |
| 結構符號 (barline) | ~0.60 | > 0.70 | +17% |

---

## 5. 相關文件連結

### 訓練腳本
- `training/yolo12_dinov3_distillation_v2.py` - 蒸餾主腳本

### 資料集
- `datasets/yolo_harmony_v2_phase8_final/` - 主要訓練資料
- `datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml` - 資料集配置

### 模型
- `harmony_omr_v2_phase8/phase8_training/weights/best.pt` - 基線模型

### 外部資源
- [DINOv2/v3 GitHub](https://github.com/facebookresearch/dinov2)
- [LightlyTrain GitHub](https://github.com/lightly-ai/lightly-train)
- [timm GitHub](https://github.com/huggingface/pytorch-image-models)

---

## 6. 風險和緩解

| 風險 | 可能性 | 緩解措施 |
|------|--------|---------|
| 蒸餾效果有限 | 中 | 嘗試更大教師模型 (vit_base) |
| 顯存不足 | 低 | RTX 5090 32GB 充足 |
| 訓練不穩定 | 中 | 使用較低學習率 + 預熱 |

---

**文檔版本**: 1.0
**更新日期**: 2025-12-20
