# 模型精進實驗計劃

**目標**: 從 mAP50=0.698 提升到 0.72+
**核心問題**: tie ↔ barline_final 互相混淆 (35% 誤分類率)

---

## 實驗 1: Focal Loss gamma 調整

**原理**: 增加 gamma 讓模型更關注困難樣本（如 tie/barline_final 混淆樣本）

```python
# 預設 gamma=1.5, 嘗試 gamma=2.0 和 gamma=2.5
model.train(
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
    epochs=100,
    imgsz=1280,
    batch=4,
    fl_gamma=2.0,  # 增加 focal loss gamma
    resume=False,
    project='harmony_omr_v2_experiments',
    name='exp1_focal_gamma_2.0'
)
```

**預期效果**: +1-2% mAP50
**風險**: 訓練時間可能變長

---

## 實驗 2: 分類損失權重調整

**原理**: 增加分類損失權重，讓模型更注重區分相似類別

```python
model.train(
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
    epochs=100,
    imgsz=1280,
    batch=4,
    cls=1.0,  # 預設 0.5, 增加到 1.0
    box=7.5,  # 保持預設
    resume=False,
    project='harmony_omr_v2_experiments',
    name='exp2_cls_weight_1.0'
)
```

**預期效果**: 改善類別混淆問題
**風險**: 可能影響定位精度

---

## 實驗 3: 弱類別 Oversampling

**原理**: 對 tie, barline_final, barline_double 進行圖片級別的過採樣

```bash
# 創建過採樣數據集
# 將包含弱類別的圖片複製 2-3 份
python create_oversampled_dataset.py \
    --source datasets/yolo_harmony_v2_phase8_final \
    --output datasets/yolo_harmony_v2_oversampled \
    --weak_classes tie,barline_final,barline_double \
    --oversample_factor 3
```

**預期效果**: +1-3% mAP50 (弱類別)
**風險**: 可能導致其他類別過擬合

---

## 實驗 4: Mosaic + MixUp 增強調整

**原理**: 增加 mosaic 和 mixup 增強，幫助模型學習更多上下文

```python
model.train(
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
    epochs=100,
    imgsz=1280,
    batch=4,
    mosaic=1.0,    # 保持
    mixup=0.15,    # 從 0 增加到 0.15
    copy_paste=0.1, # 啟用 copy-paste
    resume=False,
    project='harmony_omr_v2_experiments',
    name='exp4_augmentation'
)
```

**預期效果**: 提升泛化能力
**風險**: 訓練時間增加

---

## 實驗 5: 降低置信度 + NMS IoU 調整

**原理**: 調整後處理參數，減少漏檢

```python
# 訓練時使用較低的 NMS IoU 閾值
model.train(
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
    epochs=100,
    imgsz=1280,
    batch=4,
    nms=True,
    iou=0.6,  # 降低 IoU 閾值 (預設 0.7)
    resume=False,
    project='harmony_omr_v2_experiments',
    name='exp5_nms_iou_0.6'
)
```

**預期效果**: 減少漏檢
**風險**: 可能增加重複檢測

---

## 實驗 6: 分階段微調 (Two-Stage Fine-tuning)

**原理**: 先凍結 backbone，只訓練 head，專注於分類能力

```python
# Stage 1: 凍結 backbone，微調 head
model = YOLO('harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt')
model.train(
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
    epochs=30,
    imgsz=1280,
    batch=8,
    freeze=10,  # 凍結前 10 層
    lr0=0.0001,  # 小學習率
    project='harmony_omr_v2_experiments',
    name='exp6_finetune_head'
)
```

**預期效果**: 改善分類能力而不損失定位精度
**風險**: 效果可能有限

---

## 快速驗證實驗: Ray Tune 自動調參

**原理**: 讓算法自動搜索最佳超參數

```python
from ultralytics import YOLO

model = YOLO('harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt')
result = model.tune(
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
    epochs=30,
    iterations=50,
    optimizer='AdamW',
    plots=True,
    save=True,
    val=True
)
```

---

## 實驗優先級排序

| 優先級 | 實驗 | 原因 | 預計時間 |
|--------|------|------|----------|
| 1 | 實驗 2 (cls 權重) | 直接針對分類混淆問題 | ~8 小時 |
| 2 | 實驗 1 (Focal Loss) | 關注困難樣本 | ~10 小時 |
| 3 | 實驗 6 (分階段微調) | 風險低，快速驗證 | ~3 小時 |
| 4 | 實驗 3 (Oversampling) | 需要準備數據 | ~10 小時 |
| 5 | 實驗 4 (增強調整) | 可能有幫助 | ~10 小時 |
| 6 | Ray Tune | 自動化搜索 | ~24 小時 |

---

## 建議執行順序

1. **先跑實驗 6** (分階段微調) - 只需 3 小時，快速驗證
2. **並行跑實驗 1 + 2** - 這兩個最可能有效
3. **根據結果決定**是否需要 Oversampling 或 Ray Tune

---

*Created: 2026-01-09*
