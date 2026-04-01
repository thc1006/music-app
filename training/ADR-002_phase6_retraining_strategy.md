# ADR-002: Phase 6 Retraining Strategy — Surgical cv2 Reset for Annotation-Shift Recovery

| 欄位 | 值 |
|------|----|
| 狀態 | **PROPOSED** |
| 日期 | 2026-02-26 |
| 作者 | Claude + thc1006 |
| 前置 | ADR-001 (OpenScore notehead bbox fix) |

---

## 1. 問題描述

### 1.1 根本原因

Phase 5 模型（mAP50=0.827）是在**錯誤的 OpenScore notehead 標註**上訓練的。OpenScore 的 LilyPond glyph-group bounding box 約為 380×335px，而正確的 notehead bbox 只有 23×25px。Phase 6 已修正這些標註（ADR-001 驗證通過，799,583 noteheads 零錯誤）。

### 1.2 量化分析：模型預測 vs 修正 GT

| 指標 | Phase 5 模型預測 | Phase 6 修正 GT | 差距 |
|------|-----------------|----------------|------|
| 平均寬度 | 532.6 px | 23.6 px | **22.6x** |
| 平均高度 | 705.7 px | 25.6 px | **27.6x** |
| 平均面積 | 375,890 px² | 604 px² | **622x** |
| IoU（完美對齊） | — | — | **0.0016** |
| IoU≥0.3 匹配率 | 0/1,990 | — | **0.0%** |
| IoU≥0.5 匹配率 | 0/1,990 | — | **0.0%** |

預測信心度中位數 = 0.885（模型**高度自信地**輸出完全錯誤的框）。

94.4% 的 notehead 預測 > 200px 寬，0% 落在正確尺寸範圍（< 30px）。

### 1.3 影響範圍

**Phase 5 模型在 Phase 6 修正 GT 上的表現：**

| 指標 | Phase 5 val (舊 GT) | Phase 6 val (修正 GT) | 差異 |
|------|---------------------|----------------------|------|
| 整體 mAP50 | 0.8252 | 0.7873 | -0.038 |
| nh_filled mAP50 | 0.740 | **0.256** | **-0.484** |
| nh_hollow mAP50 | 0.639 | **0.017** | **-0.622** |
| nh_filled Recall | 0.513 | **0.180** | **-0.333** |
| nh_hollow Recall | 0.411 | **0.025** | **-0.386** |

Notehead 指標崩潰（mAP50 下降 48-62pp），但**其他 30 個 class 不受影響**。

### 1.4 架構關鍵洞察：cv2 是 class-agnostic

YOLO12s Detect Head 結構：

```
Detect (layer 21, 831,792 params, 9.0% of total):
├── cv2 (bbox regression, CLASS-AGNOSTIC): 639,936 params (6.9%)
│   ├── cv2[0] (stride=8):  Conv(128→64,3) → Conv(64→64,3) → Conv2d(64→64,1)
│   ├── cv2[1] (stride=16): Conv(256→64,3) → Conv(64→64,3) → Conv2d(64→64,1)
│   └── cv2[2] (stride=32): Conv(512→64,3) → Conv(64→64,3) → Conv2d(64→64,1)
│   Output: 4 × reg_max(16) = 64 channels (DFL offset distributions)
│
└── cv3 (classification, PER-CLASS): 191,840 params (2.1%)
    ├── cv3[0] (stride=8):  DWConv→Conv→DWConv→Conv→Conv2d(→32)
    ├── cv3[1] (stride=16): DWConv→Conv→DWConv→Conv→Conv2d(→32)
    └── cv3[2] (stride=32): DWConv→Conv→DWConv→Conv→Conv2d(→32)
    Output: 32 channels (class logits)
```

**cv2 是共享的**——所有 32 個 class 使用相同的 bbox regression 權重。
OpenScore 巨大 notehead bbox 不僅污染了 notehead 的 regression，而是**污染了整個 cv2**。

但好消息：
- Backbone (58.8%) 學到了正確的視覺特徵（其他 class 正常）
- cv3 (2.1%) 學到了正確的分類（能辨認 notehead）
- 問題完全集中在 cv2 (6.9%)

---

## 2. 方案分析

### 方案 A: 直接 Fine-Tune（當前 train_phase6_fixed.py）

**做法**: 不重置任何權重，直接用修正後的 Phase 6 資料集 fine-tune。

| 優點 | 缺點 |
|------|------|
| 最簡單，不需額外程式碼 | cv2 從「預測 500px 框」開始，需要大幅度梯度更新 |
| 已有完整腳本 | freeze=10 時 cv2 用正常 LR 可能震盪 |
|  | 不確定 AdamW 能否從 622x 的起點有效收斂 |

**風險**: 中等。cv2 可能需要很多 epoch 才能從「預測巨框」收斂到「預測小框」。初期 loss 會非常高，可能影響其他 class 的穩定性。

### 方案 B: Surgical cv2 Reset + Two-Stage（推薦）

**做法**: 手動重置 cv2（bbox regression）的所有權重為 Kaiming 初始化，保留 backbone + neck + cv3。

| 優點 | 缺點 |
|------|------|
| cv2 從乾淨起點開始，無 622x 包袱 | 需要額外 15 行 reset 程式碼 |
| 保留 cv3 的正確分類能力 | cv2 重置後初期幾個 epoch mAP 會下降 |
| 有學術支持：LP-FT (ICML 2022) | Stage 1 和 Stage 2 的切換需要 tuning |
| Surgical Fine-Tuning (ICLR 2023) |  |
| 僅重置 6.9% 的參數 |  |

**學術依據**:
- **LP-FT** (ICML 2022): Linear Probing → Fine-Tune。先訓練 head，再全模型。比直接 fine-tune **OOD 準確率高 10%**。
- **Surgical Fine-Tuning** (ICLR 2023): 對 label/output distribution shift，只調 head 層 = 最佳策略。
- **arXiv 2505.01016** (2025): YOLOv8 freeze=10（backbone only）在原始任務上**僅損失 < 0.1% mAP**。

### 方案 C: Full Head Reset（cv2 + cv3）

**做法**: 同時重置 cv2 和 cv3。

| 優點 | 缺點 |
|------|------|
| 最乾淨的 head | 浪費 cv3 已學到的分類知識 |
| 排除 cv3 可能的隱性問題 | 需要更多 epoch 重新學習分類 |

**分析**: cv3 是 per-class 的，notehead 的分類特徵可能受污染（因為它學到的 notehead "context" 是 380px 的大範圍），但也可能仍然有效（因為分類不依賴 bbox 大小）。作為 fallback 方案。

### 方案 D: 從 COCO 預訓練重新訓練

**做法**: 使用 `yolo12s.pt`（COCO 預訓練）從頭訓練。

| 優點 | 缺點 |
|------|------|
| 完全乾淨 | 浪費 5.4M backbone 在 OMR 上學到的特徵 |
| 無任何遺留問題 | 需要 200+ epoch（Phase 5 用了 200 epoch） |
|  | 預計 40-50 小時 GPU |

**分析**: 大量研究表明保留 backbone 比重訓更好（5-10% mAP 差異）。除非 B/C 都失敗，否則不應選此方案。

### 方案 E: Knowledge Distillation（保護非 notehead class）

**做法**: 用 Phase 5 模型作 teacher，student 在 notehead 上只用 GT loss，在其他 class 上加 distillation loss。

| 優點 | 缺點 |
|------|------|
| 理論上最安全 | 實作複雜度最高（自訂 loss） |
| YOLO LwF 有先例 | YOLO LwF (2025) 指出對 anchor-free 效果差 |
|  | 我們有完整資料集，LwF 設計用於缺少舊資料的場景 |

**分析**: 過度工程。我們有全部 32 class 的修正標註，且 30 個 class 的標註未變。直接 fine-tune 這些 class 的正確標註就是天然的 "regularization"。

---

## 3. 決策

### 選擇：方案 B — Surgical cv2 Reset + Two-Stage Training

**理由排序**:

1. **問題定位精確**: cv2 是唯一被污染的元件（6.9% 參數），重置它移除 622x 的 bias
2. **學術強支撐**: LP-FT + Surgical Fine-Tuning 兩篇頂會論文驗證此策略
3. **保留最大價值**: backbone (58.8%) + neck (32.2%) + cv3 (2.1%) 共 93.1% 參數不動
4. **實作簡單**: 約 15 行 PyTorch 程式碼
5. **有 fallback**: 若 mAP 不如預期，Stage 2 的 unfreeze 會自然修正

**Fallback 計劃**:
- 若 Stage 1 完成後 notehead mAP50 < 0.30 → 切換到方案 C（加重置 cv3）
- 若 Stage 2 完成後整體 mAP50 < 0.80 → 切換到方案 D（從 COCO 重訓）

---

## 4. 實作計劃

### 4.1 cv2 Surgical Reset 程式碼

```python
import torch.nn as nn
import math

def surgical_reset_cv2(model, logger):
    """Reset bbox regression head (cv2) with Kaiming init.

    Rationale: cv2 is class-agnostic and was trained on 622x oversized
    OpenScore glyph-group bboxes. Reset to clean slate while preserving
    backbone features and classification head (cv3).

    Architecture (YOLO12s):
      cv2[i]: Conv(ch→64,3) → Conv(64→64,3) → Conv2d(64→64,1)
      Total: 639,936 params (6.9% of model)
    """
    detect = model.model.model[-1]  # Detect layer (layer 21)

    reset_params = 0
    for i, seq in enumerate(detect.cv2):
        for m in seq.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                reset_params += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                reset_params += m.weight.numel() + m.bias.numel()
        # Ultralytics bias_init: box regression bias = 2.0
        seq[-1].bias.data[:] = 2.0

    logger.info("cv2 reset complete: %d params re-initialized (%.1f%% of model)",
                reset_params, reset_params / sum(p.numel() for p in model.model.parameters()) * 100)
    return model
```

### 4.2 防止 Ultralytics intersect_dicts 覆蓋 reset

**關鍵陷阱**: Ultralytics 的 `model.train()` 內部會呼叫 `intersect_dicts` 載入 checkpoint 權重。如果 checkpoint 的 cv2 shape 與新模型一致（相同 nc=32），它會**靜默地把舊的 cv2 權重載回去**，覆蓋我們的 reset。

**解決方案**: reset 後立即 save 到新的 checkpoint，然後從新 checkpoint 載入。

```python
# 1. Load model
model = YOLO(BASE_MODEL)

# 2. Reset cv2
surgical_reset_cv2(model, logger)

# 3. Save to new checkpoint (CRITICAL: prevents intersect_dicts from restoring old cv2)
reset_ckpt = Path(PROJECT) / "phase6_cv2_reset.pt"
torch.save({"model": model.model}, str(reset_ckpt))

# 4. Load from reset checkpoint for training
model = YOLO(str(reset_ckpt))
model.train(...)  # intersect_dicts will match all shapes → loads reset cv2
```

### 4.3 Two-Stage Training 配置

#### Stage 1: Freeze Backbone, Train Neck + Head（40 epochs）

```python
STAGE1_ARGS = {
    "data": DATA,
    "imgsz": 1280,
    "batch": 6,
    "nbs": 64,
    "device": "0",
    "workers": 12,
    "amp": False,
    "optimizer": "AdamW",
    "lr0": 0.001,        # cv2 從 Kaiming init 開始，需要正常 LR
    "lrf": 0.1,
    "freeze": 10,         # 凍結 backbone layers 0-9 (58.8% 參數)
    "epochs": 40,
    "patience": 20,
    # Loss weights
    "cls": 1.0,
    "box": 7.5,           # 提高 box loss 權重：cv2 需要快速收斂
    "dfl": 1.5,
    # No augmentation
    "mosaic": 0.0,
    "copy_paste": 0.0,
    "mixup": 0.0,
    "max_det": 1500,
    "compile": True,
}
```

**box=7.5 的理由**: cv2 從隨機初始化開始，bbox regression 是最需要學習的元件。提高 box loss 權重讓梯度更集中在 cv2。Phase 5 使用 box=5.0，這裡提高 50% 因為 cv2 是全新的。

#### Stage 2: Unfreeze All, Low LR（120 epochs）

```python
STAGE2_ARGS = {
    **STAGE1_ARGS,
    "freeze": None,       # 解凍所有層
    "lr0": 0.0001,        # 10x 低於 Stage 1
    "lrf": 0.01,
    "epochs": 120,        # 比 Phase 5 少（因為有 93% 預訓練參數）
    "patience": 40,
    "box": 5.0,           # 恢復正常比例（cv2 已基本收斂）
    "warmup_epochs": 2.0,
}
```

### 4.4 成功標準（分級）

| 級別 | 條件 | 行動 |
|------|------|------|
| **A (突破)** | mAP50 > 0.85 AND nh_filled Recall > 0.70 | 進入 ADR-003 部署 |
| **B (達標)** | mAP50 > 0.83 AND nh_filled mAP50 > 0.60 | 可接受，進行 Phase 7 微調 |
| **C (不足)** | mAP50 0.80-0.83 OR nh mAP50 0.40-0.60 | 切換方案 C (full head reset) |
| **D (失敗)** | mAP50 < 0.80 | 切換方案 D (COCO retrain) |

### 4.5 監控指標（每 epoch）

除標準 mAP/Recall 外，特別關注：

1. **nh_filled mAP50**: 核心指標，應在 10 epoch 內從 ~0.05 上升到 > 0.30
2. **nh_hollow mAP50**: 第二核心指標
3. **beam/barline mAP50**: 不應下降超過 2pp（防止 cv2 reset 連帶影響）
4. **box_loss**: Stage 1 初期會很高（cv2 從隨機開始），應在 5 epoch 內穩定下降
5. **val/box_loss vs val/cls_loss ratio**: 若 box_loss 佔比 > 80%，代表 cv2 仍在追趕

### 4.6 時間預估

| 階段 | Epochs | 預估時間 |
|------|--------|---------|
| cv2 Reset + Save | — | < 1 分鐘 |
| Stage 1 (freeze=10) | 40 | ~4-5 小時 |
| Stage 2 (full) | 120 | ~15-18 小時 |
| Final Eval | — | ~30 分鐘 |
| **總計** | **160** | **~20-24 小時** |

---

## 5. 研究參考文獻

| 論文 | 會議 | 核心發現 | 與本案相關性 |
|------|------|---------|-------------|
| [LP-FT](https://arxiv.org/abs/2202.10054) | ICML 2022 | 先訓練 head 再 full fine-tune，OOD +10% | 直接對應：先訓練 reset cv2 再解凍 |
| [Surgical Fine-Tuning](https://openreview.net/forum?id=APuPRxjHvZ) | ICLR 2023 | Label shift → 只調 head 層效果最佳 | 我們的情況正是 label/output shift |
| [Fine-Tune Without Forgetting](https://arxiv.org/abs/2505.01016) | arXiv 2025 | freeze=10 原始任務 < 0.1% 退化 | 確認 freeze backbone 安全 |
| [Label Error → Correction](https://arxiv.org/abs/2508.06556) | arXiv 2025 | 修正標註後重訓 mAP +16-46% | 確認我們的方向正確 |
| [YOLO LwF](https://arxiv.org/abs/2503.04688) | arXiv 2025 | LwF 對 anchor-free 效果差 | 排除方案 E |
| [Incremental Detection](https://arxiv.org/abs/1708.06977) | ICCV 2017 | 有完整資料集時不需 distillation | 排除方案 E |

---

## 6. 風險與緩解

| 風險 | 機率 | 影響 | 緩解措施 |
|------|------|------|---------|
| cv2 reset 影響非 notehead class | 中 | box regression 短暫變差 | Stage 1 freeze backbone，Stage 2 低 LR |
| Stage 1 收斂太慢 | 低 | 40 epoch 不夠 cv2 學會小框 | box=7.5 加速，patience=20 自動延長 |
| intersect_dicts 覆蓋 reset | 高 (如果沒處理) | 重置無效 | 4.2 節的 save → reload 方案 |
| Stage 2 catastrophic forgetting | 低 | 其他 class 退化 | lr0=0.0001，30 class 正確標註作 regularizer |
| Phase 5 模型不如 Ultimate v5 作 base | 低 | 起點偏低 | Phase 5 (0.827) > Ultimate v5 (0.810)，確認更好 |

---

## 7. 與現有 train_phase6_fixed.py 的差異

| 項目 | 現有腳本 | ADR-002 修改 |
|------|---------|-------------|
| cv2 reset | **無** | **新增 surgical_reset_cv2()** |
| intersect_dicts 防護 | **無** | **新增 save → reload** |
| Stage 1 box loss | 5.0 | **7.5**（加速 cv2 收斂） |
| Stage 2 epochs | 160 | **120**（有 93% 預訓練） |
| Stage 2 box loss | 5.0 | 5.0（不變） |
| Fallback 邏輯 | 無 | **新增 Stage 1 後 mAP 檢查** |
| 監控 | 基本 JSONL | **新增 cv2 收斂追蹤** |

需要修改的程式碼量：約 40 行（reset 函數 + save/reload + box 參數調整）。

---

## 8. 下一步行動

1. **等待 Phase 5 完成或決定提早停止**（目前 epoch 52/160，平台期）
2. **修改 `train_phase6_fixed.py`**：加入 cv2 reset + intersect_dicts 防護
3. **啟動 Phase 6 訓練**（GPU 空閒後）
4. **Stage 1 完成後 checkpoint review**：檢查 notehead mAP50 是否 > 0.30
5. **Stage 2 完成後 full eval**：對照 4.4 節成功標準決定下一步
