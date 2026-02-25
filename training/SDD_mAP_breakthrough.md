# SDD: OMR mAP 突破計劃 — 從 0.7763 到 >0.85

## 1. 背景與問題陳述

### 1.1 當前最佳成績
| 模型 | 數據集 | Protocol | mAP50 | mAP50-95 | P | R |
|------|--------|----------|-------|----------|---|---|
| YOLO12s Ultimate v5 | cleaned val | deploy (conf=0.25, iou=0.55) | **0.7763** | 0.7320 | 0.957 | 0.576 |
| YOLO12s Ultimate v5 | cleaned val | strict (conf=0.001, iou=0.7) | 0.7295 | 0.6873 | 0.920 | 0.589 |
| YOLO12s Ultimate v5 | original val | deploy | 0.7519 | 0.7058 | 0.917 | 0.560 |

### 1.2 核心瓶頸（TIDE 已驗證）
- **96.8% mAP 損失來自漏檢 (Miss)**：模型太保守，42% 物件未被偵測
- **分類錯誤僅 0.7%**：模型分類能力已足夠
- **根因**：20% 有毒訓練數據 + 高 cls 門檻 + 330x 類別不平衡

### 1.3 已驗證的根因
| 問題 | 證據 | 影響 |
|------|------|------|
| 有毒數據（lieder+ds2+quartets） | Cleanlab + 視覺驗證，移除後 mAP +2.44% 無需重訓 | 直接抑制 recall |
| TAL OOM (batch=8) | 40/119 batches OOM，P95=606 GT boxes/image | 訓練不穩定 |
| 微調策略錯誤 | YOLO26 a7/a10 在 epoch 2-3 達最佳後持續下降 | LR 太高破壞特徵 |
| 微小框 (~97K) | stems w=0.002 被 TAL 擴展到 16px 但 loss 仍對 2.6px 計算 | 浪費梯度信號 |

### 1.4 硬體限制
- GPU: RTX 5090 32GB
- RAM: 125GB
- 磁碟: ~26GB 可用

### 1.5 已證明失敗的方案（不再嘗試）
SAHI (-88.6%)、TTA (-15%)、WBF 同架構集成（無效）、YOLO11m 從頭訓練 (-17.7%)、YOLO26s (TFLite Android 不可用，mAP 低 2.7%)

---

## 2. 設計決策（交叉驗證結論）

### 2.1 模型選擇：YOLO12s
**依據**: 5 個研究代理一致同意。YOLO12s 在 OMR 數據集上 7/10 維度勝出，YOLO26 TFLite Android 部署不可用 (GitHub #23282)。

### 2.2 訓練起點：從 Ultimate v5 best.pt 微調
**依據**: Agent aad693c 引用 arXiv:2505.01016 證明領域微調優於從頭訓練。best.pt 已有 200 epochs OMR 領域知識。清洗 val 上 mAP50=0.7763 證明特徵品質優良。

### 2.3 Batch size: batch=6, nbs=64
**依據**: Agent a880808 分析 TAL 記憶體 = O(batch × n_max_boxes × anchors)。batch=8 在 P95 圖片(606 boxes)需 ~5GB TAL + 24GB 模型 > 32GB。batch=6 降至 ~3.8GB，消除大部分 OOM。nbs=64 梯度累積保證等效優化。

### 2.4 超參數組合：中度激進
**依據**: Agent ab21405 文獻佐證，預期 recall +8-12%。

| 參數 | 原值 | 新值 | 依據 |
|------|------|------|------|
| lr0 | 0.001 | 0.0003 | 微調需低 LR 保護已學特徵 (arXiv:2505.01016) |
| cls | 0.5 | 1.5 | 強化分類信心，預期 recall +5-8% |
| fl_gamma | 0 | 1.5 | Focal loss 抑制易分類主導梯度 (RetinaNet, ICCV 2017) |
| copy_paste | 0.0 | 0.15 | 稀有類別增強 (CVPR 2021 Copy-Paste) |
| close_mosaic | 10 | 20 | 更長真實圖片微調期 |
| label_smoothing | 0 | 0.05 | 降低過度自信 (NeurIPS 2019) |
| scale | 0.5 | 0.7 | 更大尺度變異改善多尺度偵測 |
| mixup | 0.0 | 0.1 | 輕度正則化 |
| epochs | 200 | 150 | 微調收斂更快，配合 patience |
| patience | 30 | 50 | 新增強需適應期，避免過早停止 |

---

## 3. 執行計劃

### Phase 0: 基礎設施（Day 1, ~2 小時）
1. **統一評估腳本** `training/unified_eval.py`
   - 雙 protocol: strict (conf=0.001, iou=0.7) + deploy (conf=0.25, iou=0.55)
   - 輸出 JSON + per-class CSV
   - 對所有現有模型跑一次，建立真實排行榜

2. **數據集清洗 v2** `training/create_cleaned_dataset_v2.py`
   - 移除 lieder (4,261 train / 482 val)
   - 移除 ds2 (161 train / 14 val)
   - 移除 quartets 單標註圖 (≤2 annotations, 2,058 train)
   - 過濾微小框 (w AND h < 0.003)
   - 用 symlink 零磁碟開銷
   - 預計結果: ~26,000 train / ~3,100 val

### Phase 1: 訓練（Day 1-3, ~20-30 GPU 小時）
**單一訓練腳本** `training/train_phase3_recall_boost.py`

```python
# 核心配置
model = YOLO("harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt")
model.train(
    data="datasets/yolo_harmony_v2_phase8_cleaned_v2/harmony_phase8_cleaned_v2.yaml",
    imgsz=1280,
    batch=6,
    nbs=64,
    device="0",
    workers=12,
    amp=False,
    optimizer="AdamW",
    lr0=0.0003,
    lrf=0.01,
    warmup_epochs=5,
    cos_lr=True,
    epochs=150,
    patience=50,
    cls=1.5,
    box=7.5,
    fl_gamma=1.5,
    label_smoothing=0.05,
    copy_paste=0.15,
    mixup=0.1,
    scale=0.7,
    mosaic=1.0,
    close_mosaic=20,
    save_period=10,
    plots=True,
    project="harmony_omr_v2_phase3",
    name="recall_boost_v1",
    exist_ok=True,
)
```

### Phase 2: 評估與驗證（Day 3, ~30 分鐘）
1. 用 unified_eval.py 雙 protocol 評估
2. Per-class AP 對比，確認稀有類別改善
3. 與 baseline 比較

### Gate 標準
| 指標 | 門檻 | 當前值 |
|------|------|--------|
| deploy mAP50 | > 0.7763 | 0.7763 |
| strict mAP50 | > 0.7295 | 0.7295 |
| Recall | > 0.60 | 0.576 |
| val_cls_loss | < 0.70 | ~0.73 |

---

## 4. 磁碟預算

| 項目 | 新增磁碟 |
|------|---------|
| 清洗標註副本 | ~400 MB |
| 訓練 checkpoint | ~70 MB |
| 評估報告 | ~10 MB |
| **總計** | **~500 MB** (佔 26GB 的 2%) |

---

## 5. 風險緩解

| 風險 | 緩解 |
|------|------|
| 清洗後退步 | Gate check，保留原始資料集 |
| OOM at batch=6 | fallback batch=4, nbs=64 |
| 新超參數訓練不穩定 | save_period=10，可從 checkpoint 恢復 |
| 過度微調(catastrophic forgetting) | lr0=0.0003 極低，warmup=5 |

---

## 6. 產出文件

### 新增腳本
- `training/unified_eval.py` — 雙 protocol 統一評估
- `training/create_cleaned_dataset_v2.py` — 數據清洗 v2（含 quartets + tiny box）
- `training/train_phase3_recall_boost.py` — Phase 3 訓練腳本

### 新增資料
- `training/datasets/yolo_harmony_v2_phase8_cleaned_v2/` — 清洗 v2 資料集
- `training/reports/unified_leaderboard.json` — 統一排行榜

### 修改檔案
- `CLAUDE.md` — 更新即時狀態
