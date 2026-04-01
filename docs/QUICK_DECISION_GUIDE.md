# 快速決策指南：YOLO26 OMR 訓練（2026-02-13）

> **TL;DR**: 20-epoch smoke test 成功但效能偏低（mAP50=0.28）。建議立即執行「平衡進取型」100-epoch 訓練，預期 mAP50 達 0.50-0.60。若失敗則回退「保守穩定型」。

---

## 🚦 立即決策矩陣

| 情境 | 建議方案 | 預期 mAP50 | 風險 | 時間成本 |
|------|----------|-----------|------|---------|
| **首次正式訓練** | 方案二（平衡） | 0.50-0.60 | 中 | 12-16h |
| **已有多次失敗** | 方案一（保守） | 0.45-0.50 | 低 | 18-24h |
| **追求 SOTA** | 方案三（激進） | 0.65+ | 高 | 24-36h |
| **時間緊迫** | 方案二 + Early Stop | 0.45-0.55 | 中 | 8-12h |

---

## ⚡ 5 分鐘啟動指令（推薦：方案二）

```bash
# 1. 進入專案目錄
cd /home/thc1006/dev/music-app

# 2. 創建配置（已包含在報告中）
# 詳見 docs/2026-02_YOLO_OMR_Training_Deep_Research_Report.md > Section C

# 3. 一鍵啟動訓練
cat > run_balanced_training.sh << 'EOF'
#!/bin/bash
set -e

# 環境變數
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 訓練命令
nohup python3 -m ultralytics.engine.train \
    model=yolo26s.pt \
    data=/home/thc1006/.copilot/session-state/bc083904-d73e-4b8a-9f91-9f4b9271af0f/files/harmony_phase8_subset_strat20_seed20260213.yaml \
    epochs=100 \
    batch=-1 \
    imgsz=1280 \
    workers=8 \
    amp=true \
    lr0=0.0005 \
    optimizer=AdamW \
    mosaic=1.0 \
    close_mosaic=10 \
    copy_paste=0.3 \
    degrees=0.0 \
    perspective=0.0 \
    cache=ram \
    project=harmony_omr_v2_yolo26 \
    name=yolo26s_balanced_100ep_run1 \
    > training/logs/balanced_100ep.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f training/logs/balanced_100ep.log"
EOF

chmod +x run_balanced_training.sh
./run_balanced_training.sh

# 4. 實時監控（另一 terminal）
watch -n 30 "tail -50 training/logs/balanced_100ep.log | grep -E '(Epoch|mAP50)'"
```

---

## 🎯 關鍵檢查點時間表

| 時間點 | 檢查項目 | 判斷標準 | 失敗處置 |
|--------|---------|---------|---------|
| **Epoch 10** | Loss 穩定性 | 無 NaN/Inf | 關閉 AMP，重啟 |
| **Epoch 30** | mAP 進展 | ≥ 0.40 | 延長至 150 epochs |
| **Epoch 50** | mAP 目標 | ≥ 0.50 | 繼續訓練 |
| **Epoch 80** | 平穩判斷 | 連續 10 epochs < 1% 提升 | 提早停止 |

### 快速檢查指令

```bash
# 檢查當前最佳 mAP
tail -20 runs/detect/harmony_omr_v2_yolo26/yolo26s_balanced_100ep_run1/results.csv | \
    awk -F',' 'NR>1 {if($8>max){max=$8;ep=$1}} END {printf "Best mAP50: %.4f @ Epoch %d\n", max, ep}'

# 檢查問題
grep -E "NaN|Inf|OOM" training/logs/balanced_100ep.log | tail -10

# 檢查進度
tail -1 training/logs/balanced_100ep.log | grep -oP "Epoch \d+/100"
```

---

## 🛑 緊急停止與補救

### 情況 1: Loss 出現 NaN（訓練前 20 epochs）
```bash
# 立即停止
pkill -f "ultralytics.engine.train"

# 修改配置
# amp: true → false
# lr0: 0.0005 → 0.0003
# batch: -1 → 6

# 重啟訓練
./run_balanced_training.sh
```

### 情況 2: CUDA OOM 頻繁（> 5 次/100 epochs）
```bash
# 調整 batch size
# batch: -1 → 6 或 4
# workers: 8 → 4
# cache: ram → disk
```

### 情況 3: mAP 在 50 epochs 仍 < 0.40
```bash
# 可能原因檢查
# 1. 數據品質問題（duplicate labels, annotation errors）
# 2. 類別不平衡未解決（需增強 copy_paste, cls loss weight）
# 3. 增強策略不當（檢查 mosaic, mixup 是否過度）

# 建議：完成當前訓練（觀察完整曲線），再調整策略
```

---

## 📊 訓練完成後立即驗證

```bash
# 1. 生成完整報告
python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('runs/detect/harmony_omr_v2_yolo26/yolo26s_balanced_100ep_run1/results.csv', skipinitialspace=True)

# 找出最佳模型
best_idx = results['metrics/mAP50(B)'].idxmax()
print(f"=== Best Model @ Epoch {results.loc[best_idx, 'epoch']} ===")
print(f"mAP50: {results.loc[best_idx, 'metrics/mAP50(B)']:.4f}")
print(f"Recall: {results.loc[best_idx, 'metrics/recall(B)']:.4f}")
print(f"Precision: {results.loc[best_idx, 'metrics/precision(B)']:.4f}")

# 繪製曲線
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
results.plot(x='epoch', y='metrics/mAP50(B)', ax=axes[0,0], title='mAP50')
results.plot(x='epoch', y='metrics/recall(B)', ax=axes[0,1], title='Recall')
results.plot(x='epoch', y='train/box_loss', ax=axes[1,0], title='Box Loss')
results.plot(x='epoch', y='train/cls_loss', ax=axes[1,1], title='Cls Loss')
plt.tight_layout()
plt.savefig('training/reports/balanced_100ep_curves.png')
print("Curves saved to training/reports/balanced_100ep_curves.png")
EOF

# 2. 在驗證集測試
yolo detect val \
    model=runs/detect/harmony_omr_v2_yolo26/yolo26s_balanced_100ep_run1/weights/best.pt \
    data=/home/thc1006/.copilot/session-state/bc083904-d73e-4b8a-9f91-9f4b9271af0f/files/harmony_phase8_subset_strat20_seed20260213.yaml \
    imgsz=1280 \
    conf=0.25 \
    iou=0.55

# 3. 檢查 confusion matrix
# 開啟 runs/detect/.../confusion_matrix.png
# 找出高混淆類別對（如 notehead_filled vs notehead_hollow）
```

---

## 🔄 下一步決策樹

```
訓練完成
  └─ mAP50 ≥ 0.60?
       ├─ YES → 🎉 部署準備
       │        ├─ 導出 TFLite (yolo export format=tflite)
       │        ├─ Android 集成測試
       │        └─ 效能基準測試
       │
       └─ NO → mAP50 在 0.50-0.60?
                ├─ YES → ✅ 可用，但需優化
                │        ├─ 弱類別專項增強
                │        ├─ 超參數微調（model.tune()）
                │        └─ 數據擴充
                │
                └─ NO → ⚠️ 需深度調查
                         ├─ 檢查數據品質（標註、分佈、重複）
                         ├─ 重新審視增強策略
                         ├─ 考慮更換架構（yolo26m？）
                         └─ 或回退至 Ultimate v5 stable
```

---

## 📋 預期時間線（方案二）

| 階段 | 任務 | 時間 | 累計 |
|------|------|------|------|
| **準備** | 配置創建 + 環境檢查 | 30 min | 0.5h |
| **訓練** | 100 epochs (RTX 5090) | 12-16h | 16.5h |
| **驗證** | 指標分析 + Confusion Matrix | 1h | 17.5h |
| **測試** | 測試集評估 + 錯誤分析 | 2h | 19.5h |
| **部署** | TFLite 導出 + Android 集成 | 4h | 23.5h |
| **總計** | - | **~24 小時** | - |

> **並行建議**: 訓練期間可同步進行數據品質檢查、文檔整理、Android 端架構準備。

---

## 💡 專家提示

1. **AMP 監控**: 前 20 epochs 密切關注 loss，出現 spike 立即關閉
2. **Batch Size**: `-1` 首次可能觸發 OOM，建議第一次手動設 `batch=8` 觀察
3. **Workers**: 若 CPU 使用率 > 90%，降至 4-6
4. **Cache**: 數據集 < 10GB 用 `ram`，否則用 `disk`
5. **弱類別**: 若 confusion matrix 顯示某些類別完全無法識別，需專項增強

---

## 📞 遇到問題？

### 常見問題速查

**Q: 訓練速度異常慢（< 1 epoch/hour）**  
A: 檢查 `workers` 是否過低（應 ≥ 4），`cache` 是否設為 False（改為 disk）

**Q: GPU 利用率低（< 50%）**  
A: 提高 `batch size` 或啟用 `cache=ram`

**Q: mAP 提升極慢（< 0.01/10 epochs）**  
A: 可能已收斂，檢查 validation loss 是否平穩，考慮提早停止

**Q: 某些類別完全無法檢測**  
A: 檢查該類別樣本數（< 100 instances 需大幅增強），考慮 `copy_paste=0.6`

---

## ✅ 最終檢查表（訓練前）

- [ ] 確認 GPU 可用（`nvidia-smi`）
- [ ] 確認數據路徑正確（檢查 .yaml 中的 path）
- [ ] 確認有足夠磁碟空間（至少 50GB for checkpoints + cache）
- [ ] 設定訓練監控（watch 或 TensorBoard）
- [ ] 備份當前最佳模型（若有）
- [ ] 記錄實驗到 MLflow 或 Notion

---

**祝訓練順利！有任何問題隨時回報。**  
**詳細技術細節請參閱**: `docs/2026-02_YOLO_OMR_Training_Deep_Research_Report.md`
