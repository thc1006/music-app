# YOLO26 OMR 訓練速查表 🚀

> **一頁式參考** | 2026-02-13 | 詳細文檔見 `docs/` 目錄

## 📊 當前狀況速覽

| 指標 | 值 | 狀態 |
|-----|-----|------|
| 最佳 mAP50 | 0.28241 | ⚠️ 低（目標 ≥0.50） |
| 最終 Recall | 0.22723 | ⚠️ 低 |
| CUDA OOM | 1 次/20 epochs | ✅ 可接受 |
| Duplicate Labels | 有（已自動移除） | ⚠️ 需檢查 |

## 🎯 推薦配置（直接複製使用）

```yaml
# 平衡進取型 - 100 Epochs
epochs: 100
batch: -1         # 自動 60% GPU 記憶體
imgsz: 1280
amp: true         # ⚠️ 監控 NaN
lr0: 0.0005
optimizer: AdamW
workers: 8        # 原 20 → 8
cache: ram        # 若 dataset < 10GB
mosaic: 1.0
close_mosaic: 10
copy_paste: 0.3
degrees: 0        # OMR: 不旋轉
perspective: 0
```

## ⚡ 一鍵啟動（複製貼上）

```bash
cd /home/thc1006/dev/music-app

nohup python3 -m ultralytics.engine.train \
    model=yolo26s.pt \
    data=/home/thc1006/.copilot/session-state/bc083904-d73e-4b8a-9f91-9f4b9271af0f/files/harmony_phase8_subset_strat20_seed20260213.yaml \
    epochs=100 batch=-1 imgsz=1280 workers=8 \
    amp=true lr0=0.0005 optimizer=AdamW \
    mosaic=1.0 close_mosaic=10 copy_paste=0.3 \
    degrees=0 perspective=0 cache=ram \
    project=harmony_omr_v2_yolo26 name=yolo26s_100ep \
    > training/logs/yolo26s_100ep.log 2>&1 &

echo "Training PID: $!"
```

## 🔍 實時監控

```bash
# Terminal 1: 監控進度
watch -n 30 "tail -50 training/logs/yolo26s_100ep.log | grep -E '(Epoch|mAP)'"

# Terminal 2: 檢查問題
watch -n 60 "grep -E 'NaN|Inf|OOM' training/logs/yolo26s_100ep.log | tail -5"

# 查看最佳 mAP
tail -20 runs/detect/harmony_omr_v2_yolo26/yolo26s_100ep/results.csv | \
    awk -F',' 'NR>1 {if($8>max){max=$8;ep=$1}} END {printf "Best: %.4f @ Epoch %d\n", max, ep}'
```

## 🚨 常見問題速查

### Loss 出現 NaN（< 20 epochs）
```bash
# 1. 停止訓練（找到 PID）
ps aux | grep ultralytics
# 然後用 kill <找到的PID> 停止

# 2. 修改：amp=true → false, lr0=0.0005 → 0.0003
# 3. 重啟
```

### CUDA OOM 頻繁（> 5 次）
```bash
# 調整：batch=-1 → 6, workers=8 → 4, cache=ram → disk
```

### mAP 停滯（連續 20 epochs < 1% 提升）
```bash
# 選項 1: 提早停止（節省時間）
# 選項 2: 延長至 150 epochs（觀察是否晚期提升）
# 選項 3: 檢查數據品質
```

## 📈 關鍵檢查點

| Epoch | 檢查項目 | 通過標準 | 失敗處置 |
|-------|---------|---------|---------|
| 10 | Loss 穩定性 | 無 NaN/Inf | 關閉 AMP |
| 30 | mAP 進展 | ≥ 0.40 | 延長至 150 |
| 50 | mAP 目標 | ≥ 0.50 | 繼續 |
| 80 | 平穩判斷 | 仍在提升 | 否則停止 |

## 🎓 三個方案快速選擇

| 情況 | 方案 | Batch | AMP | LR | Workers | 時間 |
|------|------|-------|-----|----|---------|----|
| 首次訓練 | 平衡 | -1 | true | 0.0005 | 8 | 12-16h |
| 曾經失敗 | 保守 | 6 | false | 0.0003 | 6 | 18-24h |
| 追求極限 | 激進 | 16 | true | 0.001 | 12 | 24-36h |

## 🧪 訓練前檢查表（30秒）

```bash
# ✅ GPU 可用
nvidia-smi

# ✅ 數據存在
ls /home/thc1006/.copilot/session-state/bc083904-d73e-4b8a-9f91-9f4b9271af0f/files/harmony_phase8_subset_strat20_seed20260213.yaml

# ✅ 磁碟空間（需 > 50GB）
df -h /home/thc1006/dev/music-app

# ✅ Python 環境
python3 -c "import ultralytics; print(ultralytics.__version__)"
```

## 📦 訓練完成後（5分鐘分析）

```bash
cd /home/thc1006/dev/music-app

# 1. 查看最佳模型
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('runs/detect/harmony_omr_v2_yolo26/yolo26s_100ep/results.csv', skipinitialspace=True)
best = df.loc[df['metrics/mAP50(B)'].idxmax()]
print(f"Best Epoch: {int(best['epoch'])}")
print(f"mAP50: {best['metrics/mAP50(B)']:.4f}")
print(f"Recall: {best['metrics/recall(B)']:.4f}")
print(f"Precision: {best['metrics/precision(B)']:.4f}")
EOF

# 2. 驗證測試
yolo detect val \
    model=runs/detect/harmony_omr_v2_yolo26/yolo26s_100ep/weights/best.pt \
    data=<test_yaml> imgsz=1280

# 3. 導出 TFLite
yolo export \
    model=runs/detect/harmony_omr_v2_yolo26/yolo26s_100ep/weights/best.pt \
    format=tflite imgsz=1280
```

## 🔗 完整文檔快速鏈接

| 文檔 | 用途 | 路徑 |
|------|------|------|
| 🎯 快速決策 | 5分鐘上手 | `docs/QUICK_DECISION_GUIDE.md` |
| 📊 完整報告 | 深度技術細節 | `docs/2026-02_YOLO_OMR_Training_Deep_Research_Report.md` |
| 📋 執行摘要 | 高層概覽 | `docs/2026-02_Research_Summary_Executive.md` |
| 📂 專案現況 | 基線與數據 | `LATEST_STATUS_AND_DATASET_SOURCES.md` |

## 💡 專家小技巧

1. **首次訓練**: 用 `batch=8`（不用 -1），觀察 GPU 記憶體後再調整
2. **AMP 監控**: 前 20 epochs 每 5 epochs 檢查一次 loss
3. **提早停止**: mAP 平穩 20 epochs 就可停，不必等 100
4. **數據快取**: 10GB 以下用 `ram`，否則用 `disk`
5. **弱類別**: 若 confusion matrix 顯示某類完全無法識別，需專項增強

## 📞 需要幫助？

- **NaN/OOM**: 見「常見問題速查」
- **mAP 太低**: 檢查數據品質 + 類別分佈
- **訓練太慢**: 調高 `batch` + 啟用 `cache=ram`
- **其他問題**: 回報當前 Epoch、mAP、Loss 值

---

**最後更新**: 2026-02-13 | **版本**: 1.0  
**預期成果**: mAP50 從 0.28 → 0.55（96% 提升）
