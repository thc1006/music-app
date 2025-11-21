# 🔥 重連後操作指南

## ✅ 已啟動的後台任務

### 主任務：全自動化預處理 + 訓練
**腳本**: `AUTO_PROCESS_ALL_DATASETS.sh`
**功能**:
1. ✅ 修復並重新轉換 DeepScoresV2（已修復 bbox 歸一化bug）
2. 🔄 嘗試下載並轉換 MUSCIMA++（如果圖片可用）
3. 🔄 嘗試轉換 PRIMUS（如果轉換器存在）
4. 🔗 合併所有可用資料集
5. 🚀 啟動 Ultra-Optimized 訓練（batch=256, workers=32）

---

## 📊 重連後立即檢查

### 1️⃣ 查看主腳本執行狀態

```bash
cd /home/thc1006/dev/music-app/training

# 查看主 log（所有階段進度）
tail -100 auto_process_all.log

# 或即時監控
tail -f auto_process_all.log
```

### 2️⃣ 查看訓練進度

```bash
# 找到最新訓練 log
ls -lt training_final_*.log | head -1

# 查看訓練進度
tail -50 training_final_*.log

# 或即時監控
tail -f $(ls -t training_final_*.log | head -1)

# 搜尋關鍵字
grep -E "Epoch|mAP|loss" $(ls -t training_final_*.log | head -1) | tail -30
```

### 3️⃣ 檢查 GPU 狀態

```bash
# 即時監控（推薦）
watch -n 2 nvidia-smi

# 或單次查詢
nvidia-smi

# 簡化版本
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader
```

### 4️⃣ 檢查訓練進程

```bash
# 查看所有 YOLO 相關進程
ps aux | grep yolo12_train

# 或使用 pgrep
pgrep -af yolo12_train
```

---

## 📈 預期結果

### 成功指標：

✅ **GPU 利用率 > 80%**
✅ **VRAM 使用 > 20GB**（batch=256 應該用滿）
✅ **無 "non-normalized" 錯誤**
✅ **Epoch 正常遞增**
✅ **mAP 逐漸上升**

### 檢查命令：

```bash
# 一鍵檢查腳本
cat << 'EOF' > check_status.sh
#!/bin/bash
echo "========== 🔥 YOLO12 訓練狀態 =========="
echo ""
echo "📊 GPU 狀態："
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader
echo ""
echo "🔄 訓練進程："
pgrep -af yolo12_train || echo "   ❌ 無運行中的訓練"
echo ""
echo "📝 最新訓練輸出（最後 15 行）："
ls -t training_final_*.log 2>/dev/null | head -1 | xargs tail -15 2>/dev/null || echo "   ⚠️  找不到 log"
echo ""
echo "📈 資料集統計："
echo "   Train: $(find datasets/yolo_harmony*/train/images -name '*.png' 2>/dev/null | wc -l) 張"
echo "   Val:   $(find datasets/yolo_harmony*/val/images -name '*.png' 2>/dev/null | wc -l) 張"
echo "   Test:  $(find datasets/yolo_harmony*/test/images -name '*.png' 2>/dev/null | wc -l) 張"
echo ""
echo "=========================================="
EOF
chmod +x check_status.sh
./check_status.sh
```

---

## 🛠️ 常見問題處理

### 問題 1: 訓練進程已停止

```bash
# 查看錯誤
tail -100 $(ls -t training_final_*.log | head -1)

# 手動重新啟動
source venv_yolo12/bin/activate
nohup python yolo12_train_ultra_optimized.py \
    --data datasets/yolo_harmony/harmony_deepscores.yaml \
    > training_manual_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 問題 2: 還有 bbox 錯誤

```bash
# 檢查是否還有無效座標
python3 << 'PYEOF'
from pathlib import Path
label_dir = Path("datasets/yolo_harmony/train/labels")
invalid = 0
for f in label_dir.glob("*.txt"):
    with open(f) as fp:
        for line in fp:
            coords = [float(x) for x in line.split()[1:5]]
            if any(c < 0 or c > 1 for c in coords):
                print(f"{f.name}: {coords}")
                invalid += 1
                if invalid >= 10:
                    break
    if invalid >= 10:
        break
print(f"\nTotal invalid: {invalid}")
PYEOF

# 如果還有錯誤，重新執行修復
./AUTO_PROCESS_ALL_DATASETS.sh
```

### 問題 3: GPU 利用率還是很低

可能原因：
1. **DataLoader 瓶頸** - 增加 workers
2. **Batch 太小** - 目前已是 256，應該足夠
3. **資料快取問題** - 確認 `cache='ram'` 有效

解決方法：
```bash
# 檢查系統 RAM 使用
free -h

# 檢查 DataLoader workers
grep "workers" yolo12_train_ultra_optimized.py

# 如果需要，手動調整並重啟
```

---

## 📁 重要檔案位置

```
/home/thc1006/dev/music-app/training/
│
├── 📜 AUTO_PROCESS_ALL_DATASETS.sh    # 主自動化腳本
├── 📜 auto_process_all.log           # 主腳本 log
├── 📜 training_final_*.log           # 訓練 log（最重要）
├── 📜 RECONNECT_INSTRUCTIONS.md      # 本檔案
│
├── 📂 datasets/
│   ├── yolo_harmony/                 # DeepScoresV2（修復後）
│   ├── yolo_harmony_muscima/         # MUSCIMA++（如果成功）
│   ├── yolo_harmony_primus/          # PRIMUS（如果成功）
│   └── yolo_harmony_combined/        # 合併版本（最終訓練用）
│
└── 📂 harmony_omr_ultra/
    └── train*/weights/
        ├── best.pt                    # 最佳模型 ⭐
        └── last.pt                    # 最新模型
```

---

## 🎯 預期時間軸

| 階段 | 預計時間 | 說明 |
|------|---------|------|
| DeepScores 轉換 | ~5 分鐘 | 24 核並行處理 |
| MUSCIMA++ 處理 | ~10 分鐘 | 下載 + 轉換（如果可用）|
| PRIMUS 處理 | ~15 分鐘 | Semantic 轉換（複雜）|
| 資料集合併 | ~2 分鐘 | 檔案複製 |
| 訓練 (600 epochs) | **8-12 小時** | 視 GPU 利用率 |

**總計**: 約 10-15 小時完成所有流程

---

## ⏸️ 緊急停止

```bash
# 停止所有訓練
pkill -f yolo12_train

# 停止主腳本（如果還在運行）
pkill -f AUTO_PROCESS_ALL_DATASETS

# 檢查是否都停止了
ps aux | grep -E "yolo12|AUTO_PROCESS"
```

---

## 📧 最後更新

**時間**: 2025-11-21 04:20
**狀態**: 全自動化腳本已啟動
**預計完成**: 2025-11-21 12:00 - 16:00

**關鍵指令**:
```bash
tail -f auto_process_all.log          # 查看主進度
tail -f $(ls -t training_final_*.log | head -1)  # 查看訓練
watch -n 2 nvidia-smi                  # 監控 GPU
```

祝訓練順利！🚀
