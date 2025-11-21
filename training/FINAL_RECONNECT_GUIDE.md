# 🛡️ 最終重連指南（防彈級流程）

## ✅ 已啟動的任務

**防彈級全自動化流程**
- 腳本: `BULLETPROOF_AUTO_PIPELINE.sh`
- 特點: **7 階段完整驗證**，確保每步都正確

### 流程階段：

1. ✅ **環境檢查** - GPU、資料集、虛擬環境
2. 🔥 **極致並行轉換** - 24核心全速（預計 7 分鐘）
3. 🔍 **轉換結果驗證** - 檔案數量、格式檢查
4. 🔍 **標註格式驗證** - **防止座標超界錯誤**
5. 🔍 **YAML 配置檢查** - 20 類別、路徑正確
6. 🚀 **GPU 訓練啟動** - Batch=256, Workers=32
7. 🔍 **訓練啟動驗證** - 確保訓練正常開始

---

## 📊 重連後立即檢查

### 1️⃣ 查看流程進度

```bash
cd /home/thc1006/dev/music-app/training

# 查看最新流程 log
tail -f bulletproof_pipeline_*.log

# 或查看完整 log
ls -lt bulletproof_pipeline_*.log | head -1 | awk '{print $9}' | xargs cat
```

### 2️⃣ 檢查是否有錯誤

```bash
# 如果流程失敗，錯誤會記錄在這裡
cat pipeline_errors.log
```

### 3️⃣ 查看訓練進度

```bash
# 找到訓練 log
ls -lt training_bulletproof_*.log | head -1

# 即時監控
tail -f training_bulletproof_*.log
```

### 4️⃣ 檢查 GPU 狀態

```bash
# 即時監控
watch -n 2 nvidia-smi

# 或單次查詢
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader
```

---

## 🎯 預期時間軸

| 階段 | 預計時間 | 說明 |
|------|---------|------|
| 環境檢查 | ~5 秒 | 驗證 GPU、資料集 |
| **極致並行轉換** | **~7 分鐘** | 24 核全速（比舊版快 6 倍）|
| 驗證標註格式 | ~30 秒 | 防止座標錯誤 |
| YAML 檢查 | ~5 秒 | 配置正確性 |
| 訓練啟動 | ~60 秒 | 載入模型、快取圖片 |
| **訓練執行** | **8-12 小時** | 600 epochs |

**總計**: 約 8-12.5 小時完成全部流程

---

## ✅ 成功指標

### 轉換階段成功：
```
✅ 轉換完成！耗時: XXX秒
✅ 數量驗證通過
✅ 所有標註檔案格式正確！
✅ YAML 配置正確
```

### 訓練啟動成功：
```
✅ 訓練已啟動！PID: XXXXX
✅ 訓練進程運行中
✅ 訓練正常進行！
```

### GPU 利用率指標：
- GPU 使用率: > 80%
- VRAM 使用: > 20GB (batch=256)
- 功率: > 200W
- 無 "non-normalized" 錯誤

---

## 🔧 如果遇到問題

### 情況 1: 轉換失敗

```bash
# 查看錯誤
cat pipeline_errors.log

# 查看詳細 log
cat bulletproof_pipeline_*.log | grep "❌"

# 手動重試
./BULLETPROOF_AUTO_PIPELINE.sh
```

### 情況 2: 訓練啟動失敗

```bash
# 查看訓練 log
tail -100 training_bulletproof_*.log

# 檢查是否有座標錯誤
grep "non-normalized" training_bulletproof_*.log

# 如果有錯誤，流程會自動停止並記錄
```

### 情況 3: GPU 利用率低

```bash
# 檢查訓練進程
ps aux | grep yolo12_train

# 查看訓練配置
grep -A 10 "Configuration:" training_bulletproof_*.log
```

---

## 📁 檔案位置

```
/home/thc1006/dev/music-app/training/
│
├── 🛡️ BULLETPROOF_AUTO_PIPELINE.sh     # 防彈級主腳本
├── 📜 bulletproof_pipeline_*.log       # 流程 log
├── 📜 training_bulletproof_*.log       # 訓練 log
├── 📜 pipeline_errors.log              # 錯誤 log（如果有）
│
├── 🔥 convert_deepscores_ULTRA_PARALLEL.py  # 極致並行轉換器
│
├── 📂 datasets/yolo_harmony/           # 轉換後的資料集
│   ├── train/images/ (xxx 張)
│   ├── train/labels/ (xxx 個)
│   ├── val/images/
│   ├── val/labels/
│   └── harmony_deepscores.yaml
│
└── 📂 harmony_omr_ultra/train*/weights/
    ├── best.pt  ⭐ 最佳模型
    └── last.pt     最新模型
```

---

## 🚀 快速檢查指令

建立快速檢查腳本：

```bash
cat > check_pipeline.sh << 'EOF'
#!/bin/bash
echo "========== 🛡️ 防彈級流程狀態 =========="
echo ""

# 1. 檢查流程進程
PIPELINE_PID=$(pgrep -f "BULLETPROOF_AUTO_PIPELINE.sh")
if [ -n "$PIPELINE_PID" ]; then
    echo "✅ 流程運行中 (PID: $PIPELINE_PID)"
else
    echo "⚠️  流程已完成或未啟動"
fi

# 2. 檢查訓練進程
TRAIN_PID=$(pgrep -f "yolo12_train_ultra_optimized.py")
if [ -n "$TRAIN_PID" ]; then
    echo "✅ 訓練運行中 (PID: $TRAIN_PID)"

    # GPU 狀態
    echo ""
    echo "🎮 GPU 狀態:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
else
    echo "⚠️  訓練未啟動"
fi

# 3. 檢查錯誤
if [ -f "pipeline_errors.log" ]; then
    echo ""
    echo "⚠️  發現錯誤 log:"
    cat pipeline_errors.log
fi

# 4. 最新進度
echo ""
echo "📝 最新進度 (流程 log 最後 10 行):"
ls -t bulletproof_pipeline_*.log 2>/dev/null | head -1 | xargs tail -10 2>/dev/null || echo "   Log 尚未建立"

# 5. 訓練進度（如果有）
if [ -n "$TRAIN_PID" ]; then
    echo ""
    echo "📊 訓練進度 (最後 5 行):"
    ls -t training_bulletproof_*.log 2>/dev/null | head -1 | xargs tail -5 2>/dev/null || echo "   訓練 log 尚未建立"
fi

echo ""
echo "=========================================="
EOF

chmod +x check_pipeline.sh
./check_pipeline.sh
```

---

## 💡 重要提醒

### ✅ 這次的改進：

1. **極致並行轉換** - 真正使用 24 核心（6-7 倍加速）
2. **三層 bbox 驗證** - 絕對不會有座標超界錯誤
3. **7 階段檢查點** - 任何問題立即停止並報告
4. **完整錯誤記錄** - 所有錯誤都會記錄在 `pipeline_errors.log`
5. **自動化訓練** - 轉換成功後自動啟動 GPU 訓練

### 🛡️ 防護機制：

- ❌ **座標超界** → 階段 3 會檢測並停止
- ❌ **檔案數量不符** → 階段 2 會檢測並停止
- ❌ **YAML 配置錯誤** → 階段 4 會檢測並停止
- ❌ **訓練啟動失敗** → 階段 7 會檢測並報告

---

**最後更新**: 2025-11-21 04:25
**預計完成**: 2025-11-21 12:00 - 17:00

**一鍵檢查**: `./check_pipeline.sh`

祝訓練順利！🚀
