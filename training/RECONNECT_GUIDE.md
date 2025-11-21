# 🔥 斷線重連指南（YOLO12 訓練持續進行中）

## 1️⃣ 查看訓練進度

```bash
cd /home/thc1006/dev/music-app/training

# 查看訓練 log（實時更新）
tail -f training_ultra.log

# 查看最近 50 行
tail -50 training_ultra.log

# 搜尋關鍵字（Epoch、GPU 使用率）
grep -E "Epoch|GPU_mem|mAP" training_ultra.log | tail -20
```

## 2️⃣ 檢查 GPU 狀態

```bash
# 即時 GPU 監控（每 2 秒更新）
watch -n 2 nvidia-smi

# 或單次查詢
nvidia-smi
```

## 3️⃣ 檢查訓練進程是否還在運行

```bash
# 方法 1：查看進程
ps aux | grep yolo12_train_ultra_optimized.py

# 方法 2：查看 PID 117213 是否存在
ps -p 117213 -o pid,ppid,cmd,%cpu,%mem,etime

# 方法 3：查看訓練輸出目錄
ls -lht harmony_omr_ultra/train*/weights/
```

## 4️⃣ 如果訓練意外停止

```bash
# 重新啟動訓練（從上次 checkpoint 恢復）
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

# 使用 resume 參數從最佳 checkpoint 繼續
nohup python yolo12_train_ultra_optimized.py \
    --data datasets/yolo_harmony/harmony_deepscores.yaml \
    --resume harmony_omr_ultra/train/weights/last.pt \
    > training_ultra_resume.log 2>&1 &

echo "Resumed training PID: $!"
```

## 5️⃣ 查看訓練結果

```bash
# 最佳模型位置
ls -lh harmony_omr_ultra/train*/weights/best.pt

# 查看訓練圖表
ls harmony_omr_ultra/train*/results.png

# 查看最終 metrics
cat harmony_omr_ultra/train*/results.csv | tail -20
```

## 6️⃣ 並行任務狀態

### 當前執行中的任務：
- **訓練主任務**：PID 117213 (nohup)
  - Log: `training_ultra.log`
  - 配置：Batch=256, Workers=32, AMP=True, cuDNN Benchmark

- **PRIMUS 轉換**（待啟動）
- **MUSCIMA++ 轉換**（待啟動）

## 7️⃣ 快速檢查腳本

建立一個快速檢查腳本：

```bash
#!/bin/bash
echo "========== YOLO12 訓練狀態 =========="
echo ""
echo "1. 訓練進程："
ps aux | grep yolo12_train_ultra_optimized.py | grep -v grep || echo "   ❌ 訓練進程已停止"
echo ""
echo "2. GPU 狀態："
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
echo ""
echo "3. 最近訓練輸出（最後 10 行）："
tail -10 training_ultra.log
echo ""
echo "4. 訓練模型檔案："
ls -lht harmony_omr_ultra/train*/weights/*.pt 2>/dev/null | head -5
echo ""
echo "=========================================="
```

儲存為 `check_training.sh` 然後執行：
```bash
chmod +x check_training.sh
./check_training.sh
```

## 8️⃣ 緊急停止訓練

```bash
# 找到 PID
ps aux | grep yolo12_train_ultra_optimized.py

# 優雅停止（建議）
kill -SIGINT <PID>

# 強制停止（最後手段）
kill -9 <PID>
```

---

**最後更新**：2025-11-21 04:12
**訓練 PID**：117213
**預計完成時間**：約 8-12 小時（600 epochs，視 GPU 利用率）
