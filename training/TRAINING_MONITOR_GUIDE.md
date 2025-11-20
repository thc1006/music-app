# YOLO12 訓練監控與恢復指南

## 當前訓練狀態

**訓練配置：**
- 模型：YOLO12s (9.27M 參數)
- 類別數：35 類（Harmony OMR V2 Ultimate）
- 總 Epochs：600
- Batch Size：24
- 資料集：1,157 訓練圖像，205 驗證圖像（1.42M 標註實例）

**預估時間：**
- 每個 epoch：約 6-7 分鐘
- 總訓練時間：約 68 小時（2.9 天）
- 預計完成：2025-11-24 02:32

---

## 訓練監控命令

### 1. 檢查訓練進程狀態

```bash
# 查看訓練進程是否運行
ps aux | grep "yolo12_train_ULTIMATE" | grep -v grep

# 查看詳細進程資訊（PID, 運行時間, CPU/記憶體使用率）
ps -p 182830 -o pid,etime,pcpu,pmem,cmd 2>/dev/null || echo "訓練進程已停止"
```

### 2. 監控訓練日誌

```bash
# 查看最新訓練輸出（最後 30 行）
tail -30 training_ultimate_35classes_batch24.log

# 實時監控訓練進度
tail -f training_ultimate_35classes_batch24.log

# 查看當前 epoch 和 batch 進度
tail -1 training_ultimate_35classes_batch24.log

# 檢查是否有錯誤（排除 OOM 警告）
grep -i "error\|failed\|exception" training_ultimate_35classes_batch24.log | grep -v "OutOfMemoryError"
```

### 3. 監控 GPU 狀態

```bash
# 查看 GPU 使用情況
nvidia-smi

# 持續監控 GPU（每 2 秒更新）
watch -n 2 nvidia-smi

# 查看詳細 GPU 資訊
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv
```

### 4. 查看訓練進度統計

```bash
# 查看已保存的權重檔案
ls -lht harmony_omr_v2_ultimate/train2/weights/

# 查看訓練指標記錄
cat harmony_omr_v2_ultimate/train2/results.csv

# 統計已完成的 epoch 數
grep -c "100%" training_ultimate_35classes_batch24.log
```

---

## Checkpoint 機制

### 自動保存的檔案

**位置：** `harmony_omr_v2_ultimate/train2/weights/`

1. **持續更新（每個 epoch）：**
   - `last.pt` - 最新的模型權重（54MB）
   - `best.pt` - 驗證 mAP 最高的模型（54MB）

2. **定期保存（每 20 epochs）：**
   - `epoch20.pt`, `epoch40.pt`, `epoch60.pt`, ...
   - 完整模型快照，包含所有訓練狀態

3. **訓練記錄：**
   - `results.csv` - 所有 epoch 的完整指標記錄
   - `labels.jpg` - 資料集標註分佈視覺化

### 查看 Checkpoint

```bash
# 列出所有保存的權重
find harmony_omr_v2_ultimate -name "*.pt" -type f | sort

# 查看權重檔案大小和更新時間
ls -lh harmony_omr_v2_ultimate/train2/weights/
```

---

## 斷線後恢復訓練

### 情境 1：訓練進程仍在運行

如果您重新連線後發現訓練進程仍在背景運行：

```bash
# 1. 確認進程狀態
ps aux | grep "yolo12_train_ULTIMATE" | grep -v grep

# 2. 查看最新訓練輸出
tail -30 training_ultimate_35classes_batch24.log

# 3. 繼續監控即可，無需任何操作
tail -f training_ultimate_35classes_batch24.log
```

### 情境 2：訓練進程已停止

如果訓練因故中斷（系統重啟、進程被殺等）：

#### 步驟 1：檢查最新 Checkpoint

```bash
# 查看最後保存的權重
ls -lt harmony_omr_v2_ultimate/train2/weights/ | head -5

# 查看已完成的 epoch 數
tail -1 harmony_omr_v2_ultimate/train2/results.csv
```

#### 步驟 2：從 Checkpoint 恢復訓練

```bash
# 方法 A：使用自動恢復（推薦）
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

# 修改訓練腳本啟用 resume
sed -i "s/'resume': False/'resume': True/" yolo12_train_ULTIMATE_35classes.py

# 重新啟動訓練
nohup python yolo12_train_ULTIMATE_35classes.py > training_resume_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "訓練已重新啟動！PID: $!"

# 等待 15 秒後檢查狀態
sleep 15
tail -30 training_resume_*.log
```

```bash
# 方法 B：手動指定 Checkpoint
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

# 使用 last.pt 恢復（從最後一個 epoch 繼續）
nohup python -c "
from ultralytics import YOLO
model = YOLO('harmony_omr_v2_ultimate/train2/weights/last.pt')
model.train(
    data='datasets/yolo_harmony_v2_35classes/harmony_deepscores_v2.yaml',
    epochs=600,
    batch=24,
    resume=True
)
" > training_resume_manual_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "訓練已恢復！PID: $!"
```

#### 步驟 3：驗證恢復成功

```bash
# 查看新訓練日誌
tail -f training_resume_*.log

# 確認 epoch 從正確位置繼續
grep -E "Epoch.*[0-9]+/600" training_resume_*.log | head -5
```

### 情境 3：需要從特定 Epoch 重新訓練

如果發現某個 epoch 後訓練出現問題：

```bash
# 使用特定 epoch 的權重
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

# 例如從 epoch 40 重新開始
nohup python -c "
from ultralytics import YOLO
model = YOLO('harmony_omr_v2_ultimate/train2/weights/epoch40.pt')
model.train(
    data='datasets/yolo_harmony_v2_35classes/harmony_deepscores_v2.yaml',
    epochs=600,
    batch=24,
    resume=True
)
" > training_from_epoch40_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "從 Epoch 40 重新訓練！PID: $!"
```

---

## 常見問題排查

### Q1: 訓練速度突然變慢

**可能原因：**
- TaskAlignedAssigner CPU fallback（正常現象）
- GPU 記憶體不足導致頻繁 swap

**檢查方法：**
```bash
# 查看是否有 OOM 警告
grep "OutOfMemoryError" training_ultimate_35classes_batch24.log | wc -l

# 查看 GPU 記憶體使用
nvidia-smi
```

**說明：**
- TaskAlignedAssigner OOM 警告是預期行為（因資料集有 1.42M 實例）
- 訓練會自動 fallback 到 CPU 處理該步驟，不影響最終結果
- 每個 batch 會增加 1-2 秒處理時間（總時間已包含此延遲）

### Q2: 如何檢查訓練是否卡住

```bash
# 查看日誌檔案是否還在更新
ls -lh training_ultimate_35classes_batch24.log

# 查看最後更新時間
stat training_ultimate_35classes_batch24.log | grep Modify

# 如果超過 15 分鐘未更新，可能卡住了
```

**解決方法：**
```bash
# 停止當前訓練
pkill -f "yolo12_train_ULTIMATE"

# 從 last.pt 恢復（參考「情境 2」）
```

### Q3: 如何查看當前最佳模型表現

```bash
# 查看 results.csv 最後一行
tail -1 harmony_omr_v2_ultimate/train2/results.csv

# 或用 Python 解析
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('harmony_omr_v2_ultimate/train2/results.csv')
print("\n最新訓練指標：")
print(df.tail(1)[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'train/box_loss', 'train/cls_loss']])
print("\n最佳 mAP50：")
best_idx = df['metrics/mAP50(B)'].idxmax()
print(df.loc[best_idx][['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']])
EOF
```

### Q4: 如何停止訓練

```bash
# 優雅停止（等待當前 epoch 完成）
# YOLO12 不支援優雅停止，建議等到 save_period (20 epochs) 的倍數時再停止

# 強制停止（立即終止）
pkill -9 -f "yolo12_train_ULTIMATE"

# 確認已停止
ps aux | grep "yolo12_train_ULTIMATE" | grep -v grep
```

**注意：** 強制停止可能損失當前 epoch 的進度，但 last.pt 會保存到上一個完成的 epoch。

---

## 訓練完成後的操作

### 1. 驗證訓練結果

```bash
# 查看最終 results.csv
tail -10 harmony_omr_v2_ultimate/train2/results.csv

# 查看最佳模型
ls -lh harmony_omr_v2_ultimate/train2/weights/best.pt
```

### 2. 模型驗證

```bash
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

# 使用 best.pt 進行驗證
python3 << 'EOF'
from ultralytics import YOLO

model = YOLO('harmony_omr_v2_ultimate/train2/weights/best.pt')
results = model.val(data='datasets/yolo_harmony_v2_35classes/harmony_deepscores_v2.yaml')

print("\n最終驗證結果：")
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
EOF
```

### 3. 匯出模型（準備部署）

```bash
# 匯出為 TFLite（用於 Android）
python3 << 'EOF'
from ultralytics import YOLO

model = YOLO('harmony_omr_v2_ultimate/train2/weights/best.pt')

# 匯出 FP16
model.export(format='tflite', half=True)

# 匯出 INT8（需要校正資料）
model.export(format='tflite', int8=True, data='datasets/yolo_harmony_v2_35classes/harmony_deepscores_v2.yaml')
EOF
```

---

## 監控腳本（快速檢查）

建立一個快速監控腳本：

```bash
cat > check_training.sh << 'EOF'
#!/bin/bash
echo "========================================="
echo "YOLO12 訓練狀態檢查"
echo "========================================="

# 檢查進程
if ps aux | grep "yolo12_train_ULTIMATE" | grep -v grep > /dev/null; then
    echo "✅ 訓練進程：運行中"
    ps aux | grep "yolo12_train_ULTIMATE" | grep -v grep | awk '{print "   PID: "$2", 運行時間: "$10", CPU: "$3"%"}'
else
    echo "❌ 訓練進程：已停止"
fi

# 檢查 GPU
echo ""
echo "🎮 GPU 狀態："
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "   VRAM: %s / %s, 使用率: %s, 功耗: %s, 溫度: %s\n", $1, $2, $3, $4, $5}'

# 當前進度
echo ""
echo "📊 訓練進度："
tail -1 training_ultimate_35classes_batch24.log | grep -oP '\d+/600' | head -1 | xargs -I {} echo "   Epoch: {}"

# 最新權重
echo ""
echo "💾 最新權重："
ls -lht harmony_omr_v2_ultimate/train2/weights/ 2>/dev/null | head -3 | tail -2 | \
    awk '{print "   "$9" - "$5" ("$6" "$7" "$8")"}'

echo "========================================="
EOF

chmod +x check_training.sh
```

使用方法：
```bash
# 執行快速檢查
./check_training.sh

# 或設定每 5 分鐘自動檢查
watch -n 300 ./check_training.sh
```

---

## 重要提醒

1. **訓練時間長達 2.9 天，請確保：**
   - 電腦不會進入休眠
   - 電源供應穩定
   - 散熱正常

2. **每 20 個 epoch 會自動保存 checkpoint**
   - 即使中斷，最多損失 20 epochs 進度
   - `last.pt` 每個 epoch 都會更新

3. **TaskAlignedAssigner OOM 是正常現象**
   - 不影響訓練
   - 已自動 CPU fallback
   - 已計入預估時間

4. **建議定期檢查（每天 1-2 次）：**
   ```bash
   ./check_training.sh
   ```

5. **訓練日誌保留：**
   - 訓練完成後保留所有日誌以供分析
   - `results.csv` 包含完整訓練曲線

---

最後更新：2025-11-21
訓練開始時間：2025-11-21 05:40
預計完成時間：2025-11-24 02:32
