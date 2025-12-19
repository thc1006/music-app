# Phase 9 完整執行計劃

**建立日期**: 2025-11-28
**預計執行時間**: 10-15 小時
**目標**: mAP50 從 0.644 提升至 0.70+

---

## 執行概覽

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 9 執行流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1                Step 2               Step 3          │
│  ┌─────────┐          ┌─────────┐          ┌─────────┐     │
│  │ 清理標註 │ ──────▶ │ 驗證數據 │ ──────▶ │ 訓練模型 │     │
│  └─────────┘          └─────────┘          └─────────┘     │
│  (30 分鐘)            (10 分鐘)            (6-9 小時)       │
│                                                             │
│                              │                              │
│                              ▼                              │
│                       ┌─────────┐                          │
│                       │ 評估結果 │                          │
│                       └─────────┘                          │
│                       (15 分鐘)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 一、快速開始（推薦）

### 方法 A：一鍵執行（背景運行）

```bash
cd ~/dev/music-app/training

# 1. 設置執行權限
chmod +x run_phase9_pipeline.sh
chmod +x check_phase9_progress.sh

# 2. 背景執行完整管道
nohup ./run_phase9_pipeline.sh > phase9_pipeline.log 2>&1 &

# 3. 查看執行狀態
tail -f phase9_pipeline.log

# 4. 監控訓練進度（另開終端）
watch -n 60 ./check_phase9_progress.sh
```

### 方法 B：分步執行（交互式）

```bash
cd ~/dev/music-app/training

# Step 1: 清理標註
python scripts/clean_tiny_annotations_phase9.py

# Step 2: 執行訓練
python yolo12_train_phase9.py

# Step 3: 監控進度（另開終端）
./check_phase9_progress.sh
```

---

## 二、詳細執行步驟

### Step 1: 清理無效標註（30 分鐘）

**目的**: 移除面積過小的無效標註

**執行**:
```bash
cd ~/dev/music-app/training
python scripts/clean_tiny_annotations_phase9.py
```

**驗證檢查點**:
- [ ] 輸出目錄存在: `datasets/yolo_harmony_v2_phase9_clean/`
- [ ] YAML 配置生成: `harmony_phase9_clean.yaml`
- [ ] 清理報告生成: `cleaning_report.json`
- [ ] 移除比例約 15-20%

**預期輸出**:
```
=== 清理報告 ===
總標註數 (清理前): 2,834,322
保留標註數: ~2,400,000
移除標註數: ~400,000
移除比例: ~15%
```

**如果失敗**:
1. 檢查輸入數據集是否存在
2. 檢查磁盤空間是否足夠（需要 ~5GB）
3. 查看錯誤訊息

---

### Step 2: 驗證清理後數據集（10 分鐘）

**執行**:
```bash
# 檢查文件數量
find datasets/yolo_harmony_v2_phase9_clean/train/labels -name "*.txt" | wc -l
find datasets/yolo_harmony_v2_phase9_clean/val/labels -name "*.txt" | wc -l

# 檢查 YAML 配置
cat datasets/yolo_harmony_v2_phase9_clean/harmony_phase9_clean.yaml

# 抽樣檢查標註
head -20 datasets/yolo_harmony_v2_phase9_clean/train/labels/*.txt | head -50
```

**驗證檢查點**:
- [ ] 訓練集 ~32,555 個文件
- [ ] 驗證集 ~3,617 個文件
- [ ] YAML 配置正確
- [ ] 標註格式正確（class_id x y w h）

---

### Step 3: 執行 Phase 9 訓練（6-9 小時）

**執行**:
```bash
# 交互式執行
python yolo12_train_phase9.py

# 或自動執行
python yolo12_train_phase9.py --auto
```

**訓練參數**:
| 參數 | 值 | 說明 |
|------|-----|------|
| epochs | 100 | 訓練輪數 |
| batch | 24 | 批次大小 |
| imgsz | 640 | 圖片尺寸 |
| lr0 | 0.0005 | 初始學習率 |
| patience | 30 | 早停耐心值 |

**驗證檢查點**:
- [ ] GPU 使用率 > 90%
- [ ] 記憶體使用 < 32GB
- [ ] 溫度 < 85°C
- [ ] 訓練損失持續下降

**監控命令**:
```bash
# 終端 1：查看 GPU 狀態
watch -n 5 nvidia-smi

# 終端 2：查看訓練進度
./check_phase9_progress.sh

# 終端 3：查看訓練日誌
tail -f harmony_omr_v2_phase9/clean_data_training/results.csv
```

---

### Step 4: 評估結果（15 分鐘）

**執行**:
```bash
# 查看最終結果
tail -1 harmony_omr_v2_phase9/clean_data_training/results.csv

# 查看訓練曲線
ls harmony_omr_v2_phase9/clean_data_training/results.png

# 查看各類別表現
cat harmony_omr_v2_phase9/clean_data_training/training_report.json
```

**成功標準**:
| 指標 | 目標 | 最低要求 |
|------|------|---------|
| mAP50 | 0.70+ | 0.68 |
| mAP50-95 | 0.62+ | 0.60 |
| barline_double | 0.35+ | 0.30 |

---

## 三、故障排除

### 問題 1：清理腳本出錯

**症狀**: `FileNotFoundError` 或 `PermissionError`

**解決**:
```bash
# 檢查輸入數據集
ls -la datasets/yolo_harmony_v2_phase8_final/

# 檢查磁盤空間
df -h

# 手動創建輸出目錄
mkdir -p datasets/yolo_harmony_v2_phase9_clean/{train,val}/{images,labels}
```

### 問題 2：訓練 OOM（記憶體不足）

**症狀**: `CUDA out of memory`

**解決**:
```bash
# 編輯訓練腳本，降低 batch size
# 在 yolo12_train_phase9.py 中修改:
# 'batch': 24 → 'batch': 16

# 或清理 GPU 記憶體後重試
nvidia-smi --gpu-reset
```

### 問題 3：訓練不收斂

**症狀**: 損失值不下降或震盪

**解決**:
```bash
# 降低學習率
# 在 yolo12_train_phase9.py 中修改:
# 'lr0': 0.0005 → 'lr0': 0.0002
```

### 問題 4：訓練中斷

**症狀**: 訓練意外停止

**解決**:
```bash
# 從最近的檢查點恢復
python -c "
from ultralytics import YOLO
model = YOLO('harmony_omr_v2_phase9/clean_data_training/weights/last.pt')
model.train(resume=True)
"
```

---

## 四、後續步驟（Phase 9 完成後）

### 如果達標（mAP50 ≥ 0.70）

```
┌─────────────────────────────────────────┐
│ 進入部署測試階段                          │
├─────────────────────────────────────────┤
│ 1. TFLite 量化                          │
│ 2. Android 整合測試                      │
│ 3. 真實樂譜測試                          │
└─────────────────────────────────────────┘
```

### 如果未達標（mAP50 < 0.68）

根據結果選擇後續方案：

| 情況 | 建議方案 | 預計時間 |
|------|---------|---------|
| mAP50 0.66-0.68 | 高解析度訓練 (768px) | 12-18h |
| mAP50 0.64-0.66 | 修復 DoReMi 數據 | 1-2 天 |
| mAP50 < 0.64 | 深度分析 + 重新規劃 | 2-3 天 |

---

## 五、文件清單

### 已創建的腳本

| 文件 | 用途 |
|------|------|
| `scripts/clean_tiny_annotations_phase9.py` | 清理無效標註 |
| `yolo12_train_phase9.py` | Phase 9 訓練腳本 |
| `run_phase9_pipeline.sh` | 自動化執行管道 |
| `check_phase9_progress.sh` | 訓練進度監控 |

### 輸出文件

| 文件 | 位置 |
|------|------|
| 清理後數據集 | `datasets/yolo_harmony_v2_phase9_clean/` |
| 清理報告 | `datasets/yolo_harmony_v2_phase9_clean/cleaning_report.json` |
| 訓練結果 | `harmony_omr_v2_phase9/clean_data_training/` |
| 最佳權重 | `harmony_omr_v2_phase9/clean_data_training/weights/best.pt` |
| 日誌文件 | `logs/phase9/` |

---

## 六、時間估算

| 步驟 | 預計時間 | 累計時間 |
|------|---------|---------|
| Step 1: 清理標註 | 30 分鐘 | 30 分鐘 |
| Step 2: 驗證數據 | 10 分鐘 | 40 分鐘 |
| Step 3: 訓練 | 6-9 小時 | 7-10 小時 |
| Step 4: 評估 | 15 分鐘 | 7-10 小時 |

**總計**: 約 7-10 小時

---

## 立即開始

```bash
# 現在執行！
cd ~/dev/music-app/training
chmod +x run_phase9_pipeline.sh check_phase9_progress.sh
nohup ./run_phase9_pipeline.sh > phase9_pipeline.log 2>&1 &
echo "Phase 9 已在背景啟動！使用 'tail -f phase9_pipeline.log' 查看進度"
```

---

*文檔生成時間: 2025-11-28*
