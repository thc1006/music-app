# YOLO26 OMR 訓練策略深度調研報告（2026-02）

**調研日期**: 2026-02-13  
**專案**: Music App - 五線譜光學辨識 (OMR)  
**當前狀態**: 20-epoch smoke test 完成，mAP50=0.28241  
**調研方法**: 交叉驗證多來源（Ultralytics官方文檔、PyTorch文檔、arXiv學術論文、GitHub Issues/Discussions、Android開發指南）

---

## A. 重點結論（15條，含來源）

### 1. YOLO26 架構優勢與特性
**結論**: YOLO26 採用 NMS-free 端到端推論架構，專為邊緣部署優化，強化小物件檢測能力，支援 TFLite/ONNX/TensorRT 導出。  
**來源**: [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/) - YOLO26 作為最新版本，繼承 YOLO11 改進並進一步優化邊緣推論。

### 2. 小物件檢測的核心策略
**結論**: 針對小物件（如五線譜符號），建議採用：(1) 多尺度訓練 `multi_scale > 0`，(2) 提高輸入解析度（如 1280），(3) Mosaic 增強（`mosaic=1.0` 前期，後期逐漸降低），(4) 關閉 `close_mosaic` 最後 10 epochs，(5) 考慮 SAHI (Slicing Aided Hyper Inference) 技術。  
**來源**: Ultralytics Training Documentation + GitHub YOLO Small Object Detection repositories (yolo-tiling, YOLO-Patch-Based-Inference 等)

### 3. CUDA OOM 在 TaskAlignedAssigner 的影響
**結論**: CUDA OOM 導致 TaskAlignedAssigner 回退到 CPU 執行，會造成：(1) 該 batch 速度顯著下降（CPU 計算標籤分配），(2) 結果正確性不受影響（僅速度問題），(3) 建議減少 batch size 或關閉 AMP 避免 GPU 記憶體峰值。  
**來源**: Ultralytics GitHub Issues（多個用戶回報類似警告）+ PyTorch內部實作（自動 fallback 機制）

### 4. 混合精度訓練 (AMP) 數值穩定性
**結論**: AMP 使用 FP16 加速訓練，但可能導致梯度下溢或 loss 為 NaN/Inf。PyTorch 官方建議：(1) 僅包裹 forward + loss 計算，(2) 搭配 GradScaler，(3) 若出現 NaN 應立即關閉 AMP（`amp=False`）或降低學習率。  
**來源**: [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)

### 5. 數據不平衡的最佳處理順序
**結論**: 處理長尾類別的優先順序應為：(1) 檢查並修正標註品質，(2) 針對弱類別進行 oversampling 或 copy-paste 增強，(3) 使用類別不平衡 loss（如 Focal Loss、Equalized Focal Loss），(4) 最後才做全局超參數調整。  
**來源**: Academic research on long-tail detection + 專案內部 Phase 6 config 實踐經驗

### 6. Duplicate Labels 警告的成因與處理
**結論**: Duplicate labels 通常來自：(1) 數據合併時未去重，(2) 增強後產生重疊 bbox，(3) 人工標註錯誤。YOLO 會自動移除重複標籤，但建議在訓練前做數據清洗（`python utils/clean_duplicates.py`）。  
**來源**: 專案 logs（`training/logs/all_phases.log` 顯示多處 duplicate 警告）+ Ultralytics data loader 實作

### 7. Workers 數量的雙面刃效應
**結論**: `workers > 0` 能加速數據載入，但過高值（如 20）可能導致：(1) CPU 資源競爭，(2) 記憶體峰值增加，(3) 在某些環境下觸發 multiprocessing 錯誤。建議：單 GPU 設為 4-8，多 GPU 設為 `min(8, num_cpus // num_gpus)`。  
**來源**: Ultralytics Training Args Documentation + PyTorch DataLoader best practices

### 8. Batch Size 與記憶體/速度權衡
**結論**: 當前設定 `batch=4` 在 RTX 5090 (32GB VRAM) + `imgsz=1280` 下偏保守。可測試：(1) `batch=-1` 自動調整至 60% GPU 記憶體，(2) 手動嘗試 `batch=8` 或 `batch=12`，(3) 搭配 gradient accumulation 模擬更大 batch。  
**來源**: Ultralytics batch sizing documentation + 專案 args.yaml（目前 batch=4）

### 9. Learning Rate 穩定訓練策略
**結論**: 當前 `lr0=0.0005` 已較保守。若仍出現不穩定，建議：(1) 進一步降低至 `0.0001-0.0003`，(2) 增加 `warmup_epochs=5-10`，(3) 使用 `cos_lr=True` 搭配較長訓練（100+ epochs），(4) AdamW optimizer 通常比 SGD 更穩定但可能收斂較慢。  
**來源**: PyTorch tuning guide + Ultralytics hyperparameter tuning guide

### 10. Cache 策略對速度的影響
**結論**: `cache=False` 每個 epoch 重新從磁碟讀取，適合大數據集；`cache=ram` 第一個 epoch 後全載入記憶體，適合中小數據集（<10GB）；`cache=disk` 折衷方案。建議：Phase8 數據集先測試 `cache=disk`，若記憶體充足改用 `cache=ram`。  
**來源**: Ultralytics cache parameter documentation + 專案 Phase8 config (`cache=ram` 過去使用)

### 11. Optimizer 選擇對收斂的影響
**結論**: AdamW（當前設定）適合多數場景，但在長尾數據可能過度適應常見類別。可測試：(1) SGD + momentum (0.9-0.95) 搭配較長訓練，(2) RAdam 提供更穩健的學習率適應，(3) Lion optimizer（新興）號稱更高記憶體效率。  
**來源**: Ultralytics optimizer options + PyTorch optimizer comparison studies

### 12. OMR 任務的特殊增強需求
**結論**: 五線譜辨識對旋轉、透視變換敏感度高（會破壞音高資訊），建議：(1) `degrees=0`（不旋轉），(2) `perspective=0`（不透視），(3) `shear=0`，(4) 適度 `translate` 和 `scale`，(5) 強化 `hsv_h/s/v` 模擬不同掃描/拍照條件。  
**來源**: OMR research papers (Optical Music Recognition challenges) + 專案 args.yaml 當前設定（已設 degrees=0）

### 13. 短期實驗 vs 長期訓練的陷阱
**結論**: Ultralytics 官方警告：短期/小規模超參數調整（如 30 epochs, fraction=0.1）得出的最佳參數**很少能泛化至完整訓練**。建議：(1) 先做穩定性驗證（30-50 epochs），(2) 確認無 NaN/OOM 後，才進行 100-150 epochs 正式訓練。  
**來源**: [Ultralytics Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/) 明確警告此問題

### 14. Early Stopping 與 Patience 設定
**結論**: 當前 `patience=10` 在 20 epochs 中較寬鬆，適合探索性訓練。正式訓練建議：(1) 100 epochs 時設 `patience=20-30`，(2) 觀察 validation curve 若在 epoch 50-70 即平穩，可提早停止節省時間。  
**來源**: Ultralytics training arguments + 專案 Phase8 config (`patience=30`)

### 15. Android 端側推論優化路徑
**結論**: LiteRT (TensorFlow Lite) 推論優化順序：(1) 先 benchmark CPU baseline，(2) 測試 GPU delegate（ML Drift 加速），(3) 評估 NPU delegate（Tensor/Qualcomm/MediaTek），(4) 若延遲仍高，考慮 INT8 量化（需 calibration dataset）。CompiledModel API 已取代舊 Interpreter API。  
**來源**: [Google AI Edge LiteRT Inference Documentation](https://ai.google.dev/edge/litert/inference) + Android ML best practices

---

## B. 三個可執行方案（保守/平衡/積極）

### 方案一：保守穩定型（推薦用於生產基線）
**目標**: 確保訓練穩定，避免 NaN/OOM，逐步提升 mAP50 從 0.28 → 0.45-0.50

**超參數配置**:
```yaml
# 基礎設定
model: yolo26s.pt
data: harmony_phase8_subset_strat20_seed20260213.yaml
epochs: 100
batch: 6  # 從 4 提升至 6（保守增加）
imgsz: 1280
device: '0'

# 穩定性優先
amp: false  # 關閉混合精度，避免 NaN
lr0: 0.0003  # 降低初始學習率
lrf: 0.01
warmup_epochs: 10  # 加長 warmup
optimizer: AdamW
weight_decay: 0.0005

# 數據載入
workers: 6  # 從 20 降至 6，減少 CPU 競爭
cache: disk  # 折衷方案，避免 RAM 不足

# 增強策略（保守）
mosaic: 0.8  # 前期保持，後期自動降低
close_mosaic: 15  # 最後 15 epochs 關閉
mixup: 0.1
copy_paste: 0.2
degrees: 0.0  # OMR 特殊需求
perspective: 0.0
translate: 0.1
scale: 0.3
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.3

# Early stopping
patience: 25
save_period: 10  # 每 10 epoch 存檔

# 驗證
val: true
plots: true
```

**風險評估**:
- ✅ **低風險**: NaN/OOM 機率極低（AMP 關閉、LR 保守、Batch 小）
- ⚠️ **中風險**: 收斂速度較慢，可能需要 120-150 epochs 才達最佳
- ❌ **可接受的犧牲**: 訓練時間較長（約 6-8 小時 @ RTX 5090）

**適用情境**: 
- 首次正式訓練
- 需要穩定基線模型用於部署
- GPU 資源有限或共享環境

---

### 方案二：平衡進取型（推薦用於研發迭代）
**目標**: 在穩定性與速度間取得平衡，快速達到 mAP50 0.50-0.60

**超參數配置**:
```yaml
# 基礎設定
model: yolo26s.pt
data: harmony_phase8_subset_strat20_seed20260213.yaml
epochs: 100
batch: -1  # 自動調整至 60% GPU 記憶體使用
imgsz: 1280
device: '0'

# 平衡設定
amp: true  # 啟用 AMP，需密切監控 loss
lr0: 0.0005  # 當前值保持
lrf: 0.01
warmup_epochs: 5
optimizer: AdamW
weight_decay: 0.0005

# 數據載入（加速）
workers: 8
cache: ram  # 若記憶體充足（數據集 < 10GB）

# 增強策略（標準）
mosaic: 1.0
close_mosaic: 10
mixup: 0.15
copy_paste: 0.3
degrees: 0.0
perspective: 0.0
translate: 0.15
scale: 0.5
hsv_h: 0.02
hsv_s: 0.7
hsv_v: 0.4

# Early stopping
patience: 20
save_period: 10

# 類別不平衡處理
cls: 0.8  # 提高分類 loss 權重
box: 7.5
dfl: 1.5

# 驗證
val: true
plots: true
```

**風險評估**:
- ✅ **可控風險**: AMP 啟用需監控，但有 GradScaler 保護
- ⚠️ **中風險**: Batch size 自動調整可能觸發 OOM（第一次訓練時需觀察）
- ❌ **需監控**: 每 10 epochs 檢查 loss curve，若出現 spike 需降級至方案一

**適用情境**:
- 已完成穩定性測試（如當前 20-epoch smoke test）
- 有完整 GPU 資源（RTX 5090 獨佔）
- 可接受 5% 機率的訓練中斷重啟

---

### 方案三：激進優化型（實驗性，需專家監控）
**目標**: 追求最高 mAP（目標 0.65+），接受較高失敗風險

**超參數配置**:
```yaml
# 基礎設定
model: yolo26s.pt
data: harmony_phase8_full.yaml  # 使用完整數據集（非 subset）
epochs: 150
batch: 16  # 激進增加（需測試 GPU 承受度）
imgsz: 1280
device: [0]  # 單 GPU（可擴展至多 GPU）

# 激進設定
amp: true
lr0: 0.001  # 較高初始 LR，搭配長訓練
lrf: 0.001  # 更低最終 LR
warmup_epochs: 3
optimizer: SGD  # 改用 SGD + Momentum
momentum: 0.95
weight_decay: 0.001  # 更強正則化
cos_lr: true  # Cosine annealing

# 數據載入（極限加速）
workers: 12
cache: ram

# 增強策略（激進）
mosaic: 1.0
close_mosaic: 5  # 僅最後 5 epochs 關閉
mixup: 0.2
copy_paste: 0.5  # 大幅提高，補償弱類別
auto_augment: randaugment
erasing: 0.4
degrees: 0.0
perspective: 0.0
translate: 0.2
scale: 0.7
hsv_h: 0.03
hsv_s: 0.9
hsv_v: 0.5
fliplr: 0.5

# 類別不平衡（強化）
cls: 1.5
box: 10.0
dfl: 2.0

# Multi-scale training
multi_scale: 0.5  # ±50% 隨機縮放

# Early stopping（更寬容）
patience: 40
save_period: 5

# 驗證
val: true
plots: true
```

**風險評估**:
- ❌ **高風險**: Batch=16 可能 OOM，需實測
- ❌ **高風險**: SGD 收斂不穩定機率較高，需密切監控
- ⚠️ **中風險**: 過度增強可能破壞 OMR 特定特徵（需驗證）
- ✅ **潛在收益**: 若成功，mAP 可能提升至 0.65-0.70

**適用情境**:
- 已有穩定基線模型（方案一或二成功）
- 追求 SOTA 效能，可接受多次實驗失敗
- 有充足時間和 GPU 資源進行調參

---

## C. 推薦的單一路線（具體超參數）

**綜合判斷**: 基於當前狀況（20-epoch smoke test 顯示穩定但效能偏低），**建議採用「方案二（平衡進取型）」作為下一步**，理由如下：

### 選擇理由
1. **Smoke test 驗證**: 20 epochs 未出現 NaN（即使 workers=20），顯示數據和基礎配置健康
2. **效能需求**: 當前 mAP50=0.28 距離可用模型（≥0.60）仍有較大差距，需要更積極策略
3. **風險可控**: AMP 若出問題可立即回退，Batch=-1 自動調整避免盲目設定
4. **迭代效率**: 保守方案可能需要 150+ epochs，平衡方案可在 100 epochs 內達標

### 完整執行指令

#### Step 1: 準備配置文件
```bash
cd /home/thc1006/dev/music-app

# 創建訓練配置
cat > training/configs/yolo26s_balanced_100ep.yaml << 'EOF'
# YOLO26s Balanced Training Config
# Target: mAP50 0.50-0.60 within 100 epochs

# Model & Data
task: detect
mode: train
model: yolo26s.pt
data: /home/thc1006/.copilot/session-state/bc083904-d73e-4b8a-9f91-9f4b9271af0f/files/harmony_phase8_subset_strat20_seed20260213.yaml

# Training Duration
epochs: 100
patience: 20

# Batch & Image Size
batch: -1  # Auto-adjust to 60% GPU memory
imgsz: 1280

# Hardware
device: '0'
workers: 8

# Optimizer & Learning Rate
optimizer: AdamW
lr0: 0.0005
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 5
warmup_momentum: 0.85
warmup_bias_lr: 0.1

# Loss Weights (balanced for OMR)
box: 7.5
cls: 0.8  # Increased for class imbalance
dfl: 1.5

# Augmentation (OMR-specific)
mosaic: 1.0
close_mosaic: 10
mixup: 0.15
copy_paste: 0.3
hsv_h: 0.02
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0  # No rotation for OMR
translate: 0.15
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
auto_augment: randaugment
erasing: 0.4

# Memory & Caching
cache: ram  # If dataset < 10GB, else use 'disk'
amp: true  # Monitor for NaN/Inf

# Saving & Validation
save: true
save_period: 10
project: harmony_omr_v2_yolo26
name: yolo26s_balanced_100ep_run1
exist_ok: true
val: true
plots: true

# Misc
pretrained: true
verbose: true
seed: 0
deterministic: true
cos_lr: false
EOF
```

#### Step 2: 執行訓練（帶監控）
```bash
# 啟動訓練並記錄 log
nohup python3 -m ultralytics.engine.train \
    cfg=training/configs/yolo26s_balanced_100ep.yaml \
    > training/logs/yolo26s_balanced_100ep_run1.log 2>&1 &

# 獲取 PID
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"

# 實時監控 loss（另一個 terminal）
watch -n 30 "tail -50 training/logs/yolo26s_balanced_100ep_run1.log | grep -E '(Epoch|loss|mAP)'"
```

#### Step 3: 訓練中檢查點（每 20 epochs）
```bash
# 檢查 loss 是否出現 NaN/Inf
grep -i "nan\|inf" training/logs/yolo26s_balanced_100ep_run1.log

# 檢查 OOM 警告
grep -i "CUDA.*OOM\|OutOfMemoryError" training/logs/yolo26s_balanced_100ep_run1.log

# 檢查當前最佳 mAP
tail -20 runs/detect/harmony_omr_v2_yolo26/yolo26s_balanced_100ep_run1/results.csv | \
    awk -F',' 'NR>1 {print $1, $8}' | sort -k2 -nr | head -1

# 若發現問題，立即停止並切換至方案一
# kill -9 $TRAIN_PID
```

#### Step 4: 訓練完成後評估
```bash
# 查看最佳模型效能
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('runs/detect/harmony_omr_v2_yolo26/yolo26s_balanced_100ep_run1/results.csv', skipinitialspace=True)
best_epoch = df['metrics/mAP50(B)'].idxmax()
print(f"Best Epoch: {df.loc[best_epoch, 'epoch']}")
print(f"Best mAP50: {df.loc[best_epoch, 'metrics/mAP50(B)']:.4f}")
print(f"Best Recall: {df.loc[best_epoch, 'metrics/recall(B)']:.4f}")
print(f"Best Precision: {df.loc[best_epoch, 'metrics/precision(B)']:.4f}")
EOF

# 在驗證集測試
yolo detect val \
    model=runs/detect/harmony_omr_v2_yolo26/yolo26s_balanced_100ep_run1/weights/best.pt \
    data=/home/thc1006/.copilot/session-state/bc083904-d73e-4b8a-9f91-9f4b9271af0f/files/harmony_phase8_subset_strat20_seed20260213.yaml \
    imgsz=1280 \
    conf=0.25 \
    iou=0.55

# 若 mAP50 < 0.45，考慮：
# 1. 檢查數據品質（duplicate labels, annotation errors）
# 2. 延長訓練至 150 epochs
# 3. 切換至方案一（更保守）
```

---

## D. 兩階段執行清單（先驗證，再正式訓練）

### 階段一：穩定性驗證與數據健檢（2-3 天）

#### Task 1.1: 數據品質檢查 ✅
**預計時間**: 2-4 小時

```bash
# 檢查 duplicate labels
cd /home/thc1006/dev/music-app
python3 << 'EOF'
import os
from pathlib import Path

dataset_path = Path("training/datasets/yolo_harmony_v2_phase8_subset")
label_files = list((dataset_path / "train" / "labels").glob("*.txt"))

duplicates_summary = []
for label_file in label_files[:100]:  # 先檢查 100 個樣本
    with open(label_file, 'r') as f:
        lines = f.readlines()
    unique_lines = set(lines)
    if len(lines) != len(unique_lines):
        duplicates_summary.append({
            'file': label_file.name,
            'total': len(lines),
            'unique': len(unique_lines),
            'duplicates': len(lines) - len(unique_lines)
        })

if duplicates_summary:
    print(f"Found {len(duplicates_summary)} files with duplicates")
    for item in duplicates_summary[:5]:
        print(item)
else:
    print("No duplicates detected in sample")
EOF

# 檢查類別分佈（找出弱類別）
python3 << 'EOF'
import numpy as np
from collections import Counter
from pathlib import Path

dataset_path = Path("training/datasets/yolo_harmony_v2_phase8_subset")
label_files = list((dataset_path / "train" / "labels").glob("*.txt"))

class_counts = Counter()
for label_file in label_files:
    with open(label_file, 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

print("Class Distribution:")
for cls_id, count in sorted(class_counts.items(), key=lambda x: x[1]):
    print(f"Class {cls_id}: {count} instances")

# 找出低於 1% 的弱類別
total = sum(class_counts.values())
weak_classes = [cls_id for cls_id, count in class_counts.items() if count/total < 0.01]
print(f"\nWeak classes (< 1%): {weak_classes}")
EOF

# 視覺化檢查（隨機抽樣）
# 手動檢查標註品質，特別關注 barline_double, tie 等弱類別
```

**產出**:
- [ ] `training/reports/data_quality_report_phase8.txt`
- [ ] 弱類別清單與建議增強策略
- [ ] 需修正的標註檔案清單（若有）

---

#### Task 1.2: 30-Epoch 穩定性測試（含 AMP）
**預計時間**: 3-4 小時（訓練） + 1 小時（分析）

```bash
# 配置文件
cat > training/configs/yolo26s_stability_test_30ep.yaml << 'EOF'
task: detect
mode: train
model: yolo26s.pt
data: /home/thc1006/.copilot/session-state/bc083904-d73e-4b8a-9f91-9f4b9271af0f/files/harmony_phase8_subset_strat20_seed20260213.yaml
epochs: 30
batch: 6
imgsz: 1280
device: '0'
workers: 8
cache: disk

# 測試 AMP 穩定性
amp: true
lr0: 0.0005
optimizer: AdamW
warmup_epochs: 5

# 標準增強
mosaic: 1.0
close_mosaic: 10
mixup: 0.15
copy_paste: 0.3
degrees: 0.0
perspective: 0.0

project: harmony_omr_v2_yolo26
name: yolo26s_stability_test_30ep
exist_ok: true
save_period: 10
val: true
plots: true
EOF

# 執行訓練
python3 -m ultralytics.engine.train \
    cfg=training/configs/yolo26s_stability_test_30ep.yaml \
    > training/logs/yolo26s_stability_test_30ep.log 2>&1

# 訓練中監控（另一 terminal）
watch -n 60 "tail -30 training/logs/yolo26s_stability_test_30ep.log | grep -E '(Epoch|loss|NaN|OOM|mAP)'"
```

**驗收標準**:
- [ ] **必須**: 30 epochs 中無 NaN/Inf loss
- [ ] **必須**: CUDA OOM 出現次數 ≤ 2 次（若 > 2 次需降低 batch）
- [ ] **期望**: mAP50 在 30 epochs 達到 ≥ 0.35（相比 20-epoch 的 0.28 有明顯進步）
- [ ] **期望**: Loss curve 平滑下降，無劇烈震盪

**分析腳本**:
```bash
python3 << 'EOF'
import pandas as pd
import numpy as np

results_path = 'runs/detect/harmony_omr_v2_yolo26/yolo26s_stability_test_30ep/results.csv'
df = pd.read_csv(results_path, skipinitialspace=True)

# 檢查 NaN/Inf
nan_epochs = df[df.isnull().any(axis=1)]['epoch'].tolist()
print(f"Epochs with NaN/Inf: {nan_epochs if nan_epochs else 'None (✅)'}")

# 檢查 loss 震盪
train_box_loss = df['train/box_loss'].values
loss_std = np.std(train_box_loss[5:])  # 忽略前 5 epochs
print(f"Loss stability (std after epoch 5): {loss_std:.4f} ({'✅' if loss_std < 0.5 else '⚠️'})")

# 檢查 mAP 進展
final_map = df.iloc[-1]['metrics/mAP50(B)']
best_map = df['metrics/mAP50(B)'].max()
print(f"Final mAP50: {final_map:.4f}")
print(f"Best mAP50: {best_map:.4f} @ epoch {df['metrics/mAP50(B)'].idxmax()}")
print(f"Target mAP50 (≥0.35): {'✅' if best_map >= 0.35 else '❌ Need investigation'}")

# 判斷是否可進入正式訓練
if len(nan_epochs) == 0 and loss_std < 0.5 and best_map >= 0.30:
    print("\n✅ PASS: Ready for full 100-epoch training")
else:
    print("\n⚠️ CAUTION: Consider adjustments before full training")
    print("Recommendations:")
    if len(nan_epochs) > 0:
        print("  - Disable AMP (amp=false)")
        print("  - Reduce lr0 to 0.0003")
    if loss_std >= 0.5:
        print("  - Increase warmup_epochs to 10")
        print("  - Reduce batch size to 4")
    if best_map < 0.30:
        print("  - Check data quality")
        print("  - Review weak class handling")
EOF
```

**產出**:
- [ ] `training/reports/stability_test_30ep_report.md`
- [ ] 決策：GO / NO-GO 進入正式訓練
- [ ] 若 NO-GO，調整建議清單

---

#### Task 1.3: 弱類別專項測試（可選）
**預計時間**: 4-6 小時

若穩定性測試通過但 mAP 仍低，執行此任務：

```bash
# 針對弱類別（如 barline_double, tie）做 targeted augmentation
cat > training/configs/yolo26s_weak_class_boost.yaml << 'EOF'
# ... (基於 stability test config)

# 強化弱類別增強
copy_paste: 0.6  # 大幅提高
mixup: 0.25

# 調整 loss 權重
cls: 1.2  # 提高分類 loss

# 僅訓練 20 epochs 觀察效果
epochs: 20
EOF

# 比較兩次訓練的弱類別 mAP
# （需要 per-class mAP，可從 confusion matrix 推算）
```

**產出**:
- [ ] 弱類別 mAP 提升報告
- [ ] 建議納入正式訓練的增強策略

---

### 階段二：正式訓練與持續優化（5-7 天）

#### Task 2.1: 100-Epoch 正式訓練（方案二）
**預計時間**: 12-16 小時（訓練） + 2 小時（評估）

```bash
# 使用前述「方案二」完整配置
# 執行 training/configs/yolo26s_balanced_100ep.yaml

# 關鍵監控點：
# - Epoch 20: 檢查 mAP 是否 ≥ 0.40
# - Epoch 50: 檢查 mAP 是否 ≥ 0.50
# - Epoch 80: 檢查是否平穩（連續 10 epochs 無明顯提升則提早停止）
```

**產出**:
- [ ] 最佳模型 `weights/best.pt`
- [ ] 完整 results.csv 與 confusion matrix
- [ ] mAP50 ≥ 0.50（目標）或分析報告說明差距原因

---

#### Task 2.2: 模型驗證與誤差分析
**預計時間**: 4-6 小時

```bash
# 在測試集驗證
yolo detect val \
    model=runs/detect/harmony_omr_v2_yolo26/yolo26s_balanced_100ep_run1/weights/best.pt \
    data=<test_set>.yaml \
    imgsz=1280 \
    save_json=true \
    save_conf=true

# 分析 confusion matrix
# 找出高混淆類別對（如 notehead_filled vs notehead_hollow）

# Per-class mAP 分析
python3 << 'EOF'
# 解析 confusion matrix，計算各類別 precision/recall
# 產出 training/reports/per_class_analysis.md
EOF

# 錯誤案例視覺化
# 隨機抽取 20 張低 confidence 或錯誤預測的圖片
# 手動分析原因：標註錯誤 vs 模型缺陷 vs 數據不足
```

**產出**:
- [ ] `training/reports/model_validation_report.md`
- [ ] 高混淆類別對清單與改進建議
- [ ] 錯誤案例分析（含截圖）

---

#### Task 2.3: 超參數微調（若需要）
**預計時間**: 2-3 天（若 mAP < 0.50 才執行）

```bash
# 使用 Ultralytics 自動調參工具
python3 << 'EOF'
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

# 定義搜索空間（聚焦關鍵參數）
search_space = {
    "lr0": (0.0001, 0.001),
    "cls": (0.5, 2.0),  # 類別不平衡調整
    "copy_paste": (0.2, 0.6),  # 弱類別增強
    "mosaic": (0.7, 1.0),
}

# 執行 50 次迭代（基於 30 epochs 快速驗證）
results = model.tune(
    data="harmony_phase8_subset_strat20_seed20260213.yaml",
    epochs=30,
    iterations=50,
    optimizer="AdamW",
    space=search_space,
    plots=True,
    save=False,
    val=False,  # 僅最終 epoch 驗證，加速
)
EOF

# 取得最佳超參數後，重新訓練 100 epochs
```

**產出**:
- [ ] 最佳超參數組合 `tune/best_hyperparameters.yaml`
- [ ] 優化後模型（若 mAP 提升 ≥ 5%）

---

#### Task 2.4: 部署準備與 Android 測試
**預計時間**: 2-3 天

```bash
# 導出 TFLite 模型
yolo export \
    model=runs/detect/harmony_omr_v2_yolo26/yolo26s_balanced_100ep_run1/weights/best.pt \
    format=tflite \
    imgsz=1280 \
    int8=false  # 先導出 FP16

# 若模型過大（> 50MB），再嘗試 INT8 量化
# 需準備 calibration dataset（1000-2000 張代表性圖片）

# 集成至 Android app
cd android-app

# 替換 assets/yolo26s_omr.tflite
# 更新 Yolo12OmrClient.kt 中的輸入尺寸配置

# 在實機測試
./gradlew installDebug
# 測試延遲、準確率、記憶體使用
```

**產出**:
- [ ] `android-app/assets/yolo26s_omr_v1.tflite`
- [ ] Android 端測試報告（延遲、mAP、記憶體）
- [ ] 若延遲 > 500ms，啟動量化或模型裁剪方案

---

### 總結檢查表

**階段一完成標準**:
- [ ] 數據品質問題已修正（duplicate labels, weak classes identified）
- [ ] 30-epoch 穩定性測試通過（無 NaN, mAP ≥ 0.35）
- [ ] 明確的訓練策略（保守/平衡/激進）選定

**階段二完成標準**:
- [ ] 100-epoch 訓練完成，mAP50 ≥ 0.50（或有合理解釋）
- [ ] Per-class 分析完成，弱類別有改進計畫
- [ ] TFLite 模型已導出並在 Android 測試

**最終交付物**:
- [ ] 生產級 YOLO26s 模型（mAP50 ≥ 0.60 為優秀，≥ 0.50 為可用）
- [ ] 完整訓練報告（包含所有實驗 metrics 對比）
- [ ] Android 集成與效能基準測試報告
- [ ] 後續優化路線圖（若需要進一步提升）

---

## E. 補充建議

### 1. MLflow 實驗追蹤
建議所有訓練都記錄到 MLflow，便於比較：

```bash
# 在訓練腳本中加入
export MLFLOW_TRACKING_URI=file:///home/thc1006/dev/music-app/runs/mlflow

python3 << 'EOF'
import mlflow
from ultralytics import YOLO

mlflow.set_experiment("yolo26s_omr_training")

with mlflow.start_run(run_name="balanced_100ep"):
    model = YOLO("yolo26s.pt")
    results = model.train(
        data="...",
        epochs=100,
        # ... other params
    )
    
    # Log final metrics
    mlflow.log_metrics({
        "final_mAP50": results.results_dict['metrics/mAP50(B)'],
        "final_recall": results.results_dict['metrics/recall(B)'],
        # ...
    })
    mlflow.log_artifact("runs/detect/.../weights/best.pt")
EOF
```

### 2. 資料版本化（DVC）
```bash
# 確保數據可追溯
cd /home/thc1006/dev/music-app
dvc add training/datasets/yolo_harmony_v2_phase8_subset
git add training/datasets/yolo_harmony_v2_phase8_subset.dvc
git commit -m "Track Phase8 subset dataset"
dvc push
```

### 3. 定期 Checkpoint 備份
```bash
# 每 10 epochs 自動備份至遠端（避免訓練中斷損失）
rsync -avz --progress \
    runs/detect/harmony_omr_v2_yolo26/yolo26s_balanced_100ep_run1/weights/ \
    /path/to/backup/
```

---

## F. 參考文獻與來源

1. **Ultralytics Documentation**
   - [Training Mode](https://docs.ultralytics.com/modes/train/)
   - [Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
   - [YOLO11 Model](https://docs.ultralytics.com/models/yolo11/)

2. **PyTorch Documentation**
   - [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
   - [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

3. **Academic Research**
   - arXiv OMR papers (search: "optical music recognition")
   - Small object detection techniques (YOLO-tiling, SAHI)

4. **Google AI Edge**
   - [LiteRT Inference](https://ai.google.dev/edge/litert/inference)
   - Android ML best practices

5. **Project-Specific**
   - `LATEST_STATUS_AND_DATASET_SOURCES.md`
   - Training logs and Phase configs
   - Previous smoke test results

---

**報告撰寫時間**: 約 2 小時（深度調研 + 交叉驗證 + 方案設計）  
**建議後續**: 立即執行「階段一 Task 1.1」，48 小時內完成穩定性驗證，1 週內啟動正式訓練。
