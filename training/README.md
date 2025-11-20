# YOLO12 訓練指南

完整的 YOLO12 四部和聲 OMR 模型訓練流程

---

## 📋 目錄

1. [快速開始](#快速開始)
2. [環境設置](#環境設置)
3. [資料集準備](#資料集準備)
4. [模型訓練](#模型訓練)
5. [模型匯出](#模型匯出)
6. [故障排除](#故障排除)

---

## 快速開始

```bash
# 1. 安裝依賴
pip install -r requirements-train.txt

# 2. 下載資料集（手動）
mkdir -p datasets
cd datasets
git clone https://github.com/OMR-Research/muscima-pp.git
cd ..

# 3. 轉換資料集
python convert_dataset.py

# 4. 訓練 YOLO12s
python yolo12_train.py --model yolo12s

# 5. 匯出 INT8 模型
python export_models.py --model harmony_omr/yolo12s_XXXXXX/weights/best.pt
```

---

## 環境設置

### 硬體需求

- **GPU**: NVIDIA RTX 5060 (8GB VRAM) 或更高
- **RAM**: 16GB 以上
- **儲存空間**: 至少 50GB 可用空間
- **作業系統**: Windows 10/11, Linux (Ubuntu 20.04+)

### 軟體需求

- **Python**: 3.10 或 3.11
- **CUDA**: 11.8 或 12.1
- **cuDNN**: 對應 CUDA 版本

### 安裝步驟

#### 1. 建立虛擬環境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 2. 安裝依賴套件

```bash
pip install -r requirements-train.txt
```

#### 3. 驗證 GPU

```bash
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

預期輸出:
```
CUDA 可用: True
GPU: NVIDIA GeForce RTX 5060
```

---

## 資料集準備

### 方法 A: 使用 MUSCIMA++（推薦）

#### 1. 下載資料集

```bash
cd datasets
git clone https://github.com/OMR-Research/muscima-pp.git
cd ..
```

#### 2. 轉換為 YOLO 格式

```bash
python convert_dataset.py \
    --input datasets/muscima-pp \
    --output datasets/yolo_harmony \
    --train-ratio 0.8
```

**預期輸出**:
```
找到 140 個 XML 標註檔案
訓練集: 112 張
驗證集: 28 張

處理訓練集...
處理驗證集...

轉換完成統計
總圖片數: 140
總標註數: 91,255
平均每張: 651.8 個標註
```

#### 3. 驗證轉換結果

```bash
ls datasets/yolo_harmony/
# 應該看到:
# images/train/  images/val/  labels/train/  labels/val/
```

### 方法 B: 使用合成資料（快速測試）

如果您想快速驗證訓練流程，可以使用小量合成資料:

```bash
python generate_synthetic_data.py --num-images 100
```

---

## 模型訓練

### YOLO12s 訓練（推薦 - 高準確度）

```bash
python yolo12_train.py \
    --model yolo12s \
    --batch 16 \
    --epochs 250
```

**預估時間**: 200-250 epochs × 8-10 小時 = **約 2-3 天**

### YOLO12n 訓練（輕量級 - 快速備援）

```bash
python yolo12_train.py \
    --model yolo12n \
    --batch 24 \
    --epochs 200
```

**預估時間**: 150-200 epochs × 4-6 小時 = **約 1-1.5 天**

### 訓練參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | yolo12s | 模型變體 (yolo12s 或 yolo12n) |
| `--batch` | 16 | Batch size（RTX 5060 建議 12-16） |
| `--epochs` | 250 | 訓練輪數（YOLO12 需要更長時間） |
| `--data` | omr_harmony.yaml | 資料集配置檔案 |

### 監控訓練進度

#### 方法 1: 查看即時 log

```bash
tail -f harmony_omr/yolo12s_XXXXXX/results.csv
```

#### 方法 2: 使用 TensorBoard

```bash
tensorboard --logdir harmony_omr/
```

開啟瀏覽器訪問: http://localhost:6006

### 訓練中斷與恢復

如果訓練中斷（Ctrl+C 或斷電），可以從最後的 checkpoint 繼續:

```bash
yolo train resume model=harmony_omr/yolo12s_XXXXXX/weights/last.pt
```

---

## 模型匯出

訓練完成後，將 .pt 模型轉換為 TFLite INT8 格式：

### 匯出 YOLO12s

```bash
python export_models.py \
    --model harmony_omr/yolo12s_20251120_XXXXXX/weights/best.pt \
    --dataset datasets/yolo_harmony
```

### 匯出 YOLO12n

```bash
python export_models.py \
    --model harmony_omr/yolo12n_20251120_YYYYYY/weights/best.pt \
    --output-name yolo12n
```

### 匯出流程說明

腳本會自動執行:

1. ✅ **匯出 FP32 TFLite** (~40MB for YOLO12s)
2. ✅ **INT8 量化** (使用 100 張代表性圖片)
3. ✅ **驗證模型** (測試推論)
4. ✅ **複製到 Android** (自動放入 `../android-app/app/src/main/assets/models/`)

### 預期輸出

```
量化效果對比
FP32 大小: 38.5 MB
INT8 大小: 10.2 MB
壓縮比: 3.77x
節省空間: 28.3 MB (73.5%)

✅ 已複製到: ../android-app/app/src/main/assets/models/yolo12s_int8.tflite
```

---

## 故障排除

### 問題 1: CUDA out of memory

**症狀**:
```
RuntimeError: CUDA out of memory
```

**解決方案**:
1. 降低 batch size:
   ```bash
   python yolo12_train.py --model yolo12s --batch 12
   ```

2. 使用 YOLO12n (更小):
   ```bash
   python yolo12_train.py --model yolo12n --batch 20
   ```

### 問題 2: 找不到資料集

**症狀**:
```
❌ 錯誤: 資料集根目錄不存在
```

**解決方案**:
1. 檢查路徑是否正確:
   ```bash
   ls datasets/yolo_harmony/images/train/
   ```

2. 重新執行轉換:
   ```bash
   python convert_dataset.py
   ```

### 問題 3: 訓練不收斂

**症狀**:
- Loss 不下降
- mAP 一直很低 (< 20%)

**可能原因與解決**:

1. **學習率太高/太低**
   - 查看 `yolo12_train.py` 的 `lr0` 參數
   - 預設 0.01 通常適合

2. **資料集標註問題**
   - 檢查 `datasets/yolo_harmony/labels/train/` 中的 .txt 檔案
   - 確保座標在 [0, 1] 範圍內

3. **Epochs 不足**
   - YOLO12 需要 200-250 epochs
   - 比 YOLO11 更長

### 問題 4: TFLite 匯出失敗

**症狀**:
```
❌ 錯誤: FP32 匯出失敗
```

**解決方案**:

1. 檢查 Ultralytics 版本:
   ```bash
   pip show ultralytics
   # 應該 >= 8.3.0
   ```

2. 升級到最新版:
   ```bash
   pip install --upgrade ultralytics
   ```

3. 如果仍失敗，嘗試先匯出 ONNX:
   ```bash
   yolo export model=best.pt format=onnx
   ```

---

## 進階配置

### 自訂資料增強

編輯 `yolo12_train.py` 中的增強參數:

```python
# 幾何變換
degrees=5.0,          # 旋轉角度
translate=0.1,        # 平移比例
scale=0.2,            # 縮放比例

# 顏色調整
hsv_h=0.015,         # 色調
hsv_s=0.5,           # 飽和度
hsv_v=0.4,           # 亮度

# Mosaic 增強
mosaic=0.5,          # 50% 機率
mixup=0.1,           # 10% 機率
```

### 多 GPU 訓練

```bash
# 使用 GPU 0 和 1
python yolo12_train.py --model yolo12s --device 0,1
```

### 自訂類別

如需修改檢測類別，編輯 `omr_harmony.yaml`:

```yaml
nc: 25  # 改為 25 個類別

names:
  0: notehead_filled
  # ... 新增更多類別
  24: new_class_name
```

---

## 檔案結構

```
training/
├── README.md                    # 本檔案
├── requirements-train.txt       # Python 依賴
├── omr_harmony.yaml            # 資料集配置
├── yolo12_train.py             # 訓練腳本
├── convert_dataset.py          # 資料集轉換
├── export_models.py            # 模型匯出
├── generate_synthetic_data.py  # 合成資料生成（可選）
├── datasets/                   # 資料集目錄
│   ├── muscima-pp/            # MUSCIMA++ 原始資料
│   └── yolo_harmony/          # YOLO 格式資料
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
└── harmony_omr/               # 訓練輸出
    ├── yolo12s_20251120_XXXXXX/
    │   ├── weights/
    │   │   ├── best.pt
    │   │   └── last.pt
    │   └── results.csv
    └── yolo12n_20251120_YYYYYY/
        └── ...
```

---

## 下一步

訓練與匯出完成後:

1. ✅ 確認 `.tflite` 模型已在 Android assets:
   ```bash
   ls ../android-app/app/src/main/assets/models/
   # 應該看到: yolo12s_int8.tflite, yolo12n_int8.tflite
   ```

2. ✅ 繼續 Android 整合（Week 2-3）

3. ✅ 參考 `docs/yolo12_omr_implementation.md` 完整文檔

---

## 聯絡與支援

- 專案 GitHub: [待補充]
- 問題回報: [待補充]
- YOLO12 官方文檔: https://docs.ultralytics.com/models/yolo12/

---

**最後更新**: 2025-11-20
**作者**: thc1006 + Claude
