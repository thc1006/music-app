# Barline Dataset Processing Scripts

這個目錄包含下載和轉換外部 OMR barline 數據集的完整工具集。

## 快速開始

### 一鍵執行完整流程

```bash
cd /home/thc1006/dev/music-app/training/scripts
chmod +x run_all_barline_pipeline.sh
./run_all_barline_pipeline.sh
```

這個腳本會自動執行：
1. 下載所有數據集
2. 轉換為 YOLO 格式
3. 合併所有數據集

## 腳本說明

### 1. download_all_datasets.py

下載所有外部 OMR 數據集。

**基本用法**：
```bash
python download_all_datasets.py
```

**下載特定數據集**：
```bash
# 只下載 AudioLabs
python download_all_datasets.py --datasets audiolabs

# 下載多個數據集
python download_all_datasets.py --datasets omr_layout doremi

# 指定輸出目錄
python download_all_datasets.py --output /path/to/output
```

**參數**：
- `--datasets`: 要下載的數據集（omr_layout, audiolabs, doremi, all）
- `--output`: 輸出目錄（預設：`/home/thc1006/dev/music-app/training/datasets/external_barlines`）

### 2. convert_omr_layout.py

轉換 OMR Layout Analysis 數據集（從 measure 邊界提取 barline）。

**用法**：
```bash
python convert_omr_layout.py

# 自定義路徑
python convert_omr_layout.py \
  --input /path/to/datasets-release \
  --output /path/to/output \
  --val-split 0.15
```

**參數**：
- `--input`: 輸入目錄（解壓後的數據集）
- `--output`: 輸出目錄
- `--val-split`: 驗證集比例（預設：0.15）

**特點**：
- 從 measure bounding boxes 提取 barline
- 自動分割訓練/驗證集
- 生成 YOLO 格式標註

### 3. convert_audiolabs.py

轉換 AudioLabs v2 數據集（從 measure 標註提取 barline）。

**用法**：
```bash
python convert_audiolabs.py

# 自定義路徑
python convert_audiolabs.py \
  --input /path/to/MeasureBoundingBoxAnnotations \
  --output /path/to/output
```

**特點**：
- 支持 JSON 格式 measure 標註
- 自動偵測行尾使用 final barline
- 智能圖片檔案搜索

### 4. convert_doremi.py

轉換 DoReMi 數據集（從 OMR XML 提取 barline）。

**用法**：
```bash
python convert_doremi.py

# 自定義路徑
python convert_doremi.py \
  --input /path/to/DoReMi_1.0 \
  --output /path/to/output
```

**特點**：
- 解析 OMR XML 格式
- 支持多種 barline 類型（barline, double, final, repeat）
- 類別名稱自動映射

### 5. merge_barline_datasets.py

合併所有已轉換的數據集。

**用法**：
```bash
python merge_barline_datasets.py

# 自定義路徑
python merge_barline_datasets.py \
  --omr-layout /path/to/omr_layout/converted \
  --audiolabs /path/to/audiolabs/converted \
  --doremi /path/to/doremi/converted \
  --output /path/to/merged
```

**特點**：
- 自動重新映射類別 ID（23-26 → 0-3）
- 生成統一的 data.yaml
- 提供詳細統計信息

### 6. run_all_barline_pipeline.sh

自動化完整流程的 Bash 腳本。

**用法**：
```bash
chmod +x run_all_barline_pipeline.sh
./run_all_barline_pipeline.sh
```

**流程**：
1. 檢查 Python 環境與依賴
2. 詢問是否下載數據集
3. 依序轉換所有已下載的數據集
4. 合併所有轉換後的數據集
5. 輸出訓練指令

## 輸出目錄結構

```
external_barlines/
├── omr_layout/
│   ├── datasets-release.zip          # 原始下載
│   ├── datasets-release/             # 解壓後
│   └── converted/                    # YOLO 格式
│       ├── images/{train,val}/
│       ├── labels/{train,val}/
│       ├── data.yaml
│       └── conversion_stats.json
│
├── audiolabs/
│   ├── audiolabs_measures.zip
│   ├── MeasureBoundingBoxAnnotations/
│   └── converted/
│       ├── images/{train,val}/
│       ├── labels/{train,val}/
│       ├── data.yaml
│       └── conversion_stats.json
│
├── doremi/
│   ├── doremi_v1.0.zip
│   ├── DoReMi_1.0/
│   └── converted/
│       ├── images/{train,val}/
│       ├── labels/{train,val}/
│       ├── data.yaml
│       └── conversion_stats.json
│
└── merged/                           # 最終合併數據集
    ├── images/{train,val}/
    ├── labels/{train,val}/
    ├── data.yaml
    └── merge_stats.json
```

## 類別映射

所有數據集統一映射到以下類別：

| 合併後 ID | 原始 ID | 類別名稱 | 描述 |
|-----------|---------|----------|------|
| 0 | 23 | barline | 普通小節線 |
| 1 | 24 | barline_double | 雙小節線 |
| 2 | 25 | barline_final | 終止線 |
| 3 | 26 | barline_repeat | 反覆記號 |

## 依賴安裝

```bash
pip install opencv-python tqdm pyyaml requests
```

或使用 requirements.txt：
```bash
pip install -r requirements.txt
```

## 故障排除

### 下載失敗

如果自動下載失敗，請手動下載：

**OMR Layout Analysis**:
```
URL: https://github.com/v-dvorak/omr-layout-analysis/releases/tag/Latest
檔案: datasets-release.zip (~2.5 GB)
目標: external_barlines/omr_layout/
```

**AudioLabs v2**:
```
URL: https://www.audiolabs-erlangen.de/resources/MIR/2019-ISMIR-LBD-Measures
檔案: MeasureBoundingBoxAnnotations.zip (248 MB)
目標: external_barlines/audiolabs/
```

**DoReMi**:
```
URL: https://github.com/steinbergmedia/DoReMi/releases
檔案: DoReMi_1.0.zip
目標: external_barlines/doremi/
```

### 轉換錯誤

常見問題：

1. **找不到圖片**：檢查數據集是否正確解壓
2. **標註格式錯誤**：確認下載的是正確版本
3. **記憶體不足**：可以分批處理較大的數據集

### 檢查轉換結果

查看統計信息：
```bash
# 查看各數據集轉換統計
cat external_barlines/omr_layout/converted/conversion_stats.json
cat external_barlines/audiolabs/converted/conversion_stats.json
cat external_barlines/doremi/converted/conversion_stats.json

# 查看合併統計
cat external_barlines/merged/merge_stats.json
```

## 使用合併數據集訓練

```bash
cd /home/thc1006/dev/music-app/training
yolo detect train \
  data=/home/thc1006/dev/music-app/training/datasets/external_barlines/merged/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16
```

## 授權說明

- **OMR Layout Analysis**: MIT License
- **AudioLabs v2**: 學術使用
- **DoReMi**: 研究使用（需確認）

使用這些數據集訓練的模型權重不受數據集授權限制，但請遵守各自的使用條款。

## 更多資訊

詳細說明請參考：
- `../datasets/external_barlines/README.md` - 數據集詳細資訊
- `../CLAUDE.md` - 專案整體說明
