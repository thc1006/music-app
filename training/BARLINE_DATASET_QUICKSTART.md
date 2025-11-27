# Barline 數據集快速開始指南

這個指南將幫助您快速下載、轉換和使用外部 barline 數據集來增強 YOLO 訓練。

## 目錄

1. [概述](#概述)
2. [一鍵執行](#一鍵執行)
3. [手動步驟](#手動步驟)
4. [數據集詳情](#數據集詳情)
5. [訓練使用](#訓練使用)
6. [故障排除](#故障排除)

---

## 概述

我們將從 3 個外部 OMR 數據集提取 barline 標註：

| 數據集 | 圖片數 | Barline 來源 | 大小 |
|--------|--------|-------------|------|
| **OMR Layout Analysis** | 7,013 | measure 邊界 | ~2.5 GB |
| **AudioLabs v2** | 940 | measure 邊界 | 248 MB |
| **DoReMi** | ~5,218 | OMR XML | ~670 MB |

**預期產出**：
- 合併數據集包含數千張圖片
- 4 種 barline 類別（barline, double, final, repeat）
- YOLO 格式，可直接用於訓練

---

## 一鍵執行

最簡單的方式是使用自動化腳本：

```bash
# 1. 進入腳本目錄
cd /home/thc1006/dev/music-app/training/scripts

# 2. 給予執行權限
chmod +x run_all_barline_pipeline.sh

# 3. 執行完整流程
./run_all_barline_pipeline.sh
```

這個腳本會：
1. 檢查依賴套件
2. 下載所有數據集（約 3 GB）
3. 轉換為 YOLO 格式
4. 合併所有數據集
5. 生成訓練指令

**預計時間**：30-60 分鐘（取決於網速）

---

## 手動步驟

如果需要更多控制，可以手動執行各步驟：

### 步驟 1: 安裝依賴

```bash
cd /home/thc1006/dev/music-app/training
pip install opencv-python tqdm pyyaml requests
```

### 步驟 2: 下載數據集

**下載所有數據集**：
```bash
cd scripts
python download_all_datasets.py
```

**或選擇性下載**：
```bash
# 只下載 AudioLabs（最小，248 MB）
python download_all_datasets.py --datasets audiolabs

# 下載 OMR Layout + DoReMi
python download_all_datasets.py --datasets omr_layout doremi
```

### 步驟 3: 轉換數據集

依序轉換每個數據集：

```bash
# 轉換 OMR Layout Analysis
python convert_omr_layout.py

# 轉換 AudioLabs v2
python convert_audiolabs.py

# 轉換 DoReMi
python convert_doremi.py
```

每個轉換腳本會：
- 讀取原始標註
- 提取 barline 相關信息
- 生成 YOLO 格式標註
- 自動分割訓練/驗證集

### 步驟 4: 合併數據集

```bash
python merge_barline_datasets.py
```

合併後的數據集位於：
```
/home/thc1006/dev/music-app/training/datasets/external_barlines/merged/
```

### 步驟 5: 檢查結果

```bash
# 查看合併統計
cat ../datasets/external_barlines/merged/merge_stats.json

# 檢查數據集配置
cat ../datasets/external_barlines/merged/data.yaml
```

---

## 數據集詳情

### 1. OMR Layout Analysis

**來源**：https://github.com/v-dvorak/omr-layout-analysis

**特點**：
- 7,013 張樂譜圖片
- 已轉換為 YOLOv8 格式
- 包含 staff, measure 等標註

**Barline 提取策略**：
- 從 `system_measure` 和 `stave_measure` 的邊界提取
- 左右邊界各創建一條垂直 barline
- Barline 寬度設為圖片寬度的 0.5%

**預期輸出**：
- 約 6,000+ 訓練圖片
- 約 1,000+ 驗證圖片
- 主要是普通 barline

### 2. AudioLabs v2

**來源**：https://www.audiolabs-erlangen.de/resources/MIR/2019-ISMIR-LBD-Measures

**特點**：
- 940 張古典音樂樂譜
- 85,980 個 measure bounding boxes
- 包含 Wagner, Beethoven, Schubert 等作曲家

**Barline 提取策略**：
- 從 JSON 格式的 measure annotations 提取
- 自動偵測行尾（y 坐標差異大）使用 final barline
- 最後一個 measure 使用 final barline

**預期輸出**：
- 約 800 訓練圖片
- 約 140 驗證圖片
- 包含較多 final barline

### 3. DoReMi

**來源**：https://github.com/steinbergmedia/DoReMi

**特點**：
- ~5,218 張 Dorico 生成的樂譜
- ~1M 標註物件（94 個類別）
- OMR XML 格式，包含完整 bounding box

**Barline 提取策略**：
- 解析 OMR XML 的 Node 元素
- 直接提取 barline 相關類別
- 支持 barline, double, final, repeat 類型

**預期輸出**：
- 約 4,400+ 訓練圖片
- 約 800+ 驗證圖片
- **唯一包含 double/repeat barline 的數據集**

---

## 訓練使用

### 基本訓練

使用合併後的數據集訓練 barline 檢測模型：

```bash
cd /home/thc1006/dev/music-app/training

# 使用 YOLOv8n（輕量）
yolo detect train \
  data=/home/thc1006/dev/music-app/training/datasets/external_barlines/merged/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=barline_detection

# 使用 YOLOv8s（更高準確度）
yolo detect train \
  data=datasets/external_barlines/merged/data.yaml \
  model=yolov8s.pt \
  epochs=150 \
  imgsz=640 \
  batch=8 \
  name=barline_detection_s
```

### 與現有模型合併

如果您已經有一個訓練好的 OMR 模型，可以：

1. **微調現有模型**：
```bash
yolo detect train \
  data=datasets/external_barlines/merged/data.yaml \
  model=path/to/your/best.pt \
  epochs=50 \
  imgsz=640 \
  freeze=10  # 凍結前 10 層
```

2. **合併數據集**：
將 `external_barlines/merged` 與您現有的數據集合併，創建新的 `data.yaml`。

### 評估模型

```bash
# 在驗證集上評估
yolo detect val \
  model=runs/detect/barline_detection/weights/best.pt \
  data=datasets/external_barlines/merged/data.yaml

# 在測試圖片上推論
yolo detect predict \
  model=runs/detect/barline_detection/weights/best.pt \
  source=path/to/test/images
```

---

## 故障排除

### 下載失敗

**問題**：網路連接錯誤或 404

**解決**：
1. 檢查網路連接
2. 手動下載（見下方連結）
3. 使用 `--datasets` 參數跳過失敗的數據集

**手動下載連結**：
- OMR Layout: https://github.com/v-dvorak/omr-layout-analysis/releases/tag/Latest
- AudioLabs: https://www.audiolabs-erlangen.de/resources/MIR/2019-ISMIR-LBD-Measures
- DoReMi: https://github.com/steinbergmedia/DoReMi/releases

### 轉換錯誤

**問題**：找不到圖片或標註檔案

**解決**：
1. 確認數據集已正確解壓
2. 檢查目錄結構是否符合預期
3. 查看錯誤訊息中的路徑

**常見問題**：
```bash
# 檢查 OMR Layout 是否正確解壓
ls datasets/external_barlines/omr_layout/datasets-release/

# 檢查 AudioLabs 是否正確解壓
ls datasets/external_barlines/audiolabs/MeasureBoundingBoxAnnotations/

# 檢查 DoReMi 是否正確解壓
ls datasets/external_barlines/doremi/DoReMi_1.0/
```

### 記憶體不足

**問題**：轉換大型數據集時記憶體不足

**解決**：
1. 關閉其他程式
2. 使用 `--val-split` 減少驗證集比例
3. 分批處理數據集

### 類別不平衡

**問題**：barline 類別分佈不均

**觀察**：
- 普通 barline: ~90%
- final barline: ~8%
- double barline: ~1.5%
- repeat barline: ~0.5%

**解決策略**：
1. 訓練時使用類別加權
2. 過採樣稀有類別
3. 合成更多 double/repeat barline 樣本

---

## 進階功能

### 自定義類別映射

如果您需要不同的類別 ID，修改 `merge_barline_datasets.py` 中的映射：

```python
self.original_to_new = {
    23: 0,  # barline
    24: 1,  # barline_double
    25: 2,  # barline_final
    26: 3   # barline_repeat
}
```

### 調整驗證集比例

```bash
# 使用 20% 驗證集
python convert_omr_layout.py --val-split 0.20
python convert_audiolabs.py --val-split 0.20
python convert_doremi.py --val-split 0.20
```

### 選擇性合併

只合併部分數據集：

```bash
python merge_barline_datasets.py \
  --omr-layout datasets/external_barlines/omr_layout/converted \
  --audiolabs "" \
  --doremi datasets/external_barlines/doremi/converted
```

---

## 檢查清單

完成以下步驟確保一切正常：

- [ ] 依賴套件已安裝（opencv-python, tqdm, pyyaml, requests）
- [ ] 至少一個數據集已下載
- [ ] 轉換腳本執行成功，無錯誤
- [ ] `merge_stats.json` 顯示合理的統計數據
- [ ] `data.yaml` 包含正確的路徑和類別
- [ ] 訓練集和驗證集目錄包含圖片和標註
- [ ] 可以開始訓練

---

## 預期結果

完成所有步驟後，您應該得到：

```
external_barlines/merged/
├── images/
│   ├── train/         # ~11,000 張圖片
│   └── val/           # ~1,900 張圖片
├── labels/
│   ├── train/         # ~11,000 個標註檔
│   └── val/           # ~1,900 個標註檔
├── data.yaml          # YOLO 配置
└── merge_stats.json   # 統計信息
```

**統計範例**：
```json
{
  "total_images": 12900,
  "train_images": 10965,
  "val_images": 1935,
  "total_barlines": 258000,
  "class_distribution": {
    "0": 232200,  // barline
    "1": 3870,    // barline_double
    "2": 20640,   // barline_final
    "3": 1290     // barline_repeat
  }
}
```

---

## 下一步

1. **訓練基礎模型**：使用合併數據集訓練 barline 檢測
2. **與主模型整合**：將 barline 檢測加入完整的 OMR 模型
3. **數據增強**：如果某些類別仍然不足，考慮合成數據
4. **持續改進**：收集更多樂譜樣本，持續優化

---

## 支援與參考

- **詳細文檔**：`scripts/README.md`
- **數據集說明**：`datasets/external_barlines/README.md`
- **問題回報**：檢查各轉換腳本的 `conversion_stats.json`

祝訓練順利！
