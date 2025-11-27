# Verovio 合成數據生成系統

高質量的樂譜 barline 合成訓練數據生成系統，使用 Verovio 渲染引擎和先進的 domain randomization 技術。

## 功能特點

- **多種 Barline 類型**: single, double, final, repeat-left, repeat-right, repeat-both
- **高質量渲染**: 使用 Verovio 的 MEI 渲染引擎
- **精確標註**: 自動提取 SVG bounding boxes
- **Domain Randomization**: 紙張紋理、掃描噪聲、幾何變換
- **YOLO 格式輸出**: 直接可用於訓練

## 安裝

```bash
cd /home/thc1006/dev/music-app/training/synthetic_generation
pip install -r requirements.txt
```

## 快速開始

### 生成測試數據（1,000 張）

```bash
python generate_synthetic_barlines.py --num-images 1000 --output-dir output
```

### 生成完整數據集（100,000 張）

```bash
python generate_synthetic_barlines.py --num-images 100000 --output-dir output --workers 8
```

## 配置

編輯 `configs/generation_config.yaml` 來自定義生成參數：

- 圖像大小
- Barline 類型分佈
- Augmentation 參數
- 字體和樣式設置

## 目錄結構

```
synthetic_generation/
├── configs/              # 配置文件
├── src/                  # 核心模組
│   ├── verovio_renderer.py
│   ├── mei_generator.py
│   ├── bbox_extractor.py
│   ├── domain_randomization.py
│   └── yolo_exporter.py
├── templates/            # MEI 模板
├── output/              # 生成的數據
│   ├── images/          # PNG 圖像
│   ├── labels/          # YOLO 標註
│   └── validation/      # 可視化樣本
└── generate_synthetic_barlines.py  # 主執行腳本
```

## 初步測試結果

**測試日期**: 2025-11-26

**測試命令**:
```bash
python generate_synthetic_barlines.py --num-images 10 --output-dir output_test --workers 1 --validation-samples 5
```

**結果**:
- ✅ 成功安裝所有依賴（verovio 5.6.0, cairosvg, 等）
- ✅ MEI 生成器正常工作
- ✅ Verovio 渲染器成功生成 SVG/PNG
- ✅ 圖像輸出正常（1024x1024 PNG）
- ⚠️ **Bbox 提取需要修正**：座標值超出 [0, 1] 範圍
- ⚠️ Verovio 警告：需要優化 MEI 模板格式

**已知問題**:

1. **Bbox 座標異常**:
   - 當前：x_center = 2.617143（應該 < 1.0）
   - 原因：SVG 座標系統與像素座標系統轉換問題
   - 計劃：重寫 `bbox_extractor.py` 使用 Verovio 的原生 bbox API

2. **Verovio 警告**:
   - "No header found in MEI data"
   - "Unsupported data.PITCHNAME"
   - 計劃：更新 MEI 模板使用 Verovio 5.x 標準格式

3. **部分圖像生成失敗**:
   - 10 張中只有 1 張成功
   - 需要添加更好的錯誤處理

## 下一步行動

### 優先級 1: 修正 Bbox 提取（必須）

```python
# 選項 A: 使用 Verovio 的 getElementAttr() API
# 選項 B: 改進 SVG 解析邏輯
# 選項 C: 使用 LilyPond 代替（備選方案）
```

### 優先級 2: 優化 MEI 生成

- 添加完整的 MEI header
- 使用正確的 MEI 5.0 語法
- 減少 Verovio 警告

### 優先級 3: 擴展測試

- 測試所有 6 種 barline 類型
- 驗證 augmentation 效果
- 生成 1,000 張圖片進行完整測試

## 輸出格式

- **Images**: PNG 格式，默認 1024x1024
- **Labels**: YOLO 格式 (class x_center y_center width height)
- **Classes**: 0=single, 1=double, 2=final, 3=repeat-left, 4=repeat-right, 5=repeat-both

## 質量控制

系統自動進行質量驗證：
- 檢查 barline 是否正確渲染
- 驗證 bounding box 合理性
- 生成可視化樣本供檢查

## 性能

- 單核：~50 張/分鐘（預估）
- 8 核並行：~300 張/分鐘（預估）
- 預計 100,000 張需要約 5-6 小時（待驗證）

## 技術細節

### MEI 格式
使用 Music Encoding Initiative (MEI) XML 格式來精確定義樂譜結構。

### Verovio 渲染
Verovio 提供高質量的 SVG/PNG 渲染，支持多種樂譜字體。

### Domain Randomization
模擬真實世界變化：
- Perlin noise 紙張紋理
- 掃描線條和斑點
- 隨機旋轉和透視變換
- JPEG 壓縮偽影
- 亮度/對比度調整

## 故障排除

### Verovio 未安裝
```bash
pip install verovio
```

### 記憶體不足
減少 `--workers` 數量或 `--batch-size`

### 渲染錯誤
檢查 MEI 模板語法，確保 Verovio 版本 >= 4.0

### Bbox 座標異常
目前已知問題，正在修復中

## 替代方案

如果 Verovio 方案遇到瓶頸，可考慮：

1. **LilyPond + Abjad**:
   - 優點：完整的 Python API，成熟穩定
   - 缺點：需要安裝 LilyPond 系統包

2. **MuseScore + Music21**:
   - 優點：支持 MusicXML
   - 缺點：需要 GUI 軟體，較慢

3. **純 SMuFL Font 渲染**:
   - 優點：極快，直接控制
   - 缺點：需要手動排版邏輯

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 許可證

MIT License
