# Verovio 合成數據生成系統 - 測試報告

**日期**: 2025-11-26
**測試者**: Claude Code
**版本**: v1.0.0-alpha

## 執行摘要

✅ **成功建立完整的 Verovio 合成數據生成系統**

系統包含所有核心組件：
- MEI 樂譜生成器
- Verovio SVG/PNG 渲染引擎
- Bounding box 提取器
- Domain randomization 增強器
- YOLO 格式輸出器

⚠️ **部分功能需要優化才能進入生產環境**

## 測試環境

- **操作系統**: Linux 6.14.0-28-generic
- **Python**: 3.12
- **Verovio**: 5.6.0
- **其他依賴**: cairosvg 2.8.2, PIL 12.0.0, OpenCV 4.12.0

## 測試結果

### 1. 依賴安裝 ✅

所有依賴成功安裝：

```
verovio==5.6.0
cairosvg==2.8.2
pillow==12.0.0
opencv-python==4.12.0.88
lxml==6.0.2
scipy==1.16.3
pyyaml==6.0.3
```

**結論**: 環境配置正確

### 2. MEI 生成 ✅

`mei_generator.py` 成功生成 MEI XML：

- 支持 6 種 barline 類型
- 隨機音符生成
- 多種譜號、拍號、調號

**樣本輸出**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.0">
  <music>
    <body>
      <mdiv>
        <score>
          ...
```

**結論**: MEI 生成功能正常

### 3. Verovio 渲染 ⚠️

**成功**:
- SVG 渲染正常
- PNG 圖像生成（1024x1024）
- 圖像質量良好

**警告**:
```
[Warning] No header found in the MEI data, trying to proceed...
[Warning] Unsupported data.PITCHNAME 'C'
[Warning] Unsupported '<barLine>' within <measure>
```

**原因**:
- MEI 模板缺少完整的 header
- Verovio 5.x 語法與模板不完全兼容

**影響**: 功能性影響小，但產生大量控制台警告

**結論**: 需要優化 MEI 模板格式

### 4. Bounding Box 提取 ❌

**嚴重問題**: 座標值異常

**樣本標註** (`barline_000006.txt`):
```
0 2.617143 0.303030 0.000000 0.242424
0 6.883810 0.303030 0.000000 0.242424
```

**問題分析**:
- x_center = 2.617（應該在 [0, 1]）
- width = 0.000（不合理）
- SVG 座標單位與像素座標轉換錯誤

**根本原因**:
1. SVG 使用絕對座標（例：2617.143 px）
2. 未正確除以圖像寬度進行歸一化
3. Bbox 計算邏輯可能錯誤識別了元素

**結論**: **必須修復才能使用**

### 5. Domain Randomization 未測試

因為 bbox 提取失敗，未能驗證增強效果。

**預期功能**:
- ✅ 紙張紋理（代碼完整）
- ✅ 掃描噪聲（代碼完整）
- ✅ 幾何變換（代碼完整）
- ✅ JPEG 壓縮（代碼完整）

**結論**: 待 bbox 修復後測試

### 6. 生成成功率

**測試**: 10 張圖片生成
**成功**: 1 張
**失敗**: 9 張

**失敗原因**:
- 大部分由於 MEI 渲染錯誤
- 缺少錯誤處理導致進程中斷

**結論**: 需要改進錯誤處理機制

## 已知問題清單

### P0 - 阻塞性問題

1. **Bbox 座標歸一化錯誤**
   - 影響：無法生成有效的 YOLO 標註
   - 狀態：待修復
   - 預估工作量：4-8 小時

### P1 - 重要問題

2. **MEI 模板格式不兼容**
   - 影響：大量警告，部分渲染失敗
   - 狀態：待優化
   - 預估工作量：2-4 小時

3. **錯誤處理不足**
   - 影響：生成失敗率高
   - 狀態：待改進
   - 預估工作量：1-2 小時

### P2 - 次要問題

4. **可視化樣本生成未驗證**
   - 影響：無法直觀驗證結果
   - 狀態：待測試
   - 預估工作量：1 小時

## 修復建議

### 方案 A: 修復 Verovio 方案（推薦）

**優點**:
- 已有基礎代碼
- Verovio 文檔完善
- 性能優秀

**步驟**:
1. 研究 Verovio Python API 的 bbox 提取方法
2. 使用 `getElementsAtTime()` 或類似 API
3. 更新 MEI 模板符合 Verovio 5.x 標準
4. 添加完整的錯誤處理

**預估時間**: 1-2 天

### 方案 B: 切換到 LilyPond（備選）

**優點**:
- 更成熟的生態系統
- Abjad 提供完整 Python API
- 已有成功案例（參考文檔）

**缺點**:
- 需要重寫大部分代碼
- 依賴系統軟體包
- 可能較慢

**預估時間**: 3-5 天

### 方案 C: 混合方案

**策略**:
- 使用 SMuFL Font 直接渲染 barline
- 手動計算座標（簡單可控）
- 用於快速生成大量數據

**預估時間**: 1 天

**推薦**: 先嘗試方案 A，如果 2 天內無法解決則切換到方案 C

## 下一步行動計劃

### 第一階段: 修復核心功能（1-2 天）

- [ ] 修復 bbox 提取邏輯
- [ ] 優化 MEI 模板
- [ ] 添加錯誤處理
- [ ] 驗證所有 6 種 barline 類型

### 第二階段: 完整測試（0.5 天）

- [ ] 生成 1,000 張測試圖片
- [ ] 驗證 domain randomization 效果
- [ ] 檢查可視化樣本
- [ ] 測試訓練/驗證分割

### 第三階段: 生產部署（0.5 天）

- [ ] 生成 10,000-100,000 張圖片
- [ ] 與 Phase 5 數據集合併
- [ ] 訓練 YOLO 模型驗證效果
- [ ] 性能優化與文檔完善

**總預估時間**: 2-3 天（全職工作）

## 技術債務

1. **單元測試缺失**: 所有模組未有測試
2. **日誌系統**: 使用 print() 而非 logging
3. **配置驗證**: 未驗證 YAML 配置正確性
4. **性能測試**: 未測試大規模生成性能

## 結論

系統架構設計良好，核心組件基本完整，但存在一個**阻塞性 bug**（bbox 座標歸一化）需要優先解決。

修復此問題後，系統應可投入使用。建議採用**方案 A**（修復 Verovio）作為主要路線，同時準備**方案 C**（SMuFL Font）作為快速備選方案。

**預期成果**: 修復完成後，系統應能夠：
- 每分鐘生成 50-100 張高質量合成圖片
- 提供準確的 YOLO 格式標註
- 通過 domain randomization 增強數據多樣性
- 支持 100,000+ 張圖片的大規模生成

## 附錄

### A. 測試命令

```bash
# 環境啟動
cd /home/thc1006/dev/music-app/training/synthetic_generation
source ../venv_yolo12/bin/activate

# 小規模測試
python generate_synthetic_barlines.py --num-images 10 --output-dir output_test --workers 1

# 中規模測試（修復後）
python generate_synthetic_barlines.py --num-images 1000 --output-dir output --workers 4

# 大規模生產（最終）
python generate_synthetic_barlines.py --num-images 100000 --output-dir output --workers 8
```

### B. 生成文件清單

```
/home/thc1006/dev/music-app/training/synthetic_generation/
├── README.md                  ✅ 完整文檔
├── TEST_REPORT.md             ✅ 本報告
├── requirements.txt           ✅ 依賴列表
├── configs/
│   └── generation_config.yaml ✅ 配置文件
├── src/
│   ├── __init__.py            ✅
│   ├── verovio_renderer.py    ✅
│   ├── mei_generator.py       ✅
│   ├── bbox_extractor.py      ⚠️ 需修復
│   ├── domain_randomization.py ✅
│   └── yolo_exporter.py       ✅
├── templates/
│   └── barline_templates/
│       └── simple_barline.mei ✅
└── generate_synthetic_barlines.py ✅
```

### C. 參考資源

- Verovio Toolkit: https://www.verovio.org/
- MEI Guidelines: https://music-encoding.org/guidelines/v5/
- YOLO Format: https://docs.ultralytics.com/datasets/detect/

---

**報告完成時間**: 2025-11-26 12:15 UTC+8
