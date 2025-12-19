# OpenScore .mscx 渲染狀態報告

## 執行日期
2025-11-27

## 任務目標
將 OpenScore Lieder .mscx 文件轉換為 YOLO 格式的訓練數據，特別針對：
- fermata (class 29): 目標解決 91% miss rate
- barline (class 23-26): 目標解決 barline 系列檢測問題

## 當前進展

### ✅ 完成項目

1. **數據提取成功**
   - 處理了 1,356 個 .mscx 文件
   - 提取了 **5,823 個 fermata** 標註
   - 提取了 246,132 個 barline 標註
   - 輸出位置：`training/openscore_analysis/`

2. **Fermata 標註詳情**
   ```
   - fermataAbove: 5,238 (90.0%)
   - fermataBelow: 583 (10.0%)
   - fermataShortAbove: 2 (<0.1%)
   - 61.3% 的文件包含 fermata
   ```

3. **Barline 標註詳情**
   ```
   - normal: 244,308 (99.3%)
   - final: 1,356 (0.5%)
   - repeat_end: 277 (0.1%)
   - repeat_start: 191 (0.1%)
   ```

### ❌ 遇到的障礙

1. **無法直接渲染 .mscx 為帶座標的圖片**
   - Verovio 無法識別 .mscx 格式（需要 MusicXML/MEI）
   - music21 無法解析 .mscx 格式
   - 需要 MuseScore CLI，但系統無 sudo 權限安裝

2. **座標問題**
   - .mscx 中只有相對偏移量（offset），沒有絕對像素座標
   - 需要渲染引擎計算最終位置

## 可行方案

### 方案 A：使用 MuseScore CLI（推薦）✨

**工具**：MuseScore 3 命令行工具

**流程**：
```bash
# 1. 安裝 MuseScore3 (需要 sudo 或 Docker)
sudo apt-get install musescore3

# 2. 批量轉換 .mscx → PNG + MusicXML
mscore3 input.mscx -o output.png
mscore3 input.mscx -o output.musicxml

# 3. 使用 Verovio 解析 MusicXML 提取座標
python render_openscore_musicxml_to_yolo.py

# 4. 對齊 fermata/barline 位置生成 YOLO 標註
```

**優勢**：
- MuseScore 是 .mscx 原生工具，完美支持
- 可生成高質量 PNG 和 MusicXML
- Verovio 可精確解析 MusicXML 座標

**所需工具**：
- MuseScore3 CLI（需安裝）
- 現有 Python 環境（已有 verovio）

### 方案 B：使用 Docker + MuseScore（無 sudo 替代）

**流程**：
```bash
# 1. 使用 Docker 運行 MuseScore（無需 sudo）
docker run -v /path/to/scores:/scores linuxserver/musescore \
  mscore3 /scores/input.mscx -o /scores/output.png

# 2. 後續同方案 A
```

**優勢**：
- 無需系統級安裝
- 可在當前環境運行

**劣勢**：
- 需要 Docker 權限

### 方案 C：手動標註工具（保底方案）

如果無法自動渲染，可以：
1. 使用 MuseScore GUI 手動導出為 PNG
2. 使用標註工具（如 LabelImg）手動標註 fermata 和 barline
3. 基於提取的 JSON 數據加速標註流程

**不推薦原因**：
- 工作量巨大（1,356 文件 × 平均 5 fermata/文件 = ~7,000 標註）
- 但可作為小規模驗證用途

### 方案 D：暫時跳過 OpenScore，使用其他數據集

**替代數據源**：
- MUSCIMA++（已有 35 fermata，已轉換完成）
- 合成數據生成（Abjad + LilyPond，Phase 5 計劃）
- DeepScores V2（需下載，~7GB）

**建議**：
- 優先完成 Phase 4 訓練（MUSCIMA++ + Rebelo）
- 並行準備 Phase 5 合成數據管道
- OpenScore 作為 Phase 6 增強數據集

## 建議行動方案

### 立即執行（本次對話）

**選項 1**：請用戶提供 sudo 權限或安裝 MuseScore3
```bash
sudo apt-get update
sudo apt-get install -y musescore3
```

**選項 2**：使用 Docker 方案
```bash
docker pull linuxserver/musescore
# 然後執行批量轉換腳本
```

### 中期方案（本周）

如果無法使用 MuseScore：
1. **優先執行 Phase 4 訓練**（已有數據集準備完成）
   ```bash
   cd /home/thc1006/dev/music-app/training
   python yolo12_train_phase4.py
   ```

2. **啟動 Phase 5 合成數據準備**（Abjad + LilyPond）
   - 安裝 Abjad: `pip install abjad`
   - 參考：`training/docs/synthetic_data_generation_guide.md`

3. **OpenScore 數據整合延後至 Phase 6**
   - 在有 MuseScore 環境後再處理

## 當前輸出文件

### 已生成文件

1. **Fermata 標註 JSON**
   - 路徑：`training/openscore_analysis/fermatas.json`
   - 內容：5,823 個 fermata 的文件路徑、小節號、聲部號、類型
   - 用途：未來與渲染圖片對齊生成 YOLO 標註

2. **Barline 標註 JSON**
   - 路徑：`training/openscore_analysis/barlines.json`
   - 內容：246,132 個 barline 的類型和位置
   - 用途：未來生成 YOLO barline 標註

3. **提取統計**
   - 路徑：`training/openscore_analysis/extraction_stats.json`
   - 內容：完整統計信息

### 腳本文件

1. **extract_openscore_fermatas.py** ✅
   - 功能：從 .mscx 提取 fermata/barline 標註
   - 狀態：已完成並成功運行

2. **render_openscore_mscx_to_yolo.py** ⚠️
   - 功能：原計劃渲染 .mscx 為 YOLO 格式
   - 狀態：受阻（無 MuseScore CLI）
   - 需要：重構為 MusicXML-based pipeline

## 數據價值評估

### OpenScore Lieder vs MUSCIMA++

| 指標 | OpenScore Lieder | MUSCIMA++ |
|------|------------------|-----------|
| **Fermata 數量** | 5,823 | 35 |
| **倍數** | **166x** | 1x |
| **文件數** | 1,356 | 140 |
| **Barline 數量** | 246,132 | ~3,330 |
| **授權** | CC-0 ✅ | CC-BY-NC-SA |

**結論**：OpenScore Lieder 是 fermata 訓練的**關鍵數據集**，值得投入資源解決渲染問題。

## 下一步建議

### 立即決策需要（請用戶選擇）

**A. 立即解決 OpenScore 渲染**
- 需要：sudo 權限或 Docker 環境
- 時間：1-2 小時設置 + 2-3 小時批量轉換
- 收益：立即獲得 5,823 fermata 標註

**B. 延後 OpenScore，優先其他訓練**
- 立即執行：Phase 4 訓練（已有數據）
- 並行準備：Phase 5 合成數據（Abjad）
- Phase 6 再整合 OpenScore

**C. 混合方案**
- 小規模測試：手動導出 50-100 文件驗證流程
- 大規模處理：等待 MuseScore 環境

### 我的推薦（基於時間投資回報）

1. **本周立即執行**：Phase 4 訓練（RTX 5090 可用時）
2. **並行準備**：Phase 5 Abjad 合成數據環境
3. **下周解決**：OpenScore 渲染問題（與系統管理員協調 MuseScore 安裝）

**理由**：
- Phase 4 數據集已完整（24,566 圖片），可立即訓練
- Phase 5 合成數據可補充 fermata（預計生成 2,000-3,000）
- OpenScore 作為 Phase 6 大規模增強，時間充裕

## 技術債務記錄

1. **缺少 MuseScore CLI**：限制 .mscx 處理能力
2. **Verovio 無 .mscx 支持**：需要中間轉換步驟
3. **座標對齊算法**：JSON 標註 → 像素座標需要開發

## 聯繫人與資源

- **MuseScore 官方文檔**：https://musescore.org/en/handbook/3/command-line-options
- **Verovio Python Toolkit**：https://github.com/rism-digital/verovio
- **OpenScore Lieder**：https://github.com/OpenScore/Lieder

---

**報告生成時間**：2025-11-27
**處理文件數**：1,356 .mscx
**提取標註數**：251,955 (5,823 fermata + 246,132 barline)
**狀態**：數據提取完成，等待渲染工具決策
