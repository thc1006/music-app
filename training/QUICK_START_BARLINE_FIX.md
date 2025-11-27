# Barline 修復 - 快速開始指南

## 🎯 一句話說明

修復 Phase 5 中 barline 標註過細/過大的問題，生成 Phase 6 修復版數據集，預期 mAP50 從 0.615 提升至 0.65-0.68。

---

## ⚡ 3 步驟執行（5 分鐘）

```bash
# 1. 切換到訓練目錄
cd /home/thc1006/dev/music-app/training

# 2. 執行一鍵腳本
./run_fix_barline.sh

# 3. 檢查結果
tail -30 datasets/yolo_harmony_v2_phase6_fixed/fix_report.txt
```

**完成！** 現在可以開始 Phase 6 訓練了。

---

## 📚 文檔導航

不知道從哪裡開始？根據你的需求選擇：

### 我想快速執行 → `run_fix_barline.sh`
```bash
./run_fix_barline.sh
```
**適合**：只想修復數據，不關心細節

---

### 我想了解原理 → `BARLINE_FIX_SUMMARY.md`
**包含**：
- 問題背景與解決方案
- 修復邏輯詳解
- 預期效果
- 技術細節

**適合**：第一次使用，想全面了解

---

### 我想看詳細說明 → `BARLINE_FIX_README.md`
**包含**：
- 完整使用指南（8000+ 字）
- FAQ 常見問題
- 故障排除
- 下一步訓練指南

**適合**：遇到問題時查閱

---

### 我想按步驟檢查 → `BARLINE_FIX_CHECKLIST.md`
**包含**：
- 執行前檢查清單
- 執行步驟
- 執行後驗證
- 問題排查

**適合**：系統性執行時使用

---

### 我想修改代碼 → `fix_barline_annotations.py`
**代碼結構**：
- 修復參數（第 16-24 行）
- 主修復類（第 57+ 行）
- 修復邏輯（第 102-200 行）

**適合**：需要自定義修復邏輯

---

### 我想驗證邏輯 → `test_fix_barline.py`
```bash
python test_fix_barline.py
```
**包含**：15+ 個測試案例
**適合**：修改代碼後驗證正確性

---

## 🔍 文件快速參考

| 文件 | 用途 | 何時使用 |
|------|------|---------|
| `run_fix_barline.sh` | 一鍵執行 | 開始修復時 |
| `BARLINE_FIX_SUMMARY.md` | 總體概覽 | 第一次閱讀 |
| `BARLINE_FIX_README.md` | 詳細指南 | 深入了解時 |
| `BARLINE_FIX_CHECKLIST.md` | 檢查清單 | 系統執行時 |
| `fix_barline_annotations.py` | 修復腳本 | 需要修改時 |
| `test_fix_barline.py` | 測試腳本 | 驗證邏輯時 |

---

## 📊 修復效果預覽

### 數據修復

| 類別 | 問題 | 修復前 | 修復後 |
|------|------|--------|--------|
| barline | 極細線 | 34.4% | **0%** ✅ |
| barline_double | 過大框 | 67.8% | **< 10%** ✅ |
| barline_final | 過大框 | 95.9% | **< 20%** ✅ |

### 訓練性能預期

| 類別 | Phase 5 | Phase 6 目標 | 提升 |
|------|---------|-------------|------|
| barline mAP50 | 0.201 | **0.50-0.60** | +150-200% |
| barline 召回率 | 9% | **45-55%** | +400-500% |
| 整體 mAP50 | 0.615 | **0.65-0.68** | +6-11% |

---

## ⏱️ 時間規劃

| 任務 | 耗時 | 說明 |
|------|------|------|
| 運行修復腳本 | 5 分鐘 | 自動化執行 |
| 檢查驗證結果 | 10 分鐘 | 查看報告和圖表 |
| 準備 Phase 6 訓練 | 5 分鐘 | 創建訓練腳本 |
| **總計（修復）** | **20 分鐘** | — |
| Phase 6 訓練 | 4-6 小時 | RTX 5090 背景運行 |

---

## ✅ 執行後檢查（3 個命令）

```bash
# 1. 檢查文件數量
ls datasets/yolo_harmony_v2_phase6_fixed/train/images/*.png | wc -l  # 應顯示: 22393

# 2. 查看修復摘要
tail -30 datasets/yolo_harmony_v2_phase6_fixed/fix_report.txt

# 3. 驗證極細線已修復
grep "極細線" datasets/yolo_harmony_v2_phase6_fixed/fix_report.txt
# 應顯示: ✅ 極細線（寬度 < 0.01）: 0 (0.0%)
```

---

## 🚀 下一步

修復完成後：

### 選項 A: 立即訓練（推薦）
```bash
# 1. 創建訓練腳本
cp yolo12_train_phase5.py yolo12_train_phase6.py

# 2. 修改配置（使用編輯器）
# - data='datasets/yolo_harmony_v2_phase6_fixed/harmony_phase6_fixed.yaml'
# - project='harmony_omr_v2_phase6'

# 3. 啟動訓練
tmux new -s phase6
source venv_yolo12/bin/activate
python yolo12_train_phase6.py
```

### 選項 B: 手動檢查
```bash
# 下載可視化圖表到本地查看
scp user@server:/path/to/datasets/yolo_harmony_v2_phase6_fixed/*.png ./
```

---

## ❓ 常見問題速查

### Q: 修復會改變圖片嗎？
**A**: 不會，只修改標註文件（.txt）。

### Q: 原始數據會被覆蓋嗎？
**A**: 不會，輸出到獨立的 `phase6_fixed` 目錄。

### Q: 修復時間多久？
**A**: 約 5 分鐘（24,910 張圖片的標註）。

### Q: 如何確認修復成功？
**A**: 查看 `fix_report.txt`，極細線應該降至 0%。

### Q: 修復後一定要訓練嗎？
**A**: 建議訓練以驗證效果，但也可以先檢查報告和可視化。

---

## 🆘 遇到問題？

1. **運行測試**：`python test_fix_barline.py`
2. **查看詳細文檔**：`BARLINE_FIX_README.md`（FAQ 章節）
3. **查看檢查清單**：`BARLINE_FIX_CHECKLIST.md`（故障排除）

---

## 📞 支援資源

| 資源 | 位置 | 用途 |
|------|------|------|
| 問題根因分析 | `barline_analysis_report.txt` | 了解問題背景 |
| Phase 5 README | `datasets/yolo_harmony_v2_phase5/README.md` | 了解數據集 |
| 合成數據研究 | `SYNTHETIC_DATA_SUMMARY.md` | 如果需要更多數據 |

---

**準備好了嗎？開始修復！** 🚀

```bash
cd /home/thc1006/dev/music-app/training && ./run_fix_barline.sh
```

---

**文檔版本**: 1.0
**最後更新**: 2025-11-26
