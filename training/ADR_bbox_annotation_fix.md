# ADR-001: OpenScore / DoReMi 標註尺寸不一致修復計畫

**狀態**: Proposed
**日期**: 2026-02-26
**前置文件**: `training/reports/notehead_recall_analysis.md`

---

## 1. 背景與問題

Phase 5 nostem 模型 mAP50=0.821，但 notehead_filled Recall 僅 0.513。
根因分析確認這是**資料標註的結構性問題**，不是模型能力問題。

### 1.1 問題本質

Val set 混合了兩套**互不相容的標註標準**：

| 來源 | 圖數 | notehead 佔比 | bbox 語義 | bbox 大小 (at 1280) |
|------|------|--------------|-----------|---------------------|
| OpenScore (lg-*) | 314 (39%) | **80.6%** | notehead+stem+beam 群組框 | ~380×335 px |
| DoReMi+MUSCIMA | 490 (61%) | **19.4%** | 正確的個別 notehead | ~10×9 px |

OpenScore 的 notehead bbox 是 LilyPond glyph group 的外接矩形，包含了整個 stem+beam 結構。
同一和弦的 noteheads 共用相同 cx、不同高度，形成巢狀重疊結構（最大 IoU=0.986）。

### 1.2 量化影響

- NMS iou=0.7 下的 Recall 理論天花板 = **0.526**（實際 0.513，差 1.3pp）
- OpenScore 的 63,313 個 notehead GT 中，**37,236 個 (59%)** 被 NMS 數學性地抑制
- 同樣問題影響 beam (5.9x)、tie (5.9x)、barline (15.4x) 等類別
- Training set 有相同問題：模型同時接收「notehead=14px」和「notehead=380px」的矛盾梯度

### 1.3 交叉驗證結果

| # | 驗證項目 | 結果 |
|---|---------|------|
| 1 | 視覺化 OpenScore bbox | ✅ 確認框了整個系統區域 |
| 2 | Notehead bbox ≈ Stem bbox 大小 | ✅ 差異 <5% |
| 3 | NMS greedy 模擬 | ✅ 天花板 0.526 |
| 4 | Phase 5 bbox = Phase 8 bbox | ✅ 10/10 相同 |
| 5 | Chord column 巢狀結構 | ✅ max IoU=0.986 |
| 6 | Training set 同問題 | ✅ OpenScore=377px, DoReMi=14px |
| 7 | 其他類別同問題 | ✅ beam/tie/barline 差 4-15x |
| 8 | Top edge = notehead 真實位置 | ✅ 間距吻合音高結構 |
| 9 | 修正後 NMS 天花板 | ✅ 0.526 → **1.000** |

---

## 2. 決策

### 採用方案：OpenScore bbox 程式化裁剪 + DoReMi 保留

修正 OpenScore 所有受影響類別的 bbox，將巨框裁剪為只包含目標符號本身的小框。
DoReMi/MUSCIMA 標註正確，保留不動。

### 否決方案

| 方案 | 否決原因 |
|------|---------|
| 移除 OpenScore notehead 標註 | 損失 80.6% 的 notehead 訓練數據 |
| 移除 DoReMi | DoReMi 的標註是正確的，且貢獻了 beam/flag/rest 等多類別 |
| 調超參數（cls weight, focal gamma） | 這是資料問題，超參數無法突破 NMS 天花板 |
| 繼續訓練更多 epoch | 天花板在 0.526，再訓練無意義 |
| 手工重新標註 | 63K+ 標註不現實 |

---

## 3. 技術方案

### 3.1 核心修正邏輯

OpenScore 的 notehead bbox 有一個關鍵特性：**bbox 的 top edge 對應 notehead 的真實 y 位置**，
bbox 向下延伸包含了 stem。

修正公式：

```python
# 估算的真實 notehead 尺寸 (normalized, 基於和弦音符間距統計)
NOTEHEAD_W = 0.01180  # ~23px at 1960 width
NOTEHEAD_H = 0.00908  # ~25px at 2772 height

# 對每個 OpenScore notehead bbox:
old_top = cy - h/2    # bbox 頂邊 = notehead 真實 y 位置
new_cy = old_top + NOTEHEAD_H / 2
new_w = NOTEHEAD_W
new_h = NOTEHEAD_H
# cx 保持不變
```

**驗證數據**：
- 修正後 NMS@0.7 天花板：0.526 → **1.000** (+47.4pp)
- 修正後 bbox 大小：~23×25px (原始), ~11×12px (at 1280)
- 與 DoReMi notehead (~28×24px 原始, ~10×9px at 1280) 尺度一致

### 3.2 受影響類別分析

不只 notehead，其他類別也有同樣的巨框問題。但修正策略因類別而異：

| 類別 | OpenScore bbox | DoReMi bbox | 比率 | 修正策略 |
|------|---------------|-------------|------|---------|
| **notehead_filled** | 380px | 10px | 38x | ✅ 裁剪到 top edge + 固定大小 |
| **notehead_hollow** | ~380px | ~11px | 35x | ✅ 同上 |
| **beam** | 408px | 69px | 6x | ⚠️ 需要獨立分析（beam 本身就是長條形） |
| **tie** | 407px | 69px | 6x | ⚠️ 需要獨立分析 |
| **barline** | 295px | 19px | 16x | ⚠️ barline 本身就是垂直線，可能合理偏大 |
| **ledger_line** | 378px | N/A | - | ⚠️ 需要獨立分析 |
| **clef_treble** | 128px | 29px | 4.5x | ⚠️ 比率較小，可能部分合理 |

**Phase 1 只修 notehead_filled 和 notehead_hollow**（最確定、影響最大）。
其他類別留到 Phase 2 獨立分析後再處理。

### 3.3 資料集處理流程

```
Phase 8 Final (33 classes, 原始標註)
  │
  ├── 1. 識別 OpenScore 圖片 (filename contains 'lg-')
  │
  ├── 2. 對 OpenScore 圖片的 notehead bbox 執行裁剪
  │     └── top_edge → notehead cy, 固定 w/h
  │
  ├── 3. 移除 stem annotations (class 2)
  │
  ├── 4. Remap class IDs (33 → 32)
  │
  └── 5. 輸出 Phase 6 dataset
        ├── train: 原始 32K 圖 (OpenScore bbox 已修正)
        └── val: cleaned_v2 2,867 圖 (OpenScore bbox 已修正)
```

### 3.4 風險與緩解

| 風險 | 嚴重度 | 緩解措施 |
|------|--------|---------|
| Top edge 假設錯誤（stem 在上方時 notehead 在底部） | 高 | 視覺化驗證 50+ 張圖；用 stem bbox 位置判斷 stem direction |
| 固定 notehead 大小不適用所有 OpenScore 字體 | 中 | 按字體 (beethoven/gonville/emmentaler/gutenberg) 分別統計 |
| 修正後仍有 DoReMi 10px notehead 偵測困難 | 低 | 這是物理限制，不影響修正的價值 |
| 改了 train+val 標註需要完全重訓練 | 低 | 本來就要重訓練，成本已計入 |

### 3.5 驗證計畫

修正完成後，在重訓練前先用以下方式驗證：

1. **視覺化抽檢**：隨機抽取 50 張 OpenScore 圖，畫修正後的 bbox，人工確認
2. **NMS 天花板驗證**：在修正後的 GT 上重跑 greedy NMS，確認天花板 ≈ 1.0
3. **統計一致性**：修正後 OpenScore 和 DoReMi 的 notehead bbox 尺寸分佈應趨近
4. **現有模型 eval**：用現有 Phase 5 模型在修正後的 val set 上跑 eval，看 recall 變化

---

## 4. 實作計畫

### Phase 1: Notehead bbox 修正 + 重訓練（預計 2 天）

| 步驟 | 工作 | 預估時間 |
|------|------|---------|
| 1.1 | 寫 `create_phase6_fixed_bbox.py` 腳本 | 2h |
| 1.2 | 處理 stem direction 判斷邏輯（用 Phase 8 的 stem bbox 位置） | 1h |
| 1.3 | 視覺化驗證 50 張圖 | 1h |
| 1.4 | 生成 Phase 6 dataset | 30min |
| 1.5 | NMS 天花板驗證 | 30min |
| 1.6 | 用現有 Phase 5 模型在 Phase 6 val 上 eval | 30min |
| 1.7 | Phase 6 訓練（two-stage, ~200 epochs） | 12-15h |
| 1.8 | 結果對比（Phase 5 vs Phase 6） | 1h |

### Phase 2: 其他類別修正（Phase 1 驗證成功後）

- 獨立分析 beam / tie / barline / ledger_line / clef 的 bbox 問題
- 各類別可能需要不同的修正策略（beam 是長條形、barline 是垂直線等）
- 每個類別需要獨立的視覺化驗證

### Phase 3: DoReMi 優化（可選）

- 如果 10px notehead 仍然拖低 recall：
  - 方案 A：從 val set 移除 DoReMi（metrics 更乾淨但代表性降低）
  - 方案 B：將 DoReMi 圖切成更小的 tiles（notehead 變大但丟失上下文）
  - 方案 C：接受現狀（DoReMi 只佔 19%，影響有限）

---

## 5. 成功標準

| 指標 | Phase 5 當前 | Phase 6 目標 | 備註 |
|------|-------------|-------------|------|
| notehead_filled Recall | 0.513 | **> 0.70** | 消除 NMS 天花板後 |
| 整體 Recall | 0.717 | **> 0.75** | notehead 佔 35% annotations |
| 整體 mAP50 | 0.821 | **> 0.85** | notehead + 其他類別連帶改善 |
| NMS 天花板 (notehead) | 0.526 | **≈ 1.0** | 修正後的驗證指標 |
| OpenScore/DoReMi bbox 尺寸比 | 38x | **< 2x** | 標註一致性 |

---

## 6. 附件與參考

- `/tmp/notehead_bbox_openscore.png` — OpenScore 巨框視覺化
- `/tmp/notehead_bbox_doremi.png` — DoReMi 正確標註視覺化
- `/tmp/chord_column_bboxes.png` — 和弦 column 巢狀 bbox 結構
- `training/reports/notehead_recall_analysis.md` — 完整根因分析報告
- `training/analysis_notehead_*.py` — 分析腳本（由背景代理生成）
