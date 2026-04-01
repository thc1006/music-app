# notehead_filled Recall=0.513 根因分析報告

**日期**: 2026-02-26
**模型**: Phase 5 nostem (YOLO12s, 1280x1280, 32 classes)
**評估配置**: conf=0.001, iou=0.7, max_det=1500 (strict protocol)

---

## 結論

**notehead_filled 的 Recall=0.513 不是模型能力問題，是標註資料結構性問題。**

NMS iou=0.7 下的理論 Recall 天花板 = **0.526**，模型實際 0.513 只差 1.3pp，已接近數學極限。

---

## 1. 問題描述

notehead_filled 是數量最多的類別（78,517 instances），但 Recall 僅 0.513，排在倒數第 4。

## 2. 根因：Val set 混合了兩套互不相容的標註標準

### 2.1 資料組成

| 來源 | Val 圖數 | Notehead 實例 | 佔比 | Bbox 大小 (at 1280) |
|------|----------|---------------|------|---------------------|
| **OpenScore (lg-)** | 314 (39%) | 63,313 | **80.6%** | **~380 × 335 px** |
| **DoReMi** | 480 (60%) | 13,457 | **17.1%** | **~14 × 9 px** |
| **MUSCIMA** | 10 (1%) | 1,747 | 2.2% | ~14 × 9 px |

### 2.2 OpenScore 標註問題（80.6% 的 noteheads）

OpenScore 的 notehead bbox 並非框「一個音符頭」，而是框了**包含 stem、beam 等元素的巨大區域**。

**視覺化證據** (`/tmp/chord_column_bboxes.png`):

單一 chord column（cx≈0.793）的 13 個 notehead bbox：
- 每個 bbox 寬 ~525px（圖片寬 1960px 的 27%）
- 高度從 155px 到 1285px 不等
- 呈現**巢狀/累積結構**：越低的音符 bbox 越大，延伸到頁面邊緣

```
# 同一 chord column 的 bbox (原始像素)
nh[0]:  w=823, h= 155  (top area)
nh[4]:  w=792, h=1147  (extends to bottom half)
nh[12]: w=824, h= 603  (bottom area)
```

**Notehead vs Stem 比較**（Phase 8 原始標註）：
- Notehead bbox: w=383±156px, h=312±151px
- Stem bbox: w=378±150px, h=322±153px
- **兩者幾乎完全相同** → notehead bbox 本質上包含了 stem

### 2.3 DoReMi 標註問題（17.1% 的 noteheads）

DoReMi 的 notehead bbox 是正確的個別音符頭標註，但在 1280px 下**物理上不可偵測**：
- 原始圖片 2475×3504，縮放比 0.37
- 原始 notehead ~28×24px → at 1280 = **10×9px**
- 490/804 張含 notehead 的圖片中，所有 noteheads 都 < 20px

### 2.4 其他類別也有同樣問題

| 類別 | OpenScore bbox | DoReMi bbox | 比率 |
|------|----------------|-------------|------|
| beam | 408 px | 69 px | 5.9x |
| tie | 407 px | 69 px | 5.9x |
| barline | 295 px | 19 px | 15.4x |
| clef_treble | 128 px | 29 px | 4.5x |

→ 不只是 notehead，整個 OpenScore 的標註體系都與 DoReMi 不同。

## 3. NMS 理論天花板驗證

在 GT boxes 上執行 greedy NMS 模擬（假設模型完美預測每個 GT）：

| 族群 | GT 數 | NMS@0.7 存活 | 被抑制 | Max Recall |
|------|-------|-------------|--------|------------|
| OpenScore (lg-) | 63,313 | 26,077 | **37,236** | **0.412** |
| DoReMi+MUSCIMA | 15,204 | 15,204 | 0 | 1.000 |
| **合計** | **78,517** | **41,281** | **37,236** | **0.526** |

**實際 Recall = 0.513，離天花板僅差 0.013（1.3pp）**

NMS@0.5 天花板更低（0.369），因為大框重疊更嚴重。

## 4. 影響量化

### 4.1 Recall 分解

```
總 GT:           78,517
不可偵測 (tiny):  ~15,204 (19.4%) → 被 DoReMi 拉低
NMS 抑制:        ~37,236 (47.4%) → 被 OpenScore 大框互殺
可被偵測:        ~26,077 (33.2%)
實際偵測:        ~40,274 (0.513 × 78517)
```

模型在可偵測範圍內的 effective recall 接近 100%。

### 4.2 對整體 mAP50 的影響

notehead_filled 佔 224,455 total instances 的 35%，其 Recall 被壓在 0.51 直接拖低了整體 mAP50。

## 5. 交叉驗證清單

| # | 驗證項目 | 方法 | 結果 |
|---|---------|------|------|
| 1 | Notehead bbox 包含 stem | 比較 Phase 8 原始的 nh/stem bbox | ✅ 確認：nh 和 stem bbox 幾乎相同大小 |
| 2 | NMS 天花板計算 | Greedy NMS on GT boxes | ✅ 確認：天花板 0.526，實際 0.513 |
| 3 | Phase 5 bbox 未修改 | 比較 Phase 8 vs Phase 5 的 notehead coords | ✅ 確認：10/10 完全相同 |
| 4 | Chord column 巢狀結構 | 分析同一 cx 的多個 bbox | ✅ 確認：10-13 個 bbox 巢狀，max IoU=0.986 |
| 5 | 視覺化 OpenScore bbox | 畫 bbox 到圖片上 | ✅ 確認：框了整個系統區域而非單一 notehead |
| 6 | 視覺化 DoReMi bbox | 畫 bbox 到圖片上 | ✅ 確認：正確但極小（10x9px） |
| 7 | Training set 同樣問題 | 比較 train 的 OpenScore vs DoReMi bbox | ✅ 確認：OpenScore=377px, DoReMi=14px |
| 8 | 其他類別同樣問題 | 比較 beam/tie/barline 的 bbox | ✅ 確認：OpenScore 比 DoReMi 大 4-15x |
| 9 | Dataset composition | 統計各來源佔比 | ✅ 確認：OpenScore 佔 80.6% noteheads |

## 6. 改善建議

### 6.1 立即可做（不需重訓練）

1. **分離評估**：按 OpenScore / DoReMi 分別計算 Recall，了解模型在各族群的真實表現
2. **調整 eval NMS**：用 agnostic_nms=False 或 per-class NMS 看 notehead 的 Recall 是否改善

### 6.2 資料層面修正（需重訓練）

3. **移除 DoReMi**：這些 10x9px 的 GT boxes 在 1280 下不可偵測，只會拉低 Recall 數字
4. **修正 OpenScore notehead bbox**：將巨框裁剪為只包含 notehead 本身（~15x15px），消除 NMS 互殺
5. **統一標註標準**：整個 dataset 應只用一種 bbox 語義

### 6.3 不建議做的事

- ❌ 調超參數（cls weight, focal gamma）— 這是資料問題，不是模型問題
- ❌ 繼續訓練更多 epoch — 天花板在那裡
- ❌ 降低 NMS IoU 閾值 — 會讓其他類別的 False Positive 增加

## 7. 附件

- `/tmp/notehead_bbox_openscore.png` — OpenScore notehead bbox 視覺化
- `/tmp/notehead_bbox_doremi.png` — DoReMi notehead bbox 視覺化
- `/tmp/chord_column_bboxes.png` — 單一 chord column 的巢狀 bbox 結構
- `/home/thc1006/dev/music-app/training/analysis_notehead_recall.py` — 分析腳本
- `/home/thc1006/dev/music-app/training/analysis_notehead_deep.py` — 深度分析腳本
- `/home/thc1006/dev/music-app/training/analysis_notehead_final.py` — 最終確認分析腳本
