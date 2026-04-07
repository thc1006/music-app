# 實驗紀錄：Phase 8/9 Pseudo-Label 與 Phase B Downstream Pipeline

**日期**: 2026-04-04 ~ 2026-04-07
**狀態**: ✅ Phase 8/9 訓練完成，Phase B downstream pipeline 完成，待進 Phase C (pitch accuracy)

---

## 1. 動機

Phase 6 修正 OpenScore notehead bbox (380×335px → 23×25px) 後，模型在 OpenScore 上**預測 0 個 notehead**。
DoReMi/MUSCIMA 上正常 (142-378/image)。根因：23×25px 在 1280 推論時只有 11×12px，太小，TAL 無法分配正樣本。

Phase 7 嘗試通用規則修正 30 class → 26/32 class 更差。

**本實驗**：用 Phase 5 模型預測 OpenScore 圖片（輸出 glyph-group 大框），後處理縮小為 40-120px pseudo-labels，取代原始標註重新訓練。

---

## 2. 方法

### 2.1 Pseudo-Label Pipeline

```
Phase 5 best.pt (混合數據訓練，認識 OpenScore 視覺特徵)
  → 對 OpenScore train+val 推論 (conf=0.10, iou=0.55, imgsz=1280)
  → 大框 (~500×600px) 經 shrink_box() 後處理
  → 輸出 YOLO 格式 pseudo-labels
```

**shrink_box() 設計**：
- 每個 class 有獨立的 target size（DoReMi reference × 3-6 倍）
- Notehead target: 0.050×0.040 normalized (~64×51px at inference)
- ±30% 隨機 jitter 避免所有 box 相同大小（Phase 6 教訓）
- Notehead 位置 top-biased（center 向上移 25%，validated in Phase 6）
- 邊界 clamping: box 不超出 [0, 1]

**代碼**: `training/create_pseudo_labels.py`
**測試**: `training/test_pseudo_label_pipeline.py` (22/22 passed, TDD)

### 2.2 Dataset

| | Train | Val |
|---|---|---|
| 總圖片 | 32,555 | 2,867 |
| DoReMi (原始 labels) | 29,951 | 2,550 |
| OpenScore (pseudo-labels) | 2,604 | 317 |
| 總標註 | 1,098,407 | 121,133 |
| OpenScore pseudo 標註 | 637,098 | 76,134 |
| Classes | 32/32 | 32/32 |

**YAML**: `datasets/yolo_harmony_v2_pseudo_v1/harmony_pseudo_v1.yaml`

### 2.3 Notehead Box Size 對比

| 來源 | Norm W | Norm H | 推論尺寸 | 備註 |
|------|--------|--------|---------|------|
| Phase 6 fixed (失敗) | 0.012 | 0.009 | 11×12px | TAL 無法學習 |
| DoReMi GT (原始) | 0.011 | 0.007 | 10×6px | 在 DoReMi 上有效 |
| **Pseudo-label** | **0.050** | **0.040** | **64×51px** | TAL 可學範圍 |
| Phase 5 prediction (原始) | 0.255 | 0.203 | 326×260px | glyph-group 大框 |

### 2.4 已知限制

1. **覆蓋率 ~30%**: Phase 5 在 conf=0.10 下只偵測到原始 GT 30-40% 的 noteheads
2. **Size gap 4-6×**: DoReMi noteheads (0.011) vs OpenScore pseudo (0.050)，同一 class 差異大
3. **位置近似**: shrink_box 位置基於 Phase 5 glyph-group 預測中心 + top-bias，非精確 notehead 位置
4. **DoReMi-only teacher 失敗**: 訓練了 DoReMi teacher (mAP50=0.874 on DoReMi val)，但在 OpenScore 上預測 0 noteheads — domain gap 太大

### 2.5 Training Config

| 參數 | Stage 1 | Stage 2 |
|------|---------|---------|
| Base model | Phase 5 best.pt | Stage 1 best.pt |
| cv2 reset | ✅ Kaiming init (639K params) | — |
| freeze | 10 (backbone) | 0 (全 unfreeze) |
| lr0 | 0.0003 | 0.0001 |
| epochs | 60 | 120 |
| patience | 30 | 40 |
| batch | 6 | 6 |
| box loss weight | 7.5 | 5.0 |
| cls | 1.0 | 1.0 |
| amp | False | False |
| compile | True | True |
| TF32 | True | True |
| mosaic/mixup/copy_paste | 0.0 | 0.0 |
| max_det | 1500 | 1500 |
| MAX_BOXES (TAL cap) | 400 | 400 |
| TAL_TOPK | 13 | 13 |

**腳本**: `training/train_phase8_pseudo.py`
**輸出**: `runs/detect/harmony_omr_v2_phase8/pseudo_v1/` (Stage 1), `pseudo_v1_stage2/` (Stage 2)

---

## 3. 結果 (在 pseudo-label val set 上)

### 3.1 Stage 1 (60 epochs, 完成 ✅)

| Epoch | mAP50 | mAP50-95 | P | R | box_loss | cls_loss |
|-------|-------|----------|-------|-------|----------|----------|
| 1 | 0.327 | 0.215 | 0.622 | 0.286 | 2.225 | 3.434 |
| 10 | 0.491 | 0.330 | 0.675 | 0.430 | 1.212 | 1.945 |
| 20 | 0.533 | 0.350 | 0.705 | 0.466 | 1.133 | 1.773 |
| 30 | 0.547 | 0.357 | 0.722 | 0.474 | 1.096 | 1.698 |
| 40 | 0.555 | 0.361 | 0.724 | 0.480 | 1.077 | 1.660 |
| 50 | 0.561 | 0.364 | 0.721 | 0.486 | 1.061 | 1.627 |
| **60** | **0.565** | **0.366** | 0.714 | 0.493 | 1.062 | 1.626 |

- 訓練時間: 23,598s (~6.6 小時)
- cv2 reset 恢復正常 (mAP50: 0.327 → 0.565)
- ep40 後趨於 plateau

### 3.2 Stage 2 (進行中, ep25/120)

| Epoch | mAP50 | mAP50-95 | P | R | box_loss | cls_loss |
|-------|-------|----------|-------|-------|----------|----------|
| 1 | 0.521 | 0.331 | 0.655 | 0.471 | 0.860 | 2.114 |
| **2 (best)** | **0.563** | **0.364** | 0.718 | 0.492 | 0.747 | 1.752 |
| 10 | 0.562 | 0.361 | 0.706 | 0.493 | 0.715 | 1.630 |
| 15 | 0.561 | 0.360 | 0.705 | 0.494 | 0.712 | 1.615 |
| 20 | 0.561 | 0.360 | 0.701 | 0.494 | 0.703 | 1.590 |
| 25 | 0.560 | 0.360 | 0.712 | 0.489 | 0.704 | 1.590 |

- **完全 plateau**: best 仍在 ep2 (0.563), 23 epochs 無改善
- patience=40, 預計 ~ep42 early stop
- 訓練時間: 18,049s (~5.0 小時) so far

---

## 4. 分析

### 4.1 mAP50=0.56 低於 Phase 6 的 0.82 — 但不可直接比較

Phase 6 val set 使用人工標註 GT (精確但 notehead bbox 太小導致失敗)。
Phase 8 val set 使用 pseudo-labels (noisy, 覆蓋率 ~30%)。
**真實性能需要用原始 GT val set 做 per-class 評估。**

### 4.2 Stage 2 Plateau 原因推測

1. **Pseudo-label 品質上限**: noisy labels 限制了模型能學到的最佳 mAP50
2. **覆蓋率不足**: 70% notehead 缺失 → 模型看到未標註的 notehead 被當作 background → 壓制 recall
3. **Size gap**: DoReMi (10px) vs OpenScore (64px) 的 4-6× 差異增加學習難度

### 4.3 關鍵問題（待 eval 回答）

- [ ] OpenScore notehead recall 是否從 0 提升？(Phase 6 = 0)
- [ ] 其他 30 class 是否退步？
- [ ] 模型預測的 bbox 大小是否合理？

---

## 5. Stage 2 結論 (2026-04-05)

**Stage 2 完全 plateau。** Best 在 ep2 (mAP50=0.5631)，之後 27+ epochs 無改善。預計 ~ep42 early stop。

### 5.1 Phase 8 最終判定：方法論失敗

mAP50=0.56 (pseudo-val) 已是此方法的上限。根因不在程式碼，在方法本身：

**根因 1：stem 方向導致 50% notehead pseudo-label 位置錯誤**

```
音樂記譜規則：
  符桿朝下 (note above middle line) → notehead 在 glyph-group 頂部 ← 我們的 top-bias ✓
  符桿朝上 (note below middle line) → notehead 在 glyph-group 底部 ← 我們的 top-bias ✗

shrink_box() 統一用 top-bias (cy - h*0.25)
→ 約 50% noteheads 的 pseudo-label 放在了錯誤的一端
→ 這些錯誤標註主動教模型「不要在這裡預測 notehead」
```

**根因 2：70% notehead 無標註 → 反向訓練**

Phase 5 at conf=0.10 只偵測 ~30% noteheads。缺失的 70% 成為 negative training signal。

**根因 3：跟 Phase 6/7 本質相同 — 用規則猜位置**

Phase 6: 規則縮 bbox → 位置太小
Phase 7: 規則移 bbox → 位置錯誤  
Phase 8: 規則 shrink + top-bias → 50% 位置錯誤

三次失敗的共同原因：**我們不知道 notehead 在 glyph-group 中的精確位置，規則無法解決此問題。**

### 5.2 Phase 7 vs Phase 8 失敗對比

| | Phase 7 | Phase 8 |
|---|---|---|
| 策略 | 規則修全部 30 class bbox | Phase 5 teacher + 規則 shrink |
| 規則 | TOP edge + CENTER | top-bias 25% + jitter |
| 驗證指標 | NMS ceiling (100%) | 視覺驗證 + TDD |
| 結果 | 26/32 更差 | plateau 0.56 |
| 共同失敗原因 | 規則位置不正確 | 規則位置不正確 |
| 差異 | 直接改 GT → 災難性退步 | 改 pseudo-label → 低 ceiling |

---

## 6. 後續方向：CV Blob Detection（2026-04-05）

Phase 8 pseudo-label 失敗後，轉向 **classical CV 直接偵測 noteheads**：
- `training/notehead_detector_cv.py` — adaptive threshold + staff line removal + connected component
- 12/12 TDD tests passed
- 4 種字體測試結果：37-80% 覆蓋率（Gutenberg 最好 80%，Beethoven/Gonville ~38%）
- **位置精準度歷來最好** — 每個框都精確在 notehead 上，不依賴 stem 方向
- 主要瓶頸：staff line removal 破壞部分 noteheads → 需要調參或改用更好的 staff removal 算法
- 待改進：coverage 37-80% → 目標 >60%

---

## 7. 關鍵文件

| 文件 | 用途 |
|------|------|
| `training/create_pseudo_labels.py` | Pseudo-label 生成 pipeline |
| `training/test_pseudo_label_pipeline.py` | 22 個 TDD 測試 |
| `training/train_phase8_pseudo.py` | Phase 8 訓練腳本 |
| `datasets/yolo_harmony_v2_pseudo_v1/` | Pseudo-label dataset |
| `runs/detect/harmony_omr_v2_phase8/pseudo_v1/` | Stage 1 output |
| `runs/detect/harmony_omr_v2_phase8/pseudo_v1_stage2/` | Stage 2 output |

---

## 7. 先前失敗方案（對比）

| 方案 | mAP50 | OpenScore NH | 失敗原因 |
|------|-------|-------------|---------|
| Phase 6 (bbox fix 23px) | 0.816* | **0** predictions | bbox 太小，TAL 無法學 |
| Phase 7 (universal fix) | 0.543 | N/A | 26/32 class 退步 |
| DoReMi-only teacher | N/A | **0** predictions | domain gap 太大 |
| Phase 8 (rule pseudo-label) | 0.56* | plateau | stem direction 錯 |
| **Phase 9 (CV noteheads)** | **0.5719 (pseudo-val)** | **~45% center-match** | box too big (fixed) |

*不同 val set，不可直接比較

---

## 8. Phase 9: CV Noteheads Training (2026-04-05 ~ 2026-04-07)

- Script: `training/train_phase8_pseudo.py` (PROJECT/NAME 已改為 phase9)
- Dataset: V2 (CV detected noteheads + Phase 5 shrink for other classes)
- Base: Phase 5 best.pt + cv2 surgical reset
- Stage 1: 60ep 完成
- Stage 2: 120ep 完成，持續上升到 ep117 (best mAP50=0.5719 on pseudo-val)
- **Weights**: `runs/detect/harmony_omr_v2_phase9/cv_noteheads_v1_stage2/weights/best.pt`

### 8.1 Phase 9 評估結果

在 Phase 6 GT val set 上：
- mAP50 strict (iou=0.7): 0.272
- mAP50 lenient (iou=0.45): 0.347

**但這些分數不可信** — 實證 Phase 6 GT 本身對 OpenScore noteheads 是錯的
（top-edge rule 對 stem-up 音符位置錯）。

### 8.2 Phase 9 實際偵測能力

- DoReMi: 100.3% (與 Phase 6 相同，無退步)
- OpenScore: 82.7% (count ratio) 但 45% (center-match) — Phase 6 為 0
- MUSCIMA: 38.4%

### 8.3 發現 CV detector floor bug (2026-04-07)

原始 `notehead_detector_cv.py` 有 `min_w_px=49, min_h_px=69` floor：
- 導致輸出 bbox 85×90px（GT 23×25px），IoU 永遠 <0.1
- **修正**: `padding_ratio=1.5→0.1`，移除 floor
- 現在輸出 ~25px，接近 GT 大小

---

## 9. Phase B: PC-first Downstream Pipeline (2026-04-07)

### 9.1 動機

9 個 training phases 都在追逐 mAP50，但 Phase 6 GT 本身是錯的，mAP50 無法作為真實評估。
OMR 社群已轉向 OMR-NED (sequence metric)。我們應該直接測試下游任務：
**`harmony_rules.py` 能否在真實樂譜上正確偵測和聲錯誤？**

### 9.2 Phase B 實作（TDD, 13 sub-tasks）

| Sub-task | Module | Tests |
|----------|--------|-------|
| B1 | `staff_detector.py` — 五線譜偵測 | 13/13 ✅ |
| B2 | `pitch_estimator.py` — 位置→MIDI | 15/15 ✅ |
| B3 | `voice_binder.py` — SATB 分配 | 9/9 ✅ |
| B4 | `measure_detector.py` — 小節切分 | 11/11 ✅ |
| B5 | `downstream_eval.py` 整合 | 12/12 ✅ |
| B6 | `test_synthetic_chorales.py` 驗證 rule engine | 10/10 ✅ |
| B7 | 5 張真實樂譜驗證 | completed |
| B8 | Phase B 決策 | 見 PHASE_B_DECISION.md |
| **Total** | | **70/70 passed** |

### 9.3 B7 真實樂譜驗證結果

在 5 張 OpenScore 上跑完整 pipeline：
- Total detections: 878
- Total noteheads: 781
- Total chords: 99
- Total violations: **530 (avg 5.35 per chord — 過高)**

Violation 分佈：
- M1 (旋律跳進): 286 (54%)
- V1 (聲部交叉): 205 (39%)
- P1 (平行 5/8): 30 (6%)
- P2 (隱伏): 9 (2%)

### 9.4 Phase B 核心發現

✅ **Pipeline 架構可行** — 從 Phase 9 detection 到 rule engine 端到端運作
❌ **Pitch estimation 不夠準確** — 導致 fake M1/V1 violations 暴增
⚠️ **Rule engine 基本正確** — synthetic 測試全通過，真實 violation 是 pitch 誤差級聯

### 9.5 Phase B 決策 (PHASE_B_DECISION.md)

**不建議**直接部署 Phase 9 到 Android。真實圖上輸出 500+ fake violations。

**建議**：進 Phase C — 改善 pitch estimation 到 ≥80% 準確度再部署。

---

## 10. 發現的子議題

### 10.1 harmony_rules.py P1 規則語義歧義

- 實作：`interval1 is (P5 or P8) AND interval2 is (P5 or P8)` — flags P5→P8 為平行
- 文件「仍為八度／五度」語義模糊
- 傳統音樂理論：平行 5 度與平行 8 度是不同規則（應為 `P5→P5` 或 `P8→P8`）
- **待使用者決定**是否修正為嚴格語義

### 10.2 多譜表 clef 偵測

- 當前 `_detect_clef_for_staff` 為每個 staff 找最近 clef
- orchestral 樂譜（多 staff + 多 clef）可能錯配
- 需要 Phase C C2 sub-task 處理

### 10.3 Grand staff voice binding

- 2 譜表 grand staff：treble=S/A, bass=T/B 是正確的
- 但多 staff orchestral 樂譜：每 staff 是樂器不是聲部
- 需要 layout 識別區分這兩種情境
