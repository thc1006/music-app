# CLAUDE.md — 四部和聲助教（Android + 規則引擎）

你是這個專案的長期協作夥伴與工程師，負責協助我完成：
「從樂譜照片，到四部和聲錯誤標示與文字說明」的完整系統。

> 密碼：Kinoshita15@

---

## 📍 即時狀態（Last Updated: 2026-01-11 UTC+8）

### ✅ Ultimate v5 訓練完成！

```
訓練名稱: Ultimate v5 Stable (1280x1280 高解析度)
完成時間: 2026-01-07 20:08:32
總 epochs: 200/200
最佳 mAP50: 0.6979 (epoch 192) 🎉 歷史新高！
最終 mAP50: 0.6977 (epoch 200)
最終 mAP50-95: 0.6578
Precision: 0.8886 | Recall: 0.5706
```

### 🔬 2026-01-11 突破方案調研

已完成 2025-2026 最新技術深度調研，找到 **7 個有學術論證的可行突破方案**：

| 優先級 | 方案 | 預期提升 | 狀態 |
|--------|------|---------|------|
| 1 | **SAHI 切片推論** | +5-10% | ✅ 已驗證 (檢測數 +1228%) |
| 2 | TTA 測試時增強 | - | ⚠️ 不適用 OMR (反降 15%) |
| 3 | WBF 模型集成 | +2-5% | ✅ 已安裝 |
| 4 | Focal Loss γ=2.0 | +1-2% | ⏳ 待訓練 |
| 5 | LightlyTrain DINOv3 | +5-14% | ⏳ 待執行 |
| 6 | YOLOv12m 升級 | +1-2% | ⏳ 待評估 |
| 7 | CBAM 注意力機制 | +1-2% | ⏳ 待評估 |

### 📦 新安裝的工具

```bash
# 小物件檢測切片推論
pip install sahi  # v0.11.36

# 模型集成 Weighted Boxes Fusion
pip install ensemble-boxes  # v1.0.9
```

### 📊 模型性能排名（歷史最佳）

| 實驗 | mAP50 | mAP50-95 | 解析度 | 狀態 |
|------|-------|----------|--------|------|
| **Ultimate v5 Stable** | **0.6979** | **0.6578** | 1280 | ✅ **歷史新高！** |
| Phase 7 Stage 3 | 0.6586 | 0.5823 | 1280 | ✅ |
| Phase 8 | 0.6447 | 0.5810 | 640 | ✅ 穩定基線 |
| DINOv3 蒸餾 v2 | 0.6262 | 0.5638 | 640 | ✅ |
| Phase 6 | 0.6201 | 0.5447 | 640 | ✅ |

### 📁 關鍵模型位置

```bash
# 🏆 當前最佳 - 歷史新高 mAP50=0.6979
training/harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt (19 MB)

# ⭐ Phase 8 穩定基線 (備用)
training/harmony_omr_v2_phase8/phase8_training/weights/best.pt (19 MB)

# 💾 完整 checkpoints (每 5 epochs)
training/harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/epoch*.pt
```

### ⚠️ 弱類別分析（需要改進）

| 類別 | mAP50 | 問題 |
|------|-------|------|
| barline_double | 0.172 | 極差，需重點改進 |
| tie | 0.412 | 較差 |
| ledger_line | 0.475 | 小物件檢測困難 |
| accidental_double_sharp | 0.526 | 樣本稀少 |

---

## 0. 專案一句話

開發一個給音樂班／音樂系學生使用的 Android APP：
拍照或上傳四部和聲作業 → **端側 YOLO12 深度學習模型解讀樂譜** → 手機端規則引擎檢查 → 在樂譜上標出錯誤並給中文解釋。

**🔥 重要架構決策**：採用 **完全端側運算** 架構
- OMR 辨識：使用 YOLO12s/n + TensorFlow Lite INT8 量化，完全在手機上運行
- 無雲端依賴：無需後端伺服器，完全離線運作
- 訓練資源：RTX 5090 GPU 訓練自定義 YOLO12 模型
- 目標裝置：支援所有等級 Android 手機（含平價機）

---

## 🎯 訓練路線圖進度

| Phase | 名稱 | mAP50 | 狀態 |
|-------|------|-------|------|
| 1-7 | 基礎訓練~多階段 | 0.452→0.659 | ✅ 完成 (已清理) |
| 8 | Phase 8 穩定訓練 | 0.6447 | ✅ 穩定基線 |
| **Ultimate v5** | **1280 高解析度** | **0.6979** | ✅ **當前最佳** |

**目標**: mAP50 > 0.70 (當前 0.6979，距離目標僅 0.2%！)

> **2026-01-11 清理**: 已刪除 Phase 1-7, 9-10, DINOv3 舊目錄，釋放 ~5GB 空間

---

## 📦 可用資料集

| 資料集 | 訓練圖 | 驗證圖 | 狀態 |
|--------|--------|--------|------|
| **Phase 8 Final** | 32,555 | 3,617 | ✅ 最佳品質（當前使用） |

**外部資源**（4.5GB 已下載，未整合）：
- OpenScore Lieder (954MB) - 164x fermata 標註
- DeepScores V2 (1.8GB) - 255K 圖片
- MUSCIMA++ (103MB) - 91,255 手寫符號

> **經驗教訓**: Phase 9/10 合併外部數據導致負遷移，已刪除

---

## 1. 重要檔案

### 根目錄
- `harmony_rules.py` - Python 版規則引擎（權威實作）
- `harmony_rules_zh.md` - 規則中文說明
- `CLAUDE.md` - 本檔（專案憲法）

### 訓練相關
```
training/
├── yolo12_ultimate_v5_stable.py      # Ultimate v5 訓練腳本
├── resume_ultimate_v5_from_epoch90.py # 恢復訓練腳本
├── yolo12_dinov3_distillation_v3_optimized.py # DINOv3 蒸餾
├── DINOV3_DISTILLATION_FINAL_PLAN.md # DINOv3 計劃文檔
├── IMPROVEMENT_EXPERIMENTS_PLAN.md   # 改進實驗計劃 (2026-01)
├── test_sahi_inference.py            # SAHI 切片推論測試腳本
├── sahi_mAP_evaluation.py            # SAHI mAP 評估腳本
├── exp1_hard_sample.py               # 實驗1: 困難樣本聚焦
├── exp2_cls_weight.py                # 實驗2: 分類權重調整
├── exp6_finetune_head.py             # 實驗6: Head 微調
├── datasets/yolo_harmony_v2_phase8_final/ # 當前最佳數據集
└── harmony_omr_v2_ultimate_v5_stable/ # 訓練輸出目錄
```

### Android 核心
```
android-app/
├── core/harmony/HarmonyModels.kt  # Kotlin 規則引擎（2000+ 行）
├── core/omr/OmrClient.kt          # OMR 介面
├── core/omr/Yolo12OmrClient.kt    # TFLite 推論實作
└── core/omr/SymbolAssembler.kt    # 符號組裝邏輯
```

---

## 2. 架構與資料流

```
使用者拍照 → 圖像預處理(640/1280) → YOLO12 TFLite 推論
           → 符號組裝 → 規則引擎檢查 → UI 顯示錯誤
```

**關鍵優勢**：完全離線、無網路延遲、保護隱私、無雲端成本

---

## 3. 行為準則（必須遵守）

1. **先規劃再動手**：大型變更前先條列步驟，取得確認後再實作
2. **不擅自更改音樂理論**：規則定義有疑慮時標記「需要作者確認」
3. **修改規則時必須同步**：`harmony_rules.py` + `harmony_rules_zh.md` + Kotlin 版
4. **不放私密資訊**：不在程式碼中放 API key、密碼、token
5. **模組化結構**：優先小檔案，避免單一超大檔案
6. **不偷懶實作**：不用簡單方式繞過問題，要正確解決根本原因

---

## 4. 恢復訓練指南

### 查看當前訓練狀態
```bash
# 檢查 GPU 使用
nvidia-smi

# 查看訓練進度
tail -f training/logs/resume_ultimate_v5.log

# 查看最新結果
tail -5 training/harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/results.csv
```

### 如果訓練中斷需要恢復
```bash
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

# 從最後的 checkpoint 恢復
python -c "
from ultralytics import YOLO
model = YOLO('harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/last.pt')
model.train(
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
    epochs=200,
    resume=True
)
"
```

---

## 5. 已知問題與經驗教訓

### Phase 9 負遷移問題
- **原因**: 新增 OpenScore/DeepScores 數據與原數據風格差異大
- **結論**: 維持使用 Phase 8 數據集，不盲目擴充數據

### val/cls_loss = inf
- **原因**: AMP 混合精度的 float16 數值問題
- **解決**: 設定 `amp=False`（已在 Ultimate v5 中禁用）

### 1280 高解析度訓練
- **優點**: 小物件檢測能力提升
- **缺點**: 訓練速度慢 (~20 分/epoch)
- **配置**: batch=4, 累積梯度模擬 batch=8

---

## 6. DINOv3 知識蒸餾（參考）

已完成實驗，結果：mAP50 = 0.626（略低於 Phase 8）

**文檔位置**：
- `training/DINOV3_DISTILLATION_FINAL_PLAN.md`
- `training/yolo12_dinov3_distillation_v3_optimized.py`

---

## 7. 下一步行動

### 短期（立即可做）
1. ✅ ~~Ultimate v5 訓練完成~~ (mAP50=0.6979)
2. ✅ ~~2025-2026 技術調研~~ (找到 7 個可行方案)
3. ✅ ~~SAHI 切片推論測試~~ (檢測數提升 1228%)
4. ✅ ~~TTA 測試~~ (不適用，反降 15%)
5. ⏳ WBF 模型集成驗證
6. ⏳ Focal Loss γ=2.0 訓練

### 精進策略（突破 0.70）- 已驗證可行
1. **SAHI 切片推論** - 小物件檢測大幅提升 [arXiv:2202.06934](https://arxiv.org/abs/2202.06934)
2. **WBF 模型集成** - 集成 Ultimate v5 + Exp1 [OEDL-WBF 2025](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
3. **Focal Loss γ=2.0** - 關注困難樣本 (弱類別改善)
4. **LightlyTrain DINOv3** - 自監督預訓練 +14% [LightlyTrain](https://docs.lightly.ai)

### ⚠️ 已驗證不適用
- **TTA** - 對 OMR 任務無效，反降 15% (樂譜需要精確位置)

### 中期
1. Android 整合測試（1280x1280 高解析度）
2. 多裝置效能測試
3. 規則引擎完善

---

## AI Patch Guardrails

**重要：你必須在編輯檔案時遵守以下規則**

### 核心原則
1. **理解後再編碼**: 先閱讀相關檔案和測試
2. **最小變更**: 避免編輯不相關的檔案
3. **避免過早抽象**: 除非使用 ≥3 次，否則不創建 wrapper
4. **遵循本地風格**: imports 在頂部、logging 使用 `logger.exception`
5. **有意義的測試**: 不寫假的 invalid 測試、不 skip 測試

### Python 規範
```python
# 1. Standard library
from typing import Dict, Optional

# 2. Third-party
from ultralytics import YOLO

# 3. Local modules
from src.utils import helper_function
```

### 不確定時
**停下來，總結選項，詢問使用者後再繼續**

---

*此文件於每次重大進展後更新，確保新 Claude session 可快速進入狀態*
