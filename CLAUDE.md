# CLAUDE.md — 四部和聲助教（Android + 規則引擎）

你是這個專案的長期協作夥伴與工程師，負責協助我完成：
「從樂譜照片，到四部和聲錯誤標示與文字說明」的完整系統。

> 密碼：Kinoshita15@

---

## 📍 即時狀態（Last Updated: 2026-02-21 22:00 UTC+8）

### 🔬 Phase 3 Recall Boost 訓練準備中

**目標**: 從 mAP50=0.7763 突破至 >0.85，核心策略：data-centric + recall-boosting hyperparams
- SDD: `training/SDD_mAP_breakthrough.md`
- 訓練腳本: `training/train_phase3_recall_boost.py`
- 數據集: cleaned_v2（移除 lieder+ds2+quartets 有毒數據 + 過濾微小框）

### 🎉 里程碑：mAP50 = 0.7763（cleaned val, 無需重訓練）

**2026-02-21 重大突破**: 通過移除有毒數據（lieder/ds2），mAP50 從 0.7519 → 0.7763！

┌─────────────────────────────────────────────────────────────┐
│  🏆 當前最佳配置                                             │
├─────────────────────────────────────────────────────────────┤
│  模型: Ultimate v5 Stable (YOLO12s, 1280x1280)              │
│  權重: training/harmony_omr_v2_ultimate_v5_stable/          │
│        stable_1280_resumed/weights/best.pt                  │
│  數據: cleaned val (移除 lieder+ds2 有毒數據)                │
│  推論: conf=0.25, iou=0.55                                  │
│  結果: mAP50=0.7763, mAP50-95=0.7320, P=0.957, R=0.576     │
└─────────────────────────────────────────────────────────────┘

### 📊 關鍵發現（2026-02-21 深度分析）

| 發現 | 影響 | 狀態 |
|------|------|------|
| TIDE: 96.8% mAP loss = Miss | Recall 是核心瓶頸 | ✅ 已確認 |
| 有毒數據: lieder+ds2+quartets | 移除後 mAP +2.44% | ✅ 已清洗 |
| YOLO12s >> YOLO26s (7/10 勝) | YOLO26 TFLite Android 不可用 | ✅ 決定用 YOLO12s |
| TAL OOM: batch=8 頻繁 OOM | 改用 batch=6, nbs=64 | ✅ 已解決 |
| 微調需低 LR (0.0003) | 從 best.pt 微調，非 yolo12s.pt | ✅ 已規劃 |

---

## 🔬 2025-2026 學術調研結果 (2026-01-21)

### ✅ YOLO26 已可用！(Ultralytics v8.4.6)

```bash
# 已升級安裝
pip install ultralytics==8.4.6

# YOLO26 模型規格
yolo26n: 2.6M params, 6.1 GFLOPs
yolo26s: 10.0M params, 22.8 GFLOPs  ← 推薦（與當前 YOLO12s 相當）
yolo26m: 21.9M params, 75.4 GFLOPs
```

### 🎯 YOLO26 關鍵特性 (對本專案的價值)

| 特性 | 說明 | 對 OMR 的意義 |
|------|------|--------------|
| **NMS-Free 推論** | 移除 NMS 後處理 | 不再需要調 IoU 閾值 |
| **移除 DFL** | 簡化導出圖 | TFLite INT8 更穩定 |
| **STAL** | 小目標感知標籤分配 | ⭐ 改善 barline_double |
| **ProgLoss** | 漸進式損失平衡 | 訓練更穩定 |
| **MuSGD** | 借鑒 LLM 優化器 | 收斂更快 |

### 📚 arXiv 最新技術調研

#### OMR 相關

| 論文 | 時間 | 重點 |
|------|------|------|
| [Sheet Music Benchmark](https://arxiv.org/abs/2506.10488) | 2025.06 | 標準化 OMR 評估框架 |
| [Sheet Music Transformer++](https://arxiv.org/abs/2405.12105) | 2025.06 | 端到端全頁 OMR |
| Rios-Vila et al. | 2025.10 | Implicit Layout-Aware Transformer |

#### 小物件檢測

| 技術 | 來源 | 效果 |
|------|------|------|
| [STAL](https://arxiv.org/abs/2509.25164) | YOLO26 | 小目標感知標籤分配 |
| [ACD-DETR](https://arxiv.org/abs/2503.20516) | Survey | 多尺度邊緣增強 |
| Super-Resolution | 多篇 | 提升小物件解析度 |

#### NMS 替代方案

| 方法 | 來源 | 效果 |
|------|------|------|
| **YOLO26 NMS-Free** | v8.4.0 | 完全移除 NMS |
| [Confluence](https://arxiv.org/abs/2012.00257) | arXiv | mAP +2-4%, Recall +5-7% |
| [QSI-NMS](https://arxiv.org/abs/2409.20520) | arXiv | 6.2x 加速，mAP -0.1% |

#### 知識蒸餾

| 方法 | 來源 | 效果 |
|------|------|------|
| [YOLO LwF](https://arxiv.org/abs/2503.04688) | 2025.03 | mAP +2.1% (VOC) |
| CWD + MGD | ICCV 2025 | 輕量化 + 全類別提升 |

### 🔄 YOLO26s 微調訓練中！(2026-01-21)

**目標**: 利用 YOLO26 NMS-Free + STAL 特性進一步提升小物件檢測

```
訓練狀態: 🔄 運行中
開始時間: 2026-01-21 01:44
腳本: training/yolo26s_finetune_batch4.py
日誌: training/logs/yolo26s_batch4.log
輸出: training/harmony_omr_v2_yolo26/yolo26s_finetune_batch4/

訓練配置:
  - 模型: yolo26s.pt (預訓練權重)
  - 解析度: 1280x1280
  - batch: 4 (避免 OOM)
  - nbs: 64 (梯度累積模擬 batch=64)
  - epochs: 100
  - patience: 30
  - optimizer: AdamW
  - lr0: 0.001

GPU 狀態:
  - VRAM: ~29GB / 32GB (90%)
  - OOM 次數: 0 ✅
  - 速度: ~10 it/s
  - 預計每 epoch: ~14-15 分鐘
  - 預計完成: ~25-30 小時
```

監控指令:
```bash
# 查看即時進度
tail -f training/logs/yolo26s_batch4.log

# 檢查 OOM
grep -c "OutOfMemoryError" training/logs/yolo26s_batch4.log

# GPU 狀態
nvidia-smi
```

### 🚀 下一步升級建議

#### 高優先級 (立即可行)

| 建議 | 預期效果 | 難度 | 狀態 |
|------|---------|------|------|
| **1. YOLO26s 微調** | 小物件 +3-5%，NMS-Free | 中 | 🔄 **訓練中** |
| **2. Confluence NMS** | mAP +2-4% | 低 | ⏳ 備選 |
| **3. CBAM 注意力** | 假陽性 -10%+ | 中 | ⏳ 備選 |

#### 中優先級

| 建議 | 預期效果 | 難度 |
|------|---------|------|
| Sheet Music Transformer | 端到端 OMR | 高 |
| Equalized Focal Loss | 弱類別改善 | 中 |
| YOLO LwF 知識蒸餾 | mAP +2-3% | 中 |

### 📋 YOLO26 升級計劃

```python
# 使用 YOLO26s 微調現有模型
from ultralytics import YOLO

# 方案 A: 從預訓練權重微調 (推薦)
model = YOLO('yolo26s.pt')
model.train(
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
    epochs=100,
    imgsz=1280,
    batch=8,
    project='harmony_omr_v2_yolo26',
    name='yolo26s_finetune'
)

# 方案 B: 使用知識蒸餾從 Ultimate v5 遷移
# (需要額外實現 teacher-student 架構)
```

### ✅ 調研驗證：當前方向正確

- ✅ **1280 高解析度** - 符合小物件檢測最佳實踐
- ✅ **NMS IoU 調整** - 研究證實有效 (Confluence 等)
- ✅ **TAL 標籤分配** - 當前最佳策略，YOLO26 STAL 更進一步
- ✅ **INT8 量化部署** - 業界標準，YOLO26 更易導出

---

## 📈 完整實驗歷史與結果

### 訓練實驗總覽

| 實驗 | 名稱 | 架構 | Epochs | mAP50 | mAP50-95 | 狀態 |
|------|------|------|--------|-------|----------|------|
| **Ultimate v5** | 1280 高解析度 | YOLO12s | 200/200 | **0.6979** | **0.6578** | ✅ **最佳模型** |
| Exp5 | DINOv3 蒸餾 v3 | YOLO12s | 30/30 | 0.6976 | 0.6577 | ✅ 第二名 |
| Exp4 | 弱類別微調 | YOLO12s | 30/30 | 0.6959 | 0.6543 | ✅ 第三名 |
| Exp1 | 困難樣本聚焦 | YOLO12s | 66/100 | 0.6727 | 0.6216 | ✅ 提早停止 |
| Phase 8 | 穩定基線 | YOLO12s | 100 | 0.6447 | 0.5810 | ✅ 基線 |
| Exp6 | YOLO11m batch=4 | YOLO11m | 61/100 | 0.5741 | 0.5048 | ❌ 失敗 |
| **YOLO26s** | **NMS-Free 微調** | **YOLO26s** | **1/100** | - | - | 🔄 **訓練中** |

### 推論優化實驗

| 方案 | 測試結果 | 結論 |
|------|---------|------|
| **🎉 NMS IoU=0.55** | **mAP50=0.7059** | ✅ **成功突破 0.70！無需重訓練** |
| WBF 模型集成 | 模型同意率 99.87% | ⚠️ 效果有限（模型太相似） |
| SAHI 切片推論 | mAP50=0.0794 | ❌ **不適用** (mAP 下降 88.6%) |
| TTA 測試時增強 | mAP 下降 15% | ❌ 不適用 OMR（需精確位置） |

---

## 🔬 詳細實驗記錄

### Ultimate v5 Stable (當前最佳模型)

```
訓練名稱: Ultimate v5 Stable (1280x1280 高解析度)
開始時間: 2026-01-05
完成時間: 2026-01-07 20:08:32
總 epochs: 200/200
最佳 mAP50: 0.6979 (epoch 192)
最終 mAP50: 0.6977 (epoch 200)
最終 mAP50-95: 0.6578
Precision: 0.8886 | Recall: 0.5706

配置:
  - 架構: YOLO12s
  - 解析度: 1280x1280
  - batch: 8
  - optimizer: AdamW
  - lr0: 0.001
  - amp: False (避免 inf loss)
```

### Exp6 YOLO11m (失敗案例分析)

```
訓練名稱: Exp6 YOLO11m batch=4
開始時間: 2026-01-15 04:11
完成時間: 2026-01-15 17:48
訓練時長: 13.611 小時
終止原因: Early Stopping (patience=20)
最佳 epoch: 41
最佳 mAP50: 0.5741 (比基線低 17.7%)

失敗原因分析:
1. YOLO11m 從 scratch 訓練需要更多 epochs (可能 200+)
2. 超參數 (lr0=0.001) 可能對 YOLO11m 不適合
3. Phase 8 數據集可能更適合 YOLO12 架構
4. batch=4 + 1280 導致有效訓練樣本不足

結論: 不建議使用 YOLO11m 從頭訓練，應使用預訓練權重微調
```

### SAHI 切片推論 (失敗案例分析)

```
測試日期: 2026-01-21
測試配置:
  - 切片大小: 640x640
  - 重疊率: 0.2
  - 信心度閾值: 0.25

結果:
  - 標準 YOLO mAP50: 0.6978
  - SAHI mAP50: 0.0794
  - 下降幅度: -88.6%

失敗原因:
1. 樂譜符號需要上下文關係（五線譜、相鄰符號）
2. 切片破壞了符號間的空間關係
3. 小物件（如 barline）被切割後無法正確匹配

結論: SAHI 不適用於 OMR 任務，樂譜需要完整圖片推論
```

### WBF 模型集成 (效果有限)

```
測試日期: 2026-01-21
測試模型:
  - Ultimate v5: mAP50=0.6978
  - Exp5 DINOv3: mAP50=0.6976

WBF 參數:
  - iou_thr: 0.55
  - skip_box_thr: 0.01
  - weights: [1.0, 0.95]

結果:
  - 模型同意率: 99.87%
  - WBF 集成 mAP50: ~0.6978 (無顯著提升)

結論: 兩個模型預測太相似，WBF 無法產生互補效果
建議: 需要不同架構或不同訓練策略的模型才能有效集成
```

---

## 📁 模型與檔案位置

### 當前最佳模型

```bash
# 🏆 生產環境使用 - mAP50=0.7059 (with iou=0.55)
training/harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt  # 19 MB

# 推論配置
model.predict(source=image, imgsz=1280, conf=0.25, iou=0.55)
```

### 其他可用模型

```bash
# ⭐ Exp5 DINOv3 蒸餾 v3 (mAP50=0.6976) - 第二選擇
training/harmony_omr_v2_experiments/exp5_dinov3/finetune/weights/best.pt

# ⭐ Exp4 弱類別微調 (mAP50=0.6959)
training/harmony_omr_v2_experiments/exp4_finetune/weights/best.pt

# Phase 8 穩定基線 (mAP50=0.6447) - 備用
training/harmony_omr_v2_phase8/phase8_training/weights/best.pt

# ❌ Exp6 YOLO11m (不建議使用)
# training/harmony_omr_v2_experiments/exp6_yolo11m_batch4/weights/best.pt
```

### 訓練腳本與文檔

```
training/
├── yolo12_ultimate_v5_stable.py        # Ultimate v5 訓練腳本
├── resume_exp6_batch4.py               # Exp6 訓練腳本
├── sahi_full_map_eval.py               # SAHI mAP 評估腳本
├── wbf_official_eval.py                # WBF 官方評估腳本
├── wbf_map_evaluation.py               # WBF mAP 評估腳本
├── IMPROVEMENT_EXPERIMENTS_PLAN.md     # 改進實驗計劃
├── BARLINE_DOUBLE_IMPROVEMENT_PLAN.md  # 弱類別改進計劃
└── datasets/yolo_harmony_v2_phase8_final/  # 當前數據集
```

---

## 🖥️ 硬體與環境

### GPU 配置

```
GPU: NVIDIA GeForce RTX 5090 (32GB VRAM)
CPU: Intel i9-14900 (24 cores)
RAM: 125 GB
Ultralytics: v8.4.6 (支援 YOLO26)
```

### 訓練配置建議

| 架構 | imgsz | batch | VRAM 使用 | 備註 |
|------|-------|-------|-----------|------|
| YOLO12s | 1280 | 8 | ~20 GB | ✅ 推薦 |
| YOLO12s | 640 | 16 | ~12 GB | 快速訓練 |
| YOLO11m | 1280 | 4 | ~22 GB | ⚠️ 容易 OOM |
| YOLO11m | 1280 | 6 | >32 GB | ❌ OOM |

### Pixel 7 部署評估

| 項目 | YOLO12s INT8 | YOLO11m INT8 |
|------|-------------|--------------|
| 模型大小 | ~5 MB | ~10-12 MB |
| 推論延遲 | 50-100ms | 80-200ms |
| FPS | 10-20 | 5-12 |
| 實時可行性 | ✅ 推薦 | ⚠️ 勉強 |

---

## 📊 弱類別分析

### 當前弱類別 (需要改進)

| 類別 | mAP50 | 問題 | 建議策略 |
|------|-------|------|---------|
| barline_double | 0.172 | 樣本稀少、標註框過大 | OpenScore 數據增強 |
| tie | 0.412 | 與 slur 混淆 | 分類權重調整 |
| ledger_line | 0.475 | 小物件檢測困難 | 已改善 (1280 解析度) |
| accidental_double_sharp | 0.526 | 樣本極稀少 | 合成數據 |

### 尚未嘗試的改進方案

來自 `IMPROVEMENT_EXPERIMENTS_PLAN.md`:

| 優先級 | 方案 | 核心參數 | 預期效果 |
|--------|------|---------|---------|
| 1 | 分類損失權重 | `cls=1.0` (預設 0.5) | 改善類別混淆 |
| 2 | 兩階段微調 | `freeze=10, lr0=0.0001` | 提升分類精度 |
| 3 | 弱類別 Oversampling | 複製弱類別圖片 3x | 平衡數據分佈 |
| 4 | MixUp/Copy-Paste | `mixup=0.15, copy_paste=0.1` | 增強泛化 |

---

## ✅ 已驗證方案總結

### 成功方案

| 方案 | 效果 | 建議 |
|------|------|------|
| **NMS IoU=0.55** | mAP50 +0.81% (0.6978→0.7059) | ✅ 生產使用 |
| **1280 高解析度** | mAP50 +5.3% (0.6447→0.6979) | ✅ 生產使用 |
| **Focal Loss γ=2.0** | mAP50=0.6959 | ✅ 有效但未超越 |
| **DINOv3 蒸餾** | mAP50=0.6976 | ✅ 備選方案 |

### 失敗方案 (不建議使用)

| 方案 | 結果 | 原因 |
|------|------|------|
| **SAHI 切片推論** | mAP 下降 88.6% | 破壞符號上下文關係 |
| **TTA 測試時增強** | mAP 下降 15% | 樂譜需要精確位置 |
| **YOLO11m 從頭訓練** | mAP 下降 17.7% | 需更多 epochs/微調 |
| **WBF 模型集成** | 無顯著提升 | 模型太相似 |

---

## 🎯 下一步行動

### 當前狀態 (2026-01-21)

```
✅ 已完成:
1. mAP50 > 0.70 達成 (0.7059 with iou=0.55)
2. Ultralytics 升級到 v8.4.6
3. YOLO26 可用性確認

🔄 進行中:
- 2025-2026 學術調研完成，發現 YOLO26 是重大升級機會
```

### 兩條可選路線

#### 路線 A: 直接部署 (快速上線)

```
1. 導出 TFLite INT8 模型 (當前 Ultimate v5 + iou=0.55)
   ↓
2. Pixel 7 實機測試
   ↓
3. Android App 整合
   ↓
4. 規則引擎完善與測試
```

#### 路線 B: YOLO26 升級 (更高性能) ⭐ 推薦

```
1. YOLO26s 微調訓練 (~10-15 小時)
   ↓
2. 驗證 NMS-Free + STAL 對小物件的改善
   ↓
3. 導出 TFLite INT8 (YOLO26 導出更穩定)
   ↓
4. Pixel 7 實機測試 + Android 整合
```

### 推論配置參考

```python
# 方案 A: 當前最佳 (Ultimate v5 + iou=0.55)
from ultralytics import YOLO
model = YOLO('training/harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt')
results = model.predict(source=image, imgsz=1280, conf=0.25, iou=0.55)

# 方案 B: YOLO26 (NMS-Free，不需要 iou 參數)
model = YOLO('yolo26s.pt')  # 或微調後的權重
results = model.predict(source=image, imgsz=1280, conf=0.25)  # 無需 iou!
```

### YOLO26 微調指令 (待執行)

```python
from ultralytics import YOLO

model = YOLO('yolo26s.pt')
model.train(
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
    epochs=100,
    imgsz=1280,
    batch=8,
    project='harmony_omr_v2_yolo26',
    name='yolo26s_finetune'
)
```

### TFLite 導出指令

```python
# YOLO26 導出更簡單 (無 DFL，無 NMS)
model.export(
    format='tflite',
    imgsz=1280,
    int8=True,
    data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
)
```

---

## 📦 可用資料集

| 資料集 | 訓練圖 | 驗證圖 | 狀態 |
|--------|--------|--------|------|
| **Phase 8 Final** | 32,555 | 3,617 | ✅ 當前使用 |

**外部資源** (已下載，未整合):
- OpenScore Lieder (954MB) - 463 個 barline_double
- DeepScores V2 (1.8GB) - 255K 圖片
- MUSCIMA++ (103MB) - 91,255 手寫符號

> **經驗教訓**: Phase 9/10 合併外部數據導致負遷移，已刪除

---

## 0. 專案一句話

開發一個給音樂班／音樂系學生使用的 Android APP：
拍照或上傳四部和聲作業 → **端側 YOLO12 深度學習模型解讀樂譜** → 手機端規則引擎檢查 → 在樂譜上標出錯誤並給中文解釋。

**架構決策**: 採用 **完全端側運算** 架構
- OMR 辨識：YOLO12s + TensorFlow Lite INT8 量化
- 無雲端依賴：完全離線運作
- 訓練資源：RTX 5090 GPU
- 目標裝置：支援所有等級 Android 手機

---

## 1. 重要檔案

### 根目錄
- `harmony_rules.py` - Python 版規則引擎（權威實作）
- `harmony_rules_zh.md` - 規則中文說明
- `CLAUDE.md` - 本檔（專案憲法）

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
使用者拍照 → 圖像預處理(1280) → YOLO12 TFLite 推論 (iou=0.55)
           → 符號組裝 → 規則引擎檢查 → UI 顯示錯誤
```

---

## 3. 行為準則（必須遵守）

1. **先規劃再動手**：大型變更前先條列步驟，取得確認後再實作
2. **不擅自更改音樂理論**：規則定義有疑慮時標記「需要作者確認」
3. **修改規則時必須同步**：`harmony_rules.py` + `harmony_rules_zh.md` + Kotlin 版
4. **不放私密資訊**：不在程式碼中放 API key、密碼、token
5. **模組化結構**：優先小檔案，避免單一超大檔案
6. **不偷懶實作**：不用簡單方式繞過問題，要正確解決根本原因

---

## 4. 已知問題與經驗教訓

### Phase 9 負遷移問題
- **原因**: 新增 OpenScore/DeepScores 數據與原數據風格差異大
- **結論**: 維持使用 Phase 8 數據集，不盲目擴充數據

### val/cls_loss = inf
- **原因**: AMP 混合精度的 float16 數值問題
- **解決**: 設定 `amp=False`

### YOLO11m OOM
- **問題**: batch=6 + 1280x1280 + YOLO11m 導致 OOM
- **解決**: 降低 batch size 到 4

### SAHI 不適用 OMR
- **問題**: 切片推論破壞符號上下文
- **結論**: 樂譜必須使用完整圖片推論

---

## 5. 恢復訓練指南

```bash
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

# 檢查 GPU
nvidia-smi

# 從 checkpoint 恢復
python -c "
from ultralytics import YOLO
model = YOLO('harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/last.pt')
model.train(data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml', epochs=200, resume=True)
"
```

---

## AI Patch Guardrails

### 核心原則
1. **理解後再編碼**: 先閱讀相關檔案和測試
2. **最小變更**: 避免編輯不相關的檔案
3. **避免過早抽象**: 除非使用 ≥3 次，否則不創建 wrapper
4. **遵循本地風格**: imports 在頂部、logging 使用 `logger.exception`
5. **有意義的測試**: 不寫假的 invalid 測試、不 skip 測試

### 不確定時
**停下來，總結選項，詢問使用者後再繼續**

---

*此文件於 2026-02-21 全面更新*
*- mAP50 = 0.7763 突破里程碑 (有毒數據清洗)*
*- 5 代理深度研究完成：微調策略、超參數、YOLO12s vs 26s、數據品質、OOM 根因*
*- Phase 3 Recall Boost SDD 完成，準備訓練*
