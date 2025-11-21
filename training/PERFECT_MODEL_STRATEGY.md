# 🏆 無敵 OMR 模型：長期戰略規劃

> **目標**：打造生產級、可靠的四部和聲樂譜辨識系統
> **時間範圍**：3-6 個月
> **最終目標**：mAP50 > 0.90，所有核心類別 > 0.85

---

## 一、核心洞察

### 1.1 從應用場景反推需求

```
用戶流程：
拍照樂譜 → OMR 辨識 → 和聲規則檢查 → 顯示錯誤

關鍵問題：OMR 需要辨識什麼才能讓規則引擎運作？
```

### 1.2 和聲規則引擎的輸入需求

```python
# harmony_rules.py 需要的資訊：
class ChordSnapshot:
    measure: int        # ← 需要 barline 辨識
    beat: float         # ← 需要 note values 辨識
    soprano: Pitch      # ← 需要 notehead + accidental + clef
    alto: Pitch         # ← 同上
    tenor: Pitch        # ← 同上
    bass: Pitch         # ← 同上
```

### 1.3 真正必要的類別（最小集合）

| 優先級 | 類別 | 用途 | 當前樣本數 | 狀態 |
|--------|------|------|-----------|------|
| P0 | notehead_filled | 音符頭 | 501,814 | ✅ |
| P0 | notehead_hollow | 音符頭 | 58,826 | ✅ |
| P0 | stem | 聲部區分 | 471,390 | ✅ |
| P0 | clef_treble | 音高參考 | 8,332 | ✅ |
| P0 | clef_bass | 音高參考 | 10,810 | ✅ |
| P0 | accidental_sharp | 升記號 | 27,645 | ✅ |
| P0 | accidental_flat | 降記號 | 19,568 | ✅ |
| P0 | accidental_natural | 還原記號 | 12,705 | ✅ |
| P0 | key_signature | 調號 | 24,146 | ✅ |
| P0 | barline | 小節線 | 5,572 | ✅ |
| P1 | accidental_double_sharp | 重升 | 338 | ⚠️ |
| P1 | accidental_double_flat | 重降 | **12** | ❌ 危機 |
| P1 | clef_alto | 中音譜號 | 2,644 | ✅ |
| P1 | clef_tenor | 次中音譜號 | 614 | ⚠️ |
| P2 | ledger_line | 加線 | 176,632 | ✅ |
| P2 | time_signature | 拍號 | 16,565 | ✅ |

---

## 二、問題診斷

### 2.1 致命問題：重升/重降記號

```
Class 17 (double_flat): 僅 12 個樣本

這意味著：
- 模型幾乎不可能學會辨識重降記號
- 當學生寫出 B♭♭ (= A) 時，系統會辨識錯誤
- 錯誤的音高 → 錯誤的和聲判斷 → 用戶失去信任
```

**為什麼這很重要？**

在四部和聲中，重升/重降雖然罕見，但出現在：
- 半音階進行
- 調性模糊的段落
- 變格終止的特殊處理

### 2.2 數據來源限制

```
當前：100% 來自 DeepScoresV2
問題：
├── 合成數據，非真實掃描
├── 固定的 5 種字體
├── 無手寫樂譜
├── 類別分布由原始樂譜決定（無法控制）
└── 重升/重降在古典樂譜中本就罕見
```

### 2.3 訓練瓶頸

| 瓶頸 | 影響 | 解法 |
|------|------|------|
| 類別極度不平衡 | 稀有類別學不會 | 合成數據 + 過採樣 |
| 數據量有限 | 泛化能力差 | 擴充數據集 |
| 單一來源 | 對真實輸入不魯棒 | 多源數據整合 |
| 解析度限制 | 小符號細節丟失 | 高解析度訓練 |

---

## 三、長期戰略路線圖

```
                          mAP50 目標
                              │
Week 1-2:   Phase 1 ─────────┼── 0.50-0.55 (基礎)
                              │
Week 3-4:   Phase 2 ─────────┼── 0.60-0.65 (類別平衡)
                              │
Week 5-8:   Phase 3 ─────────┼── 0.70-0.75 (合成數據)
                              │
Week 9-12:  Phase 4 ─────────┼── 0.80-0.85 (高解析度+架構)
                              │
Week 13-16: Phase 5 ─────────┼── 0.85-0.90 (真實數據)
                              │
Week 17+:   Phase 6 ─────────┼── 0.90+ (生產優化)
                              │
                           「無敵」
```

---

## 四、Phase 2：類別平衡訓練（關鍵轉折點）

### 4.1 類別加權損失函數

```python
# 原理：讓模型更關注稀有類別的錯誤
# 頻率越低 → 權重越高

def calculate_class_weights(class_counts, max_weight=50.0):
    """
    計算類別權重
    """
    total = sum(class_counts.values())
    weights = {}

    for cls, count in class_counts.items():
        # 反頻率加權
        weight = total / (len(class_counts) * count)
        # 限制最大權重
        weights[cls] = min(weight, max_weight)

    return weights

# 結果示例：
# Class 0 (501,814): weight = 0.1  (降權)
# Class 17 (12):     weight = 50.0 (升權 500x)
```

### 4.2 Focal Loss

```python
# 標準 Cross Entropy:
loss = -log(p)

# Focal Loss:
loss = -(1-p)^γ * log(p)

# γ = 2.0 時：
# - 高信心正確預測 (p=0.9): loss × 0.01 (忽略)
# - 低信心錯誤預測 (p=0.1): loss × 0.81 (關注)
```

### 4.3 過採樣策略

```python
# 對於 Class 17 (12 個樣本)
# 目標：至少 500 個有效訓練樣本

oversample_config = {
    17: {  # double_flat
        'repeat': 10,  # 每張圖重複 10 次
        'augmentation': 'strong',  # 強增強避免過擬合
    },
    16: {  # double_sharp
        'repeat': 5,
        'augmentation': 'medium',
    },
    # ...
}
```

---

## 五、Phase 3：合成數據生成（核心突破）

### 5.1 為什麼合成數據是關鍵

```
問題：DeepScoresV2 只有 12 個 double_flat
原因：真實樂譜中 double_flat 本就罕見
解法：自己生成包含 double_flat 的樂譜！
```

### 5.2 LilyPond 合成管道

```lilypond
% 生成包含重降記號的練習
\relative c' {
  \key c \major
  c4 deses e f |    % deses = D double-flat
  g aeses b c |     % aeses = A double-flat
}
```

### 5.3 合成數據規格

```yaml
target_distribution:
  # 每個類別目標樣本數
  notehead_filled: 100,000  # 保持
  notehead_hollow: 50,000   # 保持
  accidental_double_flat: 10,000  # 從 12 → 10,000
  accidental_double_sharp: 10,000 # 從 338 → 10,000
  clef_tenor: 5,000         # 從 614 → 5,000
  # ...

variations:
  - fonts: [emmentaler, gonville, beethoven, feta, lilyjazz, bravura]
  - sizes: [0.8x, 1.0x, 1.2x]
  - rotations: [-3°, 0°, +3°]
  - noise: [none, light, medium]
  - blur: [none, gaussian_1, gaussian_2]
```

### 5.4 自動標註

```python
# LilyPond 輸出包含精確座標
# 可以自動轉換為 YOLO 格式

def lilypond_to_yolo(ly_output, image_size):
    """
    從 LilyPond 輸出提取標註
    """
    annotations = []
    for element in ly_output.elements:
        bbox = element.bounding_box
        cls = map_to_yolo_class(element.type)
        annotations.append(to_yolo_format(cls, bbox, image_size))
    return annotations
```

---

## 六、Phase 4：高解析度 + 架構升級

### 6.1 解析度策略

```
當前：640 × 640
問題：樂譜細節豐富，640 可能不夠

實驗計畫：
├── 640px: 基準 (當前)
├── 960px: +50% 解析度
└── 1280px: +100% 解析度 (RTX 5090 可支援)

預期：每提升一級，mAP +3-5%
```

### 6.2 模型架構比較

| 模型 | 參數量 | mAP (估計) | 手機推論時間 | 建議 |
|------|--------|-----------|--------------|------|
| YOLO12n | 3M | 0.75-0.80 | 30ms | 低階機 |
| YOLO12s | 11M | 0.80-0.85 | 50ms | 中階機 |
| YOLO12m | 25M | 0.85-0.90 | 100ms | 高階機 |
| YOLO12l | 43M | 0.88-0.92 | 200ms | 僅訓練 |

### 6.3 知識蒸餾

```
策略：
1. 用 YOLO12l 訓練出最強模型 (Teacher)
2. 用 Teacher 指導 YOLO12s 訓練 (Student)
3. Student 獲得接近 Teacher 的性能，但更快
```

---

## 七、Phase 5：真實數據整合

### 7.1 數據來源

| 來源 | 類型 | 數量 | 整合難度 |
|------|------|------|----------|
| OpenOMR | 掃描樂譜 | 706 張 | 中 |
| MUSCIMA++ | 手寫樂譜 | 423 文件 | 高 |
| 自收集 | 四部和聲作業 | 待定 | 低 |
| 用戶貢獻 | 真實使用數據 | 持續增長 | 低 |

### 7.2 真實數據的價值

```
合成數據：完美、乾淨、可控
真實數據：有噪音、變形、實際使用場景

最強模型 = 合成數據（量）+ 真實數據（質）
```

---

## 八、Phase 6：生產優化

### 8.1 部署優化

```
TFLite INT8 量化：
├── 模型大小: ~50MB → ~12MB
├── 推論速度: 提升 2-4x
├── 準確度損失: < 2%
└── 記憶體使用: 降低 4x
```

### 8.2 測試時增強 (TTA)

```python
def inference_with_tta(model, image):
    """
    多角度推論，提升準確度
    """
    predictions = []
    for scale in [0.9, 1.0, 1.1]:
        for flip in [False, True]:
            augmented = augment(image, scale, flip)
            pred = model(augmented)
            predictions.append(reverse_augment(pred, scale, flip))

    return ensemble(predictions)
```

### 8.3 後處理規則

```python
def post_process(detections):
    """
    利用音樂知識優化檢測結果
    """
    # 規則 1: 音符必須在五線譜附近
    # 規則 2: 譜號必須在行首
    # 規則 3: 調號必須在譜號後
    # 規則 4: 同一位置不能有多個音符頭
    # ...
```

---

## 九、資源需求

### 9.1 計算資源

| 階段 | GPU 時間 | 說明 |
|------|----------|------|
| Phase 1 | 2-3 天 | 當前進行中 |
| Phase 2 | 3-5 天 | 類別平衡訓練 |
| Phase 3 | 1-2 週 | 大量合成數據訓練 |
| Phase 4 | 1-2 週 | 高解析度 + 大模型 |
| Phase 5 | 1 週 | 真實數據微調 |
| Phase 6 | 3-5 天 | 優化和部署 |

### 9.2 人力需求

| 任務 | 時間估計 | 技能 |
|------|----------|------|
| 合成數據管道 | 1-2 週 | Python + LilyPond |
| 數據標註驗證 | 持續 | 音樂知識 |
| 模型訓練監控 | 持續 | ML 經驗 |
| 移動端整合 | 2-3 週 | Android + TFLite |

---

## 十、風險與緩解

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| 合成數據與真實差距大 | 中 | 高 | 混合訓練 + 真實數據微調 |
| 稀有類別仍然學不好 | 中 | 高 | 專門的稀有類別分類器 |
| 手機推論太慢 | 低 | 中 | 更激進的量化 + 小模型 |
| 過擬合 | 中 | 中 | 強正則化 + 早停 |

---

## 十一、成功指標

### 11.1 量化指標

```
Milestone 1 (Phase 1 完成): mAP50 > 0.55
Milestone 2 (Phase 2 完成): mAP50 > 0.65, 所有類別 > 0.30
Milestone 3 (Phase 3 完成): mAP50 > 0.75, 核心類別 > 0.70
Milestone 4 (Phase 4 完成): mAP50 > 0.85, 核心類別 > 0.80
Milestone 5 (Phase 5 完成): mAP50 > 0.90, 核心類別 > 0.85
```

### 11.2 實際測試

```
測試集：
├── 50 張印刷四部和聲練習 (各出版社)
├── 50 張掃描樂譜 (不同掃描品質)
├── 50 張手寫四部和聲作業 (真實學生作品)
└── 50 張手機拍攝照片 (各種光線角度)

通過標準：
├── 印刷：95% 符號正確辨識
├── 掃描：90% 符號正確辨識
├── 手寫：80% 符號正確辨識
└── 拍照：85% 符號正確辨識
```

---

## 十二、立即行動項目

### 現在可以開始準備的：

1. **合成數據生成腳本** - 不影響當前訓練
2. **類別加權訓練配置** - Phase 1 完成後立即使用
3. **OpenOMR 轉換腳本** - 準備數據整合
4. **真實測試集收集** - 開始收集四部和聲作業照片

### 等待 Phase 1 完成後：

1. 執行 per-class mAP 分析
2. 決定是否需要調整類別優先級
3. 啟動 Phase 2 類別平衡訓練

---

## 結論

「無敵」模型不是一次訓練能達成的，而是需要：

1. **正確的優先級**：專注於和聲分析真正需要的類別
2. **數據為王**：合成數據解決稀有類別問題
3. **迭代優化**：每個 Phase 都要驗證和調整
4. **真實驗證**：最終用真實四部和聲作業測試

按照這個路線圖，3-6 個月內可以達到生產級準確度。

---

*文檔版本：1.0*
*創建日期：2025-11-22*
*最後更新：2025-11-22*
