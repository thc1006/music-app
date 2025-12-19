# 🎯 ULTIMATE PERFECTION ROADMAP
## 四部和聲 OMR 模型極致優化計劃

**創建日期**: 2025-12-02
**目標**: 實現業界頂尖的 OMR 模型（mAP50 > 0.85）
**預計總時間**: 4-8 週
**當前最佳**: Phase 8 - mAP50 0.6444

---

## 📊 深度分析總結

### 當前狀態診斷（Phase 8 分析）

#### ✅ **成功之處**

| 成就 | 數據 | 說明 |
|------|------|------|
| **整體性能** | mAP50 0.6444, mAP50-95 0.5809 | 已達業界中等水平 |
| **優秀類別（15個）** | mAP50 > 0.70 | clef系列、rest系列、dynamic_loud、barline_repeat |
| **穩定訓練** | 150 epochs 完整收斂 | 訓練損失持續下降 |
| **數據規模** | 32,555 訓練圖片 | 足夠的基礎數據 |

#### ❌ **關鍵瓶頸**

| 問題類型 | 嚴重程度 | 具體問題 | 影響範圍 |
|---------|---------|---------|---------|
| **1. 數據質量** | 🔴 CRITICAL | DoReMi 100% tiny bbox（4類），OpenScore 89% tiny fermata | 7個類別 |
| **2. 瓶頸類別** | 🔴 CRITICAL | 7個類別 mAP50 < 0.50 | 21% 類別 |
| **3. 未使用資源** | 🟠 HIGH | 8,726張高質量圖片未整合 | 潛在+26% 數據 |
| **4. 架構極限** | 🟡 MEDIUM | Phase 8 後期增長緩慢（80-150 epochs +0.5%） | 收斂瓶頸 |
| **5. 訓練策略** | 🟡 MEDIUM | Phase 9 配置不當導致退步 | 需優化 |

#### 🎯 **瓶頸類別詳細分析**（7個危急類別）

| 排名 | ID | 類別名稱 | mAP50 | 樣本數 | 根本原因 | 解決方案 |
|-----|----|---------:|------:|-------:|---------|---------|
| 1 | 24 | barline_double | **0.231** | 24,228 | 標註膨脹（67.8% > 0.1面積） | OpenScore +20K 標註 |
| 2 | 16 | double_sharp | **0.369** | 2,209 | 樣本稀少，分布不均 | 已有 Synthetic |
| 3 | 29 | fermata | **0.402** | 35,734 | 62% tiny bbox | OpenScore +24K |
| 4 | 8 | tie | **0.411** | 59,909 | 形狀變化大 | 需數據增強 |
| 5 | 4 | flag_8th | **0.418** | 40,484 | MUSCIMA 標註品質 | 清理+合成 |
| 6 | 27 | time_signature | **0.446** | 36,654 | Rebelo 數據品質 | 清理 |
| 7 | 32 | ledger_line | **0.466** | 246,506 | 極細線難檢測 | 高解析度訓練 |

#### 📦 **未使用的黃金資源**（8,726 張圖片）

| 數據集 | 圖片數 | 關鍵標註 | 預期提升 | 狀態 |
|--------|--------|----------|----------|------|
| **OpenScore Lieder** | 5,238 | barline_double: +20,190<br>fermata: +22,081 | barline_double: 0.23→**0.55**<br>fermata: 0.40→**0.70** | ✅ 已渲染 |
| **DeepScores Dynamics** | 700 | dynamic_soft: +3,884<br>dynamic_loud: +4,998 | dynamic_soft: 0.57→**0.75** | ✅ 已轉換 |
| **OpenScore Quartets** | 2,529 | fermata: +2,612 | fermata: 額外提升 | ✅ 已渲染 |
| **Synthetic Phase 8** | 5,940 | double_flat: +75,025<br>dynamic_loud: +30,239 | double_flat: 0.79→**0.90** | ✅ 已生成 |
| **DeepScores Fermata** | 147 | fermata: +1,712 | fermata: 額外提升 | 🟢 可用 |

---

## 🚀 終極訓練路線圖（Phase 10-15）

### 設計哲學：漸進式、針對性、系統化

```
當前狀態 (Phase 8)                                               最終目標
mAP50: 0.6444 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━➤ mAP50: 0.85+
       │                                                          │
       ├─ Phase 10: 數據質量修復   [+5-8%]  → 0.69-0.71        │
       ├─ Phase 11: 未使用數據整合 [+6-10%] → 0.75-0.78        │
       ├─ Phase 12: 架構優化      [+3-5%]  → 0.78-0.81        │
       ├─ Phase 13: 高解析度訓練   [+2-4%]  → 0.80-0.84        │
       ├─ Phase 14: 知識蒸餾      [+1-3%]  → 0.82-0.86        │
       └─ Phase 15: 生產優化      [持平]   → 0.82-0.86 (穩定) ◄┘
```

---

## 📋 Phase 10: 數據質量全面修復（Week 1）

### 目標
**修復所有數據質量問題，建立乾淨的訓練基礎**
- **預期 mAP50**: 0.644 → **0.69-0.71** (+5-8%)
- **預計時間**: 4-6 天
- **ROI**: ⭐⭐⭐⭐⭐（最高）

### 子任務

#### 10.1 DoReMi 數據修復（2天）

**問題診斷**（來自 PHASE8_COMPLETE_ANALYSIS.md）:
```
DoReMi 座標轉換錯誤導致：
- flag_16th:  14,535 樣本，100% tiny bbox → 幾乎全部無效
- flag_32nd:  5,915 樣本，100% tiny bbox → 幾乎全部無效
- rest_16th:  31,034 樣本，100% tiny bbox → 幾乎全部無效
- barline:    13,931 樣本，81.2% tiny → 大部分無效
```

**解決方案**:
```python
# scripts/fix_doremi_coordinates.py
"""
重新解析 DoReMi OMR XML，修正座標轉換邏輯
"""
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_doremi_xml_correct(xml_path):
    """
    正確解析 DoReMi XML 座標

    問題根源：
    1. 原始轉換腳本錯誤使用了 symbol 內部座標
    2. 應該使用 glyph bounding box 的絕對座標
    3. 座標系統可能需要縮放因子（檢查 resolution 標籤）
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. 獲取圖片尺寸和解析度
    img_width = int(root.find('.//ImageWidth').text)
    img_height = int(root.find('.//ImageHeight').text)
    resolution = float(root.find('.//Resolution').text)  # 可能需要這個

    bboxes = []

    # 2. 正確提取 glyph bounding boxes
    for glyph in root.findall('.//Glyph'):
        # 使用絕對座標，不是相對座標
        x = float(glyph.find('.//Left').text)
        y = float(glyph.find('.//Top').text)
        width = float(glyph.find('.//Width').text)
        height = float(glyph.find('.//Height').text)

        # 檢查是否需要縮放
        if resolution != 1.0:
            x *= resolution
            y *= resolution
            width *= resolution
            height *= resolution

        # 轉換為 YOLO 格式（normalized）
        x_center = (x + width/2) / img_width
        y_center = (y + height/2) / img_height
        w_norm = width / img_width
        h_norm = height / img_height

        # 驗證合理性
        if w_norm > 0.001 and h_norm > 0.001:  # 最小尺寸閾值
            bboxes.append({
                'class_id': map_symbol_to_class(glyph.find('.//SymbolType').text),
                'bbox': [x_center, y_center, w_norm, h_norm]
            })

    return bboxes
```

**執行步驟**:
```bash
# Day 1: 分析和修復腳本開發
cd ~/dev/music-app/training
python scripts/analyze_doremi_xml.py  # 分析原始 XML 結構
python scripts/fix_doremi_coordinates.py --dry-run  # 測試修復

# Day 2: 重新轉換 DoReMi 數據集
python scripts/fix_doremi_coordinates.py --execute
python validate_fixed_doremi.py  # 驗證修復效果
```

**預期改善**:
| 類別 | Phase 8 | Phase 10 預期 | 提升 |
|------|---------|--------------|------|
| flag_16th | 0.731 | **0.85+** | +16% |
| flag_32nd | 0.758 | **0.88+** | +16% |
| rest_16th | 0.900 | **0.95+** | +6% |
| barline | 0.574 | **0.70+** | +22% |

#### 10.2 OpenScore Fermata 重新渲染（1-2天）

**問題**: 89% tiny bbox（座標提取錯誤）

**解決方案**: 使用 Verovio 原生 bbox API
```python
# scripts/rerender_openscore_fermatas.py
import verovio

tk = verovio.toolkit()
tk.setOptions({
    "pageHeight": 2970,
    "pageWidth": 2100,
    "scale": 100,
    "breaks": "none"
})

# 載入 MusicXML
tk.loadFile("score.mxl")

# 獲取元素位置（Verovio 3.0+ 原生支持）
element_info = tk.getElementsAtTime(0)  # 逐時間點取得

# 或使用 SVG 精確解析
svg = tk.renderToSVG(1)
fermata_elements = extract_fermata_from_svg(svg)  # 正確提取座標
```

**預期改善**:
| 類別 | Phase 8 | Phase 10 預期 | 提升 |
|------|---------|--------------|------|
| fermata | 0.402 | **0.55-0.60** | +37-49% |

#### 10.3 MUSCIMA++ 標註標準化（1天）

**問題**: 標註尺度不一致（flag_8th, tie）

**解決方案**: 重新標準化 bbox 尺寸
```python
# scripts/standardize_muscima_annotations.py
"""
將 MUSCIMA++ 標註標準化到與主數據集一致的尺度
"""
def standardize_bbox_scale(bbox, class_id, reference_stats):
    """
    根據 reference_stats（從主數據集計算）調整 bbox 尺寸
    """
    ref_width = reference_stats[class_id]['median_width']
    ref_height = reference_stats[class_id]['median_height']

    # 如果 bbox 過大或過小，調整到合理範圍
    current_area = bbox['width'] * bbox['height']
    ref_area = ref_width * ref_height

    if current_area > ref_area * 3 or current_area < ref_area * 0.3:
        # 異常尺寸，需調整
        scale_factor = (ref_area / current_area) ** 0.5
        bbox['width'] *= scale_factor
        bbox['height'] *= scale_factor

    return bbox
```

#### 10.4 Tiny Bbox 智能清理（1天）

**關鍵洞察**（來自 PHASE9_CRITICAL_ANALYSIS.md）:
```
✅ 不要盲目清理所有 tiny bbox！
   - flag_16th: 84.3% tiny 但 mAP 0.731（高）→ 不應清理
   - fermata: 59.0% tiny 但 mAP 0.644（還可以）→ 不應清理
   - barline_double: 0.2% tiny 但 mAP 0.231（最差）→ tiny 不是問題

✅ 智能清理策略：
   - 只清理「已知錯誤來源」的 tiny bbox（DoReMi, OpenScore原版）
   - 保留合法的小物件（真實的 flag_16th, rest_16th）
```

**實施**:
```python
# scripts/smart_clean_tiny_annotations.py
"""
基於數據來源的智能清理
"""
def should_remove_annotation(bbox, class_id, source_dataset):
    """
    決定是否移除標註
    """
    area = bbox['width'] * bbox['height']

    # 規則 1: DoReMi 的已知錯誤類別 → 移除 tiny
    if source_dataset == 'DoReMi' and class_id in [4, 5, 6, 22, 23]:
        if area < 0.0005:
            return True, "DoReMi座標錯誤"

    # 規則 2: OpenScore 原版 fermata → 移除 tiny
    if source_dataset == 'OpenScore_original' and class_id == 29:
        if area < 0.0003:
            return True, "OpenScore座標錯誤"

    # 規則 3: 保留所有其他 tiny bbox（可能是真實小物件）
    return False, "合法小物件"
```

### Phase 10 訓練配置

```python
# yolo12_train_phase10.py
from ultralytics import YOLO

model = YOLO('harmony_omr_v2_phase8/phase8_training/weights/best.pt')  # 從 Phase 8 繼續

results = model.train(
    data='datasets/yolo_harmony_v2_phase10_clean/harmony_phase10.yaml',

    # 基礎配置
    epochs=150,
    patience=50,
    batch=24,
    imgsz=640,
    device=0,

    # 優化器（保持 Phase 8 成功配置）
    optimizer='AdamW',
    lr0=0.001,  # Phase 8 成功經驗
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,

    # 損失權重（針對瓶頸類別加權）
    cls=0.5,
    box=7.5,
    dfl=1.5,

    # 類別加權（針對修復後的類別）
    class_weights={
        4: 2.0,   # flag_8th（MUSCIMA 修復後）
        5: 2.0,   # flag_16th（DoReMi 修復後）
        6: 2.0,   # flag_32nd（DoReMi 修復後）
        22: 2.0,  # rest_16th（DoReMi 修復後）
        24: 3.0,  # barline_double（重點提升）
        29: 2.5,  # fermata（OpenScore 修復後）
    },

    # 數據增強（保持 Phase 8 配置）
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=2.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0001,
    flipud=0.0,
    fliplr=0.0,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,

    # 訓練策略
    amp=True,  # 混合精度訓練
    close_mosaic=10,
    pretrained=True,

    # 輸出
    project='harmony_omr_v2_phase10',
    name='clean_data_training',
    exist_ok=False,
    save=True,
    save_period=10,
    plots=True,
    val=True,
)
```

### Phase 10 成功標準

| 指標 | Phase 8 | Phase 10 目標 | 最低要求 |
|------|---------|--------------|---------|
| **整體 mAP50** | 0.644 | **0.71+** | 0.69 |
| **整體 mAP50-95** | 0.581 | **0.63+** | 0.61 |
| **瓶頸類別 mAP50** | 平均 0.39 | **平均 0.55+** | 平均 0.50 |
| **危急類別數量** | 7 | **≤3** | ≤5 |

---

## 📋 Phase 11: 未使用數據全面整合（Week 2）

### 目標
**整合所有未使用的高質量數據，最大化數據多樣性**
- **預期 mAP50**: 0.71 → **0.75-0.78** (+6-10%)
- **預計時間**: 4-6 天
- **ROI**: ⭐⭐⭐⭐⭐

### 數據整合策略

#### 11.1 漸進式整合（避免負遷移）

**Phase 9 失敗教訓**:
```
❌ Phase 9 一次性加入所有新數據 → 退步 9.4%
   - 域差異（Domain Shift）
   - 標註風格不一致
   - 模型混淆

✅ Phase 11 漸進式整合 → 預期成功
   - 每次只加入一個數據源
   - 訓練後評估效果
   - 有問題立即回退
```

**執行順序**（按風險排序）:

```
Phase 11A: + DeepScores Dynamics (700張)
    ↓ 訓練 50 epochs
Phase 11B: + Synthetic Phase 8 (5,940張)
    ↓ 訓練 50 epochs
Phase 11C: + OpenScore Lieder (5,238張，已修復 fermata)
    ↓ 訓練 100 epochs
Phase 11D: + OpenScore Quartets (2,529張)
    ↓ 訓練 50 epochs
Phase 11E: + DeepScores Fermata (147張)
    ↓ 訓練 50 epochs
```

**每階段評估檢查點**:
```python
def should_continue_integration(current_map, previous_map):
    """
    決定是否繼續整合下一個數據源
    """
    improvement = current_map - previous_map

    if improvement < -0.01:  # 退步 > 1%
        print("⚠️ 性能下降，停止整合，回退到上一版本")
        return False
    elif improvement > 0.02:  # 進步 > 2%
        print("✅ 顯著提升，繼續整合下一個數據源")
        return True
    else:  # 0-2% 提升
        print("➡️ 微小提升，謹慎繼續")
        return True
```

#### 11.2 各數據源整合細節

**11.2a DeepScores Dynamics**（最安全，優先整合）
```yaml
# 理由：
# - 高質量標註（專業團隊）
# - 中等規模（700張）
# - 針對性強（只有 dynamics）
# - 不會干擾其他類別

合併後數據集規模：
  訓練集：32,555 + 700 = 33,255 (+2.1%)

預期提升：
  dynamic_soft: 0.572 → 0.68+ (+19%)
  dynamic_loud: 0.903 → 0.92+ (+2%)
```

**11.2b Synthetic Phase 8**（高質量，優先度第二）
```yaml
# 理由：
# - 我們自己生成，品質可控
# - LilyPond 高品質渲染
# - 針對稀有類別（double_flat, dynamic_loud）
# - 領域隨機化已處理域差異

合併後數據集規模：
  訓練集：33,255 + 5,940 = 39,195 (+17.9%)

預期提升：
  double_flat: 0.788 → 0.90+ (+14%)
  dynamic_loud: 0.92 → 0.95+ (+3%)
```

**11.2c OpenScore Lieder**（高價值，需謹慎）
```yaml
# 理由：
# - 巨大價值（+20K barline_double, +22K fermata）
# - 但需確保 fermata 已正確修復（Phase 10.2）
# - 可能有域差異（藝術歌曲 vs 四部和聲）

合併前檢查：
  1. 確認 fermata 修復效果（抽樣100張）
  2. 驗證 barline_double 標註品質
  3. 檢查域差異（音符密度、風格）

合併後數據集規模：
  訓練集：39,195 + 5,238 = 44,433 (+13.4%)

預期提升：
  barline_double: 0.231 → 0.55+ (+138%)
  fermata: 0.60 (Phase 10後) → 0.75+ (+25%)
```

**11.2d OpenScore Quartets**（價值中等）
```yaml
# 理由：
# - 只有 fermata（2,612個）
# - 弦樂四重奏 = SATB 等價（非常契合）
# - 補充 OpenScore Lieder

合併後數據集規模：
  訓練集：44,433 + 2,529 = 46,962 (+5.7%)

預期提升：
  fermata: 0.75 → 0.78+ (+4%)
```

**11.2e DeepScores Fermata**（價值小，最後整合）
```yaml
# 理由：
# - 只有 147張圖片，1,712 fermata
# - 補充性質

合併後數據集規模：
  訓練集：46,962 + 147 = 47,109 (+0.3%)

預期提升：
  fermata: 0.78 → 0.80+ (+3%)
```

### Phase 11 訓練配置

```python
# 基本配置同 Phase 10
# 關鍵差異：

# 1. 更大的 batch size（數據增多）
batch = 28  # 從 24 提升到 28

# 2. 更長的訓練（更多數據需要更多 epochs）
epochs = 200  # 從 150 提升到 200

# 3. 更強的數據增強（處理域差異）
hsv_s = 0.8  # 從 0.7 提升（更多顏色變化）
scale = 0.6   # 從 0.5 提升（更多尺度變化）
mosaic = 1.0  # 保持（混合不同來源）

# 4. 動態類別加權（根據新增數據調整）
class_weights = {
    24: 5.0,  # barline_double（大量新數據）
    29: 3.0,  # fermata（大量新數據）
    30: 2.0,  # dynamic_soft（新數據）
    31: 1.5,  # dynamic_loud（已很好，微調）
    17: 1.5,  # double_flat（合成數據）
}

# 5. 知識保留（避免遺忘舊類別）
freeze = 0  # 不凍結任何層
pretrained = True  # 從 Phase 10 繼續
```

### Phase 11 成功標準

| 指標 | Phase 10 | Phase 11 目標 | 最低要求 |
|------|---------|--------------|---------|
| **整體 mAP50** | 0.71 | **0.77+** | 0.75 |
| **整體 mAP50-95** | 0.63 | **0.68+** | 0.66 |
| **barline_double** | 0.35 | **0.55+** | 0.50 |
| **fermata** | 0.60 | **0.80+** | 0.75 |
| **dynamic_soft** | 0.68 | **0.80+** | 0.75 |
| **危急類別數量** | 3 | **0** | ≤1 |

---

## 📋 Phase 12: YOLO 架構優化（Week 3）

### 目標
**突破架構瓶頸，引入 OMR 特定優化**
- **預期 mAP50**: 0.77 → **0.78-0.81** (+3-5%)
- **預計時間**: 5-7 天
- **ROI**: ⭐⭐⭐⭐

### 12.1 注意力機制整合

**背景**: OMR 任務特點
- 極度關注細節（旗號、臨時記號）
- 需要全局上下文（五線譜結構）
- 小物件密集排列

**方案**: CBAM（Convolutional Block Attention Module）

```python
# models/yolo12_omr_cbam.py
"""
為 YOLO12 添加 CBAM 注意力機制
專門針對 OMR 小物件檢測優化
"""
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# 在 YOLO12 backbone 的關鍵位置插入 CBAM
# - P3 層（小物件檢測）
# - P4 層（中等物件）
# - Neck 部分（特徵融合）
```

**預期效果**:
- 小物件類別（flag, rest_16th, ledger_line）: +3-5% mAP50
- 細節類別（accidental）: +2-4% mAP50

### 12.2 多尺度訓練增強

```python
# Phase 12 訓練配置
model.train(
    # 多尺度輸入（針對 OMR 中小物件多的特點）
    imgsz=640,  # 基礎解析度
    multi_scale=True,  # 啟用多尺度
    scale=0.5,  # 縮放範圍 0.5-1.5x

    # 混合尺度訓練策略（自定義）
    custom_scales=[512, 640, 768, 896],  # 訓練時隨機選擇
    scale_strategy='dynamic',  # 動態調整策略

    # 針對小物件的增強
    min_object_size=0.0005,  # 最小物件尺寸閾值（保留 flag 等）
    small_object_augment=True,  # 小物件特殊增強
)
```

### 12.3 損失函數優化

**背景**: OMR 任務不平衡
- 33 個類別，分布極度不均（最多 737K vs 最少 2K）
- 小物件多（flag, rest_16th）

**方案**: Focal Loss + Quality Focal Loss

```python
# models/omr_loss.py
"""
OMR 特定損失函數
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss: 聚焦難分類樣本
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss: 結合分類質量和定位質量
    適合 OMR 中需要精確定位的任務（barline, fermata）
    """
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, score):
        # pred: 分類預測
        # target: 分類標籤
        # score: IoU 質量分數
        weight = (score - pred.sigmoid()).abs().pow(self.beta)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
        return loss.mean()

# 整合到 YOLO12
def omr_loss_function(pred, target, iou):
    cls_loss = FocalLoss()(pred['cls'], target['cls'])
    box_loss = QualityFocalLoss()(pred['box'], target['box'], iou)
    return cls_loss + box_loss
```

### 12.4 後處理優化

**NMS 優化（針對 OMR 密集排列）**:
```python
# postprocess/omr_nms.py
"""
OMR 特定 NMS（Non-Maximum Suppression）
處理密集排列的音符和符號
"""
def omr_smart_nms(predictions, conf_thres=0.25, iou_thres=0.45, music_aware=True):
    """
    音樂感知的 NMS

    特殊規則：
    1. 音符頭不抑制 stem（它們應該共存）
    2. Barline 不抑制其他 barline（可能密集排列）
    3. 臨時記號允許與音符重疊
    """
    if not music_aware:
        return standard_nms(predictions, conf_thres, iou_thres)

    # 音樂符號關係圖
    coexist_rules = {
        (0, 2): True,  # notehead_filled 與 stem 共存
        (1, 2): True,  # notehead_hollow 與 stem 共存
        (13, 0): True,  # accidental_sharp 與 notehead 共存
        (14, 0): True,  # accidental_flat 與 notehead 共存
        (15, 0): True,  # accidental_natural 與 notehead 共存
        # ... 更多規則
    }

    output = []

    for prediction in predictions:
        # 按置信度排序
        sorted_pred = sorted(prediction, key=lambda x: x['conf'], reverse=True)

        keep = []
        for i, box1 in enumerate(sorted_pred):
            suppress = False
            for j in keep:
                box2 = sorted_pred[j]

                # 檢查是否應該共存
                class_pair = (box1['class'], box2['class'])
                if coexist_rules.get(class_pair, False):
                    continue  # 跳過抑制

                # 計算 IoU
                iou = compute_iou(box1['bbox'], box2['bbox'])
                if iou > iou_thres:
                    suppress = True
                    break

            if not suppress:
                keep.append(i)

        output.append([sorted_pred[i] for i in keep])

    return output
```

### Phase 12 訓練配置

```python
# yolo12_train_phase12.py
from ultralytics import YOLO
from models.yolo12_omr_cbam import YOLO12_OMR_CBAM

# 使用改進的架構
model = YOLO12_OMR_CBAM('harmony_omr_v2_phase11/weights/best.pt')

results = model.train(
    data='datasets/yolo_harmony_v2_phase11/harmony_phase11.yaml',

    # 基礎配置
    epochs=150,
    patience=50,
    batch=28,
    imgsz=640,
    device=0,

    # 多尺度訓練
    multi_scale=True,
    scale=0.5,
    custom_scales=[512, 640, 768, 896],

    # 改進的損失函數
    loss='omr_focal',  # 使用自定義損失
    focal_alpha=0.25,
    focal_gamma=2.0,
    qfl_beta=2.0,

    # CBAM 注意力
    cbam=True,
    cbam_ratio=16,
    cbam_kernel=7,

    # 改進的後處理
    nms_mode='music_aware',
    conf=0.25,
    iou=0.45,

    # 其他配置保持 Phase 11
    optimizer='AdamW',
    lr0=0.0008,  # 略降（微調）
    # ...
)
```

### Phase 12 成功標準

| 指標 | Phase 11 | Phase 12 目標 | 最低要求 |
|------|---------|--------------|---------|
| **整體 mAP50** | 0.77 | **0.81+** | 0.79 |
| **整體 mAP50-95** | 0.68 | **0.72+** | 0.70 |
| **小物件平均 mAP50** | ~0.70 | **0.80+** | 0.75 |
| **細節類別平均 mAP50** | ~0.65 | **0.75+** | 0.70 |

---

## 📋 Phase 13: 高解析度訓練（Week 4）

### 目標
**針對小物件和細線類別進行高解析度訓練**
- **預期 mAP50**: 0.81 → **0.80-0.84** (+2-4%)
- **預計時間**: 4-5 天
- **ROI**: ⭐⭐⭐

### 13.1 漸進式解析度提升策略

**背景**: 直接高解析度訓練的問題
- GPU 記憶體限制（RTX 5090 24GB）
- 訓練時間大幅增加
- 可能過擬合高解析度

**方案**: 漸進式解析度提升 + 知識遷移

```
Phase 13A: 640 → 768 (50 epochs)
    ↓
Phase 13B: 768 → 896 (50 epochs)
    ↓
Phase 13C: 896 → 1024 (50 epochs)
```

### 13.2 高解析度訓練配置

```python
# Phase 13A: 640 → 768
model = YOLO12_OMR_CBAM('phase12/weights/best.pt')

results_13a = model.train(
    data='datasets/yolo_harmony_v2_phase11/harmony_phase11.yaml',

    epochs=50,
    imgsz=768,  # 提升到 768
    batch=20,   # 降低 batch（記憶體限制）

    # 關鍵：從 640 模型遷移
    pretrained=True,
    freeze=0,  # 不凍結

    # 其他配置同 Phase 12
)

# Phase 13B: 768 → 896
results_13b = model.train(
    imgsz=896,
    batch=16,  # 進一步降低
    epochs=50,
    # ...
)

# Phase 13C: 896 → 1024
results_13c = model.train(
    imgsz=1024,
    batch=12,  # 最小 batch
    epochs=50,
    # ...
)
```

### 13.3 高解析度特定優化

```python
# 針對小物件的增強
augment_config = {
    # 禁用會降低細節的增強
    'mixup': 0.0,  # 不使用 mixup（會模糊細節）
    'erasing': 0.0,  # 不使用 erasing
    'mosaic': 0.5,  # 降低 mosaic（保留細節）

    # 保留細節的增強
    'degrees': 1.0,  # 降低旋轉（避免細線失真）
    'translate': 0.05,  # 降低平移
    'scale': 0.3,  # 降低縮放

    # 針對小物件的特殊處理
    'small_object_preserve': True,  # 保護小物件
    'min_visibility': 0.8,  # 確保增強後仍可見
}
```

### 13.4 針對性類別提升

**目標類別**（高解析度將最大受益）:
1. **ledger_line** (ID 32): 極細線，mAP50 0.466
2. **barline** (ID 23): 細線，mAP50 0.574
3. **barline_double** (ID 24): 細線組合，mAP50 0.55
4. **flag_8th/16th/32nd** (ID 4/5/6): 小旗號細節
5. **tie** (ID 8): 細曲線，mAP50 0.411

**策略**: 針對性加權
```python
class_weights = {
    32: 3.0,  # ledger_line（最大受益者）
    23: 2.0,  # barline
    24: 2.5,  # barline_double
    4: 2.0,   # flag_8th
    5: 2.0,   # flag_16th
    6: 2.0,   # flag_32nd
    8: 2.5,   # tie
}
```

### Phase 13 成功標準

| 指標 | Phase 12 | Phase 13 目標 | 最低要求 |
|------|---------|--------------|---------|
| **整體 mAP50** | 0.81 | **0.84+** | 0.82 |
| **整體 mAP50-95** | 0.72 | **0.75+** | 0.73 |
| **ledger_line** | 0.466 | **0.65+** | 0.60 |
| **tie** | 0.411 | **0.60+** | 0.55 |
| **barline** | 0.70 | **0.80+** | 0.75 |

---

## 📋 Phase 14: 知識蒸餾與集成學習（Week 5-6）

### 目標
**通過知識蒸餾和集成學習，壓榨最後的性能提升**
- **預期 mAP50**: 0.84 → **0.82-0.86** (+1-3%)
- **預計時間**: 7-10 天
- **ROI**: ⭐⭐⭐

### 14.1 教師-學生架構

**教師模型**: YOLO12l + Phase 13 權重（大模型）
**學生模型**: YOLO12s（生產部署模型）

```python
# Phase 14 知識蒸餾
from models.knowledge_distillation import KnowledgeDistillation

# 教師模型（高性能但大）
teacher = YOLO12l('phase13/weights/best_1024.pt')
teacher.eval()

# 學生模型（輕量但性能略低）
student = YOLO12s('yolo12s.pt')

# 蒸餾訓練
kd_trainer = KnowledgeDistillation(
    teacher=teacher,
    student=student,
    temperature=4.0,  # 蒸餾溫度
    alpha=0.7,  # 蒸餾損失權重
    beta=0.3,   # 硬標籤損失權重
)

results = kd_trainer.train(
    data='datasets/yolo_harmony_v2_phase11/harmony_phase11.yaml',
    epochs=100,
    imgsz=640,  # 學生模型使用標準解析度
    batch=32,
    # ...
)
```

**預期效果**:
- 學生模型（YOLO12s）達到接近教師模型（YOLO12l）的性能
- 模型大小：~40MB（適合移動部署）
- 推理速度：~20ms（適合實時應用）

### 14.2 模型集成（Ensemble）

**策略**: 集成多個檢查點
```python
# 集成 Phase 12/13/14 的最佳模型
models = [
    YOLO12_OMR_CBAM('phase12/weights/best.pt'),  # CBAM版本
    YOLO12_OMR_CBAM('phase13/weights/best_768.pt'),  # 768解析度
    YOLO12_OMR_CBAM('phase13/weights/best_1024.pt'),  # 1024解析度
]

# 加權集成預測
def ensemble_predict(image, models, weights=[0.3, 0.3, 0.4]):
    predictions = []
    for model, weight in zip(models, weights):
        pred = model(image)
        pred['conf'] *= weight  # 加權置信度
        predictions.append(pred)

    # 合併預測（加權 NMS）
    final_pred = weighted_nms(predictions, iou_thres=0.5)
    return final_pred
```

**預期效果**:
- 集成模型 mAP50 > 單模型 +1-2%
- 但推理時間增加 3x（不適合生產，用於驗證上限）

### Phase 14 成功標準

| 指標 | Phase 13 | Phase 14 目標 | 最低要求 |
|------|---------|--------------|---------|
| **教師模型 mAP50** | 0.84 | **0.86+** | 0.85 |
| **學生模型 mAP50** | - | **0.83+** | 0.82 |
| **學生模型大小** | - | **<50MB** | <60MB |
| **學生推理時間** | - | **<25ms** | <30ms |

---

## 📋 Phase 15: 生產優化與穩定性（Week 7-8）

### 目標
**準備生產部署，確保穩定性和魯棒性**
- **預期 mAP50**: 0.82-0.86（持平或微提升）
- **預計時間**: 7-10 天
- **ROI**: ⭐⭐⭐⭐（生產就緒）

### 15.1 TFLite 量化

```python
# export_tflite.py
from ultralytics import YOLO

model = YOLO('phase14/weights/student_best.pt')

# INT8 量化
model.export(
    format='tflite',
    imgsz=640,
    int8=True,  # INT8 量化
    data='datasets/yolo_harmony_v2_phase11/harmony_phase11.yaml',  # 校準數據集
    batch=1,
    optimize=True,
)

# 驗證量化損失
# 預期：mAP50 下降 < 2%
```

### 15.2 多場景魯棒性測試

```python
# 測試場景
test_scenarios = [
    '正常光照',
    '弱光環境',
    '過曝光',
    '模糊圖片',
    '傾斜樂譜',
    '手機拍照（低質量）',
    '掃描圖片（高質量）',
    '不同紙張顏色',
    '筆跡和修改',
]

# 針對性數據增強訓練
# 確保各場景下性能穩定
```

### 15.3 Android 整合測試

```bash
# 在多款 Android 設備上測試
devices = [
    'Pixel 8 Pro (SD 8 Gen 3)',  # 高階
    'Samsung A54 (Exynos 1380)',  # 中階
    'Redmi Note 12 (SD 4 Gen 1)',  # 入門
]

# 測試指標：
# - 推理時間
# - 記憶體使用
# - 準確度
# - 電池消耗
```

### Phase 15 成功標準

| 指標 | 目標 |
|------|------|
| **TFLite INT8 mAP50** | > 0.80（量化損失 < 3%） |
| **模型大小** | < 20MB（INT8） |
| **推理時間（高階機）** | < 50ms |
| **推理時間（中階機）** | < 150ms |
| **推理時間（入門機）** | < 500ms |
| **多場景平均 mAP50** | > 0.75 |

---

## 📊 總結：預期成果路線圖

### 性能提升預測

```
階段對比圖：

Phase 8  (基準)         0.644 ━━━━━━━━━━━━━━┫
Phase 10 (數據修復)      0.710 ━━━━━━━━━━━━━━━━━━━┫ +10.2%
Phase 11 (數據整合)      0.770 ━━━━━━━━━━━━━━━━━━━━━━━━━┫ +8.5%
Phase 12 (架構優化)      0.810 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫ +5.2%
Phase 13 (高解析度)      0.840 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫ +3.7%
Phase 14 (知識蒸餾)      0.855 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫ +1.8%
Phase 15 (生產優化)      0.850 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫ -0.6%

總提升：+32.0%（0.644 → 0.850）
```

### 各類別預期表現

| 類別類型 | Phase 8 平均 | Phase 15 預期 | 提升 |
|---------|-------------|--------------|------|
| **優秀類別**（15個，>0.70） | 0.826 | **0.90+** | +9.0% |
| **良好類別**（8個，0.60-0.70） | 0.641 | **0.82+** | +27.9% |
| **待改進類別**（3個，0.50-0.60） | 0.544 | **0.75+** | +37.9% |
| **警告類別**（5個，0.40-0.50） | 0.439 | **0.68+** | +54.9% |
| **危急類別**（2個，<0.40） | 0.300 | **0.65+** | +116.7% |

### 關鍵瓶頸類別突破

| 類別 | Phase 8 | Phase 15 預期 | 提升幅度 |
|------|---------|--------------|---------|
| **barline_double** | 0.231 | **0.70+** | **+203%** 🚀 |
| **double_sharp** | 0.369 | **0.75+** | **+103%** 🚀 |
| **fermata** | 0.402 | **0.85+** | **+111%** 🚀 |
| **tie** | 0.411 | **0.70+** | **+70%** 📈 |
| **flag_8th** | 0.418 | **0.75+** | **+79%** 📈 |
| **time_signature** | 0.446 | **0.70+** | **+57%** 📈 |
| **ledger_line** | 0.466 | **0.75+** | **+61%** 📈 |

---

## 🎯 執行優先級與風險管理

### 優先級矩陣

| Phase | ROI | 風險 | 優先級 | 建議 |
|-------|-----|------|--------|------|
| **Phase 10** | ⭐⭐⭐⭐⭐ | 🟢 低 | 🔥 P0 | **立即執行** |
| **Phase 11** | ⭐⭐⭐⭐⭐ | 🟡 中 | 🔥 P0 | **立即執行** |
| **Phase 12** | ⭐⭐⭐⭐ | 🟡 中 | 🟢 P1 | 週計劃 |
| **Phase 13** | ⭐⭐⭐ | 🟠 高 | 🟡 P2 | 評估後決定 |
| **Phase 14** | ⭐⭐⭐ | 🟠 高 | 🟡 P2 | 可選 |
| **Phase 15** | ⭐⭐⭐⭐ | 🟢 低 | 🟢 P1 | 必須（生產） |

### 風險評估

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| **DoReMi 修復失敗** | 中 | 高 | 先小規模測試，有問題立即回退 |
| **Phase 11 負遷移** | 中 | 高 | 漸進式整合，每步驗證 |
| **高解析度 OOM** | 高 | 中 | 降低 batch size，分階段提升 |
| **知識蒸餾效果不佳** | 中 | 中 | 保留 Phase 13 模型作為備選 |
| **生產部署失敗** | 低 | 高 | 充分測試，多設備驗證 |

---

## 📅 時間表（8週完整計劃）

### Week 1: Phase 10（數據修復）
- **Day 1-2**: DoReMi 座標修復
- **Day 3-4**: OpenScore Fermata 重新渲染
- **Day 5**: MUSCIMA++ 標準化 + Tiny Bbox 清理
- **Day 6-7**: Phase 10 訓練（150 epochs，~9h）

### Week 2: Phase 11（數據整合）
- **Day 1**: Phase 11A - DeepScores Dynamics（50 epochs，3h）
- **Day 2**: Phase 11B - Synthetic Phase 8（50 epochs，3h）
- **Day 3-4**: Phase 11C - OpenScore Lieder（100 epochs，6h）
- **Day 5**: Phase 11D - OpenScore Quartets（50 epochs，3h）
- **Day 6**: Phase 11E - DeepScores Fermata（50 epochs，3h）
- **Day 7**: 驗證與分析

### Week 3: Phase 12（架構優化）
- **Day 1-2**: CBAM 整合與測試
- **Day 3**: Focal Loss 實施
- **Day 4**: OMR NMS 優化
- **Day 5-7**: Phase 12 訓練（150 epochs，~9h）

### Week 4: Phase 13（高解析度）
- **Day 1-2**: Phase 13A - 640→768（50 epochs，4h）
- **Day 3-4**: Phase 13B - 768→896（50 epochs，5h）
- **Day 5-6**: Phase 13C - 896→1024（50 epochs，6h）
- **Day 7**: 評估與對比

### Week 5-6: Phase 14（知識蒸餾，可選）
- **Week 5**: 教師模型優化
- **Week 6**: 學生模型蒸餾訓練

### Week 7-8: Phase 15（生產優化）
- **Week 7**: TFLite 量化 + 多場景測試
- **Week 8**: Android 整合 + 最終驗證

---

## 🚀 立即行動計劃

### 第一步：Phase 10 啟動（今天開始）

```bash
cd ~/dev/music-app/training

# 1. 創建 Phase 10 工作目錄
mkdir -p phase10_data_quality_fix
cd phase10_data_quality_fix

# 2. 複製並修改 DoReMi 轉換腳本
cp ../scripts/convert_doremi.py ../scripts/fix_doremi_coordinates.py

# 3. 分析原始 DoReMi XML（理解問題）
python -c "
from pathlib import Path
import xml.etree.ElementTree as ET

# 隨機抽取3個 DoReMi XML 文件分析
doremi_dir = Path('../datasets/external/omr_downloads/DoReMi/DoReMi_v1/OMR_XML')
sample_files = list(doremi_dir.glob('*.xml'))[:3]

for xml_file in sample_files:
    print(f'\\n分析文件: {xml_file.name}')
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 檢查關鍵標籤
    print(f'  ImageWidth: {root.find(\".//ImageWidth\").text}')
    print(f'  ImageHeight: {root.find(\".//ImageHeight\").text}')

    # 檢查 Glyph 結構
    glyph = root.find('.//Glyph')
    if glyph:
        print(f'  Sample Glyph:')
        for child in glyph:
            print(f'    {child.tag}: {child.text}')
"

# 4. 開始修復（預計2天）
echo "✅ Phase 10 啟動成功！開始數據修復..."
```

---

## 📚 參考資料與研究

### YOLO 優化相關
- [YOLO-NAS: Neural Architecture Search](https://arxiv.org/abs/2305.12728)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)

### OMR 特定研究
- [End-to-End OMR with Attention](https://arxiv.org/abs/1804.03535)
- [Deep Learning for Music Object Detection](https://arxiv.org/abs/2107.13099)
- [OMR Datasets: A Survey](https://arxiv.org/abs/2009.05670)

### 數據增強
- [Domain Randomization for Sim2Real Transfer](https://arxiv.org/abs/1703.06907)
- [Augmentation Strategies for Training](https://arxiv.org/abs/1906.11172)

---

## ✅ 成功檢查清單

### Phase 10
- [ ] DoReMi 座標修復完成（flag_16th/32nd, rest_16th, barline）
- [ ] OpenScore Fermata 重新渲染完成
- [ ] MUSCIMA++ 標註標準化完成
- [ ] Tiny Bbox 智能清理完成
- [ ] Phase 10 訓練完成（mAP50 > 0.69）
- [ ] 所有危急類別 mAP50 > 0.50

### Phase 11
- [ ] DeepScores Dynamics 整合成功
- [ ] Synthetic Phase 8 整合成功
- [ ] OpenScore Lieder 整合成功（無負遷移）
- [ ] OpenScore Quartets 整合成功
- [ ] Phase 11 訓練完成（mAP50 > 0.75）
- [ ] barline_double mAP50 > 0.55
- [ ] fermata mAP50 > 0.75

### Phase 12
- [ ] CBAM 整合成功（小物件類別提升）
- [ ] Focal Loss 實施完成
- [ ] OMR NMS 優化完成
- [ ] Phase 12 訓練完成（mAP50 > 0.79）

### Phase 13
- [ ] 768 解析度訓練完成
- [ ] 896 解析度訓練完成
- [ ] 1024 解析度訓練完成
- [ ] ledger_line mAP50 > 0.60
- [ ] tie mAP50 > 0.55
- [ ] Phase 13 訓練完成（mAP50 > 0.82）

### Phase 14（可選）
- [ ] 知識蒸餾訓練完成
- [ ] 學生模型 mAP50 > 0.82
- [ ] 模型大小 < 50MB
- [ ] 推理時間 < 25ms

### Phase 15
- [ ] TFLite INT8 量化完成（損失 < 3%）
- [ ] 多場景測試完成（平均 mAP50 > 0.75）
- [ ] Android 多設備測試通過
- [ ] 生產部署就緒

---

## 🎓 結語

這個路線圖是基於對當前 Phase 8/9 的深度分析，結合最新的 YOLO 和 OMR 研究，以及實際數據集的詳細評估而制定的。

**核心策略**：
1. ✅ **先修復數據質量**（Phase 10）- 打好基礎
2. ✅ **再整合未使用數據**（Phase 11）- 擴大規模
3. ✅ **然後優化架構**（Phase 12）- 提升能力
4. ✅ **接著高解析度訓練**（Phase 13）- 針對性突破
5. ✅ **最後蒸餾和生產優化**（Phase 14-15）- 實用化

**預期成果**：
- **整體 mAP50**: 0.644 → **0.85+** （+32%）
- **危急類別**: 全部解決（mAP50 > 0.65）
- **生產就緒**: TFLite INT8, <20MB, <500ms（入門機）

**時間投入**: 4-8 週（取決於是否執行 Phase 14）
**資源需求**: RTX 5090 GPU, 8TB 儲存空間

**立即開始**: Phase 10 - 數據質量修復 🚀

---

**創建時間**: 2025-12-02
**作者**: Claude Code (Opus 4.5)
**版本**: v1.0
**狀態**: 📋 規劃完成，待執行
