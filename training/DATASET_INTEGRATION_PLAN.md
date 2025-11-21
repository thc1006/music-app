# 📊 YOLO12 音樂符號檢測 - 數據集整合評估與執行計畫

## 🔍 一、當前數據使用狀況（真實評估）

### 1.1 實際使用情況

**目前訓練使用的數據**：
- **唯一來源**：DeepScoresV2 數據集
- **總量**：1,362 張圖片（1,157 訓練 + 205 驗證）
- **標註**：889,833 個標註框
- **類別**：208 類 → 映射到 35 類

**數據增強方式**：
- DeepScoresV2 內建字體增強（5 種字體風格）
  - beethoven（古典風格）
  - lilyjazz（爵士風格）
  - emmentaler（LilyPond 預設）
  - gonville（替代字體）
  - gutenberg1939（復古風格）

### 1.2 誤解澄清

❌ **錯誤說法**：「只用了 30% 的數據」
✅ **實際情況**：
- DeepScoresV2：**100% 使用**（所有 1,362 張訓練圖片）
- MUSCIMA++：**0% 使用**（只是映射表引用了共享類別 ID）
- PrIMuS：**0% 使用**（未整合）

**關於 MUSCIMA++ 的澄清**：
- 映射表中的 "MUSCIMA++" 註解（如 Class 142-208）是指 DeepScoresV2 與 MUSCIMA++ 的**共享類別定義**
- 實際訓練數據**完全來自 DeepScoresV2**，並非混合數據集

## 🎯 二、其他數據集可整合性分析

### 2.1 現有但未使用的數據集

#### **MUSCIMA++ (331MB)**
- **格式**：XML 標註 + PNG 圖片
- **特點**：手寫樂譜，423 個標註文件
- **整合難度**：⭐⭐⭐⭐ (高)
- **挑戰**：
  - XML 格式需要專門解析器
  - 類別定義與 DeepScores 不完全一致
  - 手寫風格與印刷樂譜差異大
- **預期收益**：中等（可能降低印刷樂譜的準確度）

#### **PrIMuS (1.1GB)**
- **格式**：獨特的樂譜表示格式
- **特點**：87,678 張合成樂譜
- **整合難度**：⭐⭐⭐⭐⭐ (極高)
- **挑戰**：
  - 使用符號序列而非 bounding box
  - 需要複雜的座標轉換
  - 合成數據可能與真實樂譜有差距
- **預期收益**：低（格式轉換成本高，質量不確定）

### 2.2 可下載但需評估的數據集

| 數據集 | 規模 | 格式 | 整合難度 | 預期收益 | 建議 |
|--------|------|------|----------|----------|------|
| **OpenOMR** | 706 張 | XML/JSON | ⭐⭐⭐ | 高（有 stem 方向） | ✅ 推薦 |
| **HOMUS** | 15,200 符號 | 單符號圖片 | ⭐⭐ | 中（類別平衡） | ⚠️ 考慮 |
| **Audiveris** | ~500 張 | 自定格式 | ⭐⭐⭐⭐ | 低 | ❌ 不推薦 |
| **Rebelo** | 15,000 符號 | 單符號 | ⭐⭐ | 低（過於簡化） | ❌ 不推薦 |
| **CVC-MUSCIMA** | 1,000 張 | Binary masks | ⭐⭐⭐ | 中 | ⚠️ 考慮 |

## 💡 三、深度可行性評估

### 3.1 技術挑戰

1. **格式不統一問題**
   ```python
   # 每個數據集需要不同的轉換器
   converters = {
       'deepscores': parse_json_coco_format,
       'muscima': parse_xml_with_coordinates,
       'primus': parse_semantic_representation,
       'openomr': parse_mei_xml
   }
   ```

2. **類別映射複雜度**
   - DeepScores：208 類
   - MUSCIMA++：180 類（部分重疊）
   - PrIMuS：完全不同的符號體系
   - 需要建立複雜的多對多映射

3. **質量一致性**
   - 印刷 vs 手寫
   - 高解析度 vs 低解析度
   - 完整樂譜 vs 符號片段

### 3.2 投入產出比分析

| 方案 | 開發時間 | 預期 mAP 提升 | 風險 | ROI |
|------|----------|---------------|------|-----|
| 整合 MUSCIMA++ | 3-5 天 | +5% | 手寫干擾印刷 | 低 |
| 整合 PrIMuS | 7-10 天 | +3% | 格式轉換錯誤 | 極低 |
| 整合 OpenOMR | 2-3 天 | +10% | 低 | **高** |
| 優化現有 DeepScores | 1-2 天 | +15% | 極低 | **極高** |

## 🚀 四、推薦執行計畫

### Phase 1：優化現有資源（立即執行）⭐⭐⭐⭐⭐

**目標**：充分利用 DeepScoresV2 的潛力

```python
# 1. 修復類別映射問題
def fix_stem_direction():
    """基於音符位置推斷 stem 方向"""
    # 實作後處理邏輯
    pass

# 2. 資料集重新平衡
def rebalance_dataset():
    """
    - 增加驗證集到 400 張
    - 確保每類別最少 50 個樣本
    - 對稀有類別過採樣
    """
    pass

# 3. 智慧數據增強
augmentation_config = {
    'rare_classes': {
        'oversample_ratio': 5.0,
        'augmentation_intensity': 'high'
    },
    'common_classes': {
        'oversample_ratio': 1.0,
        'augmentation_intensity': 'low'
    }
}
```

**預期成果**：
- 時間：2 天
- mAP 提升：15-20%
- 風險：極低

### Phase 2：選擇性整合 OpenOMR（可選）⭐⭐⭐

**前提**：Phase 1 完成後仍需改進

```bash
# 下載 OpenOMR
wget https://github.com/openomr/openomr/archive/master.zip

# 轉換腳本
python convert_openomr.py \
    --focus-on "stem_direction,slur,tie" \
    --output-format yolo \
    --quality-filter high
```

**預期成果**：
- 時間：3 天
- 額外 mAP 提升：5-10%
- 主要解決：stem_down 和 slur 問題

### Phase 3：實驗性整合（不推薦）❌

**MUSCIMA++ 和 PrIMuS 整合**：
- 投入產出比過低
- 建議作為研究項目而非生產需求
- 可能引入更多不穩定因素

## 📊 五、數據集使用率修正

### 實際使用率評估

| 數據集 | 可用資源 | 實際使用 | 使用率 | 說明 |
|--------|---------|---------|--------|------|
| DeepScoresV2 | 1,362 張 | 1,362 張 | **100%** | 完全使用 |
| MUSCIMA++ | 423 文件 | 0 | **0%** | 格式不兼容 |
| PrIMuS | 87,678 張 | 0 | **0%** | 格式差異大 |
| **總計** | ~89,000 張 | 1,362 張 | **1.5%** | 技術上的確只用了很少 |

**但是**，這個 1.5% 的數字是**誤導性的**，因為：
1. 不同數據集的質量和適用性差異極大
2. DeepScoresV2 是最適合的數據集
3. 盲目整合可能降低性能

## 🎯 六、執行建議

### 立即行動（24-48 小時）

1. **停止當前訓練**
   ```bash
   pkill -f yolo12_train
   ```

2. **執行 Phase 1 優化**
   ```bash
   python optimize_deepscores.py --fix-classes --rebalance --augment
   ```

3. **重新訓練**
   ```bash
   python yolo12_train_optimized.py \
       --data deepscores_optimized \
       --batch 16 \
       --epochs 300
   ```

### 預期結果

- **3 天內**：mAP 0.60+，穩定無震盪
- **無需**：複雜的多數據集整合
- **風險**：最小化
- **成功率**：85%+

## 🔍 七、結論

1. **你的確使用了 DeepScoresV2 的 100%**，但整體可用資源的利用率確實很低（~1.5%）

2. **但這不是問題**，因為：
   - 其他數據集整合成本高、收益低
   - 優化現有數據比整合新數據更有效

3. **真正的問題是**：
   - 類別映射缺陷（stem_down、slur）
   - 驗證集太小
   - 類別不平衡未處理

4. **最佳策略**：
   - 先優化現有 DeepScoresV2（Phase 1）
   - 必要時才考慮 OpenOMR（Phase 2）
   - 避免 MUSCIMA++/PrIMuS（投入產出比過低）

---

**最終建議**：專注於修復現有問題，而非盲目追求數據量。質量 > 數量。