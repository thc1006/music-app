# OMR 數據集深度分析報告

> 針對四部和聲 OMR 專案的數據集評估

## 一、專案需求摘要

### 瓶頸類別（mAP50=0，必須解決）
- `accidental_double_flat` - 重降記號
- `accidental_double_sharp` - 重升記號
- `barline_double` - 雙小節線

### 低性能類別（mAP50 < 0.3）
- `fermata` - 延長記號
- `accidental_natural` - 還原記號
- `barline_final` - 終止線
- `flag_16th`, `flag_32nd` - 旗標
- `clef_tenor` - 次中音譜號
- `dynamic_soft/loud` - 力度記號

---

## 二、數據集價值排序

### ⭐⭐⭐⭐⭐ 最高價值（強烈建議下載）

#### 1. DoReMi Dataset
- **來源**: Steinberg (Dorico 軟件)
- **規模**: 6,432 張圖片，近 100 萬標註
- **類別數**: 94 類
- **標註類型**: Bounding boxes, pixel masks, MusicXML, MEI, MIDI
- **對我們的價值**:
  - ✅ 有精確的 bounding boxes
  - ✅ 類別豐富，可能包含稀有記號
  - ✅ 多種格式可交叉驗證
- **下載**: https://github.com/steinbergmedia/DoReMi/
- **優先級**: 🔴 最高

#### 2. MUSCIMA++
- **來源**: Charles University
- **規模**: 91,255 個標註符號
- **特點**: 手寫樂譜，有符號間關係標註
- **對我們的價值**:
  - ✅ 大量手寫符號（增加泛化能力）
  - ✅ 有符號關係標註（可用於後處理）
  - ⚠️ 需要格式轉換
- **下載**: https://ufal.mff.cuni.cz/muscima
- **優先級**: 🔴 最高

#### 3. Choi Accidentals Dataset
- **來源**: IRISA
- **規模**: 2,955 張臨時記號圖片
- **類別**: flats, naturals, sharps, reject
- **對我們的價值**:
  - ✅ **專門針對臨時記號**（我們的瓶頸！）
  - ✅ 座標編碼在檔名中
  - ✅ 小型，易於整合
- **下載**: https://www-intuidoc.irisa.fr/en/choi_accidentals/
- **優先級**: 🔴 最高（針對性解決 natural/sharp 混淆）

### ⭐⭐⭐⭐ 高價值（建議下載）

#### 4. Fornes Dataset
- **來源**: CVC-UAB
- **規模**: 4,100 個符號
- **類別**: Accidentals 和 Clefs
- **對我們的價值**:
  - ✅ **專門是臨時記號和譜號**
  - ✅ 可改善 clef_tenor 問題
- **下載**: http://www.cvc.uab.es/~afornes/
- **優先級**: 🟠 高

#### 5. Universal Music Symbol Collection
- **來源**: Alexander Pacha
- **規模**: ~90,000 個符號
- **類別**: 79 類（74k 手寫，16k 印刷）
- **對我們的價值**:
  - ✅ 大規模符號分類器訓練數據
  - ✅ 整合了 7 個來源
  - ⚠️ 是分類數據，非檢測數據
- **下載**: https://github.com/apacha/MusicSymbolClassifier
- **優先級**: 🟠 高

#### 6. Rebelo Dataset
- **來源**: Academic
- **規模**: 15,000 個符號
- **特點**: 包含合成修改的圖片
- **對我們的價值**:
  - ✅ 合成變異增加魯棒性
  - ✅ 有五線譜移除相關數據
- **下載**: 已有連結
- **優先級**: 🟡 中高

### ⭐⭐⭐ 中等價值（可選下載）

#### 7. AudioLabs v2
- **規模**: 940 張，85,980 個 bounding boxes
- **類別**: 小節、五線譜、系統
- **對我們的價值**:
  - ✅ 大量小節線標註
  - ✅ 可改善 barline 相關類別
  - ✅ 已有 COCO 格式
- **下載**: 直接連結可用
- **優先級**: 🟡 中

#### 8. OpenOMR Dataset
- **規模**: 706 個符號
- **類別**: 16 類（譜號、音符頭、符幹、連音線）
- **對我們的價值**:
  - ✅ 小型但專注於核心符號
  - ⚠️ 規模較小
- **下載**: http://sourceforge.net/projects/openomr/
- **優先級**: 🟡 中

### ⭐⭐ 較低價值（暫不需要）

- **CVC-MUSCIMA**: 主要用於寫手識別
- **PrIMuS**: 序列識別，非物件檢測
- **HOMUS**: 線上手寫，非我們的場景
- **IMSLP**: 純 PDF，無標註

---

## 三、下載優先順序

### 立即下載（Phase 3 必需）
1. ✅ **DoReMi** - 完整 bounding boxes
2. ✅ **Choi Accidentals** - 解決臨時記號混淆
3. ✅ **Fornes Dataset** - 臨時記號+譜號

### 後續下載（Phase 4-5）
4. **MUSCIMA++** - 手寫泛化
5. **Universal Music Symbol** - 分類器增強
6. **AudioLabs v2** - 小節線改善

---

## 四、預期效果

| 數據集 | 解決的問題 | 預期 mAP 提升 |
|--------|-----------|--------------|
| DoReMi | 整體類別覆蓋 | +5-10% |
| Choi Accidentals | natural/sharp 混淆 | +3-5% (accidentals) |
| Fornes | clef_tenor 混淆 | +2-3% (clefs) |
| **總計** | - | **+10-18%** |

---

*分析日期: 2025-11-24*
*基於 [OMR-Datasets](https://github.com/apacha/OMR-Datasets) 調研*
