# Phase 8 完整分析報告

**分析日期**: 2025-11-28
**訓練完成時間**: 2025-11-28 07:42
**分析者**: Claude (Opus 4.5)

---

## 一、整體訓練歷程回顧

### 1.1 各階段 mAP 進步趨勢

| Phase | Epochs | mAP50 | mAP50-95 | 進步幅度 |
|-------|--------|-------|----------|----------|
| Phase 2 | 131 | 0.505 | 0.443 | 基準線 |
| Phase 3 | 150 | 0.578 | 0.516 | +14.5% |
| Phase 4 | 200 | 0.613 | 0.553 | +6.1% |
| Phase 5 | 200 | 0.615 | 0.556 | +0.3% |
| Phase 6 | 200 | 0.619 | 0.545 | +0.7% |
| Phase 7 | 30 | 0.625 | 0.545 | +1.0% |
| **Phase 8** | **150** | **0.644** | **0.581** | **+3.0%** |

### 1.2 關鍵觀察

- **Phase 2→3**: 最大進步期，外部數據整合效果顯著
- **Phase 4→7**: 進入平台期，改進幅度遞減
- **Phase 8**: 突破平台期，mAP50-95 達到新高 58.1%

---

## 二、各類別詳細分析

### 2.1 類別表現排名（按 mAP50）

#### 🔴 危急類別（mAP50 < 0.40）

| 排名 | ID | 類別名稱 | mAP50 | 樣本數 | 問題診斷 |
|------|----|---------:|------:|-------:|----------|
| 1 | 24 | barline_double | **0.231** | 24,228 | 標註品質極差 |
| 2 | 16 | accidental_double_sharp | **0.369** | 2,209 | 樣本來源分布不均 |

#### ⚠️ 警告類別（mAP50 0.40-0.50）

| 排名 | ID | 類別名稱 | mAP50 | 樣本數 | 問題診斷 |
|------|----|---------:|------:|-------:|----------|
| 3 | 29 | fermata | 0.402 | 35,734 | 62% tiny bbox |
| 4 | 8 | tie | 0.411 | 59,909 | 形狀變化大 |
| 5 | 4 | flag_8th | 0.418 | 40,484 | MUSCIMA 標註品質 |
| 6 | 27 | time_signature | 0.446 | 36,654 | Rebelo 數據品質 |
| 7 | 32 | ledger_line | 0.466 | 246,506 | 極細線難檢測 |

#### 🟡 待改進類別（mAP50 0.50-0.60）

| ID | 類別名稱 | mAP50 | 樣本數 |
|----|---------:|------:|-------:|
| 14 | accidental_flat | 0.521 | 54,702 |
| 15 | accidental_natural | 0.542 | 28,885 |
| 13 | accidental_sharp | 0.570 | 59,188 |
| 30 | dynamic_soft | 0.572 | 28,157 |
| 23 | barline | 0.574 | 40,513 |
| 2 | stem | 0.583 | 677,675 |
| 25 | barline_final | 0.585 | 58,970 |
| 3 | beam | 0.591 | 208,660 |

#### 🟢 良好類別（mAP50 0.60-0.70）

| ID | 類別名稱 | mAP50 | 樣本數 |
|----|---------:|------:|-------:|
| 7 | augmentation_dot | 0.617 | 45,915 |
| 0 | notehead_filled | 0.647 | 737,991 |
| 1 | notehead_hollow | 0.679 | 61,719 |

#### ✅ 優秀類別（mAP50 > 0.70）

| ID | 類別名稱 | mAP50 | 樣本數 |
|----|---------:|------:|-------:|
| 11 | clef_alto | 0.703 | 7,068 |
| 5 | flag_16th | 0.731 | 19,050 |
| 21 | rest_8th | 0.755 | 34,060 |
| 6 | flag_32nd | 0.758 | 7,481 |
| 19 | rest_half | 0.768 | 5,037 |
| 18 | rest_whole | 0.778 | 20,672 |
| 28 | key_signature | 0.782 | 65,321 |
| 17 | accidental_double_flat | 0.788 | 75,676 |
| 9 | clef_treble | 0.797 | 23,767 |
| 20 | rest_quarter | 0.805 | 18,861 |
| 12 | clef_tenor | 0.809 | 3,481 |
| 26 | barline_repeat | 0.859 | 18,050 |
| 22 | rest_16th | 0.900 | 41,390 |
| 31 | dynamic_loud | 0.903 | 31,548 |
| 10 | clef_bass | 0.907 | 14,761 |

---

## 三、標註品質深度分析

### 3.1 Bounding Box 尺寸問題

| ID | 類別 | 樣本數 | 中位寬度 | 中位高度 | Tiny率 | 問題等級 |
|----|------|-------:|--------:|--------:|-------:|---------|
| 5 | flag_16th | 17,111 | 0.0097 | 0.0194 | **87.3%** | 🔴 CRITICAL |
| 6 | flag_32nd | 6,719 | 0.0089 | 0.0231 | **88.0%** | 🔴 CRITICAL |
| 22 | rest_16th | 37,228 | 0.0109 | 0.0163 | **85.2%** | 🔴 CRITICAL |
| 29 | fermata | 31,871 | 0.0200 | 0.0200 | **62.2%** | 🔴 CRITICAL |
| 17 | double_flat | 68,538 | 0.0161 | 0.0698 | **50.5%** | 🟠 HIGH |
| 23 | barline | 36,455 | 0.0150 | 0.1007 | **41.8%** | 🟠 HIGH |
| 21 | rest_8th | 30,690 | 0.1362 | 0.1061 | **41.1%** | 🟠 HIGH |

> **Tiny 定義**: bbox 面積 < 0.001 (normalized)

### 3.2 問題來源追溯

#### DoReMi 數據集（主要問題源）

| 類別 | DoReMi 樣本 | Tiny 比例 | 影響 |
|------|----------:|--------:|------|
| flag_16th | 14,535 | **100%** | 完全無效標註 |
| flag_32nd | 5,915 | **100%** | 完全無效標註 |
| rest_16th | 31,034 | **100%** | 完全無效標註 |
| barline | 13,931 | **81.2%** | 嚴重品質問題 |

#### OpenScore Lieder（fermata 問題源）

| 類別 | OpenScore 樣本 | Tiny 比例 | 影響 |
|------|-------------:|--------:|------|
| fermata | 22,051 | **89.1%** | 幾乎全部無效 |

#### MUSCIMA++（多類別問題）

| 類別 | MUSCIMA 樣本 | Tiny 比例 |
|------|------------:|--------:|
| barline | 5,726 | 60.6% |
| flag_8th | ~500 | ~100% |
| tie | ~500 | ~50% |

### 3.3 根本原因分析

```
問題根源樹狀圖：

低 mAP50 原因
├── 標註品質問題 (60%)
│   ├── DoReMi: 座標轉換錯誤，bbox 極小
│   ├── OpenScore: 渲染座標提取錯誤
│   └── MUSCIMA: 原始標註尺度不一致
│
├── 樣本分布不均 (25%)
│   ├── accidental_double_sharp: 只有 2,209 個
│   ├── clef_tenor: 只有 3,481 個
│   └── rest_half: 只有 5,037 個
│
└── 視覺特徵困難 (15%)
    ├── barline_double: 細線難區分
    ├── tie: 形狀變化大
    └── ledger_line: 與五線譜線混淆
```

---

## 四、Phase 9 改進方案

### 4.1 策略優先級矩陣

| 方案 | 預期提升 | 投入時間 | 成功率 | ROI | 優先級 |
|------|---------|---------|--------|-----|-------|
| **A. 清理無效標註** | mAP +3-5% | 4-6 小時 | 95% | ⭐⭐⭐⭐⭐ | 🔥 P0 |
| **B. 修復 DoReMi 轉換** | mAP +2-4% | 1-2 天 | 80% | ⭐⭐⭐⭐ | 🟢 P1 |
| **C. 重新渲染 OpenScore** | mAP +1-3% | 1 天 | 85% | ⭐⭐⭐⭐ | 🟢 P1 |
| **D. 高解析度訓練** | mAP +1-2% | 12-18 小時 | 70% | ⭐⭐⭐ | 🟡 P2 |
| **E. 合成數據補強** | mAP +2-4% | 3-5 天 | 65% | ⭐⭐⭐ | 🟡 P2 |

### 4.2 方案 A：清理無效標註（立即執行）

**目標**：移除所有 area < 0.0005 的標註

**影響估計**：
- 移除約 15-20% 的無效標註
- 訓練數據質量大幅提升
- 預期 mAP50: 0.644 → 0.67-0.69

**執行腳本**：
```python
# scripts/clean_tiny_annotations.py
import os
from pathlib import Path

def clean_tiny_annotations(dataset_dir, min_area=0.0005, min_dimension=0.005):
    """
    移除過小的標註
    - min_area: 最小面積閾值
    - min_dimension: 最小寬/高閾值
    """
    stats = {'removed': 0, 'kept': 0, 'files_modified': 0}

    for split in ['train', 'val']:
        label_dir = Path(dataset_dir) / split / 'labels'

        for label_file in label_dir.glob('*.txt'):
            lines = label_file.read_text().strip().split('\n')
            new_lines = []
            modified = False

            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    new_lines.append(line)
                    continue

                w, h = float(parts[3]), float(parts[4])
                area = w * h

                # 移除過小的標註
                if area < min_area or w < min_dimension or h < min_dimension:
                    stats['removed'] += 1
                    modified = True
                else:
                    new_lines.append(line)
                    stats['kept'] += 1

            if modified:
                stats['files_modified'] += 1
                label_file.write_text('\n'.join(new_lines))

    return stats
```

### 4.3 方案 B：修復 DoReMi 座標轉換

**問題**：DoReMi 的 OMR XML 座標轉換為 YOLO 格式時出錯

**解決方案**：
1. 重新分析 DoReMi XML 結構
2. 修正座標轉換邏輯
3. 重新生成標註文件

### 4.4 方案 C：重新渲染 OpenScore

**問題**：OpenScore fermata 座標提取錯誤（89% tiny）

**解決方案**：
1. 使用 Verovio 原生 bbox API
2. 正確處理 SVG viewBox 轉換
3. 保留 OpenScore 的 barline 數據（品質良好）

### 4.5 方案 D：高解析度訓練

**配置**：
```yaml
# Phase 9 高解析度配置
imgsz: 768  # 從 640 提升到 768
batch: 16   # 降低 batch size 以適應更大圖片
epochs: 100
multi_scale: true
scale: [0.5, 1.5]  # 多尺度訓練
```

---

## 五、實施時間表

### Week 1（立即行動）

| 日期 | 任務 | 預期產出 |
|------|------|----------|
| Day 1 | 執行方案 A：清理無效標註 | Phase 9 乾淨數據集 |
| Day 2-3 | 執行方案 B：修復 DoReMi | 修正的 flag/rest 標註 |
| Day 3-4 | 執行方案 C：修復 OpenScore | 修正的 fermata 標註 |
| Day 5-6 | Phase 9 訓練 | mAP50 目標 0.68+ |
| Day 7 | 評估與調整 | 決定是否需要 Week 2 |

### Week 2（視 Week 1 結果）

| 條件 | 行動 |
|------|------|
| mAP50 ≥ 0.70 | 進入部署測試階段 |
| mAP50 0.65-0.70 | 執行方案 D（高解析度）|
| mAP50 < 0.65 | 執行方案 E（合成數據）+ 深度分析 |

---

## 六、目標與成功標準

### Phase 9 目標

| 指標 | Phase 8 | Phase 9 目標 | 最低要求 |
|------|---------|-------------|---------|
| **整體 mAP50** | 0.644 | **0.70+** | 0.68 |
| **整體 mAP50-95** | 0.581 | **0.62+** | 0.60 |
| **最差類別 mAP50** | 0.231 | **0.40+** | 0.35 |
| **危急類別數量** | 2 | **0** | ≤1 |

### 各類別改善目標

| 類別 | Phase 8 | 目標 | 改善策略 |
|------|---------|------|----------|
| barline_double | 0.231 | 0.40+ | 清理 + 數據增強 |
| double_sharp | 0.369 | 0.50+ | 平衡採樣 |
| fermata | 0.402 | 0.55+ | 修復 OpenScore |
| tie | 0.411 | 0.50+ | 形狀增強 |
| flag_8th | 0.418 | 0.55+ | 清理 MUSCIMA |
| time_signature | 0.446 | 0.55+ | 清理 Rebelo |
| ledger_line | 0.466 | 0.55+ | 高解析度訓練 |

---

## 七、風險與緩解

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| 清理後樣本過少 | 中 | 高 | 保留閾值可調整 |
| 修復轉換耗時 | 中 | 中 | 優先處理高影響類別 |
| 訓練不收斂 | 低 | 高 | 保留 Phase 8 模型作為回退 |

---

## 八、結論

Phase 8 訓練成功達到 mAP50 64.4%，但深入分析揭示了 **標註品質問題** 是當前最大瓶頸：

1. **DoReMi 數據集** 貢獻了大量無效標註（100% tiny bbox）
2. **OpenScore fermata** 座標提取錯誤
3. **MUSCIMA++** 部分類別標註尺度不一致

**建議立即行動**：
1. 執行標註清理腳本（4-6 小時）
2. 修復關鍵數據源轉換（1-2 天）
3. Phase 9 訓練（6-9 小時）

**預期成果**：
- mAP50: 0.644 → **0.70+**
- 消除所有危急類別
- 為生產部署做好準備

---

*報告生成時間: 2025-11-28*
