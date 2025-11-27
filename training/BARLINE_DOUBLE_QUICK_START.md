# barline_double 改善快速指南

**目標**: 將 barline_double mAP50 從 0.180 提升至 0.40+

---

## 🔥 立即執行（今天）

### 方案 1A: OpenScore Lieder 渲染（最高優先級）

**為什麼**: 可獲得 **4,017 個新 barline_double 標註**（200% 增長）

**步驟 1: 安裝 Verovio（15 分鐘）**

```bash
cd /home/thc1006/dev/music-app/training
pip install verovio cairosvg pillow
```

**步驟 2: 測試渲染（30 分鐘）**

```python
# test_openscore_render.py
import verovio
import cairosvg
from pathlib import Path

# 初始化 Verovio
tk = verovio.toolkit()
tk.setOptions({
    "pageHeight": 2970,
    "pageWidth": 2100,
    "scale": 100,
    "breaks": "none"
})

# 測試文件
test_file = "datasets/external/omr_downloads/OpenScoreLieder/scores/Viardot,_Pauline/L'enfant_et_la_mere/L'enfant_et_la_mere.mxl"

if Path(test_file).exists():
    # 載入並渲染
    tk.loadFile(test_file)
    svg = tk.renderToSVG(1)

    # 轉換為 PNG
    cairosvg.svg2png(
        bytestring=svg.encode('utf-8'),
        write_to="test_openscore.png",
        dpi=300
    )
    print("✅ Test render successful: test_openscore.png")
else:
    print("❌ OpenScore Lieder not found, please download first")
```

```bash
python test_openscore_render.py
```

**步驟 3: 如果測試成功，請告訴我繼續完整渲染腳本**

---

### 方案 1B: 標註修正腳本（次高優先級）

**為什麼**: 67.8% 的 barline_double 標註框過大，需要緊縮

**步驟: 執行修正腳本（30 分鐘）**

```bash
# 創建修正腳本目錄
mkdir -p scripts

# 使用我已經提供的 fix_barline_double_annotations.py
# （完整代碼在 BARLINE_DOUBLE_IMPROVEMENT_PLAN.md 中）
python scripts/fix_barline_double_annotations.py
```

**預期輸出**:
```
=== barline_double 標註修正報告 ===
修正數量: 1277
平均面積 (修正前): 0.0751
平均面積 (修正後): 0.0285
平均縮小比例: 62.1%

✅ 數據集已保存到: datasets/yolo_harmony_v2_phase6_double_fixed
```

---

## ⚡ 短期執行（本週）

### 方案 2: 激進加權訓練

**步驟 1: 創建激進配置（15 分鐘）**

```bash
# configs/phase7_barline_extreme.yaml
# （完整配置在 BARLINE_DOUBLE_IMPROVEMENT_PLAN.md 中）
# 關鍵參數:
# - barline_double class_weight: 16.0 (提升到 16x)
# - barline_double sampling_weight: 20.0 (提升到 20x)
# - box loss: 10.0 (強化 bbox 學習)
```

**步驟 2: 啟動 Phase 7 訓練（6-9 小時）**

```bash
# 使用修正後的數據集 + 激進配置
python yolo12_train.py \
  --data datasets/yolo_harmony_v2_phase6_double_fixed/data.yaml \
  --weights harmony_omr_v2_phase6/ultimate_barline_fixed/weights/best.pt \
  --epochs 150 \
  --batch 16 \
  --project harmony_omr_v2_phase7 \
  --name extreme_barline_double \
  --cfg configs/phase7_barline_extreme.yaml
```

---

## 📊 預期結果

### Week 1 結束（方案 1A + 1B + 2）

| 指標 | Phase 6 | Phase 7 預期 | 提升 |
|------|---------|-------------|------|
| **mAP50** | 0.180 | **0.35-0.45** | +94-150% |
| **召回率** | 19.2% | **40-50%** | +108-160% |
| **精確率** | 42.1% | **55-65%** | +31-54% |

### 數據集增長

| 來源 | barline_double 標註數 |
|------|---------------------|
| Phase 6 原始 | ~2,000 |
| + OpenScore Lieder | **+4,017** |
| + 標註修正優化 | 質量提升 |
| **總計** | **~6,017** (+200%) |

---

## 🛠️ 需要創建的腳本

我可以幫你創建以下腳本（告訴我需要哪些）:

1. ✅ `test_openscore_render.py` - 測試 Verovio 渲染（已提供）
2. ⏳ `render_openscore_barlines.py` - 批量渲染 OpenScore Lieder
3. ⏳ `convert_openscore_to_yolo.py` - 提取 bbox 並轉換為 YOLO 格式
4. ⏳ `merge_openscore_phase7.py` - 合併到 Phase 7 數據集
5. ✅ `fix_barline_double_annotations.py` - 標註修正腳本（已在 PLAN 中提供）
6. ⏳ `train_phase7_extreme.py` - Phase 7 激進訓練腳本

---

## 🎯 決策點

### 現在需要你決定

1. **是否立即開始 OpenScore 渲染？**
   - ✅ 推薦: 是（最高 ROI，1-2 天完成）
   - ❌ 如果你想先嘗試其他方案

2. **是否執行標註修正？**
   - ✅ 推薦: 是（4-6 小時，快速見效）
   - ❌ 如果你認為當前標註可接受

3. **GPU 可用時間**
   - Phase 7 訓練需要 6-9 小時連續 GPU 時間
   - 建議在週末或晚上執行

---

## 📞 下一步

**請告訴我**:

1. OpenScore 渲染測試結果（是否成功生成 test_openscore.png）
2. 是否需要我創建完整的渲染腳本
3. 是否需要我創建標註修正腳本
4. 何時可以開始 Phase 7 訓練

**我會根據你的回饋**:
- 創建所需的腳本
- 提供詳細的執行指令
- 監控訓練進度
- 分析結果並提出下一步建議

---

## 📄 完整文檔

詳細分析和所有代碼請見:
- `BARLINE_DOUBLE_IMPROVEMENT_PLAN.md` - 完整改善計劃（8,000+ 字）
- `BARLINE_COMPLETE_ANALYSIS.txt` - 根本原因分析
- `OPENSCORE_LIEDER_ANALYSIS.md` - 外部數據分析

---

**總結**: 最快路徑是先執行 OpenScore 渲染測試，確認可行後批量渲染，然後合併數據集進行 Phase 7 訓練。預期 1 週內可將 barline_double mAP50 提升至 0.35-0.45。
