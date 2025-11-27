# barline_double 改善方案

**分析日期**: 2025-11-26
**當前狀態**: Phase 6 訓練完成
**問題嚴重性**: 🔴 CRITICAL - 最差表現類別

---

## 📊 當前狀況分析

### Phase 6 訓練結果

| 指標 | Phase 5 | Phase 6 | 提升 | 目標 | 差距 |
|------|---------|---------|------|------|------|
| **mAP50** | 0.140 | **0.180** | +28.6% | 0.50+ | -63.9% |
| **召回率** | 13.3% | **19.2%** | +5.9pp | 50%+ | -30.8pp |
| **精確率** | 34.5% | **42.1%** | +7.6pp | 70%+ | -27.9pp |
| **驗證實例** | 173 | 224 | +51 | - | - |

**關鍵發現**:
- ✅ Phase 6 有進步，但**遠未達標**（mAP50 僅 0.180）
- ❌ **召回率仍只有 19.2%**（漏檢 80.8%）
- ❌ 精確率 42.1%（誤檢率 57.9%）
- ⚠️ 在所有類別中表現**倒數第一**

### 根本問題分析（來自 BARLINE_COMPLETE_ANALYSIS）

| 問題 | 統計數據 | 影響 |
|------|---------|------|
| **1. 樣本稀少** | 僅 1,883 個標註（最少的 barline 類別） | 🔴 CRITICAL |
| **2. 標註框過大** | 67.8% 面積 > 0.1（異常膨脹） | 🔴 CRITICAL |
| **3. 視覺特徵模糊** | 框太大導致特徵不清晰（2 根線 + 空隙） | 🔴 CRITICAL |
| **4. 數據不平衡** | barline_final 有 58,819 個（31x more） | 🟡 WARNING |

---

## 🎯 改善策略優先級排序

### 投資報酬比分析

| 方案 | 投入時間 | 預期提升 | 成功率 | ROI | 優先級 |
|------|---------|---------|--------|-----|--------|
| **方案 1A: OpenScore Lieder 渲染** | 1-2 天 | mAP +0.10-0.15 | 90% | ⭐⭐⭐⭐⭐ | 🔥 **立即執行** |
| **方案 1B: 標註修正腳本** | 4-6 小時 | mAP +0.05-0.08 | 85% | ⭐⭐⭐⭐⭐ | 🔥 **立即執行** |
| **方案 2: 激進加權訓練** | 8-12 小時 | mAP +0.08-0.12 | 75% | ⭐⭐⭐⭐ | 🟢 **短期** |
| **方案 3: 合成數據生成** | 1-2 週 | mAP +0.15-0.20 | 70% | ⭐⭐⭐⭐ | 🟡 **中期** |
| **方案 4: 類別合併** | 2-3 天 | mAP +0.20+ | 60% | ⭐⭐⭐ | 🔵 **備選** |

---

## 🚀 方案 1A: OpenScore Lieder 渲染（立即執行）

### 為什麼這是最佳方案？

根據 `OPENSCORE_LIEDER_ANALYSIS.md`:

| 指標 | OpenScore Lieder | MUSCIMA++ | 倍數 |
|------|------------------|-----------|------|
| **barline_double 來源** | **4,017 個** | 42 | **95.6x** |
| **文件數** | 1,410 | 140 | 10.1x |
| **授權** | CC-0（商用可） | CC-BY-NC-SA | ✅ |

**具體分佈**:
- `light-heavy`: 3,554 個 → barline_final
- `heavy-light`: 463 個 → barline_double

### 實施步驟（1-2 天）

#### Day 1: 環境設置與測試渲染（4-6 小時）

```bash
# Step 1: 安裝 Verovio（推薦）
cd /home/thc1006/dev/music-app/training
pip install verovio

# Step 2: 測試渲染單個文件
python << 'EOF'
import verovio
from pathlib import Path

# 載入 Verovio toolkit
tk = verovio.toolkit()
tk.setOptions({
    "pageHeight": 2970,
    "pageWidth": 2100,
    "scale": 100,
    "adjustPageHeight": True
})

# 測試渲染
test_file = "datasets/external/omr_downloads/OpenScoreLieder/scores/Viardot,_Pauline/L'enfant_et_la_mere/L'enfant_et_la_mere.mxl"
tk.loadFile(test_file)

# 渲染為 SVG
svg = tk.renderToSVG(1)
with open("test_openscore_render.svg", "w") as f:
    f.write(svg)

print("✅ Test render successful")
EOF

# Step 3: 轉換 SVG → PNG
pip install cairosvg
python << 'EOF'
import cairosvg

cairosvg.svg2png(
    url="test_openscore_render.svg",
    write_to="test_openscore_render.png",
    dpi=300
)
print("✅ PNG conversion successful")
EOF
```

#### Day 1-2: 批量渲染腳本（8-12 小時）

創建 `scripts/render_openscore_barlines.py`:

```python
"""
OpenScore Lieder Barline 渲染腳本
優先渲染含 double/final barlines 的文件
"""
import verovio
import cairosvg
from pathlib import Path
from xml.etree import ElementTree as ET
import json
from tqdm import tqdm
import multiprocessing as mp

class OpenScoreBarlineRenderer:
    def __init__(self, openscore_dir, output_dir):
        self.openscore_dir = Path(openscore_dir)
        self.output_dir = Path(output_dir)
        self.tk = verovio.toolkit()
        self.tk.setOptions({
            "pageHeight": 2970,
            "pageWidth": 2100,
            "scale": 100,
            "breaks": "none",  # 連續渲染不分頁
        })

    def parse_musicxml_barlines(self, mxl_path):
        """解析 MusicXML 中的 barline 類型"""
        tree = ET.parse(mxl_path)
        root = tree.getroot()

        barlines = []
        for barline in root.findall('.//barline'):
            bar_style = barline.find('bar-style')
            if bar_style is not None:
                style = bar_style.text
                location = barline.get('location', 'right')
                barlines.append({
                    'style': style,
                    'location': location
                })

        return barlines

    def has_target_barlines(self, mxl_path):
        """檢查文件是否含有 double/final barlines"""
        barlines = self.parse_musicxml_barlines(mxl_path)
        target_styles = ['light-heavy', 'heavy-light', 'heavy-heavy']
        return any(b['style'] in target_styles for b in barlines)

    def render_file(self, mxl_path, output_name):
        """渲染單個文件到 PNG"""
        # 1. 載入 MusicXML
        self.tk.loadFile(str(mxl_path))

        # 2. 取得頁數
        page_count = self.tk.getPageCount()

        images = []
        for page in range(1, page_count + 1):
            # 3. 渲染 SVG
            svg = self.tk.renderToSVG(page)

            # 4. 轉換為 PNG
            png_path = self.output_dir / "images" / f"{output_name}_page{page}.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)

            cairosvg.svg2png(
                bytestring=svg.encode('utf-8'),
                write_to=str(png_path),
                dpi=300
            )

            images.append(png_path)

        return images

    def extract_barline_bboxes(self, mxl_path, svg_path):
        """
        從 SVG 中提取 barline bounding boxes
        使用 Verovio 的元素 ID 匹配
        """
        # 解析 SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Verovio 會生成帶有 ID 的 SVG 元素
        # 格式: measure-1-barline-1
        bboxes = []

        for elem in root.iter():
            elem_id = elem.get('id', '')
            if 'barline' in elem_id:
                # 取得 bounding box
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                width = float(elem.get('width', 0))
                height = float(elem.get('height', 0))

                # 轉換為 YOLO 格式 (normalized)
                # 需要知道 SVG 的總寬高
                svg_width = float(root.get('width', 2100))
                svg_height = float(root.get('height', 2970))

                x_center = (x + width / 2) / svg_width
                y_center = (y + height / 2) / svg_height
                norm_width = width / svg_width
                norm_height = height / svg_height

                # 判斷 barline 類型（需要對應 MusicXML）
                class_id = self.infer_barline_class(elem)

                bboxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': norm_width,
                    'height': norm_height
                })

        return bboxes

    def infer_barline_class(self, svg_elem):
        """
        根據 SVG 元素推斷 barline 類別
        Verovio 可能在 class 屬性中包含樣式資訊
        """
        classes = svg_elem.get('class', '').lower()

        # Verovio 的 class 命名規則
        if 'double' in classes or 'heavy-light' in classes:
            return 24  # barline_double
        elif 'final' in classes or 'light-heavy' in classes:
            return 25  # barline_final
        elif 'repeat' in classes:
            return 26  # barline_repeat
        else:
            return 23  # barline (regular)

    def batch_render(self, max_files=None, workers=4):
        """批量渲染所有符合條件的文件"""
        # 1. 找到所有 MusicXML 文件
        mxl_files = list(self.openscore_dir.rglob("*.mxl")) + \
                    list(self.openscore_dir.rglob("*.xml"))

        # 2. 過濾：只保留含 double/final barlines 的文件
        target_files = []
        for mxl in tqdm(mxl_files, desc="Filtering files"):
            if self.has_target_barlines(mxl):
                target_files.append(mxl)

        print(f"✅ Found {len(target_files)} files with double/final barlines")

        if max_files:
            target_files = target_files[:max_files]

        # 3. 批量渲染
        results = []
        for mxl in tqdm(target_files, desc="Rendering"):
            try:
                output_name = mxl.stem
                images = self.render_file(mxl, output_name)
                results.append({
                    'source': str(mxl),
                    'images': [str(img) for img in images],
                    'success': True
                })
            except Exception as e:
                results.append({
                    'source': str(mxl),
                    'error': str(e),
                    'success': False
                })

        # 4. 生成報告
        success_count = sum(1 for r in results if r['success'])
        print(f"\n✅ Rendered {success_count}/{len(target_files)} files successfully")

        # 保存報告
        with open(self.output_dir / "render_report.json", "w") as f:
            json.dump(results, f, indent=2)

        return results

# 主執行
if __name__ == "__main__":
    renderer = OpenScoreBarlineRenderer(
        openscore_dir="/home/thc1006/dev/music-app/training/datasets/external/omr_downloads/OpenScoreLieder",
        output_dir="/home/thc1006/dev/music-app/training/datasets/openscore_barlines_rendered"
    )

    # 先測試 10 個文件
    results = renderer.batch_render(max_files=10, workers=1)

    # 如果成功，渲染全部
    # results = renderer.batch_render(max_files=None, workers=4)
```

#### Day 2: 標註提取與數據集整合（4-6 小時）

```bash
# Step 1: 執行渲染
python scripts/render_openscore_barlines.py

# Step 2: 轉換為 YOLO 格式
python scripts/convert_openscore_to_yolo.py

# Step 3: 合併到 Phase 6 數據集
python scripts/merge_openscore_phase7.py
```

### 預期效果

| 指標 | Phase 6 | Phase 7 (with OpenScore) | 提升 |
|------|---------|--------------------------|------|
| **barline_double 標註數** | ~2,000 | **6,017** | +200% |
| **barline_final 標註數** | ~5,054 | **8,608** | +70% |
| **預期 mAP50** | 0.180 | **0.35-0.45** | +94-150% |
| **預期召回率** | 19.2% | **40-50%** | +108-160% |

---

## 🔧 方案 1B: 標註修正腳本（立即執行）

### 問題定位

根據 `BARLINE_COMPLETE_ANALYSIS.txt`:
- **67.8% 的 barline_double 標註面積 > 0.1**（過度膨脹）
- **平均面積: 0.0751**（應該 ≤ 0.03）

### 修正策略

創建 `scripts/fix_barline_double_annotations.py`:

```python
"""
barline_double 標註緊縮腳本
將過大的標註框智能緊縮到合理大小
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def shrink_barline_double_bbox(bbox, target_area=0.03, min_width=0.01):
    """
    智能緊縮 barline_double 標註框

    策略:
    1. double barline = 2 根細線 + 中間空隙
    2. 寬度應該約為 0.02-0.03 (normalized)
    3. 高度保持不變（跨越五線譜）
    """
    x_center, y_center, width, height = bbox

    current_area = width * height

    if current_area > 0.1:  # 過大
        # 計算合理寬度（基於 target_area）
        new_width = target_area / height
        new_width = max(new_width, min_width)  # 確保不會太細

        # 保持中心點不變
        return [x_center, y_center, new_width, height]

    return bbox  # 不需修改

def process_label_file(label_path, output_path, stats):
    """處理單個標註文件"""
    with open(label_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            modified_lines.append(line)
            continue

        class_id = int(parts[0])
        bbox = [float(x) for x in parts[1:]]

        if class_id == 24:  # barline_double
            original_area = bbox[2] * bbox[3]
            new_bbox = shrink_barline_double_bbox(bbox)
            new_area = new_bbox[2] * new_bbox[3]

            if new_area != original_area:
                stats['modified'] += 1
                stats['area_before'].append(original_area)
                stats['area_after'].append(new_area)

            modified_lines.append(f"{class_id} {' '.join(map(str, new_bbox))}\n")
        else:
            modified_lines.append(line)

    # 寫入修正後的標註
    with open(output_path, 'w') as f:
        f.writelines(modified_lines)

def main():
    # 輸入/輸出目錄
    input_dataset = Path("datasets/yolo_harmony_v2_phase6_ultimate")
    output_dataset = Path("datasets/yolo_harmony_v2_phase6_double_fixed")

    # 創建輸出目錄結構
    for split in ['train', 'val']:
        (output_dataset / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dataset / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {
        'modified': 0,
        'area_before': [],
        'area_after': []
    }

    # 處理訓練集和驗證集
    for split in ['train', 'val']:
        label_dir = input_dataset / split / "labels"
        image_dir = input_dataset / split / "images"

        output_label_dir = output_dataset / split / "labels"
        output_image_dir = output_dataset / split / "images"

        # 複製圖片（符號連結以節省空間）
        for img in image_dir.glob("*.png"):
            dst = output_image_dir / img.name
            if not dst.exists():
                dst.symlink_to(img.absolute())

        # 處理標註
        for label_path in tqdm(list(label_dir.glob("*.txt")), desc=f"Processing {split}"):
            output_path = output_label_dir / label_path.name
            process_label_file(label_path, output_path, stats)

    # 生成報告
    print(f"\n=== barline_double 標註修正報告 ===")
    print(f"修正數量: {stats['modified']}")
    if stats['area_before']:
        print(f"平均面積 (修正前): {np.mean(stats['area_before']):.4f}")
        print(f"平均面積 (修正後): {np.mean(stats['area_after']):.4f}")
        print(f"平均縮小比例: {(1 - np.mean(stats['area_after']) / np.mean(stats['area_before'])) * 100:.1f}%")

    # 更新 data.yaml
    data_yaml_content = f"""# Phase 6 barline_double 修正數據集
path: {output_dataset.absolute()}
train: train/images
val: val/images

nc: 33
names: ['notehead_filled', 'notehead_hollow', 'stem', 'beam', 'flag_8th', 'flag_16th', 'flag_32nd',
        'augmentation_dot', 'accidental_sharp', 'accidental_flat', 'accidental_double_sharp',
        'accidental_double_flat', 'accidental_natural', 'clef_treble', 'clef_bass', 'clef_alto',
        'clef_tenor', 'dynamic_f', 'dynamic_p', 'dynamic_mf', 'dynamic_mp', 'rest_whole', 'rest_half',
        'rest_quarter', 'rest_8th', 'rest_16th', 'barline', 'barline_double', 'barline_final',
        'barline_repeat', 'time_signature', 'key_signature', 'fermata']
"""

    with open(output_dataset / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    print(f"\n✅ 數據集已保存到: {output_dataset}")
    print(f"✅ 使用以下配置訓練: {output_dataset / 'data.yaml'}")

if __name__ == "__main__":
    main()
```

### 執行

```bash
cd /home/thc1006/dev/music-app/training
python scripts/fix_barline_double_annotations.py
```

### 預期效果

| 指標 | 修正前 | 修正後 | 改善 |
|------|--------|--------|------|
| **平均標註面積** | 0.0751 | **0.025-0.035** | -60-70% |
| **過大標註比例** | 67.8% | **<10%** | -85% |
| **預期 mAP50 提升** | - | **+0.05-0.08** | +28-44% |

---

## ⚡ 方案 2: 激進加權訓練（短期方案）

### 當前權重配置（Phase 6）

```yaml
class_weights:
  23: 4.0   # barline
  24: 8.0   # barline_double
  25: 2.0   # barline_final
  26: 1.0   # barline_repeat

sampling_weights:
  23: 5.0
  24: 8.0
  25: 2.0
  26: 1.5
```

### 激進方案（Phase 7）

創建 `configs/phase7_barline_extreme.yaml`:

```yaml
# Phase 7: 極端 barline_double 優化配置

stage1:
  epochs: 150
  patience: 50
  batch: 16
  imgsz: 640
  lr0: 0.001

  # 極端類別權重
  class_weights:
    23: 6.0    # barline (提升到 6x)
    24: 16.0   # barline_double (提升到 16x) ⚡
    25: 2.0    # barline_final
    26: 1.0    # barline_repeat

  # 極端採樣權重
  sampling_weights:
    23: 8.0
    24: 20.0   # barline_double (提升到 20x) ⚡
    25: 2.0
    26: 1.5

  # 激進損失權重
  box: 10.0    # 提升 bbox 損失權重
  cls: 4.0     # 提升分類損失權重
  dfl: 2.0     # 提升分佈焦點損失

  # 更激進的增強
  mosaic: 1.0
  mixup: 0.2   # 提升 mixup
  copy_paste: 0.3  # 提升 copy_paste（專門複製 barline_double）

  # 小物體優化
  scale: 0.5
  translate: 0.2

# Hard Example Mining 配置
hem:
  confidence_threshold: 0.5
  iou_threshold: 0.5
  target_classes: [23, 24, 25, 26]  # 所有 barline 類別

  # 難度評分
  false_negative_score: 3.0  # 提升 FN 權重
  low_confidence_score: 2.0
  misclassification_score: 2.0

# Stage 2: 難例微調
stage2:
  epochs: 50
  lr0: 0.0005

  class_weights:
    24: 20.0   # barline_double 極端權重 ⚡

  box: 15.0
  cls: 5.0
  dfl: 2.5
```

### 訓練腳本

```bash
# Phase 7 訓練（使用激進配置）
python custom_training/train_phase7.py \
  --config configs/phase7_barline_extreme.yaml \
  --data datasets/yolo_harmony_v2_phase6_double_fixed/data.yaml \
  --weights harmony_omr_v2_phase6/ultimate_barline_fixed/weights/best.pt
```

### 預期效果

| 指標 | Phase 6 | Phase 7 預期 | 提升 |
|------|---------|-------------|------|
| **mAP50** | 0.180 | **0.28-0.35** | +55-94% |
| **召回率** | 19.2% | **35-45%** | +82-134% |
| **精確率** | 42.1% | **55-65%** | +31-54% |

---

## 🏗️ 方案 3: 合成數據生成（中期方案）

### Verovio 合成系統（已實施 90%）

根據 `synthetic_generation/README.md`:

**當前狀態**:
- ✅ MEI 生成器完成
- ✅ Verovio 渲染器完成
- ⚠️ **Bbox 提取需要修正**（座標超出 [0,1] 範圍）
- ⚠️ 成功率低（10 張中只有 1 張成功）

### 修正計劃（2-3 天）

#### Day 1: 修正 Bbox 提取（6-8 小時）

```python
# synthetic_generation/src/bbox_extractor_v2.py
"""
使用 Verovio 原生 API 提取精確 bbox
"""
import verovio

class VerovioNativeBboxExtractor:
    def __init__(self, tk: verovio.toolkit):
        self.tk = tk

    def extract_barline_bboxes(self, page=1):
        """
        使用 Verovio 的 getElementsAtTime() API
        提取精確的 barline 位置
        """
        # Verovio 可以返回 SVG 中的元素座標
        svg_string = self.tk.renderToSVG(page)

        # 解析 SVG 獲取實際像素座標
        import xml.etree.ElementTree as ET
        root = ET.fromstring(svg_string)

        # SVG viewBox 定義了座標系統
        viewBox = root.get('viewBox')
        vb_parts = viewBox.split()
        svg_width = float(vb_parts[2])
        svg_height = float(vb_parts[3])

        bboxes = []

        # 找到所有 barline 元素
        # Verovio 使用特定的 class 名稱
        for elem in root.iter():
            elem_class = elem.get('class', '')

            if 'barLine' in elem_class:  # Verovio 命名規則
                # 取得 transform 或直接座標
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                width = float(elem.get('width', 5))  # barline 寬度通常很小
                height = float(elem.get('height', 100))  # 跨越五線譜

                # 正確的 YOLO 格式轉換
                x_center = (x + width / 2) / svg_width
                y_center = (y + height / 2) / svg_height
                norm_width = width / svg_width
                norm_height = height / svg_height

                # 確保在 [0, 1] 範圍內
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_width = max(0.001, min(1, norm_width))
                norm_height = max(0.001, min(1, norm_height))

                # 推斷類別
                class_id = self.infer_barline_type(elem)

                bboxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': norm_width,
                    'height': norm_height
                })

        return bboxes

    def infer_barline_type(self, svg_elem):
        """
        根據 SVG 元素的 class 屬性推斷類型
        Verovio 會在 class 中包含樣式資訊
        """
        classes = svg_elem.get('class', '').lower()

        # Verovio 的 barline 樣式
        if 'double' in classes or 'barlinedbl' in classes:
            return 24  # barline_double
        elif 'end' in classes or 'barlineend' in classes:
            return 25  # barline_final
        elif 'rptboth' in classes or 'repeat' in classes:
            return 26  # barline_repeat
        else:
            return 23  # regular barline

# 測試腳本
if __name__ == "__main__":
    tk = verovio.toolkit()
    extractor = VerovioNativeBboxExtractor(tk)

    # 測試文件
    tk.loadFile("test.mei")
    bboxes = extractor.extract_barline_bboxes(page=1)

    print(f"✅ Extracted {len(bboxes)} barlines")
    for bbox in bboxes:
        print(f"  Class {bbox['class_id']}: x={bbox['x_center']:.3f}, w={bbox['width']:.4f}")
```

#### Day 2-3: 批量生成（12-16 小時）

```bash
# 修正後重新生成
python synthetic_generation/generate_synthetic_barlines.py \
  --num-images 10000 \
  --output-dir datasets/synthetic_barlines_v2 \
  --workers 8 \
  --barline-double-ratio 0.3  # 30% 為 double barlines
```

### 預期產出

| 指標 | 數量 |
|------|------|
| **總圖片** | 10,000 |
| **barline_double 實例** | ~30,000 (3 per image) |
| **其他 barlines** | ~70,000 |
| **預期 mAP50 提升** | +0.15-0.20 |

---

## 🔀 方案 4: 類別合併（備選方案）

### 合併策略

將 `barline_double` 和 `barline_final` 合併為一個類別:
- **新類別**: `barline_thick` (ID 24)
- **理由**: 視覺上都是「粗/雙線」barline，分類困難

### 優點
- ✅ 樣本數增加: 1,883 + 58,819 = **60,702** (32x increase)
- ✅ 降低類別混淆
- ✅ 快速見效（2-3 天）

### 缺點
- ❌ 失去細粒度分類
- ❌ 需要重新標註數據集
- ❌ 與原始目標不符（四部和聲需要區分終止線）

### 實施難度
- 中等（需要修改數據集標籤）
- 估計時間: 2-3 天

---

## 📊 綜合改善時間表

### Week 1（立即執行）

**Day 1-2: 數據增強與修正**
- [ ] 執行方案 1A: OpenScore Lieder 渲染（優先 double barlines）
- [ ] 執行方案 1B: 標註修正腳本
- [ ] 合併數據集，創建 Phase 7 數據集

**Day 3-4: Phase 7 訓練**
- [ ] 執行方案 2: 激進加權訓練
- [ ] 使用修正後的數據集 + OpenScore 新數據
- [ ] 預期訓練時間: 6-9 小時 (RTX 5090)

**Day 5: 評估與調整**
- [ ] 評估 Phase 7 結果
- [ ] 分析 per-class 性能
- [ ] 決定是否需要方案 3 或 4

### Week 2-3（如果 Week 1 未達標）

**方案 3: 合成數據生成**
- [ ] 修正 Verovio bbox 提取
- [ ] 生成 10,000 張合成圖片
- [ ] 合併到 Phase 8 數據集
- [ ] 訓練 Phase 8

---

## 🎯 成功標準

### Phase 7 目標（Week 1 結束）

| 指標 | 當前 | 目標 | 最低要求 |
|------|------|------|---------|
| **mAP50** | 0.180 | **0.40+** | 0.35 |
| **召回率** | 19.2% | **50%+** | 40% |
| **精確率** | 42.1% | **65%+** | 55% |

### Phase 8 目標（如需 Week 2-3）

| 指標 | Phase 7 | 目標 | 最低要求 |
|------|---------|------|---------|
| **mAP50** | 0.35-0.40 | **0.50+** | 0.45 |
| **召回率** | 40-50% | **60%+** | 55% |
| **精確率** | 55-65% | **75%+** | 70% |

---

## 💡 關鍵建議

### 立即行動（今天/明天）

1. **優先執行方案 1A**: OpenScore Lieder 是最快、最確定的數據來源
   - 463 個 heavy-light barlines 可以直接解決樣本稀少問題
   - 1-2 天可完成

2. **並行執行方案 1B**: 標註修正腳本執行快速（4-6 小時）
   - 可以立即改善現有數據質量
   - 無需額外數據收集

3. **準備方案 2**: 修改訓練配置文件
   - 可以在等待數據生成時並行準備
   - 配置文件修改只需 1 小時

### 中期計劃（1-2 週）

- **如果 Phase 7 達到 mAP50 0.35+**: 可以考慮直接進入生產部署測試
- **如果 Phase 7 未達標**: 執行方案 3（合成數據生成）
- **方案 4（類別合併）**: 僅作為最後手段

### 監控指標

訓練過程中重點監控:
- ✅ barline_double 召回率 > 40%（關鍵指標）
- ✅ False Negative 數量下降
- ✅ Confusion matrix: barline_double 不再與 barline_final 混淆
- ✅ Loss 曲線: cls_loss 下降（當前 2.77，目標 < 2.0）

---

## 📂 相關文檔

- `BARLINE_COMPLETE_ANALYSIS.txt` - 根本原因分析
- `OPENSCORE_LIEDER_ANALYSIS.md` - 外部數據源分析
- `PHASE6_IMPLEMENTATION_SUMMARY.md` - 當前實施狀態
- `SYNTHETIC_DATA_SUMMARY.md` - 合成數據研究
- `synthetic_generation/README.md` - 合成系統文檔

---

## 📞 決策點

### 需要確認的問題

1. **是否立即開始 OpenScore 渲染？**
   - 推薦: ✅ 是（最高 ROI）
   - 預期時間: 1-2 天

2. **是否執行標註修正腳本？**
   - 推薦: ✅ 是（快速見效）
   - 預期時間: 4-6 小時

3. **是否使用激進加權配置？**
   - 推薦: ✅ 是（風險可控）
   - 預期時間: 8-12 小時訓練

4. **如果 Week 1 未達標，是否進入 Week 2-3 計劃？**
   - 推薦: 視 Phase 7 結果決定
   - 備選: 合成數據生成或類別合併

---

**總結**: 建議採用**雙管齊下策略**：
1. **短期（1-2 天）**: 方案 1A + 1B（數據增強 + 修正）
2. **中期（3-5 天）**: 方案 2（激進訓練）+ 評估
3. **備用（1-2 週）**: 方案 3（合成數據）或 方案 4（類別合併）

預期經過 Week 1 的努力，barline_double mAP50 可從 0.180 提升至 **0.35-0.45**，召回率從 19.2% 提升至 **40-50%**，達到可用水平。
