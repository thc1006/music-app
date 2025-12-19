# Phase 10 實作執行計劃
## 基於現有資源的數據質量修復方案

**創建日期**: 2025-12-02
**基於**: 全面目錄掃描結果
**目標**: 利用現有工具和資源，快速執行 Phase 10 數據質量修復

---

## 📊 資源現狀總結

### ✅ 已有的關鍵資源

| 類別 | 數量 | 關鍵發現 |
|------|------|----------|
| **Python 腳本** | 66個 | 完整的轉換、合併、驗證工具鏈 |
| **數據轉換腳本** | 11個 | 所有外部數據集都有轉換腳本 |
| **數據集合併腳本** | 6個 | Phase 3-5, 8 的合併經驗 |
| **渲染腳本** | 3個 | OpenScore 渲染工具已完備 |
| **訓練腳本** | 12個 | 從 Phase 2-9 的完整訓練代碼 |
| **已轉換數據集** | 23個 | YOLO 格式，隨時可用 |
| **外部數據源** | 9+個 | 高質量 OMR 數據集已下載 |

### 🎯 Phase 10 的關鍵發現

1. **DoReMi 轉換腳本已存在**：
   - `convert_doremi_to_yolo.py` (691行)
   - 需要分析和修復座標轉換邏輯

2. **OpenScore 渲染腳本已完備**：
   - `render_openscore_to_yolo.py` (341行)
   - `render_openscore_mscx_to_yolo.py` (428行)
   - `render_openscore_with_musescore.py` (496行)
   - 可以直接使用或修改

3. **MUSCIMA++ 已轉換**：
   - `yolo_muscima_converted/` 已存在
   - `convert_muscima_to_yolo.py` (335行)
   - 可能需要標準化處理

4. **數據合併經驗豐富**：
   - `merge_datasets_phase3.py` - `merge_phase8_dataset.py`
   - 有成熟的合併模板可參考

---

## 🚀 Phase 10 執行方案（基於現有資源）

### 方案調整：充分利用現有代碼

**原計劃**：從零開始寫新腳本
**調整後**：基於現有腳本修改和優化

---

## 📋 Task 1: DoReMi 座標修復（2天）

### 1.1 分析現有轉換腳本（Day 1 上午）

**目標**：理解當前 DoReMi 轉換腳本的問題

```bash
cd ~/dev/music-app/training

# Step 1: 讀取現有 DoReMi 轉換腳本
cat convert_doremi_to_yolo.py | head -100

# Step 2: 檢查 DoReMi 數據結構
ls -la datasets/external/omr_downloads/DoReMi/DoReMi_v1/OMR_XML/ | head -20

# Step 3: 抽樣檢查 3 個 XML 文件結構
python3 << 'EOF'
from pathlib import Path
import xml.etree.ElementTree as ET

doremi_dir = Path('datasets/external/omr_downloads/DoReMi/DoReMi_v1/OMR_XML')
sample_files = sorted(list(doremi_dir.glob('*.xml')))[:3]

for xml_file in sample_files:
    print(f'\n{"="*60}')
    print(f'文件: {xml_file.name}')
    print(f'{"="*60}')

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 檢查圖片尺寸
    print(f'\n[圖片信息]')
    print(f'  Width: {root.find(".//ImageWidth").text}')
    print(f'  Height: {root.find(".//ImageHeight").text}')

    # 檢查是否有 Resolution 標籤
    resolution = root.find(".//Resolution")
    if resolution is not None:
        print(f'  Resolution: {resolution.text}')

    # 檢查 Glyph 結構（前 3 個）
    glyphs = root.findall('.//Glyph')[:3]
    print(f'\n[Glyph 樣本] (共 {len(root.findall(".//Glyph"))} 個)')

    for i, glyph in enumerate(glyphs, 1):
        print(f'\n  Glyph {i}:')
        for child in glyph:
            if child.text and len(child.text) < 50:
                print(f'    {child.tag}: {child.text}')
EOF
```

**預期輸出**：了解 DoReMi XML 的確切結構和座標系統

### 1.2 定位問題並修復（Day 1 下午）

**基於**：`convert_doremi_to_yolo.py` (691行)

**修復策略**：
1. 檢查座標提取邏輯（Line ~300-400）
2. 檢查座標轉換邏輯（YOLO normalization）
3. 檢查 bbox 尺寸計算

**執行**：
```bash
# 創建修復版本
cp convert_doremi_to_yolo.py convert_doremi_to_yolo_FIXED.py

# 使用 Claude Code 修復關鍵部分
# (接下來的步驟會詳細指導)
```

### 1.3 驗證修復效果（Day 2）

```bash
# 重新轉換 DoReMi 數據集
python3 convert_doremi_to_yolo_FIXED.py \
    --source datasets/external/omr_downloads/DoReMi/DoReMi_v1 \
    --output datasets/yolo_doremi_converted_FIXED \
    --dry-run  # 先測試 10 個文件

# 對比舊版和新版的標註
python3 << 'EOF'
import sys
from pathlib import Path

old_dir = Path('datasets/yolo_doremi_converted/train/labels')
new_dir = Path('datasets/yolo_doremi_converted_FIXED/train/labels')

# 隨機選擇 5 個文件對比
sample_files = sorted(list(old_dir.glob('*.txt')))[:5]

for label_file in sample_files:
    old_path = label_file
    new_path = new_dir / label_file.name

    if not new_path.exists():
        continue

    print(f'\n{"="*60}')
    print(f'文件: {label_file.name}')
    print(f'{"="*60}')

    # 讀取舊標註
    old_lines = old_path.read_text().strip().split('\n')
    new_lines = new_path.read_text().strip().split('\n')

    print(f'  舊版標註數: {len(old_lines)}')
    print(f'  新版標註數: {len(new_lines)}')

    # 計算 bbox 面積統計
    old_areas = []
    new_areas = []

    for line in old_lines:
        if line.strip():
            parts = line.split()
            w, h = float(parts[3]), float(parts[4])
            old_areas.append(w * h)

    for line in new_lines:
        if line.strip():
            parts = line.split()
            w, h = float(parts[3]), float(parts[4])
            new_areas.append(w * h)

    if old_areas:
        print(f'  舊版平均面積: {sum(old_areas)/len(old_areas):.6f}')
        print(f'  舊版 tiny 比例: {sum(1 for a in old_areas if a < 0.0005)/len(old_areas)*100:.1f}%')

    if new_areas:
        print(f'  新版平均面積: {sum(new_areas)/len(new_areas):.6f}')
        print(f'  新版 tiny 比例: {sum(1 for a in new_areas if a < 0.0005)/len(new_areas)*100:.1f}%')

    # 顯示前 3 個標註對比
    print(f'\n  [前 3 個標註對比]')
    for i in range(min(3, len(old_lines), len(new_lines))):
        print(f'  舊: {old_lines[i]}')
        print(f'  新: {new_lines[i]}')
        print()
EOF
```

**成功標準**：
- Tiny bbox 比例從 100% 降至 < 10%
- 平均面積從 < 0.0001 提升至 > 0.001
- flag_16th, flag_32nd, rest_16th 標註合理化

---

## 📋 Task 2: OpenScore Fermata 重新渲染（1-2天）

### 2.1 分析現有渲染腳本（Day 3 上午）

**已有的渲染腳本**：
1. `render_openscore_to_yolo.py` (341行)
2. `render_openscore_mscx_to_yolo.py` (428行)
3. `render_openscore_with_musescore.py` (496行)

```bash
# 檢查哪個腳本最適合
cd ~/dev/music-app/training

# 方案 1: 查看 render_openscore_to_yolo.py
head -50 render_openscore_to_yolo.py

# 方案 2: 查看 render_openscore_mscx_to_yolo.py
head -50 render_openscore_mscx_to_yolo.py

# 檢查 OpenScore Lieder 數據現狀
ls -la datasets/yolo_openscore_lieder/train/labels/*.txt | wc -l
```

### 2.2 使用最佳腳本重新渲染（Day 3 下午 - Day 4）

**策略**：
1. 如果原腳本使用 Verovio，檢查是否正確提取座標
2. 如果座標提取有問題，修改為使用 Verovio 原生 bbox API
3. 重新渲染優先針對 fermata

```bash
# 創建修復版本
cp render_openscore_to_yolo.py render_openscore_to_yolo_FIXED.py

# 測試 10 個文件
python3 render_openscore_to_yolo_FIXED.py \
    --source datasets/external/omr_downloads/OpenScoreLieder/scores \
    --output datasets/yolo_openscore_lieder_FIXED \
    --max-files 10 \
    --focus-fermata  # 優先處理含 fermata 的文件

# 驗證 fermata bbox 尺寸
python3 << 'EOF'
from pathlib import Path

labels_dir = Path('datasets/yolo_openscore_lieder_FIXED/train/labels')

fermata_areas = []
for label_file in labels_dir.glob('*.txt'):
    for line in label_file.read_text().strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        if class_id == 29:  # fermata
            w, h = float(parts[3]), float(parts[4])
            fermata_areas.append(w * h)

if fermata_areas:
    print(f'Fermata 標註數: {len(fermata_areas)}')
    print(f'平均面積: {sum(fermata_areas)/len(fermata_areas):.6f}')
    print(f'Tiny 比例: {sum(1 for a in fermata_areas if a < 0.0003)/len(fermata_areas)*100:.1f}%')
    print(f'合理比例: {sum(1 for a in fermata_areas if a >= 0.001)/len(fermata_areas)*100:.1f}%')
EOF
```

**成功標準**：
- Fermata tiny 比例從 89% 降至 < 10%
- 平均面積 > 0.002
- 視覺檢查：bbox 緊密包圍 fermata 符號

---

## 📋 Task 3: MUSCIMA++ 標註標準化（1天）

### 3.1 分析 MUSCIMA++ 當前標註（Day 5 上午）

```bash
cd ~/dev/music-app/training

# 檢查 MUSCIMA++ 數據集
ls -la datasets/yolo_muscima_converted/

# 統計各類別的 bbox 尺寸
python3 << 'EOF'
from pathlib import Path
import numpy as np
from collections import defaultdict

labels_dir = Path('datasets/yolo_muscima_converted/train/labels')

class_stats = defaultdict(list)

for label_file in labels_dir.glob('*.txt'):
    for line in label_file.read_text().strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        w, h = float(parts[3]), float(parts[4])
        area = w * h
        class_stats[class_id].append({'w': w, 'h': h, 'area': area})

# 顯示統計
class_names = {
    4: 'flag_8th',
    8: 'tie',
    23: 'barline',
    # ... 更多
}

print('MUSCIMA++ 標註尺寸統計：')
print(f'{"類別":<15} {"數量":>6} {"中位寬度":>10} {"中位高度":>10} {"中位面積":>10} {"Tiny%":>8}')
print('='*70)

for class_id in sorted(class_stats.keys()):
    stats = class_stats[class_id]
    widths = [s['w'] for s in stats]
    heights = [s['h'] for s in stats]
    areas = [s['area'] for s in stats]

    median_w = np.median(widths)
    median_h = np.median(heights)
    median_area = np.median(areas)
    tiny_ratio = sum(1 for a in areas if a < 0.0005) / len(areas) * 100

    class_name = class_names.get(class_id, f'class_{class_id}')
    print(f'{class_name:<15} {len(stats):>6} {median_w:>10.6f} {median_h:>10.6f} {median_area:>10.6f} {tiny_ratio:>7.1f}%')
EOF
```

### 3.2 標準化處理（Day 5 下午）

**策略**：
- 對於明顯異常的 bbox（過大或過小），根據主數據集的統計調整
- 保持 MUSCIMA++ 的獨特性，不過度修改

```bash
# 創建標準化腳本（基於分析結果）
cat > standardize_muscima_annotations.py << 'EOF'
"""
MUSCIMA++ 標註標準化腳本
基於 Phase 8 主數據集的統計，調整異常尺寸的標註
"""
from pathlib import Path
import shutil
from collections import defaultdict
import numpy as np

# Phase 8 主數據集的參考統計（需要先計算）
reference_stats = {
    4: {'median_area': 0.0015},   # flag_8th
    8: {'median_area': 0.0020},   # tie
    23: {'median_area': 0.0012},  # barline
    # ... 更多
}

def standardize_bbox(class_id, bbox, ref_stats):
    """
    標準化單個 bbox
    """
    w, h = bbox['w'], bbox['h']
    area = w * h

    if class_id not in ref_stats:
        return bbox  # 沒有參考統計，保持不變

    ref_area = ref_stats[class_id]['median_area']

    # 如果面積異常（> 3x 或 < 0.3x 參考），調整
    if area > ref_area * 3 or area < ref_area * 0.3:
        scale_factor = (ref_area / area) ** 0.5
        return {
            'w': w * scale_factor,
            'h': h * scale_factor
        }

    return bbox  # 尺寸合理，不調整

def main():
    source_dir = Path('datasets/yolo_muscima_converted')
    output_dir = Path('datasets/yolo_muscima_converted_STANDARDIZED')

    # 複製整個結構
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

    # 處理所有標註文件
    for split in ['train', 'val']:
        labels_dir = output_dir / split / 'labels'

        for label_file in labels_dir.glob('*.txt'):
            lines = label_file.read_text().strip().split('\n')
            new_lines = []

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split()
                class_id = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                w, h = float(parts[3]), float(parts[4])

                # 標準化
                bbox = {'w': w, 'h': h}
                new_bbox = standardize_bbox(class_id, bbox, reference_stats)

                new_line = f"{class_id} {x} {y} {new_bbox['w']} {new_bbox['h']}"
                new_lines.append(new_line)

            # 寫入
            label_file.write_text('\n'.join(new_lines))

    print('✅ MUSCIMA++ 標註標準化完成')

if __name__ == '__main__':
    main()
EOF

# 執行標準化
python3 standardize_muscima_annotations.py
```

---

## 📋 Task 4: Tiny Bbox 智能清理（1天）

### 4.1 創建智能清理腳本（Day 6）

**基於**：
- `scripts/clean_tiny_annotations_phase9.py` (已存在)
- Phase 9 Critical Analysis 的洞察

```bash
cd ~/dev/music-app/training

# 檢查現有清理腳本
cat scripts/clean_tiny_annotations_phase9.py | head -100

# 創建智能版本
cat > scripts/clean_tiny_annotations_phase10_SMART.py << 'EOF'
"""
Phase 10 智能 Tiny Bbox 清理腳本

核心原則：
1. 不盲目清理所有 tiny bbox
2. 只清理「已知錯誤來源」的 tiny bbox
3. 保留合法的小物件（真實的 flag, rest）

基於 PHASE9_CRITICAL_ANALYSIS.md 的洞察：
- flag_16th: 84.3% tiny 但 mAP 0.731 → 保留（真實小物件）
- fermata: 59.0% tiny 但 mAP 0.644 → 保留
- barline_double: 0.2% tiny 但 mAP 0.231 → 不是 tiny 問題
"""
from pathlib import Path
import shutil
from collections import Counter

# 已知錯誤來源的數據集和類別
KNOWN_ERRORS = {
    'DoReMi': {
        'classes': [4, 5, 6, 22, 23],  # flag_8th/16th/32nd, rest_16th, barline
        'reason': 'DoReMi 座標轉換錯誤',
        'min_area': 0.0005,  # 小於此值的認為是錯誤
    },
    'OpenScore_original': {
        'classes': [29],  # fermata
        'reason': 'OpenScore 原版座標提取錯誤',
        'min_area': 0.0003,
    },
}

def get_dataset_source(image_path):
    """
    根據圖片路徑判斷數據來源
    """
    path_str = str(image_path)

    if 'doremi' in path_str.lower():
        return 'DoReMi'
    elif 'openscore' in path_str.lower() and 'original' in path_str:
        return 'OpenScore_original'

    return 'Unknown'

def should_remove(class_id, bbox, dataset_source):
    """
    決定是否移除此標註
    """
    w, h = bbox['w'], bbox['h']
    area = w * h

    # 檢查是否在已知錯誤清單中
    if dataset_source in KNOWN_ERRORS:
        error_info = KNOWN_ERRORS[dataset_source]

        if class_id in error_info['classes']:
            if area < error_info['min_area']:
                return True, error_info['reason']

    # 不在錯誤清單中，保留
    return False, "合法標註"

def clean_dataset(source_dir, output_dir):
    """
    智能清理數據集
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # 複製整個結構
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

    stats = {
        'total': 0,
        'removed': 0,
        'kept': 0,
        'reasons': Counter(),
    }

    for split in ['train', 'val']:
        images_dir = output_dir / split / 'images'
        labels_dir = output_dir / split / 'labels'

        for label_file in labels_dir.glob('*.txt'):
            # 對應的圖片路徑
            image_file = images_dir / (label_file.stem + '.png')
            dataset_source = get_dataset_source(image_file)

            lines = label_file.read_text().strip().split('\n')
            new_lines = []

            for line in lines:
                if not line.strip():
                    continue

                stats['total'] += 1
                parts = line.split()
                class_id = int(parts[0])
                w, h = float(parts[3]), float(parts[4])

                bbox = {'w': w, 'h': h}
                remove, reason = should_remove(class_id, bbox, dataset_source)

                if remove:
                    stats['removed'] += 1
                    stats['reasons'][reason] += 1
                else:
                    stats['kept'] += 1
                    new_lines.append(line)

            # 寫入
            label_file.write_text('\n'.join(new_lines))

    # 列印統計
    print('='*60)
    print('智能清理統計')
    print('='*60)
    print(f'總標註數: {stats["total"]:,}')
    print(f'保留: {stats["kept"]:,} ({stats["kept"]/stats["total"]*100:.1f}%)')
    print(f'移除: {stats["removed"]:,} ({stats["removed"]/stats["total"]*100:.1f}%)')
    print('\n移除原因分佈：')
    for reason, count in stats['reasons'].most_common():
        print(f'  - {reason}: {count:,}')
    print('='*60)

    return stats

def main():
    # 清理 Phase 8 Final 數據集
    source = 'datasets/yolo_harmony_v2_phase8_final'
    output = 'datasets/yolo_harmony_v2_phase10_clean'

    print(f'清理數據集: {source}')
    print(f'輸出目錄: {output}')
    print()

    stats = clean_dataset(source, output)

    print('\n✅ 智能清理完成！')
    print(f'   輸出位置: {output}')

if __name__ == '__main__':
    main()
EOF

# 執行智能清理
python3 scripts/clean_tiny_annotations_phase10_SMART.py
```

**成功標準**：
- 只移除 5-10% 的標註（vs Phase 9 的 15-20%）
- 保留所有合法的小物件
- flag_16th/32nd, rest_16th 的合法標註全部保留

---

## 📋 Task 5: 合併 Phase 10 數據集（1天）

### 5.1 合併所有修復後的數據（Day 7）

**基於**：
- `merge_datasets_phase3.py` - `merge_phase8_dataset.py` (成熟的合併經驗)

```bash
cd ~/dev/music-app/training

# 創建 Phase 10 合併腳本
cat > merge_phase10_dataset.py << 'EOF'
"""
Phase 10 數據集合併腳本
合併所有修復後的數據集
"""
from pathlib import Path
import shutil
from collections import Counter
import yaml

def merge_datasets(sources, output_dir, split='train'):
    """
    合併多個數據集
    """
    output_dir = Path(output_dir)
    output_images = output_dir / split / 'images'
    output_labels = output_dir / split / 'labels'

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_images': 0,
        'total_labels': 0,
        'class_counts': Counter(),
    }

    for source_name, source_dir in sources.items():
        source_dir = Path(source_dir)
        source_images = source_dir / split / 'images'
        source_labels = source_dir / split / 'labels'

        if not source_images.exists():
            print(f'⚠️  跳過 {source_name} (不存在 {split} split)')
            continue

        print(f'處理: {source_name}')

        # 複製圖片和標註
        for img_file in source_images.glob('*.png'):
            # 重命名以避免衝突
            new_name = f'{source_name}_{img_file.name}'
            shutil.copy2(img_file, output_images / new_name)

            # 對應的標註
            label_file = source_labels / (img_file.stem + '.txt')
            if label_file.exists():
                shutil.copy2(label_file, output_labels / (img_file.stem + '.txt'))

                # 統計
                stats['total_labels'] += 1
                for line in label_file.read_text().strip().split('\n'):
                    if line.strip():
                        class_id = int(line.split()[0])
                        stats['class_counts'][class_id] += 1

            stats['total_images'] += 1

        print(f'  - 圖片數: {len(list(source_images.glob("*.png")))}')

    return stats

def main():
    # 定義數據源
    sources = {
        'phase8_base': 'datasets/yolo_harmony_v2_phase8_final',  # 基礎數據
        'doremi_fixed': 'datasets/yolo_doremi_converted_FIXED',  # 修復後的 DoReMi
        'openscore_fixed': 'datasets/yolo_openscore_lieder_FIXED',  # 修復後的 OpenScore
        'muscima_std': 'datasets/yolo_muscima_converted_STANDARDIZED',  # 標準化的 MUSCIMA
    }

    output_dir = 'datasets/yolo_harmony_v2_phase10_clean'

    print('='*60)
    print('Phase 10 數據集合併')
    print('='*60)
    print()

    # 合併訓練集
    print('[訓練集]')
    train_stats = merge_datasets(sources, output_dir, split='train')

    # 合併驗證集
    print('\n[驗證集]')
    val_stats = merge_datasets(sources, output_dir, split='val')

    # 創建 YAML 配置
    yaml_config = {
        'path': str(Path(output_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 33,
        'names': {
            0: 'notehead_filled',
            1: 'notehead_hollow',
            # ... (完整的 33 個類別)
        }
    }

    yaml_path = output_dir / 'harmony_phase10.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    # 列印統計
    print('\n' + '='*60)
    print('合併統計')
    print('='*60)
    print(f'訓練圖片: {train_stats["total_images"]:,}')
    print(f'訓練標註: {train_stats["total_labels"]:,}')
    print(f'驗證圖片: {val_stats["total_images"]:,}')
    print(f'驗證標註: {val_stats["total_labels"]:,}')
    print(f'\nYAML 配置: {yaml_path}')
    print('='*60)

    print('\n✅ Phase 10 數據集合併完成！')

if __name__ == '__main__':
    main()
EOF

# 執行合併
python3 merge_phase10_dataset.py
```

---

## 📋 Task 6: Phase 10 訓練（Day 8-9）

### 6.1 創建訓練腳本（Day 8 上午）

**基於**：
- `yolo12_train_phase8.py` (203行，最成功的配置)

```bash
cd ~/dev/music-app/training

# 複製 Phase 8 腳本作為基礎
cp yolo12_train_phase8.py yolo12_train_phase10.py

# 修改關鍵配置（接下來會詳細指導）
```

**關鍵修改**：
1. 數據集路徑：`datasets/yolo_harmony_v2_phase10_clean/harmony_phase10.yaml`
2. 輸出目錄：`harmony_omr_v2_phase10/clean_data_training`
3. 預訓練模型：`harmony_omr_v2_phase8/phase8_training/weights/best.pt`
4. 類別加權：針對修復後的瓶頸類別

### 6.2 執行訓練（Day 8 下午 - Day 9）

```bash
cd ~/dev/music-app/training

# 啟動訓練（150 epochs，~9 小時）
python3 yolo12_train_phase10.py

# 或使用 nohup 背景執行
nohup python3 yolo12_train_phase10.py > logs/phase10_training.log 2>&1 &

# 監控進度
tail -f logs/phase10_training.log

# 或查看訓練曲線
ls harmony_omr_v2_phase10/clean_data_training/results.csv
```

---

## 📋 Task 7: 評估和對比（Day 10）

### 7.1 評估 Phase 10 模型

```bash
cd ~/dev/music-app/training

# 查看最終結果
tail -20 harmony_omr_v2_phase10/clean_data_training/results.csv

# 對比 Phase 8 vs Phase 10
python3 << 'EOF'
import pandas as pd

# 讀取 Phase 8 結果
phase8_df = pd.read_csv('harmony_omr_v2_phase8/phase8_training/results.csv')
phase8_final = phase8_df.iloc[-1]

# 讀取 Phase 10 結果
phase10_df = pd.read_csv('harmony_omr_v2_phase10/clean_data_training/results.csv')
phase10_final = phase10_df.iloc[-1]

# 對比
print('='*60)
print('Phase 8 vs Phase 10 對比')
print('='*60)
print(f'{"指標":<20} {"Phase 8":>12} {"Phase 10":>12} {"提升":>10}')
print('-'*60)

metrics = [
    ('mAP50', 'metrics/mAP50(B)'),
    ('mAP50-95', 'metrics/mAP50-95(B)'),
    ('Precision', 'metrics/precision(B)'),
    ('Recall', 'metrics/recall(B)'),
]

for name, col in metrics:
    p8 = phase8_final[col]
    p10 = phase10_final[col]
    improve = (p10 - p8) / p8 * 100

    print(f'{name:<20} {p8:>12.4f} {p10:>12.4f} {improve:>9.1f}%')

print('='*60)
EOF
```

**成功標準**：
- Phase 10 mAP50 > Phase 8 mAP50 (0.6444)
- 目標：mAP50 > 0.69（+7% 提升）
- 瓶頸類別至少提升 20%

---

## 📂 文件組織

### 執行後的目錄結構

```
training/
├── datasets/
│   ├── yolo_doremi_converted_FIXED/          # DoReMi 修復版
│   ├── yolo_openscore_lieder_FIXED/          # OpenScore 修復版
│   ├── yolo_muscima_converted_STANDARDIZED/  # MUSCIMA 標準化版
│   └── yolo_harmony_v2_phase10_clean/        # Phase 10 合併數據集
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── harmony_phase10.yaml
├── harmony_omr_v2_phase10/
│   └── clean_data_training/
│       ├── weights/
│       │   ├── best.pt                        # Phase 10 最佳模型
│       │   └── last.pt
│       ├── results.csv
│       └── args.yaml
├── scripts/
│   ├── clean_tiny_annotations_phase10_SMART.py
│   └── ...
├── convert_doremi_to_yolo_FIXED.py
├── render_openscore_to_yolo_FIXED.py
├── standardize_muscima_annotations.py
├── merge_phase10_dataset.py
├── yolo12_train_phase10.py
└── PHASE10_IMPLEMENTATION_PLAN.md (本文件)
```

---

## ⏱️ 時間估算

| 天數 | 任務 | 預計時間 |
|------|------|---------|
| **Day 1** | DoReMi 分析和修復 | 8小時 |
| **Day 2** | DoReMi 驗證 | 4小時 |
| **Day 3** | OpenScore 分析和修復 | 8小時 |
| **Day 4** | OpenScore 驗證 | 4小時 |
| **Day 5** | MUSCIMA 標準化 | 6小時 |
| **Day 6** | Tiny Bbox 智能清理 | 4小時 |
| **Day 7** | 數據集合併 | 4小時 |
| **Day 8** | 訓練腳本準備 + 開始訓練 | 2小時 + 訓練開始 |
| **Day 9** | 訓練繼續（自動）| 9小時（RTX 5090）|
| **Day 10** | 評估和對比 | 4小時 |

**總計**：約 7-8 個工作天（不含訓練等待時間）

---

## ✅ 成功檢查清單

### DoReMi 修復
- [ ] 分析現有轉換腳本完成
- [ ] 定位座標問題完成
- [ ] 創建修復版腳本
- [ ] 重新轉換 DoReMi 數據集
- [ ] 驗證修復效果（tiny bbox < 10%）

### OpenScore Fermata 修復
- [ ] 分析現有渲染腳本完成
- [ ] 選擇最佳渲染方案
- [ ] 創建修復版腳本
- [ ] 重新渲染 OpenScore Lieder
- [ ] 驗證 fermata bbox（tiny < 10%）

### MUSCIMA++ 標準化
- [ ] 分析當前標註統計
- [ ] 創建標準化腳本
- [ ] 執行標準化處理
- [ ] 驗證標準化效果

### Tiny Bbox 清理
- [ ] 創建智能清理腳本
- [ ] 執行清理（保留合法小物件）
- [ ] 驗證清理統計（移除 < 10%）

### 數據集合併
- [ ] 創建合併腳本
- [ ] 執行合併
- [ ] 生成 YAML 配置
- [ ] 驗證合併後數據集完整性

### Phase 10 訓練
- [ ] 創建訓練腳本
- [ ] 配置類別加權
- [ ] 啟動訓練（150 epochs）
- [ ] 訓練完成

### 評估
- [ ] 提取 Phase 10 指標
- [ ] 對比 Phase 8 vs Phase 10
- [ ] 確認提升 > 5%
- [ ] 記錄瓶頸類別改善

---

## 🚀 立即開始

```bash
# 立即開始 Day 1 任務
cd ~/dev/music-app/training

# 1. 分析 DoReMi 轉換腳本
head -100 convert_doremi_to_yolo.py

# 2. 檢查 DoReMi XML 結構
python3 << 'EOF'
from pathlib import Path
import xml.etree.ElementTree as ET

doremi_dir = Path('datasets/external/omr_downloads/DoReMi/DoReMi_v1/OMR_XML')
sample_files = sorted(list(doremi_dir.glob('*.xml')))[:3]

for xml_file in sample_files:
    print(f'\n{"="*60}')
    print(f'文件: {xml_file.name}')
    print(f'{"="*60}')

    tree = ET.parse(xml_file)
    root = tree.getroot()

    print(f'ImageWidth: {root.find(".//ImageWidth").text}')
    print(f'ImageHeight: {root.find(".//ImageHeight").text}')

    glyphs = root.findall('.//Glyph')[:2]
    print(f'\nGlyph 樣本 (共 {len(root.findall(".//Glyph"))} 個):')
    for i, glyph in enumerate(glyphs, 1):
        print(f'\n  Glyph {i}:')
        for child in glyph:
            if child.text and len(child.text) < 50:
                print(f'    {child.tag}: {child.text}')
EOF
```

---

**準備就緒！開始執行 Phase 10 吧！** 🚀
