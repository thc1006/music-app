#!/usr/bin/env python3
"""
🔧 Phase 1: 數據集優化腳本
解決 YOLO12 訓練的核心問題：
1. 類別缺失（stem_down, slur）
2. 類別不平衡
3. 驗證集過小
"""

import json
import shutil
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import os

# ============== 配置 ==============
ORIGINAL_DATASET = Path("datasets/yolo_harmony_v2_35classes")
OPTIMIZED_DATASET = Path("datasets/yolo_harmony_v2_optimized")
RANDOM_SEED = 42

# 問題類別處理策略
PROBLEM_CLASSES = {
    3: "merge_to_2",   # stem_down → stem_up (無法區分)
    30: "exclude",     # slur (無數據，暫時排除)
}

# 目標類別數量（排除問題類別後）
FINAL_NUM_CLASSES = 33  # 35 - 2

# 新的類別映射（排除 3 和 30 後重新編號）
# 原始: 0,1,2,3,4,5,...,30,...,34
# 新的: 0,1,2,skip,3,4,...,skip,...,32
OLD_TO_NEW_CLASS = {}
new_id = 0
for old_id in range(35):
    if old_id == 3:
        OLD_TO_NEW_CLASS[3] = 2  # stem_down → stem_up
    elif old_id == 30:
        OLD_TO_NEW_CLASS[30] = -1  # 排除
    else:
        OLD_TO_NEW_CLASS[old_id] = new_id
        new_id += 1

# ============== 函數 ==============

def load_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """讀取 YOLO 格式標註"""
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    labels.append((cls, x, y, w, h))
    return labels

def save_labels(label_path: Path, labels: List[Tuple[int, float, float, float, float]]):
    """保存 YOLO 格式標註"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w') as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def remap_labels(labels: List[Tuple], old_to_new: Dict[int, int]) -> List[Tuple]:
    """重新映射類別 ID"""
    remapped = []
    for cls, x, y, w, h in labels:
        new_cls = old_to_new.get(cls, -1)
        if new_cls >= 0:  # 排除 -1 (被移除的類別)
            remapped.append((new_cls, x, y, w, h))
    return remapped

def analyze_dataset(dataset_path: Path) -> Dict:
    """分析數據集統計"""
    stats = {
        'train': {'images': 0, 'labels': Counter()},
        'val': {'images': 0, 'labels': Counter()}
    }

    for split in ['train', 'val']:
        label_dir = dataset_path / split / 'labels'
        if label_dir.exists():
            for label_file in label_dir.glob('*.txt'):
                stats[split]['images'] += 1
                labels = load_labels(label_file)
                for cls, *_ in labels:
                    stats[split]['labels'][cls] += 1

    return stats

def create_optimized_dataset():
    """創建優化後的數據集"""
    print("=" * 60)
    print("🔧 Phase 1: 數據集優化")
    print("=" * 60)

    # 1. 分析原始數據集
    print("\n📊 分析原始數據集...")
    original_stats = analyze_dataset(ORIGINAL_DATASET)

    print(f"\n原始數據集統計：")
    print(f"  訓練集: {original_stats['train']['images']} 張圖片")
    print(f"  驗證集: {original_stats['val']['images']} 張圖片")

    # 顯示問題類別
    print(f"\n🔴 問題類別分析：")
    for cls, action in PROBLEM_CLASSES.items():
        train_count = original_stats['train']['labels'].get(cls, 0)
        val_count = original_stats['val']['labels'].get(cls, 0)
        print(f"  Class {cls}: train={train_count}, val={val_count} → {action}")

    # 2. 創建優化數據集目錄
    print(f"\n📁 創建優化數據集目錄: {OPTIMIZED_DATASET}")
    if OPTIMIZED_DATASET.exists():
        shutil.rmtree(OPTIMIZED_DATASET)

    for split in ['train', 'val']:
        (OPTIMIZED_DATASET / split / 'images').mkdir(parents=True)
        (OPTIMIZED_DATASET / split / 'labels').mkdir(parents=True)

    # 3. 收集所有圖片並重新分配
    print("\n🔄 重新分配訓練/驗證集...")
    all_samples = []

    for split in ['train', 'val']:
        img_dir = ORIGINAL_DATASET / split / 'images'
        label_dir = ORIGINAL_DATASET / split / 'labels'

        if img_dir.exists():
            for img_file in img_dir.glob('*.png'):
                label_file = label_dir / img_file.with_suffix('.txt').name
                if label_file.exists():
                    all_samples.append({
                        'img': img_file,
                        'label': label_file
                    })

    print(f"  總共收集: {len(all_samples)} 個樣本")

    # 4. 隨機打亂並重新分配 (80:20)
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"  新訓練集: {len(train_samples)} 張 (80%)")
    print(f"  新驗證集: {len(val_samples)} 張 (20%)")

    # 5. 處理並保存數據
    print("\n🔧 處理標註並保存...")

    new_stats = {
        'train': {'images': 0, 'labels': Counter()},
        'val': {'images': 0, 'labels': Counter()}
    }

    for split, samples in [('train', train_samples), ('val', val_samples)]:
        for sample in samples:
            # 讀取原始標註
            labels = load_labels(sample['label'])

            # 重新映射類別
            remapped = remap_labels(labels, OLD_TO_NEW_CLASS)

            if remapped:  # 只保存有標註的圖片
                # 複製圖片
                dst_img = OPTIMIZED_DATASET / split / 'images' / sample['img'].name
                shutil.copy2(sample['img'], dst_img)

                # 保存重新映射的標註
                dst_label = OPTIMIZED_DATASET / split / 'labels' / sample['label'].name
                save_labels(dst_label, remapped)

                new_stats[split]['images'] += 1
                for cls, *_ in remapped:
                    new_stats[split]['labels'][cls] += 1

    # 6. 生成新的 YAML 配置
    print("\n📝 生成新的數據集配置...")

    # 新的類別名稱（排除 stem_down 和 slur）
    original_names = [
        "notehead_filled", "notehead_hollow", "stem", "beam",  # 0-3 (原4變新3)
        "flag_8th", "flag_16th", "flag_32nd", "augmentation_dot", "tie",  # 4-8
        "clef_treble", "clef_bass", "clef_alto", "clef_tenor",  # 9-12
        "accidental_sharp", "accidental_flat", "accidental_natural",  # 13-15
        "accidental_double_sharp", "accidental_double_flat",  # 16-17
        "rest_whole", "rest_half", "rest_quarter", "rest_8th", "rest_16th",  # 18-22
        "barline", "barline_double", "barline_final", "barline_repeat",  # 23-26
        "time_signature", "key_signature",  # 27-28
        "fermata", "dynamic_soft", "dynamic_loud",  # 29-31 (原31,32,33)
        "ledger_line"  # 32 (原34)
    ]

    yaml_content = f"""# 🔧 YOLO12 Harmony OMR V2 - Optimized (33 Classes)
# 優化版本：修復類別缺失，重新平衡數據集
# Generated: 2025-11-22

path: {OPTIMIZED_DATASET.absolute()}
train: train/images
val: val/images

nc: {FINAL_NUM_CLASSES}

# 類別說明：
# - stem_down (原 Class 3) 已合併到 stem (Class 2)
# - slur (原 Class 30) 已排除（無數據）

names:
"""
    for i, name in enumerate(original_names):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = OPTIMIZED_DATASET / 'harmony_optimized.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # 7. 輸出最終統計
    print("\n" + "=" * 60)
    print("✅ 優化完成！")
    print("=" * 60)

    print(f"\n📊 新數據集統計：")
    print(f"  訓練集: {new_stats['train']['images']} 張圖片")
    print(f"  驗證集: {new_stats['val']['images']} 張圖片")
    print(f"  類別數: {FINAL_NUM_CLASSES}")

    print(f"\n🔧 主要改進：")
    print(f"  1. stem_down 合併到 stem（解決無數據問題）")
    print(f"  2. slur 暫時排除（解決無數據問題）")
    print(f"  3. 驗證集從 {original_stats['val']['images']} 張增加到 {new_stats['val']['images']} 張")

    # 類別分布
    print(f"\n📈 新類別分布（前 10 個）：")
    all_labels = new_stats['train']['labels'] + new_stats['val']['labels']
    for cls, count in all_labels.most_common(10):
        print(f"  Class {cls}: {count:,} 個標註")

    # 稀有類別警告
    print(f"\n⚠️  稀有類別（< 500 個標註）：")
    for cls in range(FINAL_NUM_CLASSES):
        count = all_labels.get(cls, 0)
        if count < 500:
            print(f"  Class {cls} ({original_names[cls] if cls < len(original_names) else 'unknown'}): {count} 個")

    print(f"\n📁 輸出位置: {OPTIMIZED_DATASET.absolute()}")
    print(f"📝 配置文件: {yaml_path.absolute()}")

    return new_stats

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    create_optimized_dataset()
