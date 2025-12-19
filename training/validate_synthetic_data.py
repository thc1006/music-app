#!/usr/bin/env python3
"""
驗證合成數據質量

檢查項目:
1. 圖片和標註數量一致
2. YOLO 格式正確
3. Bbox 座標範圍 [0, 1]
4. 統計類別分佈
5. 檢查異常值

使用方法:
    python validate_synthetic_data.py datasets/yolo_synthetic_phase8
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def load_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """讀取 YOLO 標註文件"""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            boxes.append((class_id, x, y, w, h))
    return boxes

def validate_bbox(bbox: Tuple[int, float, float, float, float]) -> List[str]:
    """驗證單個 bbox"""
    errors = []
    class_id, x, y, w, h = bbox

    # 檢查座標範圍
    if not (0 <= x <= 1):
        errors.append(f"x={x:.4f} 超出範圍 [0, 1]")
    if not (0 <= y <= 1):
        errors.append(f"y={y:.4f} 超出範圍 [0, 1]")
    if not (0 < w <= 1):
        errors.append(f"w={w:.4f} 超出範圍 (0, 1]")
    if not (0 < h <= 1):
        errors.append(f"h={h:.4f} 超出範圍 (0, 1]")

    # 檢查邊界是否超出
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2

    if x_min < 0 or x_max > 1:
        errors.append(f"x 邊界 [{x_min:.4f}, {x_max:.4f}] 超出 [0, 1]")
    if y_min < 0 or y_max > 1:
        errors.append(f"y 邊界 [{y_min:.4f}, {y_max:.4f}] 超出 [0, 1]")

    return errors

def main():
    if len(sys.argv) < 2:
        print("使用方法: python validate_synthetic_data.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'

    print("=" * 60)
    print("🔍 合成數據驗證")
    print("=" * 60)
    print(f"數據集: {dataset_dir}")
    print()

    # 1. 檢查目錄存在
    if not images_dir.exists():
        print(f"❌ 圖片目錄不存在: {images_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"❌ 標註目錄不存在: {labels_dir}")
        sys.exit(1)

    # 2. 統計文件數量
    images = sorted(images_dir.glob('*.png'))
    labels = sorted(labels_dir.glob('*.txt'))

    print(f"📊 文件數量:")
    print(f"  圖片: {len(images)}")
    print(f"  標註: {len(labels)}")

    if len(images) != len(labels):
        print(f"⚠️  圖片和標註數量不一致！")
    else:
        print(f"✅ 圖片和標註數量一致")
    print()

    # 3. 檢查文件配對
    print(f"🔍 檢查文件配對...")
    image_names = {img.stem for img in images}
    label_names = {lbl.stem for lbl in labels}

    missing_labels = image_names - label_names
    extra_labels = label_names - image_names

    if missing_labels:
        print(f"❌ 缺少標註的圖片 ({len(missing_labels)}):")
        for name in sorted(list(missing_labels)[:5]):
            print(f"    {name}.png")
        if len(missing_labels) > 5:
            print(f"    ... 還有 {len(missing_labels) - 5} 個")
    else:
        print(f"✅ 所有圖片都有對應標註")

    if extra_labels:
        print(f"⚠️  多餘的標註文件 ({len(extra_labels)}):")
        for name in sorted(list(extra_labels)[:5]):
            print(f"    {name}.txt")
        if len(extra_labels) > 5:
            print(f"    ... 還有 {len(extra_labels) - 5} 個")
    print()

    # 4. 驗證 YOLO 格式
    print(f"🔍 驗證 YOLO 格式...")
    class_counts = defaultdict(int)
    bbox_size_stats = defaultdict(list)
    total_bboxes = 0
    error_count = 0
    empty_labels = []

    for label_path in labels:
        try:
            boxes = load_yolo_label(label_path)

            if not boxes:
                empty_labels.append(label_path.name)
                continue

            total_bboxes += len(boxes)

            for bbox in boxes:
                class_id, x, y, w, h = bbox
                class_counts[class_id] += 1

                # 收集尺寸統計
                bbox_size_stats[class_id].append((w, h))

                # 驗證 bbox
                errors = validate_bbox(bbox)
                if errors:
                    if error_count < 5:  # 只顯示前 5 個錯誤
                        print(f"  ❌ {label_path.name}: {', '.join(errors)}")
                    error_count += 1

        except Exception as e:
            print(f"  ❌ 無法讀取 {label_path.name}: {e}")
            error_count += 1

    if error_count == 0:
        print(f"✅ 所有標註格式正確")
    else:
        print(f"⚠️  發現 {error_count} 個錯誤")

    if empty_labels:
        print(f"⚠️  空標註文件 ({len(empty_labels)}):")
        for name in empty_labels[:5]:
            print(f"    {name}")
        if len(empty_labels) > 5:
            print(f"    ... 還有 {len(empty_labels) - 5} 個")
    print()

    # 5. 類別統計
    print(f"📊 類別分佈:")
    print(f"  總 bboxes: {total_bboxes}")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = count / total_bboxes * 100 if total_bboxes > 0 else 0
        print(f"  Class {class_id}: {count:>6} ({percentage:5.1f}%)")
    print()

    # 6. Bbox 尺寸統計
    print(f"📏 Bbox 尺寸統計:")
    for class_id in sorted(bbox_size_stats.keys()):
        sizes = bbox_size_stats[class_id]
        widths = [w for w, h in sizes]
        heights = [h for w, h in sizes]

        print(f"  Class {class_id}:")
        print(f"    寬度: min={min(widths):.4f}, max={max(widths):.4f}, "
              f"avg={sum(widths)/len(widths):.4f}")
        print(f"    高度: min={min(heights):.4f}, max={max(heights):.4f}, "
              f"avg={sum(heights)/len(heights):.4f}")
    print()

    # 7. 每圖 bbox 數量統計
    bboxes_per_img = []
    for label_path in labels:
        boxes = load_yolo_label(label_path)
        bboxes_per_img.append(len(boxes))

    if bboxes_per_img:
        print(f"📈 每圖 Bbox 統計:")
        print(f"  最少: {min(bboxes_per_img)}")
        print(f"  最多: {max(bboxes_per_img)}")
        print(f"  平均: {sum(bboxes_per_img)/len(bboxes_per_img):.2f}")
        print(f"  中位數: {sorted(bboxes_per_img)[len(bboxes_per_img)//2]}")
    print()

    # 8. 讀取生成統計（如果存在）
    stats_file = dataset_dir / 'generation_stats.json'
    if stats_file.exists():
        print(f"📊 生成統計 (來自 {stats_file.name}):")
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(json.dumps(stats, indent=2))
        print()

    # 9. 最終總結
    print("=" * 60)
    print("✅ 驗證完成")
    print("=" * 60)

    if error_count == 0 and not empty_labels and len(images) == len(labels):
        print("🎉 數據集完全正常，可以用於訓練！")
        return 0
    else:
        print("⚠️  發現一些問題，建議檢查並修復")
        return 1

if __name__ == '__main__':
    sys.exit(main())
