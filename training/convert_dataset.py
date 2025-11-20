#!/usr/bin/env python3
"""
MUSCIMA++ 資料集轉換為 YOLO 格式

作者: thc1006 + Claude
日期: 2025-11-20

功能:
1. 解析 MUSCIMA++ XML 標註檔案
2. 轉換為 YOLO txt 格式 (class x_center y_center width height)
3. 自動分割訓練/驗證集 (80/20)
4. 生成資料集統計報告

使用方式:
    python convert_dataset.py --input datasets/muscima-pp --output datasets/yolo_harmony
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
import json
import random
from collections import defaultdict
from PIL import Image
import shutil

# ============= 類別定義 =============

# 四部和聲 OMR 20 個類別（對應 omr_harmony.yaml）
HARMONY_CLASSES = [
    "notehead_filled",    # 0
    "notehead_hollow",    # 1
    "stem_up",            # 2
    "stem_down",          # 3
    "beam",               # 4
    "flag",               # 5
    "clef_treble",        # 6
    "clef_bass",          # 7
    "clef_alto",          # 8
    "clef_tenor",         # 9
    "accidental_sharp",   # 10
    "accidental_flat",    # 11
    "accidental_natural", # 12
    "rest_quarter",       # 13
    "rest_half",          # 14
    "rest_whole",         # 15
    "barline",            # 16
    "time_signature",     # 17
    "key_signature",      # 18
    "staffline"           # 19
]

CLASS_TO_IDX = {c: i for i, c in enumerate(HARMONY_CLASSES)}

# MUSCIMA++ 類別映射到我們的類別
MUSCIMA_TO_HARMONY = {
    # 音符頭
    'noteheadFull': 'notehead_filled',
    'noteheadHalf': 'notehead_hollow',
    'noteheadWhole': 'notehead_hollow',
    'noteheadFullSmall': 'notehead_filled',
    'noteheadHalfSmall': 'notehead_hollow',

    # 符幹與符尾
    'stem': 'stem_up',  # 預設，後續可根據方向判斷
    'beam': 'beam',
    'flag8thUp': 'flag',
    'flag8thDown': 'flag',
    'flag16thUp': 'flag',
    'flag16thDown': 'flag',

    # 譜號
    'g-clef': 'clef_treble',
    'f-clef': 'clef_bass',
    'c-clef': 'clef_alto',  # C clef 可能是 alto 或 tenor

    # 變音記號
    'sharp': 'accidental_sharp',
    'flat': 'accidental_flat',
    'natural': 'accidental_natural',
    'doubleSharp': 'accidental_sharp',
    'doubleFlat': 'accidental_flat',

    # 休止符
    'restQuarter': 'rest_quarter',
    'restHalf': 'rest_half',
    'restWhole': 'rest_whole',
    'rest8th': 'rest_quarter',
    'rest16th': 'rest_quarter',

    # 小節線與其他
    'barline': 'barline',
    'thinBarline': 'barline',
    'thickBarline': 'barline',
    'repeatDot': 'barline',

    # 拍號與調號
    'timeSignature': 'time_signature',
    'keySignature': 'key_signature',

    # 五線譜線
    'staffLine': 'staffline',
}


# ============= XML 解析 =============

def parse_muscima_xml(xml_path: Path) -> List[Tuple[str, List[int]]]:
    """
    解析 MUSCIMA++ XML 標註檔案

    返回: [(class_name, [left, top, width, height]), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    for node in root.findall('.//Node'):
        # 取得類別名稱
        class_elem = node.find('ClassName')
        if class_elem is None:
            continue

        muscima_class = class_elem.text

        # 映射到我們的類別
        harmony_class = MUSCIMA_TO_HARMONY.get(muscima_class)
        if harmony_class is None:
            continue  # 跳過不需要的類別

        # 取得 bounding box
        try:
            top = int(node.find('Top').text)
            left = int(node.find('Left').text)
            width = int(node.find('Width').text)
            height = int(node.find('Height').text)
        except (AttributeError, ValueError) as e:
            print(f"⚠️  警告: 無法解析 bbox: {e}")
            continue

        # 基本驗證
        if width <= 0 or height <= 0:
            continue

        annotations.append((harmony_class, [left, top, width, height]))

    return annotations


def convert_to_yolo_format(
    annotations: List[Tuple[str, List[int]]],
    img_width: int,
    img_height: int
) -> List[str]:
    """
    轉換為 YOLO 格式

    YOLO 格式: <class_id> <x_center> <y_center> <width> <height>
    所有值正規化到 [0, 1]
    """
    yolo_lines = []

    for class_name, (left, top, width, height) in annotations:
        class_id = CLASS_TO_IDX[class_name]

        # 轉換為中心點座標
        x_center = left + width / 2
        y_center = top + height / 2

        # 正規化
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height

        # 確保在 [0, 1] 範圍內
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        yolo_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
        yolo_lines.append(yolo_line)

    return yolo_lines


# ============= 主轉換流程 =============

def convert_muscima_to_yolo(input_dir: Path, output_dir: Path, train_ratio: float = 0.8):
    """
    主轉換流程
    """
    print("=" * 60)
    print("MUSCIMA++ → YOLO 格式轉換")
    print("=" * 60)
    print(f"輸入目錄: {input_dir}")
    print(f"輸出目錄: {output_dir}")
    print(f"訓練/驗證比例: {train_ratio:.0%} / {1-train_ratio:.0%}")
    print()

    # 建立輸出目錄結構
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # 尋找所有 XML 和圖片檔案
    xml_dir = input_dir / 'v2.0' / 'data' / 'annotations'
    img_dir = input_dir / 'v2.0' / 'data' / 'images'

    if not xml_dir.exists():
        # 嘗試其他可能的路徑
        xml_dir = input_dir / 'data' / 'annotations'
        img_dir = input_dir / 'data' / 'images'

    if not xml_dir.exists() or not img_dir.exists():
        print(f"❌ 錯誤: 找不到 XML 或圖片目錄")
        print(f"   XML: {xml_dir}")
        print(f"   IMG: {img_dir}")
        return

    xml_files = list(xml_dir.glob('*.xml'))
    print(f"找到 {len(xml_files)} 個 XML 標註檔案")

    if len(xml_files) == 0:
        print("❌ 錯誤: 沒有找到任何 XML 檔案")
        return

    # 統計資訊
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'class_counts': defaultdict(int),
        'train_images': 0,
        'val_images': 0,
        'skipped': 0
    }

    # 隨機分割訓練/驗證集
    random.shuffle(xml_files)
    train_count = int(len(xml_files) * train_ratio)
    train_files = xml_files[:train_count]
    val_files = xml_files[train_count:]

    print(f"訓練集: {len(train_files)} 張")
    print(f"驗證集: {len(val_files)} 張")
    print()

    # 處理訓練集
    print("處理訓練集...")
    for xml_file in train_files:
        process_file(xml_file, img_dir, output_dir, 'train', stats)

    # 處理驗證集
    print("處理驗證集...")
    for xml_file in val_files:
        process_file(xml_file, img_dir, output_dir, 'val', stats)

    # 印出統計資訊
    print("\n" + "=" * 60)
    print("轉換完成統計")
    print("=" * 60)
    print(f"總圖片數: {stats['total_images']}")
    print(f"總標註數: {stats['total_annotations']}")
    print(f"平均每張: {stats['total_annotations'] / max(1, stats['total_images']):.1f} 個標註")
    print(f"訓練集: {stats['train_images']} 張")
    print(f"驗證集: {stats['val_images']} 張")
    print(f"跳過: {stats['skipped']} 張（無圖片或無標註）")
    print()

    print("各類別統計:")
    for class_name in HARMONY_CLASSES:
        count = stats['class_counts'][class_name]
        print(f"  {class_name:25s}: {count:5d}")
    print()

    # 儲存統計資訊
    stats_file = output_dir / 'conversion_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_images': stats['total_images'],
            'total_annotations': stats['total_annotations'],
            'train_images': stats['train_images'],
            'val_images': stats['val_images'],
            'class_counts': dict(stats['class_counts'])
        }, f, indent=2)

    print(f"統計資訊已儲存: {stats_file}")


def process_file(xml_file: Path, img_dir: Path, output_dir: Path, split: str, stats: Dict):
    """處理單一檔案"""
    # 找對應的圖片
    img_name = xml_file.stem + '.png'
    img_path = img_dir / img_name

    if not img_path.exists():
        # 嘗試 jpg
        img_name = xml_file.stem + '.jpg'
        img_path = img_dir / img_name

    if not img_path.exists():
        stats['skipped'] += 1
        return

    # 解析 XML
    try:
        annotations = parse_muscima_xml(xml_file)
    except Exception as e:
        print(f"⚠️  警告: 無法解析 {xml_file.name}: {e}")
        stats['skipped'] += 1
        return

    if len(annotations) == 0:
        stats['skipped'] += 1
        return

    # 取得圖片尺寸
    try:
        with Image.open(img_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"⚠️  警告: 無法讀取 {img_path.name}: {e}")
        stats['skipped'] += 1
        return

    # 轉換為 YOLO 格式
    yolo_lines = convert_to_yolo_format(annotations, img_width, img_height)

    if len(yolo_lines) == 0:
        stats['skipped'] += 1
        return

    # 複製圖片
    dst_img = output_dir / 'images' / split / img_name
    shutil.copy(img_path, dst_img)

    # 儲存 YOLO 標註
    label_name = xml_file.stem + '.txt'
    dst_label = output_dir / 'labels' / split / label_name
    with open(dst_label, 'w') as f:
        f.write('\n'.join(yolo_lines))

    # 更新統計
    stats['total_images'] += 1
    stats['total_annotations'] += len(yolo_lines)
    stats[f'{split}_images'] += 1

    for class_name, _ in annotations:
        stats['class_counts'][class_name] += 1


# ============= 命令列介面 =============

def main():
    parser = argparse.ArgumentParser(
        description='將 MUSCIMA++ 資料集轉換為 YOLO 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=Path('datasets/muscima-pp'),
        help='MUSCIMA++ 資料集目錄 (預設: datasets/muscima-pp)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('datasets/yolo_harmony'),
        help='輸出目錄 (預設: datasets/yolo_harmony)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='訓練集比例 (預設: 0.8)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='隨機種子 (預設: 42)'
    )

    args = parser.parse_args()

    # 設定隨機種子
    random.seed(args.seed)

    # 轉換
    convert_muscima_to_yolo(args.input, args.output, args.train_ratio)


if __name__ == '__main__':
    main()
