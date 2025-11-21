#!/usr/bin/env python3
"""
DeepScoresV2 → YOLO 格式轉換（多進程並行優化版本）
針對 RTX 5090 + i9-14900 (24 cores) 優化

特點：
- 多進程並行處理 (24 workers)
- 內存映射減少 I/O
- 批次寫入優化
- 實時進度顯示
"""

import json
import os
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import numpy as np

from deepscores_to_harmony_mapping import get_harmony_class_id, get_class_name

# 配置
DEEPSCORES_DIR = Path("datasets/ds2_dense")
OUTPUT_DIR = Path("datasets/yolo_harmony")
NUM_WORKERS = min(24, cpu_count())  # 使用全部核心

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    將 COCO bbox [x, y, width, height] 轉換為 YOLO 格式 [x_center, y_center, width, height]
    所有值歸一化到 [0, 1]，並確保在有效範圍內
    """
    x, y, w, h = bbox

    # 確保 bbox 在圖片範圍內
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    w = max(1, min(w, img_width - x))
    h = max(1, min(h, img_height - y))

    # 計算中心點和尺寸（歸一化）
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    # 最終檢查：確保所有值在 [0, 1] 範圍內
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height

def process_single_image(args):
    """
    處理單張圖片的轉換（供多進程調用）

    Args:
        args: (image_data, annotations_dict, images_dir, labels_dir)

    Returns:
        (success: bool, image_id: int, num_annotations: int)
    """
    image_data, annotations_dict, images_dir, labels_dir = args

    image_id = image_data['id']
    filename = image_data['filename']
    img_width = image_data['width']
    img_height = image_data['height']

    # 檢查圖片是否存在
    src_img_path = DEEPSCORES_DIR / "images" / filename
    if not src_img_path.exists():
        return False, image_id, 0

    # 獲取該圖片的所有標註
    ann_ids = image_data.get('ann_ids', [])
    if not ann_ids:
        return False, image_id, 0

    # 轉換標註
    yolo_annotations = []
    for ann_id in ann_ids:
        if ann_id not in annotations_dict:
            continue

        annotation = annotations_dict[ann_id]

        # cat_id 是一個列表，可能包含多個類別
        cat_ids = annotation.get('cat_id', [])
        if not cat_ids:
            continue

        # 轉換邊界框（只取第一次）
        bbox = annotation['a_bbox']  # [x, y, width, height]
        x_center, y_center, width, height = convert_bbox_to_yolo(
            bbox, img_width, img_height
        )

        # 處理每個類別（一個標註可能有多個類別）
        for cat_id_str in cat_ids:
            if cat_id_str is None:
                continue
            deepscores_class = int(cat_id_str)

            # 映射到 Harmony 類別
            harmony_class = get_harmony_class_id(deepscores_class)
            if harmony_class == -1:
                continue  # 跳過未映射的類別

            # YOLO 格式: class x_center y_center width height
            yolo_annotations.append(
                f"{harmony_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

    if not yolo_annotations:
        return False, image_id, 0

    # 複製圖片
    dst_img_path = images_dir / filename
    shutil.copy2(src_img_path, dst_img_path)

    # 寫入標註文件
    label_filename = Path(filename).stem + '.txt'
    label_path = labels_dir / label_filename
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))

    return True, image_id, len(yolo_annotations)

def convert_dataset_split(json_path, split_name):
    """
    轉換單個數據集分割（train 或 test）

    Args:
        json_path: DeepScores JSON 路徑
        split_name: 'train' or 'test'
    """
    print(f"\n{'='*60}")
    print(f"開始轉換 {split_name} 集")
    print(f"{'='*60}")

    # 建立輸出目錄
    split_dir = OUTPUT_DIR / split_name
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 讀取 JSON
    print(f"讀取 {json_path.name}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 獲取標註字典（DeepScoresV2 格式已經是 dict）
    print("讀取標註字典...")
    annotations_dict = data.get('annotations', {})

    images = data['images']
    print(f"總圖片數: {len(images)}")
    print(f"總標註數: {len(annotations_dict)}")

    # 準備多進程參數
    args_list = [
        (img, annotations_dict, images_dir, labels_dir)
        for img in images
    ]

    # 多進程並行轉換
    print(f"使用 {NUM_WORKERS} 個進程並行轉換...")
    successful = 0
    total_annotations = 0

    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_image, args_list),
            total=len(images),
            desc=f"轉換 {split_name}",
            unit="img"
        ))

    # 統計結果
    for success, img_id, num_ann in results:
        if success:
            successful += 1
            total_annotations += num_ann

    print(f"\n{split_name} 集轉換完成:")
    print(f"  成功: {successful} / {len(images)} 圖片")
    print(f"  總標註: {total_annotations}")
    print(f"  平均每圖: {total_annotations / successful:.1f} 個標註" if successful > 0 else "  平均每圖: 0")

    return successful, total_annotations

def split_train_val(train_ratio=0.85):
    """
    將 train 集分割為 train 和 val

    Args:
        train_ratio: 訓練集佔比（剩下的為驗證集）
    """
    print(f"\n{'='*60}")
    print(f"分割 train 為 train/val ({train_ratio:.0%} / {1-train_ratio:.0%})")
    print(f"{'='*60}")

    train_images_dir = OUTPUT_DIR / 'train' / 'images'
    train_labels_dir = OUTPUT_DIR / 'train' / 'labels'

    val_images_dir = OUTPUT_DIR / 'val' / 'images'
    val_labels_dir = OUTPUT_DIR / 'val' / 'labels'
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # 獲取所有圖片
    image_files = list(train_images_dir.glob('*.png'))
    np.random.seed(42)  # 固定隨機種子
    np.random.shuffle(image_files)

    # 計算分割點
    split_idx = int(len(image_files) * train_ratio)
    val_files = image_files[split_idx:]

    print(f"移動 {len(val_files)} 張圖片到 val 集...")

    # 移動到 val
    for img_path in tqdm(val_files, desc="移動到 val"):
        label_path = train_labels_dir / (img_path.stem + '.txt')

        # 移動圖片和標註
        shutil.move(str(img_path), str(val_images_dir / img_path.name))
        if label_path.exists():
            shutil.move(str(label_path), str(val_labels_dir / label_path.name))

    print(f"分割完成:")
    print(f"  Train: {len(list(train_images_dir.glob('*.png')))} 張")
    print(f"  Val:   {len(list(val_images_dir.glob('*.png')))} 張")

def create_yaml_config():
    """生成 YOLO 訓練配置 YAML"""
    yaml_content = f"""# ============================================
# YOLO12 四部和聲資料集配置 (DeepScoresV2)
# ============================================

# 資料集根路徑（相對於此 yaml 檔案）
path: {OUTPUT_DIR.absolute()}

# 訓練/驗證/測試集路徑
train: train/images
val: val/images
test: test/images

# 類別數量
nc: 20

# 類別名稱（索引 0-19）
names:
  0: notehead_filled
  1: notehead_hollow
  2: stem_up
  3: stem_down
  4: beam
  5: flag
  6: clef_treble
  7: clef_bass
  8: clef_alto
  9: clef_tenor
  10: accidental_sharp
  11: accidental_flat
  12: accidental_natural
  13: rest_quarter
  14: rest_half
  15: rest_whole
  16: barline
  17: time_signature
  18: key_signature
  19: staffline
"""

    yaml_path = OUTPUT_DIR / 'harmony_deepscores.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n生成配置文件: {yaml_path}")

def main():
    """主函數"""
    print(f"\n{'='*60}")
    print(f"DeepScoresV2 → YOLO 並行轉換工具")
    print(f"CPU 核心: {NUM_WORKERS}")
    print(f"{'='*60}")

    # 清空輸出目錄
    if OUTPUT_DIR.exists():
        print(f"清空現有輸出目錄: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 轉換 train 和 test 集
    train_json = DEEPSCORES_DIR / 'deepscores_train.json'
    test_json = DEEPSCORES_DIR / 'deepscores_test.json'

    train_success, train_annotations = convert_dataset_split(train_json, 'train')
    test_success, test_annotations = convert_dataset_split(test_json, 'test')

    # 分割 train 為 train/val
    split_train_val(train_ratio=0.85)

    # 生成 YAML 配置
    create_yaml_config()

    print(f"\n{'='*60}")
    print(f"轉換完成！")
    print(f"{'='*60}")
    print(f"訓練集: {OUTPUT_DIR / 'train'}")
    print(f"驗證集: {OUTPUT_DIR / 'val'}")
    print(f"測試集: {OUTPUT_DIR / 'test'}")
    print(f"配置文件: {OUTPUT_DIR / 'harmony_deepscores.yaml'}")

    print(f"\n統計:")
    final_train_count = len(list((OUTPUT_DIR / 'train' / 'images').glob('*.png')))
    final_val_count = len(list((OUTPUT_DIR / 'val' / 'images').glob('*.png')))
    print(f"  Train: {final_train_count} 張")
    print(f"  Val:   {final_val_count} 張")
    print(f"  Test:  {test_success} 張")
    print(f"  Total: {final_train_count + final_val_count + test_success} 張")

if __name__ == "__main__":
    main()
