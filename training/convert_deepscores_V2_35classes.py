#!/usr/bin/env python3
"""
DeepScoresV2 → YOLO 極致並行版本
真正使用全部 24 核心！

優化策略：
1. 預先載入所有 JSON 到記憶體（避免重複讀取）
2. 使用 chunksize 批次處理
3. 減少進程間通訊開銷
4. 使用 ProcessPoolExecutor 替代 Pool
"""

import json
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from deepscores_to_harmony_mapping_v2_35classes import get_harmony_class_id

# 配置
DEEPSCORES_DIR = Path("datasets/ds2_dense")
OUTPUT_DIR = Path("datasets/yolo_harmony")
NUM_WORKERS = 24
CHUNK_SIZE = 10  # 每個 worker 一次處理 10 張圖

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """bbox 轉換（已修復邊界問題）"""
    x, y, w, h = bbox

    # 確保 bbox 在圖片範圍內
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    w = max(1, min(w, img_width - x))
    h = max(1, min(h, img_height - y))

    # 歸一化
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    # 最終裁切
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height

def process_image_batch(args):
    """
    批次處理多張圖片（減少進程啟動開銷）

    Args:
        args: (image_batch, annotations_dict, images_dir, labels_dir)

    Returns:
        [(success, image_id, num_annotations), ...]
    """
    image_batch, annotations_dict, images_dir, labels_dir = args
    results = []

    for image_data in image_batch:
        image_id = image_data['id']
        filename = image_data['filename']
        img_width = image_data['width']
        img_height = image_data['height']

        # 檢查圖片
        src_img_path = DEEPSCORES_DIR / "images" / filename
        if not src_img_path.exists():
            results.append((False, image_id, 0))
            continue

        # 獲取標註
        ann_ids = image_data.get('ann_ids', [])
        if not ann_ids:
            results.append((False, image_id, 0))
            continue

        # 轉換標註
        yolo_annotations = []
        for ann_id in ann_ids:
            if ann_id not in annotations_dict:
                continue

            annotation = annotations_dict[ann_id]
            cat_ids = annotation.get('cat_id', [])
            if not cat_ids:
                continue

            bbox = annotation['a_bbox']
            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox, img_width, img_height
            )

            for cat_id_str in cat_ids:
                if cat_id_str is None:
                    continue
                harmony_class = get_harmony_class_id(int(cat_id_str))
                if harmony_class == -1:
                    continue

                yolo_annotations.append(
                    f"{harmony_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

        if not yolo_annotations:
            results.append((False, image_id, 0))
            continue

        # 複製圖片
        dst_img_path = images_dir / filename
        shutil.copy2(src_img_path, dst_img_path)

        # 寫入標註
        label_filename = Path(filename).stem + '.txt'
        label_path = labels_dir / label_filename
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

        results.append((True, image_id, len(yolo_annotations)))

    return results

def convert_dataset_split(json_path, split_name):
    """轉換資料集（極致並行版本）"""
    print(f"\n{'='*60}")
    print(f"🔥 極致並行轉換 {split_name} 集")
    print(f"{'='*60}")

    # 建立目錄
    split_dir = OUTPUT_DIR / split_name
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 一次性讀取 JSON（避免每個 worker 重複讀取）
    print(f"📥 讀取 {json_path.name}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations_dict = data.get('annotations', {})
    images = data['images']

    print(f"  總圖片: {len(images)}")
    print(f"  總標註: {len(annotations_dict)}")

    # 將圖片分成批次（每批 CHUNK_SIZE 張）
    batches = []
    for i in range(0, len(images), CHUNK_SIZE):
        batch = images[i:i + CHUNK_SIZE]
        batches.append((batch, annotations_dict, images_dir, labels_dir))

    print(f"  批次數: {len(batches)} (每批 {CHUNK_SIZE} 張)")
    print(f"  Workers: {NUM_WORKERS}")

    # 使用 ProcessPoolExecutor 極致並行
    successful = 0
    total_annotations = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任務
        futures = {executor.submit(process_image_batch, batch): batch
                  for batch in batches}

        # 顯示進度
        with tqdm(total=len(images), desc=f"轉換 {split_name}", unit="img") as pbar:
            for future in as_completed(futures):
                batch_results = future.result()
                for success, img_id, num_ann in batch_results:
                    if success:
                        successful += 1
                        total_annotations += num_ann
                    pbar.update(1)

    print(f"\n✅ {split_name} 集轉換完成:")
    print(f"  成功: {successful} / {len(images)}")
    print(f"  標註: {total_annotations}")

    return successful, total_annotations

def split_train_val(train_ratio=0.85):
    """分割 train/val"""
    print(f"\n{'='*60}")
    print(f"分割 train/val")
    print(f"{'='*60}")

    train_images_dir = OUTPUT_DIR / 'train' / 'images'
    train_labels_dir = OUTPUT_DIR / 'train' / 'labels'
    val_images_dir = OUTPUT_DIR / 'val' / 'images'
    val_labels_dir = OUTPUT_DIR / 'val' / 'labels'
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(train_images_dir.glob('*.png'))
    np.random.seed(42)
    np.random.shuffle(image_files)

    split_idx = int(len(image_files) * train_ratio)
    val_files = image_files[split_idx:]

    for img_path in tqdm(val_files, desc="移動到 val"):
        label_path = train_labels_dir / (img_path.stem + '.txt')
        shutil.move(str(img_path), str(val_images_dir / img_path.name))
        if label_path.exists():
            shutil.move(str(label_path), str(val_labels_dir / label_path.name))

    print(f"✅ 分割完成")

def create_yaml_config():
    """生成 YAML 配置"""
    yaml_content = f"""# YOLO12 四部和聲資料集配置
path: {OUTPUT_DIR.absolute()}
train: train/images
val: val/images
test: test/images
nc: 20
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

    print(f"\n✅ 配置: {yaml_path}")

def main():
    """主函數"""
    print(f"\n{'='*60}")
    print(f"🔥 DeepScoresV2 → YOLO 極致並行轉換")
    print(f"  CPU 核心: {NUM_WORKERS}")
    print(f"  批次大小: {CHUNK_SIZE}")
    print(f"{'='*60}")

    # 清空輸出
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # 轉換
    train_json = DEEPSCORES_DIR / 'deepscores_train.json'
    test_json = DEEPSCORES_DIR / 'deepscores_test.json'

    convert_dataset_split(train_json, 'train')
    convert_dataset_split(test_json, 'test')

    # 分割
    split_train_val(train_ratio=0.85)

    # 生成配置
    create_yaml_config()

    print(f"\n{'='*60}")
    print(f"✅ 全部完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
