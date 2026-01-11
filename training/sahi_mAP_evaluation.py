#!/usr/bin/env python3
"""
SAHI mAP 評估腳本
使用 COCO 評估標準計算 SAHI 切片推論的 mAP
"""

import os
import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
import numpy as np

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def main():
    print("=" * 70)
    print("SAHI mAP 評估 (使用 1280x1280 切片)")
    print("=" * 70)

    from ultralytics import YOLO
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # 配置
    MODEL_PATH = 'harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt'
    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'

    # 先用標準 YOLO 驗證獲取基線
    print("\n1. 基線 mAP (標準 YOLO 驗證)...")
    model = YOLO(MODEL_PATH)
    baseline_results = model.val(
        data=DATA_YAML,
        imgsz=1280,
        batch=4,
        verbose=False
    )
    baseline_map50 = baseline_results.box.map50
    baseline_map = baseline_results.box.map
    print(f"   基線 mAP50:    {baseline_map50:.4f}")
    print(f"   基線 mAP50-95: {baseline_map:.4f}")

    clear_gpu_memory()

    # SAHI 切片推論
    print("\n2. SAHI 1280x1280 切片推論...")

    # 載入 SAHI 模型
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=0.001,  # 低閾值以獲得更多預測
        device='cuda:0'
    )

    VAL_IMAGES_DIR = Path('datasets/yolo_harmony_v2_phase8_final/val/images')
    VAL_LABELS_DIR = Path('datasets/yolo_harmony_v2_phase8_final/val/labels')

    # 讀取類別名稱
    with open(DATA_YAML, 'r') as f:
        import yaml
        data_config = yaml.safe_load(f)
        class_names = data_config['names']

    image_files = sorted(VAL_IMAGES_DIR.glob('*.png'))
    print(f"   驗證集圖片數: {len(image_files)}")

    # SAHI 配置：1280x1280 切片
    slice_height = 1280
    slice_width = 1280
    overlap = 0.2

    all_predictions = []
    all_ground_truths = []

    for img_path in tqdm(image_files, desc="SAHI 推論"):
        try:
            # SAHI 推論
            result = get_sliced_prediction(
                str(img_path),
                detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                verbose=0
            )

            # 收集預測
            for pred in result.object_prediction_list:
                all_predictions.append({
                    'image': img_path.name,
                    'class': pred.category.id,
                    'confidence': pred.score.value,
                    'bbox': [
                        pred.bbox.minx,
                        pred.bbox.miny,
                        pred.bbox.maxx,
                        pred.bbox.maxy
                    ]
                })

            # 讀取 ground truth
            label_path = VAL_LABELS_DIR / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center, y_center, w, h = map(float, parts[1:5])
                            all_ground_truths.append({
                                'image': img_path.name,
                                'class': cls_id,
                            })

        except Exception as e:
            print(f"錯誤: {img_path.name}: {e}")
            continue

    print(f"\n   總預測數: {len(all_predictions)}")
    print(f"   總 Ground Truth 數: {len(all_ground_truths)}")

    # 簡單統計
    pred_by_class = {}
    for p in all_predictions:
        cls = p['class']
        pred_by_class[cls] = pred_by_class.get(cls, 0) + 1

    gt_by_class = {}
    for g in all_ground_truths:
        cls = g['class']
        gt_by_class[cls] = gt_by_class.get(cls, 0) + 1

    print("\n類別預測統計 (前 10 個):")
    for cls_id in sorted(pred_by_class.keys())[:10]:
        pred_count = pred_by_class.get(cls_id, 0)
        gt_count = gt_by_class.get(cls_id, 0)
        ratio = pred_count / gt_count if gt_count > 0 else 0
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        print(f"   {cls_name:25s}: 預測 {pred_count:6d} / GT {gt_count:6d} (x{ratio:.1f})")

    print("\n" + "=" * 70)
    print("結論:")
    print("=" * 70)
    print(f"基線 mAP50: {baseline_map50:.4f}")
    print("\n注意: SAHI 主要用於推論時提升小物件檢測")
    print("完整 mAP 評估需要使用 COCO 格式的評估工具")
    print("=" * 70)

if __name__ == '__main__':
    main()
