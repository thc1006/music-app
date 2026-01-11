#!/usr/bin/env python3
"""
SAHI 切片推論測試腳本
比較標準推論 vs SAHI 切片推論的效果
"""

import os
import sys
import torch
import gc
from pathlib import Path
from tqdm import tqdm
import numpy as np

def clear_gpu_memory():
    """清理 GPU 記憶體"""
    torch.cuda.empty_cache()
    gc.collect()

def test_sahi_inference():
    print("=" * 70)
    print("Phase 2: SAHI 切片推論測試")
    print("=" * 70)

    # 導入必要模組
    from ultralytics import YOLO
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction, get_prediction
    from sahi.utils.coco import Coco

    # 配置
    MODEL_PATH = 'harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt'
    VAL_IMAGES_DIR = 'datasets/yolo_harmony_v2_phase8_final/val/images'
    VAL_LABELS_DIR = 'datasets/yolo_harmony_v2_phase8_final/val/labels'

    # 獲取驗證集圖片列表 (只測試前 100 張以節省時間)
    image_files = sorted(Path(VAL_IMAGES_DIR).glob('*.png'))[:100]
    print(f"\n測試圖片數: {len(image_files)}")

    # 載入 SAHI 模型
    print("\n載入模型...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',  # SAHI 使用 yolov8 接口兼容 yolo12
        model_path=MODEL_PATH,
        confidence_threshold=0.25,
        device='cuda:0'
    )

    clear_gpu_memory()

    # 測試配置
    slice_configs = [
        {"name": "標準推論 (無切片)", "slice_height": None, "slice_width": None},
        {"name": "SAHI 640x640", "slice_height": 640, "slice_width": 640, "overlap": 0.2},
        {"name": "SAHI 512x512", "slice_height": 512, "slice_width": 512, "overlap": 0.2},
        {"name": "SAHI 320x320 (小物件)", "slice_height": 320, "slice_width": 320, "overlap": 0.3},
    ]

    results = {}

    for config in slice_configs:
        print(f"\n{'=' * 50}")
        print(f"測試: {config['name']}")
        print('=' * 50)

        total_detections = 0

        for img_path in tqdm(image_files, desc=config['name']):
            try:
                if config['slice_height'] is None:
                    # 標準推論
                    result = get_prediction(
                        str(img_path),
                        detection_model,
                        verbose=0
                    )
                else:
                    # SAHI 切片推論
                    result = get_sliced_prediction(
                        str(img_path),
                        detection_model,
                        slice_height=config['slice_height'],
                        slice_width=config['slice_width'],
                        overlap_height_ratio=config.get('overlap', 0.2),
                        overlap_width_ratio=config.get('overlap', 0.2),
                        verbose=0
                    )

                total_detections += len(result.object_prediction_list)

            except Exception as e:
                print(f"錯誤處理 {img_path.name}: {e}")
                continue

        avg_detections = total_detections / len(image_files)
        results[config['name']] = {
            'total': total_detections,
            'avg': avg_detections
        }

        print(f"總檢測數: {total_detections}")
        print(f"平均每張: {avg_detections:.1f}")

        clear_gpu_memory()

    # 結果總結
    print("\n" + "=" * 70)
    print("SAHI 測試結果總結")
    print("=" * 70)

    baseline = results.get("標準推論 (無切片)", {}).get('avg', 0)

    for name, data in results.items():
        diff = data['avg'] - baseline if baseline > 0 else 0
        diff_pct = (diff / baseline * 100) if baseline > 0 else 0
        print(f"{name:25s}: {data['avg']:6.1f} 檢測/張 ({diff_pct:+.1f}%)")

    print("\n" + "=" * 70)
    print("注意: 檢測數增加不一定代表更好，需要結合 mAP 評估")
    print("SAHI 主要優勢在於提高小物件的召回率")
    print("=" * 70)

    return results

if __name__ == '__main__':
    test_sahi_inference()
