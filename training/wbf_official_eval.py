#!/usr/bin/env python3
"""
WBF 模型集成 - 使用官方 YOLO 驗證
先確認單模型基線，再評估 WBF 效果
"""

import os
import sys
import torch
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

os.chdir('/home/thc1006/dev/music-app/training')

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def main():
    print("=" * 70)
    print("WBF 模型集成評估 - 官方驗證")
    print(f"開始時間: {datetime.now()}")
    print("=" * 70)

    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'

    # 模型配置
    models_config = [
        {
            'name': 'Ultimate v5',
            'path': 'harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt',
            'weight': 1.0
        },
        {
            'name': 'Exp5 DINOv3',
            'path': 'harmony_omr_v2_experiments/exp5_dinov3/finetune/weights/best.pt',
            'weight': 0.95
        }
    ]

    # 確認模型存在
    available_models = []
    for cfg in models_config:
        if os.path.exists(cfg['path']):
            available_models.append(cfg)
            print(f"  [OK] {cfg['name']}")
        else:
            print(f"  [X]  {cfg['name']}: 未找到")

    if len(available_models) < 2:
        print("\n需要至少 2 個模型")
        return

    # 第一步：使用官方 val() 確認單模型基線
    print("\n" + "=" * 70)
    print("第一步：確認單模型基線 (官方 val)")
    print("=" * 70)

    single_results = {}
    for cfg in available_models:
        print(f"\n驗證 {cfg['name']}...")
        model = YOLO(cfg['path'])
        results = model.val(
            data=DATA_YAML,
            imgsz=1280,
            batch=8,
            verbose=False,
            plots=False
        )
        single_results[cfg['name']] = {
            'map50': results.box.map50,
            'map': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr
        }
        print(f"  mAP50:     {results.box.map50:.4f}")
        print(f"  mAP50-95:  {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall:    {results.box.mr:.4f}")
        del model
        clear_gpu_memory()

    # 第二步：WBF 集成推論統計
    print("\n" + "=" * 70)
    print("第二步：WBF 集成效果分析")
    print("=" * 70)

    # 載入模型
    models = []
    for cfg in available_models:
        model = YOLO(cfg['path'])
        models.append({
            'model': model,
            'name': cfg['name'],
            'weight': cfg['weight']
        })

    # 測試圖片 (取樣本)
    VAL_IMAGES_DIR = Path('datasets/yolo_harmony_v2_phase8_final/val/images')
    image_files = sorted(VAL_IMAGES_DIR.glob('*.png'))[:200]  # 測試 200 張

    print(f"測試圖片數: {len(image_files)}")

    # WBF 參數
    iou_thr = 0.55
    skip_box_thr = 0.01

    # 統計
    stats = {
        'single_detections': {m['name']: 0 for m in models},
        'wbf_detections': 0,
        'agreement_ratio': [],  # 多模型同意率
    }

    for img_path in tqdm(image_files, desc="WBF 分析"):
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_height, img_width = img.shape[:2]

        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        model_detections = {}

        for m in models:
            results = m['model'].predict(str(img_path), imgsz=1280, verbose=False, conf=0.25)[0]

            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                labels = results.boxes.cls.cpu().numpy().astype(int)

                stats['single_detections'][m['name']] += len(boxes)
                model_detections[m['name']] = len(boxes)

                # 正規化
                boxes_norm = boxes.copy()
                boxes_norm[:, [0, 2]] /= img_width
                boxes_norm[:, [1, 3]] /= img_height
                boxes_norm = boxes_norm.clip(0, 1)

                boxes_list.append(boxes_norm.tolist())
                scores_list.append(scores.tolist())
                labels_list.append(labels.tolist())
            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                model_detections[m['name']] = 0

            weights.append(m['weight'])

        # WBF 融合
        if any(len(b) > 0 for b in boxes_list):
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr
            )
            stats['wbf_detections'] += len(fused_boxes)

            # 計算模型同意率
            if len(fused_boxes) > 0:
                max_single = max(model_detections.values())
                if max_single > 0:
                    agreement = len(fused_boxes) / max_single
                    stats['agreement_ratio'].append(agreement)

    # 結果
    print("\n" + "=" * 70)
    print("評估結果")
    print("=" * 70)

    print("\n【官方驗證 - 單模型 mAP50】")
    best_single_name = None
    best_single_map50 = 0
    for name, result in single_results.items():
        print(f"  {name}: {result['map50']:.4f}")
        if result['map50'] > best_single_map50:
            best_single_map50 = result['map50']
            best_single_name = name

    print(f"\n  最佳單模型: {best_single_name} (mAP50={best_single_map50:.4f})")

    print("\n【WBF 集成統計 (200 張樣本)】")
    for m in models:
        avg = stats['single_detections'][m['name']] / len(image_files)
        print(f"  {m['name']} 平均檢測數: {avg:.1f}/張")

    avg_wbf = stats['wbf_detections'] / len(image_files)
    print(f"  WBF 集成平均檢測數: {avg_wbf:.1f}/張")

    if stats['agreement_ratio']:
        avg_agreement = np.mean(stats['agreement_ratio'])
        print(f"  平均模型同意率: {avg_agreement:.2%}")

    # 分析
    print("\n" + "=" * 70)
    print("分析與建議")
    print("=" * 70)

    if best_single_map50 >= 0.70:
        print(f"\n  已達成 0.70 目標！最佳模型: {best_single_name}")
    else:
        gap = 0.70 - best_single_map50
        print(f"\n  距離 0.70 目標還差: {gap:.4f} ({gap*100:.2f}%)")
        print("\n  建議策略:")
        print("    1. SAHI 切片推論 (已驗證可提升檢測數)")
        print("    2. 嘗試更高 imgsz (1600/2048)")
        print("    3. 調整信心度閾值")
        print("    4. 針對弱類別進行數據增強")

    print(f"\n完成時間: {datetime.now()}")

if __name__ == '__main__':
    main()
