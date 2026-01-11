#!/usr/bin/env python3
"""
WBF (Weighted Boxes Fusion) 模型集成測試
集成 Ultimate v5 + Exp1 Hard Sample 模型
"""

import os
import torch
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def normalize_boxes(boxes, img_width, img_height):
    """將 boxes 正規化到 [0,1] 範圍"""
    if len(boxes) == 0:
        return np.array([])
    boxes = np.array(boxes)
    boxes[:, [0, 2]] /= img_width
    boxes[:, [1, 3]] /= img_height
    return boxes.clip(0, 1)

def denormalize_boxes(boxes, img_width, img_height):
    """將 boxes 從 [0,1] 反正規化"""
    if len(boxes) == 0:
        return np.array([])
    boxes = np.array(boxes)
    boxes[:, [0, 2]] *= img_width
    boxes[:, [1, 3]] *= img_height
    return boxes

def main():
    print("=" * 70)
    print("Phase 3: WBF 模型集成測試")
    print("=" * 70)

    # 模型路徑
    models_config = [
        {
            'name': 'Ultimate v5',
            'path': 'harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt',
            'weight': 1.0  # 最佳模型，權重較高
        },
        {
            'name': 'Exp1 Hard Sample',
            'path': 'harmony_omr_v2_experiments/exp1_hard_sample_focus/weights/best.pt',
            'weight': 0.8  # 次要模型
        }
    ]

    # 檢查模型是否存在
    available_models = []
    for cfg in models_config:
        if os.path.exists(cfg['path']):
            available_models.append(cfg)
            print(f"✅ 找到模型: {cfg['name']} ({cfg['path']})")
        else:
            print(f"❌ 未找到模型: {cfg['name']} ({cfg['path']})")

    if len(available_models) < 2:
        print("\n⚠️ 需要至少 2 個模型才能進行集成")
        print("將使用單模型進行基線測試")

        if len(available_models) == 1:
            # 單模型測試
            model = YOLO(available_models[0]['path'])
            results = model.val(
                data='datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml',
                imgsz=1280,
                batch=4,
                verbose=False
            )
            print(f"\n單模型 {available_models[0]['name']}:")
            print(f"  mAP50:    {results.box.map50:.4f}")
            print(f"  mAP50-95: {results.box.map:.4f}")
        return

    # 載入模型
    print("\n載入模型...")
    models = []
    for cfg in available_models:
        model = YOLO(cfg['path'])
        models.append({
            'model': model,
            'name': cfg['name'],
            'weight': cfg['weight']
        })
        print(f"  ✅ {cfg['name']} 已載入")

    # 測試圖片
    VAL_IMAGES_DIR = Path('datasets/yolo_harmony_v2_phase8_final/val/images')
    image_files = sorted(VAL_IMAGES_DIR.glob('*.png'))[:50]  # 測試前 50 張

    print(f"\n測試圖片數: {len(image_files)}")

    # WBF 參數
    iou_thr = 0.5
    skip_box_thr = 0.001

    # 集成推論
    total_boxes_single = 0
    total_boxes_ensemble = 0

    for img_path in tqdm(image_files, desc="WBF 集成推論"):
        # 獲取圖片尺寸
        import cv2
        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]

        # 收集所有模型的預測
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []

        for m in models:
            results = m['model'].predict(str(img_path), imgsz=1280, verbose=False)[0]

            if len(results.boxes) > 0:
                # 獲取預測框 (xyxy 格式)
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                labels = results.boxes.cls.cpu().numpy().astype(int)

                # 正規化 boxes
                boxes_norm = normalize_boxes(boxes, img_width, img_height)

                boxes_list.append(boxes_norm.tolist())
                scores_list.append(scores.tolist())
                labels_list.append(labels.tolist())
                weights.append(m['weight'])

                total_boxes_single += len(boxes)
            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                weights.append(m['weight'])

        # WBF 集成
        if any(len(b) > 0 for b in boxes_list):
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr
            )
            total_boxes_ensemble += len(fused_boxes)

    # 結果統計
    print("\n" + "=" * 70)
    print("WBF 集成結果")
    print("=" * 70)
    avg_single = total_boxes_single / len(image_files) / len(models)
    avg_ensemble = total_boxes_ensemble / len(image_files)

    print(f"單模型平均檢測數: {avg_single:.1f}/張")
    print(f"WBF 集成檢測數:   {avg_ensemble:.1f}/張")
    print(f"變化: {(avg_ensemble/avg_single - 1)*100:+.1f}%")

    print("\n" + "=" * 70)
    print("結論:")
    print("=" * 70)
    print("WBF 通過融合多模型預測，可以:")
    print("  1. 減少單模型的誤報 (透過多模型一致性)")
    print("  2. 提高召回率 (多模型互補)")
    print("  3. 提升整體 mAP (需完整評估)")
    print("\n建議: 使用 SAHI + WBF 組合進行最終推論")
    print("=" * 70)

if __name__ == '__main__':
    main()
