#!/usr/bin/env python3
"""
WBF (Weighted Boxes Fusion) mAP 評估
集成 Ultimate v5 + Exp5 DINOv3 模型並計算 mAP

目標: 突破 mAP50 > 0.70
"""

import os
import sys
import json
import torch
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# 確保在正確目錄
os.chdir('/home/thc1006/dev/music-app/training')

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def normalize_boxes(boxes, img_width, img_height):
    """將 boxes 正規化到 [0,1] 範圍"""
    if len(boxes) == 0:
        return np.array([]).reshape(0, 4)
    boxes = np.array(boxes).copy()
    boxes[:, [0, 2]] /= img_width
    boxes[:, [1, 3]] /= img_height
    return boxes.clip(0, 1)

def denormalize_boxes(boxes, img_width, img_height):
    """將 boxes 從 [0,1] 反正規化"""
    if len(boxes) == 0:
        return np.array([]).reshape(0, 4)
    boxes = np.array(boxes).copy()
    boxes[:, [0, 2]] *= img_width
    boxes[:, [1, 3]] *= img_height
    return boxes

def load_ground_truth(label_path, img_width, img_height):
    """載入 YOLO 格式的 ground truth"""
    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return np.array([]).reshape(0, 4), np.array([])

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])

                # 轉換為 xyxy 格式
                x1 = (x_center - w/2) * img_width
                y1 = (y_center - h/2) * img_height
                x2 = (x_center + w/2) * img_width
                y2 = (y_center + h/2) * img_height

                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

    return np.array(boxes) if boxes else np.array([]).reshape(0, 4), np.array(labels)

def calculate_iou(box1, box2):
    """計算兩個框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def calculate_ap(precision, recall):
    """計算 AP (使用 11 點插值法)"""
    recall = np.array(recall)
    precision = np.array(precision)

    # 添加起始和結束點
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # 確保 precision 是遞減的
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # 找出 recall 變化的點
    indices = np.where(recall[1:] != recall[:-1])[0] + 1

    # 計算 AP
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return ap

def evaluate_predictions(all_predictions, all_ground_truths, num_classes, iou_threshold=0.5):
    """計算 mAP"""
    aps = []

    for cls in range(num_classes):
        # 收集該類別的所有預測和 GT
        cls_predictions = []
        cls_gt_count = 0

        for img_id in all_predictions:
            preds = all_predictions[img_id]
            gts = all_ground_truths[img_id]

            # 過濾該類別的預測
            for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
                if label == cls:
                    cls_predictions.append({
                        'img_id': img_id,
                        'box': box,
                        'score': score
                    })

            # 計算該類別的 GT 數量
            cls_gt_count += np.sum(gts['labels'] == cls)

        if cls_gt_count == 0:
            continue

        # 按分數排序
        cls_predictions.sort(key=lambda x: x['score'], reverse=True)

        # 計算 TP/FP
        tp = np.zeros(len(cls_predictions))
        fp = np.zeros(len(cls_predictions))
        gt_matched = defaultdict(set)

        for i, pred in enumerate(cls_predictions):
            img_id = pred['img_id']
            pred_box = pred['box']

            gts = all_ground_truths[img_id]
            gt_boxes = gts['boxes'][gts['labels'] == cls]

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx not in gt_matched[img_id]:
                tp[i] = 1
                gt_matched[img_id].add(best_gt_idx)
            else:
                fp[i] = 1

        # 計算累積 TP/FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # 計算 precision 和 recall
        recall = tp_cumsum / cls_gt_count
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # 計算 AP
        ap = calculate_ap(precision, recall)
        aps.append(ap)

    return np.mean(aps) if aps else 0

def main():
    print("=" * 70)
    print("WBF 模型集成 mAP 評估")
    print(f"開始時間: {datetime.now()}")
    print("=" * 70)

    # 模型配置 - 使用最佳的兩個模型
    models_config = [
        {
            'name': 'Ultimate v5',
            'path': 'harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt',
            'weight': 1.0,
            'baseline_map50': 0.6979
        },
        {
            'name': 'Exp5 DINOv3',
            'path': 'harmony_omr_v2_experiments/exp5_dinov3/finetune/weights/best.pt',
            'weight': 0.95,
            'baseline_map50': 0.6976
        },
        {
            'name': 'Exp4 Finetune',
            'path': 'harmony_omr_v2_experiments/exp4_finetune/weights/best.pt',
            'weight': 0.9,
            'baseline_map50': 0.6959
        }
    ]

    # 檢查模型
    available_models = []
    for cfg in models_config:
        if os.path.exists(cfg['path']):
            available_models.append(cfg)
            print(f"  [OK] {cfg['name']}: {cfg['path']}")
            print(f"       基線 mAP50: {cfg['baseline_map50']}")
        else:
            print(f"  [X]  {cfg['name']}: 未找到")

    if len(available_models) < 2:
        print("\n需要至少 2 個模型進行集成")
        return

    # 載入模型
    print(f"\n載入 {len(available_models)} 個模型...")
    models = []
    for cfg in available_models:
        print(f"  載入 {cfg['name']}...", end=' ')
        model = YOLO(cfg['path'])
        models.append({
            'model': model,
            'name': cfg['name'],
            'weight': cfg['weight']
        })
        print("OK")
        clear_gpu_memory()

    # 數據路徑
    VAL_IMAGES_DIR = Path('datasets/yolo_harmony_v2_phase8_final/val/images')
    VAL_LABELS_DIR = Path('datasets/yolo_harmony_v2_phase8_final/val/labels')

    image_files = sorted(VAL_IMAGES_DIR.glob('*.png'))
    print(f"\n驗證集圖片數: {len(image_files)}")

    # 獲取類別數
    yaml_path = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
    import yaml
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    num_classes = data_config['nc']
    print(f"類別數: {num_classes}")

    # WBF 參數
    iou_thr = 0.55  # WBF IoU 閾值
    skip_box_thr = 0.01  # 跳過低分數框

    print(f"\nWBF 參數:")
    print(f"  IoU 閾值: {iou_thr}")
    print(f"  分數閾值: {skip_box_thr}")
    print(f"  模型權重: {[m['weight'] for m in models]}")

    # 收集所有預測和 GT
    all_predictions = {}
    all_ground_truths = {}
    single_model_predictions = {m['name']: {} for m in models}

    print(f"\n開始推論...")

    for img_path in tqdm(image_files, desc="WBF 集成"):
        img_id = img_path.stem

        # 讀取圖片尺寸
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_height, img_width = img.shape[:2]

        # 載入 GT
        label_path = VAL_LABELS_DIR / f"{img_id}.txt"
        gt_boxes, gt_labels = load_ground_truth(label_path, img_width, img_height)
        all_ground_truths[img_id] = {
            'boxes': gt_boxes,
            'labels': gt_labels
        }

        # 收集所有模型預測
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []

        for m in models:
            results = m['model'].predict(str(img_path), imgsz=1280, verbose=False, conf=0.01)[0]

            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                labels = results.boxes.cls.cpu().numpy().astype(int)

                # 保存單模型預測
                single_model_predictions[m['name']][img_id] = {
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                }

                # 正規化用於 WBF
                boxes_norm = normalize_boxes(boxes, img_width, img_height)
                boxes_list.append(boxes_norm.tolist())
                scores_list.append(scores.tolist())
                labels_list.append(labels.tolist())
            else:
                single_model_predictions[m['name']][img_id] = {
                    'boxes': np.array([]).reshape(0, 4),
                    'scores': np.array([]),
                    'labels': np.array([])
                }
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])

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

            # 反正規化
            fused_boxes = denormalize_boxes(np.array(fused_boxes), img_width, img_height)

            all_predictions[img_id] = {
                'boxes': fused_boxes,
                'scores': np.array(fused_scores),
                'labels': np.array(fused_labels).astype(int)
            }
        else:
            all_predictions[img_id] = {
                'boxes': np.array([]).reshape(0, 4),
                'scores': np.array([]),
                'labels': np.array([])
            }

    # 計算 mAP
    print("\n計算 mAP...")

    # WBF 集成 mAP
    wbf_map50 = evaluate_predictions(all_predictions, all_ground_truths, num_classes, iou_threshold=0.5)

    # 單模型 mAP
    single_maps = {}
    for m in models:
        single_preds = {}
        for img_id in all_ground_truths:
            if img_id in single_model_predictions[m['name']]:
                single_preds[img_id] = single_model_predictions[m['name']][img_id]
            else:
                single_preds[img_id] = {
                    'boxes': np.array([]).reshape(0, 4),
                    'scores': np.array([]),
                    'labels': np.array([])
                }
        single_maps[m['name']] = evaluate_predictions(single_preds, all_ground_truths, num_classes, iou_threshold=0.5)

    # 結果
    print("\n" + "=" * 70)
    print("評估結果")
    print("=" * 70)

    # 建立基線對照
    baselines = {
        'Ultimate v5': 0.6979,
        'Exp5 DINOv3': 0.6976,
        'Exp4 Finetune': 0.6959
    }

    print("\n單模型 mAP50:")
    for m in models:
        baseline = baselines.get(m['name'], 0)
        diff = (single_maps[m['name']] - baseline) * 100
        print(f"  {m['name']}: {single_maps[m['name']]:.4f} (基線: {baseline}, 差異: {diff:+.2f}%)")

    print(f"\nWBF 集成 mAP50: {wbf_map50:.4f}")

    best_single = max(single_maps.values())
    improvement = (wbf_map50 - best_single) * 100
    print(f"\n與最佳單模型比較: {improvement:+.2f}%")

    baseline = 0.6979
    vs_baseline = (wbf_map50 - baseline) * 100
    print(f"與基線 (0.6979) 比較: {vs_baseline:+.2f}%")

    if wbf_map50 > 0.70:
        print("\n" + "=" * 70)
        print("  SUCCESS: 突破 0.70 目標!")
        print("=" * 70)
    else:
        gap = 0.70 - wbf_map50
        print(f"\n距離 0.70 目標還差: {gap:.4f} ({gap*100:.2f}%)")

    print(f"\n完成時間: {datetime.now()}")

    return wbf_map50

if __name__ == '__main__':
    main()
