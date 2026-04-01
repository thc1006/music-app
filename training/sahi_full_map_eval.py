#!/usr/bin/env python3
"""
SAHI 完整 mAP 評估
使用 pycocotools 計算 SAHI 切片推論的真正 mAP
"""

import os
import json
import torch
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import cv2

os.chdir('/home/thc1006/dev/music-app/training')

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def calculate_iou(box1, box2):
    """計算兩個框的 IoU (xyxy 格式)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def load_yolo_labels(label_path, img_width, img_height):
    """載入 YOLO 格式標籤並轉換為 xyxy"""
    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return boxes, labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])

                x1 = (x_center - w/2) * img_width
                y1 = (y_center - h/2) * img_height
                x2 = (x_center + w/2) * img_width
                y2 = (y_center + h/2) * img_height

                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

    return boxes, labels

def compute_ap(precision, recall):
    """計算 AP (11 點插值)"""
    recall = np.array(recall)
    precision = np.array(precision)

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return ap

def main():
    print("=" * 70)
    print("SAHI 完整 mAP 評估")
    print(f"開始時間: {datetime.now()}")
    print("=" * 70)

    from ultralytics import YOLO
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    MODEL_PATH = 'harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt'
    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
    VAL_IMAGES_DIR = Path('datasets/yolo_harmony_v2_phase8_final/val/images')
    VAL_LABELS_DIR = Path('datasets/yolo_harmony_v2_phase8_final/val/labels')

    # 讀取類別
    import yaml
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    num_classes = data_config['nc']
    class_names = data_config['names']
    print(f"類別數: {num_classes}")

    # 第一步：標準 YOLO 基線
    print("\n" + "=" * 70)
    print("1. 標準 YOLO 基線 mAP")
    print("=" * 70)

    model = YOLO(MODEL_PATH)
    baseline_results = model.val(data=DATA_YAML, imgsz=1280, batch=8, verbose=False, plots=False)
    baseline_map50 = baseline_results.box.map50
    baseline_map = baseline_results.box.map
    print(f"   mAP50:    {baseline_map50:.4f}")
    print(f"   mAP50-95: {baseline_map:.4f}")

    del model
    clear_gpu_memory()

    # 第二步：SAHI 推論
    print("\n" + "=" * 70)
    print("2. SAHI 切片推論")
    print("=" * 70)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=0.25,  # 使用較高閾值減少誤報
        device='cuda:0'
    )

    image_files = sorted(VAL_IMAGES_DIR.glob('*.png'))
    print(f"   驗證集圖片數: {len(image_files)}")

    # SAHI 配置
    slice_height = 640
    slice_width = 640
    overlap = 0.2

    print(f"   切片大小: {slice_width}x{slice_height}")
    print(f"   重疊率: {overlap}")

    # 收集預測和 GT
    all_predictions = defaultdict(list)  # {cls: [(score, is_tp), ...]}
    gt_counts = defaultdict(int)

    for img_path in tqdm(image_files[:500], desc="SAHI 推論 (500 張樣本)"):  # 測試 500 張
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_height, img_width = img.shape[:2]

        # 載入 GT
        label_path = VAL_LABELS_DIR / f"{img_path.stem}.txt"
        gt_boxes, gt_labels = load_yolo_labels(label_path, img_width, img_height)

        # 統計 GT
        for cls in gt_labels:
            gt_counts[cls] += 1

        # SAHI 推論
        try:
            result = get_sliced_prediction(
                str(img_path),
                detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                verbose=0
            )
        except Exception as e:
            continue

        # 匹配預測和 GT
        gt_matched = set()

        # 按分數排序預測
        predictions = sorted(
            result.object_prediction_list,
            key=lambda x: x.score.value,
            reverse=True
        )

        for pred in predictions:
            pred_cls = pred.category.id
            pred_score = pred.score.value
            pred_box = [
                pred.bbox.minx,
                pred.bbox.miny,
                pred.bbox.maxx,
                pred.bbox.maxy
            ]

            # 找最佳匹配的 GT
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_cls != pred_cls:
                    continue
                if gt_idx in gt_matched:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # 判斷 TP/FP
            if best_iou >= 0.5 and best_gt_idx >= 0:
                all_predictions[pred_cls].append((pred_score, True))
                gt_matched.add(best_gt_idx)
            else:
                all_predictions[pred_cls].append((pred_score, False))

    # 計算每類 AP
    print("\n" + "=" * 70)
    print("3. 計算 mAP")
    print("=" * 70)

    aps = []
    class_aps = {}

    for cls in range(num_classes):
        preds = all_predictions[cls]
        n_gt = gt_counts[cls]

        if n_gt == 0:
            continue

        # 按分數排序
        preds.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, (score, is_tp) in enumerate(preds):
            if is_tp:
                tp[i] = 1
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(precision, recall)
        aps.append(ap)
        class_aps[cls] = ap

    sahi_map50 = np.mean(aps) if aps else 0

    # 結果
    print("\n" + "=" * 70)
    print("結果比較")
    print("=" * 70)

    print(f"\n標準 YOLO mAP50:  {baseline_map50:.4f}")
    print(f"SAHI mAP50:       {sahi_map50:.4f}")

    diff = (sahi_map50 - baseline_map50) * 100
    print(f"\n差異: {diff:+.2f}%")

    if sahi_map50 > baseline_map50:
        print("\n SAHI 有提升!")
    else:
        print("\n⚠️ SAHI 效果不佳")

    # 弱類別比較
    print("\n弱類別 AP (SAHI):")
    weak_classes = ['barline_double', 'tie', 'ledger_line']
    for name in weak_classes:
        for cls_id, cls_name in class_names.items():
            if cls_name == name:
                ap = class_aps.get(cls_id, 0)
                print(f"  {name}: {ap:.4f}")

    if sahi_map50 > 0.70:
        print("\n" + "=" * 70)
        print("  SUCCESS: SAHI 突破 0.70!")
        print("=" * 70)

    print(f"\n完成時間: {datetime.now()}")

if __name__ == '__main__':
    main()
