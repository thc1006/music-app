#!/usr/bin/env python3
"""
Phase 6: YOLO11m 升級訓練
從 YOLO11s 升級到 YOLO11m (更大容量模型)

目標: 利用更大模型容量改善弱類別檢測
- YOLO11s: ~9M 參數
- YOLO11m: ~20M 參數 (2x+ 容量)

策略: 從 YOLO11m 預訓練權重開始，完整訓練
"""

import os
import gc
import torch
from pathlib import Path
from datetime import datetime

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def main():
    print("=" * 70)
    print("Phase 6: YOLO11m 升級訓練")
    print("=" * 70)
    print(f"開始時間: {datetime.now()}")

    from ultralytics import YOLO

    # 配置
    BASE_MODEL = 'yolo11m.pt'  # YOLO11m 預訓練權重
    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
    OUTPUT_DIR = 'harmony_omr_v2_experiments'
    NAME = 'exp6_yolo11m'

    print(f"\n基礎模型: {BASE_MODEL}")
    print(f"數據集: {DATA_YAML}")

    # 載入模型
    print("\n載入 YOLO11m 預訓練模型...")
    model = YOLO(BASE_MODEL)

    # 模型信息
    print(f"模型類型: YOLO11m (Medium)")

    clear_gpu_memory()

    # 訓練配置
    # YOLO11m 比 YOLO11s 大，需要降低 batch size
    train_config = {
        'data': DATA_YAML,
        'epochs': 100,          # 從頭訓練需要更多 epochs
        'imgsz': 1280,          # 保持高解析度
        'batch': 3,             # YOLO11m 更大，降低 batch
        'device': 0,
        'workers': 8,
        'patience': 15,         # 更長耐心
        'save': True,
        'save_period': 10,
        'project': OUTPUT_DIR,
        'name': NAME,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,           # 從頭訓練用較大學習率
        'lrf': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'cos_lr': True,
        'amp': False,           # 禁用 AMP 避免數值問題
        'nbs': 64,              # 梯度累積

        # 數據增強
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.3,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.5,
        'mixup': 0.1,

        'verbose': True,
    }

    print("\n訓練配置:")
    print(f"  模型: YOLO11m (升級)")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch size: {train_config['batch']}")
    print(f"  Image size: {train_config['imgsz']}")
    print(f"  Learning rate: {train_config['lr0']}")

    # 開始訓練
    print("\n" + "=" * 70)
    print("開始 YOLO11m 訓練...")
    print("=" * 70)

    start_time = datetime.now()

    try:
        results = model.train(**train_config)

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("訓練完成!")
        print("=" * 70)
        print(f"總時間: {duration}")

        # 驗證結果
        print("\n執行最終驗證...")
        val_results = model.val(
            data=DATA_YAML,
            imgsz=1280,
            batch=3,
            verbose=False
        )

        print(f"\n最終結果:")
        print(f"  mAP50:    {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")
        print(f"  Precision: {val_results.box.mp:.4f}")
        print(f"  Recall:    {val_results.box.mr:.4f}")

        # 與基線比較
        baseline_map50 = 0.6979
        improvement = (val_results.box.map50 - baseline_map50) * 100
        print(f"\n與 Ultimate v5 基線比較:")
        print(f"  基線 mAP50: {baseline_map50:.4f}")
        print(f"  變化: {improvement:+.2f}%")

        # 弱類別檢查
        print(f"\n弱類別 mAP50:")
        weak_classes = {
            'barline_double': 24,
            'tie': 8,
            'ledger_line': 9,
        }
        for name, idx in weak_classes.items():
            try:
                ap50 = val_results.box.ap50[idx]
                print(f"  {name}: {ap50:.4f}")
            except:
                pass

    except Exception as e:
        print(f"\n訓練錯誤: {e}")
        raise
    finally:
        clear_gpu_memory()

if __name__ == '__main__':
    main()
