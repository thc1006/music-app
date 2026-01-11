#!/usr/bin/env python3
"""
Phase 4: 弱類別微調訓練
基於 Ultimate v5 最佳模型進行微調
目標: 改善弱類別 (barline_double, tie, ledger_line) 的檢測性能
策略: 使用更高的 cls loss 權重 + 更長的訓練
"""

import os
import torch
import gc
from pathlib import Path
from datetime import datetime

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def main():
    print("=" * 70)
    print("Phase 4: 弱類別微調訓練")
    print("=" * 70)

    from ultralytics import YOLO

    # 配置
    BASE_MODEL = 'harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt'
    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
    OUTPUT_DIR = 'harmony_omr_v2_experiments/exp4_finetune'

    # 確認模型存在
    if not os.path.exists(BASE_MODEL):
        print(f"錯誤: 找不到基礎模型 {BASE_MODEL}")
        return

    print(f"\n基礎模型: {BASE_MODEL}")
    print(f"數據集: {DATA_YAML}")
    print(f"輸出目錄: {OUTPUT_DIR}")

    # 創建輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 載入模型
    print("\n載入模型...")
    model = YOLO(BASE_MODEL)

    clear_gpu_memory()

    # 微調訓練配置 - 穩定版 (無 OOM)
    # 使用與 Ultimate v5 相同的配置確保穩定性
    train_config = {
        'data': DATA_YAML,
        'epochs': 30,           # 微調 epochs
        'imgsz': 1280,          # 保持高解析度
        'batch': 4,             # 與 Ultimate v5 相同，確保無 OOM
        'device': 0,
        'workers': 8,
        'patience': 10,         # 早停
        'save': True,
        'save_period': 10,
        'project': 'harmony_omr_v2_experiments',
        'name': 'exp4_finetune',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.00005,         # 微調使用很小的學習率
        'lrf': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 2,
        'cos_lr': True,
        'amp': False,           # 禁用 AMP 避免數值問題
        'nbs': 64,              # 名義 batch size (梯度累積模擬)

        # Loss 權重調整
        'cls': 1.0,             # 提高分類損失權重 (默認 0.5)
        'box': 7.5,             # 保持框回歸權重

        # 數據增強 (保守設定，避免破壞已學特徵)
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 0.0,         # 不旋轉
        'translate': 0.05,
        'scale': 0.2,
        'flipud': 0.0,          # 不上下翻轉
        'fliplr': 0.0,          # 不左右翻轉
        'mosaic': 0.3,
        'mixup': 0.05,

        'verbose': True,
    }

    print("\n訓練配置:")
    print(f"  Cls Loss 權重: {train_config['cls']}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch size: {train_config['batch']}")
    print(f"  Image size: {train_config['imgsz']}")
    print(f"  Learning rate: {train_config['lr0']}")

    # 開始訓練
    print("\n" + "=" * 70)
    print("開始微調訓練...")
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
            batch=4,
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

    except Exception as e:
        print(f"\n訓練錯誤: {e}")
        raise
    finally:
        clear_gpu_memory()

if __name__ == '__main__':
    main()
