#!/usr/bin/env python3
"""
Phase 5: DINOv3 知識蒸餾 (修復版)
基於 Ultimate v5 最佳模型進行 DINOv3 蒸餾

修復:
1. 使用 Ultimate v5 作為基礎模型 (mAP50=0.6979)
2. 禁用 AMP 避免 cls_loss=inf 問題
3. 適當的 batch size 和 epochs 設定
"""

import os
import sys
import gc
import torch
from pathlib import Path
from datetime import datetime

# 配置
BASE_MODEL = Path('harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt')
DATASET_DIR = Path('datasets/yolo_harmony_v2_phase8_final')
OUTPUT_DIR = Path('harmony_omr_v2_experiments/exp5_dinov3')

# DINOv3 蒸餾配置
DISTILLATION_CONFIG = {
    'method': 'distillation',
    'teacher': 'dinov2_vitl14_reg',  # DINOv2 Large (最大可用教師)
    'epochs': 50,                     # 蒸餾 epochs
    'batch_size': 32,                 # 適合 32GB VRAM
}

# 微調配置
FINETUNE_CONFIG = {
    'epochs': 30,
    'imgsz': 1280,
    'batch': 6,
    'lr0': 0.00005,
    'workers': 8,
    'amp': False,  # 關鍵: 禁用 AMP 避免數值問題
}


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def check_environment():
    """檢查環境是否就緒"""
    print("=" * 70)
    print("Phase 5: DINOv3 蒸餾環境檢查")
    print("=" * 70)

    errors = []

    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    else:
        errors.append("CUDA 不可用")

    # 模型
    if BASE_MODEL.exists():
        print(f"基礎模型: {BASE_MODEL}")
    else:
        errors.append(f"找不到基礎模型: {BASE_MODEL}")

    # 資料集
    if DATASET_DIR.exists():
        train_count = len(list((DATASET_DIR / 'train/images').glob('*')))
        print(f"資料集: {train_count:,} 訓練圖片")
    else:
        errors.append("資料集不存在")

    # LightlyTrain
    try:
        import lightly_train
        print(f"LightlyTrain: {lightly_train.__version__}")
    except ImportError:
        errors.append("lightly-train 未安裝")

    if errors:
        print(f"\n錯誤: {errors}")
        return False

    print("\n環境檢查通過")
    return True


def run_distillation():
    """Step 1: DINOv3 蒸餾預訓練"""
    print("\n" + "=" * 70)
    print("Step 1: DINOv3 蒸餾預訓練")
    print("=" * 70)

    import lightly_train
    from ultralytics import YOLO

    pretrain_out = OUTPUT_DIR / 'pretrain'
    pretrain_out.mkdir(parents=True, exist_ok=True)

    # 載入模型
    print(f"\n載入 Ultimate v5 模型...")
    model = YOLO(str(BASE_MODEL))

    print(f"\n蒸餾配置:")
    print(f"  教師: {DISTILLATION_CONFIG['teacher']}")
    print(f"  學生: Ultimate v5 (mAP50=0.6979)")
    print(f"  Epochs: {DISTILLATION_CONFIG['epochs']}")
    print(f"  Batch: {DISTILLATION_CONFIG['batch_size']}")

    try:
        lightly_train.pretrain(
            out=str(pretrain_out),
            data=str(DATASET_DIR / 'train/images'),
            model=model,
            method=DISTILLATION_CONFIG['method'],
            method_args={
                'teacher': DISTILLATION_CONFIG['teacher'],
            },
            epochs=DISTILLATION_CONFIG['epochs'],
            batch_size=DISTILLATION_CONFIG['batch_size'],
            num_workers=FINETUNE_CONFIG['workers'],
            overwrite=True,
        )
        print("\n蒸餾預訓練完成!")
        return pretrain_out

    except Exception as e:
        print(f"\n蒸餾失敗: {e}")
        print("將直接使用 Ultimate v5 進行微調...")
        return None


def run_finetuning(pretrain_dir=None):
    """Step 2: YOLO 微調"""
    print("\n" + "=" * 70)
    print("Step 2: YOLO 微調")
    print("=" * 70)

    from ultralytics import YOLO

    # 選擇模型
    if pretrain_dir:
        exported = pretrain_dir / 'exported_models/exported_last.pt'
        if exported.exists():
            model_path = str(exported)
            print(f"使用蒸餾後模型: {model_path}")
        else:
            model_path = str(BASE_MODEL)
            print(f"蒸餾模型不存在，使用 Ultimate v5")
    else:
        model_path = str(BASE_MODEL)
        print(f"使用 Ultimate v5 模型")

    model = YOLO(model_path)

    finetune_out = OUTPUT_DIR / 'finetune'

    # 訓練配置
    train_config = {
        'data': str(DATASET_DIR / 'harmony_phase8_final.yaml'),
        'epochs': FINETUNE_CONFIG['epochs'],
        'imgsz': FINETUNE_CONFIG['imgsz'],
        'batch': FINETUNE_CONFIG['batch'],
        'device': 0,
        'workers': FINETUNE_CONFIG['workers'],
        'patience': 10,
        'save': True,
        'project': str(OUTPUT_DIR),
        'name': 'finetune',
        'exist_ok': True,
        'optimizer': 'AdamW',
        'lr0': FINETUNE_CONFIG['lr0'],
        'lrf': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 2,
        'cos_lr': True,
        'amp': FINETUNE_CONFIG['amp'],  # 關閉 AMP

        # 保守的數據增強
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 0.0,
        'translate': 0.05,
        'scale': 0.2,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.3,
        'mixup': 0.05,

        'verbose': True,
    }

    print(f"\n微調配置:")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Image size: {train_config['imgsz']}")
    print(f"  Batch: {train_config['batch']}")
    print(f"  AMP: {train_config['amp']}")

    start_time = datetime.now()

    try:
        results = model.train(**train_config)

        duration = datetime.now() - start_time
        print(f"\n微調完成! 耗時: {duration}")

        # 驗證
        val_results = model.val(
            data=str(DATASET_DIR / 'harmony_phase8_final.yaml'),
            imgsz=1280,
            batch=4,
            verbose=False
        )

        print(f"\n最終結果:")
        print(f"  mAP50:    {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")

        baseline = 0.6979
        improvement = (val_results.box.map50 - baseline) * 100
        print(f"\n與 Ultimate v5 比較: {improvement:+.2f}%")

        return val_results

    except Exception as e:
        print(f"\n微調失敗: {e}")
        raise


def main():
    print("=" * 70)
    print("Phase 5: DINOv3 知識蒸餾 (修復版)")
    print("=" * 70)
    print(f"開始時間: {datetime.now()}")

    # 環境檢查
    if not check_environment():
        return

    clear_gpu_memory()

    # 創建輸出目錄
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: 蒸餾
    pretrain_dir = run_distillation()

    clear_gpu_memory()

    # Step 2: 微調
    results = run_finetuning(pretrain_dir)

    print("\n" + "=" * 70)
    print("Phase 5 完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
