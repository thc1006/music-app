#!/usr/bin/env python3
"""
Exp6 YOLO11m 恢復訓練 - RTX 5090 優化版
充分利用 32GB VRAM + 24 核 i9-14900

優化策略：
1. batch=6 (從 3 提升，配合梯度累積達到有效 batch=24)
2. workers=16 (利用 24 核 CPU)
3. 啟用 AMP FP16 混合精度 (RTX 5090 的 Tensor Core 優化)
4. cache='ram' 利用 125GB 系統記憶體
"""

import os
import gc
import torch
from datetime import datetime

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def main():
    print("=" * 70)
    print("Exp6 YOLO11m 恢復訓練 - RTX 5090 優化版")
    print("=" * 70)
    print(f"恢復時間: {datetime.now()}")

    # 顯示硬體資訊
    print(f"\n硬體配置:")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CPU 核心: {os.cpu_count()}")

    from ultralytics import YOLO

    # 從 checkpoint 恢復
    CHECKPOINT = 'harmony_omr_v2_experiments/exp6_yolo11m/weights/last.pt'
    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'

    print(f"\n從 checkpoint 恢復: {CHECKPOINT}")

    # 載入模型
    model = YOLO(CHECKPOINT)
    clear_gpu_memory()

    # RTX 5090 優化配置
    # 32GB VRAM 可以支援 batch=6 @ 1280x1280
    # 有效 batch = 6 * (64/6) ≈ 64 (通過梯度累積)
    train_config = {
        'data': DATA_YAML,
        'epochs': 100,
        'imgsz': 1280,
        'batch': 6,              # 從 3 提升到 6 (RTX 5090 32GB)
        'device': 0,
        'workers': 16,           # 利用 24 核 i9-14900
        'cache': 'ram',          # 利用 125GB 系統記憶體加速數據載入
        'patience': 20,          # 更長耐心讓模型收斂
        'save': True,
        'save_period': 5,        # 更頻繁保存
        'resume': True,          # 關鍵：從 checkpoint 恢復
        'exist_ok': True,

        # 優化器配置
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,      # 恢復訓練減少 warmup
        'cos_lr': True,
        'nbs': 64,               # 梯度累積目標 batch size

        # 混合精度 - RTX 5090 Blackwell 架構優化
        # 注意：如果出現 inf loss，改為 False
        'amp': True,

        # 數據增強（保持原設定）
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
        'plots': True,
    }

    print("\n優化配置 (RTX 5090 專用):")
    print(f"  Batch size: {train_config['batch']} (原 3 → 6, 2x 提升)")
    print(f"  Workers: {train_config['workers']} (原 8 → 16)")
    print(f"  Cache: {train_config['cache']} (利用 125GB RAM)")
    print(f"  AMP: {train_config['amp']} (FP16 混合精度)")
    print(f"  有效 batch: ~{train_config['nbs']} (梯度累積)")

    # 開始訓練
    print("\n" + "=" * 70)
    print("開始恢復訓練 (從 epoch 3)...")
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
            batch=6,
            verbose=True
        )

        print(f"\n===== 最終結果 =====")
        print(f"  mAP50:     {val_results.box.map50:.4f}")
        print(f"  mAP50-95:  {val_results.box.map:.4f}")
        print(f"  Precision: {val_results.box.mp:.4f}")
        print(f"  Recall:    {val_results.box.mr:.4f}")

        # 與基線比較
        baseline_map50 = 0.6979
        improvement = (val_results.box.map50 - baseline_map50) * 100
        print(f"\n與 Ultimate v5 基線比較:")
        print(f"  基線 mAP50: {baseline_map50:.4f}")
        print(f"  變化: {improvement:+.2f}%")

        if val_results.box.map50 > 0.70:
            print("\n🎉🎉🎉 突破 0.70 目標！🎉🎉🎉")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n⚠️ OOM 錯誤！請降低 batch size")
            print(f"建議：將 batch 從 6 降到 4")
        raise
    except Exception as e:
        print(f"\n訓練錯誤: {e}")
        raise
    finally:
        clear_gpu_memory()

if __name__ == '__main__':
    main()
