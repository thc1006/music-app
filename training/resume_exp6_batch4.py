#!/usr/bin/env python3
"""
Exp6 YOLO11m 恢復訓練 - RTX 5090 穩定版 (batch=4)
避免 OOM，確保訓練穩定

變更:
- batch: 6 → 4 (避免 TaskAlignedAssigner OOM)
- amp: True (保持 FP16 加速)
- 其他配置不變
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
    print("Exp6 YOLO11m 恢復訓練 - batch=4 穩定版")
    print("=" * 70)
    print(f"開始時間: {datetime.now()}")

    # 顯示硬體資訊
    print(f"\n硬體配置:")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CPU 核心: {os.cpu_count()}")

    from ultralytics import YOLO

    # 從原始 YOLO11m 預訓練權重重新開始（不用之前的 checkpoint）
    # 因為之前的訓練有 OOM 問題可能影響了模型
    BASE_MODEL = 'yolo11m.pt'
    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
    OUTPUT_DIR = 'harmony_omr_v2_experiments'
    NAME = 'exp6_yolo11m_batch4'

    print(f"\n從頭訓練: {BASE_MODEL}")
    print(f"輸出目錄: {OUTPUT_DIR}/{NAME}")

    # 載入模型
    model = YOLO(BASE_MODEL)
    clear_gpu_memory()

    # 穩定配置 - batch=4 避免 OOM
    train_config = {
        'data': DATA_YAML,
        'epochs': 100,
        'imgsz': 1280,
        'batch': 4,              # 從 6 降到 4，避免 OOM
        'device': 0,
        'workers': 16,           # 利用 24 核 i9-14900
        'cache': 'ram',          # 利用 125GB 系統記憶體
        'patience': 20,
        'save': True,
        'save_period': 5,
        'project': OUTPUT_DIR,
        'name': NAME,
        'exist_ok': True,

        # 優化器配置
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'cos_lr': True,
        'nbs': 64,               # 梯度累積目標 batch size

        # 混合精度 - RTX 5090 優化
        'amp': True,

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
        'plots': True,
    }

    print("\n穩定配置 (避免 OOM):")
    print(f"  Batch size: {train_config['batch']} (原 6 → 4)")
    print(f"  Workers: {train_config['workers']}")
    print(f"  Cache: {train_config['cache']}")
    print(f"  AMP: {train_config['amp']}")
    print(f"  有效 batch: ~{train_config['nbs']} (梯度累積)")
    print(f"  預估 VRAM: ~24-26 GB (安全範圍)")

    # 開始訓練
    print("\n" + "=" * 70)
    print("開始訓練 (從頭開始，batch=4 穩定版)...")
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
            print(f"\n⚠️ OOM 錯誤！batch=4 仍然不夠")
            print(f"建議：將 batch 降到 3 或禁用 AMP")
        raise
    except Exception as e:
        print(f"\n訓練錯誤: {e}")
        raise
    finally:
        clear_gpu_memory()

if __name__ == '__main__':
    main()
