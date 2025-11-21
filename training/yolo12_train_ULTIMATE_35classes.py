#!/usr/bin/env python3
"""
🏆 YOLO12 RTX 5090 Ultimate Training - 35 Classes
終極完整方案：batch=28，極致穩定
"""

import torch
import sys
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# PyTorch 優化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(24)

# 🏆 終極配置 (35 類，batch=28)
ULTIMATE_CONFIG = {
    'epochs': 600,
    'batch': 24,  # ⭐ 穩定性優先（避免 TaskAlignedAssigner OOM）
    'imgsz': 640,
    'patience': 100,

    # 學習率
    'lr0': 0.01,
    'lrf': 0.01,
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'cos_lr': True,

    # YOLO12 資料增強
    'degrees': 5.0,
    'translate': 0.1,
    'scale': 0.9,
    'shear': 2.0,
    'perspective': 0.0001,
    'hsv_h': 0.015,
    'hsv_s': 0.5,
    'hsv_v': 0.4,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.1,
    'flipud': 0.0,
    'fliplr': 0.0,
    'close_mosaic': 10,

    # 硬體優化
    'device': 0,
    'workers': 8,
    'amp': True,
    'cache': False,
    'multi_scale': False,
    'rect': False,

    # 輸出配置
    'project': 'harmony_omr_v2_ultimate',
    'save_period': 20,
    'plots': True,
    'verbose': True,
    'seed': 42,
    'resume': False,
    'exist_ok': False,
    'pretrained': True,
}

def main():
    print("\n" + "=" * 70)
    print(" " * 10 + "🏆 YOLO12 Ultimate Training - 35 Classes")
    print("=" * 70)

    # GPU 檢查
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(1)

    print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 載入模型
    print(f"\n📥 Loading: yolo12s.pt")
    model = YOLO('yolo12s.pt')

    # 配置摘要
    print("\n⚙️  Configuration:")
    print(f"   Classes: 35 (Ultimate)")
    print(f"   Batch: {ULTIMATE_CONFIG['batch']} (避免 OOM)")
    print(f"   Epochs: {ULTIMATE_CONFIG['epochs']}")
    print(f"   Workers: {ULTIMATE_CONFIG['workers']}")
    print(f"   AMP: {ULTIMATE_CONFIG['amp']}")

    # 開始訓練
    print("\n" + "=" * 70)
    print(" " * 25 + "🔥 Training Started")
    print("=" * 70)

    results = model.train(
        data='datasets/yolo_harmony_v2_35classes/harmony_deepscores_v2.yaml',
        **ULTIMATE_CONFIG,
    )

    print("\n" + "=" * 70)
    print(" " * 20 + "✅ Training Completed Successfully")
    print("=" * 70)

if __name__ == "__main__":
    main()
