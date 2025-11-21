#!/usr/bin/env python3
"""
🔧 YOLO12 RTX 5090 Fixed Training - 35 Classes
修正版訓練腳本：解決類別不平衡和訓練震盪問題
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

# 🔧 修正配置（解決訓練震盪）
FIXED_CONFIG = {
    'epochs': 300,  # 減少 epochs，專注穩定性
    'batch': 16,    # 降低 batch size 提高穩定性
    'imgsz': 640,
    'patience': 50,  # 更早停止

    # 🔧 關鍵修正：降低學習率
    'lr0': 0.005,   # ⭐ 從 0.01 降到 0.005
    'lrf': 0.001,   # ⭐ 最終學習率也降低
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5.0,  # 更長的 warmup
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.05,  # 降低 warmup bias lr
    'cos_lr': True,

    # 🔧 減少資料增強（提高穩定性）
    'degrees': 3.0,      # 從 5.0 降到 3.0
    'translate': 0.05,   # 從 0.1 降到 0.05
    'scale': 0.5,        # 從 0.9 降到 0.5（減少縮放變化）
    'shear': 1.0,        # 從 2.0 降到 1.0
    'perspective': 0.0,  # 關閉透視變換
    'hsv_h': 0.01,      # 從 0.015 降到 0.01
    'hsv_s': 0.3,       # 從 0.5 降到 0.3
    'hsv_v': 0.3,       # 從 0.4 降到 0.3
    'mosaic': 0.5,      # ⭐ 從 1.0 降到 0.5（50% 機率）
    'mixup': 0.0,       # 保持關閉
    'copy_paste': 0.0,  # ⭐ 關閉 copy_paste（太激進）
    'flipud': 0.0,
    'fliplr': 0.0,
    'close_mosaic': 20,  # 更早關閉 mosaic

    # 硬體優化
    'device': 0,
    'workers': 8,
    'amp': True,
    'cache': False,
    'multi_scale': False,
    'rect': False,

    # 輸出配置
    'project': 'harmony_omr_v2_fixed',
    'name': 'train_stable',
    'save_period': 10,
    'plots': True,
    'verbose': True,
    'seed': 42,
    'resume': False,
    'exist_ok': False,
    'pretrained': True,

    # 🔧 新增：處理類別不平衡
    'box': 7.5,
    'cls': 1.0,  # 可以考慮增加到 2.0 強化分類
    'dfl': 1.5,
}

def main():
    print("\n" + "=" * 70)
    print(" " * 10 + "🔧 YOLO12 Fixed Training - Stable Version")
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
    print("\n⚙️  Fixed Configuration:")
    print(f"   Classes: 35")
    print(f"   Batch: {FIXED_CONFIG['batch']} (穩定優先)")
    print(f"   Epochs: {FIXED_CONFIG['epochs']}")
    print(f"   LR: {FIXED_CONFIG['lr0']} → {FIXED_CONFIG['lrf']} (降低)")
    print(f"   Mosaic: {FIXED_CONFIG['mosaic']} (50%)")
    print(f"   Copy-Paste: Disabled")

    # 開始訓練
    print("\n" + "=" * 70)
    print(" " * 15 + "🚀 Starting Stable Training")
    print("=" * 70 + "\n")

    # 訓練模型
    results = model.train(
        data='datasets/yolo_harmony_v2_35classes/harmony_deepscores_v2.yaml',
        **FIXED_CONFIG
    )

    print("\n" + "=" * 70)
    print(" " * 20 + "✅ Training Completed!")
    print("=" * 70)

    # 顯示最終結果
    print(f"\nBest model saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()