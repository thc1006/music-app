#!/usr/bin/env python3
"""
🚀 YOLO12 RTX 5090 Optimized Training - 33 Classes
Phase 1 優化版：解決 OOM、mAP 震盪問題
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

# 🚀 Phase 1 優化配置
OPTIMIZED_CONFIG = {
    # 基礎設定
    'epochs': 300,          # 減少 epochs，專注穩定
    'batch': 16,            # ⭐ 降低 batch 避免 OOM (72% VRAM)
    'imgsz': 640,
    'patience': 50,         # 更早停止

    # ⭐ 關鍵優化：降低學習率
    'lr0': 0.005,           # 從 0.01 降到 0.005
    'lrf': 0.01,            # 最終學習率比例
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5.0,   # 更長 warmup
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.05,
    'cos_lr': True,

    # ⭐ 減少資料增強（提高穩定性）
    'degrees': 3.0,         # 從 5.0 降到 3.0
    'translate': 0.05,      # 從 0.1 降到 0.05
    'scale': 0.5,           # 從 0.9 降到 0.5
    'shear': 1.0,           # 從 2.0 降到 1.0
    'perspective': 0.0,     # 關閉透視變換
    'hsv_h': 0.01,
    'hsv_s': 0.3,
    'hsv_v': 0.3,
    'mosaic': 0.5,          # ⭐ 從 1.0 降到 0.5
    'mixup': 0.0,
    'copy_paste': 0.0,      # ⭐ 關閉 copy_paste
    'flipud': 0.0,
    'fliplr': 0.0,
    'erasing': 0.2,         # 從 0.4 降到 0.2
    'close_mosaic': 20,     # 更早關閉 mosaic

    # 硬體優化
    'device': 0,
    'workers': 8,
    'amp': True,            # 保持 AMP
    'cache': False,
    'multi_scale': False,
    'rect': False,

    # 損失權重
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,

    # 輸出配置
    'project': 'harmony_omr_v2_optimized',
    'name': 'train_phase1',
    'save_period': 10,      # 每 10 epochs 保存
    'plots': True,
    'verbose': True,
    'seed': 42,
    'resume': False,
    'exist_ok': False,
    'pretrained': True,

    # 驗證設定
    'val': True,
    'save': True,
}

def main():
    print("\n" + "=" * 70)
    print(" " * 10 + "🚀 YOLO12 Optimized Training - Phase 1")
    print("=" * 70)

    # GPU 檢查
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n🎮 GPU: {gpu_name}")
    print(f"💾 VRAM: {vram:.2f} GB")

    # 載入模型
    print(f"\n📥 Loading: yolo12s.pt")
    model = YOLO('yolo12s.pt')

    # 配置摘要
    print("\n⚙️  Optimized Configuration:")
    print(f"   Classes: 33 (優化版，排除問題類別)")
    print(f"   Batch: {OPTIMIZED_CONFIG['batch']} (穩定優先)")
    print(f"   Epochs: {OPTIMIZED_CONFIG['epochs']}")
    print(f"   LR: {OPTIMIZED_CONFIG['lr0']} → {OPTIMIZED_CONFIG['lr0'] * OPTIMIZED_CONFIG['lrf']}")
    print(f"   Mosaic: {OPTIMIZED_CONFIG['mosaic']} (50%)")
    print(f"   Warmup: {OPTIMIZED_CONFIG['warmup_epochs']} epochs")

    print("\n🔧 主要優化：")
    print("   1. Batch 16 避免 CUDA OOM")
    print("   2. LR 0.005 減少震盪")
    print("   3. 減少資料增強提高穩定性")
    print("   4. 驗證集從 205→273 張")
    print("   5. 移除問題類別 (stem_down, slur)")

    # 開始訓練
    print("\n" + "=" * 70)
    print(" " * 15 + "🔥 Starting Optimized Training")
    print("=" * 70 + "\n")

    # 使用優化後的數據集
    data_yaml = 'datasets/yolo_harmony_v2_optimized/harmony_optimized.yaml'

    if not Path(data_yaml).exists():
        print(f"❌ 找不到優化數據集配置: {data_yaml}")
        print("請先執行: python optimize_dataset_phase1.py")
        sys.exit(1)

    # 訓練模型
    results = model.train(
        data=data_yaml,
        **OPTIMIZED_CONFIG
    )

    print("\n" + "=" * 70)
    print(" " * 20 + "✅ Training Completed!")
    print("=" * 70)

    print(f"\n📊 最終結果:")
    print(f"   Best model: {results.save_dir}/weights/best.pt")
    print(f"   Last model: {results.save_dir}/weights/last.pt")

    return results

if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent)
    main()
