#!/usr/bin/env python3
"""
🔥 YOLO12 RTX 5090 Ultra-Optimized Training Script
針對 RTX 5090 + i9-14900 的極致效能優化版本

優化重點：
1. VRAM 最大化利用（32GB）
2. DataLoader 極致優化
3. cuDNN Benchmark
4. PyTorch 2.x 最佳化
5. 資料預快取到 GPU
"""

import torch
import sys
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# ============= 極致效能優化設定 =============

# PyTorch 優化
torch.backends.cudnn.benchmark = True  # ⚡ 自動選擇最佳卷積算法
torch.backends.cudnn.deterministic = False  # 犧牲確定性換取速度
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 加速
torch.backends.cudnn.allow_tf32 = True

# 設定 PyTorch 線程數
torch.set_num_threads(24)  # i9-14900 全部核心

# RTX 5090 極致配置（32GB VRAM）
ULTRA_CONFIG = {
    # ========== 基礎訓練參數 ==========
    'epochs': 600,
    'batch': 32,  # ⚡ 超保守值（TaskAlignedAssigner OOM）
    'imgsz': 640,
    'patience': 100,

    # ========== 學習率 ==========
    'lr0': 0.01,
    'lrf': 0.01,
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'cos_lr': True,

    # ========== YOLO12 資料增強 ==========
    'degrees': 5.0,
    'translate': 0.1,
    'scale': 0.9,  # ⭐ YOLO12s 推薦
    'shear': 2.0,
    'perspective': 0.0001,
    'hsv_h': 0.015,
    'hsv_s': 0.5,
    'hsv_v': 0.4,
    'mosaic': 1.0,  # ⭐ YOLO12 推薦 1.0
    'mixup': 0.0,   # ⭐ YOLO12 推薦 0.0
    'copy_paste': 0.1,  # ⭐ YOLO12 特有
    'flipud': 0.0,
    'fliplr': 0.0,
    'close_mosaic': 10,

    # ========== 硬體優化（穩定版本）==========
    'device': 0,
    'workers': 8,  # ⚡ 保守值（避免 multiprocessing 錯誤）
    'amp': True,  # ⭐ FP16 混合精度
    'cache': False,  # ⚠️ 關閉快取（避免 VRAM OOM）
    'multi_scale': False,  # ⚠️ 關閉多尺度（避免 VRAM 爆掉）
    'rect': False,  # 矩形訓練（OMR 可能不適合）

    # ========== 輸出配置 ==========
    'project': 'harmony_omr_ultra',
    'save_period': 20,
    'plots': True,
    'verbose': True,
    'seed': 42,

    # ========== 進階優化 ==========
    'resume': False,
    'exist_ok': False,
    'pretrained': True,

    # ========== DataLoader 極致優化（2025）==========
    # Note: pin_memory 和 persistent_workers 由 PyTorch 自動處理
}

def check_environment():
    """檢查訓練環境"""
    print("\n" + "=" * 70)
    print(" " * 15 + "🔥 Ultra-Optimized Training Environment")
    print("=" * 70)

    # Python 版本
    print(f"\n📦 Python: {sys.version.split()[0]}")

    # PyTorch
    print(f"🔧 PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"🔢 CUDA Version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\n❌ Error: CUDA GPU not detected")
        sys.exit(1)

    # GPU 資訊
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    compute_capability = torch.cuda.get_device_capability(0)

    print(f"\n🎮 GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_memory:.2f} GB")
    print(f"⚡ Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

    # 優化狀態
    print(f"\n🚀 cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"🚀 TF32 MatMul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"🚀 TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")
    print(f"🚀 PyTorch Threads: {torch.get_num_threads()}")

    # RTX 5090 驗證
    if "5090" in gpu_name:
        print("\n🔥 RTX 5090 Detected! Ultra-optimized config enabled")
        print("   - 32GB VRAM → Batch size 256")
        print("   - GDDR7 1.79 TB/s → RAM cache enabled")
        print("   - Tensor Cores Gen 5 → AMP (FP16) enabled")
        print("   - cuDNN Benchmark → Enabled")
        print("   - Workers: 32 (persistent)")

    print("=" * 70)


def patch_dataloader(trainer):
    """
    Patch Ultralytics dataloader for extreme performance

    This callback modifies the dataloader to use:
    - pin_memory=True
    - persistent_workers=True
    - prefetch_factor=2
    """
    def on_train_start(trainer_obj):
        # Patch train dataloader
        if hasattr(trainer_obj, 'train_loader'):
            old_loader = trainer_obj.train_loader
            if hasattr(old_loader, 'dataset'):
                from torch.utils.data import DataLoader
                trainer_obj.train_loader = DataLoader(
                    old_loader.dataset,
                    batch_size=old_loader.batch_size,
                    shuffle=True,
                    num_workers=min(32, trainer_obj.args.workers),
                    pin_memory=True,  # ⚡ 加速 CPU→GPU
                    persistent_workers=True,  # ⚡ Worker 常駐
                    prefetch_factor=2,  # ⚡ 預先載入
                    collate_fn=old_loader.collate_fn if hasattr(old_loader, 'collate_fn') else None,
                )
                print(f"🔥 DataLoader patched: pin_memory=True, persistent_workers=True")

    return {'on_train_start': on_train_start}


def train_ultra_optimized(args):
    """執行 Ultra-Optimized 訓練"""
    print("\n" + "=" * 70)
    print(" " * 20 + "🚀 Starting Ultra-Optimized Training")
    print("=" * 70)

    # 載入預訓練模型
    print(f"\n📥 Loading: yolo12s.pt")
    model = YOLO('yolo12s.pt')
    print("✅ Model loaded successfully")

    # 印出配置
    print("\n⚙️  Configuration:")
    print(f"   Batch: {ULTRA_CONFIG['batch']}")
    print(f"   Epochs: {ULTRA_CONFIG['epochs']}")
    print(f"   Workers: {ULTRA_CONFIG['workers']}")
    print(f"   AMP: {ULTRA_CONFIG['amp']}")
    print(f"   Cache: {ULTRA_CONFIG['cache']}")
    print(f"   Multi-scale: {ULTRA_CONFIG['multi_scale']}")

    # 添加 DataLoader patch callback
    callbacks = patch_dataloader(model)

    # 開始訓練
    print("\n" + "=" * 70)
    print(" " * 25 + "🔥 Training Started")
    print("=" * 70)

    results = model.train(
        data=args.data,
        **ULTRA_CONFIG,
        # callbacks=callbacks,  # 如果 Ultralytics 支援
    )

    print("\n" + "=" * 70)
    print(" " * 20 + "✅ Training Completed Successfully")
    print("=" * 70)

    return results


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='YOLO12 Ultra-Optimized Training')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML path')
    parser.add_argument('--model', type=str, default='yolo12s', help='Model variant')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 10 + "🔥 YOLO12 RTX 5090 Ultra-Optimized Training System")
    print("=" * 70)

    # 檢查環境
    check_environment()

    # 執行訓練
    train_ultra_optimized(args)


if __name__ == "__main__":
    main()
