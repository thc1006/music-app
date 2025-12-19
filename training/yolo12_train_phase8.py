#!/usr/bin/env python3
"""
Phase 8 Training Script - Maximum GPU Utilization
Optimized for RTX 5090 (32GB VRAM)

Key improvements in Phase 8:
- double_flat (Class 17): 651 → 68,538 (+10,428%)
- dynamic_loud (Class 31): 1,309 → 28,688 (+2,091%)
- fermata (Class 29): 11,041 → 31,871 (+189%)
- barline_double (Class 24): 4,038 → 21,623 (+436%)
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================

# Paths
BASE_DIR = Path('/home/thc1006/dev/music-app/training')
DATASET_YAML = BASE_DIR / 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
OUTPUT_DIR = BASE_DIR / 'harmony_omr_v2_phase8'
PRETRAINED_MODEL = BASE_DIR / 'harmony_omr_v2_phase7/stage4_polish/weights/best.pt'

# Training parameters optimized for RTX 5090 (32GB VRAM)
TRAINING_CONFIG = {
    # Model
    'model': str(PRETRAINED_MODEL),

    # Dataset
    'data': str(DATASET_YAML),

    # Training duration
    'epochs': 150,
    'patience': 30,  # Early stopping

    # Batch size - optimized for RTX 5090 (avoiding OOM)
    'batch': 24,  # Reduced from 48 to avoid CUDA OOM errors

    # Image size
    'imgsz': 640,  # Standard size for balance

    # Learning rate
    'lr0': 0.001,  # Lower for fine-tuning
    'lrf': 0.01,   # Final LR = lr0 * lrf

    # Optimizer
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,

    # Warmup
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # Augmentation - moderate for fine-tuning
    'hsv_h': 0.015,
    'hsv_s': 0.5,
    'hsv_v': 0.3,
    'degrees': 5.0,      # Slight rotation
    'translate': 0.1,
    'scale': 0.3,
    'shear': 2.0,
    'perspective': 0.0001,
    'flipud': 0.0,       # No vertical flip for sheet music
    'fliplr': 0.0,       # No horizontal flip for sheet music
    'mosaic': 0.5,       # Reduced mosaic
    'mixup': 0.1,
    'copy_paste': 0.0,   # Disabled for sheet music

    # Loss weights
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,

    # Other
    'workers': 16,       # Maximize CPU workers
    'device': 0,         # GPU 0
    'amp': True,         # Mixed precision
    'cache': 'ram',      # Cache dataset in RAM for speed
    'rect': False,       # Rectangular training
    'close_mosaic': 20,  # Close mosaic last 20 epochs
    'nbs': 64,           # Nominal batch size

    # Output
    'project': str(OUTPUT_DIR),
    'name': 'phase8_training',
    'exist_ok': True,
    'plots': True,
    'save': True,
    'save_period': 10,   # Save checkpoint every 10 epochs
    'verbose': True,
}


def check_gpu():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    free_mem = (torch.cuda.get_device_properties(device).total_memory -
                torch.cuda.memory_allocated(device)) / 1e9

    print(f"GPU: {gpu_name}")
    print(f"Total Memory: {total_mem:.1f} GB")
    print(f"Free Memory: {free_mem:.1f} GB")

    return gpu_name, total_mem


def check_dataset():
    """Verify dataset exists and is valid."""
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")

    print(f"Dataset: {DATASET_YAML}")
    print(f"Pretrained: {PRETRAINED_MODEL}")

    if not PRETRAINED_MODEL.exists():
        print("⚠️ Pretrained model not found, will use fresh YOLO12s")
        return 'yolo12s.pt'

    return str(PRETRAINED_MODEL)


def main():
    print("=" * 70)
    print("Phase 8 Training - Maximum GPU Utilization")
    print("=" * 70)

    # Check GPU
    gpu_name, total_mem = check_gpu()

    # Adjust batch size based on GPU memory (conservative to avoid OOM)
    if total_mem >= 30:  # 32GB VRAM
        TRAINING_CONFIG['batch'] = 24  # Reduced from 48 due to OOM issues
        print(f"Using batch size 24 for {total_mem:.0f}GB VRAM (conservative)")
    elif total_mem >= 20:
        TRAINING_CONFIG['batch'] = 16
        print(f"Using batch size 16 for {total_mem:.0f}GB VRAM")
    else:
        TRAINING_CONFIG['batch'] = 8
        print(f"Using batch size 8 for {total_mem:.0f}GB VRAM")

    # Check dataset
    model_path = check_dataset()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)

    # Initialize model
    model = YOLO(model_path)

    # Start training
    results = model.train(**TRAINING_CONFIG)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    # Print final metrics
    if results:
        print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    # Save final report
    report_path = OUTPUT_DIR / 'phase8_training' / 'training_report.txt'
    with open(report_path, 'w') as f:
        f.write("Phase 8 Training Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Dataset: {DATASET_YAML}\n")
        f.write(f"Pretrained: {model_path}\n")
        f.write(f"Batch Size: {TRAINING_CONFIG['batch']}\n")
        f.write(f"Epochs: {TRAINING_CONFIG['epochs']}\n")
        if results:
            f.write(f"\nFinal Metrics:\n")
            for key, value in results.results_dict.items():
                f.write(f"  {key}: {value}\n")

    print(f"\nReport saved to: {report_path}")
    print(f"Best model: {OUTPUT_DIR}/phase8_training/weights/best.pt")


if __name__ == '__main__':
    main()
