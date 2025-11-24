#!/usr/bin/env python3
"""
Phase 3 Training Script: External Data Integration
===================================================
Continues from Phase 2 best model with comprehensive merged dataset.

Key improvements:
- Merged dataset with ~14K+ images (Phase2 + Fornes + Choi + DoReMi)
- Significantly more samples for rare classes (double_sharp, double_flat, etc.)
- Stable training parameters with early stopping

Model: YOLO12s
Base weights: Phase 2 best.pt
Dataset: yolo_harmony_v2_phase3 (merged external datasets)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure YOLO12 is available
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run:")
    print("  pip install ultralytics")
    sys.exit(1)

# Configuration
BASE_DIR = Path("/home/thc1006/dev/music-app/training")
PHASE2_WEIGHTS = BASE_DIR / "harmony_omr_v2_phase2" / "balanced_training" / "weights" / "best.pt"
DATASET_YAML = BASE_DIR / "datasets" / "yolo_harmony_v2_phase3" / "harmony_phase3.yaml"
OUTPUT_DIR = BASE_DIR / "harmony_omr_v2_phase3"

# Training hyperparameters
TRAINING_CONFIG = {
    # Basic settings
    "epochs": 150,
    "batch": 16,  # Conservative for stability
    "imgsz": 640,
    "device": 0,  # GPU 0

    # Learning rate
    "lr0": 0.001,  # Lower initial LR since continuing from Phase 2
    "lrf": 0.01,   # Final LR factor
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Warmup
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # Early stopping
    "patience": 30,  # Stop if no improvement for 30 epochs

    # Data augmentation (moderate - dataset is already diverse)
    "mosaic": 0.5,
    "mixup": 0.0,
    "copy_paste": 0.0,  # Disabled - not suitable for music notation
    "degrees": 0.0,     # No rotation - music notation is orientation-sensitive
    "translate": 0.1,
    "scale": 0.3,
    "shear": 0.0,
    "flipud": 0.0,      # No vertical flip
    "fliplr": 0.0,      # No horizontal flip
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.4,

    # Loss weights
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    # Other
    "workers": 8,
    "cache": True,
    "amp": True,  # Mixed precision for speed
    "val": True,
    "plots": True,
    "save": True,
    "save_period": 10,  # Save checkpoint every 10 epochs

    # Output
    "project": str(OUTPUT_DIR),
    "name": "external_data_training",
    "exist_ok": True,
}


def check_prerequisites():
    """Check that all prerequisites are met before training."""
    print("="*60)
    print("PHASE 3 TRAINING: External Data Integration")
    print("="*60)

    errors = []

    # Check Phase 2 weights
    print(f"\n[1] Checking Phase 2 weights...")
    if PHASE2_WEIGHTS.exists():
        size_mb = PHASE2_WEIGHTS.stat().st_size / (1024 * 1024)
        print(f"    Found: {PHASE2_WEIGHTS}")
        print(f"    Size: {size_mb:.1f} MB")
    else:
        errors.append(f"Phase 2 weights not found: {PHASE2_WEIGHTS}")

    # Check dataset YAML
    print(f"\n[2] Checking dataset configuration...")
    if DATASET_YAML.exists():
        print(f"    Found: {DATASET_YAML}")

        # Verify dataset structure
        dataset_dir = DATASET_YAML.parent
        train_images = dataset_dir / "train" / "images"
        val_images = dataset_dir / "val" / "images"

        if train_images.exists():
            train_count = len(list(train_images.glob("*")))
            print(f"    Train images: {train_count}")
        else:
            errors.append(f"Train images directory not found: {train_images}")

        if val_images.exists():
            val_count = len(list(val_images.glob("*")))
            print(f"    Val images: {val_count}")
        else:
            errors.append(f"Val images directory not found: {val_images}")
    else:
        errors.append(f"Dataset YAML not found: {DATASET_YAML}")
        errors.append("Run merge_datasets_phase3.py first!")

    # Check GPU
    print(f"\n[3] Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"    GPU: {gpu_name}")
            print(f"    Memory: {gpu_memory:.1f} GB")
        else:
            errors.append("No CUDA GPU available")
    except ImportError:
        errors.append("PyTorch not installed")

    # Check output directory
    print(f"\n[4] Output directory...")
    print(f"    {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if errors:
        print("\n" + "="*60)
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        print("="*60)
        return False

    print("\n" + "="*60)
    print("All prerequisites met!")
    print("="*60)
    return True


def print_training_config():
    """Print the training configuration."""
    print("\n" + "-"*60)
    print("TRAINING CONFIGURATION")
    print("-"*60)

    key_params = [
        ("Base weights", str(PHASE2_WEIGHTS.name)),
        ("Dataset", str(DATASET_YAML.name)),
        ("Epochs", TRAINING_CONFIG["epochs"]),
        ("Batch size", TRAINING_CONFIG["batch"]),
        ("Image size", TRAINING_CONFIG["imgsz"]),
        ("Initial LR", TRAINING_CONFIG["lr0"]),
        ("Early stopping patience", TRAINING_CONFIG["patience"]),
        ("Mosaic aug", TRAINING_CONFIG["mosaic"]),
        ("Mixed precision", TRAINING_CONFIG["amp"]),
    ]

    for name, value in key_params:
        print(f"  {name:<25}: {value}")

    print("-"*60)


def train():
    """Run Phase 3 training."""

    # Prerequisites check
    if not check_prerequisites():
        print("\nPlease fix the errors above before training.")
        sys.exit(1)

    # Print config
    print_training_config()

    # Load model from Phase 2 weights
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    print(f"Loading from: {PHASE2_WEIGHTS}")

    model = YOLO(str(PHASE2_WEIGHTS))
    print("Model loaded successfully!")

    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        results = model.train(
            data=str(DATASET_YAML),
            **TRAINING_CONFIG
        )

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Print results summary
        print("\nResults Summary:")
        print(f"  Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

        # Model location
        best_model = OUTPUT_DIR / "external_data_training" / "weights" / "best.pt"
        if best_model.exists():
            print(f"\nBest model saved to:")
            print(f"  {best_model}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results may be available in:")
        print(f"  {OUTPUT_DIR}")

    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    train()
