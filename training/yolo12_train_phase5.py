#!/usr/bin/env python3
"""
Phase 5 Training Script: Fermata & Barline Enhancement
=======================================================

Building on Phase 4 with enhanced fermata and barline data:
1. +27.9% fermata annotations (9,510 → 12,160)
2. Enhanced barline samples from synthetic generation
3. Cleaned dataset (2M+ duplicate labels removed)
4. DeepScoresV2 high-quality fermata data added

Key improvements over Phase 4:
- Fermata: 9,510 → 12,160 (+27.9%)
- Barline_double: 1,734 → 1,883 (+8.6%)
- Data quality: 100% clean (no duplicates)

Target class improvements:
- fermata (class 29): mAP50 0.286 → 0.45-0.55
- barline (class 23): mAP50 0.222 → 0.40-0.50
- barline_double (class 24): mAP50 0.195 → 0.40-0.50

Training optimizations:
- Slightly higher copy_paste (0.35) to boost rare classes
- cls loss weight increased (1.2) for better classification
- 200 epochs with early stopping
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Run Phase 5 training with enhanced fermata/barline data."""

    # Paths
    BASE_DIR = Path("/home/thc1006/dev/music-app/training")
    PHASE4_WEIGHTS = BASE_DIR / "harmony_omr_v2_phase4" / "enhanced_rare_class" / "weights" / "best.pt"
    PHASE5_DATASET = BASE_DIR / "datasets" / "yolo_harmony_v2_phase5" / "harmony_phase5.yaml"
    OUTPUT_DIR = BASE_DIR / "harmony_omr_v2_phase5"

    # Verify paths
    if not PHASE4_WEIGHTS.exists():
        print(f"ERROR: Phase 4 weights not found: {PHASE4_WEIGHTS}")
        print("Looking for alternative weights...")
        # Try Phase 3 weights as fallback
        alt_weights = BASE_DIR / "harmony_omr_v2_phase3" / "external_data_training" / "weights" / "best.pt"
        if alt_weights.exists():
            PHASE4_WEIGHTS = alt_weights
            print(f"Using Phase 3 weights: {PHASE4_WEIGHTS}")
        else:
            print("No suitable weights found!")
            sys.exit(1)

    if not PHASE5_DATASET.exists():
        print(f"ERROR: Phase 5 dataset config not found: {PHASE5_DATASET}")
        sys.exit(1)

    print("=" * 70)
    print("PHASE 5 TRAINING: Fermata & Barline Enhancement")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {PHASE4_WEIGHTS}")
    print(f"Dataset: {PHASE5_DATASET}")
    print(f"Output: {OUTPUT_DIR}")

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Load previous best model
    print("\n[1] Loading base model...")
    model = YOLO(str(PHASE4_WEIGHTS))

    # Training configuration optimized for Phase 5
    train_config = {
        # Basic settings
        "data": str(PHASE5_DATASET),
        "epochs": 200,
        "patience": 30,  # Early stopping patience
        "batch": 16,     # Safe for RTX GPU
        "imgsz": 640,

        # Output settings
        "project": str(OUTPUT_DIR),
        "name": "fermata_barline_enhanced",
        "exist_ok": True,
        "save_period": 10,  # Save checkpoint every 10 epochs

        # Learning rate (fine-tuning from Phase 4)
        "lr0": 0.0003,   # Lower LR for fine-tuning
        "lrf": 0.01,     # Final LR ratio
        "warmup_epochs": 3.0,

        # Key augmentations
        "mosaic": 0.5,      # Proven effective
        "mixup": 0.15,      # Slightly more mixup for diversity
        "copy_paste": 0.35, # KEY: Increased for rare classes

        # Music-specific settings
        "flipud": 0.0,   # No vertical flip
        "fliplr": 0.0,   # No horizontal flip (music is directional)
        "degrees": 0.0,  # No rotation (staff lines must be horizontal)

        # Scale and translate
        "scale": 0.3,
        "translate": 0.1,

        # HSV augmentation (moderate)
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.4,

        # Loss weights (emphasize classification for rare classes)
        "box": 7.5,
        "cls": 1.2,      # Increased for rare class emphasis
        "dfl": 1.5,

        # Other settings
        "cache": True,   # Cache images for faster training
        "device": "0",   # GPU 0
        "workers": 8,
        "amp": True,     # Mixed precision
        "close_mosaic": 15,  # Disable mosaic in last 15 epochs
        "deterministic": True,
        "seed": 42,

        # Validation
        "val": True,
        "plots": True,
    }

    print("\n[2] Training Configuration:")
    print("-" * 50)
    key_params = ['epochs', 'lr0', 'copy_paste', 'mosaic', 'mixup', 'cls', 'batch']
    for param in key_params:
        print(f"  {param}: {train_config[param]}")

    print("\n[3] Phase 5 Dataset Statistics:")
    print("  - Total images: 24,910 (22,393 train + 2,517 val)")
    print("  - Fermata annotations: 12,160 (+27.9% vs Phase 4)")
    print("  - Barline annotations: 25,958")
    print("  - Barline_double: 1,883 (+8.6% vs Phase 4)")
    print("  - Data quality: 100% clean (2M+ duplicates removed)")

    print("\n[4] Key Phase 5 Improvements:")
    print("  - DeepScoresV2 fermata data: +2,244 high-quality annotations")
    print("  - Synthetic fermata data: +206 diverse samples")
    print("  - Synthetic barline data: +475 annotations")
    print("  - copy_paste: 0.3 -> 0.35 (rare class boost)")
    print("  - cls loss weight: 1.0 -> 1.2 (classification emphasis)")
    print("  - lr0: 0.0005 -> 0.0003 (finer tuning)")

    print("\n[5] Target Improvements:")
    print("  - fermata mAP50: 0.286 -> 0.45-0.55")
    print("  - barline mAP50: 0.222 -> 0.40-0.50")
    print("  - barline_double mAP50: 0.195 -> 0.40-0.50")

    print("\n[6] Starting training...")
    print("=" * 70)

    # Start training
    results = model.train(**train_config)

    print("\n" + "=" * 70)
    print("PHASE 5 TRAINING COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Print final metrics
    if results:
        print("\nFinal Metrics:")
        try:
            print(f"  Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"  Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        except:
            print("  Results available in training output directory")

    print(f"\nModel saved to: {OUTPUT_DIR}/fermata_barline_enhanced/weights/")
    print("\nNext steps:")
    print("  1. Evaluate per-class mAP improvements")
    print("  2. Compare fermata/barline detection vs Phase 4")
    print("  3. If targets not met, consider Phase 6 with more synthetic data")

    return results


if __name__ == "__main__":
    main()
