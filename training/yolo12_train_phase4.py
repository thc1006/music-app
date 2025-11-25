#!/usr/bin/env python3
"""
Phase 4 Training Script: Enhanced Rare Class Detection

Based on deep research findings:
1. Copy-paste augmentation: +6.1 mAP on rare classes (CVPR 2021)
2. Lower LR for fine-tuning: 0.0005 (from Phase 3's 0.001)
3. Extended training: 200 epochs (proven effective for OMR)
4. Focal loss implicit in YOLO's cls_loss

Key improvements over Phase 3:
- copy_paste: 0.0 -> 0.3 (major improvement for rare classes)
- lr0: 0.001 -> 0.0005 (finer tuning)
- epochs: 150 -> 200 (extended training)
- Enhanced dataset with MUSCIMA++ and Rebelo samples

Target classes to improve:
- fermata (class 29): +1,270 samples from MUSCIMA++/Rebelo
- accidental_natural (class 15): +7,781 samples
- barline (class 23): +5,979 samples
- barline_double (class 24): +506 samples
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Run Phase 4 training with optimized configuration."""

    # Paths
    BASE_DIR = Path("/home/thc1006/dev/music-app/training")
    PHASE3_WEIGHTS = BASE_DIR / "harmony_omr_v2_phase3" / "external_data_training" / "weights" / "best.pt"
    PHASE4_DATASET = BASE_DIR / "datasets" / "yolo_harmony_v2_phase4" / "harmony_phase4.yaml"
    OUTPUT_DIR = BASE_DIR / "harmony_omr_v2_phase4"

    # Verify paths
    if not PHASE3_WEIGHTS.exists():
        print(f"ERROR: Phase 3 weights not found: {PHASE3_WEIGHTS}")
        sys.exit(1)

    if not PHASE4_DATASET.exists():
        print(f"ERROR: Phase 4 dataset config not found: {PHASE4_DATASET}")
        sys.exit(1)

    print("=" * 70)
    print("PHASE 4 TRAINING: Enhanced Rare Class Detection")
    print("=" * 70)
    print(f"\nBase model: {PHASE3_WEIGHTS}")
    print(f"Dataset: {PHASE4_DATASET}")
    print(f"Output: {OUTPUT_DIR}")

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Load Phase 3 best model
    print("\n[1] Loading Phase 3 best model...")
    model = YOLO(str(PHASE3_WEIGHTS))

    # Training configuration based on research
    # Key findings:
    # - Copy-paste: +6.1 mAP on rare classes (most effective augmentation)
    # - Lower LR: 0.0005 for fine-tuning from pretrained
    # - Extended epochs: 200+ for OMR tasks
    # - Mosaic: 0.5 (good for object detection)
    # - No horizontal flip (music is left-to-right)

    train_config = {
        # Basic settings
        "data": str(PHASE4_DATASET),
        "epochs": 200,
        "patience": 30,  # Early stopping patience
        "batch": 16,     # Safe for RTX GPU
        "imgsz": 640,

        # Output settings
        "project": str(OUTPUT_DIR),
        "name": "enhanced_rare_class",
        "exist_ok": True,
        "save_period": 10,  # Save checkpoint every 10 epochs

        # Learning rate (lower for fine-tuning)
        "lr0": 0.0005,   # Reduced from 0.001 for finer tuning
        "lrf": 0.01,     # Final LR ratio
        "warmup_epochs": 3.0,

        # Key augmentations
        "mosaic": 0.5,      # Proven effective
        "mixup": 0.1,       # Light mixup for diversity
        "copy_paste": 0.3,  # KEY: +6.1 mAP on rare classes!

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

        # Loss weights (increase cls for rare classes)
        "box": 7.5,
        "cls": 1.0,      # Increased from 0.5 to emphasize classification
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

    print("\n[3] Key Phase 4 Improvements:")
    print("  - copy_paste: 0.0 -> 0.3 (critical for rare classes)")
    print("  - lr0: 0.001 -> 0.0005 (finer tuning)")
    print("  - cls loss weight: 0.5 -> 1.0 (emphasize classification)")
    print("  - epochs: 150 -> 200 (extended training)")
    print("  - mixup: 0.0 -> 0.1 (additional diversity)")
    print("  - Dataset: +10,000 images with enhanced rare classes")

    print("\n[4] Starting training...")
    print("=" * 70)

    # Start training
    results = model.train(**train_config)

    print("\n" + "=" * 70)
    print("PHASE 4 TRAINING COMPLETE")
    print("=" * 70)

    # Print final metrics
    if results:
        print("\nFinal Metrics:")
        print(f"  Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

    print(f"\nModel saved to: {OUTPUT_DIR}/enhanced_rare_class/weights/")
    print("\nNext steps:")
    print("  1. Evaluate per-class mAP improvements")
    print("  2. Check fermata, natural, barline detection rates")
    print("  3. If mAP50 < 0.65, consider Phase 5 with synthetic data")

    return results


if __name__ == "__main__":
    main()
