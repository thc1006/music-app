#!/usr/bin/env python3
"""
Component Test Suite
====================

Tests all Phase 6 components independently to verify functionality
before running the full training pipeline.

Usage:
    python test_components.py [component]

    component:
        - all (default): Test all components
        - loss: Test barline_focused_loss
        - miner: Test hard_example_miner
        - sampler: Test weighted_sampler
        - integration: Test full integration
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_barline_focused_loss():
    """Test custom loss function."""
    print("\n" + "="*70)
    print("TEST 1: Barline-Focused Loss Function")
    print("="*70)

    from barline_focused_loss import (
        BarlineFocusedLoss,
        FocalLoss,
        create_class_weights_from_analysis,
    )

    # Test 1.1: FocalLoss
    print("\n[1.1] Testing FocalLoss...")
    focal = FocalLoss(alpha=0.25, gamma=2.0)

    inputs = torch.randn(4, 33)  # 4 samples, 33 classes
    targets = torch.tensor([23, 24, 25, 26])  # barline classes

    loss = focal(inputs, targets)
    print(f"  Focal loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("  ✓ FocalLoss working")

    # Test 1.2: BarlineFocusedLoss
    print("\n[1.2] Testing BarlineFocusedLoss...")
    loss_fn = BarlineFocusedLoss(
        num_classes=33,
        class_weights={23: 4.0, 24: 8.0, 25: 2.0, 26: 1.0},
    )

    batch_idx = torch.tensor([0, 0, 1, 1])
    cls = torch.tensor([23, 24, 25, 26])
    bboxes = torch.tensor([
        [0.5, 0.5, 0.002, 0.025],  # Very thin barline
        [0.5, 0.5, 0.26, 0.25],    # barline_double
        [0.5, 0.5, 0.31, 0.26],    # barline_final
        [0.5, 0.5, 0.10, 0.31],    # barline_repeat
    ])

    cls_weights, bbox_weights = loss_fn.get_loss_weights(batch_idx, cls, bboxes)
    print(f"  Class weights: {cls_weights.tolist()}")
    print(f"  Bbox weights: {bbox_weights.tolist()}")

    # Verify thin barline gets extra weight
    assert bbox_weights[0] > 1.0, "Thin barline should have higher bbox weight"
    assert cls_weights[1] > cls_weights[0], "barline_double should have highest class weight"
    print("  ✓ Dynamic weighting working")

    # Test 1.3: Weight calculation from stats
    print("\n[1.3] Testing weight calculation from stats...")
    stats = {
        23: {'count': 25958, 'mAP50': 0.201, 'recall': 0.09},
        24: {'count': 1883, 'mAP50': 0.140, 'recall': 0.133},
        25: {'count': 58819, 'mAP50': 0.708, 'recall': 0.525},
        26: {'count': 18994, 'mAP50': 0.879, 'recall': 0.830},
    }

    weights = create_class_weights_from_analysis(stats, 'hybrid')
    print("  Calculated weights (hybrid strategy):")
    for class_id, weight in weights.items():
        class_name = {23: 'barline', 24: 'barline_double',
                     25: 'barline_final', 26: 'barline_repeat'}[class_id]
        print(f"    {class_name:20s}: {weight:.2f}x")

    # barline_double should have highest weight (worst performance + low count)
    assert weights[24] > weights[23], "barline_double should have highest weight"
    print("  ✓ Weight calculation working")

    print("\n✅ Barline-Focused Loss: ALL TESTS PASSED")
    return True


def test_hard_example_miner():
    """Test hard example mining (without actual model)."""
    print("\n" + "="*70)
    print("TEST 2: Hard Example Miner")
    print("="*70)

    from hard_example_miner import HardExampleMiner, HardExample

    # Test 2.1: HardExample dataclass
    print("\n[2.1] Testing HardExample dataclass...")
    example = HardExample(
        image_path="/path/to/image.jpg",
        example_type='FN',
        class_id=23,
        class_name='barline',
        bbox=[0.5, 0.5, 0.002, 0.025],
        difficulty_score=2.0,
    )
    print(f"  Created example: {example.class_name}, difficulty={example.difficulty_score}")
    print("  ✓ HardExample dataclass working")

    # Test 2.2: IoU calculation
    print("\n[2.2] Testing IoU calculation...")

    # Create a mock miner (without actual model)
    class MockMiner:
        def calculate_iou(self, box1, box2):
            # Use actual implementation
            from hard_example_miner import HardExampleMiner
            # Create temporary instance just for this method
            temp = HardExampleMiner.__new__(HardExampleMiner)
            return temp.calculate_iou(box1, box2)

    miner = MockMiner()

    # Test cases
    box1 = np.array([0.5, 0.5, 0.1, 0.1])  # center, size
    box2_identical = np.array([0.5, 0.5, 0.1, 0.1])
    box2_partial = np.array([0.55, 0.55, 0.1, 0.1])
    box2_no_overlap = np.array([0.8, 0.8, 0.1, 0.1])

    iou_identical = miner.calculate_iou(box1, box2_identical)
    iou_partial = miner.calculate_iou(box1, box2_partial)
    iou_no_overlap = miner.calculate_iou(box1, box2_no_overlap)

    print(f"  IoU (identical): {iou_identical:.4f}")
    print(f"  IoU (partial): {iou_partial:.4f}")
    print(f"  IoU (no overlap): {iou_no_overlap:.4f}")

    assert abs(iou_identical - 1.0) < 0.01, "Identical boxes should have IoU ~1.0"
    assert 0 < iou_partial < 1, "Partial overlap should have 0 < IoU < 1"
    assert iou_no_overlap < 0.01, "No overlap should have IoU ~0"
    print("  ✓ IoU calculation working")

    print("\n✅ Hard Example Miner: BASIC TESTS PASSED")
    print("  (Full test requires trained model - see test_with_model())")
    return True


def test_weighted_sampler():
    """Test weighted batch sampler."""
    print("\n" + "="*70)
    print("TEST 3: Weighted Batch Sampler")
    print("="*70)

    from weighted_sampler import (
        WeightedBatchSampler,
        analyze_label_distribution,
        calculate_optimal_weights,
    )

    # Test 3.1: Create mock labels directory
    print("\n[3.1] Creating mock labels for testing...")
    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    labels_dir = Path(temp_dir) / "labels"
    labels_dir.mkdir()

    # Create mock label files
    # File 1: Contains barline (23)
    with open(labels_dir / "img1.txt", 'w') as f:
        f.write("23 0.5 0.5 0.002 0.025\n")  # barline
        f.write("0 0.3 0.3 0.01 0.01\n")     # notehead

    # File 2: Contains barline_double (24)
    with open(labels_dir / "img2.txt", 'w') as f:
        f.write("24 0.5 0.5 0.26 0.25\n")    # barline_double
        f.write("0 0.3 0.3 0.01 0.01\n")

    # File 3: Contains only noteheads
    with open(labels_dir / "img3.txt", 'w') as f:
        f.write("0 0.3 0.3 0.01 0.01\n")
        f.write("1 0.4 0.4 0.01 0.01\n")

    print(f"  Created {len(list(labels_dir.glob('*.txt')))} mock label files")

    # Test 3.2: WeightedBatchSampler
    print("\n[3.2] Testing WeightedBatchSampler...")
    sampler = WeightedBatchSampler(
        labels_dir=str(labels_dir),
        class_weights={23: 5.0, 24: 8.0},
        num_samples=10,
        replacement=True,
    )

    # Sample some indices
    indices = list(sampler)
    print(f"  Sampled indices (10 samples): {indices}")
    assert len(indices) == 10, "Should sample exactly 10 indices"
    print("  ✓ WeightedBatchSampler working")

    # Test 3.3: Optimal weight calculation
    print("\n[3.3] Testing optimal weight calculation...")
    weights = calculate_optimal_weights(
        labels_dir=str(labels_dir),
        target_classes=[23, 24],
        strategy='inverse_sqrt',
        target_ratio=8.0,
    )
    print(f"  Calculated weights: {weights}")
    assert 23 in weights and 24 in weights, "Should calculate weights for target classes"
    print("  ✓ Optimal weight calculation working")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    print("\n✅ Weighted Batch Sampler: ALL TESTS PASSED")
    return True


def test_integration():
    """Test integration of all components."""
    print("\n" + "="*70)
    print("TEST 4: Component Integration")
    print("="*70)

    from barline_focused_loss import BarlineFocusedLoss
    from weighted_sampler import WeightedBatchSampler

    print("\n[4.1] Testing loss + sampler integration...")

    # Create loss function
    loss_fn = BarlineFocusedLoss(num_classes=33)
    config = loss_fn.get_training_config()

    print("  Loss function config:")
    for key in ['box', 'cls', 'dfl', 'copy_paste']:
        print(f"    {key}: {config[key]}")

    # Verify config is compatible
    assert 'box' in config and 'cls' in config, "Config should have loss weights"
    assert config['copy_paste'] >= 0.3, "copy_paste should be significant"

    print("  ✓ Loss configuration valid")

    print("\n[4.2] Testing config loading...")
    config_path = Path(__file__).parent / "configs" / "phase6_config.yaml"

    if config_path.exists():
        import yaml
        with open(config_path) as f:
            phase6_config = yaml.safe_load(f)

        print("  Loaded phase6_config.yaml:")
        print(f"    Stage 1 epochs: {phase6_config['stage1']['epochs']}")
        print(f"    Stage 2 epochs: {phase6_config['stage2']['epochs']}")
        print(f"    HEM target classes: {phase6_config['hem']['target_classes']}")

        # Verify consistency
        stage1_cls_weights = phase6_config['stage1']['class_weights']
        assert 23 in stage1_cls_weights, "Config should have barline weight"
        assert stage1_cls_weights[24] > stage1_cls_weights[23], \
            "barline_double should have higher weight"

        print("  ✓ Config file valid and consistent")
    else:
        print(f"  ⚠ Config file not found: {config_path}")
        return False

    print("\n✅ Component Integration: ALL TESTS PASSED")
    return True


def test_with_model(model_path=None, data_yaml=None):
    """
    Test with actual trained model (optional).

    This requires a trained model and dataset.
    """
    print("\n" + "="*70)
    print("TEST 5: Full Test with Trained Model")
    print("="*70)

    if model_path is None or data_yaml is None:
        print("\n⚠ Skipping model test (no model/data provided)")
        print("  To run full test:")
        print("    python test_components.py model /path/to/best.pt /path/to/data.yaml")
        return None

    from hard_example_miner import HardExampleMiner

    # Check paths
    model_path = Path(model_path)
    data_yaml = Path(data_yaml)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    if not data_yaml.exists():
        print(f"❌ Data config not found: {data_yaml}")
        return False

    print(f"\n[5.1] Loading model from {model_path}...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return False

    print("\n[5.2] Testing Hard Example Mining on small sample...")
    miner = HardExampleMiner(
        model=model,
        data_yaml=str(data_yaml),
        output_dir="test_hem_output",
        target_classes=[23, 24],
    )

    # Mine just first 10 images for testing
    stats = miner.mine_validation_set(max_images=10)

    print(f"\n  Hard examples found: {len(miner.hard_examples)}")
    print(f"  Stats: {stats}")

    if len(miner.hard_examples) > 0:
        print("  ✓ Hard example mining working")
    else:
        print("  ⚠ No hard examples found (might be normal for 10 images)")

    # Cleanup
    import shutil
    if Path("test_hem_output").exists():
        shutil.rmtree("test_hem_output")

    print("\n✅ Model Test: PASSED")
    return True


def main():
    """Main test runner."""
    print("="*70)
    print("Phase 6 Component Test Suite")
    print("="*70)

    # Parse arguments
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
    else:
        test_name = "all"

    results = {}

    if test_name in ["all", "loss"]:
        results['loss'] = test_barline_focused_loss()

    if test_name in ["all", "miner"]:
        results['miner'] = test_hard_example_miner()

    if test_name in ["all", "sampler"]:
        results['sampler'] = test_weighted_sampler()

    if test_name in ["all", "integration"]:
        results['integration'] = test_integration()

    if test_name == "model":
        if len(sys.argv) >= 4:
            model_path = sys.argv[2]
            data_yaml = sys.argv[3]
            results['model'] = test_with_model(model_path, data_yaml)
        else:
            print("\nERROR: Model test requires model and data paths")
            print("Usage: python test_components.py model /path/to/best.pt /path/to/data.yaml")
            sys.exit(1)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for test, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
            all_passed = False
        else:
            status = "⚠ SKIPPED"

        print(f"  {test:20s}: {status}")

    if all_passed and len(results) > 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou're ready to run Phase 6 training:")
        print("  python custom_training/train_phase6.py")
    elif len(results) == 0:
        print("\n⚠ No tests were run")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please fix errors before running full training.")

    print("="*70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
