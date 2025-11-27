#!/usr/bin/env python3
"""
Weighted Batch Sampler for Barline-Focused Training
====================================================

Implements class-weighted sampling to increase exposure to rare classes:

1. Per-image class-based weighting
2. Oversampling images containing barline classes
3. Integration with Ultralytics DataLoader

Strategy:
- Images with barline (class 23): 5x sampling probability
- Images with barline_double (class 24): 8x sampling probability
- Images with barline_final (class 25): 2x sampling probability
- Other images: 1x baseline probability

This ensures the model sees more barline examples during training,
addressing the 91% miss rate for barline class.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.utils.data import Sampler


class WeightedBatchSampler(Sampler):
    """
    Weighted sampler that oversamples images containing target classes.

    This sampler assigns higher probability to images containing
    difficult classes (e.g., barline, barline_double).
    """

    def __init__(
        self,
        labels_dir: str,
        class_weights: Optional[Dict[int, float]] = None,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            labels_dir: Directory containing YOLO format label files
            class_weights: Dict mapping class_id to sampling weight multiplier
            num_samples: Number of samples per epoch (None = dataset size)
            replacement: Sample with replacement (True for oversampling)
            seed: Random seed for reproducibility
        """
        self.labels_dir = Path(labels_dir)
        self.replacement = replacement
        self.seed = seed

        # Default weights for barline classes
        if class_weights is None:
            class_weights = {
                23: 5.0,   # barline - critical (recall 9%)
                24: 8.0,   # barline_double - worst (mAP 0.140)
                25: 2.0,   # barline_final
                26: 1.5,   # barline_repeat
            }
        self.class_weights = class_weights

        # Scan labels directory
        print(f"WeightedBatchSampler: Scanning {labels_dir}...")
        self.label_files = sorted(self.labels_dir.glob('*.txt'))
        self.num_images = len(self.label_files)

        if num_samples is None:
            num_samples = self.num_images
        self.num_samples = num_samples

        # Calculate per-image weights
        self.image_weights = self._calculate_image_weights()

        # Statistics
        self._print_statistics()

        print(f"WeightedBatchSampler initialized:")
        print(f"  Images: {self.num_images}")
        print(f"  Samples per epoch: {self.num_samples}")
        print(f"  Replacement: {replacement}")

    def _calculate_image_weights(self) -> np.ndarray:
        """
        Calculate sampling weight for each image.

        Returns:
            Array of weights (num_images,)
        """
        weights = np.ones(self.num_images, dtype=np.float32)

        for idx, label_file in enumerate(self.label_files):
            try:
                with open(label_file) as f:
                    classes_in_image = set()
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            classes_in_image.add(class_id)

                    # Apply maximum weight among all classes in image
                    image_weight = 1.0
                    for class_id in classes_in_image:
                        if class_id in self.class_weights:
                            image_weight = max(
                                image_weight,
                                self.class_weights[class_id]
                            )

                    weights[idx] = image_weight

            except Exception as e:
                # If label file is empty or invalid, keep weight = 1.0
                pass

        return weights

    def _print_statistics(self):
        """Print sampling statistics."""
        print("\nSampling weight distribution:")

        weight_counts = Counter(self.image_weights)
        for weight in sorted(weight_counts.keys()):
            count = weight_counts[weight]
            pct = 100.0 * count / self.num_images
            print(f"  Weight {weight:.1f}x: {count} images ({pct:.1f}%)")

        # Effective oversampling
        avg_weight = self.image_weights.mean()
        print(f"\nAverage weight: {avg_weight:.2f}x")
        print(f"Effective oversampling: {avg_weight:.1f}x baseline")

    def __iter__(self):
        """
        Generate sample indices.

        Yields:
            Index of sampled image
        """
        # Set random seed for reproducibility
        rng = np.random.RandomState(self.seed)

        # Normalize weights to probabilities
        probs = self.image_weights / self.image_weights.sum()

        # Sample indices
        indices = rng.choice(
            self.num_images,
            size=self.num_samples,
            replace=self.replacement,
            p=probs,
        )

        return iter(indices.tolist())

    def __len__(self):
        """Number of samples per epoch."""
        return self.num_samples


class ProgressiveWeightedSampler(WeightedBatchSampler):
    """
    Progressive sampler that gradually increases focus on hard classes.

    In early epochs, use balanced sampling.
    In later epochs, increase weight on hard classes.

    This prevents overfitting to hard examples early in training.
    """

    def __init__(
        self,
        labels_dir: str,
        class_weights: Optional[Dict[int, float]] = None,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        seed: int = 42,
        warmup_epochs: int = 50,
        max_weight_scale: float = 1.0,
    ):
        """
        Args:
            warmup_epochs: Number of epochs before applying full weights
            max_weight_scale: Maximum weight scaling factor
        """
        self.warmup_epochs = warmup_epochs
        self.max_weight_scale = max_weight_scale
        self.current_epoch = 0

        super().__init__(
            labels_dir=labels_dir,
            class_weights=class_weights,
            num_samples=num_samples,
            replacement=replacement,
            seed=seed,
        )

    def set_epoch(self, epoch: int):
        """Update current epoch for progressive weighting."""
        self.current_epoch = epoch

        # Calculate weight scaling
        if epoch < self.warmup_epochs:
            # Linear ramp from 0 to max_weight_scale
            scale = (epoch / self.warmup_epochs) * self.max_weight_scale
        else:
            scale = self.max_weight_scale

        # Apply scaling to base weights
        # Start from uniform weights and scale up
        base_weights = np.ones(self.num_images, dtype=np.float32)
        scaled_weights = base_weights + (self.image_weights - base_weights) * scale

        self.image_weights = scaled_weights

        print(f"Epoch {epoch}: Weight scaling = {scale:.2f}x")


def create_weighted_dataloader(
    dataset,
    batch_size: int,
    labels_dir: str,
    class_weights: Optional[Dict[int, float]] = None,
    num_workers: int = 8,
    progressive: bool = False,
    **kwargs,
):
    """
    Create DataLoader with weighted sampling.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        labels_dir: Path to labels directory
        class_weights: Per-class sampling weights
        num_workers: Number of data loading workers
        progressive: Use progressive weighting
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader with weighted sampler
    """
    from torch.utils.data import DataLoader

    # Create sampler
    if progressive:
        sampler = ProgressiveWeightedSampler(
            labels_dir=labels_dir,
            class_weights=class_weights,
            num_samples=len(dataset),
        )
    else:
        sampler = WeightedBatchSampler(
            labels_dir=labels_dir,
            class_weights=class_weights,
            num_samples=len(dataset),
        )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        **kwargs,
    )

    return dataloader, sampler


def analyze_label_distribution(
    labels_dir: str,
    target_classes: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    Analyze class distribution in dataset.

    Args:
        labels_dir: Directory containing YOLO labels
        target_classes: Classes to analyze (None = all)

    Returns:
        Dict mapping class_id to count
    """
    labels_dir = Path(labels_dir)
    label_files = sorted(labels_dir.glob('*.txt'))

    class_counts = Counter()
    images_with_class = defaultdict(int)

    print(f"Analyzing {len(label_files)} label files...")

    for label_file in label_files:
        classes_in_image = set()

        try:
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])

                        if target_classes is None or class_id in target_classes:
                            class_counts[class_id] += 1
                            classes_in_image.add(class_id)

            # Count images containing each class
            for class_id in classes_in_image:
                images_with_class[class_id] += 1

        except Exception:
            pass

    # Print results
    print("\nClass distribution:")
    print(f"{'Class ID':<10} {'Instances':<12} {'Images':<10} {'Avg/Image':<10}")
    print("-" * 50)

    for class_id in sorted(class_counts.keys()):
        instances = class_counts[class_id]
        images = images_with_class[class_id]
        avg = instances / images if images > 0 else 0

        print(f"{class_id:<10} {instances:<12} {images:<10} {avg:<10.2f}")

    return dict(class_counts)


def calculate_optimal_weights(
    labels_dir: str,
    target_classes: List[int],
    strategy: str = 'inverse_sqrt',
    target_ratio: float = 5.0,
) -> Dict[int, float]:
    """
    Calculate optimal sampling weights based on class distribution.

    Args:
        labels_dir: Directory containing labels
        target_classes: Classes to calculate weights for
        strategy: Weighting strategy
            - 'inverse': 1 / count
            - 'inverse_sqrt': 1 / sqrt(count)
            - 'effective_samples': Based on Cui et al. 2019
        target_ratio: Target oversampling ratio for rarest class

    Returns:
        Dict mapping class_id to weight
    """
    # Get class distribution (images containing each class)
    labels_dir = Path(labels_dir)
    label_files = sorted(labels_dir.glob('*.txt'))

    images_with_class = defaultdict(int)

    for label_file in label_files:
        classes_in_image = set()
        try:
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id in target_classes:
                            classes_in_image.add(class_id)

            for class_id in classes_in_image:
                images_with_class[class_id] += 1
        except Exception:
            pass

    # Calculate weights
    counts = {c: images_with_class[c] for c in target_classes}
    max_count = max(counts.values())
    min_count = min(counts.values())

    weights = {}

    for class_id, count in counts.items():
        if strategy == 'inverse':
            weight = max_count / count
        elif strategy == 'inverse_sqrt':
            weight = np.sqrt(max_count / count)
        elif strategy == 'effective_samples':
            # Based on "Class-Balanced Loss Based on Effective Number of Samples"
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, count)
            weight = (1.0 - beta) / effective_num
        else:
            weight = 1.0

        weights[class_id] = float(weight)

    # Normalize to target ratio
    max_weight = max(weights.values())
    for class_id in weights:
        weights[class_id] = (weights[class_id] / max_weight) * target_ratio

    # Ensure minimum weight = 1.0
    min_weight = min(weights.values())
    if min_weight < 1.0:
        for class_id in weights:
            weights[class_id] /= min_weight

    return weights


# Example usage
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Weighted Batch Sampler - Test Mode")
    print("=" * 70)

    # Test configuration
    if len(sys.argv) > 1:
        labels_dir = sys.argv[1]
    else:
        labels_dir = "/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase5/labels/train"

    if not Path(labels_dir).exists():
        print(f"\nERROR: Labels directory not found: {labels_dir}")
        print("Please provide labels directory as argument.")
        sys.exit(1)

    # Analyze distribution
    print("\n1. Analyzing label distribution...")
    barline_classes = [23, 24, 25, 26]
    analyze_label_distribution(labels_dir, target_classes=barline_classes)

    # Calculate optimal weights
    print("\n2. Calculating optimal weights...")
    for strategy in ['inverse_sqrt', 'inverse', 'effective_samples']:
        print(f"\nStrategy: {strategy}")
        weights = calculate_optimal_weights(
            labels_dir,
            target_classes=barline_classes,
            strategy=strategy,
            target_ratio=8.0,
        )
        for class_id, weight in weights.items():
            class_names = {23: 'barline', 24: 'barline_double',
                          25: 'barline_final', 26: 'barline_repeat'}
            print(f"  {class_names[class_id]:20s}: {weight:.2f}x")

    # Test sampler
    print("\n3. Testing WeightedBatchSampler...")
    sampler = WeightedBatchSampler(
        labels_dir=labels_dir,
        class_weights={23: 5.0, 24: 8.0, 25: 2.0, 26: 1.5},
        num_samples=1000,
    )

    # Sample a few batches
    print("\nSampling first 10 indices:")
    indices = list(sampler)[:10]
    print(f"  Indices: {indices}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
