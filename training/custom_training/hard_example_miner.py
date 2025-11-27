#!/usr/bin/env python3
"""
Hard Example Mining (HEM) System
=================================

Identifies and curates hard examples for focused training:

1. False Negatives (FN): Ground truth objects that model missed
2. Low Confidence Predictions: Detections with conf < threshold
3. High IoU Misclassifications: Correct localization, wrong class

Target barline issues from Phase 5:
- barline (ID 23): 91% missed (recall 9%)
- barline_double (ID 24): 86.7% missed (recall 13.3%)

Strategy:
1. Run inference on validation set
2. Identify hard examples using multiple criteria
3. Create curated hard example subset
4. Use for Stage 2 fine-tuning

Output:
- hard_examples_{class_name}.txt: List of image paths with hard examples
- hard_examples_stats.json: Statistics and analysis
- hard_examples/: Visualizations (optional)
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class HardExample:
    """Represents a single hard example."""
    image_path: str
    example_type: str  # 'FN', 'low_conf', 'misclass'
    class_id: int
    class_name: str
    bbox: List[float]  # [x, y, w, h] normalized
    confidence: float = 0.0  # 0 for FN
    predicted_class: Optional[int] = None
    iou: float = 0.0  # IoU with GT (if applicable)
    difficulty_score: float = 1.0  # Higher = harder


class HardExampleMiner:
    """
    Hard Example Mining system for YOLO models.

    Analyzes model predictions to identify difficult examples
    that need additional training focus.
    """

    def __init__(
        self,
        model,
        data_yaml: str,
        output_dir: str = "hard_examples_analysis",
        target_classes: Optional[List[int]] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        low_conf_threshold: float = 0.3,
    ):
        """
        Args:
            model: Trained YOLO model or path to weights
            data_yaml: Path to dataset YAML config
            output_dir: Directory to save hard example analysis
            target_classes: List of class IDs to focus on (None = all)
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching predictions to GT
            low_conf_threshold: Threshold for identifying low-conf predictions
        """
        self.model = model
        self.data_yaml = Path(data_yaml)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Target classes (default: barline classes)
        if target_classes is None:
            target_classes = [23, 24, 25, 26]  # All barline classes
        self.target_classes = set(target_classes)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.low_conf_threshold = low_conf_threshold

        # Load class names from data YAML
        self.class_names = self._load_class_names()

        # Storage for hard examples
        self.hard_examples: List[HardExample] = []
        self.stats: Dict = defaultdict(lambda: defaultdict(int))

        print(f"HardExampleMiner initialized:")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Target classes: {[self.class_names[i] for i in target_classes]}")
        print(f"  Conf threshold: {conf_threshold}")
        print(f"  IoU threshold: {iou_threshold}")

    def _load_class_names(self) -> Dict[int, str]:
        """Load class names from data YAML."""
        import yaml
        with open(self.data_yaml) as f:
            data = yaml.safe_load(f)
        names = data.get('names', {})
        return {i: name for i, name in enumerate(names.values())}

    def calculate_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray,
    ) -> float:
        """
        Calculate IoU between two boxes in [x, y, w, h] format.

        Args:
            box1, box2: Boxes in normalized [x, y, w, h] format

        Returns:
            IoU score
        """
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2

        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2

        # Calculate intersection
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def analyze_image(
        self,
        image_path: str,
        gt_labels_path: str,
    ) -> List[HardExample]:
        """
        Analyze a single image for hard examples.

        Args:
            image_path: Path to image
            gt_labels_path: Path to ground truth labels (YOLO format)

        Returns:
            List of hard examples found in this image
        """
        hard_examples = []

        # Load ground truth
        gt_boxes = []
        if os.path.exists(gt_labels_path):
            with open(gt_labels_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        gt_boxes.append((class_id, bbox))

        # Filter GT to target classes
        gt_boxes = [(c, b) for c, b in gt_boxes if c in self.target_classes]

        if not gt_boxes:
            return []  # No target class in this image

        # Run inference
        try:
            results = self.model.predict(
                image_path,
                conf=0.001,  # Very low threshold to catch all predictions
                iou=0.5,
                verbose=False,
            )
        except Exception as e:
            print(f"Warning: Failed to predict {image_path}: {e}")
            return []

        if not results or len(results) == 0:
            # Model produced no predictions - all GT are FN
            for class_id, bbox in gt_boxes:
                hard_examples.append(HardExample(
                    image_path=str(image_path),
                    example_type='FN',
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    bbox=bbox,
                    confidence=0.0,
                    difficulty_score=2.0,  # High difficulty
                ))
            return hard_examples

        # Extract predictions
        result = results[0]
        pred_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                # Convert to normalized [x, y, w, h]
                xyxy = box.xyxy[0].cpu().numpy()
                img_h, img_w = result.orig_shape
                x = (xyxy[0] + xyxy[2]) / 2 / img_w
                y = (xyxy[1] + xyxy[3]) / 2 / img_h
                w = (xyxy[2] - xyxy[0]) / img_w
                h = (xyxy[3] - xyxy[1]) / img_h
                bbox = [x, y, w, h]
                pred_boxes.append((class_id, conf, bbox))

        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()

        for pred_idx, (pred_cls, pred_conf, pred_bbox) in enumerate(pred_boxes):
            if pred_cls not in self.target_classes:
                continue

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, (gt_cls, gt_bbox) in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = self.calculate_iou(
                    np.array(pred_bbox),
                    np.array(gt_bbox),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if this is a hard example
            if best_iou >= self.iou_threshold:
                # Good localization
                matched_pred.add(pred_idx)
                if best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    gt_cls, gt_bbox = gt_boxes[best_gt_idx]

                    # Check for misclassification
                    if pred_cls != gt_cls:
                        hard_examples.append(HardExample(
                            image_path=str(image_path),
                            example_type='misclass',
                            class_id=gt_cls,
                            class_name=self.class_names[gt_cls],
                            bbox=gt_bbox,
                            confidence=pred_conf,
                            predicted_class=pred_cls,
                            iou=best_iou,
                            difficulty_score=1.5,
                        ))

                    # Check for low confidence (even if correct class)
                    elif pred_conf < self.low_conf_threshold:
                        hard_examples.append(HardExample(
                            image_path=str(image_path),
                            example_type='low_conf',
                            class_id=gt_cls,
                            class_name=self.class_names[gt_cls],
                            bbox=gt_bbox,
                            confidence=pred_conf,
                            iou=best_iou,
                            difficulty_score=1.0 + (self.low_conf_threshold - pred_conf),
                        ))

        # Identify False Negatives (unmatched GT)
        for gt_idx, (gt_cls, gt_bbox) in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                hard_examples.append(HardExample(
                    image_path=str(image_path),
                    example_type='FN',
                    class_id=gt_cls,
                    class_name=self.class_names[gt_cls],
                    bbox=gt_bbox,
                    confidence=0.0,
                    difficulty_score=2.0,  # High difficulty
                ))

        return hard_examples

    def mine_validation_set(
        self,
        val_images_dir: Optional[str] = None,
        val_labels_dir: Optional[str] = None,
        max_images: Optional[int] = None,
    ) -> Dict:
        """
        Mine hard examples from validation set.

        Args:
            val_images_dir: Path to validation images (auto-detected from YAML)
            val_labels_dir: Path to validation labels (auto-detected from YAML)
            max_images: Maximum number of images to analyze (None = all)

        Returns:
            Statistics dict
        """
        # Auto-detect paths from YAML if not provided
        if val_images_dir is None or val_labels_dir is None:
            import yaml
            with open(self.data_yaml) as f:
                data = yaml.safe_load(f)

            dataset_root = self.data_yaml.parent / data.get('path', '')
            if val_images_dir is None:
                val_images_dir = dataset_root / data.get('val', 'images/val')
            if val_labels_dir is None:
                # Assume labels are in parallel directory
                val_labels_dir = str(val_images_dir).replace('/images/', '/labels/')

        val_images_dir = Path(val_images_dir)
        val_labels_dir = Path(val_labels_dir)

        print(f"\nMining hard examples from validation set:")
        print(f"  Images: {val_images_dir}")
        print(f"  Labels: {val_labels_dir}")

        # Get all images
        image_files = sorted(val_images_dir.glob('*.jpg')) + \
                     sorted(val_images_dir.glob('*.png'))

        if max_images:
            image_files = image_files[:max_images]

        print(f"  Total images: {len(image_files)}")

        # Analyze each image
        self.hard_examples = []
        for img_path in tqdm(image_files, desc="Analyzing images"):
            # Get corresponding label file
            label_path = val_labels_dir / (img_path.stem + '.txt')

            # Analyze
            examples = self.analyze_image(str(img_path), str(label_path))
            self.hard_examples.extend(examples)

            # Update stats
            for ex in examples:
                self.stats[ex.class_name][ex.example_type] += 1

        # Generate report
        return self.generate_report()

    def generate_report(self) -> Dict:
        """
        Generate analysis report and save results.

        Returns:
            Statistics dict
        """
        print(f"\n{'='*70}")
        print("Hard Example Mining Results")
        print(f"{'='*70}")

        # Overall stats
        total_hard = len(self.hard_examples)
        print(f"\nTotal hard examples found: {total_hard}")

        # Per-class breakdown
        print("\nPer-class breakdown:")
        print(f"{'Class':<20} {'FN':<8} {'Low Conf':<10} {'Misclass':<10} {'Total':<8}")
        print("-" * 70)

        class_stats = {}
        for class_name in sorted(self.stats.keys()):
            fn = self.stats[class_name]['FN']
            low_conf = self.stats[class_name]['low_conf']
            misclass = self.stats[class_name]['misclass']
            total = fn + low_conf + misclass

            print(f"{class_name:<20} {fn:<8} {low_conf:<10} {misclass:<10} {total:<8}")

            class_stats[class_name] = {
                'FN': fn,
                'low_conf': low_conf,
                'misclass': misclass,
                'total': total,
            }

        # Save hard example lists
        self._save_hard_example_lists()

        # Save statistics JSON
        stats_dict = {
            'total_hard_examples': total_hard,
            'per_class': class_stats,
            'config': {
                'conf_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'low_conf_threshold': self.low_conf_threshold,
            }
        }

        stats_path = self.output_dir / 'hard_examples_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

        print(f"\nStatistics saved to: {stats_path}")

        return stats_dict

    def _save_hard_example_lists(self):
        """Save lists of images containing hard examples."""
        # Group by class
        by_class: Dict[str, Set[str]] = defaultdict(set)

        for ex in self.hard_examples:
            by_class[ex.class_name].add(ex.image_path)

        # Save per-class lists
        for class_name, image_paths in by_class.items():
            output_file = self.output_dir / f"hard_examples_{class_name}.txt"
            with open(output_file, 'w') as f:
                for img_path in sorted(image_paths):
                    f.write(f"{img_path}\n")

            print(f"  {class_name}: {len(image_paths)} images -> {output_file}")

        # Save all hard examples details
        all_examples_file = self.output_dir / 'hard_examples_detailed.json'
        with open(all_examples_file, 'w') as f:
            json.dump(
                [asdict(ex) for ex in self.hard_examples],
                f,
                indent=2,
            )

    def create_hard_example_dataset(
        self,
        source_images_dir: str,
        source_labels_dir: str,
        output_dataset_dir: str,
        min_difficulty: float = 1.0,
    ):
        """
        Create a curated dataset containing only hard examples.

        Args:
            source_images_dir: Source validation images directory
            source_labels_dir: Source validation labels directory
            output_dataset_dir: Output directory for hard example dataset
            min_difficulty: Minimum difficulty score to include
        """
        output_dir = Path(output_dataset_dir)
        images_dir = output_dir / 'images' / 'train'
        labels_dir = output_dir / 'labels' / 'train'

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Get unique image paths meeting difficulty threshold
        hard_image_paths = set()
        for ex in self.hard_examples:
            if ex.difficulty_score >= min_difficulty:
                hard_image_paths.add(ex.image_path)

        print(f"\nCreating hard example dataset:")
        print(f"  Total hard images: {len(hard_image_paths)}")
        print(f"  Output: {output_dir}")

        # Copy images and labels
        copied = 0
        for img_path in tqdm(hard_image_paths, desc="Copying files"):
            img_path = Path(img_path)
            label_path = Path(str(img_path).replace('/images/', '/labels/').replace(
                img_path.suffix, '.txt'
            ))

            # Copy image
            dst_img = images_dir / img_path.name
            if img_path.exists():
                shutil.copy2(img_path, dst_img)

            # Copy label
            dst_label = labels_dir / label_path.name
            if label_path.exists():
                shutil.copy2(label_path, dst_label)
                copied += 1

        print(f"  Copied {copied} image-label pairs")

        # Create dataset YAML
        yaml_content = f"""# Hard Example Dataset for Stage 2 Training
# Auto-generated by HardExampleMiner

path: {output_dir.absolute()}
train: images/train
val: images/train  # Use same for validation in Stage 2

# Copy class names from original dataset
nc: 33
names: {{
  0: notehead_filled, 1: notehead_hollow, 2: stem, 3: beam,
  4: flag_8th, 5: flag_16th, 6: flag_32nd, 7: augmentation_dot,
  8: clef_treble, 9: clef_bass, 10: clef_alto, 11: clef_tenor,
  12: accidental_sharp, 13: accidental_flat, 14: accidental_double_sharp,
  15: accidental_double_flat, 16: accidental_natural,
  17: rest_quarter, 18: rest_half, 19: rest_whole, 20: rest_8th,
  21: time_signature, 22: key_signature,
  23: barline, 24: barline_double, 25: barline_final, 26: barline_repeat,
  27: dynamic_piano, 28: dynamic_forte, 29: fermata,
  30: slur, 31: tie, 32: ledger_line
}}

# Hard example statistics
# Total images: {len(hard_image_paths)}
# Min difficulty: {min_difficulty}
"""
        yaml_path = output_dir / 'hard_examples.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"  Dataset config: {yaml_path}")

        return str(yaml_path)


# Example usage
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Hard Example Mining System - Test Mode")
    print("=" * 70)

    # Test configuration
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "/home/thc1006/dev/music-app/training/harmony_omr_v2_phase5/fermata_barline_enhanced/weights/best.pt"

    if len(sys.argv) > 2:
        data_yaml = sys.argv[2]
    else:
        data_yaml = "/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase5/harmony_phase5.yaml"

    print(f"\nModel: {model_path}")
    print(f"Dataset: {data_yaml}")

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Please provide model path as first argument.")
        sys.exit(1)

    if not os.path.exists(data_yaml):
        print(f"\nERROR: Dataset config not found at {data_yaml}")
        print("Please provide data YAML path as second argument.")
        sys.exit(1)

    # Load model
    print("\nLoading model...")
    from ultralytics import YOLO
    model = YOLO(model_path)

    # Create miner
    miner = HardExampleMiner(
        model=model,
        data_yaml=data_yaml,
        output_dir="hard_examples_analysis",
        target_classes=[23, 24, 25, 26],  # All barline classes
        conf_threshold=0.5,
        iou_threshold=0.5,
        low_conf_threshold=0.3,
    )

    # Mine validation set (test on first 100 images)
    print("\nMining validation set (first 100 images for testing)...")
    stats = miner.mine_validation_set(max_images=100)

    # Create hard example dataset
    print("\nCreating hard example dataset...")
    hard_yaml = miner.create_hard_example_dataset(
        source_images_dir="auto",
        source_labels_dir="auto",
        output_dataset_dir="hard_examples_dataset",
        min_difficulty=1.5,
    )

    print("\n" + "=" * 70)
    print("Hard example mining completed!")
    print(f"Results saved to: {miner.output_dir}")
    print(f"Hard example dataset: {hard_yaml}")
    print("=" * 70)
