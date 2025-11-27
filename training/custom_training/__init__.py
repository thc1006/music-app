"""
Custom Training Components for YOLO12 OMR
==========================================

This package provides advanced training components for improving
barline detection in optical music recognition:

- barline_focused_loss: Custom loss function with per-class weighting
- hard_example_miner: Hard Example Mining (HEM) system
- weighted_sampler: Class-weighted data sampling
- train_phase6: Multi-stage training orchestration

Target improvements:
- barline (class 23): mAP50 0.201 → 0.50-0.60
- barline_double (class 24): mAP50 0.140 → 0.40-0.50
- barline_final (class 25): improve precision
"""

__version__ = "1.0.0"
__author__ = "Harmony OMR Team"

from .barline_focused_loss import BarlineFocusedLoss
from .hard_example_miner import HardExampleMiner
from .weighted_sampler import WeightedBatchSampler

__all__ = [
    "BarlineFocusedLoss",
    "HardExampleMiner",
    "WeightedBatchSampler",
]
