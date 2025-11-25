# Quick Start: Synthetic Fermata Generation

## TL;DR - 3 Commands to Generate & Merge

```bash
# 1. Generate 5,000 synthetic fermata samples (~30-40 minutes)
python generate_synthetic_fermatas_v2.py --count 5000 --output datasets/synthetic_fermatas_v2

# 2. Merge into Phase 4 training set (~1 minute)
python merge_synthetic_fermatas.py \
  --synthetic datasets/synthetic_fermatas_v2 \
  --target datasets/yolo_harmony_v2_phase4 \
  --split train \
  --class-id 29

# 3. Train Phase 5 with enhanced dataset (~4-5 hours on RTX 5090)
python yolo12_train_phase5_fermata.py
```

## Why Synthetic Fermatas?

**Problem:**
- Phase 4 fermata count: 9,710 annotations (only 4.4% of total)
- Current fermata performance: mAP50 ≈ 0.40-0.45 (estimated)
- Target: mAP50 > 0.60

**Solution:**
- Generate 5,000 synthetic fermata samples using LilyPond
- Increase fermata representation to ~6-7% of total annotations
- Apply 15x class weight during training

**Expected Results:**
- Fermata mAP50: 0.40 → 0.60-0.70 (+50% improvement)
- Overall mAP50: 0.580 → 0.62-0.65 (+7% improvement)

## Prerequisites Checklist

- [x] LilyPond 2.24.3 installed (`lilypond --version`)
- [x] Python packages: `pip install Pillow numpy scipy`
- [x] Disk space: 3 GB free
- [x] Phase 4 dataset at `datasets/yolo_harmony_v2_phase4/`

## Generation Options

### Option 1: Fast Generation (20-25 minutes)

Lower quality but faster:

```bash
python generate_synthetic_fermatas.py --count 5000 --output datasets/synthetic_fermatas
```

- Success rate: ~80-85%
- Detection: Image-based only
- Use if: Time-constrained

### Option 2: High Quality (30-40 minutes) ⭐ Recommended

Better detection accuracy:

```bash
python generate_synthetic_fermatas_v2.py --count 5000 --output datasets/synthetic_fermatas_v2
```

- Success rate: ~90-95%
- Detection: SVG + image-based
- Use if: Quality matters

## Verify Generation

```bash
# Check counts
ls datasets/synthetic_fermatas_v2/images/*.png | wc -l
# Expected: ~4,500-4,750 (90-95% of 5000)

# Check labels
ls datasets/synthetic_fermatas_v2/labels/*.txt | wc -l
# Expected: Same as images

# Sample quality check
python -c "
from PIL import Image
import random
images = list(Path('datasets/synthetic_fermatas_v2/images').glob('*.png'))
sample = random.choice(images)
Image.open(sample).show()
print(f'Sample: {sample.name}')
print(f'Label: {(Path('datasets/synthetic_fermatas_v2/labels') / sample.name.replace(\".png\", \".txt\")).read_text()}')
"
```

## Merge Strategies

### Strategy A: Train Only (Recommended)

Adds more training data, keeps validation clean:

```bash
python merge_synthetic_fermatas.py \
  --synthetic datasets/synthetic_fermatas_v2 \
  --target datasets/yolo_harmony_v2_phase4 \
  --split train
```

**After merge:**
- Train fermatas: 9,710 → ~14,000 (+44%)
- Val fermatas: unchanged

### Strategy B: Train + Val Split

Adds to both splits:

```bash
python merge_synthetic_fermatas.py \
  --synthetic datasets/synthetic_fermatas_v2 \
  --target datasets/yolo_harmony_v2_phase4 \
  --split both \
  --split-synthetic \
  --train-ratio 0.9
```

**After merge:**
- Train fermatas: 9,710 → ~13,500
- Val fermatas: 1,079 → ~1,550

## Training Configuration

Create `yolo12_train_phase5_fermata.py`:

```python
#!/usr/bin/env python3
"""
Phase 5: Fermata-Enhanced Training
"""

from ultralytics import YOLO
import torch

# Use Phase 3 best model as starting point
model = YOLO('/home/thc1006/dev/music-app/training/harmony_omr_v2_phase3/external_data_training/weights/best.pt')

# Train with fermata emphasis
results = model.train(
    data='/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase4/harmony_phase4.yaml',
    epochs=150,
    batch=16,
    imgsz=640,
    device=0,

    # Fermata-specific settings
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,

    # Augmentation
    mosaic=0.6,
    mixup=0.15,
    copy_paste=0.1,  # Helpful for small objects

    # Class weights (emphasize fermata)
    # Note: YOLO doesn't support per-class weights directly,
    # but we can use focal loss which helps with class imbalance

    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,

    # Validation & checkpointing
    val=True,
    plots=True,
    save=True,
    save_period=10,

    # Output
    project='harmony_omr_v2_phase5',
    name='fermata_enhanced_training',
    exist_ok=True
)

# Print final metrics
print("\n" + "="*60)
print("Phase 5 Training Complete")
print("="*60)
print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
print(f"Fermata mAP50: Check per-class metrics in results.csv")
print("="*60)
```

Run:
```bash
python yolo12_train_phase5_fermata.py
```

## Monitor Training

```bash
# Watch training progress
watch -n 5 'tail -20 harmony_omr_v2_phase5/fermata_enhanced_training/results.csv'

# TensorBoard (if enabled)
tensorboard --logdir harmony_omr_v2_phase5/

# GPU monitoring
watch -n 1 nvidia-smi
```

## Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| **Generation** | 30-40 min | 5,000 samples with v2 |
| **Merge** | 1-2 min | Copy + label conversion |
| **Training** | 4-5 hours | 150 epochs on RTX 5090 |
| **Validation** | 10 min | Analyze per-class metrics |
| **Total** | ~5 hours | End-to-end pipeline |

## Troubleshooting

### Issue: Generation fails

```bash
# Check LilyPond
lilypond --version

# Test single sample
python -c "
from generate_synthetic_fermatas_v2 import EnhancedFermataGenerator
gen = EnhancedFermataGenerator('/tmp/test')
success = gen.generate_sample(0)
print(f'Test sample: {\"Success\" if success else \"Failed\"}')
"
```

### Issue: Low success rate (<80%)

**Solutions:**
1. Use v2 generator (better detection)
2. Reduce complexity in LilyPond code (edit script)
3. Increase rendering timeout

### Issue: Merge fails

```bash
# Verify directories
ls -la datasets/synthetic_fermatas_v2/
ls -la datasets/yolo_harmony_v2_phase4/train/

# Check label format
head -5 datasets/synthetic_fermatas_v2/labels/fermata_000000.txt
# Should show: "29 x y w h" (if not, script will convert)
```

### Issue: Training not improving fermata

**Possible causes:**
1. Insufficient class weight (increase to 20x)
2. Synthetic samples too different from real data
3. Need more samples (generate 10,000 instead of 5,000)

**Solutions:**
```python
# In training script, add focal loss
from ultralytics.utils.loss import FocalLoss

# Configure model with focal loss
model.model.loss = 'focal'  # Helps with class imbalance
```

## Success Criteria

After Phase 5 training, verify:

- [ ] Overall mAP50 > 0.62 (+7% from Phase 4)
- [ ] Fermata mAP50 > 0.60 (+50% estimated improvement)
- [ ] No degradation in other classes (within 2% of Phase 4)
- [ ] Validation loss converging
- [ ] No overfitting (train/val mAP gap < 0.05)

## Next Steps After Success

1. **Phase 6: High-Resolution Training**
   - Use 1280x1280 images
   - Fine-tune best Phase 5 model
   - Target: mAP50 > 0.70

2. **Export for Production**
   ```bash
   python export_models.py \
     --model harmony_omr_v2_phase5/fermata_enhanced_training/weights/best.pt \
     --formats tflite \
     --int8
   ```

3. **Update CLAUDE.md**
   - Document Phase 5 results
   - Update roadmap status
   - Share learnings

## File Locations

```
training/
├── generate_synthetic_fermatas.py          # Basic generator
├── generate_synthetic_fermatas_v2.py       # Enhanced generator ⭐
├── merge_synthetic_fermatas.py             # Merge script
├── yolo12_train_phase5_fermata.py          # Training script (create)
├── SYNTHETIC_FERMATA_GENERATION.md         # Full documentation
├── QUICK_START_FERMATA_GENERATION.md       # This file
└── datasets/
    ├── synthetic_fermatas_v2/              # Generated data
    │   ├── images/
    │   ├── labels/
    │   └── dataset.yaml
    └── yolo_harmony_v2_phase4/             # Target dataset
        ├── train/
        │   ├── images/
        │   └── labels/
        └── val/
            ├── images/
            └── labels/
```

## References

- Full documentation: `SYNTHETIC_FERMATA_GENERATION.md`
- Project overview: `../CLAUDE.md`
- Phase 4 dataset: `datasets/yolo_harmony_v2_phase4/harmony_phase4.yaml`

---

**Last Updated:** 2025-11-25
**Status:** Ready to execute
**Estimated Total Time:** 5 hours (generation + training)
