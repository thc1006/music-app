# Synthetic Fermata Generation Guide

## Overview

This guide covers generating synthetic fermata samples using LilyPond to address the fermata class imbalance in the Phase 4 dataset.

**Current Status:**
- Phase 4 fermata count: 9,710 annotations (4.4% of total)
- Target: Add 5,000+ synthetic fermata samples
- Goal: Achieve mAP50 > 0.60 for fermata class

## Prerequisites

### System Requirements

1. **LilyPond** (version 2.24+)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install lilypond

   # Verify installation
   lilypond --version
   ```

2. **Python Packages**
   ```bash
   pip install Pillow numpy scipy
   ```

3. **Disk Space**
   - ~2-3 GB for 5,000 samples
   - Temporary space: ~5 GB during generation

## Generation Scripts

### Option 1: Basic Generator (`generate_synthetic_fermatas.py`)

Faster but less accurate fermata detection.

```bash
python generate_synthetic_fermatas.py \
  --count 5000 \
  --output datasets/synthetic_fermatas \
  --seed 42
```

**Features:**
- Image-based fermata detection
- 3-5 minutes per 100 samples
- ~80-85% success rate

### Option 2: Enhanced Generator (`generate_synthetic_fermatas_v2.py`) ⭐ Recommended

More accurate fermata detection using SVG output.

```bash
python generate_synthetic_fermatas_v2.py \
  --count 5000 \
  --output datasets/synthetic_fermatas_v2 \
  --seed 42
```

**Features:**
- SVG + image-based detection
- Multiple detection strategies
- 5-8 minutes per 100 samples
- ~90-95% success rate

## Parameters

### Generation Configuration

The scripts automatically vary:

| Parameter | Values | Description |
|-----------|--------|-------------|
| **Note values** | whole, half, quarter, eighth | Duration of notes with fermatas |
| **Clefs** | treble, bass, alto, tenor | Different clef types |
| **Positions** | note, rest, barline | Where fermatas appear |
| **Time signatures** | 4/4, 3/4, 2/4, 6/8, etc. | Measure organization |
| **Staff size** | 18-24 | Visual size variation |
| **Fermata count** | 1-3 per image | Multiple fermatas |
| **Accidentals** | 40-60% chance | Sharp/flat additions |
| **Dynamics** | 30-60% chance | Expression marks |
| **Background** | white, cream, textured | Visual variation |

### Augmentations Applied

1. **Brightness**: ±15%
2. **Contrast**: ±10%
3. **Gaussian blur**: 0-0.5px radius (30% chance)
4. **Background noise**: ±8 intensity (textured only)

## Output Format

### Directory Structure

```
datasets/synthetic_fermatas_v2/
├── images/
│   ├── fermata_000000.png
│   ├── fermata_000001.png
│   └── ...
├── labels/
│   ├── fermata_000000.txt
│   ├── fermata_000001.txt
│   └── ...
└── dataset.yaml
```

### Label Format (YOLO)

```
29 0.631007 0.084274 0.022883 0.030645
29 0.478261 0.085887 0.022883 0.027419
```

Format: `class_id x_center y_center width height` (normalized 0-1)

**Note:** Synthetic labels use class ID 0, which is remapped to 29 (fermata) during merge.

## Merging with Phase 4 Dataset

### Step 1: Generate Synthetic Data

```bash
cd /home/thc1006/dev/music-app/training

python generate_synthetic_fermatas_v2.py \
  --count 5000 \
  --output datasets/synthetic_fermatas_v2 \
  --seed 42
```

Expected output:
```
Progress: 5000/5000 (4750 successful, 250 failed)
Success rate: 95.0%
```

### Step 2: Review Samples

```bash
# View random samples
ls datasets/synthetic_fermatas_v2/images/ | head -20

# Check label counts
wc -l datasets/synthetic_fermatas_v2/labels/*.txt | tail -1
```

### Step 3: Merge into Phase 4

**Option A: Merge all into training set**

```bash
python merge_synthetic_fermatas.py \
  --synthetic datasets/synthetic_fermatas_v2 \
  --target datasets/yolo_harmony_v2_phase4 \
  --split train \
  --class-id 29
```

**Option B: Split then merge into both train/val**

```bash
python merge_synthetic_fermatas.py \
  --synthetic datasets/synthetic_fermatas_v2 \
  --target datasets/yolo_harmony_v2_phase4 \
  --split both \
  --class-id 29 \
  --split-synthetic \
  --train-ratio 0.9
```

### Step 4: Verify Merge

```bash
# Count total images
ls datasets/yolo_harmony_v2_phase4/train/images/ | wc -l

# Count fermata annotations
grep -r "^29 " datasets/yolo_harmony_v2_phase4/train/labels/ | wc -l
```

Expected after merge:
- Original: 9,710 fermatas
- After merge: ~14,000-14,500 fermatas (+45-50%)

## Training with Enhanced Dataset

### Update Training Script

Create `yolo12_train_phase5_fermata.py` with fermata-specific optimizations:

```python
# Class weights (emphasize fermata)
class_weights = {
    29: 15.0,  # fermata - 15x weight
    # ... other classes with standard weights
}

# Training hyperparameters
model.train(
    data='datasets/yolo_harmony_v2_phase4/harmony_phase4.yaml',
    epochs=150,
    batch=16,
    imgsz=640,
    device=0,

    # Fermata-specific augmentation
    mosaic=0.6,
    mixup=0.15,
    copy_paste=0.1,  # Enable for fermata

    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,

    # Validation
    val=True,
    plots=True,
    save_period=10
)
```

### Expected Improvements

| Metric | Phase 4 | Phase 5 Target |
|--------|---------|----------------|
| **Overall mAP50** | 0.580 | 0.62-0.65 |
| **Fermata mAP50** | ~0.40 | **0.60-0.70** |
| **Training time** | 4h | 4-5h |

## Troubleshooting

### Issue: Low Success Rate (<80%)

**Cause:** Fermata detection failing

**Solutions:**
1. Use v2 generator (enhanced detection)
2. Reduce `fermata_count` parameter (edit script)
3. Increase staff_size minimum (clearer rendering)

### Issue: Labels Missing or Incorrect

**Cause:** Detection algorithm false positives

**Solutions:**
1. Manual review: `python review_samples.py` (create if needed)
2. Filter by bbox size: remove too large/small boxes
3. Use stricter detection thresholds

### Issue: LilyPond Timeouts

**Cause:** Complex scores taking too long

**Solutions:**
1. Increase timeout in script (line ~310): `timeout=60`
2. Reduce notes per measure
3. Simplify time signatures

### Issue: Merge Failed

**Cause:** Directory structure mismatch

**Solutions:**
```bash
# Verify structure
tree -L 2 datasets/synthetic_fermatas_v2/
tree -L 2 datasets/yolo_harmony_v2_phase4/

# Fix if needed
mkdir -p datasets/yolo_harmony_v2_phase4/train/{images,labels}
```

## Performance Tips

### Parallel Generation

Generate multiple batches in parallel:

```bash
# Terminal 1
python generate_synthetic_fermatas_v2.py --count 2500 --output datasets/syn_fermata_1 --seed 42

# Terminal 2
python generate_synthetic_fermatas_v2.py --count 2500 --output datasets/syn_fermata_2 --seed 123

# Then merge both
python merge_synthetic_fermatas.py --synthetic datasets/syn_fermata_1 --target datasets/yolo_harmony_v2_phase4 --split train
python merge_synthetic_fermatas.py --synthetic datasets/syn_fermata_2 --target datasets/yolo_harmony_v2_phase4 --split train
```

### Faster Generation

Reduce quality for more speed:

```python
# Edit script, line ~310
'-dresolution=200',  # Instead of 300
```

### Storage Optimization

Compress after generation:

```bash
# PNG optimization
cd datasets/synthetic_fermatas_v2/images/
optipng *.png

# Or use lossy compression
mogrify -quality 95 *.png
```

## Quality Control

### Manual Review

Sample 100 random images:

```bash
cd datasets/synthetic_fermatas_v2/images/
ls | shuf -n 100 > /tmp/review_samples.txt

# Open with image viewer
cat /tmp/review_samples.txt | xargs -I {} xdg-open {}
```

### Automated Validation

Create `validate_synthetic.py`:

```python
#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

def validate_sample(img_path, label_path):
    """Check if fermata bbox overlaps with actual content."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    with open(label_path) as f:
        for line in f:
            _, x, y, bw, bh = map(float, line.split())

            # Convert to pixel coords
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            # Extract region
            region = img[y1:y2, x1:x2]

            # Check if region has content
            if region.size == 0:
                return False, "Empty bbox"

            dark_pixels = (region < 200).sum()
            dark_ratio = dark_pixels / region.size

            if dark_ratio < 0.05:
                return False, f"Too few dark pixels: {dark_ratio:.2%}"

    return True, "OK"

# Run validation
images_dir = Path('datasets/synthetic_fermatas_v2/images')
labels_dir = Path('datasets/synthetic_fermatas_v2/labels')

valid = 0
invalid = 0

for img_file in images_dir.glob('*.png'):
    label_file = labels_dir / f"{img_file.stem}.txt"
    is_valid, msg = validate_sample(img_file, label_file)

    if is_valid:
        valid += 1
    else:
        invalid += 1
        print(f"Invalid: {img_file.name} - {msg}")

print(f"\nValidation: {valid} valid, {invalid} invalid ({100*valid/(valid+invalid):.1f}%)")
```

Run:
```bash
python validate_synthetic.py
```

## Advanced Customization

### Add More Fermata Variants

Edit `generate_synthetic_fermatas_v2.py`:

```python
# Line ~150
POSITIONS = ['note', 'rest', 'barline', 'chord']  # Add 'chord'

def _generate_chord_with_fermata(self, config):
    """Generate a chord with fermata."""
    pitches = random.sample(self.PITCHES, 3)
    octaves = random.choices(self.OCTAVES, k=3)

    chord = ' '.join(f"{p}{o}" for p, o in zip(pitches, octaves))
    return f"<{chord}>{config.note_value}\\fermata"
```

### Custom Staff Spacing

```python
# Line ~180, edit staff_size calculation
staff_space_factor = (config.staff_size - 16) / 3  # Wider range
```

### Different Fermata Styles

LilyPond supports multiple fermata types:

```python
FERMATA_TYPES = [
    '\\fermata',           # Normal
    '\\shortfermata',      # Short
    '\\longfermata',       # Long
    '\\verylongfermata',   # Very long
]

# Use in generation
fermata_type = random.choice(self.FERMATA_TYPES)
note = f"{pitch}{octave}{config.note_value}{fermata_type}"
```

## Integration Checklist

- [ ] LilyPond installed and tested
- [ ] Python dependencies installed
- [ ] Generated 5,000+ synthetic fermata samples
- [ ] Reviewed sample quality (>90% success rate)
- [ ] Merged into Phase 4 dataset
- [ ] Verified fermata annotation count increased
- [ ] Updated training script with fermata class weight
- [ ] Created Phase 5 training configuration
- [ ] Started training with merged dataset
- [ ] Monitored fermata mAP50 improvement

## References

- LilyPond documentation: http://lilypond.org/doc/
- YOLO label format: https://docs.ultralytics.com/datasets/
- Phase 4 dataset: `/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase4/`
- CLAUDE.md roadmap: Phase 4 → Phase 5 transition

## Maintenance

### Regenerate After Updates

If LilyPond or detection logic is updated:

```bash
# Backup old dataset
mv datasets/synthetic_fermatas_v2 datasets/synthetic_fermatas_v2.backup

# Regenerate
python generate_synthetic_fermatas_v2.py --count 5000 --output datasets/synthetic_fermatas_v2

# Compare quality
python compare_datasets.py datasets/synthetic_fermatas_v2.backup datasets/synthetic_fermatas_v2
```

### Version Control

```bash
# Add scripts only (not datasets)
git add generate_synthetic_fermatas*.py merge_synthetic_fermatas.py
git add SYNTHETIC_FERMATA_GENERATION.md
git commit -m "feat: Synthetic fermata generation pipeline"

# .gitignore should exclude:
datasets/synthetic_fermatas*/
```

---

**Last Updated:** 2025-11-25
**Author:** Claude Code
**Status:** Ready for Phase 5 training
