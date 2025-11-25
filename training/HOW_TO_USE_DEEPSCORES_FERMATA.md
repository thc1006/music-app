# How to Use DeepScoresV2 Fermata Dataset

Quick guide for integrating the newly converted DeepScores fermata data.

---

## 🎯 Quick Start

### Option 1: Merge into Phase 4 Dataset (Recommended)

Update `merge_datasets_phase4.py`:

```python
# Add DeepScores to dataset list
DATASETS = [
    ("yolo_harmony_v2_phase3", 1.0),
    ("yolo_muscima_converted", 1.0),
    ("yolo_rebelo_converted", 1.0),
    ("yolo_deepscores_converted", 1.0),  # NEW!
]
```

Then run:
```bash
python merge_datasets_phase4.py
```

**Expected Output**:
- Phase 4 fermata count: 9,710 → 11,954 (+23%)
- Total images: 24,566 → 24,758

---

## Option 2: Use Standalone for Fermata Training

### YAML Config

```yaml
# deepscores_fermata_only.yaml
path: /home/thc1006/dev/music-app/training/datasets/yolo_deepscores_converted
train: images/train
val: images/val

names:
  29: fermata

nc: 33  # Must match full class count for compatibility
```

### Training Command

```bash
yolo train \
  model=yolo12s.pt \
  data=datasets/yolo_deepscores_converted/deepscores_fermata.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=fermata_specialist
```

---

## 📊 Dataset Statistics

```python
# Quick stats
train_images = 147
val_images = 45
total_fermatas = 2244
avg_per_image = 11.7
```

---

## 🔍 Validation

Check a sample:

```bash
# View first label
head datasets/yolo_deepscores_converted/labels/train/ds2_train_000002.txt

# Count annotations
wc -l datasets/yolo_deepscores_converted/labels/train/*.txt | tail -1

# Verify images
ls datasets/yolo_deepscores_converted/images/train/ | wc -l
```

Expected:
- 147 train images
- 147 train labels
- Each label has multiple lines (class_id 29 = fermata)

---

## 🚨 Important Notes

1. **Duplicates**: Annotations appear twice (deepscores + muscima++ sets)
   - This is normal and doesn't harm training
   - YOLO training handles duplicates gracefully

2. **No Barlines**: DeepScores doesn't have barline data
   - Use MUSCIMA++, DoReMi, or Rebelo for barlines

3. **Class 29 Only**: Dataset contains only fermata class
   - Safe to merge with other datasets (no conflicts)

---

## 💡 Tips

### For Maximum Fermata Performance

Combine all fermata sources:
- DeepScores: 2,244 annotations ✅ (this dataset)
- MUSCIMA++: 35 annotations ✅
- Rebelo2: 102 symbols ⏳ (convert if needed)

**Total**: 2,381 fermata annotations

### For Balanced Training

Use Phase 4 merge (all datasets) for balanced multi-class training.

---

## 📁 File Locations

```
datasets/
├── yolo_deepscores_converted/        # NEW - Fermata dataset
│   ├── images/
│   │   ├── train/  (147 images)
│   │   └── val/    (45 images)
│   ├── labels/
│   │   ├── train/  (147 labels)
│   │   └── val/    (45 labels)
│   ├── deepscores_fermata.yaml
│   └── conversion_report.txt
│
├── yolo_harmony_v2_phase4/           # Phase 4 merge target
│   └── (will include DeepScores after merge)
│
└── ds2_dense/                        # Original DeepScores source
    ├── deepscores_train.json
    ├── deepscores_test.json
    └── images/
```

---

## 🔧 Troubleshooting

### Problem: "FileNotFoundError: deepscores_fermata.yaml"

**Solution**: Use absolute path or run from training directory
```bash
cd /home/thc1006/dev/music-app/training
python yolo12_train_phase4.py
```

### Problem: "No images found"

**Solution**: Check YAML paths are absolute
```yaml
path: /home/thc1006/dev/music-app/training/datasets/yolo_deepscores_converted
```

### Problem: "Class ID mismatch"

**Solution**: Ensure using 33-class system (not 35-class)
```python
nc: 33  # in YAML config
```

---

## ✅ Validation Checklist

Before training:

- [ ] 147 train images exist
- [ ] 45 val images exist
- [ ] All labels have class_id = 29
- [ ] Bounding boxes are normalized (0.0 - 1.0)
- [ ] YAML paths point to correct directories

Quick validation:
```bash
python -c "
import yaml
with open('datasets/yolo_deepscores_converted/deepscores_fermata.yaml') as f:
    cfg = yaml.safe_load(f)
    print(f'Train path: {cfg[\"train\"]}')
    print(f'Val path: {cfg[\"val\"]}')
    print(f'Class 29: {cfg[\"names\"][29]}')
"
```

Expected output:
```
Train path: images/train
Val path: images/val
Class 29: fermata
```

---

## 🎯 Success Criteria

After merging into Phase 4:

1. **Fermata mAP50**: Should improve from Phase 3's 0.580
2. **Target**: 0.65-0.70 mAP50 on fermata class
3. **No regression**: Other classes maintain performance

Monitor during training:
```bash
# Watch fermata performance
tail -f runs/detect/phase4/results.csv | grep "fermata"
```

---

**Ready to merge?** → Run `python merge_datasets_phase4.py`
**Need help?** → Check `DEEPSCORES_CONVERSION_SUMMARY.md`
