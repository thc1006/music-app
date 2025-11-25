# Phase 4 Dataset Duplicate Label Analysis Report

**Date**: 2025-11-25
**Analyst**: Claude Code
**Dataset**: yolo_harmony_v2_phase4

---

## Executive Summary

The Phase 4 dataset contains **systematic duplicate labels** affecting approximately **15% of training images**, with an estimated **1.6 million duplicate annotations** across the entire dataset. Each affected label appears **exactly 2 consecutive times**, indicating a bug in the data generation pipeline rather than data merging issues.

### Impact Metrics

| Metric | Value |
|--------|-------|
| **Total Training Images** | 22,110 |
| **Affected Images** | ~3,317 (15.0%) |
| **Duplicate Labels (sampled)** | 14,415 / 36,333 (39.7%) |
| **Estimated Total Duplicates** | ~1,593,578 |
| **Storage Waste** | ~40% of label file size |
| **Training Impact** | Each duplicate counted twice in loss calculation |

---

## Root Cause Analysis

### 1. **Origin of Duplicates**

The duplicates were **NOT introduced during Phase 4 merging**. They exist in the source datasets:

```
Phase 1 (yolo_harmony_v2_optimized) → HAS DUPLICATES ❌
  ↓ (copied to)
Phase 2 (yolo_harmony_v2_phase2) → HAS DUPLICATES ❌
  ↓ (merged to)
Phase 3 (yolo_harmony_v2_phase3) → HAS DUPLICATES ❌
  ↓ (merged to)
Phase 4 (yolo_harmony_v2_phase4) → HAS DUPLICATES ❌
```

### 2. **Duplication Pattern**

**Key Findings**:
- ✅ **100% of duplicates are consecutive** (line N followed by line N+1)
- ✅ **Every duplicate appears exactly 2 times** (no 3x or higher)
- ✅ **Only affects files with `lg-` prefix and `-aug-` pattern**
- ❌ **MUSCIMA++ data**: Clean (0.1% duplicates)
- ❌ **Rebelo data**: Clean (0% duplicates)
- ❌ **External datasets**: Clean

**Example from `lg-687429493056531979-aug-beethoven--page-21.txt`**:
```
Line 1: 32 0.816582 0.117244 0.366837 0.078644
Line 2: 32 0.816582 0.117244 0.366837 0.078644  ← DUPLICATE
Line 3: 32 0.766837 0.779762 0.466327 0.440476
Line 4: 32 0.766837 0.779762 0.466327 0.440476  ← DUPLICATE
...
```

### 3. **Affected Data Sources**

| Source Prefix | Files Sampled | Duplicate Rate | Status |
|---------------|---------------|----------------|--------|
| **p3** (Phase 3) | 126 | **41.8%** | ❌ Affected |
| **muscima** | 3 | 0.1% | ✅ Clean |
| **rebelo** | 71 | 0.0% | ✅ Clean |

The `p3_` prefix files contain Phase 2 data (prefixed as `p2_` in Phase 3), which inherited duplicates from Phase 1.

### 4. **Root Cause: Data Generation Bug**

The duplicates originate from **Phase 1 data generation** (likely synthetic data from LilyPond or augmentation scripts). The pattern suggests:

1. **Hypothesis**: A data augmentation or synthetic generation script wrote each bounding box annotation **twice**
2. **Affected files**: Only files matching pattern `lg-*-aug-*` (LilyPond-generated, augmented)
3. **Likely culprit**: One of these scripts:
   - `phase3_lilypond_precise_synthesis.py`
   - `phase3_synthetic_pipeline.py`
   - `synthetic_data_generator.py`
   - Or the original Phase 1 LilyPond rendering pipeline

**Evidence**:
```python
# Pattern in affected files:
lg-687429493056531979-aug-beethoven--page-21.txt
lg-90621209-aug-beethoven-_oversample_31_36.txt
lg-101766503886095953-aug-gonville--page-1.txt
#  ↑           ↑
#  lg- prefix  -aug- pattern
```

---

## Quantitative Analysis

### Sample Statistics (200 training files)

```
Total files checked:     200
Files with duplicates:   30 (15.0%)
Total labels:            36,333
Unique labels:           21,918
Duplicate labels:        14,415 (39.7%)
```

### Extrapolation to Full Dataset

```
Estimated affected files:     3,317 / 22,110 (15%)
Estimated duplicate labels:   ~1,593,578
Average duplicates per file:  ~480 (when file is affected)
```

### Class Distribution Impact

Most common duplicated classes (in affected files):
```
Class 0 (notehead_filled):  ~4,800 duplicates
Class 2 (stem):             ~4,600 duplicates
Class 1 (notehead_hollow):  ~3,000 duplicates
Class 3 (beam):             ~1,800 duplicates
Class 32 (ledger_line):     ~1,200 duplicates
```

---

## Training Impact

### Performance Impact

1. **Loss Calculation**: Each duplicate annotation contributes to loss twice, effectively **doubling the weight** of these annotations
2. **Class Balance**: Already common classes (notehead, stem) become even more dominant
3. **Overfitting Risk**: Model may memorize duplicate patterns
4. **mAP Accuracy**: Validation metrics may be **inflated** if validation set also has duplicates

### Storage Impact

```
Label file size waste:    ~40% of affected files
Disk space wasted:        ~50-100 MB (label files only)
Memory during training:   ~40% more annotations loaded per batch
```

---

## Validation Set Analysis

**Sample Statistics (100 validation files)**:
```
Files with duplicates:   13 (13.0%)
Duplicate ratio:         38.1%
```

⚠️ **Critical**: Validation set is also affected, which means:
- mAP metrics may not reflect true performance
- Model evaluation is biased toward duplicate patterns

---

## Recommendations

### Immediate Actions (Priority 1)

1. **Create Deduplication Script**
   ```python
   # Pseudo-code
   for label_file in all_label_files:
       lines = read_lines(label_file)
       unique_lines = remove_consecutive_duplicates(lines)
       write_lines(label_file, unique_lines)
   ```

2. **Verify Deduplication**
   - Re-run analysis on cleaned dataset
   - Confirm 0% duplicate rate
   - Validate that images still match labels

3. **Re-generate Dataset Config**
   - Update class distribution statistics
   - Recalculate dataset size
   - Update CLAUDE.md with corrected numbers

### Before Next Training (Priority 2)

1. **Identify Original Bug**
   - Search for double-write loops in:
     - `synthetic_data_generator.py`
     - `phase3_lilypond_precise_synthesis.py`
     - Any script that generates `lg-*-aug-*` files

2. **Fix Data Generation Pipeline**
   - Prevent duplicates in future synthetic data
   - Add validation step after generation

3. **Regenerate Phase 1 Data** (Optional, if time permits)
   - Clean regeneration from scratch
   - Ensures no legacy bugs

### Code Quality (Priority 3)

1. **Add Dataset Validation**
   ```python
   def validate_dataset(dataset_path):
       """Check for duplicate labels, missing files, etc."""
       issues = []
       for label_file in glob(dataset_path / "**/*.txt"):
           if has_consecutive_duplicates(label_file):
               issues.append(f"Duplicates in {label_file}")
       return issues
   ```

2. **Pre-training Sanity Check**
   - Always run validation before training
   - Log dataset statistics (class counts, file counts, etc.)

---

## Detailed Code Analysis

### Merge Script Analysis (`merge_datasets_phase4.py`)

**Lines 40-87**: `copy_dataset()` function
```python
# This function is CORRECT ✅
# It simply copies files as-is, does NOT introduce duplicates
dst_img = dst_images / new_name
if not dst_img.exists():
    shutil.copy2(img_path, dst_img)  # Copy image

dst_lbl = dst_labels / new_label
if not dst_lbl.exists():
    shutil.copy2(label_path, dst_lbl)  # Copy label (preserves duplicates if present)
```

**Verdict**: Merge script is innocent. It propagates existing duplicates but doesn't create them.

### Phase 2 Oversampling (`yolo12_train_phase2_balanced.py`)

**Lines 209-216**: Oversampling loop
```python
# This code is CORRECT ✅
# It copies entire image+label pairs, preserving duplicates
for i in range(factor - 1):
    new_name = f"{img_name}_oversample_{cls}_{i}"
    shutil.copy2(src_img, dst_img)  # Copy image
    shutil.copy2(src_lbl, dst_lbl)  # Copy label
```

**Verdict**: Oversampling script is innocent. It amplifies existing issues but doesn't create duplicates.

---

## Deduplication Implementation

### Recommended Script

```python
#!/usr/bin/env python3
"""
Deduplicate consecutive duplicate labels in YOLO format.
Safe operation: only removes exact consecutive duplicates.
"""

from pathlib import Path
from collections import defaultdict

def deduplicate_label_file(label_path):
    """Remove consecutive duplicate lines from a label file."""
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        return 0

    # Remove consecutive duplicates
    unique_lines = [lines[0]]
    for line in lines[1:]:
        if line != unique_lines[-1]:
            unique_lines.append(line)

    duplicates_removed = len(lines) - len(unique_lines)

    if duplicates_removed > 0:
        with open(label_path, 'w') as f:
            f.writelines(unique_lines)

    return duplicates_removed

def main():
    dataset_path = Path("/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase4")

    total_files = 0
    total_duplicates = 0
    affected_files = 0

    for split in ['train', 'val']:
        label_dir = dataset_path / split / "labels"

        for label_file in label_dir.glob("*.txt"):
            total_files += 1
            removed = deduplicate_label_file(label_file)

            if removed > 0:
                affected_files += 1
                total_duplicates += removed

    print(f"Processed {total_files} files")
    print(f"Affected files: {affected_files}")
    print(f"Total duplicates removed: {total_duplicates:,}")

if __name__ == "__main__":
    main()
```

---

## Testing Plan

### Before Deduplication
1. Backup Phase 4 dataset
2. Count total labels: `wc -l train/labels/*.txt val/labels/*.txt`
3. Sample 10 files and manually verify duplicates

### After Deduplication
1. Re-count total labels (should be ~40% less in affected files)
2. Verify no consecutive duplicates remain
3. Check image-label correspondence
4. Validate YOLO can load dataset

### Training Validation
1. Run 10 epochs on cleaned dataset
2. Compare mAP with Phase 3 results
3. Monitor class-wise AP improvements

---

## Conclusion

The duplicate label problem is **systematic, severe, but easily fixable**. It originates from Phase 1 data generation (likely LilyPond synthetic pipeline) and affects **~15% of files** with **~1.6M duplicate annotations**.

**Immediate fix**: Run deduplication script (20-30 minutes)
**Long-term fix**: Identify and patch data generation bug
**Expected benefit**: More accurate training, better class balance, 40% storage savings

---

## Appendix: Sample Files with Duplicates

### High Duplicate Count Files
```
p3_p2_lg-90621209-aug-beethoven-_oversample_31_36.txt
  - Total: 1,657 labels
  - Unique: 896 labels
  - Duplicates: 761 (45.9%)

p3_p2_lg-754102968543342864-aug-emmentaler--page-2_oversample_24_7.txt
  - Total: 3,159 labels
  - Unique: 1,829 labels
  - Duplicates: 1,330 (42.1%)

p3_p2_lg-101766503886095953-aug-gonville--page-1.txt
  - Total: 621 labels
  - Unique: 315 labels
  - Duplicates: 306 (49.3%)
```

### Clean Files (for comparison)
```
muscima_muscima_CVC-MUSCIMA_W-01_N-19_D-ideal.txt
  - Total: 457 labels
  - Unique: 457 labels
  - Duplicates: 0 (0%)

rebelo_rebelo1_Barline_0000.txt
  - Total: 1 label
  - Unique: 1 label
  - Duplicates: 0 (0%)
```

---

**Report Generated**: 2025-11-25
**Next Steps**: Create and run deduplication script before Phase 4 training
