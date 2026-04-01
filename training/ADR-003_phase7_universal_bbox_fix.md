# ADR-003: Phase 7 — Universal OpenScore Bbox Fix for All 30 Classes

**Status**: Approved
**Date**: 2026-02-27
**Author**: Claude + User
**Supersedes**: ADR-001 (notehead-only fix), extends ADR-002 (Phase 6 strategy)

---

## Context

Phase 6 fixed only **2/33 classes** (notehead_filled, notehead_hollow) and removed stems.
Deep analysis on 2026-02-27 revealed:

- **32/33 classes** have oversized OpenScore glyph-group bboxes (6x to 7746x area ratio)
- NMS@0.7 ceiling for unfixed classes: **57.1%** (42.9% annotations killed)
- After applying universal rules: **99.6%** ceiling (+42.5pp)

## Root Cause

LilyPond renders music as **glyph groups** — compound objects where a single bounding box
encompasses ALL graphical elements of a musical event (notehead + stem + beam + flag + dots +
accidentals + ledger lines). OpenScore annotations inherit these glyph-group bboxes.

### Glyph-Group Types (Empirically Verified)

| Type | Members | Evidence (bbox sharing %) |
|------|---------|--------------------------|
| **Note Column** | notehead, beam, flag, augdot, ledger_line | 37-80% sharing with notehead |
| **System Start** | clef + barline_repeat | 100% sharing (clef↔barline_repeat) |
| **Key Signature** | key_sig + accidentals | 100% sharing (key_sig↔accidentals) |
| **System End** | barline_final + tie | 71% sharing |
| **Standalone** | rests, time_sig, fermata, dynamics, barlines | <2% sharing |

## Decision: Two Universal Position Rules

### Rule 1 — TOP Edge (for glyph-group members)

```
new_cy = bbox_top + ref_h / 2
new_w  = ref_w
new_h  = ref_h
```

**Applies to** (NMS ceiling result):
| Class | Before | After | Δ |
|-------|--------|-------|---|
| ledger_line | 37.3% | **100.0%** | +62.7pp |
| beam | 53.3% | **99.9%** | +46.6pp |
| accidental_flat | 71.0% | **100.0%** | +29.0pp |
| accidental_sharp | 73.8% | **100.0%** | +26.2pp |
| tie | 83.9% | **100.0%** | +16.1pp |
| accidental_natural | 83.9% | **100.0%** | +16.1pp |
| augmentation_dot | 74.1% | **100.0%** | +25.9pp |
| flag_8th | 78.1% | **99.9%** | +21.8pp |
| fermata | 72.1% | **98.4%** | +26.3pp |

**Why TOP works universally**: In LilyPond glyph groups, the primary symbol (notehead,
beam endpoint, accidental) is always at the topmost position. The bbox extends downward
to encompass stems, flags, and other dependent elements.

### Rule 2 — CENTER (for standalone symbols)

```
new_w = ref_w
new_h = ref_h
# cx, cy unchanged
```

**Applies to** (NMS ceiling result):
| Class | Before | After | Δ |
|-------|--------|-------|---|
| time_signature | 42.8% | **99.8%** | +57.0pp |
| rest_whole | 70.2% | **100.0%** | +29.8pp |
| rest_8th | 75.0% | **100.0%** | +25.0pp |
| rest_16th | 77.0% | **100.0%** | +23.0pp |
| rest_quarter | 78.5% | **100.0%** | +21.5pp |

**Why CENTER works**: Standalone symbols own their entire glyph-group bbox.
The symbol is approximately centered within it. Shrinking to reference size
while keeping the center point preserves the correct position.

### Rule 3 — Barlines (special: keep height, shrink width)

```
new_w = ref_w
# h, cx, cy unchanged
```

Barlines span full staff height. Only the width is oversized.

### Rule 4 — Key Signature (keep as-is OR special processing)

Key_signature bbox IS the actual key signature area (containing multiple accidentals).
Individual accidentals within key_sig share the key_sig bbox — their true positions
need music-theory inference (which sharp/flat at which staff position).

**Decision**: For Phase 7, apply TOP rule to accidentals (already validated at 100% NMS ceiling).
Key_signature class itself: keep original bbox (it represents the whole key sig area).

## Reference Sizes (DoReMi Median, Normalized)

| Class | ref_w | ref_h | DoReMi n |
|-------|-------|-------|----------|
| notehead_filled | 0.01180 | 0.00908 | 15,204 |
| notehead_hollow | 0.01180 | 0.00908 | 299 |
| beam | 0.04150 | 0.00742 | 3,512 |
| flag_8th | 0.00889 | 0.01969 | 430 |
| flag_16th | 0.00970 | 0.01941 | 1,728 |
| flag_32nd | 0.00889 | 0.02312 | 610 |
| augmentation_dot | 0.00364 | 0.00257 | 423 |
| tie | 0.03475 | 0.00628 | 615 |
| accidental_sharp | 0.00848 | 0.01684 | 998 |
| accidental_flat | 0.00768 | 0.01484 | 775 |
| accidental_natural | 0.00566 | 0.01627 | 684 |
| accidental_double_sharp | 0.00848 | 0.00599 | 94 |
| accidental_double_flat | 0.01601 | 0.06861 | 7,088 |
| rest_whole | 0.00970 | 0.00343 | 49 |
| rest_half | 0.00970 | 0.00343 | 15 |
| rest_quarter | 0.00929 | 0.01798 | 254 |
| rest_8th | 0.00848 | 0.01027 | 1,388 |
| rest_16th | 0.01091 | 0.01627 | 3,531 |
| clef_treble | 0.02263 | 0.04195 | 751 |
| clef_bass | 0.02343 | 0.02140 | 190 |
| barline | 0.01500 | — | 3,300 |
| barline_double | 0.01000 | — | 2,372 |
| barline_final | 0.01000 | — | 623 |
| barline_repeat | 0.01562 | 0.13281 | 103 |
| time_signature | 0.01455 | 0.01256 | 194 |
| fermata | 0.02000 | 0.02000 | 2,762 |
| dynamic_soft | 0.01535 | 0.00999 | 49 |
| dynamic_loud | 0.01944 | 0.40745 | 2,776 |
| ledger_line | 0.01205 | 0.00513 | 476 |

## Classes with Insufficient DoReMi Reference

| Class | Issue | Proposed Solution |
|-------|-------|-------------------|
| clef_alto | DoReMi ref 0.90×0.90 (bad) | Use manual measurement |
| clef_tenor | 0 DoReMi samples | Use clef_alto reference |
| rest_half | Only 15 DoReMi samples | Validate manually, use rest_whole ref |
| dynamic_soft | Only 49 DoReMi samples | Validate manually |
| dynamic_loud | h=0.40745 suspicious | Needs manual validation |

## Implementation Plan

### Phase 7a: Fix ALL classes (extend create_phase6_fixed_bbox.py)

```python
# Universal fix algorithm
RULES = {
    # TOP edge rule (glyph-group members)
    "top": [0,1,3,4,5,6,7,8,13,14,15,16,17,29,32],  # nh,beam,flag,augdot,tie,acc,fermata,ledger
    # CENTER rule (standalone)
    "center": [18,19,20,21,22,27,30],  # rests, time_sig, dynamic_soft
    # BARLINE rule (keep height, shrink width)
    "barline": [23,24,25,26],
    # CLEF rule (keep height, shrink width to ref)
    "clef": [9,10,11,12],
    # KEEP as-is
    "keep": [28],  # key_signature
}
```

### Phase 7b: Retrain with surgical cv2 reset (same as Phase 6)

Same ADR-002 strategy: reset cv2 bbox regression head, two-stage training.

### Success Criteria

| Metric | Phase 6 (noteheads only) | Phase 7 Target |
|--------|-------------------------|----------------|
| NMS@0.7 ceiling (all classes) | ~65% | **>95%** |
| mAP50 (strict eval) | 0.813 (Stage 2 ep7) | **>0.87** |
| notehead_filled Recall | ~0.50 | **>0.65** |
| beam mAP50 | TBD | **>0.75** |
| ledger_line mAP50 | TBD | **>0.60** |

## Risks

1. **DoReMi reference sizes may not be perfectly representative** — mitigation: validate
   with visual inspection on 20+ images before full dataset generation
2. **Accidental positions within key signatures need music-theory inference** — mitigation:
   Phase 7a uses simple TOP rule (100% NMS ceiling), Phase 7b can add refined positioning
3. **Some classes have very few DoReMi samples** — mitigation: manual measurement + validation

## Timeline

| Step | Duration | Dependencies |
|------|----------|-------------|
| Write create_phase7_dataset.py | 2-3 hours | ADR-003 approved |
| Validate on 20+ images visually | 1 hour | Dataset script |
| Generate full dataset | 30 min | Validation passed |
| NMS ceiling validation (all classes) | 30 min | Dataset generated |
| Train Phase 7 (120 epochs + cv2 reset) | 20-24 hours | GPU available |
| Evaluate | 1 hour | Training complete |

**Total: ~26-30 hours (mostly GPU training)**
