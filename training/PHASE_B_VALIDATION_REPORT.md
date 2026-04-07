# Phase B Validation Report

**Date**: 2026-04-07
**Scope**: End-to-end Python pipeline validation (Phase 9 → harmony_rules.py)

## Summary

Full Phase B pipeline (B1-B6) implemented with strict TDD. 60+ unit tests pass.
End-to-end run on 5 real OpenScore images produces parseable harmony analysis,
but the violation counts are far too high (avg 5.35 per chord), indicating
pitch estimation is not yet accurate enough for production.

## Test Results (B1-B6 unit tests)

| Module | Tests | Status |
|--------|-------|--------|
| staff_detector | 13 | ✅ |
| pitch_estimator | 15 | ✅ |
| voice_binder | 9 | ✅ |
| measure_detector | 11 | ✅ |
| downstream_pipeline | 12 | ✅ |
| synthetic_chorales | 10 | ✅ |
| **Total** | **70** | **✅** |

## B7 Real Image Validation (5 OpenScore images)

| Image | Detect | Noteheads | Chords | Violations | Violations/Chord |
|-------|--------|-----------|--------|------------|------------------|
| Beethoven p2 | 203 | 177 | 22 | 127 | 5.8 |
| Gonville p1 | 142 | 129 | 15 | 75 | 5.0 |
| Gutenberg p1 | 154 | 133 | 18 | 99 | 5.5 |
| Emmentaler p1 | 236 | 219 | 29 | 171 | 5.9 |
| Beethoven p2' | 143 | 123 | 15 | 58 | 3.9 |
| **Total** | **878** | **781** | **99** | **530** | **5.35** |

### Violation breakdown (rule types)

- **M1** (melodic interval): 286 (54%) — most common
- **V1** (voice crossing): 205 (39%)
- **P1** (parallel 5th/8ve): 30 (6%)
- **P2** (hidden interval): 9 (2%)

## Interpretation

**~5 violations per chord is way too high** — real harmony music has <0.1
violations per chord on average. This indicates:

### Primary issue: pitch estimation is inaccurate

The pipeline correctly detects noteheads (781 of them in 5 images), segments
into measures (99 chords), and runs the rule engine end-to-end. But the
MIDI pitches are mostly wrong because:

1. **Clef detection is per-staff not per-stave-system**: multi-staff images
   may have multiple clefs (treble+bass) but our code picks one per staff
2. **Staff detection finds "lines" that aren't staves**: text regions,
   dynamic markings, or beam groups may produce false staff detections
3. **Voice binding assumes SATB order** — in orchestral scores, each staff
   is a different instrument, not a voice; the top-staff=SA, bottom=TB
   heuristic fails
4. **Accidentals are not properly scoped** to a measure (they can apply
   to subsequent notes of the same pitch class)

### Secondary issue: M1 and V1 are spike-prone

- M1 (melodic intervals) fires on large jumps → if pitch estimation is wrong,
  consecutive "wrong pitches" create fake huge jumps → spurious M1 violations
- V1 (voice crossing) fires when SATB ordering is violated → wrong pitch
  assignment creates fake crossings

**Both M1 and V1 are second-order errors caused by wrong pitches.**

## Pipeline Works, Pitches Don't

The GOOD news:
- Detection layer (Phase 9) works: 781 noteheads on 5 images
- Assembly layer works: staves detected, measures segmented, voices bound
- Rule engine works: rules fire on synthetic test cases
- Pipeline is end-to-end wireable

The BAD news:
- Pitch estimation accuracy is probably <30% on these images
- Until pitch estimation is accurate, violation counts are meaningless

## Recommendations (for B8 decision)

1. **Don't deploy Phase 9 to Android yet** — downstream task output is not
   reliable enough. Users would see hundreds of fake violations.

2. **Focus next effort on pitch accuracy**:
   - Detect clef PER STAVE (not per multi-staff system)
   - Handle grand staves (treble+bass as pair)
   - Improve staff line detection (filter out text/beam regions)
   - Scope accidentals to measure + pitch-class

3. **Build a small ground-truth test set**:
   - Pick ONE simple 4-voice chorale image with KNOWN correct pitches
   - Measure: what % of CV-detected noteheads get correct MIDI?
   - Iterate pitch estimation until accuracy ≥80%

4. **Alternatively, evaluate on simpler content first**:
   - Single-staff monophonic melody (trivial voice assignment)
   - Test pitch accuracy in isolation
   - Then move to 4-voice

## Files Produced (Phase B)

- `training/staff_detector.py` — staff line detection
- `training/pitch_estimator.py` — MIDI pitch from position
- `training/voice_binder.py` — SATB voice binding
- `training/measure_detector.py` — measure segmentation
- `training/downstream_eval.py` — updated integrated pipeline
- `training/test_staff_detector.py` — 13 tests
- `training/test_pitch_estimator.py` — 15 tests
- `training/test_voice_binder.py` — 9 tests
- `training/test_measure_detector.py` — 11 tests
- `training/test_synthetic_chorales.py` — 10 tests
- `training/test_downstream_pipeline.py` — 12 tests

## Phase B Conclusion

✅ **Pipeline architecture validated** — Phase 9 → rules engine wiring works
❌ **Pitch estimation not yet deployment-ready** — needs per-stave clef detection
→ **Next: Phase C** — improve pitch estimation with gold-standard measurement

Do NOT proceed to Android integration until pitch accuracy ≥80% on a known
test case.
