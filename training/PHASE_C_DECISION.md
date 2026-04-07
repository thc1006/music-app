# Phase C Decision (2026-04-07)

## TL;DR

**Phase C complete. Do NOT ship Phase 9 to Android yet, but for a different
reason than Phase B.** Phase B's blocker — 530 fake violations on
orchestral pages — is fully resolved. New blocker: we still cannot prove
pitch accuracy on real 4-part music because we have no Bach chorale ground
truth. Next: **Phase D — chorale ground truth + multi-clef pitch accuracy**.

## What changed since Phase B decision

| Sub-task | Status | Output |
|----------|--------|--------|
| C1: Pitch accuracy harness | ✅ | 8/8 GT MIDI exact match (treble only) |
| C2: Per-stave clef detection | ✅ | `clef_detector.py`, greedy 1-to-1 matching, 16 tests |
| C3: Filter false staves | ⏸️ deferred | Not blocking; C4 makes false staves harmless |
| C4: Layout-aware voice binding | ✅ | `voice_binder.py` dispatcher, 19 tests |
| C5: Measure-scoped accidentals | ✅ | `accidental_resolver.py`, 12 tests |
| C6: Re-validate real images | ✅ | `PHASE_C_VALIDATION_REPORT.md` |
| C7: Final decision | ✅ | this document |

## C6 numbers (the real test)

### Same 5 OpenScore images Phase B used (orchestral, 10–14 staves)

|              | Phase B | Phase C |
|--------------|--------:|--------:|
| Detections   |     878 |     878 |
| Noteheads    |     781 |     781 |
| Chords       |      99 |   **0** |
| Violations   |     530 |   **0** |
| v/c          |    5.35 |    0.00 |

Detection layer is byte-identical (Phase 9 unchanged). The 530→0 drop comes
from C4 layout dispatch correctly refusing to bind voices on out-of-scope
orchestral content. **The rule engine no longer hallucinates.**

### Bartók String Quartet 5 mvt 3 (5 real 4-stave images)

| Total | det | nh | chords | viol |
|-------|----:|---:|-------:|-----:|
|       | 749 | 334 |     32 |   62 |

- 1.94 v/c (vs 5.35 baseline = 2.76× lower)
- Rule mix: M1=30, V1=25, P1=7 (more balanced than Phase B's 54%/39%/6%)

This proves the pipeline still produces output on legitimate 4-part content.
But Bartók is atonal — many "violations" are deliberate, so this number is
**not** a quality score, only a sanity check.

## What Phase C actually fixed

1. **Layout dispatch (C4) is the load-bearing change.** The Phase B blocker
   was treating orchestral instrument lines as SATB voices. C4 now refuses
   to grade material the rule engine wasn't designed for — that's the right
   call, not a regression.

2. **Per-stave clef (C2) eliminates one entire pitch-error vector** — every
   staff now has its own clef instead of defaulting to treble.

3. **Accidental persistence (C5) eliminates "G♯ at beat 1, then naturals
   for the rest of the measure" type errors.** This was an unscored Phase B
   bug that would have shown up immediately on any real chorale.

## What Phase C did NOT fix (and what's still blocking deployment)

1. **No Bach chorale ground truth.** All test data is either:
   - Orchestral (correctly skipped now, but unscored)
   - Bartók atonal (sanity check only, can't validate rule outputs)
   - Synthetic (B6 tests, validated rule logic but not real OMR)

2. **Multi-clef pitch accuracy is unmeasured on real images.** C1 covers
   8 noteheads, all on one treble staff. We don't know if bass clef pitch
   estimation is correct on real images.

3. **Per-stave clef detection accuracy on real images is unmeasured.** C2
   tests are synthetic. The C2-Fix6 measurement showed beethoven-2
   violations went up because the OLD code had a silent duplicate-assignment
   bug — we don't know yet if NEW C2 is right on most real images.

4. **C3 (false staff filtering) is deferred.** False staves caused by text
   blocks or beam regions still exist; they no longer crash the pipeline
   because C4 returns `[]` on weird staff counts, but a text region that
   produces an extra "staff" turning a 4-stave quartet into a 5-stave
   detection would still cause C4 to skip the chord.

## Decision

### Recommended: Phase D — chorale ground truth before deployment

**Required for Android deployment**:

1. **D1**: Curate 3–5 Bach chorale page images
   - Source: OpenScore Lieder corpus, Music21 Bach corpus, or render from
     `.mxl` files we already have
   - Each image: 2-stave grand staff, treble + bass clef, 4-voice harmony

2. **D2**: Extend `pitch_ground_truth.json` to multi-clef
   - Hand-label 30+ noteheads per image, both clefs
   - Re-run `pitch_accuracy_harness.measure_pitch_accuracy()`
   - Target: ≥80% MIDI accuracy

3. **D3**: Manual rule audit on one chorale
   - Pick a chorale where music theory says "0 violations" (or known
     deliberate violations)
   - Compare predicted violations to that ground truth
   - Target: false-positive rate < 0.5 v/c on a clean chorale

4. **D4** (only if D1–D3 pass): TFLite INT8 export + Android port

### Not recommended (now or before D)

- ❌ Deploying Phase 9 to Android. Pitch correctness on real chorales is
  unmeasured. Users would see violations they can't verify.
- ❌ Re-training Phase 9. Detection is not the bottleneck — symbol assembly
  is. Re-training won't help.
- ❌ Going to Phase C3 (false staff filtering). It's not blocking; deal
  with it only if a real chorale image fails because of a false staff.

### What to do if no chorale GT can be curated

Fallback: ship Phase 9 to Android as a pure **symbol detector** (boxes only,
no harmony analysis), and iterate harmony analysis on-device with manual
test cases. This is the path Phase B's "Alternative path" section
described. Slower than Phase D but unblocks shipping if curation stalls.

## Quality gates

| Gate | Phase B status | Phase C status | Required for ship |
|------|---------------|---------------|-------------------|
| Detection works | ✅ | ✅ | ✅ |
| Pipeline runs end-to-end | ✅ | ✅ | ✅ |
| Rule engine fires on synthetic | ✅ | ✅ | ✅ |
| No false-positive flood on out-of-scope content | ❌ (5.35 v/c) | ✅ (0 v/c) | ✅ |
| Pitch accuracy on real treble staff | unmeasured | ✅ (8/8) | ✅ |
| Pitch accuracy on real bass staff | unmeasured | unmeasured | ❌ blocking |
| Rule false-positive rate on clean chorale | unmeasured | unmeasured | ❌ blocking |

## Status transition

- Phase A: **complete** (detection wiring)
- Phase B: **complete** (pipeline modules + synthetic validation)
- Phase C: **complete** (layout-aware assembly + measure accidentals)
- **Phase D: TODO** (chorale GT + multi-clef accuracy)
- Phase E (future): TFLite export + Android port

## Files produced in Phase C

```
training/
├── pitch_accuracy_harness.py      # C1
├── pitch_ground_truth.json        # C1
├── test_pitch_accuracy.py         # C1 tests
├── clef_detector.py               # C2
├── test_clef_detector.py          # C2 tests
├── voice_binder.py                # modified for C4
├── test_voice_binder.py           # updated for C4
├── test_voice_binder_layout.py    # C4 dispatcher tests
├── accidental_resolver.py         # C5
├── test_accidental_resolver.py    # C5 tests
├── downstream_eval.py             # integrates C2 + C5
├── c6_revalidate_phase_b.py       # C6 — Phase B image rerun
├── c6_revalidate_4part.py         # C6 — Bartók 4-stave run
├── phase_c_revalidation.json      # C6 output
├── phase_c_4part.json             # C6 output
├── PHASE_C_VALIDATION_REPORT.md   # C6 report
└── PHASE_C_DECISION.md            # this file (C7)
```

73 Phase C unit tests + 70 Phase B unit tests = **143 total, all passing**
(plus 1 skipped — pre-existing).
