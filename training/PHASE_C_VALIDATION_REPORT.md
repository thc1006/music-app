# Phase C Validation Report

**Date**: 2026-04-07
**Scope**: End-to-end Python pipeline re-validation after Phase C interventions
**Predecessor**: `PHASE_B_VALIDATION_REPORT.md` (5.35 violations/chord on real images)

## Summary

Phase C delivers three independent fixes (C2/C4/C5) that collectively eliminate
the entire 530-violation false-positive load Phase B reported on orchestral
images, while preserving meaningful chord output on legitimate 4-stave content.
The change is **not** an improvement in pitch accuracy on orchestral material —
it is the correct architectural decision to **decline to evaluate** material the
rule engine was never designed for.

## Phase C interventions (recap)

| ID | Module | Change | Tests |
|----|--------|--------|-------|
| C1 | `pitch_accuracy_harness.py` | Quantitative MIDI accuracy harness | 8 GT noteheads, 100% baseline |
| C2 | `clef_detector.py` | Per-stave clef assignment, greedy 1-to-1 matching | 16 |
| C4 | `voice_binder.py` | Layout dispatch: 2-stave grand staff / 4-stave quartet / else skip | 19 |
| C5 | `accidental_resolver.py` | In-measure accidental persistence per (staff, step) | 12 |

C3 (false-staff filtering) was deferred — the false-staff problem on text/dynamic
markings is a real issue but not blocking the rule pipeline now that C4 refuses to
bind voices on out-of-scope layouts.

## C6 Test 1: same 5 OpenScore images Phase B used

These are LilyPond engraver-augmented orchestral pages (10-14 staves each).

| Image | Stv | B-Det | C-Det | B-Nh | C-Nh | B-Ch | C-Ch | B-V | C-V |
|-------|----:|------:|------:|-----:|-----:|-----:|-----:|----:|----:|
| Beethoven p2  | 12 | 203 | 203 | 177 | 177 | 22 | **0** | 127 | **0** |
| Gonville p1   | 12 | 142 | 142 | 129 | 129 | 15 | **0** |  75 | **0** |
| Gutenberg p1  | 12 | 154 | 154 | 133 | 133 | 18 | **0** |  99 | **0** |
| Emmentaler p1 | 14 | 236 | 236 | 219 | 219 | 29 | **0** | 171 | **0** |
| Beethoven p2' | 10 | 143 | 143 | 123 | 123 | 15 | **0** |  58 | **0** |
| **TOTAL**     | 60 | 878 | 878 | 781 | 781 | 99 | **0** | 530 | **0** |

- Detection layer (Phase 9) is **byte-identical** between Phase B and Phase C —
  C2/C4/C5 are downstream-only.
- Chord count: 99 → 0. Violations: 530 → 0.
- Cause: every image has 10–14 staves; C4 layout dispatcher returns `[]`
  because the rule engine targets 2-stave grand-staff or 4-stave quartet
  4-part harmony, not orchestral instrumentation.

This is the **correct** behavior. Phase B was producing 5.35 violations/chord
because it was treating orchestral instrument lines as SATB voices, which
is musically nonsensical. Refusing to grade out-of-scope content is better
than hallucinating thousands of fake errors.

## C6 Test 2: legitimate 4-stave content (Bartók String Quartet 5 mvt 3)

| # | Stv | Det | Nh | Chord | Viol |
|---|----:|----:|---:|------:|-----:|
| 1 |  4 | 102 |  42 |  0 |  0 |
| 2 |  4 | 193 |  79 |  6 | 16 |
| 3 |  4 | 178 |  92 | 11 | 30 |
| 4 |  4 | 186 |  85 | 15 | 16 |
| 5 |  4 |  90 |  36 |  0 |  0 |
| **T**| 20 | 749 | 334 | **32** | **62** |

- Violations/chord = **1.94**
- Rule breakdown: M1=30, V1=25, P1=7

For comparison: Phase B reported **5.35 v/c** on orchestral with M1=54%,
V1=39%, P1=6%. Phase C on real 4-stave content shows **2.76× lower** v/c
density and a markedly different rule distribution.

**Caveat (important)**: Bartók is atonal and uses parallel motion deliberately.
The 1.94 v/c on Bartók is **not** a clean evaluation of pitch correctness —
it's a sanity check that the pipeline produces output. The "real" downstream
metric still requires Bach-chorale-style 4-part test data, which we don't
have curated yet.

## Detection layer is unchanged

| Total over 5 Phase B images | Phase B | Phase C |
|-----------------------------|--------:|--------:|
| Detections                  |     878 |     878 |
| Noteheads                   |     781 |     781 |

Phase 9 `best.pt` is the same model. The 100% improvement in violations
comes entirely from the assembly layer.

## Test status

| Suite | Tests | Pass |
|-------|------:|-----:|
| pitch_accuracy   | (C1)  |  8/8 GT MIDI exact match |
| clef_detector    | 16    | ✅ |
| voice_binder     | 9     | ✅ |
| voice_binder_layout | 10 | ✅ |
| accidental_resolver | 12 | ✅ |
| downstream_pipeline | 12 | ✅ |
| **C-suite total** | **73 + 1 skip** | **✅** |

Plus the Phase B suites (staff_detector, pitch_estimator, measure_detector,
synthetic_chorales) continue to pass — full system is at 141 + 1 skip.

## What Phase C does NOT prove

1. **It does not prove pitch accuracy on real images.** C1 measures pitch
   on 8 hand-curated noteheads from one staff, all treble. Phase C did not
   add a multi-clef ground-truth set on real orchestral / quartet content.
   On the Bartók run, MIDI values are not validated against any ground truth.

2. **It does not prove the rule engine is correct on Bartók.** Bartók
   intentionally uses parallel motion. The 1.94 v/c is a sanity check that
   the pipeline runs; it is not a quality score.

3. **It does not improve detection.** Phase 9 weights are unchanged. Notehead
   recall on real OpenScore is still ~45% by center-match against the broken
   Phase 6 GT (see MEMORY.md).

## What Phase C does prove

1. **The rule engine no longer hallucinates 530 violations on orchestral
   pages**, because C4 refuses to bind voices on incompatible layouts.

2. **Per-stave clef detection works** — C2 unit tests verify uniqueness and
   greedy matching on synthetic detections, and the C2-Fix6 measurement
   showed Beethoven-2 violations went from 58 → 73 only because the OLD code
   had a silent bug that masked downstream errors.

3. **In-measure accidental persistence works** — C5 tests cover sharp/flat
   propagation, natural cancellation, pitch-class scoping, and cross-measure
   reset.

4. **The pipeline produces output on real 4-stave content** — Bartók run
   shows 32 chords / 62 violations across 5 images, with realistic rule
   distribution (no degenerate "all M1" pattern).

## Decision (C7 input)

**Recommended**: do NOT deploy Phase 9 to Android yet. C4 prevents fake
violations, but we still cannot confidently report on real 4-part music
because:

- No Bach chorale test images in our val set (curate from OpenScore Lieder
  or Music21 corpus next).
- Pitch ground truth covers only 8 noteheads on 1 image / 1 clef.
- Per-stave clef accuracy on real images is not measured (only synthetic).

**Recommended next steps** (Phase D, if pursued):

1. Curate 3–5 Bach chorale page images with full ground-truth MIDI annotations.
2. Extend C1 harness to multi-clef (treble + bass) and run on chorales.
3. Manual rule audit: pick one chorale where we know the right answer,
   compare predicted violations to a music theory student's grading.
4. Only after pitch accuracy ≥ 80% on chorales: TFLite export + Android port.

## Files added in Phase C

| File | Purpose |
|------|---------|
| `pitch_accuracy_harness.py`     | C1 — quantitative pitch metric |
| `pitch_ground_truth.json`       | C1 — 8-notehead GT |
| `clef_detector.py`              | C2 — per-stave clef assignment |
| `accidental_resolver.py`        | C5 — in-measure accidental state |
| `c6_revalidate_phase_b.py`      | C6 — re-runs Phase B images |
| `c6_revalidate_4part.py`        | C6 — runs Bartók 4-stave |
| `phase_c_revalidation.json`     | C6 — output of Phase B re-run |
| `phase_c_4part.json`            | C6 — output of 4-stave run |
| `test_pitch_accuracy.py`        | C1 tests |
| `test_clef_detector.py`         | C2 tests |
| `test_voice_binder_layout.py`   | C4 tests |
| `test_accidental_resolver.py`   | C5 tests |

Plus modifications to:
- `voice_binder.py` (layout dispatcher)
- `downstream_eval.py` (clef_detector + accidental_resolver integration)

## Conclusion

✅ **Phase B's primary failure mode is fixed.** 530 fake violations on
orchestral pages → 0, by recognizing that the rule engine should refuse
out-of-scope content rather than guess at it.

⚠️ **Phase C is not the green-light to deploy.** Pitch correctness on real
4-part music is still unmeasured. Phase D (chorale GT) is required before
TFLite export.

→ **Going to C7 (final decision) with this report as input.**
