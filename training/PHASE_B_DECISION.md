# Phase B Decision (2026-04-07)

## TL;DR

**Phase B DO NOT SHIP Phase 9 to Android yet.** The downstream pipeline works
structurally, but pitch estimation accuracy is too low. Real harmony errors
cannot yet be reliably identified.

## What we validated

✅ Phase 9 model detects noteheads correctly (781 on 5 images)
✅ Staff line detection works (B1)
✅ Pitch estimation math is correct on synthetic data (B2, 15/15 tests)
✅ Voice binding logic works (B3)
✅ Measure segmentation works (B4)
✅ Full pipeline wires end-to-end (B5)
✅ Rule engine fires on synthetic violations (B6)

## What failed

❌ On real images, pitch estimation produces mostly-wrong MIDI pitches
❌ Wrong pitches cascade into fake M1/V1/P1 violations (avg 5.35/chord)
❌ Real harmony errors buried in noise

## Root causes (best guesses, need verification)

1. **Clef detection per-staff, not per-system** — multi-staff grand staves
   have both treble and bass clefs but code picks one
2. **Staff line false positives** — text/beam regions may be detected as staves
3. **Voice binding heuristic wrong for orchestral scores** — top staff ≠ S+A
   if it's actually Violin 1
4. **Accidentals not scoped to measure** — key signature + bar accidentals
   should persist to same-pitch notes in the measure

## Next phase: Phase C (Pitch Accuracy)

Goals:
1. Pitch accuracy ≥80% on a known 4-voice chorale image
2. False-positive violation count ≤0.5 per chord on valid progression

### Phase C Sub-tasks

#### C1: Build pitch accuracy test harness
- Pick ONE simple OpenScore chorale image
- Manually label correct MIDI for 20-30 noteheads
- Script: measure accuracy = correct / total

#### C2: Per-stave clef detection
- Detect clef objects PER staff (not once per image)
- Handle grand staves (treble+bass paired vertically)

#### C3: Filter false staves
- Reject "staff" detections that don't have 5 roughly-equal lines
- Reject staves too close to image top/bottom (page numbers, text)

#### C4: Grand staff voice binding
- If 2 staves close together → treat as grand staff (treble=S/A, bass=T/B)
- If 4+ staves → orchestral layout (needs different heuristic)

#### C5: Measure-scoped accidentals
- Parse accidentals at detection time
- Propagate within measure for same-pitch notes
- Reset at barline

#### C6: Re-validate on same 5 images
- Target: avg violations/chord < 2 (vs current 5.35)
- If achieved → proceed to Android integration

## Alternative path: Ship Phase 9 as "symbol detector only"

Instead of trying to get full harmony analysis working in Python first,
we could:

1. Ship Phase 9 to Android as a pure **symbol detector**
2. Let the Android `SymbolAssembler.kt` do the pitch/voice/measure work
3. Use Kotlin debugging tools to iterate on pitch estimation there

**Pros**: Kotlin has better ecosystem for production music apps (e.g., music21
equivalents may exist). Android UI allows interactive pitch correction.
**Cons**: Slower iteration, debugging harder. Wastes all our Python work.

## Decision

**Recommended path: Phase C (continue in Python)**

Reasons:
- Phase B infrastructure is all built; Phase C is incremental
- Python iteration is 10x faster than Android for logic development
- Once pitch is accurate, Android port is straightforward (mostly data
  structures and API calls)
- Pitch accuracy is testable in isolation (single chorale image)

**Not recommended**: Deploy to Android now. Users would see 500+ false
violations per 5 images — pipeline output is not yet trustworthy.

## Status transition

- Phase A: **complete** (Phase 9 detection → pipeline wiring)
- Phase B: **complete** (modular pipeline + synthetic rule validation)
- **Phase C: START** (pitch accuracy iteration)
- Phase D (future): Android integration
