# Bach Chorale Ground Truth Set (Phase D)

5 J. S. Bach chorales rendered from `music21.corpus` via LilyPond at 300 DPI,
3300×2550 px each. Each image contains the chorale's first LilyPond page —
4 staves (S/A/T/B), no lyrics, no titles.

## Files

| Image       | BWV       | Title                          | Measures (whole) | Notes (whole) |
|-------------|-----------|--------------------------------|-----------------:|--------------:|
| bwv269.png  | BWV 269   | Aus meines Herzens Grunde      | 24 | 229 |
| bwv267.png  | BWV 267   | Sei Lob und Ehr                | 20 | 330 |
| bwv270.png  | BWV 270   | Befiehl du deine Wege          | 15 | 207 |
| bwv354.png  | BWV 354   | Es ist genug                   | 12 | 235 |
| bwv377.png  | BWV 377   | Christus, der ist mein Leben   | 10 | 154 |

(Image is page 1 only; remaining content is on pages 2-N which we don't use yet.)

## How they were rendered

`d1_render_chorales.py` runs:

1. `music21.corpus.parse('bach/bwvNNN')`
2. Strip lyrics + titles
3. `score.write(fmt='lilypond')` → `.ly` source
4. Patch the `.ly` with our compact `\paper{}` block (11×8.5 in, no titles,
   set-global-staff-size 22)
5. `lilypond --png -dresolution=300` → high-DPI PNG

## Recommended detection settings for these images

The Phase 9 model was trained on 1280-wide engraver-augmented val images,
so the default `run_pipeline()` parameters (`imgsz=1280, conf=0.25`) under-detect
on these high-DPI 3300-wide chorale renders. Use:

```python
run_pipeline(image_path, imgsz=1600, conf=0.10, iou=0.55)
```

Validation results (`d1_validate_chorales.py` with the above settings):

| BWV | Stv | Det | Nh | Chords | Violations |
|----:|----:|----:|---:|-------:|-----------:|
| 269 | 4 | 69 | 65 | 3 | 5 |
| 267 | 4 | 78 | 75 | 6 | 17 |
| 270 | 4 | 74 | 69 | 9 | 25 |
| 354 | 4 | 70 | 64 | 5 | 26 |
| 377 | 4 | 73 | 69 | 7 | 28 |
| **TOT** | 20 | 364 | 342 | **30** | **101** |

All 5 chorales:
- Detect 4 staves → C4 layout dispatcher takes the `_bind_quartet` path
- Produce ≥3 chords → rule engine has data to grade
- Notehead count looks plausible against music21's page-1 expected count
  (whole-chorale recall percentages in `validation_summary.json` are
  misleading since they compare against the full N-page chorale, not
  just the rendered first page)

## What this set is FOR

- **D2**: extend `pitch_ground_truth.json` to ≥30 multi-clef noteheads per
  chorale (target ≥80% MIDI accuracy)
- **D3**: manual rule audit on one clean chorale (target FP rate <0.5 v/c)

## What this set is NOT

- Not a training set
- Not a benchmark for other models
- Not a substitute for real photographed sheet music (Phase E will need
  photo-quality test data)

## Files in this directory

- `bwv*.png`                — 5 rendered chorale images (3300×2550)
- `README.md`               — this file
- `validation_summary.json` — `d1_validate_chorales.py` output

To regenerate from scratch, run `training/d1_render_chorales.py` —
it pulls from `music21.corpus` and writes to this directory.
