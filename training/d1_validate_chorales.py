"""
Phase D1 — Validate the rendered Bach chorales by running Phase 9 + pipeline.

Expected:
  - 4 staves detected on each chorale (4-voice SATB)
  - Reasonable notehead count (≥80% of music21's note count)
  - Pipeline produces chords (validates _bind_quartet code path)

Output: training/datasets/chorale_gt/validation_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from downstream_eval import run_pipeline
from staff_detector import detect_staves


CHORALE_DIR = PROJECT_ROOT / "training/datasets/chorale_gt"
BWV_LIST = ["269", "267", "270", "354", "377"]


def get_expected_notes(bwv: str) -> dict:
    """Use music21 to count expected notes per voice in this chorale."""
    from music21 import corpus, note
    score = corpus.parse(f"bach/bwv{bwv}")
    counts = {}
    for i, p in enumerate(score.parts):
        notes = list(p.recurse().getElementsByClass(note.Note))
        counts[p.partName or f"part{i}"] = len(notes)
    counts["total"] = sum(counts.values())
    counts["measures"] = len(score.parts[0].getElementsByClass("Measure"))
    return counts


def main():
    rows = []
    for bwv in BWV_LIST:
        img_path = CHORALE_DIR / f"bwv{bwv}.png"
        if not img_path.exists():
            print(f"MISSING: {img_path}")
            continue

        # Music21 ground truth note count (over WHOLE chorale, not just page 1)
        expected = get_expected_notes(bwv)

        # Detect staves
        staves = detect_staves(str(img_path))
        n_staves = len(staves)

        # Run full pipeline with chorale-tuned settings
        # (3300×2550 LilyPond renders need imgsz=1600, conf=0.10)
        result = run_pipeline(str(img_path), imgsz=1600, conf=0.10, iou=0.55)
        n_det = result["num_detections"]
        n_nh = result["num_noteheads"]
        n_ch = result["num_chords"]
        n_v = len(result["violations"])

        rows.append({
            "bwv": bwv,
            "image": img_path.name,
            "staves": n_staves,
            "expected_total_notes": expected["total"],
            "expected_per_voice": {k: v for k, v in expected.items() if k not in ("total", "measures")},
            "expected_measures": expected["measures"],
            "detections": n_det,
            "noteheads": n_nh,
            "chords": n_ch,
            "violations": n_v,
            "notehead_recall_vs_total": (
                n_nh / expected["total"] if expected["total"] else 0
            ),
        })

    print()
    print(f"{'BWV':<5} {'Stv':>4} {'ExpN':>5} {'Det':>5} {'Nh':>5} "
          f"{'Recall':>7} {'Chord':>6} {'Viol':>5}")
    print("-" * 56)
    for r in rows:
        print(f"{r['bwv']:<5} {r['staves']:>4} "
              f"{r['expected_total_notes']:>5} "
              f"{r['detections']:>5} {r['noteheads']:>5} "
              f"{r['notehead_recall_vs_total']*100:>6.1f}% "
              f"{r['chords']:>6} {r['violations']:>5}")
    print("-" * 56)

    print()
    print("Notes:")
    print("  - 'ExpN' is total notes across the WHOLE chorale (music21).")
    print("  - 'Recall' is rendered-page noteheads / total chorale notes —")
    print("    so it will be <100% if the rendered image is page 1 of N pages.")
    print("  - We care about: (a) staves==4, (b) chords>0, (c) recall reasonable")
    print("    given page-1-only constraint.")

    out = CHORALE_DIR / "validation_summary.json"
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
