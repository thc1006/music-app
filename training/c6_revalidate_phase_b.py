"""
Phase C6 — Re-validate Phase C pipeline on the same 5 OpenScore images
that Phase B used as the baseline (5.35 violations/chord).

Compare:
  - Phase B baseline (pre-C2/C3/C4/C5):  878 det, 781 nh, 99 chords, 530 v, 5.35 v/c
  - Phase C (post-C2 + C4 + C5):         current run on same images

Phase C interventions:
  C2: per-stave clef detection (greedy 1-to-1 matching)
  C4: layout-aware voice binding (skip non-2/4-stave layouts)
  C5: in-measure accidental persistence

Expected:
  - non-{2,4}-stave images → chords=0 (C4 skip is the right behavior on
    orchestral content; the 5 Phase B images are likely 8-18 stave so
    most/all should produce 0 chords now)
  - if any image is 2 or 4 stave, voice binding will run and chords > 0

If chords=0 on all 5 → confirms Phase B's "wrong layout" diagnosis was
correct AND that C4 prevents fake violations on out-of-scope content.
This is a NEGATIVE-RESULT validation: we're not pretending to be right
on orchestral music.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from downstream_eval import run_pipeline, run_phase9_detection
from staff_detector import detect_staves


VAL_DIR = PROJECT_ROOT / "training/datasets/yolo_harmony_v2_phase6_fixed/val/images"

# Same 5 OpenScore engraver-augmented images Phase B reported on
PHASE_B_IMAGES = [
    "phase7_phase6_base_p4_p3_p2_lg-102414375-aug-beethoven--page-2_oversample_12_3.png",
    "phase7_phase6_base_p4_p3_p2_lg-102548668-aug-gonville--page-1.png",
    "phase7_phase6_base_p4_p3_p2_lg-102548668-aug-gutenberg1939--page-1.png",
    "phase7_phase6_base_p4_p3_p2_lg-105569450-aug-emmentaler--page-1_oversample_6_1.png",
    "phase7_phase6_base_p4_p3_p2_lg-11466156-aug-beethoven--page-2.png",
]

# Phase B baseline numbers (from PHASE_B_VALIDATION_REPORT.md, ordered to match)
PHASE_B_BASELINE = [
    {"label": "Beethoven p2",  "det": 203, "nh": 177, "chords": 22, "viol": 127},
    {"label": "Gonville p1",   "det": 142, "nh": 129, "chords": 15, "viol":  75},
    {"label": "Gutenberg p1",  "det": 154, "nh": 133, "chords": 18, "viol":  99},
    {"label": "Emmentaler p1", "det": 236, "nh": 219, "chords": 29, "viol": 171},
    {"label": "Beethoven p2'", "det": 143, "nh": 123, "chords": 15, "viol":  58},
]


def label_for(filename: str, baseline: dict) -> str:
    return baseline["label"]


def main():
    print("=" * 78)
    print("Phase C6 — Re-validation on Phase B test images")
    print("=" * 78)

    rows = []
    total_b = {"det": 0, "nh": 0, "chords": 0, "viol": 0}
    total_c = {"det": 0, "nh": 0, "chords": 0, "viol": 0, "staves": 0}

    rule_counter = Counter()

    for img_name, baseline in zip(PHASE_B_IMAGES, PHASE_B_BASELINE):
        img_path = VAL_DIR / img_name
        if not img_path.exists():
            print(f"  MISSING: {img_path}")
            continue

        # Get staff count separately so we can report layout dispatch behavior
        staves = detect_staves(str(img_path))
        n_staves = len(staves)

        result = run_pipeline(str(img_path))
        n_det = result["num_detections"]
        n_nh = result["num_noteheads"]
        n_ch = result["num_chords"]
        n_v = len(result["violations"])

        for v in result["violations"]:
            rule_counter[v.rule_id] += 1

        rows.append({
            "label": baseline["label"],
            "image": img_name,
            "staves": n_staves,
            "phase_b": baseline,
            "phase_c": {"det": n_det, "nh": n_nh, "chords": n_ch, "viol": n_v},
        })

        for k in ("det", "nh", "chords", "viol"):
            total_b[k] += baseline[k]
        total_c["det"] += n_det
        total_c["nh"] += n_nh
        total_c["chords"] += n_ch
        total_c["viol"] += n_v
        total_c["staves"] += n_staves

    # Print results
    print()
    print(f"{'Image':<14} {'Stv':>4} {'B-Det':>6} {'C-Det':>6} {'B-Nh':>6} {'C-Nh':>6} "
          f"{'B-Ch':>5} {'C-Ch':>5} {'B-V':>5} {'C-V':>5}")
    print("-" * 78)
    for r in rows:
        b = r["phase_b"]
        c = r["phase_c"]
        print(f"{r['label']:<14} {r['staves']:>4} "
              f"{b['det']:>6} {c['det']:>6} "
              f"{b['nh']:>6} {c['nh']:>6} "
              f"{b['chords']:>5} {c['chords']:>5} "
              f"{b['viol']:>5} {c['viol']:>5}")
    print("-" * 78)
    print(f"{'TOTAL':<14} {total_c['staves']:>4} "
          f"{total_b['det']:>6} {total_c['det']:>6} "
          f"{total_b['nh']:>6} {total_c['nh']:>6} "
          f"{total_b['chords']:>5} {total_c['chords']:>5} "
          f"{total_b['viol']:>5} {total_c['viol']:>5}")

    b_vc = total_b["viol"] / max(total_b["chords"], 1)
    c_vc = total_c["viol"] / max(total_c["chords"], 1)
    print()
    print(f"Phase B violations/chord:  {b_vc:.2f}")
    print(f"Phase C violations/chord:  {c_vc:.2f}")
    if b_vc > 0 and c_vc > 0:
        print(f"Improvement (lower=better):{b_vc/c_vc:.2f}x")
    elif total_c["chords"] == 0:
        print("Phase C: 0 chords on all images (C4 layout dispatch correctly")
        print("         skipped these as non-4-part orchestral content)")
    print()
    print("Rule breakdown (Phase C):")
    for rule_id, count in rule_counter.most_common():
        print(f"  {rule_id}: {count}")
    print()

    # Save JSON
    out_path = PROJECT_ROOT / "training/phase_c_revalidation.json"
    with open(out_path, "w") as f:
        json.dump({
            "phase_b_total": total_b,
            "phase_c_total": total_c,
            "phase_b_vc": b_vc,
            "phase_c_vc": c_vc,
            "rule_breakdown": dict(rule_counter),
            "rows": rows,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
