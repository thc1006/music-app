"""
Phase C6 — Re-validate Phase C pipeline on actual 4-stave (string quartet)
content. This is the inverse of c6_revalidate_phase_b.py: confirm the
pipeline still produces chords on legitimate 4-part material.

Test material: Bartók String Quartet No. 5 mvt 3 (DoReMi, 4 staves per system).

Compare:
  - Phase B baseline   = 5.35 v/c on orchestral (mismeasured)
  - Phase C orchestral = 0.00 v/c on Phase B images (correctly skipped)
  - Phase C 4-part     = ?    v/c on real 4-stave content
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from downstream_eval import run_pipeline
from staff_detector import detect_staves


VAL_DIR = PROJECT_ROOT / "training/datasets/yolo_harmony_v2_phase10_1/val/images"

# 5 Bartók 4-stave images previously confirmed to detect 4 staves
QUARTET_IMAGES = [
    "phase7_phase6_base_p4_p3_doremi_doremi_Bartok - String Quartet 5 mvt 3-008.png",
    "phase7_phase6_base_p4_p3_doremi_doremi_Bartok - String Quartet 5 mvt 3-010.png",
    "phase7_phase6_base_p4_p3_doremi_doremi_Bartok - String Quartet 5 mvt 3-012.png",
    "phase7_phase6_base_p4_p3_doremi_doremi_Bartok - String Quartet 5 mvt 3-013.png",
    "phase7_phase6_base_p4_p3_doremi_doremi_Bartok - String Quartet 5 mvt 3-025.png",
]


def main():
    print("=" * 78)
    print("Phase C6 — Re-validation on 4-stave content (Bartók String Quartet 5)")
    print("=" * 78)
    print()

    rows = []
    total = {"det": 0, "nh": 0, "chords": 0, "viol": 0, "staves_total": 0}
    rule_counter = Counter()

    for img_name in QUARTET_IMAGES:
        img_path = VAL_DIR / img_name
        if not img_path.exists():
            print(f"  MISSING: {img_path}")
            continue

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
            "image": img_name,
            "staves": n_staves,
            "det": n_det,
            "nh": n_nh,
            "chords": n_ch,
            "viol": n_v,
        })
        total["det"] += n_det
        total["nh"] += n_nh
        total["chords"] += n_ch
        total["viol"] += n_v
        total["staves_total"] += n_staves

    print(f"{'#':>3} {'Stv':>4} {'Det':>5} {'Nh':>5} {'Chord':>6} {'Viol':>5}")
    print("-" * 36)
    for i, r in enumerate(rows):
        print(f"{i+1:>3} {r['staves']:>4} {r['det']:>5} {r['nh']:>5} "
              f"{r['chords']:>6} {r['viol']:>5}")
    print("-" * 36)
    print(f"{'TOT':>3} {total['staves_total']:>4} {total['det']:>5} {total['nh']:>5} "
          f"{total['chords']:>6} {total['viol']:>5}")

    if total["chords"] > 0:
        vc = total["viol"] / total["chords"]
        print()
        print(f"Violations/chord: {vc:.2f}")
        print(f"  (Phase B baseline on orchestral: 5.35 — but mismeasured)")
        print(f"  (Phase C C4 layout dispatch produces 0 v/c on those orchestral imgs)")
    else:
        print()
        print("No chords formed — staves were not 4 or quartet binding failed.")

    print()
    print("Rule breakdown:")
    for rule_id, count in rule_counter.most_common():
        print(f"  {rule_id}: {count}")

    out_path = PROJECT_ROOT / "training/phase_c_4part.json"
    with open(out_path, "w") as f:
        json.dump({
            "total": total,
            "violations_per_chord": (
                total["viol"] / total["chords"] if total["chords"] else 0
            ),
            "rule_breakdown": dict(rule_counter),
            "rows": rows,
        }, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
