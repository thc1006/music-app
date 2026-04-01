#!/usr/bin/env python3
"""
Phase A: Inference-Time Recall Improvement Evaluation
=====================================================
Sweeps conf and iou thresholds on the existing best model (no retraining)
to find the configuration that maximizes mAP50 while keeping Precision > 0.85.

Two stages:
  1. Confidence threshold sweep  (conf in [0.05..0.30], iou=0.55)
  2. Combined conf+iou sweep     (top conf values x iou in [0.45..0.60])

Results are saved to reports/phase_a_inference_results.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = (
    "/home/thc1006/dev/music-app/training/"
    "harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt"
)
DATA_YAML = (
    "/home/thc1006/dev/music-app/training/"
    "datasets/yolo_harmony_v2_phase8_cleaned_v2/harmony_phase8_cleaned_v2.yaml"
)
IMGSZ = 1280
DEVICE = "0"

# Stage 1: conf sweep with fixed iou
CONF_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
FIXED_IOU = 0.55

# Stage 2: combined sweep
IOU_VALUES = [0.45, 0.50, 0.55, 0.60]

OUTPUT_JSON = "/home/thc1006/dev/music-app/training/reports/phase_a_inference_results.json"

# Constraints
MIN_PRECISION = 0.85


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_val(model, conf: float, iou: float) -> dict[str, float]:
    """Run model.val() and return a compact metrics dict."""
    result = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        conf=conf,
        iou=iou,
        device=DEVICE,
        verbose=False,
    )
    return {
        "conf": conf,
        "iou": iou,
        "mAP50": round(float(result.box.map50), 5),
        "mAP50_95": round(float(result.box.map), 5),
        "precision": round(float(result.box.mp), 5),
        "recall": round(float(result.box.mr), 5),
    }


def print_row(m: dict[str, float], header: bool = False) -> None:
    """Pretty-print one metrics row."""
    if header:
        print(
            f"{'conf':>6s}  {'iou':>5s}  {'mAP50':>8s}  {'mAP50-95':>9s}  "
            f"{'Precision':>10s}  {'Recall':>8s}"
        )
        print("-" * 58)
    print(
        f"{m['conf']:6.2f}  {m['iou']:5.2f}  {m['mAP50']:8.4f}  {m['mAP50_95']:9.4f}  "
        f"{m['precision']:10.4f}  {m['recall']:8.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from ultralytics import YOLO

    tz = timezone(timedelta(hours=8))
    start = time.time()
    print(f"[{datetime.now(tz):%Y-%m-%d %H:%M:%S}] Phase A Inference Evaluation")
    print(f"  Model : {MODEL_PATH}")
    print(f"  Data  : {DATA_YAML}")
    print(f"  imgsz : {IMGSZ}")
    print()

    model = YOLO(MODEL_PATH)

    # ------------------------------------------------------------------
    # Stage 1 – Confidence threshold sweep (iou fixed at 0.55)
    # ------------------------------------------------------------------
    print("=" * 62)
    print("Stage 1: Confidence Threshold Sweep  (iou=0.55)")
    print("=" * 62)

    stage1_results: list[dict[str, float]] = []
    for i, conf in enumerate(CONF_VALUES):
        t0 = time.time()
        m = run_val(model, conf=conf, iou=FIXED_IOU)
        elapsed = time.time() - t0
        stage1_results.append(m)
        print_row(m, header=(i == 0))
        print(f"  ({elapsed:.1f}s)")

    # Pick top-N conf values that satisfy precision constraint, ranked by mAP50
    valid_confs = [r for r in stage1_results if r["precision"] >= MIN_PRECISION]
    if not valid_confs:
        # Relax: just use all
        valid_confs = sorted(stage1_results, key=lambda r: r["mAP50"], reverse=True)
    else:
        valid_confs = sorted(valid_confs, key=lambda r: r["mAP50"], reverse=True)

    # Take top 3 unique conf values for the combined sweep
    top_confs = [r["conf"] for r in valid_confs[:3]]
    print(f"\nTop conf values for Stage 2 (precision >= {MIN_PRECISION}): {top_confs}")
    print()

    # ------------------------------------------------------------------
    # Stage 2 – Combined conf + iou sweep
    # ------------------------------------------------------------------
    print("=" * 62)
    print("Stage 2: Combined conf + iou Sweep")
    print("=" * 62)

    stage2_results: list[dict[str, float]] = []
    first = True
    for conf in top_confs:
        for iou in IOU_VALUES:
            # Skip if already tested in Stage 1
            already = any(
                r["conf"] == conf and r["iou"] == iou for r in stage1_results
            )
            if already:
                dup = next(
                    r for r in stage1_results if r["conf"] == conf and r["iou"] == iou
                )
                stage2_results.append(dup)
                print_row(dup, header=first)
                print("  (cached from Stage 1)")
                first = False
                continue

            t0 = time.time()
            m = run_val(model, conf=conf, iou=iou)
            elapsed = time.time() - t0
            stage2_results.append(m)
            print_row(m, header=first)
            print(f"  ({elapsed:.1f}s)")
            first = False

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    all_results = stage1_results + [
        r for r in stage2_results if r not in stage1_results
    ]
    # De-duplicate
    seen = set()
    unique: list[dict[str, float]] = []
    for r in all_results:
        key = (r["conf"], r["iou"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Best by mAP50 with precision constraint
    constrained = [r for r in unique if r["precision"] >= MIN_PRECISION]
    best_map50 = max(constrained, key=lambda r: r["mAP50"]) if constrained else max(unique, key=lambda r: r["mAP50"])

    # Best by Recall with precision constraint
    best_recall = max(constrained, key=lambda r: r["recall"]) if constrained else max(unique, key=lambda r: r["recall"])

    # Best by mAP50-95 with precision constraint
    best_map50_95 = max(constrained, key=lambda r: r["mAP50_95"]) if constrained else max(unique, key=lambda r: r["mAP50_95"])

    elapsed_total = time.time() - start

    print()
    print("=" * 62)
    print("RESULTS SUMMARY")
    print("=" * 62)
    print(f"\nTotal configurations tested: {len(unique)}")
    print(f"Total elapsed time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"Precision constraint: >= {MIN_PRECISION}")
    print(f"Configs meeting constraint: {len(constrained)}/{len(unique)}")

    print(f"\n--- Best mAP50 (precision >= {MIN_PRECISION}) ---")
    print_row(best_map50, header=True)

    print(f"\n--- Best Recall (precision >= {MIN_PRECISION}) ---")
    print_row(best_recall, header=True)

    print(f"\n--- Best mAP50-95 (precision >= {MIN_PRECISION}) ---")
    print_row(best_map50_95, header=True)

    # Full table sorted by mAP50 descending
    print("\n--- All Results (sorted by mAP50 desc) ---")
    sorted_all = sorted(unique, key=lambda r: r["mAP50"], reverse=True)
    for i, r in enumerate(sorted_all):
        flag = ""
        if r["precision"] < MIN_PRECISION:
            flag = "  [!prec<0.85]"
        print_row(r, header=(i == 0))
        if flag:
            print(flag)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "metadata": {
            "model": MODEL_PATH,
            "data": DATA_YAML,
            "imgsz": IMGSZ,
            "device": DEVICE,
            "precision_constraint": MIN_PRECISION,
            "timestamp": datetime.now(tz).isoformat(),
            "elapsed_seconds": round(elapsed_total, 1),
        },
        "stage1_conf_sweep": {
            "fixed_iou": FIXED_IOU,
            "results": stage1_results,
        },
        "stage2_combined_sweep": {
            "top_confs_selected": top_confs,
            "iou_values": IOU_VALUES,
            "results": stage2_results,
        },
        "all_results_sorted_by_map50": sorted_all,
        "best": {
            "best_mAP50": best_map50,
            "best_recall": best_recall,
            "best_mAP50_95": best_map50_95,
        },
    }

    out = Path(OUTPUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
