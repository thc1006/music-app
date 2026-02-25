#!/usr/bin/env python3
"""Sweep confidence and IoU thresholds to find optimal deploy settings.

Tests the Ultimate v5 best model across a grid of conf/iou values
to find the combination that maximizes mAP50.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

MODEL = Path(
    "/home/thc1006/dev/music-app/training/"
    "harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt"
)
DATA = "datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml"
IMGSZ = 1280

CONF_VALUES = [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
IOU_VALUES = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(MODEL))
    print(f"Model: {MODEL}")
    print(f"Data:  {DATA}")
    print(f"Testing {len(CONF_VALUES)} conf x {len(IOU_VALUES)} iou = {len(CONF_VALUES)*len(IOU_VALUES)} combinations\n")

    results = []

    for conf in CONF_VALUES:
        for iou in IOU_VALUES:
            t0 = time.time()
            metrics = model.val(
                data=DATA,
                imgsz=IMGSZ,
                conf=conf,
                iou=iou,
                device="0",
                verbose=False,
                plots=False,
            )
            elapsed = time.time() - t0

            row = {
                "conf": conf,
                "iou": iou,
                "mAP50": round(float(metrics.box.map50), 5),
                "mAP50_95": round(float(metrics.box.map), 5),
                "precision": round(float(metrics.box.mp), 5),
                "recall": round(float(metrics.box.mr), 5),
                "time_s": round(elapsed, 1),
            }
            results.append(row)

            flag = " ★" if row["mAP50"] > 0.7519 else ""
            print(
                f"conf={conf:.3f} iou={iou:.2f} | "
                f"mAP50={row['mAP50']:.4f} mAP50-95={row['mAP50_95']:.4f} | "
                f"P={row['precision']:.3f} R={row['recall']:.3f} | "
                f"{elapsed:.0f}s{flag}"
            )
            sys.stdout.flush()

    # Find best
    best = max(results, key=lambda r: r["mAP50"])
    print(f"\n{'='*80}")
    print(f"BEST: conf={best['conf']:.3f} iou={best['iou']:.2f} → mAP50={best['mAP50']:.5f}")
    print(f"       mAP50-95={best['mAP50_95']:.5f} P={best['precision']:.4f} R={best['recall']:.4f}")

    # Top 5
    top5 = sorted(results, key=lambda r: r["mAP50"], reverse=True)[:5]
    print(f"\nTop 5:")
    for i, r in enumerate(top5, 1):
        print(f"  {i}. conf={r['conf']:.3f} iou={r['iou']:.2f} → mAP50={r['mAP50']:.5f} P={r['precision']:.3f} R={r['recall']:.3f}")

    # Save
    out = Path("reports/conf_iou_sweep.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"best": best, "all": results}, f, indent=2)
    print(f"\nSaved to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
