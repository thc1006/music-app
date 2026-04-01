#!/usr/bin/env python3
"""
驗證目前部署基線模型（Ultimate v5 + iou=0.55）是否達到最低門檻。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = (
    "harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt"
)
DEFAULT_DATA = "datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate production baseline model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .pt model")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to data yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-map50", type=float, default=0.70)
    parser.add_argument("--min-recall", type=float, default=0.55)
    parser.add_argument(
        "--output-json",
        default="reports/baseline_validation.json",
        help="Summary output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    result = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    metrics = {
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "thresholds": {
            "min_map50": args.min_map50,
            "min_recall": args.min_recall,
        },
        "passed": bool(
            result.box.map50 >= args.min_map50 and result.box.mr >= args.min_recall
        ),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
