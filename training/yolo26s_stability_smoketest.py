#!/usr/bin/env python3
"""
YOLO26s 穩定性 smoke test。
重點是先驗證是否會再出現 NaN/Inf，再決定是否進入長訓練。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_DATA = "datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 stability smoke test")
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="harmony_omr_v2_yolo26")
    parser.add_argument("--name", default="yolo26s_stability_smoketest")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", choices=["false", "true", "ram", "disk"], default="false")
    return parser.parse_args()


def has_nan_loss(results_csv: Path) -> bool:
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results.csv: {results_csv}")
    with results_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key, value in row.items():
                if "loss" not in key:
                    continue
                try:
                    numeric = float(value)
                except ValueError:
                    return True
                if math.isnan(numeric) or math.isinf(numeric):
                    return True
    return False


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    cache = {"false": False, "true": True, "ram": "ram", "disk": "disk"}[args.cache]

    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "fraction": args.fraction,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.01,
        "warmup_epochs": 5,
        "warmup_momentum": 0.85,
        "weight_decay": 0.0005,
        "patience": 10,
        "amp": False,
        "cache": cache,
        "workers": args.workers,
        "nbs": 64,
        "close_mosaic": 10,
        "plots": True,
        "val": True,
        "verbose": True,
    }
    model.train(**train_args)

    save_dir = Path(getattr(model.trainer, "save_dir", Path(args.project) / args.name))
    results_csv = save_dir / "results.csv"
    if has_nan_loss(results_csv):
        raise RuntimeError(
            f"偵測到 NaN/Inf loss，建議停止 YOLO26 路線並檢查設定: {results_csv}"
        )

    print(f"Smoke test completed without NaN/Inf loss: {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
