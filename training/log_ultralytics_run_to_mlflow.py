#!/usr/bin/env python3
"""
將 Ultralytics results.csv 匯入 MLflow，作為輕量治理入口。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log Ultralytics run to MLflow")
    parser.add_argument("--results-csv", required=True, help="Path to results.csv")
    parser.add_argument("--experiment", default="harmony_omr")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tracking-uri", default="file:./training/mlruns")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra params in key=value format (repeatable)",
    )
    return parser.parse_args()


def parse_extra_params(raw_params: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in raw_params:
        if "=" not in item:
            raise ValueError(f"Invalid --param format: {item}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def read_rows(results_csv: Path) -> list[dict[str, str]]:
    with results_csv.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise KeyError(f"Missing metric key: {key}")
    return float(value)


def main() -> int:
    args = parse_args()
    results_csv = Path(args.results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "mlflow 尚未安裝，請先執行: pip install mlflow"
        ) from exc

    rows = read_rows(results_csv)
    if not rows:
        raise RuntimeError(f"No rows in results.csv: {results_csv}")

    last = rows[-1]
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", "0") or 0.0))
    extras = parse_extra_params(args.param)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or results_csv.parent.name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("results_csv", str(results_csv))
        mlflow.log_param("num_epochs_recorded", len(rows))
        for key, value in extras.items():
            mlflow.log_param(key, value)

        mlflow.log_metric("last_map50", as_float(last, "metrics/mAP50(B)"))
        mlflow.log_metric("last_map50_95", as_float(last, "metrics/mAP50-95(B)"))
        mlflow.log_metric("last_precision", as_float(last, "metrics/precision(B)"))
        mlflow.log_metric("last_recall", as_float(last, "metrics/recall(B)"))
        mlflow.log_metric("best_map50", as_float(best, "metrics/mAP50(B)"))
        mlflow.log_metric("best_epoch", as_float(best, "epoch"))
        mlflow.log_artifact(str(results_csv))

    print(f"Logged run to MLflow: experiment={args.experiment}, run_name={run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
