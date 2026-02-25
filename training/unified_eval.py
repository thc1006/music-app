#!/usr/bin/env python3
"""
Unified evaluation script for YOLO OMR models.

Runs two evaluation protocols:
  - strict:  conf=0.001, iou=0.7  (academic / fair comparison)
  - deploy:  conf=0.25,  iou=0.55 (production / deployment target)

Usage (from training/ directory):
    python unified_eval.py --models path/to/best.pt [path2.pt ...] \
        --data datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = [
    "notehead_filled",
    "notehead_hollow",
    "stem",
    "beam",
    "flag_8th",
    "flag_16th",
    "flag_32nd",
    "augmentation_dot",
    "tie",
    "clef_treble",
    "clef_bass",
    "clef_alto",
    "clef_tenor",
    "accidental_sharp",
    "accidental_flat",
    "accidental_natural",
    "accidental_double_sharp",
    "accidental_double_flat",
    "rest_whole",
    "rest_half",
    "rest_quarter",
    "rest_8th",
    "rest_16th",
    "barline",
    "barline_double",
    "barline_final",
    "barline_repeat",
    "time_signature",
    "key_signature",
    "fermata",
    "dynamic_soft",
    "dynamic_loud",
    "ledger_line",
]

PROTOCOLS: dict[str, dict[str, float]] = {
    "strict": {"conf": 0.001, "iou": 0.7},
    "deploy": {"conf": 0.25, "iou": 0.55},
}

LEADERBOARD_PATH = Path("reports/unified_leaderboard.json")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified two-protocol YOLO evaluation (strict + deploy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL_PATH",
        help="One or more .pt model weight files to evaluate.",
    )
    parser.add_argument(
        "--data",
        default="datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml",
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size (default: 1280).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help='Device to use: "0" for first GPU, "cpu" for CPU (default: 0).',
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write per-model JSON and CSV files (default: reports).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8).",
    )
    parser.add_argument(
        "--no-update-leaderboard",
        action="store_true",
        help="Skip updating the shared leaderboard JSON.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def run_protocol(
    model: Any,
    protocol_name: str,
    protocol_cfg: dict[str, float],
    data: str,
    imgsz: int,
    device: str,
    workers: int,
) -> dict[str, Any]:
    """Run model.val() for a single protocol and return structured results."""
    print(
        f"  [{protocol_name}] conf={protocol_cfg['conf']}, iou={protocol_cfg['iou']}"
    )
    result = model.val(
        data=data,
        imgsz=imgsz,
        conf=protocol_cfg["conf"],
        iou=protocol_cfg["iou"],
        device=device,
        workers=workers,
        verbose=False,
        plots=False,
        save_json=False,
    )

    box = result.box

    # Per-class AP values — shape: (num_classes,)
    # ap50 is per-class AP at IoU=0.50
    # ap   is per-class AP at IoU=0.50:0.95
    ap50_per_class: list[float] = (
        box.ap50.tolist() if hasattr(box, "ap50") else []
    )
    ap_per_class: list[float] = (
        box.ap.tolist() if hasattr(box, "ap") else []
    )

    # Build per-class dict using canonical class names.
    # The validator may have fewer classes than CLASS_NAMES if some are absent
    # from the validation split; we pair by index and fall back gracefully.
    per_class: dict[str, dict[str, float]] = {}
    num_model_classes = max(len(ap50_per_class), len(ap_per_class))
    for idx in range(num_model_classes):
        cls_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        per_class[cls_name] = {
            "ap50": float(ap50_per_class[idx]) if idx < len(ap50_per_class) else 0.0,
            "ap50_95": float(ap_per_class[idx]) if idx < len(ap_per_class) else 0.0,
        }

    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        "conf": protocol_cfg["conf"],
        "iou": protocol_cfg["iou"],
    }


def evaluate_model(
    model_path: Path,
    data: str,
    imgsz: int,
    device: str,
    workers: int,
) -> dict[str, Any]:
    """Evaluate one model under both protocols. Returns a full results dict."""
    from ultralytics import YOLO  # imported here to keep module importable without GPU

    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))

    protocols_results: dict[str, Any] = {}
    for proto_name, proto_cfg in PROTOCOLS.items():
        protocols_results[proto_name] = run_protocol(
            model=model,
            protocol_name=proto_name,
            protocol_cfg=proto_cfg,
            data=data,
            imgsz=imgsz,
            device=device,
            workers=workers,
        )

    return {
        "model": str(model_path),
        "data": data,
        "imgsz": imgsz,
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
        "protocols": protocols_results,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_model_json(results: dict[str, Any], output_dir: Path) -> Path:
    """Save per-model JSON result file."""
    model_stem = Path(results["model"]).parent.parent.name  # e.g. stable_1280_resumed
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"eval_{model_stem}_{ts}.json"
    filename.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved JSON: {filename}")
    return filename


def save_per_class_csv(results: dict[str, Any], output_dir: Path) -> Path:
    """Save per-class AP breakdown as CSV."""
    model_stem = Path(results["model"]).parent.parent.name
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"eval_{model_stem}_{ts}_per_class.csv"

    # Gather all class names across both protocols
    all_classes: list[str] = CLASS_NAMES[:]

    rows: list[dict[str, str]] = []
    for cls_name in all_classes:
        row: dict[str, str] = {"class": cls_name}
        for proto_name in PROTOCOLS:
            proto = results["protocols"].get(proto_name, {})
            per_class = proto.get("per_class", {})
            cls_data = per_class.get(cls_name, {})
            row[f"{proto_name}_ap50"] = f"{cls_data.get('ap50', 0.0):.4f}"
            row[f"{proto_name}_ap50_95"] = f"{cls_data.get('ap50_95', 0.0):.4f}"
        rows.append(row)

    fieldnames = ["class"] + [
        f"{p}_{m}"
        for p in PROTOCOLS
        for m in ("ap50", "ap50_95")
    ]

    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved CSV:  {filename}")
    return filename


def update_leaderboard(all_results: list[dict[str, Any]]) -> None:
    """Append or update entries in the shared leaderboard JSON."""
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if LEADERBOARD_PATH.exists():
        try:
            existing = json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []

    # Overwrite entries with the same model path; append new ones.
    existing_by_model = {entry["model"]: entry for entry in existing}
    for result in all_results:
        existing_by_model[result["model"]] = result

    # Sort descending by deploy mAP50 for convenience.
    leaderboard = sorted(
        existing_by_model.values(),
        key=lambda r: r["protocols"].get("deploy", {}).get("map50", 0.0),
        reverse=True,
    )

    LEADERBOARD_PATH.write_text(
        json.dumps(leaderboard, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nLeaderboard updated: {LEADERBOARD_PATH} ({len(leaderboard)} entries)")


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------


def _col(text: str, width: int, align: str = "<") -> str:
    """Pad/truncate text to fixed column width."""
    text = str(text)
    if len(text) > width:
        text = text[: width - 1] + "…"
    return f"{text:{align}{width}}"


def print_summary_table(all_results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    col_widths = {
        "model": 40,
        "proto": 7,
        "map50": 8,
        "map50_95": 10,
        "prec": 8,
        "recall": 8,
    }
    sep = "-" * (sum(col_widths.values()) + len(col_widths) * 3 + 1)

    header = (
        "| "
        + _col("Model", col_widths["model"])
        + " | "
        + _col("Proto", col_widths["proto"])
        + " | "
        + _col("mAP50", col_widths["map50"])
        + " | "
        + _col("mAP50-95", col_widths["map50_95"])
        + " | "
        + _col("Prec", col_widths["prec"])
        + " | "
        + _col("Recall", col_widths["recall"])
        + " |"
    )

    print(f"\n{sep}")
    print(header)
    print(sep)

    for result in all_results:
        model_label = Path(result["model"]).parent.parent.name or Path(result["model"]).stem
        for proto_name in PROTOCOLS:
            proto = result["protocols"].get(proto_name, {})
            map50 = proto.get("map50", 0.0)
            map50_95 = proto.get("map50_95", 0.0)
            prec = proto.get("precision", 0.0)
            recall = proto.get("recall", 0.0)

            row = (
                "| "
                + _col(model_label, col_widths["model"])
                + " | "
                + _col(proto_name, col_widths["proto"])
                + " | "
                + _col(f"{map50:.4f}", col_widths["map50"])
                + " | "
                + _col(f"{map50_95:.4f}", col_widths["map50_95"])
                + " | "
                + _col(f"{prec:.4f}", col_widths["prec"])
                + " | "
                + _col(f"{recall:.4f}", col_widths["recall"])
                + " |"
            )
            print(row)
        print(sep)

    print()


def print_per_class_table(results: dict[str, Any]) -> None:
    """Print per-class AP breakdown for a single model."""
    model_label = Path(results["model"]).stem
    print(f"\nPer-class breakdown: {model_label}")

    col_w = {
        "cls": 28,
        "ap": 10,
    }
    proto_names = list(PROTOCOLS.keys())
    header_parts = [_col("Class", col_w["cls"])]
    for p in proto_names:
        header_parts.append(_col(f"{p} AP50", col_w["ap"]))
        header_parts.append(_col(f"{p} AP50-95", col_w["ap"]))
    sep = "-" * (col_w["cls"] + len(proto_names) * 2 * col_w["ap"] + (1 + len(proto_names) * 2) * 3 + 1)
    header = "| " + " | ".join(header_parts) + " |"
    print(sep)
    print(header)
    print(sep)

    for cls_name in CLASS_NAMES:
        row_parts = [_col(cls_name, col_w["cls"])]
        for p in proto_names:
            per_class = results["protocols"].get(p, {}).get("per_class", {})
            cls_data = per_class.get(cls_name, {})
            ap50 = cls_data.get("ap50", 0.0)
            ap50_95 = cls_data.get("ap50_95", 0.0)
            row_parts.append(_col(f"{ap50:.4f}", col_w["ap"]))
            row_parts.append(_col(f"{ap50_95:.4f}", col_w["ap"]))
        print("| " + " | ".join(row_parts) + " |")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = [Path(p) for p in args.models]
    missing = [p for p in model_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: model not found: {p}", file=sys.stderr)
        return 1

    all_results: list[dict[str, Any]] = []

    for model_path in model_paths:
        try:
            results = evaluate_model(
                model_path=model_path,
                data=args.data,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
            )
        except Exception as exc:
            print(f"ERROR evaluating {model_path}: {exc}", file=sys.stderr)
            continue

        save_model_json(results, output_dir)
        save_per_class_csv(results, output_dir)
        print_per_class_table(results)
        all_results.append(results)

    if not all_results:
        print("No models evaluated successfully.", file=sys.stderr)
        return 2

    print_summary_table(all_results)

    if not args.no_update_leaderboard:
        update_leaderboard(all_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
