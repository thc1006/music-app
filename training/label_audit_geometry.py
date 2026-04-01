"""
label_audit_geometry.py

Label quality audit script for YOLO-format bounding box annotations.
Checks geometric anomalies in bounding box annotations and produces
a CSV report and JSON statistics summary.

Usage:
    python label_audit_geometry.py \
        --data-dir datasets/yolo_harmony_v2_phase8_final \
        --output-dir reports/

For the Phase 8 Final dataset (~2.53M annotations, 36,172 label files).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = [
    "notehead_filled",        # 0
    "notehead_hollow",        # 1
    "stem",                   # 2
    "beam",                   # 3
    "flag_8th",               # 4
    "flag_16th",              # 5
    "flag_32nd",              # 6
    "augmentation_dot",       # 7
    "tie",                    # 8
    "clef_treble",            # 9
    "clef_bass",              # 10
    "clef_alto",              # 11
    "clef_tenor",             # 12
    "accidental_sharp",       # 13
    "accidental_flat",        # 14
    "accidental_natural",     # 15
    "accidental_double_sharp",# 16
    "accidental_double_flat", # 17
    "rest_whole",             # 18
    "rest_half",              # 19
    "rest_quarter",           # 20
    "rest_8th",               # 21
    "rest_16th",              # 22
    "barline",                # 23
    "barline_double",         # 24
    "barline_final",          # 25
    "barline_repeat",         # 26
    "time_signature",         # 27
    "key_signature",          # 28
    "fermata",                # 29
    "dynamic_soft",           # 30
    "dynamic_loud",           # 31
    "ledger_line",            # 32
]

NUM_CLASSES = len(CLASS_NAMES)

# Anomaly thresholds
SIGMA_THRESHOLD = 3.0          # flag boxes >3σ from per-class mean
MIN_AREA = 1e-4                # area < 0.0001 → "extremely_small"
MAX_AREA = 0.5                 # area > 0.5    → "extremely_large"
MAX_ASPECT_RATIO = 20.0        # w/h > 20      → "degenerate_wide"
MIN_ASPECT_RATIO = 1.0 / 20.0  # w/h < 1/20   → "degenerate_tall"

# CSV field names
CSV_FIELDS = [
    "split",
    "image_path",
    "label_file",
    "line_number",
    "class_id",
    "class_name",
    "cx",
    "cy",
    "w",
    "h",
    "area",
    "aspect_ratio",
    "anomaly_type",
    "severity",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Annotation:
    """Single YOLO annotation row."""
    split: str
    label_file: str
    line_number: int
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        if self.h == 0.0:
            return float("inf")
        return self.w / self.h


@dataclass
class AnomalyRecord:
    """One flagged annotation."""
    split: str
    label_file: str
    line_number: int
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    area: float
    aspect_ratio: float
    anomaly_type: str
    severity: str

    def to_csv_row(self) -> dict[str, Any]:
        image_path = label_file_to_image_path(self.label_file)
        return {
            "split": self.split,
            "image_path": image_path,
            "label_file": self.label_file,
            "line_number": self.line_number,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "cx": f"{self.cx:.6f}",
            "cy": f"{self.cy:.6f}",
            "w": f"{self.w:.6f}",
            "h": f"{self.h:.6f}",
            "area": f"{self.area:.8f}",
            "aspect_ratio": f"{self.aspect_ratio:.4f}",
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def label_file_to_image_path(label_file: str) -> str:
    """
    Convert a label file path to its probable image path.
    YOLO convention: labels/ -> images/, strip .txt extension.
    """
    p = label_file.replace("/labels/", "/images/")
    if p.endswith(".txt"):
        p = p[:-4]
    return p


def _class_name(class_id: int) -> str:
    if 0 <= class_id < NUM_CLASSES:
        return CLASS_NAMES[class_id]
    return f"unknown_{class_id}"


# ---------------------------------------------------------------------------
# Worker function (used with multiprocessing)
# ---------------------------------------------------------------------------

def _parse_label_file(args: tuple[str, str, str]) -> list[tuple]:
    """
    Parse a single YOLO label file and return a list of raw tuples.
    Each tuple: (split, label_file, line_number, class_id, cx, cy, w, h)
    Invalid / unparseable lines are silently skipped.
    """
    split, label_file, abs_path = args
    rows: list[tuple] = []
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue
                rows.append((split, label_file, lineno, class_id, cx, cy, w, h))
    except OSError:
        pass
    return rows


# ---------------------------------------------------------------------------
# Loading annotations
# ---------------------------------------------------------------------------

def collect_label_files(data_dir: Path, splits: list[str]) -> list[tuple[str, str, str]]:
    """
    Collect (split, relative_label_path, absolute_label_path) for all label files.
    """
    tasks: list[tuple[str, str, str]] = []
    for split in splits:
        labels_dir = data_dir / split / "labels"
        if not labels_dir.is_dir():
            logger.warning("Labels directory not found: %s", labels_dir)
            continue
        files = sorted(labels_dir.glob("*.txt"))
        logger.info("  %s: %d label files found", split, len(files))
        for fp in files:
            rel = str(fp.relative_to(data_dir.parent))
            tasks.append((split, rel, str(fp)))
    return tasks


def load_all_annotations(
    data_dir: Path,
    splits: list[str],
    num_workers: int,
) -> list[tuple]:
    """
    Load all annotations in parallel using multiprocessing.
    Returns a flat list of raw tuples.
    """
    tasks = collect_label_files(data_dir, splits)
    total_files = len(tasks)
    logger.info("Loading %d label files with %d workers ...", total_files, num_workers)

    all_rows: list[tuple] = []
    chunk_size = max(1, total_files // (num_workers * 8))

    t0 = time.monotonic()
    with mp.Pool(processes=num_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_parse_label_file, tasks, chunksize=chunk_size),
            start=1,
        ):
            all_rows.extend(result)
            if i % 5000 == 0 or i == total_files:
                elapsed = time.monotonic() - t0
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(
                    "  Parsed %d/%d files  |  %d annotations so far  |  %.0f files/s",
                    i,
                    total_files,
                    len(all_rows),
                    rate,
                )

    elapsed = time.monotonic() - t0
    logger.info(
        "Loaded %d annotations from %d files in %.1fs",
        len(all_rows),
        total_files,
        elapsed,
    )
    return all_rows


# ---------------------------------------------------------------------------
# Per-class statistics
# ---------------------------------------------------------------------------

def compute_class_stats(
    all_rows: list[tuple],
) -> dict[int, dict[str, Any]]:
    """
    For each class, compute mean/std/min/max for w, h, area, aspect_ratio.
    Returns a dict keyed by class_id.
    """
    # Accumulate per class
    # Using dict of lists to avoid quadratic memory; convert to np arrays once.
    per_class: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"w": [], "h": [], "area": [], "aspect_ratio": []}
    )

    for row in all_rows:
        _, _, _, class_id, _, _, w, h = row
        area = w * h
        ar = (w / h) if h != 0 else float("inf")
        d = per_class[class_id]
        d["w"].append(w)
        d["h"].append(h)
        d["area"].append(area)
        # Exclude inf from aspect_ratio stats (handled separately as anomaly)
        if ar != float("inf"):
            d["aspect_ratio"].append(ar)

    stats: dict[int, dict[str, Any]] = {}
    for class_id, d in per_class.items():
        entry: dict[str, Any] = {"count": len(d["w"]), "class_name": _class_name(class_id)}
        for metric in ("w", "h", "area", "aspect_ratio"):
        # aspect_ratio list may differ in length (excluded inf)
            arr = np.array(d[metric], dtype=np.float64)
            if len(arr) == 0:
                entry[metric] = {"mean": None, "std": None, "min": None, "max": None, "count": 0}
                continue
            entry[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": int(len(arr)),
            }
        stats[class_id] = entry

    return stats


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _sigma_severity(deviation_sigma: float) -> str:
    """Map sigma deviation to a human-readable severity label."""
    if deviation_sigma >= 6.0:
        return "critical"
    if deviation_sigma >= 4.5:
        return "high"
    if deviation_sigma >= SIGMA_THRESHOLD:
        return "medium"
    return "low"


def detect_anomalies(
    all_rows: list[tuple],
    stats: dict[int, dict[str, Any]],
) -> list[AnomalyRecord]:
    """
    Flag each annotation that violates any geometric constraint.
    An annotation may produce multiple AnomalyRecord entries (one per violation type).
    """
    anomalies: list[AnomalyRecord] = []

    # Precompute per-class thresholds for sigma-based checks
    sigma_bounds: dict[int, dict[str, tuple[float, float]]] = {}
    for class_id, s in stats.items():
        bounds: dict[str, tuple[float, float]] = {}
        for metric in ("w", "h", "area"):
            ms = s.get(metric, {})
            mean = ms.get("mean")
            std = ms.get("std")
            if mean is not None and std is not None and std > 0:
                bounds[metric] = (
                    mean - SIGMA_THRESHOLD * std,
                    mean + SIGMA_THRESHOLD * std,
                )
            else:
                bounds[metric] = (None, None)  # type: ignore[assignment]
        sigma_bounds[class_id] = bounds

    for row in all_rows:
        split, label_file, lineno, class_id, cx, cy, w, h = row
        area = w * h
        ar = (w / h) if h != 0.0 else float("inf")

        violations: list[tuple[str, str]] = []  # (anomaly_type, severity)

        # --- Hard geometric checks ---

        # Zero dimension
        if w == 0.0:
            violations.append(("zero_width", "critical"))
        if h == 0.0:
            violations.append(("zero_height", "critical"))

        # Coordinates out of range [0, 1]
        half_w = w / 2.0
        half_h = h / 2.0
        if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
            violations.append(("center_out_of_range", "high"))
        if (cx - half_w) < 0.0 or (cx + half_w) > 1.0:
            violations.append(("box_exceeds_image_width", "medium"))
        if (cy - half_h) < 0.0 or (cy + half_h) > 1.0:
            violations.append(("box_exceeds_image_height", "medium"))

        # Extremely small area
        if 0.0 < area < MIN_AREA:
            violations.append(("extremely_small_area", "high"))

        # Extremely large area
        if area > MAX_AREA:
            violations.append(("extremely_large_area", "high"))

        # Degenerate aspect ratio (skip if zero dimension already flagged)
        if w > 0.0 and h > 0.0:
            if ar > MAX_ASPECT_RATIO:
                violations.append(("degenerate_wide_aspect_ratio", "medium"))
            elif ar < MIN_ASPECT_RATIO:
                violations.append(("degenerate_tall_aspect_ratio", "medium"))

        # Infinite aspect ratio (h == 0, w > 0)
        if ar == float("inf") and w > 0.0:
            violations.append(("infinite_aspect_ratio", "critical"))

        # --- Sigma-based statistical outlier checks ---
        if class_id in sigma_bounds:
            b = sigma_bounds[class_id]
            s = stats[class_id]

            for metric, val in (("w", w), ("h", h), ("area", area)):
                lo, hi = b.get(metric, (None, None))
                ms = s.get(metric, {})
                mean = ms.get("mean")
                std_ = ms.get("std")
                if lo is None or mean is None or std_ is None or std_ == 0:
                    continue
                if val < lo or val > hi:
                    deviation = abs(val - mean) / std_
                    sev = _sigma_severity(deviation)
                    violations.append((f"outlier_{metric}_sigma_{deviation:.1f}", sev))

        # Emit one AnomalyRecord per violation
        for anomaly_type, severity in violations:
            anomalies.append(
                AnomalyRecord(
                    split=split,
                    label_file=label_file,
                    line_number=lineno,
                    class_id=class_id,
                    class_name=_class_name(class_id),
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    area=area,
                    aspect_ratio=ar if ar != float("inf") else -1.0,
                    anomaly_type=anomaly_type,
                    severity=severity,
                )
            )

    return anomalies


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def build_console_summary(
    all_rows: list[tuple],
    anomalies: list[AnomalyRecord],
    stats: dict[int, dict[str, Any]],
) -> str:
    """Build a human-readable console summary string."""
    total_annotations = len(all_rows)
    total_anomalies = len(anomalies)
    unique_files_with_anomalies = len({a.label_file for a in anomalies})

    # Count anomaly types
    type_counts: dict[str, int] = defaultdict(int)
    severity_counts: dict[str, int] = defaultdict(int)
    for a in anomalies:
        base_type = a.anomaly_type.split("_sigma_")[0] if "_sigma_" in a.anomaly_type else a.anomaly_type
        type_counts[base_type] += 1
        severity_counts[a.severity] += 1

    lines = [
        "",
        "=" * 70,
        "  LABEL GEOMETRY AUDIT — SUMMARY",
        "=" * 70,
        f"  Total annotations examined : {total_annotations:>12,}",
        f"  Total anomaly records       : {total_anomalies:>12,}",
        f"  Unique files with anomalies : {unique_files_with_anomalies:>12,}",
        f"  Anomaly rate (records)      : {total_anomalies / max(total_annotations, 1) * 100:>11.3f}%",
        "",
        "  Severity breakdown:",
    ]
    for sev in ("critical", "high", "medium", "low"):
        cnt = severity_counts.get(sev, 0)
        lines.append(f"    {sev:<10}: {cnt:>10,}")

    lines += ["", "  Anomaly type breakdown (top 20):"]
    for atype, cnt in sorted(type_counts.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"    {atype:<45}: {cnt:>8,}")

    lines += ["", "  Per-class annotation counts and area stats:"]
    lines.append(
        f"  {'cls':>3}  {'name':<30}  {'count':>8}  "
        f"{'area_mean':>10}  {'area_std':>10}  {'ar_mean':>8}"
    )
    lines.append("  " + "-" * 76)
    for class_id in sorted(stats.keys()):
        s = stats[class_id]
        name = s["class_name"]
        count = s["count"]
        area_s = s.get("area", {})
        ar_s = s.get("aspect_ratio", {})
        area_mean = area_s.get("mean")
        area_std = area_s.get("std")
        ar_mean = ar_s.get("mean")
        lines.append(
            f"  {class_id:>3}  {name:<30}  {count:>8,}  "
            f"{area_mean if area_mean is not None else 'N/A':>10.6f}  "
            f"{area_std  if area_std  is not None else 'N/A':>10.6f}  "
            f"{ar_mean   if ar_mean   is not None else 'N/A':>8.3f}"
        )

    lines += ["", "=" * 70, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(anomalies: list[AnomalyRecord], output_path: Path) -> None:
    """Write anomaly records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in anomalies:
            writer.writerow(record.to_csv_row())
    logger.info("CSV report written: %s  (%d rows)", output_path, len(anomalies))


def write_json_stats(
    stats: dict[int, dict[str, Any]],
    anomaly_summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Write JSON statistics file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert int keys to strings for JSON serialization
    serializable_stats: dict[str, Any] = {}
    for class_id, s in stats.items():
        serializable_stats[str(class_id)] = s

    payload = {
        "audit_meta": {
            "sigma_threshold": SIGMA_THRESHOLD,
            "min_area_threshold": MIN_AREA,
            "max_area_threshold": MAX_AREA,
            "max_aspect_ratio": MAX_ASPECT_RATIO,
            "min_aspect_ratio": MIN_ASPECT_RATIO,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
        },
        "anomaly_summary": anomaly_summary,
        "per_class_stats": serializable_stats,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("JSON stats written: %s", output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO label geometry audit — flags bounding box anomalies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/yolo_harmony_v2_phase8_final"),
        help="Path to dataset root containing train/ and val/ subdirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write audit reports.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to audit (default: train val).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 2),
        help="Number of worker processes for parallel file loading.",
    )
    parser.add_argument(
        "--csv-name",
        default="label_audit_geometry.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--json-name",
        default="label_audit_geometry_stats.json",
        help="Output JSON stats filename.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    splits: list[str] = args.splits
    num_workers: int = max(1, args.workers)

    if not data_dir.is_dir():
        logger.error("Data directory does not exist: %s", data_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name
    json_path = output_dir / args.json_name

    logger.info("Data dir   : %s", data_dir.resolve())
    logger.info("Output dir : %s", output_dir.resolve())
    logger.info("Splits     : %s", splits)
    logger.info("Workers    : %d", num_workers)

    # ---- Step 1: Load all annotations ----
    t_start = time.monotonic()
    all_rows = load_all_annotations(data_dir, splits, num_workers)

    if not all_rows:
        logger.error("No annotations found. Check --data-dir and --splits.")
        return 1

    # ---- Step 2: Compute per-class statistics ----
    logger.info("Computing per-class statistics ...")
    stats = compute_class_stats(all_rows)

    # ---- Step 3: Detect anomalies ----
    logger.info("Detecting anomalies (%d annotations) ...", len(all_rows))
    t_detect = time.monotonic()
    anomalies = detect_anomalies(all_rows, stats)
    logger.info(
        "Anomaly detection completed in %.1fs — %d records flagged",
        time.monotonic() - t_detect,
        len(anomalies),
    )

    # ---- Step 4: Build summary ----
    type_counts: dict[str, int] = defaultdict(int)
    severity_counts: dict[str, int] = defaultdict(int)
    class_anomaly_counts: dict[int, int] = defaultdict(int)
    for a in anomalies:
        base_type = (
            a.anomaly_type.split("_sigma_")[0]
            if "_sigma_" in a.anomaly_type
            else a.anomaly_type
        )
        type_counts[base_type] += 1
        severity_counts[a.severity] += 1
        class_anomaly_counts[a.class_id] += 1

    anomaly_summary: dict[str, Any] = {
        "total_annotations": len(all_rows),
        "total_anomaly_records": len(anomalies),
        "unique_files_with_anomalies": len({a.label_file for a in anomalies}),
        "anomaly_rate_pct": round(len(anomalies) / max(len(all_rows), 1) * 100, 4),
        "by_severity": dict(severity_counts),
        "by_anomaly_type": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "by_class_id": {
            str(cid): {
                "class_name": _class_name(cid),
                "anomaly_records": cnt,
            }
            for cid, cnt in sorted(class_anomaly_counts.items())
        },
    }

    # ---- Step 5: Console summary ----
    summary_str = build_console_summary(all_rows, anomalies, stats)
    print(summary_str)

    # ---- Step 6: Write outputs ----
    write_csv(anomalies, csv_path)
    write_json_stats(stats, anomaly_summary, json_path)

    total_elapsed = time.monotonic() - t_start
    logger.info("Audit complete in %.1fs", total_elapsed)
    logger.info("CSV  -> %s", csv_path.resolve())
    logger.info("JSON -> %s", json_path.resolve())

    return 0


if __name__ == "__main__":
    sys.exit(main())
