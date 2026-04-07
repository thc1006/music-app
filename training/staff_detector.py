"""
Staff Line Detector — classical CV

Detects the 5 horizontal lines of each staff in a sheet music image.
Uses horizontal projection profile + peak detection + clustering into
5-line groups.

Output: list[Staff], where each Staff exposes:
  - line_ys: list of 5 y-coordinates (top to bottom)
  - spacing: average line-to-line distance
  - x_min, x_max: staff horizontal bounds
  - y_to_step(y): converts a pixel y to a diatonic step index
                   (top line = 0, next space = 1, next line = 2, ... bottom line = 8)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Staff:
    """A single 5-line music staff."""
    line_ys: list[float]          # 5 y-coordinates, sorted ascending (top to bottom)
    spacing: float                 # average line-to-line spacing
    x_min: int                     # staff left bound
    x_max: int                     # staff right bound

    def y_to_step(self, y: float) -> int:
        """Convert a pixel y-coordinate to a diatonic step from top line.

        Top line  = step 0
        Top space = step 1
        2nd line  = step 2
        ...
        Bottom line = step 8

        Values above top line are negative; below bottom line are > 8.
        """
        if self.spacing <= 0:
            return 0
        top_y = self.line_ys[0]
        # 2 diatonic steps per line spacing (one line + one space)
        steps = (y - top_y) / (self.spacing / 2)
        return int(round(steps))


def _find_staff_line_y_peaks(
    binary: np.ndarray,
    min_line_width_ratio: float = 0.5,
) -> list[int]:
    """Find y-coordinates where horizontal dark content spans most of the image width.

    Staff lines are long horizontal runs. Using horizontal projection:
    count black pixels per row; staff lines produce tall peaks.
    """
    H, W = binary.shape
    # Sum black pixels per row (binary is inverted: foreground=255)
    row_sums = np.sum(binary, axis=1) // 255  # number of foreground pixels per row
    min_line_width = int(W * min_line_width_ratio)
    peaks = np.where(row_sums >= min_line_width)[0]

    # Merge adjacent peaks (staff lines are usually 1-3 px thick)
    if len(peaks) == 0:
        return []
    merged: list[int] = [int(peaks[0])]
    for p in peaks[1:]:
        if p - merged[-1] <= 3:
            merged[-1] = int(p)  # keep latest in the cluster
        else:
            merged.append(int(p))
    return merged


def _cluster_lines_into_staves(
    line_ys: list[int],
    tolerance_factor: float = 0.3,
) -> list[list[int]]:
    """Group line y-coords into staves of 5 lines each.

    Strategy: compute spacings between consecutive lines; within a staff,
    spacings are roughly uniform. Between staves, there's a bigger gap.
    Use median spacing as reference and split where the gap exceeds
    (1 + tolerance_factor) × median.
    """
    if len(line_ys) < 5:
        return []

    # Compute all pairwise gaps
    gaps = [line_ys[i + 1] - line_ys[i] for i in range(len(line_ys) - 1)]
    median_gap = float(np.median(gaps))

    # Split points: gaps significantly larger than median are inter-staff
    threshold = median_gap * (1.0 + tolerance_factor * 5)  # heuristic

    groups: list[list[int]] = []
    current: list[int] = [line_ys[0]]
    for i, gap in enumerate(gaps):
        if gap > threshold:
            groups.append(current)
            current = [line_ys[i + 1]]
        else:
            current.append(line_ys[i + 1])
    groups.append(current)

    # Keep only groups with exactly 5 lines (real staves)
    # If a group has >5, it might contain multiple staves; try to split
    staves: list[list[int]] = []
    for group in groups:
        if len(group) == 5:
            staves.append(group)
        elif len(group) > 5:
            # Try to split into 5-line chunks
            # Simple heuristic: look for the largest internal gap
            while len(group) >= 5:
                staves.append(group[:5])
                group = group[5:]
        # Groups with <5 lines are discarded (probably noise)
    return staves


def _estimate_staff_bounds(
    binary: np.ndarray,
    line_ys: list[int],
    margin: int = 5,
) -> tuple[int, int]:
    """Find left/right extent of the staff by looking at the lines' actual length.

    Sample a strip of pixels around each staff line and find where the
    dark content starts/ends.
    """
    H, W = binary.shape
    x_mins: list[int] = []
    x_maxs: list[int] = []
    for y in line_ys:
        y_lo = max(0, y - 1)
        y_hi = min(H, y + 2)
        strip = binary[y_lo:y_hi, :]
        col_sums = np.sum(strip, axis=0)
        nonzero = np.where(col_sums > 0)[0]
        if len(nonzero) > 0:
            x_mins.append(int(nonzero[0]))
            x_maxs.append(int(nonzero[-1]))
    if not x_mins:
        return 0, W
    x_min = max(0, min(x_mins) - margin)
    x_max = min(W, max(x_maxs) + margin)
    return x_min, x_max


def detect_staves(image_path: str) -> list[Staff]:
    """Detect all 5-line staves in a sheet music image.

    Args:
        image_path: path to the input image

    Returns:
        list of Staff objects, one per detected staff
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    H, W = img.shape

    # Binarize: dark ink = 255
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find y-coordinates of staff lines via horizontal projection
    line_ys = _find_staff_line_y_peaks(binary, min_line_width_ratio=0.5)
    if len(line_ys) < 5:
        return []

    # Group into 5-line staves
    staff_groups = _cluster_lines_into_staves(line_ys)

    staves: list[Staff] = []
    for group in staff_groups:
        if len(group) != 5:
            continue
        spacings = [group[i + 1] - group[i] for i in range(4)]
        avg_spacing = sum(spacings) / 4
        if avg_spacing <= 0:
            continue
        x_min, x_max = _estimate_staff_bounds(binary, group)
        staves.append(Staff(
            line_ys=[float(y) for y in group],
            spacing=float(avg_spacing),
            x_min=x_min,
            x_max=x_max,
        ))
    return staves


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python staff_detector.py <image_path>")
        sys.exit(1)

    staves = detect_staves(sys.argv[1])
    print(f"Detected {len(staves)} staves:")
    for i, s in enumerate(staves):
        print(f"  Staff {i}: lines={[int(y) for y in s.line_ys]} "
              f"spacing={s.spacing:.1f} x=[{s.x_min}, {s.x_max}]")
