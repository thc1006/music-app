"""
CV-based Notehead Detector for OpenScore Sheet Music

Uses classical computer vision (morphology + connected components) to find
noteheads directly in sheet music images — no ML model needed.

Why this works:
  - Filled noteheads are small dark ellipses (~15-25px) on/between staff lines
  - Hollow noteheads are small ring shapes
  - These features are visually unique and detectable with blob analysis

Why not ML-based:
  - Phase 6/7/8 all failed because training labels had wrong positions
  - CV detects directly from pixels — no label quality dependency

Usage:
    from notehead_detector_cv import detect_noteheads
    detections = detect_noteheads("path/to/image.png")
    # Returns: [(cls, cx, cy, w, h), ...] in YOLO normalized format
"""
import cv2
import numpy as np
from pathlib import Path


def detect_noteheads(
    image_path: str,
    min_area: int = 40,
    max_area: int = 1500,
    min_circularity: float = 0.3,
    min_solidity: float = 0.5,
    padding_ratio: float = 0.1,
) -> list[tuple]:
    """Detect filled and hollow noteheads in a sheet music image.

    Pipeline:
      1. CLAHE contrast enhancement + adaptive threshold
      2. Detect and remove staff lines (horizontal morphology)
      3. Connected component analysis on remaining blobs
      4. Filter by area, circularity, aspect ratio
      5. Classify as filled (cls=0) or hollow (cls=1)

    Args:
        image_path: path to sheet music image
        min_area: minimum blob area in pixels²
        max_area: maximum blob area in pixels²
        min_circularity: minimum 4π·area/perimeter² (circle=1.0)
        min_solidity: minimum area/convex_hull_area
        padding_ratio: bbox padding as fraction of blob size

    Returns:
        List of (cls, cx, cy, w, h) in normalized YOLO format.
        cls: 0 = notehead_filled, 1 = notehead_hollow
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    H, W = img.shape

    # Step 0: CLAHE contrast enhancement (recovers noteheads in low-contrast areas)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Step 1: Adaptive threshold
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=8
    )

    # Step 2: Remove staff lines using horizontal morphology
    # Staff lines are long horizontal structures
    staff_kernel_len = max(W // 30, 20)  # ~65px for 1960-wide images
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (staff_kernel_len, 1))
    staff_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Subtract staff lines from binary image
    no_staff = cv2.subtract(binary, staff_lines)

    # Step 3: Light cleanup — close small gaps from staff line removal
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(no_staff, cv2.MORPH_CLOSE, close_kernel)

    # Step 4: Find contours
    contours, hierarchy = cv2.findContours(
        cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]  # shape: (N, 4) — [next, prev, child, parent]

    detections = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Bounding rect
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        # Aspect ratio filter (noteheads are roughly circular/oval)
        aspect = w / h
        if aspect < 0.35 or aspect > 2.8:
            continue

        # Circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        # Solidity (area / convex hull area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area

        # Skip child contours (inner holes)
        parent = hierarchy[i][3]
        if parent >= 0:
            continue

        if solidity < min_solidity:
            continue

        # Classify filled (cls=0) vs hollow (cls=1)
        # Use interior brightness on ORIGINAL grayscale — more robust than
        # contour hierarchy (which breaks when staff removal cuts hollow rings)
        cls = 0  # default: filled
        if w >= 6 and h >= 6:
            # Crop interior region (center 50%) from original grayscale
            ix1 = x + w // 4
            iy1 = y + h // 4
            ix2 = x + 3 * w // 4
            iy2 = y + 3 * h // 4
            interior = img[iy1:iy2, ix1:ix2]
            if interior.size > 0:
                interior_mean = float(np.mean(interior))
                # Hollow noteheads have bright interior (>180), filled have dark (<100)
                if interior_mean > 170:
                    cls = 1  # hollow

        # Add padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        bx = max(0, x - pad_x)
        by = max(0, y - pad_y)
        bw = min(W - bx, w + 2 * pad_x)
        bh = min(H - by, h + 2 * pad_y)

        # NOTE (2026-04-07): Removed min_box_px floor. Was min_w_px=49, min_h_px=69.
        # The floor silently overrode padding_ratio changes, producing 85×90px boxes
        # when raw blobs are 21×21px. GT is 23×25px, so oversized boxes → IoU<0.1.
        # Phase 6 failure was about training labels, not inference boxes; this floor
        # was misapplied. Raw blob + small padding = correct output size.
        # Sanity minimum to catch degenerate blobs (pure noise):
        if bw < 8:
            bw = 8
        if bh < 8:
            bh = 8
        bw = min(W - bx, bw)
        bh = min(H - by, bh)

        # Convert to normalized YOLO format (cx, cy, w, h)
        cx_norm = (bx + bw / 2) / W
        cy_norm = (by + bh / 2) / H
        w_norm = bw / W
        h_norm = bh / H

        detections.append((cls, float(cx_norm), float(cy_norm),
                           float(w_norm), float(h_norm)))

    return detections


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python notehead_detector_cv.py <image_path>")
        sys.exit(1)

    dets = detect_noteheads(sys.argv[1])
    print(f"Found {len(dets)} noteheads")
    for cls, cx, cy, w, h in dets[:10]:
        print(f"  cls={cls} cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")
