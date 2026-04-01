#!/usr/bin/env python3
"""
Offline copy-paste augmentation for tail (rare) classes.

This script identifies the bottom-N classes by instance count, extracts all
bounding-box crops that belong to those classes from the training images, then
randomly pastes the crops onto other training images (head-class-dominant images
are preferred as backgrounds) to boost each tail class toward a target count.

Original images and labels are never modified.  All output lives under a new
directory tree so the two datasets can coexist safely.

Usage
-----
    python tail_copy_paste.py \\
        --data-dir  datasets/yolo_harmony_v2_phase8_final \\
        --output-dir datasets/yolo_harmony_v2_phase8_cleaned \\
        --target-count 10000 \\
        --tail-n 8 \\
        --seed 42

Output layout
-------------
    <output-dir>/
        train_augmented/
            images/   aug_<original_stem>_<idx>.png
            labels/   aug_<original_stem>_<idx>.txt
        augmentation_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required.  Install it with:  pip install Pillow")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy is required.  Install it with:  pip install numpy")
    sys.exit(1)

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

# Minimum pixel size (either dimension) below which a crop is discarded.
MIN_CROP_PX: int = 5

# Maximum IoU with an existing box allowed when choosing a paste location.
MAX_PASTE_IOU: float = 0.20

# How many placement attempts before we give up on a particular patch/image pair.
MAX_PLACEMENT_ATTEMPTS: int = 40

# How many background images to try before moving on to a new patch.
MAX_BG_ATTEMPTS: int = 10

# Batch size for iterating images when building the patch pool (memory control).
PATCH_EXTRACTION_BATCH: int = 500


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class YoloBox:
    """A single YOLO-format annotation row (cx cy w h are normalized 0-1)."""
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_pixel(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) in pixel coordinates."""
        x1 = int((self.cx - self.w / 2) * img_w)
        y1 = int((self.cy - self.h / 2) * img_h)
        x2 = int((self.cx + self.w / 2) * img_w)
        y2 = int((self.cy + self.h / 2) * img_h)
        return (
            max(0, x1),
            max(0, y1),
            min(img_w, x2),
            min(img_h, y2),
        )

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class Patch:
    """A PIL crop extracted from a training image, ready for pasting."""
    class_id: int
    image: Image.Image          # RGBA so alpha channel carries original shape
    source_stem: str            # for traceability


@dataclass
class ImageRecord:
    """Everything we know about a training image and its annotations."""
    image_path: Path
    label_path: Path
    boxes: list[YoloBox]

    @property
    def stem(self) -> str:
        return self.image_path.stem

    def dominant_class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for box in self.boxes:
            counts[box.class_id] += 1
        return dict(counts)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_label_file(label_path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if not label_path.exists():
        return boxes
    with label_path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                boxes.append(YoloBox(cls, cx, cy, w, h))
            except ValueError:
                logger.warning("Skipping malformed label line: %s", raw_line.rstrip())
    return boxes


def write_label_file(label_path: Path, boxes: list[YoloBox]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as fh:
        for box in boxes:
            fh.write(box.to_yolo_line() + "\n")


def load_image_records(images_dir: Path, labels_dir: Path) -> list[ImageRecord]:
    """Scan all images in images_dir and pair them with their label files."""
    records: list[ImageRecord] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = read_label_file(label_path)
        records.append(ImageRecord(img_path, label_path, boxes))
    return records


# ---------------------------------------------------------------------------
# Instance-count analysis
# ---------------------------------------------------------------------------

def count_instances(records: list[ImageRecord]) -> dict[int, int]:
    """Return {class_id: total_instance_count} across all records."""
    counts: dict[int, int] = defaultdict(int)
    for rec in records:
        for box in rec.boxes:
            counts[box.class_id] += 1
    return dict(counts)


def select_tail_classes(
    counts: dict[int, int],
    tail_n: int,
    all_class_ids: set[int],
) -> list[int]:
    """Return the bottom-tail_n class IDs ordered by ascending instance count.

    Classes with zero instances in the dataset are also included (they rank
    lowest of all).
    """
    # Merge dataset classes and any zero-count classes from all_class_ids.
    full_counts: dict[int, int] = {cls: 0 for cls in all_class_ids}
    full_counts.update(counts)
    ordered = sorted(full_counts.items(), key=lambda x: x[1])
    return [cls_id for cls_id, _ in ordered[:tail_n]]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union for two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_valid_paste_location(
    patch_w: int,
    patch_h: int,
    canvas_w: int,
    canvas_h: int,
    existing_boxes_px: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    """Sample a random top-left (x, y) such that the pasted patch does not
    overlap existing boxes by more than MAX_PASTE_IOU.

    Returns None if MAX_PLACEMENT_ATTEMPTS are exhausted.
    """
    if patch_w > canvas_w or patch_h > canvas_h:
        return None
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        x1 = rng.randint(0, canvas_w - patch_w)
        y1 = rng.randint(0, canvas_h - patch_h)
        x2 = x1 + patch_w
        y2 = y1 + patch_h
        candidate = (x1, y1, x2, y2)
        overlap_ok = all(
            iou(candidate, eb) <= MAX_PASTE_IOU for eb in existing_boxes_px
        )
        if overlap_ok:
            return (x1, y1)
    return None


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches_for_class(
    records: list[ImageRecord],
    class_id: int,
    rng: random.Random,
) -> list[Patch]:
    """Extract PIL crops for every bounding box of class_id in the dataset."""
    patches: list[Patch] = []
    batch: list[ImageRecord] = []

    def _process_batch(batch: list[ImageRecord]) -> None:
        for rec in batch:
            relevant = [b for b in rec.boxes if b.class_id == class_id]
            if not relevant:
                continue
            try:
                img = Image.open(rec.image_path).convert("RGBA")
            except Exception as exc:
                logger.warning("Cannot open %s: %s", rec.image_path, exc)
                continue
            iw, ih = img.size
            for box in relevant:
                x1, y1, x2, y2 = box.to_pixel(iw, ih)
                pw, ph = x2 - x1, y2 - y1
                if pw < MIN_CROP_PX or ph < MIN_CROP_PX:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                patches.append(Patch(class_id=class_id, image=crop, source_stem=rec.stem))

    for i, rec in enumerate(records):
        batch.append(rec)
        if len(batch) >= PATCH_EXTRACTION_BATCH or i == len(records) - 1:
            _process_batch(batch)
            batch = []

    rng.shuffle(patches)
    return patches


# ---------------------------------------------------------------------------
# Background image selection
# ---------------------------------------------------------------------------

def build_head_class_preference(
    records: list[ImageRecord],
    tail_class_ids: set[int],
) -> list[ImageRecord]:
    """Return records sorted so that images dominated by head-class boxes come
    first.  This gives tail-class patches a visually plausible musical context.
    """

    def head_fraction(rec: ImageRecord) -> float:
        if not rec.boxes:
            return 0.0
        head_count = sum(1 for b in rec.boxes if b.class_id not in tail_class_ids)
        return head_count / len(rec.boxes)

    return sorted(records, key=head_fraction, reverse=True)


# ---------------------------------------------------------------------------
# Single augmented-image generation
# ---------------------------------------------------------------------------

def generate_augmented_image(
    bg_record: ImageRecord,
    patches: list[Patch],
    class_id: int,
    n_pastes: int,
    rng: random.Random,
) -> tuple[Image.Image, list[YoloBox]] | None:
    """Paste n_pastes copies of class_id patches onto bg_record's image.

    Returns (augmented_PIL_image, updated_box_list) or None if no valid
    placement could be found even after trying several patches.
    """
    try:
        canvas = Image.open(bg_record.image_path).convert("RGBA")
    except Exception as exc:
        logger.warning("Cannot open background %s: %s", bg_record.image_path, exc)
        return None

    cw, ch = canvas.size
    new_boxes = list(bg_record.boxes)  # shallow copy, boxes are immutable dataclasses
    existing_px = [b.to_pixel(cw, ch) for b in new_boxes]

    pasted = 0
    patch_indices = list(range(len(patches)))
    rng.shuffle(patch_indices)

    for pidx in patch_indices:
        if pasted >= n_pastes:
            break
        patch = patches[pidx]
        pw, ph = patch.image.size
        if pw < MIN_CROP_PX or ph < MIN_CROP_PX:
            continue

        loc = find_valid_paste_location(pw, ph, cw, ch, existing_px, rng)
        if loc is None:
            continue

        x1, y1 = loc
        x2, y2 = x1 + pw, y1 + ph

        # Paste using the alpha channel as mask so patch edges blend.
        canvas.paste(patch.image, (x1, y1), patch.image)

        # Record new annotation.
        cx = (x1 + x2) / 2 / cw
        cy = (y1 + y2) / 2 / ch
        nw = pw / cw
        nh = ph / ch
        new_box = YoloBox(class_id, cx, cy, nw, nh)
        new_boxes.append(new_box)
        existing_px.append((x1, y1, x2, y2))
        pasted += 1

    if pasted == 0:
        return None

    return canvas.convert("RGB"), new_boxes


# ---------------------------------------------------------------------------
# Main augmentation loop
# ---------------------------------------------------------------------------

def augment_tail_class(
    class_id: int,
    class_name: str,
    current_count: int,
    target_count: int,
    patches: list[Patch],
    bg_records: list[ImageRecord],
    out_images_dir: Path,
    out_labels_dir: Path,
    rng: random.Random,
    global_idx: dict[int, int],
) -> int:
    """Generate augmented images until class_id reaches target_count.

    Returns the number of new instances successfully pasted.
    """
    if not patches:
        logger.warning("Class %d (%s): no patches extracted, skipping.", class_id, class_name)
        return 0

    needed = max(0, target_count - current_count)
    if needed == 0:
        logger.info("Class %d (%s): already at %d >= %d, skipping.", class_id, class_name, current_count, target_count)
        return 0

    logger.info(
        "Class %d (%s): current=%d  target=%d  need=%d  patches=%d",
        class_id, class_name, current_count, target_count, needed, len(patches),
    )

    # How many pastes per augmented image?  Aim for 3-6 per image so we do not
    # bloat the dataset too much.
    pastes_per_image = max(1, min(6, needed // max(1, len(bg_records) // 4)))

    added = 0
    bg_pool = list(bg_records)  # will cycle

    while added < needed:
        bg_record = rng.choice(bg_pool)
        remaining = needed - added
        n_pastes = min(pastes_per_image, remaining)

        result = generate_augmented_image(bg_record, patches, class_id, n_pastes, rng)
        if result is None:
            # Try a different background next time.
            continue

        aug_img, aug_boxes = result
        idx = global_idx[class_id]
        global_idx[class_id] += 1

        out_stem = f"aug_{bg_record.stem}_{idx}"
        img_out_path = out_images_dir / f"{out_stem}.png"
        lbl_out_path = out_labels_dir / f"{out_stem}.txt"

        try:
            aug_img.save(img_out_path)
        except Exception as exc:
            logger.warning("Failed to save %s: %s", img_out_path, exc)
            continue

        write_label_file(lbl_out_path, aug_boxes)

        # Count how many of class_id were actually added.
        actually_added = sum(1 for b in aug_boxes if b.class_id == class_id) - sum(
            1 for b in bg_record.boxes if b.class_id == class_id
        )
        added += max(0, actually_added)

    logger.info("Class %d (%s): finished, added ~%d instances.", class_id, class_name, added)
    return added


# ---------------------------------------------------------------------------
# CLI and entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline copy-paste augmentation for tail (rare) classes in a "
            "YOLO-format dataset.  Original files are never modified."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Root directory of the YOLO dataset (contains train/ and val/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Output root directory.  Augmented files are written to "
            "<output-dir>/train_augmented/{images,labels}/."
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=10_000,
        help="Target number of effective instances per tail class (default: 10000).",
    )
    parser.add_argument(
        "--tail-n",
        type=int,
        default=8,
        help="Number of bottom classes (by instance count) to augment (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--min-crop-px",
        type=int,
        default=MIN_CROP_PX,
        help=f"Minimum pixel size for a crop to be used (default: {MIN_CROP_PX}).",
    )
    parser.add_argument(
        "--max-iou",
        type=float,
        default=MAX_PASTE_IOU,
        help=f"Max IoU allowed with existing boxes when placing a patch (default: {MAX_PASTE_IOU}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Allow callers to override module-level constants via CLI.
    global MIN_CROP_PX, MAX_PASTE_IOU
    MIN_CROP_PX = args.min_crop_px
    MAX_PASTE_IOU = args.max_iou

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ---- Resolve paths -------------------------------------------------------
    data_dir: Path = args.data_dir.resolve()
    train_images_dir = data_dir / "train" / "images"
    train_labels_dir = data_dir / "train" / "labels"

    for p in (train_images_dir, train_labels_dir):
        if not p.exists():
            logger.error("Expected directory does not exist: %s", p)
            sys.exit(1)

    out_dir: Path = args.output_dir.resolve()
    out_images_dir = out_dir / "train_augmented" / "images"
    out_labels_dir = out_dir / "train_augmented" / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all training records -------------------------------------------
    logger.info("Scanning training images in %s …", train_images_dir)
    records = load_image_records(train_images_dir, train_labels_dir)
    logger.info("Found %d training images.", len(records))

    if not records:
        logger.error("No training images found.  Aborting.")
        sys.exit(1)

    # ---- Count instances per class -------------------------------------------
    instance_counts = count_instances(records)
    all_class_ids = set(instance_counts.keys())

    logger.info("Instance counts per class:")
    for cls_id, cnt in sorted(instance_counts.items(), key=lambda x: x[1]):
        logger.info("  class %2d: %7d", cls_id, cnt)

    # ---- Select tail classes -------------------------------------------------
    tail_class_ids = select_tail_classes(instance_counts, args.tail_n, all_class_ids)
    tail_set = set(tail_class_ids)
    logger.info("Tail classes selected (bottom %d): %s", args.tail_n, tail_class_ids)

    # ---- Read class names from YAML if available ----------------------------
    yaml_path = data_dir / "harmony_phase8_final.yaml"
    class_names: dict[int, str] = {}
    if yaml_path.exists():
        try:
            import re
            text = yaml_path.read_text()
            for m in re.finditer(r"(\d+):\s+(\S+)", text):
                class_names[int(m.group(1))] = m.group(2)
        except Exception:
            pass

    def cls_name(cls_id: int) -> str:
        return class_names.get(cls_id, f"class_{cls_id}")

    # ---- Build a sorted background pool (head-class-dominant first) ----------
    bg_records = build_head_class_preference(records, tail_set)

    # ---- Augment each tail class ---------------------------------------------
    global_idx: dict[int, int] = defaultdict(int)
    report: dict = {
        "args": {
            "data_dir": str(data_dir),
            "output_dir": str(out_dir),
            "target_count": args.target_count,
            "tail_n": args.tail_n,
            "seed": args.seed,
        },
        "tail_classes": {},
        "total_augmented_images": 0,
    }

    for class_id in tail_class_ids:
        name = cls_name(class_id)
        current = instance_counts.get(class_id, 0)

        logger.info("=" * 60)
        logger.info("Extracting patches for class %d (%s) …", class_id, name)
        patches = extract_patches_for_class(records, class_id, rng)
        logger.info("  Extracted %d usable patches.", len(patches))

        if not patches:
            report["tail_classes"][str(class_id)] = {
                "name": name,
                "before": current,
                "added": 0,
                "patches_found": 0,
                "status": "skipped_no_patches",
            }
            continue

        added = augment_tail_class(
            class_id=class_id,
            class_name=name,
            current_count=current,
            target_count=args.target_count,
            patches=patches,
            bg_records=bg_records,
            out_images_dir=out_images_dir,
            out_labels_dir=out_labels_dir,
            rng=rng,
            global_idx=global_idx,
        )

        report["tail_classes"][str(class_id)] = {
            "name": name,
            "before": current,
            "added": added,
            "after_estimated": current + added,
            "patches_found": len(patches),
            "status": "ok",
        }

    # ---- Tally augmented images ----------------------------------------------
    total_aug = sum(global_idx.values())
    report["total_augmented_images"] = total_aug
    logger.info("=" * 60)
    logger.info("Total augmented image files written: %d", total_aug)
    logger.info("Output directory: %s", out_dir)

    # ---- Write report --------------------------------------------------------
    report_path = out_dir / "augmentation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report written to %s", report_path)

    # ---- Final summary -------------------------------------------------------
    logger.info("")
    logger.info("Summary:")
    logger.info("  %-6s  %-30s  %8s  %8s  %8s", "Class", "Name", "Before", "Added", "After~")
    for cls_id in tail_class_ids:
        entry = report["tail_classes"].get(str(cls_id), {})
        logger.info(
            "  %-6d  %-30s  %8d  %8d  %8d",
            cls_id,
            entry.get("name", "?"),
            entry.get("before", 0),
            entry.get("added", 0),
            entry.get("after_estimated", entry.get("before", 0)),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
