"""
TDD Red Phase: Phase 5 Teacher Pseudo-Label Pipeline Tests

These tests define the acceptance criteria for the pseudo-labeling pipeline.
All tests should FAIL initially (Red), then pass after implementation (Green).

Pipeline overview:
  Phase 5 model predicts on OpenScore images (large glyph-group boxes)
  → post-process to shrink boxes to TAL-learnable size
  → create relabeled dataset (DoReMi original + OpenScore pseudo-labeled)

Key lessons from Phase 6/7 failures:
  - Phase 6: shrunk to 23×25px (0.012 norm) → 11px at inference → TAL can't learn → 0 predictions
  - Phase 7: rule-based fix for 30 classes → 26/32 WORSE
  - Phase 6: all OpenScore noteheads same size → unnatural
"""
import pytest
from pathlib import Path
from collections import defaultdict
import statistics

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
TRAINING_DIR = PROJECT_ROOT / "training"

PHASE5_WEIGHTS = PROJECT_ROOT / "runs/detect/harmony_omr_v2_phase5/nostem_v1_stage2/weights/best.pt"
DOREMI_DATASET = TRAINING_DIR / "datasets/yolo_harmony_v2_doremi_only"
PHASE6_DATASET = TRAINING_DIR / "datasets/yolo_harmony_v2_phase6_fixed"
PSEUDO_DATASET = TRAINING_DIR / "datasets/yolo_harmony_v2_pseudo_v1"

NUM_CLASSES = 32
NAMES = [
    'notehead_filled', 'notehead_hollow', 'beam', 'flag_8th', 'flag_16th',
    'flag_32nd', 'augmentation_dot', 'tie', 'clef_treble', 'clef_bass',
    'clef_alto', 'clef_tenor', 'accidental_sharp', 'accidental_flat',
    'accidental_natural', 'accidental_double_sharp', 'accidental_double_flat',
    'rest_whole', 'rest_half', 'rest_quarter', 'rest_8th', 'rest_16th',
    'barline', 'barline_double', 'barline_final', 'barline_repeat',
    'time_signature', 'key_signature', 'dynamic_loud', 'dynamic_soft',
    'fermata', 'ledger_line',
]

# Phase 6 failure: 0.012 normalized → 11px at 1280 inference → TAL can't learn
# Guard: flag any box where BOTH w and h are below this as dangerous
PHASE6_DANGER_THRESHOLD = 0.012

# Pseudo-label target range for noteheads:
#   DoReMi GT = 0.011×0.007 (works on DoReMi, but failed on OpenScore at same size)
#   We target 3-8× DoReMi reference → ~0.03-0.09 normalized
#   This gives ~38-115px at 1280 inference → TAL can comfortably handle
NOTEHEAD_MIN_DIM = 0.025  # ~32px at inference, >2× Phase 6 failure
NOTEHEAD_MAX_DIM = 0.12   # ~154px at inference, well below glyph-group

# Generic max for all classes: smaller than glyph-group (~0.25) to be useful
GENERIC_MAX_BOX_NORM = 0.15

# DoReMi reference sizes (median normalized w, h from training set)
DOREMI_REF = {
    'notehead_filled':  (0.01131, 0.00685),
    'notehead_hollow':  (0.01131, 0.00685),
    'beam':             (0.03596, 0.00742),
    'flag_8th':         (0.00889, 0.01969),
    'flag_16th':        (0.00970, 0.01941),
    'flag_32nd':        (0.00889, 0.02312),
    'augmentation_dot': (0.00364, 0.00258),
    'tie':              (0.03615, 0.00932),
    'clef_treble':      (0.02263, 0.04195),
    'clef_bass':        (0.02343, 0.03125),
    'accidental_sharp': (0.00848, 0.01684),
    'accidental_flat':  (0.00768, 0.01484),
    'accidental_natural': (0.00566, 0.01627),
    'rest_whole':       (0.00970, 0.00343),
    'rest_half':        (0.00970, 0.00659),
    'rest_quarter':     (0.00929, 0.01798),
    'rest_8th':         (0.00848, 0.01027),
    'rest_16th':        (0.01091, 0.01627),
    'barline':          (0.01500, 0.08505),
    'barline_repeat':   (0.01953, 0.11250),
    'time_signature':   (0.01455, 0.01256),
    'key_signature':    (0.01650, 0.05008),
    'dynamic_loud':     (0.02000, 0.02000),
    'dynamic_soft':     (0.01535, 0.00999),
    'ledger_line':      (0.01177, 0.00508),
}

# Classes with unreliable DoReMi reference (0.9×0.9 = glyph-group, not tight bbox)
UNRELIABLE_DOREMI_CLASSES = {'clef_alto', 'clef_tenor', 'accidental_double_sharp',
                             'accidental_double_flat', 'barline_double',
                             'barline_final', 'fermata'}

# Inference IoU — must match pipeline's actual setting (production uses 0.55)
INFERENCE_IOU = 0.55


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _read_yolo_labels(label_dir: Path):
    """Read all YOLO label files, return dict: {stem: [(cls, cx, cy, w, h), ...]}"""
    result = {}
    for lf in sorted(label_dir.glob("*.txt")):
        annotations = []
        if lf.stat().st_size == 0:
            result[lf.stem] = annotations
            continue
        for line in lf.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                annotations.append((cls_id, cx, cy, w, h))
        result[lf.stem] = annotations
    return result


def _is_openscore(filename: str) -> bool:
    """OpenScore filenames contain 'lg-' (LilyPond-generated)."""
    return "lg-" in filename


# ──────────────────────────────────────────────────────────────────────
# Unit Tests: shrink_box() logic (pure function, no model needed)
# ──────────────────────────────────────────────────────────────────────

class TestShrinkBoxLogic:
    """Test the core box shrinking function in isolation."""

    @pytest.fixture
    def shrink_fn(self):
        """Import the shrink function from the pipeline module."""
        from create_pseudo_labels import shrink_box
        return shrink_box

    def test_notehead_shrunk_within_target_range(self, shrink_fn):
        """Notehead box must be in [NOTEHEAD_MIN_DIM, NOTEHEAD_MAX_DIM].

        Too small → Phase 6 failure (TAL can't learn).
        Too large → still glyph-group, defeats purpose.
        """
        pred_box = (0.50, 0.30, 0.25, 0.20)  # cx, cy, w, h (normalized)

        for cls_id in [0, 1]:  # notehead_filled, notehead_hollow
            _, _, new_w, new_h = shrink_fn(cls_id, *pred_box)
            assert new_w >= NOTEHEAD_MIN_DIM, (
                f"cls {cls_id}: w={new_w:.4f} < {NOTEHEAD_MIN_DIM} — too small for TAL"
            )
            assert new_h >= NOTEHEAD_MIN_DIM, (
                f"cls {cls_id}: h={new_h:.4f} < {NOTEHEAD_MIN_DIM} — too small for TAL"
            )
            assert new_w <= NOTEHEAD_MAX_DIM, (
                f"cls {cls_id}: w={new_w:.4f} > {NOTEHEAD_MAX_DIM} — still too large"
            )
            assert new_h <= NOTEHEAD_MAX_DIM, (
                f"cls {cls_id}: h={new_h:.4f} > {NOTEHEAD_MAX_DIM} — still too large"
            )

    def test_shrunk_box_stays_inside_image(self, shrink_fn):
        """Shrunk box must be fully inside [0, 1] even for edge predictions."""
        edge_boxes = [
            (0.02, 0.02, 0.25, 0.20),  # top-left corner
            (0.98, 0.98, 0.25, 0.20),  # bottom-right corner
            (0.50, 0.01, 0.30, 0.15),  # top edge
            (0.01, 0.50, 0.20, 0.25),  # left edge
        ]
        for cls_id in [0, 1, 2, 8, 22]:  # various classes
            for box in edge_boxes:
                cx, cy, w, h = shrink_fn(cls_id, *box)
                assert cx - w / 2 >= -1e-6, f"cls={cls_id} box={box}: left edge < 0 (cx={cx}, w={w})"
                assert cy - h / 2 >= -1e-6, f"cls={cls_id} box={box}: top edge < 0 (cy={cy}, h={h})"
                assert cx + w / 2 <= 1 + 1e-6, f"cls={cls_id} box={box}: right edge > 1 (cx={cx}, w={w})"
                assert cy + h / 2 <= 1 + 1e-6, f"cls={cls_id} box={box}: bottom edge > 1 (cy={cy}, h={h})"

    def test_notehead_top_bias_with_boundary_conflict(self, shrink_fn):
        """When glyph-group is near top of image, top-bias must yield to boundary.

        Reviewer C3: top-bias and boundary constraint must coexist.
        """
        # Prediction near top of image: cy=0.05, top edge at 0.05-0.10=-0.05
        near_top_box = (0.50, 0.05, 0.25, 0.20)

        for cls_id in [0, 1]:
            cx, cy, w, h = shrink_fn(cls_id, *near_top_box)
            # Must stay inside image
            assert cy - h / 2 >= -1e-6, (
                f"cls={cls_id}: top edge {cy - h/2:.4f} < 0 — boundary violated"
            )
            # Should still try to be near top of prediction if possible
            # (but not required if it would violate boundary)

    def test_shrunk_box_center_near_prediction(self, shrink_fn):
        """Shrunk box center should be within the original prediction box."""
        pred_box = (0.50, 0.30, 0.25, 0.20)
        cx_orig, cy_orig, w_orig, h_orig = pred_box

        for cls_id in [0, 1, 2, 7, 22]:
            cx, cy, _, _ = shrink_fn(cls_id, *pred_box)
            assert abs(cx - cx_orig) <= w_orig / 2, (
                f"cls {cls_id}: cx={cx:.4f} too far from pred cx={cx_orig:.4f}"
            )
            assert abs(cy - cy_orig) <= h_orig / 2, (
                f"cls {cls_id}: cy={cy:.4f} too far from pred cy={cy_orig:.4f}"
            )

    def test_notehead_center_biased_toward_top(self, shrink_fn):
        """For noteheads in center of image, center_y should be ≤ prediction center_y.

        Phase 6 validated: notehead sits at TOP edge of glyph-group.
        YOLO: cy=0 is image top, so smaller cy = higher position.
        """
        # Use a box safely away from boundaries to avoid conflict
        pred_box = (0.50, 0.50, 0.25, 0.20)
        _, cy_orig, _, _ = pred_box

        for cls_id in [0, 1]:
            _, cy, _, _ = shrink_fn(cls_id, *pred_box)
            assert cy <= cy_orig, (
                f"Notehead cls {cls_id}: cy={cy:.4f} should be ≤ pred cy={cy_orig:.4f} "
                f"(notehead is at TOP of glyph-group)"
            )

    def test_shrink_adds_size_variation(self, shrink_fn):
        """Multiple calls must produce diverse sizes (jitter).

        Phase 6 lesson: all 63K noteheads identical → unnatural.
        Require ≥15 unique sizes in 50 calls to mandate continuous jitter.
        """
        pred_box = (0.50, 0.30, 0.25, 0.20)
        sizes = set()
        for _ in range(50):
            _, _, w, h = shrink_fn(0, *pred_box)
            sizes.add((round(w, 5), round(h, 5)))

        assert len(sizes) >= 15, (
            f"Only {len(sizes)} unique sizes in 50 calls — too discrete, "
            f"need continuous jitter (Phase 6 had 1)"
        )

    def test_different_classes_get_different_aspect_ratios(self, shrink_fn):
        """Beam (wide+thin) vs notehead (roughly square) must differ meaningfully."""
        pred_box = (0.50, 0.30, 0.25, 0.20)

        median_aspect = {}
        for cls_id in [0, 2]:  # notehead_filled, beam
            ratios = []
            for _ in range(30):
                _, _, w, h = shrink_fn(cls_id, *pred_box)
                ratios.append(w / h if h > 0 else 0)
            median_aspect[cls_id] = statistics.median(ratios)

        nh_ar = median_aspect[0]
        beam_ar = median_aspect[2]
        # Beam should be wider relative to height (DoReMi: 0.036/0.007 ≈ 5:1)
        # Notehead is roughly square (DoReMi: 0.011/0.007 ≈ 1.6:1)
        assert beam_ar > nh_ar * 1.5, (
            f"Beam aspect ratio ({beam_ar:.2f}) should be >1.5× notehead ({nh_ar:.2f})"
        )

    def test_unreliable_classes_still_produce_valid_output(self, shrink_fn):
        """Classes without reliable DoReMi reference must still output valid boxes.

        Reviewer L5: 7 classes have no good reference. shrink_fn must not crash
        or fall back to glyph-group size for these.
        """
        pred_box = (0.50, 0.30, 0.25, 0.20)
        unreliable_ids = [NAMES.index(n) for n in UNRELIABLE_DOREMI_CLASSES if n in NAMES]

        for cls_id in unreliable_ids:
            cx, cy, w, h = shrink_fn(cls_id, *pred_box)
            # Must not return the original glyph-group size
            assert w < 0.20, f"cls {NAMES[cls_id]}: w={w:.4f} — still glyph-group sized"
            assert h < 0.20, f"cls {NAMES[cls_id]}: h={h:.4f} — still glyph-group sized"
            # Must be positive
            assert w > 0 and h > 0, f"cls {NAMES[cls_id]}: zero-sized box"


# ──────────────────────────────────────────────────────────────────────
# Integration Tests: Phase 5 Teacher Coverage
# ──────────────────────────────────────────────────────────────────────

class TestPhase5TeacherCoverage:
    """Validate Phase 5 model can detect symbols on OpenScore images."""

    @pytest.fixture(scope="class")
    def phase5_predictions(self):
        """Run Phase 5 on 10 OpenScore val images, cache results."""
        from ultralytics import YOLO
        import os

        model = YOLO(str(PHASE5_WEIGHTS))
        val_img_dir = PHASE6_DATASET / "val/images"
        os_imgs = sorted([f for f in os.listdir(val_img_dir) if "lg-" in f])[:10]
        assert len(os_imgs) > 0, "No OpenScore val images found"

        predictions = {}
        for fname in os_imgs:
            r = model.predict(
                str(val_img_dir / fname), imgsz=1280, conf=0.25,
                iou=INFERENCE_IOU, max_det=1500, verbose=False,
            )
            boxes = r[0].boxes
            preds = []
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                xywhn = boxes.xywhn[i]
                preds.append((cls, float(xywhn[0]), float(xywhn[1]),
                              float(xywhn[2]), float(xywhn[3])))
            predictions[fname] = preds
        return predictions

    def test_detects_noteheads_on_openscore(self, phase5_predictions):
        """Phase 5 must detect ≥20 noteheads per OpenScore image on average."""
        nh_counts = []
        for preds in phase5_predictions.values():
            nh = sum(1 for cls, *_ in preds if cls in (0, 1))
            nh_counts.append(nh)

        avg_nh = statistics.mean(nh_counts)
        assert avg_nh >= 20, (
            f"Average {avg_nh:.1f} noteheads/image — too few for pseudo-labeling"
        )

    def test_detects_multiple_classes(self, phase5_predictions):
        """Phase 5 must detect ≥12 unique classes across OpenScore images."""
        all_classes = set()
        for preds in phase5_predictions.values():
            for cls, *_ in preds:
                all_classes.add(cls)

        assert len(all_classes) >= 12, (
            f"Only {len(all_classes)} classes detected — need broader coverage"
        )

    def test_prediction_boxes_are_glyph_group_sized(self, phase5_predictions):
        """Phase 5 notehead predictions should be large, confirming need for shrink."""
        nh_widths = []
        for preds in phase5_predictions.values():
            for cls, _, _, w, h in preds:
                if cls in (0, 1):
                    nh_widths.append(w)

        assert len(nh_widths) > 0, "No notehead predictions — Phase 5 model broken?"

        median_w = statistics.median(nh_widths)
        assert median_w > 0.10, (
            f"Median notehead width {median_w:.4f} — expected large glyph-group boxes (>0.10)"
        )


# ──────────────────────────────────────────────────────────────────────
# Integration Tests: Pseudo-Label Dataset Quality
# ──────────────────────────────────────────────────────────────────────

class TestPseudoLabelDataset:
    """Validate the final relabeled dataset after pipeline completes."""

    @pytest.fixture(scope="class")
    def pseudo_labels(self):
        """Load pseudo-label dataset labels."""
        assert PSEUDO_DATASET.exists(), (
            f"Pseudo-label dataset not found at {PSEUDO_DATASET}. "
            f"Run create_pseudo_labels.py first."
        )
        return {
            "train": _read_yolo_labels(PSEUDO_DATASET / "train/labels"),
            "val": _read_yolo_labels(PSEUDO_DATASET / "val/labels"),
        }

    @pytest.fixture(scope="class")
    def original_labels(self):
        """Load Phase 6 dataset labels for comparison."""
        return {
            "train": _read_yolo_labels(PHASE6_DATASET / "train/labels"),
            "val": _read_yolo_labels(PHASE6_DATASET / "val/labels"),
        }

    def test_dataset_exists(self):
        """Pseudo-label dataset directory must exist with expected structure."""
        for split in ["train", "val"]:
            assert (PSEUDO_DATASET / split / "images").is_dir(), f"Missing {split}/images"
            assert (PSEUDO_DATASET / split / "labels").is_dir(), f"Missing {split}/labels"
        assert (PSEUDO_DATASET / "harmony_pseudo_v1.yaml").is_file(), "Missing YAML config"

    def test_image_count_matches_original(self, pseudo_labels, original_labels):
        """Pseudo dataset must have same number of label files as original."""
        for split in ["train", "val"]:
            pseudo_count = len(pseudo_labels[split])
            orig_count = len(original_labels[split])
            assert pseudo_count == orig_count, (
                f"{split}: pseudo has {pseudo_count}, original has {orig_count}"
            )

    def test_doremi_labels_unchanged(self, pseudo_labels, original_labels):
        """DoReMi labels must be IDENTICAL to original — only OpenScore changed."""
        for split in ["train", "val"]:
            doremi_files = [f for f in original_labels[split] if not _is_openscore(f)]
            mismatches = 0
            for fname in doremi_files:  # check ALL DoReMi files
                orig = original_labels[split].get(fname, [])
                pseudo = pseudo_labels[split].get(fname, [])
                if len(orig) != len(pseudo):
                    mismatches += 1
                    continue
                for a, b in zip(sorted(orig), sorted(pseudo)):
                    if a != b:
                        mismatches += 1
                        break

            assert mismatches == 0, (
                f"{split}: {mismatches}/{len(doremi_files)} DoReMi files differ"
            )

    def test_openscore_has_pseudo_labels(self, pseudo_labels):
        """OpenScore files must have pseudo-label annotations (not empty).

        Grounded in Phase 5 detection capacity (~57 noteheads/image at conf=0.25).
        """
        for split in ["train", "val"]:
            os_files = [f for f in pseudo_labels[split] if _is_openscore(f)]
            empty_count = sum(1 for f in os_files if len(pseudo_labels[split][f]) == 0)

            if len(os_files) == 0:
                continue

            nonempty_ratio = 1 - (empty_count / len(os_files))
            assert nonempty_ratio >= 0.90, (
                f"{split}: only {nonempty_ratio:.0%} OpenScore files have annotations — "
                f"Phase 5 should detect symbols on >90% of images"
            )

    def test_class_coverage(self, pseudo_labels):
        """Pseudo-label dataset must cover ≥28/32 classes in training set."""
        classes_seen = set()
        for anns in pseudo_labels["train"].values():
            for cls, *_ in anns:
                classes_seen.add(cls)

        assert len(classes_seen) >= 28, (
            f"Only {len(classes_seen)}/32 classes — "
            f"missing: {[NAMES[c] for c in (set(range(32)) - classes_seen)]}"
        )

    def test_openscore_notehead_not_phase6_tiny(self, pseudo_labels):
        """No pseudo-label notehead should have BOTH w and h below Phase 6 danger zone.

        Phase 6: 0.012 norm → 11px at inference → TAL failure.
        """
        tiny_count = 0
        total_count = 0
        for fname, anns in pseudo_labels["train"].items():
            if not _is_openscore(fname):
                continue
            for cls, cx, cy, w, h in anns:
                if cls not in (0, 1):
                    continue
                total_count += 1
                if w < PHASE6_DANGER_THRESHOLD and h < PHASE6_DANGER_THRESHOLD:
                    tiny_count += 1

        assert total_count > 0, "No OpenScore noteheads in pseudo dataset"

        tiny_ratio = tiny_count / total_count
        assert tiny_ratio < 0.01, (
            f"{tiny_count}/{total_count} ({tiny_ratio:.1%}) notehead boxes in Phase 6 danger zone "
            f"(both dims < {PHASE6_DANGER_THRESHOLD})"
        )

    def test_openscore_notehead_in_target_range(self, pseudo_labels):
        """OpenScore notehead boxes should mostly be in [NOTEHEAD_MIN, NOTEHEAD_MAX]."""
        out_of_range = 0
        total = 0
        for fname, anns in pseudo_labels["train"].items():
            if not _is_openscore(fname):
                continue
            for cls, cx, cy, w, h in anns:
                if cls not in (0, 1):
                    continue
                total += 1
                if w < NOTEHEAD_MIN_DIM or h < NOTEHEAD_MIN_DIM:
                    out_of_range += 1
                elif w > NOTEHEAD_MAX_DIM or h > NOTEHEAD_MAX_DIM:
                    out_of_range += 1

        assert total > 0, "No OpenScore noteheads in pseudo dataset"

        in_range_ratio = 1 - (out_of_range / total)
        assert in_range_ratio >= 0.90, (
            f"Only {in_range_ratio:.0%} noteheads in target range "
            f"[{NOTEHEAD_MIN_DIM}, {NOTEHEAD_MAX_DIM}] — "
            f"{out_of_range}/{total} out of range"
        )

    def test_openscore_notehead_size_diversity(self, pseudo_labels):
        """OpenScore noteheads must NOT all be identical size.

        Phase 6: all 63K = exactly 0.01180×0.00908 → 1 unique size.
        DoReMi has 930 unique sizes. Require ≥50 for pseudo-labels.
        """
        sizes = set()
        for fname, anns in pseudo_labels["train"].items():
            if not _is_openscore(fname):
                continue
            for cls, cx, cy, w, h in anns:
                if cls in (0, 1):
                    sizes.add((round(w, 5), round(h, 5)))

        assert len(sizes) >= 50, (
            f"Only {len(sizes)} unique notehead sizes — need continuous diversity"
        )


# ──────────────────────────────────────────────────────────────────────
# Smoke test: module importability
# ──────────────────────────────────────────────────────────────────────

class TestModuleImport:
    """Verify the pipeline module exists and is importable."""

    def test_import_create_pseudo_labels(self):
        """create_pseudo_labels module must be importable."""
        import importlib
        spec = importlib.util.find_spec("create_pseudo_labels")
        assert spec is not None, (
            "Cannot find create_pseudo_labels module. "
            "Create training/create_pseudo_labels.py first."
        )

    def test_shrink_box_function_exists(self):
        """shrink_box() function must be exported."""
        from create_pseudo_labels import shrink_box
        assert callable(shrink_box)

    def test_generate_pseudo_labels_function_exists(self):
        """generate_pseudo_labels() function must be exported."""
        from create_pseudo_labels import generate_pseudo_labels
        assert callable(generate_pseudo_labels)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
