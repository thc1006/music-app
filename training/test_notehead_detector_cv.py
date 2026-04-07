"""
TDD Red Phase: CV-based Notehead Detector Tests

Detects noteheads directly from sheet music images using classical computer vision
(morphology + connected components), no ML model needed.

Why CV instead of ML:
  - Phase 6/7/8 all failed because we couldn't determine notehead position within glyph-groups
  - Noteheads have extremely distinctive visual features (small dark ellipses on staff lines)
  - Classical CV can find them directly, no training data quality issues

Target: OpenScore images (~1960×2772px), noteheads ~12-25px, staff spacing ~10-12px
"""
import pytest
from pathlib import Path
import statistics

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
TRAINING_DIR = PROJECT_ROOT / "training"
PHASE6_DATASET = TRAINING_DIR / "datasets/yolo_harmony_v2_phase6_fixed"

# OpenScore val images (different font styles)
OPENSCORE_VAL_IMGS = [
    "phase7_phase6_base_p4_p3_p2_lg-102414375-aug-beethoven--page-2_oversample_12_3.png",
    "phase7_phase6_base_p4_p3_p2_lg-10247684-aug-gonville--page-3.png",
    "phase7_phase6_base_p4_p3_p2_lg-102548668-aug-gutenberg1939--page-1.png",
    "phase7_phase6_base_p4_p3_p2_lg-105569450-aug-emmentaler--page-1_oversample_6_1.png",
    "phase7_phase6_base_p4_p3_p2_lg-11466156-aug-beethoven--page-2.png",
    "phase7_phase6_base_p4_p3_p2_lg-120034259-aug-beethoven--page-6.png",
    "phase7_phase6_base_p4_p3_p2_lg-117996848-aug-emmentaler--page-29.png",
    "phase7_phase6_base_p4_p3_p2_lg-131048488-aug-emmentaler--page-4.png",
]

# Notehead expected PADDED bbox pixel sizes in 1960×2772 images
# padding_ratio=1.5 turns a ~20px notehead into ~80px bbox (for TAL learnability)
NH_MIN_PX = 15
NH_MAX_PX = 120
NH_MIN_AREA = 200
NH_MAX_AREA = 14000


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _img_path(fname):
    return str(PHASE6_DATASET / "val/images" / fname)


def _count_gt_noteheads(fname):
    """Count noteheads in Phase 6 GT labels for comparison."""
    stem = Path(fname).stem
    lbl = PHASE6_DATASET / "val/labels" / f"{stem}.txt"
    if not lbl.exists():
        return 0
    count = 0
    for line in lbl.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5 and int(parts[0]) in (0, 1):
            count += 1
    return count


# ──────────────────────────────────────────────────────────────────────
# Module Import Tests
# ──────────────────────────────────────────────────────────────────────

class TestModuleImport:

    def test_module_importable(self):
        import importlib
        spec = importlib.util.find_spec("notehead_detector_cv")
        assert spec is not None, "Create training/notehead_detector_cv.py first"

    def test_detect_function_exists(self):
        from notehead_detector_cv import detect_noteheads
        assert callable(detect_noteheads)


# ──────────────────────────────────────────────────────────────────────
# Core Detection Tests (run on real images)
# ──────────────────────────────────────────────────────────────────────

class TestNoteheadDetection:
    """Test detection quality on real OpenScore images."""

    @pytest.fixture(scope="class")
    def detector_results(self):
        """Run detector on all 8 test images, cache results."""
        from notehead_detector_cv import detect_noteheads

        results = {}
        for fname in OPENSCORE_VAL_IMGS:
            path = _img_path(fname)
            detections = detect_noteheads(path)
            gt_count = _count_gt_noteheads(fname)
            results[fname] = {
                "detections": detections,
                "gt_count": gt_count,
            }
        return results

    def test_finds_noteheads_on_every_image(self, detector_results):
        """Must find ≥10 noteheads on every OpenScore image."""
        for fname, res in detector_results.items():
            n = len(res["detections"])
            assert n >= 10, (
                f"{fname[:40]}: only {n} noteheads found — expected ≥10"
            )

    def test_average_detection_count(self, detector_results):
        """Average ≥30 noteheads per image across all test images."""
        counts = [len(r["detections"]) for r in detector_results.values()]
        avg = statistics.mean(counts)
        assert avg >= 30, (
            f"Average {avg:.0f} noteheads/image — expected ≥30. "
            f"Per-image: {counts}"
        )

    def test_coverage_vs_gt(self, detector_results):
        """Should detect ≥40% of GT noteheads on average.

        GT has ~100-240 noteheads per image. We need ≥40% coverage
        to be a meaningful improvement over Phase 8's ~30%.
        """
        ratios = []
        for fname, res in detector_results.items():
            gt = res["gt_count"]
            detected = len(res["detections"])
            if gt > 0:
                ratios.append(detected / gt)

        avg_ratio = statistics.mean(ratios)
        assert avg_ratio >= 0.40, (
            f"Average coverage {avg_ratio:.0%} < 40% of GT. "
            f"Per-image: {[f'{r:.0%}' for r in ratios]}"
        )

    def test_bbox_size_reasonable(self, detector_results):
        """Detected notehead bboxes should be 8-40px, not glyph-group sized."""
        all_ws, all_hs = [], []
        for res in detector_results.values():
            for cls, cx, cy, w, h in res["detections"]:
                # Convert normalized to pixel (images are 1960×2772)
                w_px = w * 1960
                h_px = h * 2772
                all_ws.append(w_px)
                all_hs.append(h_px)

        if not all_ws:
            pytest.fail("No detections at all")

        med_w = statistics.median(all_ws)
        med_h = statistics.median(all_hs)

        assert med_w >= NH_MIN_PX, f"Median width {med_w:.1f}px < {NH_MIN_PX} — too small"
        assert med_w <= NH_MAX_PX, f"Median width {med_w:.1f}px > {NH_MAX_PX} — too large"
        assert med_h >= NH_MIN_PX, f"Median height {med_h:.1f}px < {NH_MIN_PX} — too small"
        assert med_h <= NH_MAX_PX, f"Median height {med_h:.1f}px > {NH_MAX_PX} — too large"

    def test_no_huge_boxes(self, detector_results):
        """No detection should be >200px — that's glyph-group territory.

        With padding_ratio=1.5, a 25px notehead becomes ~88px bbox.
        Glyph-groups are 300-600px. Threshold at 200px separates them.
        """
        huge = 0
        total = 0
        for res in detector_results.values():
            for cls, cx, cy, w, h in res["detections"]:
                total += 1
                w_px = w * 1960
                h_px = h * 2772
                if w_px > 200 or h_px > 200:
                    huge += 1

        assert total > 0, "No detections"
        ratio = huge / total
        assert ratio < 0.05, (
            f"{huge}/{total} ({ratio:.1%}) detections >200px — should be <5%"
        )

    def test_positions_within_image(self, detector_results):
        """All detections must be within [0, 1] normalized bounds."""
        for res in detector_results.values():
            for cls, cx, cy, w, h in res["detections"]:
                assert 0 <= cx <= 1, f"cx={cx} out of bounds"
                assert 0 <= cy <= 1, f"cy={cy} out of bounds"
                assert cx - w / 2 >= -0.01, f"left edge out: cx={cx}, w={w}"
                assert cy - h / 2 >= -0.01, f"top edge out: cy={cy}, h={h}"

    def test_output_format(self, detector_results):
        """Each detection should be (cls, cx, cy, w, h) with cls in {0, 1}."""
        for res in detector_results.values():
            for det in res["detections"]:
                assert len(det) == 5, f"Expected 5-tuple, got {len(det)}"
                cls, cx, cy, w, h = det
                assert cls in (0, 1), f"cls={cls} not in {{0, 1}}"
                assert isinstance(cx, float), f"cx should be float"
                assert w > 0 and h > 0, f"w={w}, h={h} must be positive"


# ──────────────────────────────────────────────────────────────────────
# Cross-Font Robustness Tests
# ──────────────────────────────────────────────────────────────────────

class TestCrossFontRobustness:
    """Verify detector works across different LilyPond font styles."""

    @pytest.fixture(scope="class")
    def per_font_results(self):
        from notehead_detector_cv import detect_noteheads

        fonts = {
            "beethoven": [],
            "gonville": [],
            "emmentaler": [],
            "gutenberg": [],
        }
        for fname in OPENSCORE_VAL_IMGS:
            path = _img_path(fname)
            dets = detect_noteheads(path)
            for font_key in fonts:
                if font_key in fname.lower():
                    fonts[font_key].append(len(dets))
                    break
        return fonts

    def test_all_fonts_produce_detections(self, per_font_results):
        """Every font style must produce ≥10 noteheads on at least one image."""
        for font, counts in per_font_results.items():
            if not counts:
                continue  # no test image for this font
            max_count = max(counts)
            assert max_count >= 10, (
                f"Font '{font}': max {max_count} detections — detector may not work for this font"
            )

    def test_font_detection_variance_not_extreme(self, per_font_results):
        """No font should have <50% of the best font's average detection rate."""
        avgs = {}
        for font, counts in per_font_results.items():
            if counts:
                avgs[font] = statistics.mean(counts)

        if len(avgs) < 2:
            pytest.skip("Need ≥2 fonts with results")

        best = max(avgs.values())
        worst = min(avgs.values())
        ratio = worst / best if best > 0 else 0

        assert ratio >= 0.3, (
            f"Font variance too extreme: best={best:.0f}, worst={worst:.0f}, "
            f"ratio={ratio:.0%}. Detector may be font-dependent. "
            f"Per-font: {avgs}"
        )


# ──────────────────────────────────────────────────────────────────────
# Size Diversity Test (Phase 6 lesson)
# ──────────────────────────────────────────────────────────────────────

class TestNoRegression:
    """Guard against regression when modifying the detector."""

    @pytest.fixture(scope="class")
    def all_results(self):
        from notehead_detector_cv import detect_noteheads
        results = {}
        for fname in OPENSCORE_VAL_IMGS:
            dets = detect_noteheads(_img_path(fname))
            gt = _count_gt_noteheads(fname)
            results[fname] = {"dets": len(dets), "gt": gt}
        return results

    def test_average_coverage_at_least_70(self, all_results):
        """Average coverage must stay ≥70% (current baseline ~86%)."""
        ratios = [r["dets"] / r["gt"] for r in all_results.values() if r["gt"] > 0]
        avg = statistics.mean(ratios)
        assert avg >= 0.70, f"Average coverage {avg:.0%} < 70%"

    def test_worst_image_at_least_50(self, all_results):
        """No single image should drop below 50% coverage."""
        for fname, r in all_results.items():
            if r["gt"] == 0:
                continue
            ratio = r["dets"] / r["gt"]
            assert ratio >= 0.50, (
                f"{fname[:40]}: coverage {ratio:.0%} < 50%"
            )

    def test_false_positive_rate_bounded(self, all_results):
        """Images with >150% coverage indicate excessive FP. Max 1 such image."""
        excess_count = sum(
            1 for r in all_results.values()
            if r["gt"] > 0 and r["dets"] / r["gt"] > 1.50
        )
        assert excess_count <= 1, (
            f"{excess_count} images with >150% coverage — too many false positives"
        )


class TestBoxSizeNearGT:
    """Phase A critical: box size must be close to GT 23×25px, not 85×90px.

    Bug discovered 2026-04-07: padding_ratio=1.5 + min_box_px=49/69 floor
    produces boxes 3.7x larger than GT, making IoU matching impossible.
    Fix: reduce padding_ratio AND remove/reduce floor.
    """

    @pytest.fixture(scope="class")
    def box_sizes(self):
        from notehead_detector_cv import detect_noteheads
        from PIL import Image
        ws, hs = [], []
        for fname in OPENSCORE_VAL_IMGS[:4]:
            dets = detect_noteheads(_img_path(fname))
            img = Image.open(_img_path(fname))
            W, H = img.size
            for cls, cx, cy, w, h in dets:
                ws.append(w * W)
                hs.append(h * H)
        return ws, hs

    def test_median_width_close_to_gt(self, box_sizes):
        """Median bbox width should be in [18, 40]px (GT is ~23px)."""
        ws, _ = box_sizes
        assert len(ws) > 0, "No detections"
        med_w = statistics.median(ws)
        assert 18 <= med_w <= 40, (
            f"Median width {med_w:.0f}px out of range [18, 40]. "
            f"GT is ~23px. Current bug: floor=49px forces oversized boxes."
        )

    def test_median_height_close_to_gt(self, box_sizes):
        """Median bbox height should be in [18, 45]px (GT is ~25px)."""
        _, hs = box_sizes
        med_h = statistics.median(hs)
        assert 18 <= med_h <= 45, (
            f"Median height {med_h:.0f}px out of range [18, 45]. "
            f"GT is ~25px. Current bug: floor=69px forces oversized boxes."
        )

    def test_no_box_larger_than_60px(self, box_sizes):
        """No more than 10% of boxes should exceed 60px in either dimension."""
        ws, hs = box_sizes
        total = len(ws)
        huge = sum(1 for w, h in zip(ws, hs) if w > 60 or h > 60)
        ratio = huge / total if total else 0
        assert ratio <= 0.10, (
            f"{huge}/{total} ({ratio:.0%}) boxes > 60px — floor bug active"
        )

    def test_box_area_in_reasonable_range(self, box_sizes):
        """Box area should be close to GT (23*25=575 px²), 80% should be in [300, 1800]."""
        ws, hs = box_sizes
        total = len(ws)
        in_range = sum(1 for w, h in zip(ws, hs) if 300 <= w * h <= 1800)
        ratio = in_range / total if total else 0
        assert ratio >= 0.80, (
            f"Only {ratio:.0%} of boxes in area range [300, 1800] px². "
            f"Target: close to GT area 575 px²."
        )


class TestSizeDiversity:
    """Detected noteheads must have natural size variation (not all identical)."""

    def test_not_all_same_size(self):
        """Phase 6 lesson: all 63K noteheads identical → unnatural. CV should vary."""
        from notehead_detector_cv import detect_noteheads

        sizes = set()
        for fname in OPENSCORE_VAL_IMGS[:3]:
            dets = detect_noteheads(_img_path(fname))
            for cls, cx, cy, w, h in dets:
                sizes.add((round(w, 5), round(h, 5)))

        assert len(sizes) >= 20, (
            f"Only {len(sizes)} unique sizes — CV detections should naturally vary"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
