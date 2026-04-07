"""
TDD Red: Per-stave clef detector tests.

Phase C C2 — improve clef detection accuracy by detecting clef PER STAVE
(not just assuming treble for all). Currently downstream_eval defaults to
treble when no clef is found, which is wrong for orchestral scores.

Test strategy:
- Synthetic detections (known clef positions) → verify correct staff assignment
- Real images → verify clefs at expected x position (left of staff)
"""
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
sys.path.insert(0, str(PROJECT_ROOT))


def _make_staff(top_y, spacing=16, x_min=0, x_max=1800):
    from staff_detector import Staff
    return Staff(
        line_ys=[top_y + i * spacing for i in range(5)],
        spacing=spacing,
        x_min=x_min,
        x_max=x_max,
    )


def _make_clef(cls_id, cx, cy, w=100, h=200):
    """Build a clef detection dict."""
    return {
        "class_id": cls_id,
        "class_name": {8: "clef_treble", 9: "clef_bass", 10: "clef_alto", 11: "clef_tenor"}[cls_id],
        "cx": float(cx),
        "cy": float(cy),
        "w": float(w),
        "h": float(h),
        "confidence": 0.9,
    }


class TestModuleImport:
    def test_module_importable(self):
        import importlib.util
        spec = importlib.util.find_spec("clef_detector")
        assert spec is not None, "Create training/clef_detector.py first"

    def test_assign_clefs_function_exists(self):
        from clef_detector import assign_clefs_to_staves
        assert callable(assign_clefs_to_staves)


class TestSingleStaffSingleClef:
    """One staff with one clef nearby — basic case."""

    def test_treble_clef_assigned_to_staff(self):
        from clef_detector import assign_clefs_to_staves
        staff = _make_staff(top_y=100)  # staff at y 100-164
        # Clef at left, vertically centered on staff
        clef = _make_clef(cls_id=8, cx=50, cy=132, h=120)
        result = assign_clefs_to_staves([clef], [staff])
        assert result[0] == "treble"

    def test_bass_clef_assigned_to_staff(self):
        from clef_detector import assign_clefs_to_staves
        staff = _make_staff(top_y=100)
        clef = _make_clef(cls_id=9, cx=50, cy=132, h=80)
        result = assign_clefs_to_staves([clef], [staff])
        assert result[0] == "bass"

    def test_alto_clef_assigned_to_staff(self):
        from clef_detector import assign_clefs_to_staves
        staff = _make_staff(top_y=100)
        clef = _make_clef(cls_id=10, cx=50, cy=132, h=100)
        result = assign_clefs_to_staves([clef], [staff])
        assert result[0] == "alto"


class TestNoClef:
    """When no clef is detected for a staff, fallback should be reasonable."""

    def test_no_clefs_returns_default(self):
        from clef_detector import assign_clefs_to_staves
        staff = _make_staff(top_y=100)
        result = assign_clefs_to_staves([], [staff])
        # Should return some clef (treble is the most common default)
        assert result[0] in ("treble", "bass", "alto", "tenor")

    def test_clef_far_away_uses_default(self):
        """Clef detection that's 10× spacing away from any staff is rejected."""
        from clef_detector import assign_clefs_to_staves
        staff = _make_staff(top_y=100, spacing=16)
        # Clef 500 pixels away (way too far)
        clef = _make_clef(cls_id=9, cx=50, cy=900, h=100)
        result = assign_clefs_to_staves([clef], [staff])
        # No reasonable association → default treble
        assert result[0] == "treble"


class TestClefMustBeAtLeft:
    """Clefs sit at the LEFT of the staff. Detections far right should be rejected."""

    def test_clef_at_right_of_staff_rejected(self):
        from clef_detector import assign_clefs_to_staves
        staff = _make_staff(top_y=100, x_min=0, x_max=1800)
        # "clef" at the right side — must be a false positive
        clef = _make_clef(cls_id=9, cx=1700, cy=132)
        result = assign_clefs_to_staves([clef], [staff])
        # Should default to treble, NOT use the misplaced clef
        assert result[0] == "treble"


class TestMultiStaffMultiClef:
    """Multiple staves each with their own clef."""

    def test_two_staves_two_different_clefs(self):
        """Top staff treble, bottom staff bass — each gets correct clef."""
        from clef_detector import assign_clefs_to_staves
        top_staff = _make_staff(top_y=100, spacing=16)  # y 100-164
        bot_staff = _make_staff(top_y=300, spacing=16)  # y 300-364
        treble = _make_clef(cls_id=8, cx=50, cy=132)  # near top staff
        bass = _make_clef(cls_id=9, cx=50, cy=332)  # near bottom staff
        result = assign_clefs_to_staves([treble, bass], [top_staff, bot_staff])
        assert result[0] == "treble"
        assert result[1] == "bass"

    def test_three_staves_three_clefs(self):
        from clef_detector import assign_clefs_to_staves
        s1 = _make_staff(top_y=100, spacing=16)
        s2 = _make_staff(top_y=300, spacing=16)
        s3 = _make_staff(top_y=500, spacing=16)
        c1 = _make_clef(cls_id=8, cx=50, cy=132)   # treble
        c2 = _make_clef(cls_id=10, cx=50, cy=332)  # alto
        c3 = _make_clef(cls_id=9, cx=50, cy=532)   # bass
        result = assign_clefs_to_staves([c1, c2, c3], [s1, s2, s3])
        assert result[0] == "treble"
        assert result[1] == "alto"
        assert result[2] == "bass"


class TestPartialDetection:
    """Common real-world case: some staves detected, others missed."""

    def test_two_staves_only_top_clef_detected(self):
        """When clef detection misses one staff, the other should still get its clef."""
        from clef_detector import assign_clefs_to_staves
        top_staff = _make_staff(top_y=100, spacing=16)
        bot_staff = _make_staff(top_y=300, spacing=16)
        treble = _make_clef(cls_id=8, cx=50, cy=132)
        # No clef for bottom staff
        result = assign_clefs_to_staves([treble], [top_staff, bot_staff])
        assert result[0] == "treble"
        # Bottom should default (treble or based on layout heuristics)
        assert result[1] in ("treble", "bass", "alto", "tenor")


class TestClefUniqueness:
    """C2-Fix: each clef detection must be assigned to AT MOST one staff.

    Without this constraint, a clef bbox that overlaps two adjacent staves
    (after padding) would be matched to both, producing duplicate-clef
    artifacts in orchestral scores.
    """

    def test_one_clef_one_staff_no_double_use(self):
        """One clef + two adjacent staves: only the closest staff gets it."""
        from clef_detector import assign_clefs_to_staves
        s1 = _make_staff(top_y=100, spacing=16)  # y 100-164
        s2 = _make_staff(top_y=200, spacing=16)  # y 200-264
        # Clef bbox that with padding overlaps BOTH staves
        # cy=180, h=200 → top=80, bot=280 — straddles both
        clef = _make_clef(cls_id=9, cx=50, cy=180, h=200)
        result = assign_clefs_to_staves([clef], [s1, s2])
        # Exactly one staff should get bass; the other defaults
        bass_count = sum(1 for c in result if c == "bass")
        assert bass_count == 1, (
            f"One clef should yield one bass assignment, got {bass_count}: {result}"
        )

    def test_two_clefs_two_staves_unique_assignment(self):
        """Two clefs and two staves: each clef assigned to one distinct staff."""
        from clef_detector import assign_clefs_to_staves
        s1 = _make_staff(top_y=100, spacing=16)
        s2 = _make_staff(top_y=300, spacing=16)
        # Both clefs are valid for both staves (with very loose padding)
        c1 = _make_clef(cls_id=8, cx=50, cy=132, h=80)  # closer to s1
        c2 = _make_clef(cls_id=9, cx=50, cy=332, h=80)  # closer to s2
        result = assign_clefs_to_staves([c1, c2], [s1, s2])
        assert result[0] == "treble"
        assert result[1] == "bass"
        # Specifically, both clefs got used (no duplicates)
        assert "treble" in result
        assert "bass" in result

    def test_three_staves_two_clefs_partial_assignment(self):
        """Three staves but only two clef detections — third defaults."""
        from clef_detector import assign_clefs_to_staves
        s1 = _make_staff(top_y=100, spacing=16)
        s2 = _make_staff(top_y=300, spacing=16)
        s3 = _make_staff(top_y=500, spacing=16)
        c1 = _make_clef(cls_id=8, cx=50, cy=132, h=80)
        c2 = _make_clef(cls_id=9, cx=50, cy=532, h=80)
        result = assign_clefs_to_staves([c1, c2], [s1, s2, s3])
        assert result[0] == "treble"  # got c1
        assert result[2] == "bass"    # got c2
        assert result[1] == "treble"  # default (no clef for middle staff)
        # Each detection assigned to at most one staff
        # (no way for c1 or c2 to be in result twice)
        assert result.count("treble") + result.count("bass") <= 3

    def test_more_clefs_than_staves(self):
        """4 clefs, 2 staves: only 2 clefs used, 2 ignored."""
        from clef_detector import assign_clefs_to_staves
        s1 = _make_staff(top_y=100, spacing=16)
        s2 = _make_staff(top_y=300, spacing=16)
        # 4 clefs, all valid for some staff
        c1 = _make_clef(cls_id=8, cx=50, cy=132)
        c2 = _make_clef(cls_id=9, cx=50, cy=140)  # also near s1
        c3 = _make_clef(cls_id=10, cx=50, cy=332)
        c4 = _make_clef(cls_id=11, cx=50, cy=340)  # also near s2
        result = assign_clefs_to_staves([c1, c2, c3, c4], [s1, s2])
        assert len(result) == 2
        # Each staff got the closest clef; extras ignored
        assert result[0] in ("treble", "bass")  # closest of c1 or c2
        assert result[1] in ("alto", "tenor")    # closest of c3 or c4


class TestRealImageUniqueness:
    """Real-world: Beethoven test image has 7 clefs / 12 staves.
    Verify no clef detection is double-assigned."""

    def test_real_image_no_clef_assigned_to_multiple_staves(self):
        """On the Beethoven test image, ensure each Phase 9 clef detection
        is assigned to at most one staff (no double-counting)."""
        from clef_detector import assign_clefs_to_staves
        from downstream_eval import run_phase9_detection
        from staff_detector import detect_staves

        VAL_DIR = PROJECT_ROOT / "training/datasets/yolo_harmony_v2_phase6_fixed/val/images"
        IMG = str(VAL_DIR / "phase7_phase6_base_p4_p3_p2_lg-102414375-aug-beethoven--page-2_oversample_12_3.png")

        staves = detect_staves(IMG)
        detections = run_phase9_detection(IMG)
        clef_clss = {8, 9, 10, 11}
        clefs = [d for d in detections if d.get("class_id") in clef_clss]

        result = assign_clefs_to_staves(clefs, staves)
        # Sanity: at most as many non-default assignments as clef detections
        non_treble_count = sum(1 for c in result if c != "treble")
        # Phase 9 detected 7 clefs (2 treble, 3 bass, 2 alto)
        # After 1-to-1 matching, non-treble assignments should be ≤ 5
        # (3 bass + 2 alto = 5 max possible non-treble)
        assert non_treble_count <= 5, (
            f"Got {non_treble_count} non-treble assignments but Phase 9 only "
            f"detected 5 non-treble clefs. Some clef must be double-assigned."
        )


class TestClefDistanceFiltering:
    """Clefs should only associate with staves whose y-range overlaps clef y-range."""

    def test_clef_within_staff_y_range_assigned(self):
        from clef_detector import assign_clefs_to_staves
        staff = _make_staff(top_y=100, spacing=16)  # y 100-164
        # Clef y=132 is within staff y range
        clef = _make_clef(cls_id=8, cx=50, cy=132, h=140)
        result = assign_clefs_to_staves([clef], [staff])
        assert result[0] == "treble"

    def test_clef_close_to_staff_centerline(self):
        from clef_detector import assign_clefs_to_staves
        staff = _make_staff(top_y=100, spacing=16)
        # Centerline at y=132
        # Clef cy slightly off but still within reasonable distance
        clef = _make_clef(cls_id=9, cx=50, cy=140, h=80)
        result = assign_clefs_to_staves([clef], [staff])
        assert result[0] == "bass"


class TestRealImageClefAssignment:
    """Use real Phase 9 detections + real staff_detector output."""

    def test_real_image_returns_one_clef_per_staff(self):
        from clef_detector import assign_clefs_to_staves
        from downstream_eval import run_phase9_detection
        from staff_detector import detect_staves

        VAL_DIR = PROJECT_ROOT / "training/datasets/yolo_harmony_v2_phase6_fixed/val/images"
        IMG = str(VAL_DIR / "phase7_phase6_base_p4_p3_p2_lg-102414375-aug-beethoven--page-2_oversample_12_3.png")

        staves = detect_staves(IMG)
        detections = run_phase9_detection(IMG)
        clef_clss = {8, 9, 10, 11}
        clefs = [d for d in detections if d.get("class_id") in clef_clss]

        result = assign_clefs_to_staves(clefs, staves)
        # One clef name per staff
        assert len(result) == len(staves)
        # All values are valid clef names
        for clef_name in result:
            assert clef_name in ("treble", "bass", "alto", "tenor")

    def test_real_image_at_least_some_non_treble(self):
        """A real orchestral image should have at least 1 non-treble assignment
        if Phase 9 detected any non-treble clef. (Beethoven test image has bass+alto.)"""
        from clef_detector import assign_clefs_to_staves
        from downstream_eval import run_phase9_detection
        from staff_detector import detect_staves

        VAL_DIR = PROJECT_ROOT / "training/datasets/yolo_harmony_v2_phase6_fixed/val/images"
        IMG = str(VAL_DIR / "phase7_phase6_base_p4_p3_p2_lg-102414375-aug-beethoven--page-2_oversample_12_3.png")

        staves = detect_staves(IMG)
        detections = run_phase9_detection(IMG)
        clef_clss = {8, 9, 10, 11}
        clefs = [d for d in detections if d.get("class_id") in clef_clss]

        result = assign_clefs_to_staves(clefs, staves)
        non_treble = [c for c in result if c != "treble"]
        # Phase 9 detected bass + alto on this image, so we should have ≥1 non-treble
        assert len(non_treble) >= 1, (
            f"Expected ≥1 non-treble clef, got all-treble: {result}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
