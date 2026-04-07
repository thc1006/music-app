"""
TDD Red Phase: Staff line detector tests

Staff line detection is the foundation for pitch estimation.
We need to find the 5 horizontal lines of each staff in a sheet music image.
"""
import pytest
from pathlib import Path

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
VAL_DIR = PROJECT_ROOT / "training/datasets/yolo_harmony_v2_phase6_fixed/val/images"

# Test images with known staff structure
SINGLE_STAFF_IMG = str(VAL_DIR / "phase7_phase6_base_p4_p3_p2_lg-102548668-aug-gutenberg1939--page-1.png")
MULTI_STAFF_IMG = str(VAL_DIR / "phase7_phase6_base_p4_p3_p2_lg-102414375-aug-beethoven--page-2_oversample_12_3.png")


class TestModuleImport:
    def test_module_importable(self):
        import importlib.util
        spec = importlib.util.find_spec("staff_detector")
        assert spec is not None, "Create training/staff_detector.py first"

    def test_detect_staves_function_exists(self):
        from staff_detector import detect_staves
        assert callable(detect_staves)

    def test_staff_dataclass_exists(self):
        from staff_detector import Staff
        assert Staff is not None


class TestStaffDetection:
    """Test basic staff line detection on real images."""

    def test_returns_list_of_staffs(self):
        from staff_detector import detect_staves
        staves = detect_staves(SINGLE_STAFF_IMG)
        assert isinstance(staves, list)

    def test_detects_at_least_one_staff(self):
        from staff_detector import detect_staves
        staves = detect_staves(SINGLE_STAFF_IMG)
        assert len(staves) >= 1, "Should detect at least 1 staff"

    def test_each_staff_has_5_lines(self):
        """Each Staff object should contain exactly 5 line y-coordinates."""
        from staff_detector import detect_staves
        staves = detect_staves(SINGLE_STAFF_IMG)
        for staff in staves:
            assert hasattr(staff, "line_ys"), "Staff must have line_ys attribute"
            assert len(staff.line_ys) == 5, (
                f"Staff should have 5 lines, got {len(staff.line_ys)}"
            )

    def test_lines_are_sorted_top_to_bottom(self):
        """Line y-coords should be in ascending order (top=smaller y)."""
        from staff_detector import detect_staves
        staves = detect_staves(SINGLE_STAFF_IMG)
        for staff in staves:
            ys = staff.line_ys
            assert ys == sorted(ys), f"Lines not sorted: {ys}"

    def test_staff_spacing_consistent(self):
        """Within a single staff, line spacing should be roughly uniform (±20%)."""
        from staff_detector import detect_staves
        staves = detect_staves(SINGLE_STAFF_IMG)
        for staff in staves:
            ys = staff.line_ys
            spacings = [ys[i+1] - ys[i] for i in range(4)]
            mean_spacing = sum(spacings) / 4
            for s in spacings:
                assert 0.7 * mean_spacing <= s <= 1.3 * mean_spacing, (
                    f"Inconsistent spacing: {spacings} (mean={mean_spacing:.1f})"
                )

    def test_staff_has_spacing_attribute(self):
        """Staff should expose an average spacing for pitch calculation."""
        from staff_detector import detect_staves
        staves = detect_staves(SINGLE_STAFF_IMG)
        for staff in staves:
            assert hasattr(staff, "spacing"), "Staff must have spacing attribute"
            assert staff.spacing > 0

    def test_staff_has_bounds(self):
        """Staff should know its x-range (for associating noteheads)."""
        from staff_detector import detect_staves
        staves = detect_staves(SINGLE_STAFF_IMG)
        for staff in staves:
            assert hasattr(staff, "x_min"), "Staff must have x_min"
            assert hasattr(staff, "x_max"), "Staff must have x_max"
            assert staff.x_min < staff.x_max


class TestMultiStaffDetection:
    """Multi-staff orchestral images."""

    def test_detects_multiple_staves(self):
        """Orchestral score should detect multiple staves (>3)."""
        from staff_detector import detect_staves
        staves = detect_staves(MULTI_STAFF_IMG)
        assert len(staves) >= 3, f"Expected ≥3 staves, got {len(staves)}"

    def test_staves_do_not_overlap(self):
        """Detected staves should have non-overlapping y-ranges."""
        from staff_detector import detect_staves
        staves = detect_staves(MULTI_STAFF_IMG)
        # Sort by top y
        staves_sorted = sorted(staves, key=lambda s: s.line_ys[0])
        for i in range(len(staves_sorted) - 1):
            top = staves_sorted[i]
            bottom = staves_sorted[i + 1]
            assert top.line_ys[-1] < bottom.line_ys[0], (
                f"Staves overlap: {top.line_ys} and {bottom.line_ys}"
            )


class TestStaffUtilities:
    """Test convenience methods on Staff object."""

    def test_staff_y_to_step_conversion(self):
        """Staff should convert a y-coord to a diatonic step from top line."""
        from staff_detector import detect_staves
        staves = detect_staves(SINGLE_STAFF_IMG)
        if not staves:
            pytest.skip("No staves detected")
        staff = staves[0]
        # Top line = step 0
        step_at_top = staff.y_to_step(staff.line_ys[0])
        assert step_at_top == 0, f"Top line step should be 0, got {step_at_top}"
        # Bottom line = step 8 (4 lines down × 2 steps per line)
        step_at_bottom = staff.y_to_step(staff.line_ys[-1])
        assert step_at_bottom == 8, f"Bottom line step should be 8, got {step_at_bottom}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
