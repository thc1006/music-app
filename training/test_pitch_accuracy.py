"""
TDD Red: Pitch accuracy harness tests.

This is the foundation for Phase C — measuring how accurate our pitch
estimator is on real images, so we know what to fix.

Strategy:
  - Build a small `ground_truth` dict mapping (image, notehead_index) → expected MIDI
  - Run Phase 9 + downstream pipeline to get predicted pitches
  - Compare and report accuracy
  - This score is the metric we'll improve in C2-C5

Ground truth is established by visual inspection of small cropped images
(too small to hit dimension limits in image-loading tools).
"""
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
sys.path.insert(0, str(PROJECT_ROOT))


class TestModuleImport:
    def test_module_importable(self):
        import importlib.util
        spec = importlib.util.find_spec("pitch_accuracy_harness")
        assert spec is not None, "Create training/pitch_accuracy_harness.py first"

    def test_measure_function_exists(self):
        from pitch_accuracy_harness import measure_pitch_accuracy
        assert callable(measure_pitch_accuracy)

    def test_save_crops_function_exists(self):
        """Helper to save small crops around predicted noteheads for visual inspection."""
        from pitch_accuracy_harness import save_notehead_crops
        assert callable(save_notehead_crops)

    def test_load_ground_truth_function_exists(self):
        from pitch_accuracy_harness import load_ground_truth
        assert callable(load_ground_truth)


class TestGroundTruthFormat:

    def test_ground_truth_is_loadable(self):
        """A YAML/JSON file should map image → list of (notehead_index, expected_midi)."""
        from pitch_accuracy_harness import load_ground_truth
        gt = load_ground_truth()
        assert isinstance(gt, dict)
        # Even an empty dict is valid initially

    def test_ground_truth_value_format(self):
        """Each entry should be {image_basename: list[{idx, midi, ...}]}."""
        from pitch_accuracy_harness import load_ground_truth
        gt = load_ground_truth()
        for image_name, entries in gt.items():
            assert isinstance(image_name, str)
            assert isinstance(entries, list)
            for e in entries:
                assert "idx" in e or "cx" in e
                assert "midi" in e
                assert isinstance(e["midi"], int)
                assert 21 <= e["midi"] <= 108


class TestAccuracyMeasurement:

    def test_perfect_match_returns_100_percent(self):
        """If predictions exactly match ground truth, accuracy = 100%."""
        from pitch_accuracy_harness import measure_pitch_accuracy
        # Synthetic case: 3 noteheads with known midi
        predicted = [
            {"cx": 100, "cy": 100, "midi": 60},
            {"cx": 200, "cy": 100, "midi": 62},
            {"cx": 300, "cy": 100, "midi": 64},
        ]
        ground_truth = [
            {"cx": 100, "cy": 100, "midi": 60},
            {"cx": 200, "cy": 100, "midi": 62},
            {"cx": 300, "cy": 100, "midi": 64},
        ]
        result = measure_pitch_accuracy(predicted, ground_truth)
        assert result["exact_accuracy"] == 1.0
        assert result["matched_count"] == 3
        assert result["total_gt"] == 3

    def test_no_match_returns_zero(self):
        from pitch_accuracy_harness import measure_pitch_accuracy
        predicted = [{"cx": 100, "cy": 100, "midi": 60}]
        ground_truth = [{"cx": 100, "cy": 100, "midi": 72}]
        result = measure_pitch_accuracy(predicted, ground_truth)
        assert result["exact_accuracy"] == 0.0
        assert result["matched_count"] == 0

    def test_within_semitone_tolerance(self):
        """A semitone-off prediction should count for 'lenient' accuracy."""
        from pitch_accuracy_harness import measure_pitch_accuracy
        predicted = [{"cx": 100, "cy": 100, "midi": 61}]
        ground_truth = [{"cx": 100, "cy": 100, "midi": 60}]
        result = measure_pitch_accuracy(
            predicted, ground_truth, tolerance_semitones=1
        )
        assert result["exact_accuracy"] == 0.0
        assert result["lenient_accuracy"] == 1.0  # within ±1 semitone

    def test_position_matching_uses_xy_distance(self):
        """Predictions are matched to GT by spatial proximity."""
        from pitch_accuracy_harness import measure_pitch_accuracy
        # Predictions and GT are in the same order but not at exact same coords
        predicted = [
            {"cx": 105, "cy": 102, "midi": 60},  # ~5px off
            {"cx": 200, "cy": 100, "midi": 62},
        ]
        ground_truth = [
            {"cx": 100, "cy": 100, "midi": 60},
            {"cx": 200, "cy": 100, "midi": 62},
        ]
        result = measure_pitch_accuracy(predicted, ground_truth, max_match_dist=20)
        assert result["exact_accuracy"] == 1.0


class TestBaselineMeasurement:
    """Run the harness on a real image with sample ground truth."""

    def test_baseline_on_real_image_runs_without_crash(self):
        """Even with empty ground truth, the function should not crash."""
        from pitch_accuracy_harness import run_baseline
        VAL_DIR = PROJECT_ROOT / "training/datasets/yolo_harmony_v2_phase6_fixed/val/images"
        test_img = str(VAL_DIR / "phase7_phase6_base_p4_p3_p2_lg-102414375-aug-beethoven--page-2_oversample_12_3.png")
        result = run_baseline(test_img)
        assert "predictions" in result
        assert "ground_truth_count" in result
        assert isinstance(result["predictions"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
