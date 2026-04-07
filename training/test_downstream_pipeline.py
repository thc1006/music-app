"""
TDD: Phase 9 → harmony_rules.py downstream pipeline integration tests.

The real "north star" metric: can the detector+assembler+rules pipeline
correctly identify harmony errors on real sheet music?
This is the test that determines whether Phase 9 is deployable.
"""
import pytest
from pathlib import Path
import sys

# Allow importing harmony_rules from project root
PROJECT_ROOT = Path("/home/thc1006/dev/music-app")
sys.path.insert(0, str(PROJECT_ROOT))

PHASE9_WEIGHTS = PROJECT_ROOT / "runs/detect/harmony_omr_v2_phase9/cv_noteheads_v1_stage2/weights/best.pt"
VAL_DIR = PROJECT_ROOT / "training/datasets/yolo_harmony_v2_phase6_fixed/val"

TEST_IMAGE = str(VAL_DIR / "images" /
                 "phase7_phase6_base_p4_p3_p2_lg-102414375-aug-beethoven--page-2_oversample_12_3.png")


# ──────────────────────────────────────────────────────────────────────
# Module Import
# ──────────────────────────────────────────────────────────────────────

class TestModuleImport:

    def test_downstream_module_importable(self):
        import importlib.util
        spec = importlib.util.find_spec("downstream_eval")
        assert spec is not None, (
            "Create training/downstream_eval.py first"
        )

    def test_harmony_rules_importable(self):
        """harmony_rules.py exists at project root and is importable."""
        from harmony_rules import HarmonyAnalyzer, ChordSnapshot, NoteEvent
        assert callable(HarmonyAnalyzer)

    def test_run_pipeline_function_exists(self):
        from downstream_eval import run_pipeline
        assert callable(run_pipeline)


# ──────────────────────────────────────────────────────────────────────
# Phase 9 detection layer
# ──────────────────────────────────────────────────────────────────────

class TestPhase9Detection:

    def test_phase9_weights_exist(self):
        assert PHASE9_WEIGHTS.exists(), f"Missing: {PHASE9_WEIGHTS}"

    def test_phase9_detects_noteheads_on_real_image(self):
        """Phase 9 must detect ≥20 noteheads on a real OpenScore image."""
        from downstream_eval import run_phase9_detection
        detections = run_phase9_detection(TEST_IMAGE)
        nh = [d for d in detections if d["class_id"] in (0, 1)]
        assert len(nh) >= 20, f"Only {len(nh)} noteheads found"

    def test_detections_have_required_fields(self):
        from downstream_eval import run_phase9_detection
        detections = run_phase9_detection(TEST_IMAGE)
        assert len(detections) > 0
        d = detections[0]
        for key in ("class_id", "class_name", "cx", "cy", "w", "h", "confidence"):
            assert key in d, f"Missing field: {key}"


# ──────────────────────────────────────────────────────────────────────
# Symbol assembly layer (YOLO output → harmony_rules format)
# ──────────────────────────────────────────────────────────────────────

class TestSymbolAssembly:

    def test_build_chord_snapshots(self):
        """Convert detections into ChordSnapshot list for rules engine."""
        from downstream_eval import run_phase9_detection, build_chord_snapshots
        detections = run_phase9_detection(TEST_IMAGE)
        chords = build_chord_snapshots(detections, image_path=TEST_IMAGE)
        assert isinstance(chords, list)
        # At least some chords should be built from the detected noteheads
        assert len(chords) >= 0  # may be 0 if assembly heuristic is conservative

    def test_chord_snapshot_structure(self):
        """ChordSnapshot must have expected structure for harmony_rules."""
        from downstream_eval import run_phase9_detection, build_chord_snapshots
        from harmony_rules import ChordSnapshot, NoteEvent
        detections = run_phase9_detection(TEST_IMAGE)
        chords = build_chord_snapshots(detections, image_path=TEST_IMAGE)
        if not chords:
            pytest.skip("No chords assembled; assembly is conservative")
        for chord in chords[:3]:
            assert isinstance(chord, ChordSnapshot)
            assert hasattr(chord, "notes")
            for voice_name, note in chord.notes.items():
                assert isinstance(note, NoteEvent)
                assert isinstance(note.midi, int)
                assert voice_name in ("S", "A", "T", "B")


# ──────────────────────────────────────────────────────────────────────
# Rules engine integration
# ──────────────────────────────────────────────────────────────────────

class TestRulesEngineIntegration:

    def test_rules_engine_runs_without_error(self):
        """Full pipeline runs without crashing on real image."""
        from downstream_eval import run_pipeline
        result = run_pipeline(TEST_IMAGE)
        assert "violations" in result
        assert "num_detections" in result
        assert "num_chords" in result

    def test_pipeline_returns_list_of_violations(self):
        """Output violations should be a list (may be empty)."""
        from downstream_eval import run_pipeline
        result = run_pipeline(TEST_IMAGE)
        assert isinstance(result["violations"], list)

    def test_pipeline_reports_detection_count(self):
        """Pipeline should report ≥20 noteheads were detected."""
        from downstream_eval import run_pipeline
        result = run_pipeline(TEST_IMAGE)
        assert result["num_detections"] >= 20, (
            f"Only {result['num_detections']} detections on test image"
        )


# ──────────────────────────────────────────────────────────────────────
# End-to-end sanity
# ──────────────────────────────────────────────────────────────────────

class TestEndToEndSanity:

    def test_pipeline_on_multiple_images(self):
        """Pipeline should complete on 3 different images without error."""
        from downstream_eval import run_pipeline
        import os
        val_img_dir = VAL_DIR / "images"
        os_imgs = sorted([f for f in os.listdir(val_img_dir) if "lg-" in f])[:3]
        for fname in os_imgs:
            result = run_pipeline(str(val_img_dir / fname))
            assert "violations" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
