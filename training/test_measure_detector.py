"""
TDD Red: Measure detector tests.

Uses barline x-positions to segment noteheads into measures.
"""
import pytest


def _make_barline(x, y=150):
    return {"class_id": 22, "class_name": "barline", "cx": x, "cy": y,
            "w": 3, "h": 100, "confidence": 0.9}


def _make_barline_final(x):
    return {"class_id": 24, "class_name": "barline_final", "cx": x, "cy": 150,
            "w": 5, "h": 100, "confidence": 0.9}


def _make_notehead(x, y=100):
    return {"class_id": 0, "class_name": "notehead_filled", "cx": x, "cy": y,
            "w": 20, "h": 25, "confidence": 0.9}


class TestModuleImport:
    def test_module_importable(self):
        import importlib.util
        spec = importlib.util.find_spec("measure_detector")
        assert spec is not None

    def test_segment_measures_exists(self):
        from measure_detector import segment_measures
        assert callable(segment_measures)


class TestBarlineSegmentation:

    def test_barlines_segment_three_measures(self):
        """3 barlines → 3 measures (before, between, after)."""
        from measure_detector import segment_measures
        detections = [
            _make_notehead(x=50),
            _make_notehead(x=150),  # in measure 1
            _make_barline(x=200),   # barline
            _make_notehead(x=250),  # in measure 2
            _make_barline(x=400),
            _make_notehead(x=450),  # in measure 3
        ]
        measures = segment_measures(detections)
        assert len(measures) == 3

    def test_measure_numbers_start_at_1(self):
        from measure_detector import segment_measures
        detections = [
            _make_notehead(x=50),
            _make_barline(x=100),
            _make_notehead(x=150),
        ]
        measures = segment_measures(detections)
        assert measures[0]["number"] == 1

    def test_measure_numbers_ascending(self):
        from measure_detector import segment_measures
        detections = [
            _make_notehead(x=50),
            _make_barline(x=100),
            _make_notehead(x=150),
            _make_barline(x=200),
            _make_notehead(x=250),
        ]
        measures = segment_measures(detections)
        nums = [m["number"] for m in measures]
        assert nums == sorted(nums)
        assert nums[0] == 1


class TestNoteheadAssignment:

    def test_each_notehead_belongs_to_a_measure(self):
        from measure_detector import segment_measures
        nh1 = _make_notehead(x=50)
        nh2 = _make_notehead(x=250)
        detections = [nh1, _make_barline(x=200), nh2]
        measures = segment_measures(detections)
        # nh1 in measure 1, nh2 in measure 2
        assert any(nh1 in m["noteheads"] for m in measures)
        assert any(nh2 in m["noteheads"] for m in measures)

    def test_noteheads_sorted_by_x_within_measure(self):
        from measure_detector import segment_measures
        detections = [
            _make_notehead(x=150),
            _make_notehead(x=50),
            _make_notehead(x=100),
            _make_barline(x=200),
        ]
        measures = segment_measures(detections)
        first = measures[0]
        xs = [n["cx"] for n in first["noteheads"]]
        assert xs == sorted(xs)


class TestBeatAssignment:

    def test_noteheads_assigned_beat_number(self):
        """Each notehead gets a beat index within its measure."""
        from measure_detector import segment_measures
        detections = [
            _make_notehead(x=50),
            _make_notehead(x=100),
            _make_notehead(x=150),
            _make_barline(x=200),
        ]
        measures = segment_measures(detections)
        first = measures[0]
        # beat_in_measure can be a float (chord position)
        beats = [n.get("beat_in_measure") for n in first["noteheads"]]
        assert all(b is not None for b in beats)
        assert beats == sorted(beats)


class TestEdgeCases:

    def test_no_barlines_single_measure(self):
        """If no barlines, everything is measure 1."""
        from measure_detector import segment_measures
        detections = [_make_notehead(x=50), _make_notehead(x=150)]
        measures = segment_measures(detections)
        assert len(measures) == 1
        assert measures[0]["number"] == 1
        assert len(measures[0]["noteheads"]) == 2

    def test_empty_detections_returns_empty(self):
        from measure_detector import segment_measures
        measures = segment_measures([])
        assert measures == []

    def test_final_barline_counted(self):
        from measure_detector import segment_measures
        detections = [
            _make_notehead(x=50),
            _make_barline_final(x=100),
        ]
        measures = segment_measures(detections)
        assert len(measures) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
