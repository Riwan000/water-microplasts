"""
Tests for Step 3: two_tier.py
Covers compute_tier1() and compute_tier2().
"""

import pytest

from src.sizing.two_tier import (
    PLASTIC_CLASSES,
    Tier1Summary,
    compute_tier1,
    compute_tier2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(class_name: str, confidence: float = 0.9, particle_id: int = 1) -> dict:
    return {
        "particle_id": particle_id,
        "class_name": class_name,
        "confidence": confidence,
        "length_um": 50.0,
        "width_um": 10.0,
        "aspect_ratio": 5.0,
        "surface_area_um2": 500.0,
        "bbox": [10.0, 20.0, 60.0, 30.0],
    }


# ---------------------------------------------------------------------------
# compute_tier1
# ---------------------------------------------------------------------------

class TestComputeTier1:
    def test_basic_count_and_concentration(self):
        """3 fibres in 1 L → 3 particles/L."""
        detections = [_make_detection("fibre", particle_id=i) for i in range(3)]
        result = compute_tier1(detections, filtered_volume_litres=1.0)
        assert isinstance(result, Tier1Summary)
        assert result.total_plastic_count == 3
        assert result.particles_per_litre == pytest.approx(3.0, abs=0.001)

    def test_concentration_formula(self):
        """10 particles in 0.5 L → 20 particles/L."""
        detections = [_make_detection("fragment", particle_id=i) for i in range(10)]
        result = compute_tier1(detections, filtered_volume_litres=0.5)
        assert result.particles_per_litre == pytest.approx(20.0, abs=0.001)

    def test_algae_excluded_from_count(self):
        """Algae detections must not contribute to Tier 1 count."""
        detections = [
            _make_detection("fibre", particle_id=1),
            _make_detection("algae", particle_id=2),
            _make_detection("fragment", particle_id=3),
        ]
        result = compute_tier1(detections, filtered_volume_litres=1.0)
        assert result.total_plastic_count == 2

    def test_all_plastic_classes_counted(self):
        """Each of the 4 plastic morphotypes must be counted."""
        detections = [
            _make_detection("fibre"),
            _make_detection("fragment"),
            _make_detection("pellet"),
            _make_detection("foam_film"),
        ]
        result = compute_tier1(detections, filtered_volume_litres=1.0)
        assert result.total_plastic_count == 4

    def test_empty_detections(self):
        """No detections → count 0 and concentration 0."""
        result = compute_tier1([], filtered_volume_litres=1.0)
        assert result.total_plastic_count == 0
        assert result.particles_per_litre == 0.0

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_tier1([], filtered_volume_litres=0.0)

    def test_negative_volume_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_tier1([], filtered_volume_litres=-1.0)

    def test_unknown_class_excluded(self):
        """Unknown / unclassified debris should not appear in Tier 1."""
        detections = [
            _make_detection("unclassified_debris"),
            _make_detection("fibre"),
        ]
        result = compute_tier1(detections, filtered_volume_litres=1.0)
        assert result.total_plastic_count == 1


# ---------------------------------------------------------------------------
# compute_tier2
# ---------------------------------------------------------------------------

class TestComputeTier2:
    def test_returns_list_of_dicts(self):
        detections = [_make_detection("fibre")]
        result = compute_tier2(detections)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_all_fields_present(self):
        """Every Tier 2 row must contain the required output keys."""
        required_keys = {
            "particle_id", "class_name", "confidence",
            "length_um", "width_um", "aspect_ratio", "surface_area_um2",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        }
        result = compute_tier2([_make_detection("fibre")])
        assert required_keys.issubset(result[0].keys())

    def test_algae_excluded_from_tier2(self):
        detections = [
            _make_detection("fibre", particle_id=1),
            _make_detection("algae", particle_id=2),
        ]
        result = compute_tier2(detections)
        assert len(result) == 1
        assert result[0]["class_name"] == "fibre"

    def test_all_plastic_classes_in_tier2(self):
        detections = [_make_detection(cls, particle_id=i) for i, cls in enumerate(PLASTIC_CLASSES)]
        result = compute_tier2(detections)
        assert len(result) == len(PLASTIC_CLASSES)

    def test_confidence_rounded_to_4dp(self):
        det = _make_detection("pellet", confidence=0.876543)
        result = compute_tier2([det])
        assert result[0]["confidence"] == pytest.approx(0.8765, abs=0.0001)

    def test_bbox_fields_populated(self):
        det = _make_detection("foam_film")
        result = compute_tier2([det])
        assert result[0]["bbox_x1"] == 10.0
        assert result[0]["bbox_y2"] == 30.0

    def test_empty_detections_returns_empty_list(self):
        assert compute_tier2([]) == []
