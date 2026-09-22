"""
Tests for Step 3: sizing_engine.py
Covers is_in_focus() and measure_particle().
"""

import numpy as np
import pytest

from src.sizing.sizing_engine import ParticleDimensions, is_in_focus, measure_particle


# ---------------------------------------------------------------------------
# is_in_focus
# ---------------------------------------------------------------------------

class TestIsInFocus:
    def test_sharp_image_passes(self):
        """A synthetic image with a hard black/white edge has high Laplacian variance."""
        # 50x50 image: left half black, right half white — very sharp edge
        sharp = np.zeros((50, 50, 3), dtype=np.uint8)
        sharp[:, 25:] = 255
        assert is_in_focus(sharp, sharpness_threshold=100.0) is True

    def test_blurry_image_fails(self):
        """A uniform grey image has near-zero Laplacian variance."""
        blurry = np.full((50, 50, 3), 128, dtype=np.uint8)
        assert is_in_focus(blurry, sharpness_threshold=100.0) is False

    def test_custom_threshold_low_accepts_blurry(self):
        """Setting threshold to 0 accepts everything including flat images."""
        flat = np.full((30, 30, 3), 200, dtype=np.uint8)
        assert is_in_focus(flat, sharpness_threshold=0.0) is True

    def test_returns_bool(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = is_in_focus(img)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# measure_particle
# ---------------------------------------------------------------------------

class TestMeasureParticle:
    def test_square_polygon(self):
        """A 10×10 pixel square → length == width == 10 * px_to_um."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        dims = measure_particle(polygon, px_to_um=1.0)
        assert isinstance(dims, ParticleDimensions)
        assert dims.length_um == pytest.approx(10.0, abs=0.1)
        assert dims.width_um == pytest.approx(10.0, abs=0.1)
        assert dims.aspect_ratio == pytest.approx(1.0, abs=0.01)

    def test_elongated_fibre_shape(self):
        """A 2×20 px rectangle should have aspect_ratio ≈ 10."""
        polygon = [(0, 0), (20, 0), (20, 2), (0, 2)]
        dims = measure_particle(polygon, px_to_um=1.0)
        # length should be ~20, width ~2
        assert dims.length_um >= dims.width_um
        assert dims.aspect_ratio == pytest.approx(10.0, abs=0.5)

    def test_px_to_um_scaling(self):
        """Calibration factor should scale lengths linearly."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        dims_1x = measure_particle(polygon, px_to_um=1.0)
        dims_2x = measure_particle(polygon, px_to_um=2.0)
        assert dims_2x.length_um == pytest.approx(dims_1x.length_um * 2, abs=0.1)
        assert dims_2x.surface_area_um2 == pytest.approx(dims_1x.surface_area_um2 * 4, abs=1.0)

    def test_surface_area_is_length_times_width(self):
        polygon = [(0, 0), (4, 0), (4, 6), (0, 6)]
        dims = measure_particle(polygon, px_to_um=0.5)
        expected_area = dims.length_um * dims.width_um
        assert dims.surface_area_um2 == pytest.approx(expected_area, abs=0.01)

    def test_aspect_ratio_always_gte_one(self):
        """aspect_ratio must always be >= 1 regardless of polygon orientation."""
        wide = [(0, 0), (20, 0), (20, 2), (0, 2)]
        tall = [(0, 0), (2, 0), (2, 20), (0, 20)]
        for poly in [wide, tall]:
            dims = measure_particle(poly, px_to_um=1.0)
            assert dims.aspect_ratio >= 1.0

    def test_too_few_points_raises(self):
        """Fewer than 3 polygon points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            measure_particle([(0, 0), (1, 1)], px_to_um=1.0)

    def test_three_point_triangle(self):
        """Three-point polygon (triangle) is the minimum valid input."""
        triangle = [(0, 0), (10, 0), (5, 8)]
        dims = measure_particle(triangle, px_to_um=1.0)
        assert dims.length_um > 0
        assert dims.width_um > 0
