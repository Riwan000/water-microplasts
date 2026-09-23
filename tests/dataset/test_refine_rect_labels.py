from pathlib import Path

import numpy as np

from src.dataset.refine_rect_labels import (
    accept_refinement,
    is_axis_aligned_rect,
    is_mostly_grid,
    mask_to_polygon,
    parse_label_file,
    strip_grid,
)


def test_is_axis_aligned_rect_detects_box_fallback() -> None:
    rect = np.array([[0.1, 0.2], [0.4, 0.2], [0.4, 0.5], [0.1, 0.5]])

    assert is_axis_aligned_rect(rect)


def test_is_axis_aligned_rect_rejects_real_polygons() -> None:
    quad = np.array([[0.1, 0.2], [0.4, 0.25], [0.4, 0.5], [0.1, 0.5]])
    pentagon = np.array([[0.1, 0.2], [0.4, 0.2], [0.4, 0.5], [0.2, 0.6], [0.1, 0.5]])

    assert not is_axis_aligned_rect(quad)
    assert not is_axis_aligned_rect(pentagon)


def test_accept_refinement_keeps_thin_fibre_inside_box() -> None:
    mask = np.zeros((100, 100), bool)
    for i in range(10, 90):
        mask[i, i - 1:i + 2] = True  # thin diagonal fibre

    assert accept_refinement(mask, (5, 5, 95, 95))


def test_accept_refinement_rejects_empty_filled_or_escaping_masks() -> None:
    empty = np.zeros((100, 100), bool)
    filled = np.zeros((100, 100), bool)
    filled[10:90, 10:90] = True
    escaping = np.zeros((100, 100), bool)
    escaping[40:60, 0:100] = True

    assert not accept_refinement(empty, (10, 10, 90, 90))
    assert not accept_refinement(filled, (10, 10, 90, 90))
    assert not accept_refinement(escaping, (40, 40, 60, 60))


def test_mask_to_polygon_traces_largest_component() -> None:
    mask = np.zeros((50, 50), np.uint8)
    mask[10:40, 20:24] = 1
    mask[2:4, 2:4] = 1

    polygon = mask_to_polygon(mask)

    assert polygon is not None
    assert polygon[:, 0].min() >= 20 and polygon[:, 0].max() <= 23
    assert mask_to_polygon(np.zeros((10, 10), np.uint8)) is None


def test_parse_label_file_skips_malformed_lines(tmp_path: Path) -> None:
    label = tmp_path / "a.txt"
    label.write_text("0 0.1 0.2 0.4 0.2 0.4 0.5 0.1 0.5\n0 0.1 0.2\n")

    instances = parse_label_file(label)

    assert len(instances) == 1
    assert instances[0].points.shape == (4, 2)


def test_is_mostly_grid_flags_masks_on_dark_grid_lines() -> None:
    gray = np.full((50, 50), 160, np.uint8)
    gray[:, 20:23] = 45  # grid line
    on_grid = np.zeros((50, 50), bool)
    on_grid[5:45, 20:23] = True
    fibre = np.zeros((50, 50), bool)
    fibre[5:45, 30:32] = True

    assert is_mostly_grid(on_grid, gray)
    assert not is_mostly_grid(fibre, gray)


def test_strip_grid_removes_grid_line_and_its_halo_only() -> None:
    gray = np.full((40, 40), 160, np.uint8)
    gray[:, 20] = 45
    mask = np.zeros((40, 40), bool)
    mask[5, 0:40] = True  # fibre crossing the grid line

    stripped = strip_grid(mask, gray)

    assert not stripped[5, 17:24].any()
    assert stripped[5, 0:10].all() and stripped[5, 30:40].all()
