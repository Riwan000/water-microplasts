from pathlib import Path

import numpy as np
import pytest

from src.dataset.label_studio_to_yolo import image_path_from_url, region_to_points, task_to_lines
from src.dataset.prelabel_label_studio import (
    build_label_config,
    local_file_url,
    polygon_to_percent,
    select_unlabelled,
)

NAMES = ["fibre", "fragment", "pellet", "foam_film", "algae"]


def _polygon(label: str, points: list[list[float]]) -> dict:
    return {"type": "polygonlabels", "value": {"points": points, "polygonlabels": [label]}}


def _rect(label: str, x: float, y: float, w: float, h: float, rotation: float = 0) -> dict:
    return {"type": "rectanglelabels",
            "value": {"x": x, "y": y, "width": w, "height": h, "rotation": rotation, "rectanglelabels": [label]}}


def test_select_unlabelled_excludes_labelled_and_is_seeded() -> None:
    pool = [Path(f"real_{i:03d}.jpg") for i in range(20)]
    labelled = {"real_000.jpg", "real_001.jpg"}

    picked = select_unlabelled(pool, labelled, count=5, seed=1)

    assert len(picked) == 5
    assert not {p.name for p in picked} & labelled
    assert picked == select_unlabelled(pool, labelled, count=5, seed=1)


def test_polygon_to_percent_scales_to_image_size() -> None:
    assert polygon_to_percent(np.array([[50, 25], [100, 50]]), 200, 100) == [[25.0, 25.0], [50.0, 50.0]]


def test_local_file_url_round_trips_through_converter(tmp_path: Path) -> None:
    image = tmp_path / "data" / "x" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    url = local_file_url(image, tmp_path)

    assert url == "/data/local-files/?d=data/x/a.jpg"
    assert image_path_from_url(url, tmp_path) == tmp_path / "data" / "x" / "a.jpg"


def test_label_config_has_polygon_and_box_tools_for_every_class() -> None:
    config = build_label_config(NAMES)

    assert "<PolygonLabels" in config and "<RectangleLabels" in config
    assert all(config.count(f'value="{n}"') == 2 for n in NAMES)


def test_region_to_points_converts_polygon_and_rectangle() -> None:
    assert region_to_points(_polygon("fibre", [[10, 20], [30, 40], [50, 20]])) == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.2)]
    assert np.allclose(region_to_points(_rect("fibre", 10, 20, 30, 40)), [(0.1, 0.2), (0.4, 0.2), (0.4, 0.6), (0.1, 0.6)])


def test_region_to_points_rejects_rotated_rectangles() -> None:
    with pytest.raises(ValueError):
        region_to_points(_rect("fibre", 10, 20, 30, 40, rotation=15))


def test_task_to_lines_skips_unreviewed_and_cancelled_tasks() -> None:
    assert task_to_lines({"predictions": [{"result": [_polygon("fibre", [[1, 1], [2, 2], [3, 1]])]}]}, NAMES) is None
    assert task_to_lines({"annotations": [{"was_cancelled": True, "result": []}]}, NAMES) is None


def test_task_to_lines_writes_class_ids_and_keeps_empty_reviewed_photos() -> None:
    task = {"annotations": [{"result": [_polygon("fragment", [[10, 10], [20, 10], [20, 20]]), _rect("fibre", 0, 0, 10, 10)]}]}

    lines = task_to_lines(task, NAMES)

    assert lines[0].startswith("1 ") and lines[1].startswith("0 ")
    assert len(lines[1].split()) == 9
    assert task_to_lines({"annotations": [{"result": []}]}, NAMES) == []


def test_task_to_lines_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError):
        task_to_lines({"annotations": [{"result": [_polygon("glass", [[1, 1], [2, 2], [3, 1]])]}]}, NAMES)
