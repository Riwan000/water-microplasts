from pathlib import Path

import pytest

from src.training.benchmark import BenchmarkResult, load_report, promote_champion, write_report


def _result(name: str, map50_95: float = 0.5) -> BenchmarkResult:
    return BenchmarkResult(model_name=name, mask_map50_95=map50_95, thin_fibre_recall=0.6,
                           latency_ms_gpu=10.0, latency_ms_cpu=90.0)


def test_promote_champion_copies_best_weights_to_champion_path(tmp_path: Path) -> None:
    weights = tmp_path / "models" / "yolo11s-seg" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"weights")
    champion_path = tmp_path / "out" / "champion.pt"

    result = promote_champion(_result("yolo11s-seg"), str(tmp_path / "models"), str(champion_path))

    assert result == champion_path
    assert champion_path.read_bytes() == b"weights"


def test_promote_champion_is_noop_when_champion_path_is_the_source(tmp_path: Path) -> None:
    weights = tmp_path / "yolo11s-seg" / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    best.write_bytes(b"weights")

    result = promote_champion(_result("yolo11s-seg"), str(tmp_path), str(best))

    assert result == best
    assert best.read_bytes() == b"weights"


def test_promote_champion_raises_when_best_weights_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="yolo11s-seg"):
        promote_champion(_result("yolo11s-seg"), str(tmp_path), str(tmp_path / "champion.pt"))


def test_load_report_round_trips_write_report(tmp_path: Path) -> None:
    results = [_result("a", 0.4), _result("b", 0.7)]
    report = tmp_path / "report.json"

    write_report(results, str(report))

    assert load_report(str(report)) == results
