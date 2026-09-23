import os
from pathlib import Path

import pytest

from src.training.train import archive_existing_run


def test_archive_existing_run_returns_none_when_no_previous_run(tmp_path: Path) -> None:
    assert archive_existing_run(tmp_path / "yolo11s-seg") is None


def test_archive_existing_run_moves_previous_run_under_archive(tmp_path: Path) -> None:
    run_dir = tmp_path / "yolo11s-seg"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"old")

    archived = archive_existing_run(run_dir)

    assert not run_dir.exists()
    assert archived.parent == tmp_path / "archive"
    assert archived.name.startswith("yolo11s-seg-")
    assert (archived / "weights" / "best.pt").read_bytes() == b"old"


def test_archive_existing_run_refuses_to_overwrite_existing_archive(tmp_path: Path) -> None:
    run_dir = tmp_path / "yolo11s-seg"
    run_dir.mkdir()
    first = archive_existing_run(run_dir)
    run_dir.mkdir()
    # Pin the new run's mtime to the archived one so both map to the same name.
    os.utime(run_dir, (first.stat().st_atime, first.stat().st_mtime))

    with pytest.raises(FileExistsError):
        archive_existing_run(run_dir)
    assert first.exists()
