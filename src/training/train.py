"""
Step 2: Multi-Model Training
Trains each candidate segmentation architecture (yolov8n-seg, yolo11n-seg,
yolo11s-seg) on data/yolo_dataset and writes weights under models/<arch>/.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

CANDIDATE_MODELS = ["yolov8n-seg", "yolo11n-seg", "yolo11s-seg"]
MODELS_DIR = "models"
IMG_SIZE = 640
# Conservative for an 8GB laptop GPU + 14GB system RAM: avoids the CUDA OOM
# auto-retry (default batch=16) and the multi-worker dataloader spawn storm
# (default workers=8) that exhausted available memory on this machine.
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 2
ARCHIVE_DIR = "archive"


def archive_existing_run(run_dir: Path) -> Path | None:
    """Move a previous run out of the way so a new run never overwrites it.

    Keeps models/<name>/ as the stable path benchmark.py reads, while old
    runs are preserved under models/archive/<name>-<timestamp>/.
    """
    if not run_dir.exists():
        return None
    stamp = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    destination = run_dir.parent / ARCHIVE_DIR / f"{run_dir.name}-{stamp}"
    if destination.exists():
        raise FileExistsError(f"Archive target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(run_dir), str(destination))
    except PermissionError as e:
        raise PermissionError(
            f"Cannot archive {run_dir}: a file is in use (stop the backend if it loads these weights)."
        ) from e
    return destination


def train_model(
    model_name: str,
    dataset_yaml: str,
    epochs: int,
    device: str,
    batch: int = DEFAULT_BATCH,
    workers: int = DEFAULT_WORKERS,
) -> str:
    """Train one candidate model and return the path to its best weights."""
    archived = archive_existing_run(Path(MODELS_DIR) / model_name)
    if archived:
        print(f"[Train] Archived previous run to {archived}")
    model = YOLO(f"{model_name}.pt")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=IMG_SIZE,
        device=device,
        batch=batch,
        workers=workers,
        # ultralytics rebases a relative `project` under runs/<task>/, so this
        # must be absolute to actually land under MODELS_DIR as intended.
        project=str(Path(MODELS_DIR).resolve()),
        name=model_name,
        exist_ok=True,
    )
    return str(Path(results.save_dir) / "weights" / "best.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train microplastic segmentation candidates.")
    parser.add_argument("--model", choices=CANDIDATE_MODELS, required=True)
    parser.add_argument("--dataset", default="data/yolo_dataset/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    best_weights = train_model(args.model, args.dataset, args.epochs, args.device, args.batch, args.workers)
    print(f"[Train] {args.model} best weights: {best_weights}")


if __name__ == "__main__":
    main()
