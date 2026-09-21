"""
Step 2: Multi-Model Training
Trains each candidate segmentation architecture (yolov8n-seg, yolo11n-seg,
yolo11s-seg) on data/yolo_dataset and writes weights under models/<arch>/.
"""

import argparse
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


def train_model(
    model_name: str,
    dataset_yaml: str,
    epochs: int,
    device: str,
    batch: int = DEFAULT_BATCH,
    workers: int = DEFAULT_WORKERS,
) -> str:
    """Train one candidate model and return the path to its best weights."""
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
