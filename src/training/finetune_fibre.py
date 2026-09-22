"""
Fine-tunes the yolo11s-seg champion on the fibre rehearsal set built by
build_fibre_finetune_set.py: warm-started from its own best.pt, frozen
backbone, short epoch budget, no mosaic - adapting to real fibre appearance
rather than relearning features from scratch.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = "models"
IMG_SIZE = 640
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune yolo11s-seg on the fibre rehearsal set.")
    parser.add_argument("--weights", default="models/yolo11s-seg/weights/best.pt")
    parser.add_argument("--dataset", default="data/finetune_fibre/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="yolo11s-seg-finetune-fibre")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.train(
        data=args.dataset,
        epochs=args.epochs,
        patience=args.patience,
        freeze=args.freeze,
        imgsz=IMG_SIZE,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        cache="ram",
        mosaic=0,
        project=str(Path(MODELS_DIR).resolve()),
        name=args.name,
        exist_ok=True,
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[Finetune] best weights: {best_weights}")


if __name__ == "__main__":
    main()
