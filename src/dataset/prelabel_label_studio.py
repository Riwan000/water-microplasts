"""
Pre-labels unlabelled real filter photos for review in Label Studio.

Samples photos from data/cleaned/benchmark_real/test that are not already in
a labelled batch, runs the champion model, tightens each detection's outline
with SAM (box prompt, grid lines stripped; falls back to the model's own
mask), and writes Label Studio import files:

  <out>/tasks.json          tasks with polygon predictions (pre-annotations)
  <out>/label_config.xml    labeling interface: polygons to keep/edit,
                            rectangles to quickly box missed particles

Images are served by Label Studio's local-files storage rooted at the
project directory (see README "Labelling real photos").
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.backend.config import load_settings
from src.dataset.refine_rect_labels import (
    SAM_WEIGHTS,
    accept_refinement,
    is_mostly_grid,
    mask_to_polygon,
    strip_grid,
)
from src.training.build_fibre_finetune_set import DATASET_YAML_NAMES

POOL_DIR = Path("data/cleaned/benchmark_real/test")
REAL_LABELS_ROOT = Path("data/real_labels")
DEFAULT_CONF = 0.25
PERCENT = 100.0
LABEL_COLOURS = ["#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#14b8a6"]


def select_unlabelled(pool: list[Path], labelled_names: set[str], count: int, seed: int) -> list[Path]:
    candidates = sorted(p for p in pool if p.name not in labelled_names)
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, min(count, len(candidates))))


def polygon_to_percent(polygon_px: np.ndarray, width: int, height: int) -> list[list[float]]:
    return [[round(float(x) / width * PERCENT, 3), round(float(y) / height * PERCENT, 3)] for x, y in polygon_px]


def local_file_url(path: Path, document_root: Path) -> str:
    return "/data/local-files/?d=" + path.resolve().relative_to(document_root.resolve()).as_posix()


def build_label_config(names: list[str]) -> str:
    labels = "\n".join(
        f'    <Label value="{n}" background="{LABEL_COLOURS[i % len(LABEL_COLOURS)]}"/>' for i, n in enumerate(names)
    )
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>\n'
        '  <Header value="Keep/fix polygons; delete false detections; draw a BOX around any missed particle."/>\n'
        '  <PolygonLabels name="label" toName="image" strokeWidth="2" opacity="0.25" pointSize="small">\n'
        f"{labels}\n"
        "  </PolygonLabels>\n"
        '  <RectangleLabels name="box" toName="image" strokeWidth="2">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


def outline_detections(sam, image: np.ndarray, boxes: list[list[int]], model_masks: np.ndarray) -> list[np.ndarray]:
    """SAM-tightened polygon per detection, falling back to the model's own mask."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sam_masks = sam(image, bboxes=boxes, verbose=False)[0].masks.data.cpu().numpy() if boxes else []
    polygons = []
    for box, model_mask, sam_mask in zip(boxes, model_masks, sam_masks):
        sam_mask = cv2.resize(sam_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        use_sam = accept_refinement(sam_mask, tuple(box)) and not is_mostly_grid(sam_mask, gray)
        polygon = mask_to_polygon(strip_grid(sam_mask, gray)) if use_sam else None
        polygons.append(polygon if polygon is not None else mask_to_polygon(model_mask))
    return polygons


def prelabel_image(model, sam, path: Path, conf: float, document_root: Path, model_version: str) -> dict:
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    result = model.predict(image, conf=conf, retina_masks=True, verbose=False)[0]
    regions = []
    if result.masks is not None:
        boxes = [[int(v) for v in b] for b in result.boxes.xyxy.cpu().numpy()]
        model_masks = result.masks.data.cpu().numpy() > 0.5
        polygons = outline_detections(sam, image, boxes, model_masks)
        for i, polygon in enumerate(polygons):
            if polygon is None:
                continue
            regions.append({
                "id": f"p{i}",
                "from_name": "label", "to_name": "image", "type": "polygonlabels",
                "original_width": w, "original_height": h,
                "score": round(float(result.boxes.conf[i]), 4),
                "value": {
                    "points": polygon_to_percent(polygon, w, h),
                    "polygonlabels": [result.names[int(result.boxes.cls[i])]],
                },
            })
    return {
        "data": {"image": local_file_url(path, document_root)},
        "predictions": [{"model_version": model_version, "result": regions}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-label real photos for Label Studio review.")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--weights", default=None, help="Defaults to CHAMPION_WEIGHTS_PATH")
    parser.add_argument("--out", default="data/label_studio/batch2")
    args = parser.parse_args()

    from ultralytics import SAM, YOLO

    weights = args.weights or load_settings().champion_weights_path
    labelled = {p.name for p in REAL_LABELS_ROOT.glob("*/images/*.jpg")}
    photos = select_unlabelled(sorted(POOL_DIR.glob("*.jpg")), labelled, args.count, args.seed)

    model, sam = YOLO(weights), SAM(SAM_WEIGHTS)
    document_root = Path.cwd()
    tasks = [prelabel_image(model, sam, p, args.conf, document_root, Path(weights).parent.parent.name) for p in photos]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps(tasks))
    (out / "label_config.xml").write_text(build_label_config(DATASET_YAML_NAMES))
    n_regions = sum(len(t["predictions"][0]["result"]) for t in tasks)
    print(f"[Prelabel] {len(tasks)} photos, {n_regions} pre-drawn outlines -> {out / 'tasks.json'}")


if __name__ == "__main__":
    main()
