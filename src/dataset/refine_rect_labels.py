"""
Refines box-fallback fibre labels into real polygons with SAM.

build_fibre_finetune_set.py writes a 4-point rectangle whenever a reviewed
instance had no polygon. For a thin diagonal fibre that rectangle is mostly
background, which teaches the model rectangular masks and deflates mask
metrics. This prompts SAM with each rectangle as a box and replaces it with
the traced fibre polygon when the result passes sanity checks; otherwise the
rectangle is kept.

Non-destructive: reads data/real_labels/<batch>/ and writes a new batch
data/real_labels/<batch>_sam/ (images, labels, dataset.yaml) plus overlays/
showing old rectangles (red) vs refined polygons (green) for spot-checking.
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REAL_LABELS_ROOT = Path("data/real_labels")
SAM_WEIGHTS = "sam2.1_b.pt"
RECT_POINTS = 4
# A traced fibre must cover a meaningful but not near-total share of its box:
# near 0 means SAM missed it, near 1 means SAM just filled the box.
MIN_BOX_FILL = 0.02
MAX_BOX_FILL = 0.90
MIN_POLYGON_POINTS = 3
POLYGON_EPSILON_PX = 1.0
# The filter's printed grid lines are near-black (~40-60 grey) while fibres are
# pale-to-mid grey; SAM prompted with a box that crosses a grid line sometimes
# traces the line instead. Reject masks made mostly of grid-dark pixels.
GRID_DARK_LEVEL = 70
MAX_DARK_FRACTION = 0.5
GRID_HALO_PX = 7
OLD_COLOUR = (0, 0, 255)
NEW_COLOUR = (0, 255, 0)


@dataclass(frozen=True)
class Instance:
    class_id: int
    points: np.ndarray  # (N, 2) normalised xy


def parse_label_file(path: Path) -> list[Instance]:
    instances = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 1 + 2 * MIN_POLYGON_POINTS:
            continue
        instances.append(Instance(int(parts[0]), np.array(parts[1:], float).reshape(-1, 2)))
    return instances


def is_axis_aligned_rect(points: np.ndarray) -> bool:
    """True for the x1,y1,x2,y1,x2,y2,x1,y2 box fallback written by polygon_for()."""
    if len(points) != RECT_POINTS:
        return False
    xs, ys = points[:, 0], points[:, 1]
    return len(np.unique(xs.round(5))) == 2 and len(np.unique(ys.round(5))) == 2


def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    """Largest external contour of a binary mask as an (N, 2) pixel polygon."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, POLYGON_EPSILON_PX, closed=True).reshape(-1, 2)
    return approx if len(approx) >= MIN_POLYGON_POINTS else None


def accept_refinement(mask: np.ndarray, box_px: tuple[int, int, int, int]) -> bool:
    """Keep SAM's mask only if it plausibly traces one fibre inside the box."""
    x1, y1, x2, y2 = box_px
    box_area = max((x2 - x1) * (y2 - y1), 1)
    inside = mask[y1:y2, x1:x2].sum()
    total = mask.sum()
    if total == 0:
        return False
    fill = inside / box_area
    return MIN_BOX_FILL <= fill <= MAX_BOX_FILL and inside / total >= 0.9


def is_mostly_grid(mask: np.ndarray, gray: np.ndarray) -> bool:
    """True when most of the mask sits on near-black grid-line pixels."""
    total = mask.sum()
    return total > 0 and (gray[mask] < GRID_DARK_LEVEL).sum() / total > MAX_DARK_FRACTION


def strip_grid(mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Drop grid-line pixels (plus their anti-aliased grey halo) a fibre mask leaked onto."""
    grid = cv2.dilate((gray < GRID_DARK_LEVEL).astype(np.uint8), np.ones((GRID_HALO_PX,) * 2, np.uint8))
    return mask & (grid == 0)


def format_line(class_id: int, points: np.ndarray) -> str:
    return f"{class_id} " + " ".join(f"{c:.5f}" for c in points.reshape(-1))


def refine_image(sam, image_path: Path, instances: list[Instance]) -> tuple[list[str], np.ndarray, dict]:
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    scale = np.array([w, h])
    overlay = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stats = {"rects": 0, "refined": 0, "kept_rect": 0, "polygons": 0}

    rect_idx = [i for i, inst in enumerate(instances) if is_axis_aligned_rect(inst.points)]
    boxes = []
    for i in rect_idx:
        px = instances[i].points * scale
        boxes.append([int(px[:, 0].min()), int(px[:, 1].min()), int(np.ceil(px[:, 0].max())), int(np.ceil(px[:, 1].max()))])

    masks = []
    if boxes:
        result = sam(image, bboxes=boxes, verbose=False)[0]
        masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                 for m in result.masks.data.cpu().numpy()]

    refined = dict(zip(rect_idx, zip(boxes, masks)))
    lines = []
    for i, inst in enumerate(instances):
        if i not in refined:
            stats["polygons"] += 1
            lines.append(format_line(inst.class_id, inst.points))
            continue
        stats["rects"] += 1
        box, mask = refined[i]
        plausible = accept_refinement(mask, box) and not is_mostly_grid(mask, gray)
        polygon = mask_to_polygon(strip_grid(mask, gray)) if plausible else None
        cv2.rectangle(overlay, box[:2], box[2:], OLD_COLOUR, 1)
        if polygon is None:
            stats["kept_rect"] += 1
            lines.append(format_line(inst.class_id, inst.points))
            continue
        stats["refined"] += 1
        cv2.polylines(overlay, [polygon.astype(np.int32)], True, NEW_COLOUR, 1)
        lines.append(format_line(inst.class_id, polygon / scale))
    return lines, overlay, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine rectangle fibre labels into polygons with SAM.")
    parser.add_argument("--batch-name", default="fibre_batch1")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import SAM

    src = REAL_LABELS_ROOT / args.batch_name
    dst = REAL_LABELS_ROOT / f"{args.batch_name}_sam"
    if dst.exists():
        raise SystemExit(f"{dst} already exists; remove it or pick another batch name.")
    for sub in ("images", "labels", "overlays"):
        (dst / sub).mkdir(parents=True)

    sam = SAM(SAM_WEIGHTS)
    sam.to("cuda" if args.device != "cpu" else "cpu")
    totals = {"rects": 0, "refined": 0, "kept_rect": 0, "polygons": 0}
    for image_path in sorted((src / "images").glob("*.jpg")):
        label_path = src / "labels" / f"{image_path.stem}.txt"
        instances = parse_label_file(label_path) if label_path.exists() else []
        lines, overlay, stats = refine_image(sam, image_path, instances)
        shutil.copy2(image_path, dst / "images" / image_path.name)
        (dst / "labels" / label_path.name).write_text("\n".join(lines) + ("\n" if lines else ""))
        cv2.imwrite(str(dst / "overlays" / image_path.name), overlay)
        totals = {k: totals[k] + stats[k] for k in totals}

    yaml = (src / "dataset.yaml").read_text().replace(str(src.resolve()).replace("\\", "/"), str(dst.resolve()).replace("\\", "/"))
    (dst / "dataset.yaml").write_text(yaml)
    (dst / "refine_report.json").write_text(json.dumps(totals, indent=2))
    print(f"[Refine] {totals['refined']}/{totals['rects']} rectangles refined, "
          f"{totals['kept_rect']} kept as rectangles, {totals['polygons']} polygons unchanged -> {dst}")


if __name__ == "__main__":
    main()
