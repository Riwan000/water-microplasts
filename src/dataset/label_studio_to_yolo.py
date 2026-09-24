"""
Converts a Label Studio JSON export into a YOLO-seg real label batch.

Only tasks with a submitted (non-cancelled) annotation are converted, so
photos you have not reviewed never become ground truth. Polygons are written
as-is; rectangles (quick boxes around missed particles) are written as
4-point rectangles for refine_rect_labels.py to turn into outlines with SAM:

  python -m src.dataset.label_studio_to_yolo --export export.json --batch-name fibre_batch2
  python -m src.dataset.refine_rect_labels --batch-name fibre_batch2
"""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.training.build_fibre_finetune_set import DATASET_YAML_NAMES

REAL_LABELS_ROOT = Path("data/real_labels")
PERCENT = 100.0


def image_path_from_url(url: str, document_root: Path) -> Path:
    """Resolve a /data/local-files/?d=<relative path> URL back to a file."""
    relative = parse_qs(urlparse(url).query).get("d", [""])[0]
    if not relative:
        raise ValueError(f"Not a local-files URL: {url}")
    return document_root / relative


def region_to_points(region: dict) -> list[tuple[float, float]] | None:
    """Normalised (0-1) polygon for a polygon or rectangle region; None for other types."""
    value = region["value"]
    if region["type"] == "polygonlabels":
        return [(x / PERCENT, y / PERCENT) for x, y in value["points"]]
    if region["type"] == "rectanglelabels":
        if value.get("rotation", 0):
            raise ValueError("Rotated rectangles are not supported; draw axis-aligned boxes.")
        x1, y1 = value["x"] / PERCENT, value["y"] / PERCENT
        x2, y2 = x1 + value["width"] / PERCENT, y1 + value["height"] / PERCENT
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return None


def region_label(region: dict) -> str:
    return region["value"][region["type"]][0]


def clamp(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def task_to_lines(task: dict, names: list[str]) -> list[str] | None:
    """YOLO-seg lines for a reviewed task, or None when it has no usable annotation."""
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not annotations:
        return None
    lines = []
    for region in annotations[-1]["result"]:
        points = region_to_points(region)
        if points is None:
            continue
        label = region_label(region)
        if label not in names:
            raise ValueError(f"Unknown label '{label}'; expected one of {names}")
        coords = " ".join(f"{clamp(x):.5f} {clamp(y):.5f}" for x, y in points)
        lines.append(f"{names.index(label)} {coords}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Label Studio JSON export to a YOLO-seg batch.")
    parser.add_argument("--export", required=True, help="Label Studio export (JSON format)")
    parser.add_argument("--batch-name", required=True)
    args = parser.parse_args()

    out = REAL_LABELS_ROOT / args.batch_name
    if out.exists():
        raise SystemExit(f"{out} already exists; pick another batch name.")
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()

    tasks = json.loads(Path(args.export).read_text(encoding="utf-8"))
    document_root = Path.cwd()
    converted = skipped = regions = 0
    for task in tasks:
        lines = task_to_lines(task, DATASET_YAML_NAMES)
        if lines is None:
            skipped += 1
            continue
        src = image_path_from_url(task["data"]["image"], document_root)
        shutil.copy2(src, out / "images" / src.name)
        (out / "labels" / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        converted += 1
        regions += len(lines)

    root = str(out.resolve()).replace("\\", "/")
    (out / "dataset.yaml").write_text(
        f"path: {root}\ntrain: images\nval: images\nnc: {len(DATASET_YAML_NAMES)}\n"
        f"names: {DATASET_YAML_NAMES}\n"
    )
    print(f"[Convert] {converted} reviewed photos ({regions} regions) -> {out}; skipped {skipped} unreviewed")


if __name__ == "__main__":
    main()
