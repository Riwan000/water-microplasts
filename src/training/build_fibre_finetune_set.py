"""
Builds the fibre fine-tune dataset from a reviewed real-image label batch:
  - copies the reviewed real images + writes YOLO-seg fibre-only labels to
    data/real_labels/<batch_name>/
  - samples a fraction of the existing synthetic train set for rehearsal
  - writes a combined train.txt + dataset.yaml under data/finetune_fibre/

Ground truth source: JSON docs exported from the "Fibre Ground Truth" review
artifact (one per image, with model + human-added fibre instances and a
"kept"/"rejected" decision on each).
"""

import argparse
import json
import random
import shutil
from pathlib import Path

FIBRE_CLASS_ID = 0
REAL_IMAGES_SRC = Path("data/cleaned/benchmark_real/test")
SYNTH_TRAIN_IMAGES = Path("data/yolo_dataset/images/train")
SYNTH_VAL_IMAGES = Path("data/yolo_dataset/images/val")
DATASET_YAML_NAMES = ["fibre", "fragment", "pellet", "foam_film", "algae"]


def polygon_for(instance: dict) -> list[float]:
    poly = instance.get("polygon")
    if poly and len(poly) >= 3:
        return [c for point in poly for c in point]
    x1, y1, x2, y2 = instance["bbox"]
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def build_real_labels(reviewed_dir: Path, out_dir: Path) -> list[Path]:
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    written = []
    for doc_path in sorted(reviewed_dir.glob("*.json")):
        doc = json.loads(doc_path.read_text())
        kept = [i for i in doc["instances"] if i.get("decision") == "kept"]

        src_img = REAL_IMAGES_SRC / doc["file"]
        dst_img = images_out / doc["file"]
        shutil.copyfile(src_img, dst_img)

        label_path = labels_out / (doc_path.stem + ".txt")
        lines = [
            f"{FIBRE_CLASS_ID} " + " ".join(f"{c:.5f}" for c in polygon_for(inst))
            for inst in kept
        ]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        written.append(dst_img.resolve())

    return written


def sample_synthetic(fraction: float, seed: int) -> list[Path]:
    all_images = sorted(SYNTH_TRAIN_IMAGES.glob("*.jpg"))
    rng = random.Random(seed)
    n = int(len(all_images) * fraction)
    return [p.resolve() for p in rng.sample(all_images, n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fibre fine-tune dataset.")
    parser.add_argument("--reviewed-dir", required=True, help="Dir of exported per-image review JSON docs")
    parser.add_argument("--batch-name", default="fibre_batch1")
    parser.add_argument("--synth-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/finetune_fibre")
    args = parser.parse_args()

    real_out = Path("data/real_labels") / args.batch_name
    real_images = build_real_labels(Path(args.reviewed_dir), real_out)
    print(f"[Build] Wrote {len(real_images)} real fibre-labeled images to {real_out}")

    synth_images = sample_synthetic(args.synth_fraction, args.seed)
    print(f"[Build] Sampled {len(synth_images)} synthetic images ({args.synth_fraction:.0%} of train set)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train.txt"
    train_txt.write_text("\n".join(str(p) for p in real_images + synth_images) + "\n")

    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "path: .\n"
        f"train: {train_txt.resolve()}\n"
        f"val: {SYNTH_VAL_IMAGES.resolve()}\n"
        f"nc: {len(DATASET_YAML_NAMES)}\n"
        "names:\n" + "".join(f"- {n}\n" for n in DATASET_YAML_NAMES)
    )
    print(f"[Build] Combined train set: {len(real_images)} real + {len(synth_images)} synthetic = "
          f"{len(real_images) + len(synth_images)} images")
    print(f"[Build] Dataset yaml: {dataset_yaml}")


if __name__ == "__main__":
    main()
