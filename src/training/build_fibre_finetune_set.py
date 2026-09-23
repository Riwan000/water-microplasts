"""
Builds the fibre fine-tune dataset from a reviewed real-image label batch:
  - copies the reviewed real images + writes YOLO-seg fibre-only labels to
    data/real_labels/<batch_name>/
  - holds out a seeded fraction of the real images as a real-only test set
    (test.txt, used as the val split so checkpoint selection sees real data)
  - samples a fraction of the existing synthetic train set for rehearsal
  - writes a combined train.txt (real train images repeated --real-repeat
    times so they are not drowned out by synthetic data) + dataset.yaml
    under data/finetune_fibre/

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


def split_real(images: list[Path], holdout_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Deterministically split real images into (train, test); test is never trained on."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    shuffled = sorted(images)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * holdout_fraction))
    return sorted(shuffled[n_test:]), sorted(shuffled[:n_test])


def sample_synthetic(fraction: float, seed: int) -> list[Path]:
    all_images = sorted(SYNTH_TRAIN_IMAGES.glob("*.jpg"))
    rng = random.Random(seed)
    n = int(len(all_images) * fraction)
    return [p.resolve() for p in rng.sample(all_images, n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fibre fine-tune dataset.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reviewed-dir", help="Dir of exported per-image review JSON docs")
    source.add_argument("--from-existing", action="store_true",
                        help="Reuse the already-built data/real_labels/<batch-name>/ images + labels")
    parser.add_argument("--batch-name", default="fibre_batch1")
    parser.add_argument("--synth-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--real-repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/finetune_fibre")
    args = parser.parse_args()

    real_out = Path("data/real_labels") / args.batch_name
    if args.from_existing:
        real_images = [p.resolve() for p in sorted((real_out / "images").glob("*.jpg"))]
        if not real_images:
            raise SystemExit(f"No images found under {real_out / 'images'}")
        print(f"[Build] Reusing {len(real_images)} real fibre-labeled images from {real_out}")
    else:
        real_images = build_real_labels(Path(args.reviewed_dir), real_out)
        print(f"[Build] Wrote {len(real_images)} real fibre-labeled images to {real_out}")

    real_train, real_test = split_real(real_images, args.holdout_fraction, args.seed)
    print(f"[Build] Real split: {len(real_train)} train / {len(real_test)} held-out test")

    synth_images = sample_synthetic(args.synth_fraction, args.seed)
    print(f"[Build] Sampled {len(synth_images)} synthetic images ({args.synth_fraction:.0%} of train set)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train.txt"
    train_images = real_train * args.real_repeat + synth_images
    train_txt.write_text("\n".join(str(p) for p in train_images) + "\n")
    test_txt = out_dir / "test.txt"
    test_txt.write_text("\n".join(str(p) for p in real_test) + "\n")

    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "path: .\n"
        f"train: {train_txt.resolve()}\n"
        f"val: {test_txt.resolve()}\n"
        f"nc: {len(DATASET_YAML_NAMES)}\n"
        "names:\n" + "".join(f"- {n}\n" for n in DATASET_YAML_NAMES)
    )
    print(f"[Build] Combined train set: {len(real_train)} real x{args.real_repeat} + "
          f"{len(synth_images)} synthetic = {len(train_images)} entries; val = {len(real_test)} real held-out")
    print(f"[Build] Dataset yaml: {dataset_yaml}")


if __name__ == "__main__":
    main()
