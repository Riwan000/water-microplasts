"""
Step 2: Comparative Benchmark
Evaluates each trained candidate on mask mAP50-95, thin-fibre recall, and
inference latency (RTX 4060 GPU + CPU fallback), then writes
models/benchmark_report.json.
"""

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ultralytics import YOLO

from src.backend.config import load_settings
from src.training.train import CANDIDATE_MODELS, MODELS_DIR

FIBRE_CLASS_NAME = "fibre"
LATENCY_WARMUP_RUNS = 3
LATENCY_TIMED_RUNS = 20


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    mask_map50_95: float
    thin_fibre_recall: float
    latency_ms_gpu: float
    latency_ms_cpu: float


def _measure_latency_ms(weights_path: str, sample_image: str, device: str) -> float:
    model = YOLO(weights_path)
    for _ in range(LATENCY_WARMUP_RUNS):
        model.predict(sample_image, device=device, verbose=False)

    start = time.perf_counter()
    for _ in range(LATENCY_TIMED_RUNS):
        model.predict(sample_image, device=device, verbose=False)
    elapsed = time.perf_counter() - start
    return (elapsed / LATENCY_TIMED_RUNS) * 1000.0


def benchmark_model(model_name: str, weights_path: str, dataset_yaml: str, sample_image: str) -> BenchmarkResult:
    """Run the comparative benchmark for one trained model."""
    model = YOLO(weights_path)
    metrics = model.val(data=dataset_yaml, device="0", verbose=False)

    fibre_row = next((row for row in metrics.summary() if row["Class"] == FIBRE_CLASS_NAME), None)
    thin_fibre_recall = float(fibre_row["Mask-R"]) if fibre_row else 0.0

    return BenchmarkResult(
        model_name=model_name,
        mask_map50_95=float(metrics.seg.map),
        thin_fibre_recall=thin_fibre_recall,
        latency_ms_gpu=_measure_latency_ms(weights_path, sample_image, device="0"),
        latency_ms_cpu=_measure_latency_ms(weights_path, sample_image, device="cpu"),
    )


def write_report(results: list[BenchmarkResult], out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def _pick_champion(results: list[BenchmarkResult]) -> BenchmarkResult:
    return max(results, key=lambda r: (r.mask_map50_95, r.thin_fibre_recall))


def load_report(report_path: str) -> list[BenchmarkResult]:
    with open(report_path) as f:
        return [BenchmarkResult(**row) for row in json.load(f)]


def promote_champion(champion: BenchmarkResult, models_dir: str, champion_path: str) -> Path:
    """Copy the champion's best.pt to the path the backend loads (CHAMPION_WEIGHTS_PATH)."""
    source = Path(models_dir) / champion.model_name / "weights" / "best.pt"
    if not source.exists():
        raise FileNotFoundError(f"Champion weights missing at {source}; cannot promote {champion.model_name}.")
    destination = Path(champion_path)
    if destination.exists() and destination.resolve() == source.resolve():
        return destination  # CHAMPION_WEIGHTS_PATH already points at this run's best.pt
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _promote(results: list[BenchmarkResult]) -> None:
    champion = _pick_champion(results)
    destination = promote_champion(champion, MODELS_DIR, load_settings().champion_weights_path)
    print(f"[Benchmark] Champion: {champion.model_name} "
          f"(mask_mAP50-95={champion.mask_map50_95:.4f}, thin_fibre_recall={champion.thin_fibre_recall:.4f}) "
          f"-> promoted to {destination}")


def run_benchmark() -> list[BenchmarkResult]:
    dataset_yaml = "data/yolo_dataset/dataset.yaml"
    val_images_dir = Path("data/yolo_dataset/images/val")
    sample_image = str(next(val_images_dir.glob("*.jpg")))

    results = []
    for model_name in CANDIDATE_MODELS:
        weights_path = Path(MODELS_DIR) / model_name / "weights" / "best.pt"
        if not weights_path.exists():
            print(f"[Benchmark] Skipping {model_name}: no weights at {weights_path}")
            continue
        print(f"[Benchmark] Evaluating {model_name}...")
        results.append(benchmark_model(model_name, str(weights_path), dataset_yaml, sample_image))

    if not results:
        raise SystemExit("No trained candidate weights found. Run train.py for each candidate first.")

    write_report(results, str(Path(MODELS_DIR) / "benchmark_report.json"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidates and promote the champion.")
    parser.add_argument("--promote-only", action="store_true",
                        help="Skip evaluation; promote the champion from the existing benchmark_report.json.")
    args = parser.parse_args()

    if args.promote_only:
        _promote(load_report(str(Path(MODELS_DIR) / "benchmark_report.json")))
    else:
        _promote(run_benchmark())


if __name__ == "__main__":
    main()
