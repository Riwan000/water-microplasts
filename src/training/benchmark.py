"""
Step 2: Comparative Benchmark
Evaluates each trained candidate on mask mAP50-95, thin-fibre recall, and
inference latency (RTX 4060 GPU + CPU fallback), then writes
models/benchmark_report.json.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ultralytics import YOLO

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


def main() -> None:
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
    champion = _pick_champion(results)
    print(f"[Benchmark] Champion: {champion.model_name} "
          f"(mask_mAP50-95={champion.mask_map50_95:.4f}, thin_fibre_recall={champion.thin_fibre_recall:.4f})")


if __name__ == "__main__":
    main()
