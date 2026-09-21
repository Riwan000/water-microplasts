# models/

Step 2 output — not committed except this file and `benchmark_report.json`.

- `yolov8n-seg/`, `yolo11n-seg/`, `yolo11s-seg/` — per-architecture training run weights,
  created by `src/training/train.py`. Weight files (`*.pt`, `*.pth`, `*.onnx`) are gitignored.
- `benchmark_report.json` — written by `src/training/benchmark.py`; records the comparative
  mask mAP50-95, thin-fibre recall, and inference-latency results used to pick the champion
  model referenced by `CHAMPION_WEIGHTS_PATH` in `src/backend/config.py`.
