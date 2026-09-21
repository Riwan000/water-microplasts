# Trial Run — 2026-09-22

Archived snapshot of Step 2 (Multi-Model Training & Benchmarking), preserved because
`models/<arch>/` gets overwritten by `exist_ok=True` on the next training run.

- **Dataset**: `data/yolo_dataset` — 2,500 composited scenes (2,000 train / 500 val + 1,000
  negative patches), class-balanced via inverse-frequency weighted sampling in
  `src/dataset/compose_scenes.py`. This is a reduced trial dataset, not the full 20,000-scene set.
- **Training**: 20 epochs per candidate, batch=8, workers=2, imgsz=640, RTX 4060 Laptop GPU
  (conservative settings due to limited system RAM on this machine).
- **Champion**: `yolo11s-seg` — highest mask mAP50-95 (0.579) and highest thin-fibre recall
  (0.651) of the 3 candidates; see `benchmark_report.json` for full numbers.

Treat these results as directional (pipeline validation), not production-final. A production
training run should use the full dataset and more epochs before deployment.
