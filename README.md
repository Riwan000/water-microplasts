# water-microplasts

A low-cost benchtop tool that counts, classifies, and sizes suspected microplastic
particles in a water sample from a single microscope photo — no spectroscopy
hardware, no dedicated GPU, no trained analyst required.

You take a photo of a filtered water sample under a cheap USB microscope, upload
it, and get back: how many plastic particles it contains, what shape each one is
(fibre, fragment, pellet, film/foam), how big each one is in micrometres, and a
particles-per-litre concentration — plus an annotated image and an optional AI
assistant to explain what the numbers mean.

It is a **screening instrument**, not an analytical one. It answers "is this water
contaminated, and roughly how much" — not "which polymer is this." See
[`docs/microplastic-counter-prd.md`](docs/microplastic-counter-prd.md) for the
full product spec and [`docs/revised-microplastic-detector-plan.md`](docs/revised-microplastic-detector-plan.md)
for the technical plan.

---

## How it works

1. **Capture** — photograph a water sample on a thin, fixed-depth chamber
   (gridded membrane filter paper) under an IBELL USB microscope. The grid lines
   double as an optical ruler for calibration.
2. **Detect** — a YOLO11s instance-segmentation model finds every particle in the
   frame and outlines it with a pixel-accurate polygon.
3. **Filter** — each detection passes through:
   - a **focus gate** (Laplacian sharpness) that discards blurry/out-of-plane crops,
   - a **quarantine gate** that routes low-confidence detections to an
     "Unclassified Debris" bucket instead of the plastic count.
4. **Size & classify** — surviving detections get a morphotype label (fibre,
   fragment, pellet, foam/film — algae is detected but excluded from the plastic
   count) plus length, width, aspect ratio, and surface area in µm, computed from
   the polygon and a user-supplied px-to-µm calibration factor.
5. **Report** — results are aggregated into two tiers:
   - **Tier 1** — total plastic count and particles/L (`count / filtered volume`).
   - **Tier 2** — full per-particle breakdown for the table/CSV/PDF export.
6. **Explain (optional)** — an AI assistant, grounded in the actual counts for
   that sample, answers plain-language questions like "is this water safe to
   drink?".

---

## Architecture

```
┌─────────────────────┐        HTTP        ┌──────────────────────────┐
│  Streamlit frontend   │ ─────────────────▶ │   FastAPI backend          │
│  (src/frontend_streamlit) │                │   (src/backend)             │
│                        │ ◀───────────────── │                            │
│  - image upload        │   JSON + base64   │  POST /inference/image     │
│  - Tier 1/2 dashboard   │   PNG overlay     │  GET  /export/{pdf,csv}    │
│  - particle inspector   │                   │  POST /assistant/chat      │
│  - AI chat dock         │                   │  GET  /health               │
└─────────────────────┘                     └──────────────────────────┘
                                                          │
                                              ┌────────────┴────────────┐
                                              │  Ultralytics YOLO11s-seg  │
                                              │  models/champion.pt       │
                                              └────────────┬────────────┘
                                                            │
                                              ┌────────────┴────────────┐
                                              │  src/sizing               │
                                              │  quarantine_gate.py       │
                                              │  sizing_engine.py         │
                                              │  two_tier.py              │
                                              └───────────────────────────┘
```

| Layer | Path | Role |
|---|---|---|
| Frontend | `src/frontend_streamlit/` | Streamlit dashboard — upload, results, particle inspector, AI chat dock, PDF/CSV export buttons |
| Backend API | `src/backend/` | FastAPI app: inference, export, assistant routes; CORS + settings |
| Sizing logic | `src/sizing/` | Quarantine gate, focus/sharpness gate, µm measurement, Tier 1/2 aggregation |
| Dataset pipeline | `src/dataset/` | Cleans raw filter-paper/particle images and composites synthetic training scenes |
| Training pipeline | `src/training/` | Trains and benchmarks candidate YOLO-seg architectures, fine-tunes on real fibres |

---

## Requirements

- Python **3.10+**
- `pip` and a virtual environment tool (`venv`, `conda`, etc.)
- No GPU required (CPU inference works; GPU speeds up training/benchmarking)
- A trained model checkpoint at `models/champion.pt` (see [Models](#models) below)

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd water-microplasts

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 3b. (NVIDIA GPU, optional) swap the CPU-only torch that ultralytics pulls in
#     for the CUDA build — ~4x faster inference. cu126 needs driver >= 560.
pip install --force-reinstall --no-deps torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 4. Configure environment variables
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```

Edit `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `CHAMPION_WEIGHTS_PATH` | `models/champion.pt` | Path to the trained segmentation model used for inference |
| `CORS_ORIGINS` | `http://localhost:8501` | Origins allowed to call the API (comma-separated) |
| `OPENROUTER_API_KEY` | *(empty)* | API key for the AI assistant. **Optional** — leave blank and the `/assistant` endpoint is simply not mounted; everything else works fine without it |
| `OPENROUTER_MODEL` | `google/gemini-2.0-flash-exp:free` | Model used for assistant chat, via [OpenRouter](https://openrouter.ai/) |

---

## Running it

Start the backend and frontend in two terminals:

```bash
# Terminal 1 — API (http://localhost:8000)
uvicorn src.backend.main:app --reload

# Terminal 2 — dashboard (http://localhost:8501)
streamlit run src/frontend_streamlit/app.py
```

Then open **http://localhost:8501**, upload a filter-paper image, set the
filtered volume and px-to-µm calibration in the sidebar, and run inference.

If the frontend and backend run on different hosts/ports, point the dashboard at
the API with:

```bash
API_BASE_URL=http://your-api-host:8000 streamlit run src/frontend_streamlit/app.py
```

---

## Models

The `champion` model referenced by `CHAMPION_WEIGHTS_PATH` is whichever
architecture won the comparative benchmark in `models/benchmark_report.json`.
Model weights are **not committed** (see `.gitignore`) — you need to either
train your own or drop a compatible `.pt` file at `models/champion.pt`.

Candidates evaluated so far (`mask mAP50-95` / thin-fibre recall / CPU latency):

| Model | mAP50-95 | Thin-fibre recall | CPU latency |
|---|---|---|---|
| yolov8n-seg | 0.562 | 0.639 | 96 ms |
| yolo11n-seg | 0.561 | 0.646 | 88 ms |
| **yolo11s-seg** | **0.684** | **0.762** | 179 ms |

`yolo11s-seg` currently wins on both accuracy and fibre recall and is the
champion architecture. See `models/README.md` for details and
`src/training/finetune_fibre.py` for the real-fibre fine-tune experiment on top
of it.

### Training your own

```bash
# 1. Build the synthetic training dataset (see docs/ML_DATASET_PLAN.md)
python -m src.dataset.clean_backgrounds
python -m src.dataset.clean_case1
python -m src.dataset.clean_particles
python -m src.dataset.extract_masks
python -m src.dataset.prepare_case0
python -m src.dataset.compose_scenes

# 2. Train and benchmark candidate architectures
python -m src.training.train
python -m src.training.benchmark   # writes models/benchmark_report.json

# 3. Optional: fine-tune the champion on real (non-synthetic) fibre images
python -m src.training.build_fibre_finetune_set
python -m src.training.finetune_fibre
```

---

## Testing

```bash
pytest
```

Tests live under `tests/`, mirroring `src/` (`tests/sizing/`, `tests/backend/`,
`tests/dataset/`, `tests/training/`). Config is in `pytest.ini`.

---

## Project structure

```
src/
  backend/               FastAPI app
    main.py              App entrypoint, router wiring, /health
    config.py             Settings loaded from .env
    routes/
      inference.py        POST /inference/image — the core detect→size→report pipeline
      export.py            GET /export/pdf|csv/{sample_id}
      assistant.py          POST /assistant/chat — AI Q&A grounded in sample results
    services/
      csv_exporter.py       Tier 2 CSV report builder
      pdf_exporter.py        Tier 1/2 PDF lab report builder
      openrouter_client.py    OpenRouter API wrapper

  frontend_streamlit/     Streamlit dashboard (Phase 1 interim UI)
    app.py                 Upload, calibration sidebar, results, export buttons
    components/
      charts.py             Concentration gauge + morphotype/size distribution charts
      inspector.py           Clickable particle gallery + "Unclassified Debris" review tab
      chat_dock.py            AI chatbot dock

  sizing/                 Pure-logic particle measurement & aggregation
    quarantine_gate.py      Confidence threshold gate
    sizing_engine.py          Focus filter + µm measurement from polygon
    two_tier.py               Tier 1 (count/concentration) + Tier 2 (per-particle) aggregation

  dataset/                 Raw-data cleaning + synthetic scene generation
  training/               Model training, benchmarking, fine-tuning

tests/                    pytest suite, mirrors src/
docs/                     Product spec, ML dataset plan, technical roadmap
models/                   Trained weights (gitignored) + benchmark_report.json
data/                     Raw/cleaned/composited datasets (gitignored)
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check; reports whether the AI assistant is enabled |
| `POST` | `/inference/image` | Upload an image (`file`) + `filtered_volume_ml`, `px_to_um` form fields → Tier 1/2 results, quarantine list, annotated PNG (base64) |
| `GET` | `/export/pdf/{sample_id}` | Download a PDF lab report for a completed inference session |
| `GET` | `/export/csv/{sample_id}` | Download a CSV audit trail for a completed inference session |
| `POST` | `/assistant/chat` | Ask a water-safety question; answer is grounded in that sample's counts (only mounted if `OPENROUTER_API_KEY` is set) |

Interactive docs are available at `http://localhost:8000/docs` while the backend
is running.

---

## Status & roadmap

This is **Phase 1**: static image upload, Vision AI, sizing, and a Streamlit
dashboard. Live camera streaming and a React frontend rebuild are Phase 2 — see
[`docs/revised-microplastic-detector-plan.md`](docs/revised-microplastic-detector-plan.md).
pH/TDS IoT hardware integration is deferred further still.

**Explicitly out of scope:** polymer identification (PE/PP/PET/PS) — that needs
FTIR/Raman spectroscopy hardware this project deliberately excludes.
