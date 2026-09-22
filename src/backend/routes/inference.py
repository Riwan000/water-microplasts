"""
Step 4: Image-upload inference endpoint.

POST /inference/image
  - Accepts a filter-paper image + sample metadata (volume, px_to_um).
  - Runs the champion YOLO11s-seg model.
  - Passes each detection through the quarantine gate, focus filter,
    and sizing engine.
  - Returns Tier 1 / Tier 2 results, quarantine bucket, and an
    annotated image (polygon overlays) as base64 PNG.
  - Persists the session to data/sessions/{sample_id}.csv for export.

Live camera streaming is a Phase 2 addition.
"""

import base64
import io
import os
import uuid
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

from src.backend.config import load_settings
from src.backend.services.csv_exporter import build_csv_report
from src.sizing.quarantine_gate import passes_quarantine_gate
from src.sizing.sizing_engine import is_in_focus, measure_particle
from src.sizing.two_tier import compute_tier1, compute_tier2

router = APIRouter()


def _instance_contours(mask_chw: np.ndarray, orig_w: int, orig_h: int) -> list[np.ndarray]:
    """Split one instance's raw mask into its separate connected contours.

    ``result.masks.xy`` concatenates every disconnected blob of a single
    instance's mask into one point array (common for fragmented real-world
    masks split by grid lines or occlusion), which draws as bogus straight
    lines bridging unrelated blobs if treated as one closed polygon. Finding
    contours directly on the raw mask keeps each blob separate.

    Args:
        mask_chw: One instance's binary mask at the model's mask resolution
            (``result.masks.data[i]``), NOT the original image size.
        orig_w: Original image width in pixels.
        orig_h: Original image height in pixels.

    Returns:
        Contours in original-image pixel coordinates, largest area first.
    """
    resized = cv2.resize(mask_chw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)

# ── Colour map for polygon overlays (BGR) ────────────────────────────────────
_CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "fibre":     (255, 100,  50),   # blue-ish
    "fragment":  ( 50, 200, 255),   # orange
    "pellet":    ( 50, 255, 100),   # green
    "foam_film": (200,  50, 255),   # purple
    "algae":     (100, 255, 200),   # teal
}
_DEFAULT_COLOUR = (200, 200, 200)

# ── In-memory session store: sample_id → {"tier1": ..., "tier2": [...]} ──────
_SESSION_STORE: dict[str, dict] = {}

# ── Sessions directory for CSV persistence ────────────────────────────────────
_SESSIONS_DIR = Path("data/sessions")


# ── Cached model loader (loads once per process) ──────────────────────────────
@lru_cache(maxsize=1)
def _load_model() -> YOLO:
    settings = load_settings()
    weights = settings.champion_weights_path
    if not Path(weights).exists():
        raise RuntimeError(
            f"Champion weights not found at '{weights}'. "
            "Set CHAMPION_WEIGHTS_PATH in .env."
        )
    return YOLO(weights)


# ── Response schema ───────────────────────────────────────────────────────────
class ParticleDetail(BaseModel):
    particle_id: int
    class_name: str
    confidence: float
    in_focus: bool
    length_um: float
    width_um: float
    aspect_ratio: float
    surface_area_um2: float
    bbox: list[float]           # [x1, y1, x2, y2]


class Tier1Result(BaseModel):
    total_plastic_count: int
    particles_per_litre: float


class InferenceResponse(BaseModel):
    sample_id: str
    tier1: Tier1Result
    tier2: list[ParticleDetail]
    quarantine: list[ParticleDetail]
    annotated_image_b64: str    # base64 PNG with polygon overlays
    px_to_um: float


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("/image", response_model=InferenceResponse)
async def infer_image(
    file: UploadFile,
    filtered_volume_ml: Annotated[float, Form()] = 100.0,
    px_to_um: Annotated[float, Form()] = 0.5,
) -> InferenceResponse:
    """Run the champion segmentation model on an uploaded filter-paper image.

    Args:
        file: JPEG or PNG microscope image of a membrane filter paper.
        filtered_volume_ml: Volume of water filtered (mL). Converted to
            litres for concentration calculation.
        px_to_um: Calibration factor — micrometres per pixel. Use the
            Streamlit sidebar slider; derive from grid-line spacing.

    Returns:
        InferenceResponse with Tier 1/2 results, quarantine bucket, and
        the annotated image encoded as a base64 PNG string.
    """
    # ── 1. Decode uploaded image ──────────────────────────────────────────────
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image. Upload a valid JPEG or PNG.")

    filtered_volume_litres = filtered_volume_ml / 1000.0

    # ── 2. Run YOLO inference ─────────────────────────────────────────────────
    model = _load_model()
    results = model.predict(source=img_bgr, task="segment", verbose=False)
    result  = results[0]

    class_names: dict[int, str] = model.names
    masks = result.masks   # may be None if no detections
    boxes = result.boxes

    annotated = img_bgr.copy()
    confirmed: list[ParticleDetail] = []
    quarantine: list[ParticleDetail] = []

    mask_data = masks.data.cpu().numpy() if masks is not None else None

    if masks is not None and len(masks) > 0:
        for idx, box in enumerate(boxes):
            conf     = float(box.conf[0])
            cls_id   = int(box.cls[0])
            cls_name = class_names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)

            # ── Focus check ───────────────────────────────────────────────────
            crop = img_bgr[y1:y2, x1:x2]
            focused = is_in_focus(crop) if crop.size > 0 else False

            # ── Split mask into separate blobs (avoids bridging artifacts) ──────
            contours = _instance_contours(mask_data[idx], img_bgr.shape[1], img_bgr.shape[0])

            # ── Sizing (largest connected blob only) ─────────────────────────────
            pts = [(float(p[0][0]), float(p[0][1])) for p in contours[0]] if contours else []
            if len(pts) >= 3:
                dims = measure_particle(pts, px_to_um)
            else:
                # degenerate mask — skip sizing, use bbox fallback
                from src.sizing.sizing_engine import ParticleDimensions
                w_px = float(x2 - x1)
                h_px = float(y2 - y1)
                long_px  = max(w_px, h_px)
                short_px = min(w_px, h_px)
                dims = ParticleDimensions(
                    length_um=round(long_px * px_to_um, 3),
                    width_um=round(short_px * px_to_um, 3),
                    aspect_ratio=round(long_px / short_px, 3) if short_px > 0 else 1.0,
                    surface_area_um2=round(long_px * short_px * px_to_um ** 2, 3),
                )

            detail = ParticleDetail(
                particle_id=idx,
                class_name=cls_name,
                confidence=round(conf, 4),
                in_focus=focused,
                length_um=dims.length_um,
                width_um=dims.width_um,
                aspect_ratio=dims.aspect_ratio,
                surface_area_um2=dims.surface_area_um2,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
            )

            # ── Quarantine gate ───────────────────────────────────────────────
            if passes_quarantine_gate(conf):
                confirmed.append(detail)
            else:
                quarantine.append(detail)

            # ── Draw polygon overlay (each blob drawn separately) ────────────────
            colour = _CLASS_COLOURS.get(cls_name, _DEFAULT_COLOUR)
            if contours:
                overlay = annotated.copy()
                cv2.fillPoly(overlay, contours, colour)
                annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
                cv2.polylines(annotated, contours, isClosed=True, color=colour, thickness=2)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            # Label: class + confidence
            label = f"[Q]{cls_name}" if not passes_quarantine_gate(conf) else cls_name
            cv2.putText(
                annotated, f"{label} {conf:.2f}",
                (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
            )

    # ── 3. Tier 1 / Tier 2 ───────────────────────────────────────────────────
    all_dicts = [
        {**d.model_dump(), "bbox": d.bbox}
        for d in confirmed
    ]
    tier1_summary = compute_tier1(all_dicts, filtered_volume_litres)
    tier2_rows    = compute_tier2(all_dicts)

    # ── 4. Encode annotated image ─────────────────────────────────────────────
    _, png_buf = cv2.imencode(".png", annotated)
    annotated_b64 = base64.b64encode(png_buf.tobytes()).decode("utf-8")

    # ── 5. Persist session to CSV ─────────────────────────────────────────────
    sample_id = str(uuid.uuid4())[:8].upper()
    _SESSION_STORE[sample_id] = {
        "tier1": tier1_summary.__dict__,
        "tier2": tier2_rows,
        "annotated_image_b64": annotated_b64,
    }

    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    csv_bytes = build_csv_report(sample_id, tier2_rows)
    (_SESSIONS_DIR / f"{sample_id}.csv").write_bytes(csv_bytes)

    # ── 6. Build and return response ──────────────────────────────────────────
    return InferenceResponse(
        sample_id=sample_id,
        tier1=Tier1Result(**tier1_summary.__dict__),
        tier2=confirmed,
        quarantine=quarantine,
        annotated_image_b64=annotated_b64,
        px_to_um=px_to_um,
    )


def get_session(sample_id: str) -> dict:
    """Retrieve a stored session by ID (used by export routes)."""
    if sample_id not in _SESSION_STORE:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found. Run inference first.")
    return _SESSION_STORE[sample_id]

