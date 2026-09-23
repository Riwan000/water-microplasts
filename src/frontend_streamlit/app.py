"""
Step 5: Streamlit Dashboard (Phase 1 Interim)
Run with: streamlit run src/frontend_streamlit/app.py

Static image upload -> FastAPI inference -> Tier 1/2 results, particle
inspector, and AI chat dock. Live video feed is deferred to the Phase 2
React rebuild — see docs/revised-microplastic-detector-plan.md.
"""

import base64
import io
import os
import sys
from pathlib import Path
from collections import Counter

# Ensure project root is in sys.path so 'src' imports resolve reliably
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import requests
import streamlit as st
from PIL import Image, ImageOps

from src.frontend_streamlit.components import chat_dock, charts, inspector

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
_ANNOTATED_IMAGE_DISPLAY_SIZE = (560, 360)  # (w, h) box the annotated image is fit to, scaled up or down as needed

st.set_page_config(page_title="Microplastic Detector", layout="wide")


def _fit_to_box(image: Image.Image, box_size: tuple[int, int]) -> Image.Image:
    """Scale image to fill box_size (up or down) while preserving aspect ratio."""
    return ImageOps.contain(image, box_size, Image.Resampling.LANCZOS)


@st.cache_data(ttl=30)
def _assistant_enabled() -> bool:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        return resp.json().get("assistant_enabled") == "true"
    except requests.RequestException:
        return False


def _run_inference(file_bytes: bytes, filename: str, filtered_volume_ml: float, px_to_um: float) -> dict:
    response = requests.post(
        f"{API_BASE}/inference/image",
        files={"file": (filename, file_bytes, "application/octet-stream")},
        data={"filtered_volume_ml": filtered_volume_ml, "px_to_um": px_to_um},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _render_export_buttons(sample_id: str) -> None:
    st.sidebar.subheader("Export")
    col_pdf, col_csv = st.sidebar.columns(2)

    with col_pdf:
        if st.button("Prepare PDF"):
            try:
                resp = requests.get(f"{API_BASE}/export/pdf/{sample_id}", timeout=30)
                resp.raise_for_status()
                st.session_state["_pdf_bytes"] = resp.content
            except requests.RequestException as exc:
                st.error(f"Could not generate PDF: {exc}")
        if st.session_state.get("_pdf_bytes"):
            st.download_button(
                "Download PDF",
                data=st.session_state["_pdf_bytes"],
                file_name=f"microplastic_{sample_id}.pdf",
                mime="application/pdf",
            )

    with col_csv:
        if st.button("Prepare CSV"):
            try:
                resp = requests.get(f"{API_BASE}/export/csv/{sample_id}", timeout=30)
                resp.raise_for_status()
                st.session_state["_csv_bytes"] = resp.content
            except requests.RequestException as exc:
                st.error(f"Could not generate CSV: {exc}")
        if st.session_state.get("_csv_bytes"):
            st.download_button(
                "Download CSV",
                data=st.session_state["_csv_bytes"],
                file_name=f"microplastic_{sample_id}.csv",
                mime="text/csv",
            )


def main() -> None:
    st.sidebar.title("Microplastic Detector")
    st.sidebar.caption("Phase 1 — static image upload. Live streaming ships in Phase 2.")

    filtered_volume_ml = st.sidebar.number_input("Sample volume (mL)", min_value=0.0, value=100.0)
    px_to_um = st.sidebar.number_input(
        "Calibration (µm / pixel)", min_value=0.0, value=0.5, step=0.01,
        help="Derive from grid-line spacing on the filter paper image.",
    )
    uploaded_file = st.sidebar.file_uploader("Upload a filter-paper image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image from the sidebar to run detection.")
        return

    file_bytes = uploaded_file.getvalue()
    cache_key = (uploaded_file.name, len(file_bytes), filtered_volume_ml, px_to_um)

    if st.session_state.get("_cache_key") != cache_key:
        with st.spinner("Running detection..."):
            try:
                result = _run_inference(file_bytes, uploaded_file.name, filtered_volume_ml, px_to_um)
            except requests.RequestException as exc:
                st.error(
                    "Could not reach the inference backend. Is it running? "
                    f"(`uvicorn src.backend.main:app`) — {exc}"
                )
                return
        st.session_state["_cache_key"] = cache_key
        st.session_state["_result"] = result
        st.session_state["_pdf_bytes"] = None
        st.session_state["_csv_bytes"] = None
        st.session_state["chat_history"] = []  # fresh sample, fresh conversation

    result = st.session_state["_result"]
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    annotated_bytes = base64.b64decode(result["annotated_image_b64"])
    annotated_image = Image.open(io.BytesIO(annotated_bytes)).convert("RGB")

    tier1 = result["tier1"]
    tier2 = result["tier2"]
    quarantine = result["quarantine"]

    top_left, top_right = st.columns([1, 1])
    with top_left:
        st.image(_fit_to_box(annotated_image, _ANNOTATED_IMAGE_DISPLAY_SIZE), caption="Annotated detections")
    with top_right:
        charts.render_gauge(tier1)
        charts.render_presence(tier1)

    assistant_on = _assistant_enabled()
    tab_labels = ["Morphology & Size", "Particle Inspector"]
    if assistant_on:
        tab_labels.append("Ask the Assistant")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        charts.render_morphology(tier2)
    with tabs[1]:
        inspector.render(image, tier2, quarantine)

    # AI assistant needs OPENROUTER_API_KEY (see GitHub issue: re-enable AI
    # assistant) — the backend doesn't mount /assistant without one, so the
    # dock is hidden until /health reports it's actually available.
    if assistant_on:
        with tabs[2]:
            particle_counts = dict(Counter(p["class_name"] for p in tier2))
            chat_dock.render(
                particle_counts=particle_counts,
                filtered_volume_litres=filtered_volume_ml / 1000.0,
                total_plastic_count=tier1.get("total_plastic_count", 0),
                particles_per_litre=tier1.get("particles_per_litre", 0.0),
                quarantine_count=len(quarantine),
                api_base=API_BASE,
            )

    _render_export_buttons(result["sample_id"])


if __name__ == "__main__":
    main()
