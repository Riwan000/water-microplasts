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
from collections import Counter

import requests
import streamlit as st
from PIL import Image

from src.frontend_streamlit.components import chat_dock, charts, inspector

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Microplastic Detector", layout="wide")


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
    st.subheader("Export")
    col_pdf, col_csv = st.columns(2)

    with col_pdf:
        if st.button("Prepare PDF report"):
            try:
                resp = requests.get(f"{API_BASE}/export/pdf/{sample_id}", timeout=30)
                resp.raise_for_status()
                st.session_state["_pdf_bytes"] = resp.content
            except requests.RequestException as exc:
                st.error(f"Could not generate PDF: {exc}")
        if st.session_state.get("_pdf_bytes"):
            st.download_button(
                "Download PDF report",
                data=st.session_state["_pdf_bytes"],
                file_name=f"microplastic_{sample_id}.pdf",
                mime="application/pdf",
            )

    with col_csv:
        if st.button("Prepare CSV report"):
            try:
                resp = requests.get(f"{API_BASE}/export/csv/{sample_id}", timeout=30)
                resp.raise_for_status()
                st.session_state["_csv_bytes"] = resp.content
            except requests.RequestException as exc:
                st.error(f"Could not generate CSV: {exc}")
        if st.session_state.get("_csv_bytes"):
            st.download_button(
                "Download CSV report",
                data=st.session_state["_csv_bytes"],
                file_name=f"microplastic_{sample_id}.csv",
                mime="text/csv",
            )


def main() -> None:
    st.title("Microplastic Detector")
    st.caption("Phase 1 — static image upload. Live microscope streaming ships in Phase 2.")

    filtered_volume_ml = st.sidebar.number_input("Sample volume (mL)", min_value=0.0, value=100.0)
    px_to_um = st.sidebar.number_input(
        "Calibration (µm / pixel)", min_value=0.0, value=0.5, step=0.01,
        help="Derive from grid-line spacing on the filter paper image.",
    )

    uploaded_file = st.file_uploader("Upload a filter-paper image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to run detection.")
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
    st.image(annotated_bytes, caption="Annotated detections", use_container_width=True)

    tier1 = result["tier1"]
    tier2 = result["tier2"]
    quarantine = result["quarantine"]

    charts.render(tier1, tier2)
    inspector.render(image, tier2, quarantine)

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
