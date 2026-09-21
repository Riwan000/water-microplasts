"""
Step 5: Streamlit Dashboard (Phase 1 Interim)
Run with: streamlit run src/frontend_streamlit/app.py

Static image upload -> FastAPI inference -> Tier 1/2 results, particle
inspector, and AI chat dock. Live video feed is deferred to the Phase 2
React rebuild — see docs/revised-microplastic-detector-plan.md.
"""

import streamlit as st

from src.frontend_streamlit.components import chat_dock, charts, inspector

st.set_page_config(page_title="Microplastic Detector", layout="wide")


def main() -> None:
    st.title("Microplastic Detector")
    st.caption("Phase 1 — static image upload. Live microscope streaming ships in Phase 2.")

    uploaded_file = st.file_uploader("Upload a filter-paper image", type=["jpg", "jpeg", "png"])
    st.sidebar.number_input("Sample volume (mL)", min_value=0.0, value=100.0)

    if uploaded_file is None:
        st.info("Upload an image to run detection.")
        return

    st.warning("Inference is not wired up yet — Step 4 (FastAPI backend) is still pending.")
    charts.render_placeholder()
    inspector.render_placeholder()
    chat_dock.render_placeholder()


if __name__ == "__main__":
    main()
