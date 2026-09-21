"""Concentration gauge + morphotype/size distribution charts."""

import streamlit as st


def render_placeholder() -> None:
    st.subheader("Concentration & Morphology")
    st.caption("Populated once Step 4 inference results are wired in.")


def render(tier1: dict, tier2: list[dict]) -> None:
    raise NotImplementedError("Wire up Plotly/Altair gauge and histograms here (Step 5).")
