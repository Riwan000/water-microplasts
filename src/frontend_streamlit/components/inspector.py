"""Clickable particle inspector gallery + 'Unclassified Debris' review tab."""

import streamlit as st


def render_placeholder() -> None:
    st.subheader("Particle Inspector")
    st.caption("Populated once Step 4 inference results are wired in.")


def render(detections: list[dict]) -> None:
    raise NotImplementedError("Wire up the particle inspector gallery here (Step 5).")
