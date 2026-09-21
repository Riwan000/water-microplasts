"""AI chatbot dock backed by the FastAPI /assistant/chat endpoint."""

import streamlit as st


def render_placeholder() -> None:
    st.subheader("Ask the Assistant")
    st.caption("Populated once Step 4's /assistant/chat endpoint is wired in.")


def render(particle_counts: dict, filtered_volume_litres: float) -> None:
    raise NotImplementedError("Wire up the assistant chat dock here (Step 5).")
