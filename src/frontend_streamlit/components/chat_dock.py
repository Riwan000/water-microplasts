"""AI chatbot dock backed by the FastAPI /assistant/chat endpoint."""

import requests
import streamlit as st

_CHAT_HISTORY_KEY = "chat_history"


def render_placeholder() -> None:
    st.subheader("Ask the Assistant")
    st.caption("Populated once Step 4's /assistant/chat endpoint is wired in.")


def render(
    particle_counts: dict,
    filtered_volume_litres: float,
    total_plastic_count: int = 0,
    particles_per_litre: float = 0.0,
    quarantine_count: int = 0,
    api_base: str = "http://localhost:8000",
) -> None:
    """Render the AI chat dock, grounded in the current sample's results.

    Args:
        particle_counts: Count of confirmed particles per class_name.
        filtered_volume_litres: Sample volume filtered, in litres.
        total_plastic_count: Tier 1 total confirmed plastic particle count.
        particles_per_litre: Tier 1 concentration.
        quarantine_count: Number of particles routed to "Unclassified Debris".
        api_base: Base URL of the FastAPI backend.
    """
    st.subheader("Ask the Assistant")

    if _CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[_CHAT_HISTORY_KEY] = []

    for message in st.session_state[_CHAT_HISTORY_KEY]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask about this sample's water safety...")
    if not question:
        return

    st.session_state[_CHAT_HISTORY_KEY].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{api_base}/assistant/chat",
                    json={
                        "question": question,
                        "particle_counts": particle_counts,
                        "filtered_volume_litres": filtered_volume_litres,
                        "total_plastic_count": total_plastic_count,
                        "particles_per_litre": particles_per_litre,
                        "quarantine_count": quarantine_count,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                reply = response.json()["reply"]
            except requests.RequestException as exc:
                reply = f"Assistant unavailable right now ({exc}). Try again shortly."
        st.write(reply)

    st.session_state[_CHAT_HISTORY_KEY].append({"role": "assistant", "content": reply})
