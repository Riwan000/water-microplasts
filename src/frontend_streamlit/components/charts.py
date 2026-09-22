"""Concentration gauge + morphotype/size distribution charts."""

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_placeholder() -> None:
    st.subheader("Concentration & Morphology")
    st.caption("Populated once Step 4 inference results are wired in.")


def render(tier1: dict, tier2: list[dict]) -> None:
    """Render the concentration gauge and morphology charts.

    Args:
        tier1: Dict with ``total_plastic_count`` and ``particles_per_litre``.
        tier2: Per-particle dicts as returned by ``compute_tier2()``.
    """
    st.subheader("Concentration & Morphology")

    concentration = tier1.get("particles_per_litre", 0.0)
    total_count = tier1.get("total_plastic_count", 0)

    col1, col2 = st.columns([1, 2])

    with col1:
        # Gauge range scales to the observed value — this is a relative
        # visual indicator, not a claim about any specific WHO/EFSA/GESAMP
        # numeric threshold (none is hardcoded here; the AI assistant can
        # discuss regulatory context in its own reply).
        gauge_max = max(concentration * 1.5, 10.0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=concentration,
            number={"suffix": " /L"},
            title={"text": "Particles per Litre"},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": "#048A81"},
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Total plastic particles", total_count)

    with col2:
        if not tier2:
            st.info("No confirmed plastic particles to chart yet.")
        else:
            df = pd.DataFrame(tier2)

            class_counts = Counter(df["class_name"])
            bar_df = pd.DataFrame(
                {"class_name": list(class_counts.keys()), "count": list(class_counts.values())}
            )
            bar_fig = px.bar(
                bar_df, x="class_name", y="count",
                title="Morphotype Breakdown", color="class_name",
                labels={"class_name": "Morphotype", "count": "Count"},
            )
            bar_fig.update_layout(height=280, showlegend=False, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(bar_fig, use_container_width=True)

    if tier2:
        df = pd.DataFrame(tier2)
        hist_fig = px.histogram(
            df, x="length_um", color="class_name", nbins=20,
            title="Particle Size Distribution (Length, µm)",
            labels={"length_um": "Length (µm)", "count": "Count"},
        )
        hist_fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(hist_fig, use_container_width=True)
