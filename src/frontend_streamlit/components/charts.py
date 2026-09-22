"""Concentration gauge + morphotype/size distribution charts."""

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_placeholder() -> None:
    st.caption("Populated once Step 4 inference results are wired in.")


def render_gauge(tier1: dict, height: int = 230) -> None:
    """Render the concentration gauge + total-count metric.

    Args:
        tier1: Dict with ``total_plastic_count`` and ``particles_per_litre``.
        height: Gauge figure height in pixels, tuned to fit alongside the
            annotated image without pushing the page past one viewport.
    """
    concentration = tier1.get("particles_per_litre", 0.0)
    total_count = tier1.get("total_plastic_count", 0)

    # Gauge range scales to the observed value — this is a relative
    # visual indicator, not a claim about any specific WHO/EFSA/GESAMP
    # numeric threshold (none is hardcoded here; the AI assistant can
    # discuss regulatory context in its own reply).
    gauge_max = max(concentration * 1.5, 10.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=concentration,
        number={"suffix": " /L", "font": {"size": 36}},
        title={"text": "Particles per Litre", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, gauge_max], "tickfont": {"size": 10}},
            "bar": {"color": "#048A81"},
        },
    ))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10))

    gauge_col, metric_col = st.columns([4, 1])
    with gauge_col:
        st.plotly_chart(fig, use_container_width=True)
    with metric_col:
        st.markdown(f"<div style='margin-top:{height // 2}px'></div>", unsafe_allow_html=True)
        st.metric("Total plastic particles", total_count)


def render_presence(tier1: dict) -> None:
    """Render a plain-language Yes/No microplastics-present indicator.

    Args:
        tier1: Dict with ``total_plastic_count``.
    """
    present = tier1.get("total_plastic_count", 0) > 0
    if present:
        st.error("**Microplastics present: Yes**")
    else:
        st.success("**Microplastics present: No**")


def render_morphology(tier2: list[dict], height: int = 300) -> None:
    """Render the morphotype breakdown and particle size distribution.

    Args:
        tier2: Per-particle dicts as returned by ``compute_tier2()``.
        height: Per-chart height in pixels.
    """
    if not tier2:
        st.info("No confirmed plastic particles to chart yet.")
        return

    df = pd.DataFrame(tier2)

    col1, col2 = st.columns(2)

    with col1:
        class_counts = Counter(df["class_name"])
        bar_df = pd.DataFrame(
            {"class_name": list(class_counts.keys()), "count": list(class_counts.values())}
        )
        bar_fig = px.bar(
            bar_df, x="class_name", y="count",
            title="Morphotype Breakdown", color="class_name",
            labels={"class_name": "Morphotype", "count": "Count"},
        )
        bar_fig.update_layout(height=height, showlegend=False, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(bar_fig, use_container_width=True)

    with col2:
        hist_fig = px.histogram(
            df, x="length_um", color="class_name", nbins=20,
            title="Particle Size Distribution (Length, µm)",
            labels={"length_um": "Length (µm)", "count": "Count"},
        )
        hist_fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(hist_fig, use_container_width=True)
