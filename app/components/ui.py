"""Reusable Streamlit UI components."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def page_title(title: str, description: str) -> None:
    """Render a consistent page heading."""
    st.title(title)
    st.caption(description)
    st.divider()


def metric_row(metrics: list[tuple[str, str]]) -> None:
    """Render a row of Streamlit metric cards."""
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)


def risk_badge(category: str) -> None:
    """Render a clear risk category label."""
    colors = {
        "LOW": ("green", "Low"),
        "MEDIUM": ("orange", "Medium"),
        "HIGH": ("red", "High"),
    }
    color, label = colors.get(category, ("gray", category.title()))
    st.markdown(
        f"<span style='background:{color};color:white;padding:0.35rem "
        f"0.7rem;border-radius:0.5rem;font-weight:700'>{label} Risk</span>",
        unsafe_allow_html=True,
    )


def safe_plotly_bar(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
) -> None:
    """Render an empty-safe horizontal or vertical Plotly bar chart."""
    if frame.empty:
        st.info("No data is available for this chart.")
        return
    figure = px.bar(
        frame,
        x=x,
        y=y,
        color=color,
        title=title,
        template="plotly_white",
    )
    figure.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(figure, width="stretch")
