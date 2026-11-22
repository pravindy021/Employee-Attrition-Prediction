"""Project-relative paths and artifact discovery helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config.settings import ProjectSettings


@st.cache_resource
def project_root() -> Path:
    """Return the project root without relying on an absolute OS path."""
    return ProjectSettings.load().root


def project_path(*parts: str) -> Path:
    """Resolve a path beneath the project root."""
    return project_root().joinpath(*parts)
