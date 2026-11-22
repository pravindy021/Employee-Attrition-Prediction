"""Streamlit AppTest coverage for dashboard page imports and startup."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_app_starts_without_exceptions() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py")
    app.run(timeout=30)
    assert not app.exception


def test_dashboard_navigation_elements_exist() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py")
    app.run(timeout=30)
    assert len(app.radio) >= 1
    assert len(app.sidebar) >= 1
