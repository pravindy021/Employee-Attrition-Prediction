"""Tests for Phase 12 deployment preparation."""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "app.py",
        "requirements.txt",
        "runtime.txt",
        ".streamlit/config.toml",
        "configs/project.json",
        "data/raw/Palo Alto Networks.csv",
        "models/phase6_attrition_risk_model.joblib",
        "models/phase5/logistic_regression_tuned.joblib",
        "reports/DEMOGRAPHIC_METRICS.csv",
    ],
)
def test_deployment_required_files_exist(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).is_file()


def test_runtime_is_python_312() -> None:
    runtime = (PROJECT_ROOT / "runtime.txt").read_text(encoding="utf-8").strip()
    assert runtime.startswith("python-3.12")


def test_project_paths_are_relative() -> None:
    app_text = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "D:\\\\" not in app_text
    assert "C:\\\\" not in app_text


def test_requirements_do_not_contain_development_only_tools() -> None:
    requirements = (
        PROJECT_ROOT / "requirements.txt"
    ).read_text(encoding="utf-8").lower()
    assert "jupyter" not in requirements
    assert "ipykernel" not in requirements
