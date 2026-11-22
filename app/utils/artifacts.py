"""Cached loading for dashboard data, models, and reports."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from app.utils.paths import project_path
from src.utils.logging_config import configure_logging
from src.features.preprocessing import (
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
    resolve_preprocessing_spec,
)


@st.cache_data
def load_employee_data() -> pd.DataFrame:
    """Load the raw project dataset for historical analytics."""
    candidates = [
        project_path("data", "raw", "Palo Alto Networks.csv"),
        project_path("data", "Palo Alto Networks.csv"),
    ]
    for path in candidates:
        if path.exists():
            configure_logging().info("Loaded employee analytics dataset.")
            return pd.read_csv(path)
    raise FileNotFoundError("No project employee CSV artifact was found.")


@st.cache_resource
def load_calibrated_model():
    """Load the saved Phase 6 calibrated model."""
    path = project_path("models", "phase6_attrition_risk_model.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Missing calibrated model artifact: {path}")
    return joblib.load(path)


@st.cache_resource
def load_tuned_logistic_model():
    """Load the saved underlying tuned logistic pipeline."""
    path = project_path("models", "phase5", "logistic_regression_tuned.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned logistic model artifact: {path}")
    return joblib.load(path)


@st.cache_resource
def load_risk_config() -> dict:
    """Load the Phase 6 threshold and risk-category configuration."""
    path = project_path("data", "processed", "phase6_risk_scoring_config.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Missing risk configuration artifact: {path}")
    return joblib.load(path)


@st.cache_data
def load_csv_report(relative_path: str) -> pd.DataFrame:
    """Load an existing CSV report or raise a clear missing-artifact error."""
    path = project_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing report artifact: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_json_report(relative_path: str) -> dict:
    """Load an existing JSON report."""
    import json

    path = project_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing report artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def model_feature_columns() -> tuple[list[str], list[str]]:
    """Return operational model and sensitive audit columns."""
    frame = load_employee_data()
    spec = resolve_preprocessing_spec(frame)
    model_columns = [
        column
        for column in frame.columns
        if column not in spec.excluded_columns
    ]
    return model_columns, list(SENSITIVE_COLUMNS)
