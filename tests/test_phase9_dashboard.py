"""Tests for Phase 9 dashboard integration."""

from __future__ import annotations

import pandas as pd
import pytest

from app.utils.artifacts import (
    load_employee_data,
    load_risk_config,
    model_feature_columns,
)
from app.utils.prediction import (
    build_profile_frame,
    predict_profile,
)
from src.models.risk_scoring import classify_risk


def test_dashboard_modules_import() -> None:
    from app.components.ui import page_title
    from app.utils.paths import project_root

    assert callable(page_title)
    assert project_root().name == "Employee-Attrition-Prediction"


def test_existing_employee_and_risk_artifacts_load() -> None:
    frame = load_employee_data()
    config = load_risk_config()
    model_columns, sensitive = model_feature_columns()

    assert len(frame) == 1470
    assert "Attrition" in frame.columns
    assert config["threshold"] == pytest.approx(0.21)
    assert "Age" in sensitive
    assert "Gender" in sensitive
    assert "MaritalStatus" in sensitive
    assert not set(sensitive).intersection(set(model_columns))


def test_risk_category_assignment() -> None:
    categories = classify_risk([0.29, 0.30, 0.60, 0.61])
    assert categories.tolist() == ["LOW", "MEDIUM", "MEDIUM", "HIGH"]


def test_profile_builder_handles_missing_optional_values() -> None:
    frame = pd.DataFrame(
        {
            "Department": ["Sales", "R&D"],
            "OverTime": ["No", "Yes"],
            "MonthlyIncome": [3000, 5000],
        }
    )
    profile = build_profile_frame(
        frame,
        ["Department", "OverTime", "MonthlyIncome"],
        {"Department": "Sales"},
    )
    assert len(profile) == 1
    assert profile["Department"].iloc[0] == "Sales"
    assert "OverTime" in profile.columns
    assert "MonthlyIncome" in profile.columns


def test_prediction_helper_validates_calibrated_probability() -> None:
    class StubModel:
        def predict_proba(self, profile):
            assert len(profile) == 1
            return [[0.72, 0.28]]

    profile = pd.DataFrame({"Department": ["Sales"]})
    result = predict_profile(StubModel(), profile, threshold=0.21)
    assert result.probability == pytest.approx(0.28)
    assert result.risk_category == "LOW"
    assert result.threshold_prediction == 1
