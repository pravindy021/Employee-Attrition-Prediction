"""Pytest tests for Phase 7 SHAP explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.explainability.shap_analysis import (
    calculate_shap_values,
    generate_individual_explanation,
    global_importance,
    initialize_linear_explainer,
    protect_sensitive_features,
)


def test_sensitive_features_are_excluded() -> None:
    names = ["Age", "Overtime Flag", "Gender Male", "Monthly Income"]
    assert protect_sensitive_features(names) == ["Overtime Flag", "Monthly Income"]


def test_linear_explainer_output_dimensions() -> None:
    background = np.array(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
        dtype="float64",
    )
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42).fit(
        background,
        np.array([0, 0, 1]),
    )
    explainer = initialize_linear_explainer(model, background)
    result = calculate_shap_values(explainer, background)
    assert result.shap_values.shape == (3, 2)


def test_global_importance_matches_feature_names() -> None:
    values = np.array([[0.2, -0.1], [-0.3, 0.4]])
    importance = global_importance(values, ["Overtime", "Income"])
    assert len(importance) == 2
    assert set(importance["feature"]) == {"Overtime", "Income"}


def test_individual_explanation_and_risk_boundary() -> None:
    explanation = generate_individual_explanation(
        shap_values=np.array([0.4, -0.2]),
        transformed_values=np.array([1.0, 0.5]),
        feature_names=["Overtime", "Monthly Income"],
        calibrated_probability=0.61,
        threshold_prediction=1,
        record_id="record_1",
    )
    assert explanation["risk_category"] == "HIGH"
    assert explanation["calibrated_attrition_probability"] == 0.61
    assert "causal" in explanation["human_readable_explanation"]


def test_invalid_individual_probability_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        generate_individual_explanation(
            shap_values=np.array([0.1]),
            transformed_values=np.array([1.0]),
            feature_names=["Overtime"],
            calibrated_probability=1.2,
            threshold_prediction=1,
            record_id="record_1",
        )
