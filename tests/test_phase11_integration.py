"""Phase 11 integration, leakage, and reproducibility tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.utils.artifacts import (
    load_calibrated_model,
    load_employee_data,
    load_risk_config,
    model_feature_columns,
)
from app.utils.prediction import build_profile_frame, predict_profile
from src.features.preprocessing import SENSITIVE_COLUMNS
from src.models.risk_scoring import classify_risk


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_saved_calibrated_model_loads_and_predicts_valid_probability() -> None:
    frame = load_employee_data()
    model = load_calibrated_model()
    model_columns, _ = model_feature_columns()
    profile = build_profile_frame(frame, model_columns, {})
    result = predict_profile(model, profile, threshold=0.21)

    assert 0.0 <= result.probability <= 1.0
    assert result.risk_category in {"LOW", "MEDIUM", "HIGH"}
    assert result.threshold_prediction in {0, 1}


def test_repeated_predictions_are_reproducible() -> None:
    frame = load_employee_data()
    model = load_calibrated_model()
    model_columns, _ = model_feature_columns()
    profile = build_profile_frame(frame, model_columns, {})

    first = predict_profile(model, profile, 0.21).probability
    second = predict_profile(model, profile, 0.21).probability
    assert first == pytest.approx(second)


def test_train_test_indices_are_disjoint() -> None:
    split = joblib.load(
        PROJECT_ROOT / "data" / "processed" / "phase3_split.joblib"
    )
    train = set(split["train_indices"].tolist())
    test = set(split["test_indices"].tolist())
    assert train.isdisjoint(test)
    assert len(train) == 1176
    assert len(test) == 294


def test_sensitive_attributes_are_not_model_inputs() -> None:
    frame = load_employee_data()
    model_columns, sensitive = model_feature_columns()
    assert set(sensitive) == set(SENSITIVE_COLUMNS)
    assert set(sensitive).isdisjoint(set(model_columns))
    assert "Attrition" not in model_columns


def test_original_csv_hash_is_unchanged() -> None:
    original = PROJECT_ROOT / "data" / "Palo Alto Networks.csv"
    raw_copy = PROJECT_ROOT / "data" / "raw" / "Palo Alto Networks.csv"
    assert original.exists()
    assert raw_copy.exists()

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert digest(original) == digest(raw_copy)
    assert digest(original) == (
        "389735e8909136a51fe487040df8f8662c2d222bb3505a83745c5a8c5bf20c7c".lower()
    )


def test_boundary_risk_categories() -> None:
    categories = classify_risk([0.299, 0.300, 0.600, 0.601])
    assert categories.tolist() == ["LOW", "MEDIUM", "MEDIUM", "HIGH"]


def test_phase8_and_phase7_reports_load() -> None:
    demographic = pd.read_csv(
        PROJECT_ROOT / "reports" / "DEMOGRAPHIC_METRICS.csv"
    )
    shap_importance = pd.read_csv(
        PROJECT_ROOT / "reports" / "explainability" / "shap_global_importance.csv"
    )
    individual = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "explainability"
            / "individual_explanations.json"
        ).read_text(encoding="utf-8")
    )
    assert not demographic.empty
    assert not shap_importance.empty
    assert len(individual) >= 1


def test_configured_threshold_matches_phase6() -> None:
    config = load_risk_config()
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "phase6_validation_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert config["threshold"] == pytest.approx(0.21)
    assert metadata["selected_threshold"] == pytest.approx(0.21)
