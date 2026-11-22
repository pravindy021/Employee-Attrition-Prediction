"""Dashboard prediction helpers without retraining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.risk_scoring import classify_risk, validate_probabilities


@dataclass(frozen=True)
class PredictionResult:
    """A calibrated propensity and its operational risk category."""

    probability: float
    threshold_prediction: int
    risk_category: str


def build_profile_frame(
    template: pd.DataFrame,
    model_columns: list[str],
    values: dict,
) -> pd.DataFrame:
    """Create a complete model-input row from validated form values."""
    if template.empty:
        raise ValueError("Cannot build a profile without dataset defaults.")

    row = {}
    for column in model_columns:
        if column in values:
            row[column] = values[column]
        elif pd.api.types.is_numeric_dtype(template[column]):
            row[column] = float(template[column].median())
        else:
            mode = template[column].mode(dropna=True)
            row[column] = mode.iloc[0] if not mode.empty else ""
    return pd.DataFrame([row], columns=model_columns)


def predict_profile(
    calibrated_model,
    profile: pd.DataFrame,
    threshold: float,
) -> PredictionResult:
    """Predict with the existing calibrated model only."""
    probability_matrix = np.asarray(
        calibrated_model.predict_proba(profile),
        dtype="float64",
    )
    if probability_matrix.ndim != 2 or probability_matrix.shape[1] < 2:
        raise ValueError("The prediction model must return two probability columns.")
    probabilities = validate_probabilities(probability_matrix[:, 1])
    probability = float(probabilities[0])
    prediction = int(probability >= threshold)
    category = str(classify_risk([probability])[0])
    return PredictionResult(
        probability=probability,
        threshold_prediction=prediction,
        risk_category=category,
    )
