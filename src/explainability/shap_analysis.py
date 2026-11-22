"""SHAP analysis utilities for the fitted logistic regression pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


SENSITIVE_FEATURES = ("Age", "Gender", "MaritalStatus")


@dataclass(frozen=True)
class ExplanationResult:
    """SHAP values and associated transformed feature names."""

    shap_values: np.ndarray
    feature_names: tuple[str, ...]
    base_value: float


def dense_matrix(matrix) -> np.ndarray:
    """Convert sparse or dense matrices to a NumPy array."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return names emitted by the fitted feature-engineering pipeline."""
    preprocessing = pipeline.named_steps["preprocessing"]
    column_transformer = preprocessing.named_steps["preprocessor"]
    names = column_transformer.get_feature_names_out()
    return [
        str(name).replace("__", "_").replace("_", " ", 1)
        if str(name).startswith(("numeric_", "categorical_"))
        else str(name)
        for name in names
    ]


def transform_features(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Transform raw model-input records using the fitted pipeline."""
    preprocessing = pipeline.named_steps["preprocessing"]
    transformed = preprocessing.transform(frame)
    return dense_matrix(transformed)


def initialize_linear_explainer(
    classifier,
    background: np.ndarray,
) -> shap.LinearExplainer:
    """Initialize a linear SHAP explainer for logistic regression."""
    if not hasattr(classifier, "coef_"):
        raise TypeError("LinearExplainer requires a fitted linear classifier.")
    if background.ndim != 2 or background.shape[0] == 0:
        raise ValueError("SHAP background must be a non-empty 2D matrix.")
    return shap.LinearExplainer(classifier, background)


def calculate_shap_values(
    explainer: shap.LinearExplainer,
    transformed: np.ndarray,
) -> ExplanationResult:
    """Calculate SHAP values and normalize SHAP's possible result containers."""
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[2] == 2 else values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2D SHAP value matrix, received shape {values.shape}"
        )
    if values.shape[0] != transformed.shape[0]:
        raise ValueError("SHAP output row count does not match input row count.")

    base = explainer.expected_value
    if isinstance(base, np.ndarray):
        base = base[1] if base.size > 1 else base[0]
    return ExplanationResult(
        shap_values=values,
        feature_names=tuple(),
        base_value=float(base),
    )


def clean_feature_name(name: str) -> str:
    """Convert encoded column names into readable labels."""
    label = name
    for prefix in ("numeric_", "categorical_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    label = label.replace("_", " ")
    return label.strip()


def protect_sensitive_features(names: Iterable[str]) -> list[str]:
    """Exclude sensitive feature names and their encoded categories."""
    protected: list[str] = []
    for name in names:
        normalized = str(name).lower().replace("_", " ").replace("-", " ")
        words = normalized.split()
        is_sensitive = any(
            normalized == sensitive.lower()
            or normalized.startswith(f"{sensitive.lower()} ")
            or normalized.startswith(f"{sensitive.lower()}_")
            for sensitive in SENSITIVE_FEATURES
        )
        del words
        if not is_sensitive:
            protected.append(str(name))
    return protected


def validate_sensitivity_exclusion(
    feature_names: Iterable[str],
) -> None:
    """Raise if an explanation list exposes a sensitive field."""
    protected = protect_sensitive_features(feature_names)
    if len(protected) != len(tuple(feature_names)):
        raise ValueError("Sensitive feature found in explanation output.")


def global_importance(
    shap_values: np.ndarray,
    feature_names: Iterable[str],
) -> pd.DataFrame:
    """Calculate mean absolute SHAP importance and positive/negative averages."""
    names = list(feature_names)
    if shap_values.shape[1] != len(names):
        raise ValueError("SHAP value columns do not match feature names.")

    mean_absolute = np.abs(shap_values).mean(axis=0)
    positive_average = np.where(
        shap_values > 0,
        shap_values,
        np.nan,
    )
    negative_average = np.where(
        shap_values < 0,
        shap_values,
        np.nan,
    )
    positive_column = []
    negative_column = []
    for column_index in range(shap_values.shape[1]):
        positive_values = positive_average[:, column_index]
        positive_values = positive_values[~np.isnan(positive_values)]
        negative_values = negative_average[:, column_index]
        negative_values = negative_values[~np.isnan(negative_values)]
        positive_column.append(
            float(positive_values.mean()) if positive_values.size else 0.0
        )
        negative_column.append(
            float(negative_values.mean()) if negative_values.size else 0.0
        )
    table = pd.DataFrame(
        {
            "feature": names,
            "mean_absolute_shap": mean_absolute,
            "average_positive_shap": positive_column,
            "average_negative_shap": negative_column,
        }
    )
    table["average_positive_shap"] = table[
        "average_positive_shap"
    ].fillna(0.0)
    table["average_negative_shap"] = table[
        "average_negative_shap"
    ].fillna(0.0)
    return table.sort_values("mean_absolute_shap", ascending=False)


def generate_individual_explanation(
    shap_values: np.ndarray,
    transformed_values: np.ndarray,
    feature_names: Iterable[str],
    calibrated_probability: float,
    threshold_prediction: int,
    record_id: str,
    top_n: int = 8,
) -> dict[str, object]:
    """Generate a human-readable individual explanation."""
    if not 0.0 <= calibrated_probability <= 1.0:
        raise ValueError("Calibrated probability must be between 0 and 1.")
    if shap_values.ndim != 1:
        raise ValueError("A single-record SHAP vector is required.")

    names = list(feature_names)
    if len(names) != len(shap_values) or len(names) != len(transformed_values):
        raise ValueError("Explanation feature dimensions do not match.")

    safe_names = protect_sensitive_features(names)
    if len(safe_names) != len(names):
        raise ValueError("Sensitive feature detected in the explanation set.")

    ranking = np.argsort(np.abs(shap_values))[::-1][:top_n]
    positive = [
        {
            "feature": names[index],
            "contribution": float(shap_values[index]),
            "transformed_value": float(transformed_values[index]),
        }
        for index in ranking
        if shap_values[index] > 0
    ]
    negative = [
        {
            "feature": names[index],
            "contribution": float(shap_values[index]),
            "transformed_value": float(transformed_values[index]),
        }
        for index in ranking
        if shap_values[index] < 0
    ]

    if calibrated_probability < 0.30:
        risk_category = "LOW"
    elif calibrated_probability <= 0.60:
        risk_category = "MEDIUM"
    else:
        risk_category = "HIGH"

    text_parts = [
        f"Record {record_id} has a calibrated attrition propensity of "
        f"{calibrated_probability:.1%}, categorized as {risk_category} risk."
    ]
    if positive:
        text_parts.append(
            "The largest increasing features are "
            + ", ".join(item["feature"] for item in positive[:3])
            + "."
        )
    if negative:
        text_parts.append(
            "The largest decreasing features are "
            + ", ".join(item["feature"] for item in negative[:3])
            + "."
        )
    text_parts.append(
        "These are model associations, not causal reasons or a certainty of resignation."
    )

    return {
        "anonymous_record_id": record_id,
        "calibrated_attrition_probability": float(calibrated_probability),
        "risk_category": risk_category,
        "threshold_prediction": int(threshold_prediction),
        "positive_contributors": positive,
        "negative_contributors": negative,
        "human_readable_explanation": " ".join(text_parts),
    }
