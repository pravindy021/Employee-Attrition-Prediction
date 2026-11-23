"""Employee attrition risk scoring and threshold utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60


def validate_probabilities(probabilities) -> np.ndarray:
    """Validate and return a one-dimensional probability array."""
    values = np.asarray(probabilities, dtype="float64").reshape(-1)
    if values.size == 0:
        raise ValueError("Probabilities must not be empty.")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities must contain only finite values.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Probabilities must be between 0 and 1.")
    return values


def classify_risk(probabilities) -> np.ndarray:
    """Map probabilities to Low, Medium, and High risk categories."""
    values = validate_probabilities(probabilities)
    categories = np.full(values.shape, "MEDIUM", dtype=object)
    categories[values < LOW_RISK_MAX] = "LOW"
    categories[values > MEDIUM_RISK_MAX] = "HIGH"
    return categories


def threshold_predictions(
    probabilities,
    threshold: float,
) -> np.ndarray:
    """Apply a configurable classification threshold."""
    values = validate_probabilities(probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    return (values >= threshold).astype("int64")


def optimize_threshold(
    y_true,
    probabilities,
    minimum_recall: float = 0.65,
):
    """Choose a training-only threshold balancing F1 and recall."""
    y_true_array = np.asarray(y_true).reshape(-1)
    values = validate_probabilities(probabilities)
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("Minimum recall must be between 0 and 1.")

    thresholds = np.arange(0.05, 0.951, 0.01)
    rows = []
    eligible = []
    for threshold in thresholds:
        prediction = threshold_predictions(values, float(threshold))
        recall = recall_score(y_true_array, prediction, zero_division=0)
        row = {
            "threshold": float(threshold),
            "precision": float(
                precision_score(y_true_array, prediction, zero_division=0)
            ),
            "recall": float(recall),
            "f1": float(f1_score(y_true_array, prediction, zero_division=0)),
        }
        rows.append(row)
        if recall >= minimum_recall:
            eligible.append(row)

    tradeoffs = pd.DataFrame(rows)
    candidates = pd.DataFrame(eligible) if eligible else tradeoffs
    selected = candidates.sort_values(
        ["f1", "recall", "precision"],
        ascending=False,
    ).iloc[0]
    return float(selected["threshold"]), tradeoffs


def build_employee_risk_table(
    test_frame: pd.DataFrame,
    y_true,
    probabilities,
    threshold: float,
) -> pd.DataFrame:
    """Create an anonymous employee-level risk table."""
    values = validate_probabilities(probabilities)
    if len(test_frame) != len(values):
        raise ValueError("Test frame and probability lengths differ.")

    result = pd.DataFrame(
        {
            "anonymous_record_id": [
                f"record_{index}" for index in test_frame.index
            ],
            "department": test_frame["Department"].to_numpy(),
            "job_role": test_frame["JobRole"].to_numpy(),
            "over_time": test_frame["OverTime"].to_numpy(),
            "attrition_probability": values,
            "risk_category": classify_risk(values),
            "threshold_prediction": threshold_predictions(values, threshold),
            "historical_attrition": np.asarray(y_true).reshape(-1),
        }
    )
    return result


def build_department_risk_summary(
    employee_risk_table: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate employee-risk estimates by department."""
    summary = (
        employee_risk_table.groupby("department", observed=True)
        .agg(
            employee_count=("anonymous_record_id", "size"),
            mean_attrition_probability=("attrition_probability", "mean"),
            high_risk_count=("risk_category", lambda values: int((values == "HIGH").sum())),
            medium_risk_count=("risk_category", lambda values: int((values == "MEDIUM").sum())),
            low_risk_count=("risk_category", lambda values: int((values == "LOW").sum())),
        )
        .reset_index()
    )
    summary["high_risk_percent"] = (
        summary["high_risk_count"] / summary["employee_count"] * 100
    )
    return summary.sort_values(
        "mean_attrition_probability",
        ascending=False,
    )
