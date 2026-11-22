"""Group fairness metrics for held-out employee predictions."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio or zero when the denominator is zero."""
    return float(numerator / denominator) if denominator else 0.0


def group_metrics(
    y_true,
    y_pred,
    probabilities,
    groups,
) -> pd.DataFrame:
    """Calculate descriptive and classification metrics for each demographic group."""
    y_true_array = np.asarray(y_true).reshape(-1)
    y_pred_array = np.asarray(y_pred).reshape(-1)
    probabilities_array = np.asarray(probabilities, dtype="float64").reshape(-1)
    group_array = np.asarray(groups).reshape(-1)

    if not (
        len(y_true_array)
        == len(y_pred_array)
        == len(probabilities_array)
        == len(group_array)
    ):
        raise ValueError("Fairness metric inputs must have matching lengths.")

    frame = pd.DataFrame(
        {
            "actual": y_true_array,
            "predicted": y_pred_array,
            "probability": probabilities_array,
            "group": group_array,
        }
    )

    rows = []
    for group, subset in frame.groupby(
        "group",
        observed=True,
        dropna=False,
    ):
        actual = subset["actual"].to_numpy()
        predicted = subset["predicted"].to_numpy()

        positive_actual = actual == 1
        negative_actual = actual == 0
        positive_predicted = predicted == 1
        negative_predicted = predicted == 0

        true_positive = int((positive_actual & positive_predicted).sum())
        false_negative = int((positive_actual & negative_predicted).sum())
        true_negative = int((negative_actual & negative_predicted).sum())
        false_positive = int((negative_actual & positive_predicted).sum())

        positive_count = int(positive_actual.sum())
        negative_count = int(negative_actual.sum())
        predicted_positive_count = int(positive_predicted.sum())

        rows.append(
            {
                "group": str(group),
                "sample_size": int(len(subset)),
                "actual_positive_count": positive_count,
                "predicted_positive_count": predicted_positive_count,
                "selection_rate": safe_ratio(
                    predicted_positive_count,
                    len(subset),
                ),
                "true_positive_count": true_positive,
                "false_positive_count": false_positive,
                "true_negative_count": true_negative,
                "false_negative_count": false_negative,
                "tpr_recall": safe_ratio(
                    true_positive,
                    positive_count,
                ),
                "fnr": safe_ratio(
                    false_negative,
                    positive_count,
                ),
                "fpr": safe_ratio(
                    false_positive,
                    negative_count,
                ),
                "tnr_specificity": safe_ratio(
                    true_negative,
                    negative_count,
                ),
                "precision": safe_ratio(
                    true_positive,
                    true_positive + false_positive,
                ),
                "accuracy": safe_ratio(
                    true_positive + true_negative,
                    len(subset),
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No fairness groups were available.")
    return result


def fairness_summary(group_table: pd.DataFrame) -> dict[str, float | str]:
    """Calculate maximum-minus-minimum group fairness differences."""
    if "selection_rate" not in group_table or "tpr_recall" not in group_table:
        raise ValueError("Group table is missing required fairness columns.")

    selection_rates = group_table["selection_rate"].to_numpy()
    recall_rates = group_table["tpr_recall"].to_numpy()
    false_positive_rates = group_table["fpr"].to_numpy()
    precisions = group_table["precision"].to_numpy()

    eligible = group_table[
        (group_table["actual_positive_count"] > 0)
        & (group_table["sample_size"] >= 20)
    ]

    return {
        "demographic_parity_difference": float(
            selection_rates.max() - selection_rates.min()
        ),
        "equal_opportunity_difference": float(
            eligible["tpr_recall"].max()
            - eligible["tpr_recall"].min()
        )
        if not eligible.empty
        else 0.0,
        "false_positive_rate_difference": float(
            false_positive_rates.max() - false_positive_rates.min()
        ),
        "precision_difference": float(
            precisions.max() - precisions.min()
        ),
    }
