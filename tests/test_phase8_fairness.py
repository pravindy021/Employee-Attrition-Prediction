"""Tests for Phase 8 fairness metrics and audit separation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fairness.fairness_metrics import (
    fairness_summary,
    group_metrics,
)


def test_group_metrics_basic() -> None:
    table = group_metrics(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 0]),
        probabilities=np.array([0.1, 0.3, 0.8, 0.2]),
        groups=np.array(["A", "A", "B", "B"]),
    )
    assert set(table["group"]) == {"A", "B"}
    assert table["sample_size"].sum() == 4
    assert table.loc[table["group"] == "A", "fpr"].iloc[0] == 0.5
    assert table.loc[table["group"] == "B", "fnr"].iloc[0] == 0.5


def test_zero_division_metrics_are_stable() -> None:
    table = group_metrics(
        y_true=np.array([0, 0]),
        y_pred=np.array([0, 0]),
        probabilities=np.array([0.1, 0.2]),
        groups=np.array(["A", "A"]),
    )
    assert table["precision"].iloc[0] == 0.0
    assert table["tpr_recall"].iloc[0] == 0.0
    assert table["fnr"].iloc[0] == 0.0


def test_small_subgroup_reproducibility() -> None:
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])
    probabilities = np.array([0.1, 0.8, 0.2])
    groups = np.array(["A", "A", "B"])

    first = group_metrics(y_true, y_pred, probabilities, groups)
    second = group_metrics(y_true, y_pred, probabilities, groups)
    pd.testing.assert_frame_equal(first, second)


def test_missing_groups_are_preserved_as_values() -> None:
    table = group_metrics(
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        probabilities=np.array([0.1, 0.8]),
        groups=np.array(["Missing", "A"]),
    )
    assert "Missing" in set(table["group"])


def test_mismatched_input_lengths_raise() -> None:
    with pytest.raises(ValueError):
        group_metrics(
            y_true=np.array([0, 1]),
            y_pred=np.array([0]),
            probabilities=np.array([0.1, 0.8]),
            groups=np.array(["A", "B"]),
        )


def test_fairness_summary_is_reproducible() -> None:
    table = pd.DataFrame(
        {
            "selection_rate": [0.2, 0.4],
            "tpr_recall": [0.5, 0.8],
            "fpr": [0.1, 0.3],
            "precision": [0.6, 0.7],
            "actual_positive_count": [10, 10],
            "sample_size": [100, 100],
        }
    )
    first = fairness_summary(table)
    second = fairness_summary(table)
    assert first == second
    assert first["demographic_parity_difference"] == pytest.approx(0.2)
