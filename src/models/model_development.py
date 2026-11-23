"""Phase 4 model registry and classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def build_model_registry(class_weight_strategy: str = "balanced") -> dict[str, Any]:
    """Build a reproducible Phase 4 model registry.

    XGBoost is optional at call time but is included when its package import works.
    """
    del class_weight_strategy
    return {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
            solver="lbfgs",
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=42,
        ),
    }


def add_xgboost_if_available(
    registry: dict[str, Any],
    negative_count: int,
    positive_count: int,
) -> tuple[dict[str, Any], bool]:
    """Add XGBoost only when the installed package can be imported."""
    try:
        from xgboost import XGBClassifier
    except (ImportError, OSError) as exc:
        del exc
        return registry, False

    scale_pos_weight = (
        negative_count / positive_count if positive_count else 1.0
    )
    registry["xgboost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    return registry, True


def positive_probability(model: Any, features) -> np.ndarray:
    """Return positive-class probabilities or equivalent scores."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        score = model.decision_function(features)
        return np.asarray(score).reshape(-1)
    raise TypeError("Model does not provide predict_proba or decision_function.")


def calculate_classification_metrics(
    y_true,
    y_pred,
    y_probability,
) -> dict[str, float]:
    """Calculate the Phase 4 model metrics used in the comparison."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(
            average_precision_score(y_true, y_probability)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "specificity": float(specificity),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def select_promising_models(
    comparison,
    candidate_names: list[str] | None = None,
    limit: int = 3,
):
    """Select the highest PR-AUC candidate models from measured baseline results."""
    frame = comparison.copy()
    if candidate_names:
        frame = frame[frame["model"].isin(candidate_names)]
    if "pr_auc" not in frame.columns:
        raise ValueError("Comparison data must contain a pr_auc column.")
    return (
        frame.sort_values(
            ["pr_auc", "recall", "roc_auc"],
            ascending=False,
        )
        .head(limit)
    )
