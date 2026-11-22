"""Execute Phase 4 baseline model development."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_csv
from src.features.preprocessing import (
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    resolve_preprocessing_spec,
)
from src.models.model_development import (
    add_xgboost_if_available,
    build_model_registry,
    calculate_classification_metrics,
    positive_probability,
)
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 baseline model development."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/Palo Alto Networks.csv",
    )
    parser.add_argument(
        "--split-artifact",
        default="data/processed/phase3_split.joblib",
    )
    parser.add_argument(
        "--models-dir",
        default="models/phase4",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures/phase4",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def save_roc_and_pr_curves(
    y_true,
    probability_by_model: dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    """Save aggregate ROC and precision-recall figures."""
    fig, axis = plt.subplots(figsize=(9, 7))
    for name, probability in probability_by_model.items():
        RocCurveDisplay.from_predictions(
            y_true,
            probability,
            name=name,
            ax=axis,
        )
    axis.set_title("Out-of-Fold ROC Curves")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "phase4_roc_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 7))
    for name, probability in probability_by_model.items():
        PrecisionRecallDisplay.from_predictions(
            y_true,
            probability,
            name=name,
            ax=axis,
        )
    axis.set_title("Out-of-Fold Precision-Recall Curves")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "phase4_precision_recall_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_confusion_matrices(
    y_true,
    prediction_by_model: dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    """Save a confusion-matrix grid for all Phase 4 models."""
    names = list(prediction_by_model)
    rows = (len(names) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows))
    axes = np.asarray(axes).reshape(-1)

    for axis, name in zip(axes, names):
        matrix = np.array(
            [
                [
                    int(((y_true == 0) & (prediction_by_model[name] == 0)).sum()),
                    int(((y_true == 0) & (prediction_by_model[name] == 1)).sum()),
                ],
                [
                    int(((y_true == 1) & (prediction_by_model[name] == 0)).sum()),
                    int(((y_true == 1) & (prediction_by_model[name] == 1)).sum()),
                ],
            ]
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Stayed", "Attrition"],
            yticklabels=["Stayed", "Attrition"],
            ax=axis,
        )
        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")

    for axis in axes[len(names):]:
        axis.axis("off")

    fig.suptitle("Out-of-Fold Confusion Matrices")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "phase4_confusion_matrices.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_report(
    report_path: Path,
    comparison: pd.DataFrame,
    xgboost_available: bool,
    cv_folds: int,
) -> None:
    """Write measured Phase 4 model-development findings."""
    ranked = comparison.sort_values(
        ["pr_auc", "recall", "roc_auc"],
        ascending=False,
    )
    selected = ranked.iloc[0]

    lines = [
        "# Phase 4 Machine Learning Model Development",
        "",
        "## Scope",
        "",
        "This report uses only the 1,176-record training partition. "
        "The 294-record test partition remains untouched for later final evaluation.",
        "",
        "## Method",
        "",
        f"- Stratified cross-validation folds: **{cv_folds}**",
        "- Random seed: **42**",
        "- Preprocessing: existing feature engineering plus train-fold-only imputation, scaling, and encoding",
        "- Class imbalance: initial balanced class weights or equivalent XGBoost weighting",
        "- SMOTE: **not applied**",
        "- XGBoost available: "
        f"**{'yes' if xgboost_available else 'no'}**",
        "",
        "## Model comparison",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced accuracy | Specificity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in ranked.iterrows():
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | "
            f"{row['pr_auc']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['specificity']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"Using out-of-fold training results, the highest measured PR-AUC model is "
            f"**{selected['model']}** (PR-AUC **{selected['pr_auc']:.4f}**, "
            f"recall **{selected['recall']:.4f}**).",
            "",
            "This is a Phase 4 candidate selection only. It has not been evaluated on "
            "the held-out test set. Accuracy was not used as the sole ranking criterion "
            "because attrition is imbalanced.",
            "",
            "## Business interpretation",
            "",
            "- Recall is important because missed at-risk employees are a high-cost false negative.",
            "- Precision and specificity protect against excessive false alerts.",
            "- PR-AUC is especially relevant when the positive class is minority.",
            "- Final model suitability also requires calibration, threshold analysis, "
            "fairness auditing, and held-out test evaluation in later phases.",
            "",
            "## Leakage controls",
            "",
            "- The test partition was not loaded for model fitting or cross-validation.",
            "- Preprocessing is embedded in a pipeline and is re-fitted inside each training fold.",
            "- Sensitive fields remain excluded from model inputs.",
            "- The target column is not an input feature.",
            "",
            "## Limitations",
            "",
            "- Cross-validation results estimate training-fold performance, not final test performance.",
            "- Feature associations do not establish causation.",
            "- Probability thresholds have not yet been optimized.",
            "- SHAP, fairness, and final risk thresholds are separate later phases.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = configure_logging()
    sns.set_theme(style="whitegrid")

    if args.folds < 2:
        logger.error("Cross-validation folds must be at least 2.")
        return 1

    try:
        frame = load_csv(args.csv_path)
        split = joblib.load(args.split_artifact)
        train_indices = split["train_indices"]

        spec = resolve_preprocessing_spec(frame)
        model_columns = [
            column
            for column in frame.columns
            if column not in spec.excluded_columns
        ]
        x_train = frame.loc[train_indices, model_columns].copy()
        y_train = frame.loc[train_indices, TARGET_COLUMN].astype("int64").copy()

        registry = build_model_registry()
        class_counts = y_train.value_counts()
        registry, xgboost_available = add_xgboost_if_available(
            registry,
            negative_count=int(class_counts.get(0, 0)),
            positive_count=int(class_counts.get(1, 0)),
        )

        cv = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.random_state,
        )
        comparison_rows = []
        probability_by_model = {}
        prediction_by_model = {}
        classification_reports = {}

        models_dir = Path(args.models_dir)
        reports_dir = Path(args.reports_dir)
        figures_dir = Path(args.figures_dir)
        for directory in (models_dir, reports_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        for name, classifier in registry.items():
            preprocessing = build_preprocessing_pipeline(spec)
            pipeline = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("classifier", classifier),
                ]
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UndefinedMetricWarning,
                )
                cross_validate(
                    pipeline,
                    x_train,
                    y_train,
                    cv=cv,
                    scoring={
                        "accuracy": "accuracy",
                        "precision": "precision",
                        "recall": "recall",
                        "f1": "f1",
                        "roc_auc": "roc_auc",
                        "average_precision": "average_precision",
                        "balanced_accuracy": "balanced_accuracy",
                    },
                    n_jobs=-1,
                    return_train_score=False,
                )

            from sklearn.model_selection import cross_val_predict

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UndefinedMetricWarning,
                )
                oof_pred_values = cross_val_predict(
                    pipeline,
                    x_train,
                    y_train,
                    cv=cv,
                    method="predict",
                    n_jobs=-1,
                )
                oof_probability = cross_val_predict(
                    pipeline,
                    x_train,
                    y_train,
                    cv=cv,
                    method="predict_proba",
                    n_jobs=-1,
                )[:, 1]

            metrics = calculate_classification_metrics(
                y_train,
                oof_pred_values,
                oof_probability,
            )
            row = {"model": name, **metrics}
            comparison_rows.append(row)
            probability_by_model[name] = oof_probability
            prediction_by_model[name] = oof_pred_values
            classification_reports[name] = classification_report(
                y_train,
                oof_pred_values,
                target_names=["Stayed", "Attrition"],
                digits=4,
                zero_division=0,
            )

            pipeline.fit(x_train, y_train)
            joblib.dump(
                pipeline,
                models_dir / f"{name}.joblib",
            )

            logger.info(
                "%s: PR-AUC=%.4f recall=%.4f",
                name,
                metrics["pr_auc"],
                metrics["recall"],
            )

        comparison = pd.DataFrame(comparison_rows)
        comparison = comparison.sort_values(
            ["pr_auc", "recall", "roc_auc"],
            ascending=False,
        )
        comparison.to_csv(
            reports_dir / "model_comparison.csv",
            index=False,
        )

        metadata = {
            "phase": 4,
            "random_state": args.random_state,
            "cross_validation_folds": args.folds,
            "train_row_count": int(len(x_train)),
            "test_row_count": 294,
            "test_set_evaluated": False,
            "xgboost_available": xgboost_available,
            "smote_applied": False,
            "sensitive_columns_excluded": SENSITIVE_COLUMNS,
            "models": comparison.to_dict(orient="records"),
        }
        (reports_dir / "phase4_model_results.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        write_report(
            reports_dir / "phase4_model_evaluation.md",
            comparison,
            xgboost_available,
            args.folds,
        )
        save_roc_and_pr_curves(
            y_train,
            probability_by_model,
            figures_dir,
        )
        save_confusion_matrices(
            y_train,
            prediction_by_model,
            figures_dir,
        )

        logger.info(
            "Saved model comparison: %s",
            reports_dir / "model_comparison.csv",
        )
        logger.info(
            "Saved evaluation report: %s",
            reports_dir / "phase4_model_evaluation.md",
        )
        logger.info(
            "Saved trained models under: %s",
            models_dir,
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
