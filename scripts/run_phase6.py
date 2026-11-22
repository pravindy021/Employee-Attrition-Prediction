"""Execute Phase 6 held-out model validation and risk scoring."""

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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    brier_score_loss,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_csv
from src.features.preprocessing import (
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
    resolve_preprocessing_spec,
)
from src.models.model_development import calculate_classification_metrics
from src.models.risk_scoring import (
    build_department_risk_summary,
    build_employee_risk_table,
    optimize_threshold,
)
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 6 test validation and employee risk scoring."
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
        "--tuned-model-dir",
        default="models/phase5",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="data/processed",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures/phase6",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def save_calibration_plot(
    y_true,
    probabilities_by_name: dict[str, np.ndarray],
    figure_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    for name, probabilities in probabilities_by_name.items():
        observed, predicted = calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", label=name)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed attrition fraction")
    axis.set_title("Held-Out Probability Calibration")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_threshold_plot(
    tradeoffs,
    threshold: float,
    figure_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        tradeoffs["threshold"],
        tradeoffs["precision"],
        label="Precision",
    )
    axis.plot(
        tradeoffs["threshold"],
        tradeoffs["recall"],
        label="Recall",
    )
    axis.plot(
        tradeoffs["threshold"],
        tradeoffs["f1"],
        label="F1",
    )
    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        label=f"Selected threshold {threshold:.2f}",
    )
    axis.set_xlabel("Classification threshold")
    axis.set_ylabel("Metric value")
    axis.set_title("Training OOF Precision-Recall-F1 Trade-Off")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_test_curves(
    y_true,
    probabilities,
    figure_prefix: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        name="Selected calibrated logistic model",
        ax=axis,
    )
    axis.set_title("Held-Out ROC Curve")
    fig.tight_layout()
    fig.savefig(
        figure_prefix.with_name("phase6_test_roc.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 7))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        name="Selected calibrated logistic model",
        ax=axis,
    )
    axis.set_title("Held-Out Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(
        figure_prefix.with_name("phase6_test_precision_recall.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_test_confusion(
    y_true,
    y_pred,
    figure_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Stayed", "Attrition"],
        cmap="Blues",
        ax=axis,
    )
    axis.set_title("Held-Out Confusion Matrix")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_phase6_report(
    report_path: Path,
    candidate_test: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    selected_method: str,
    threshold: float,
    employee_count_by_risk: pd.Series,
) -> None:
    best = candidate_test.sort_values(
        ["pr_auc", "recall", "roc_auc"],
        ascending=False,
    ).iloc[0]

    lines = [
        "# Phase 6 Model Validation and Employee Risk Scoring",
        "",
        "## Scope and data",
        "",
        "- Training rows used for threshold and calibration decisions: **1,176**",
        "- Held-out test rows evaluated for the first time: **294**",
        "- Candidate models: top tuned candidates by Phase 5 training PR-AUC",
        "- Historical target: `Attrition = 1`",
        "- No 30/60/90-day outcome horizon is present in the dataset.",
        "",
        "This is an estimate of historical attrition propensity, not a time-bound prediction "
        "of who will leave within 30, 60, or 90 days.",
        "",
        "## Held-out candidate validation",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in candidate_test.iterrows():
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | "
            f"{row['pr_auc']:.4f} | {row['brier_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"The highest held-out PR-AUC candidate in this validation run is "
            f"**{best['model']}** at **{best['pr_auc']:.4f}**.",
            "",
            "## Calibration",
            "",
            "| Method | Training OOF Brier score |",
            "|---|---:|",
        ]
    )
    for _, row in calibration_summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['training_oof_brier']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"Selected calibration method: **{selected_method}**",
            "",
            "Calibration was selected using training-only out-of-fold predictions.",
            "The held-out test set was used only to report final validation performance.",
            "",
            "## Threshold decision",
            "",
            f"- Selected threshold: **{threshold:.2f}**",
            "- Selection objective: minimum recall 65%, then highest training OOF F1",
            "- Lower thresholds generally increase recall and false positives.",
            "- Higher thresholds generally increase precision and missed-risk count.",
            "",
            "## Initial risk categories",
            "",
            "- LOW: probability below 30%",
            "- MEDIUM: probability from 30% through 60%",
            "- HIGH: probability above 60%",
            "",
            "These are configurable operational assumptions, not scientific boundaries.",
            "A probability is a relative estimate, not certainty.",
            "",
            "## Employee risk distribution",
            "",
        ]
    )
    for category in ["LOW", "MEDIUM", "HIGH"]:
        lines.append(
            f"- {category}: **{int(employee_count_by_risk.get(category, 0))}**"
        )

    lines.extend(
        [
            "",
            "## Business and ethical interpretation",
            "",
            "- Risk scores support HR research and human review, not automated employment decisions.",
            "- Historical patterns may reproduce organizational bias.",
            "- Age, Gender, and MaritalStatus remain separate from model inputs.",
            "- The reports do not establish that a feature causes attrition.",
            "- A high score does not determine that an employee will resign.",
            "",
            "## Limitations",
            "",
            "- The held-out data is a single temporal-independent split, not an external production sample.",
            "- No survey source, outcome horizon, or company provenance beyond the supplied file was verified.",
            "- Risk categories require stakeholder and legal review before operational use.",
            "- Calibration and threshold behavior may change under future data drift.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = configure_logging()
    sns.set_theme(style="whitegrid")

    try:
        frame = load_csv(args.csv_path)
        split = joblib.load(args.split_artifact)
        train_indices = split["train_indices"]
        test_indices = split["test_indices"]

        spec = resolve_preprocessing_spec(frame)
        model_columns = [
            column
            for column in frame.columns
            if column not in spec.excluded_columns
        ]
        x_train = frame.loc[train_indices, model_columns].copy()
        y_train = frame.loc[train_indices, TARGET_COLUMN].astype("int64").copy()
        x_test = frame.loc[test_indices, model_columns].copy()
        y_test = frame.loc[test_indices, TARGET_COLUMN].astype("int64").copy()

        tuned_dir = Path(args.tuned_model_dir)
        reports_dir = Path(args.reports_dir)
        artifacts_dir = Path(args.artifacts_dir)
        final_models_dir = Path(args.models_dir)
        figures_dir = Path(args.figures_dir)
        for directory in (
            reports_dir,
            artifacts_dir,
            final_models_dir,
            figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        candidate_names = [
            "logistic_regression",
            "gradient_boosting",
            "xgboost",
        ]
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=args.random_state,
        )

        candidate_test_rows = []
        test_probabilities = {}
        for name in candidate_names:
            model_path = tuned_dir / f"{name}_tuned.joblib"
            if not model_path.exists():
                continue
            pipeline = joblib.load(model_path)
            probabilities = pipeline.predict_proba(x_test)[:, 1]
            predictions = (probabilities >= 0.5).astype("int64")
            metrics = calculate_classification_metrics(
                y_test,
                predictions,
                probabilities,
            )
            candidate_test_rows.append(
                {
                    "model": name,
                    "brier_score": float(
                        brier_score_loss(y_test, probabilities)
                    ),
                    **metrics,
                }
            )
            test_probabilities[name] = np.asarray(probabilities)

        if not candidate_test_rows:
            logger.error("No tuned candidate artifacts were found.")
            return 1

        candidate_test = pd.DataFrame(candidate_test_rows).sort_values(
            ["pr_auc", "recall", "roc_auc"],
            ascending=False,
        )
        candidate_test.to_csv(
            reports_dir / "test_model_validation.csv",
            index=False,
        )

        logistic_pipeline = joblib.load(
            tuned_dir / "logistic_regression_tuned.joblib"
        )

        raw_oof = cross_val_predict(
            clone(logistic_pipeline),
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        calibration_summary_rows = []
        calibration_oof = {}
        for method in ("sigmoid", "isotonic"):
            calibrator = CalibratedClassifierCV(
                estimator=clone(logistic_pipeline),
                method=method,
                cv=cv,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UndefinedMetricWarning,
                )
                oof_probabilities = cross_val_predict(
                    calibrator,
                    x_train,
                    y_train,
                    cv=cv,
                    method="predict_proba",
                    n_jobs=-1,
                )[:, 1]
            calibration_oof[method] = oof_probabilities
            calibration_summary_rows.append(
                {
                    "method": method,
                    "training_oof_brier": float(
                        brier_score_loss(y_train, oof_probabilities)
                    ),
                }
            )

        calibration_summary = pd.DataFrame(calibration_summary_rows)
        selected_calibration = calibration_summary.sort_values(
            "training_oof_brier"
        ).iloc[0]["method"]

        final_calibrator = CalibratedClassifierCV(
            estimator=clone(logistic_pipeline),
            method=selected_calibration,
            cv=cv,
        )
        final_calibrator.fit(x_train, y_train)
        final_probabilities = final_calibrator.predict_proba(x_test)[:, 1]
        final_predictions_raw = final_calibrator.predict(x_test)

        threshold, oof_tradeoffs = optimize_threshold(
            y_train,
            calibration_oof[selected_calibration],
            minimum_recall=0.65,
        )
        final_predictions = (
            final_probabilities >= threshold
        ).astype("int64")

        final_metrics = calculate_classification_metrics(
            y_test,
            final_predictions,
            final_probabilities,
        )
        final_metrics["brier_score"] = float(
            brier_score_loss(y_test, final_probabilities)
        )
        final_metrics["threshold"] = threshold

        employee_risk = build_employee_risk_table(
            frame.loc[test_indices],
            y_test,
            final_probabilities,
            threshold,
        )
        department_risk = build_department_risk_summary(employee_risk)

        employee_count_by_risk = employee_risk["risk_category"].value_counts()
        employee_risk.to_csv(
            artifacts_dir / "employee_risk_scores.csv",
            index=False,
        )
        department_risk.to_csv(
            artifacts_dir / "department_risk_summary.csv",
            index=False,
        )

        joblib.dump(
            final_calibrator,
            final_models_dir / "phase6_attrition_risk_model.joblib",
        )
        joblib.dump(
            {
                "model": final_calibrator,
                "threshold": threshold,
                "risk_categories": {
                    "low": "<0.30",
                    "medium": "0.30-0.60",
                    "high": ">0.60",
                },
            },
            artifacts_dir / "phase6_risk_scoring_config.joblib",
        )

        calibration_summary.to_csv(
            reports_dir / "probability_calibration.csv",
            index=False,
        )
        oof_tradeoffs.to_csv(
            reports_dir / "threshold_tradeoffs.csv",
            index=False,
        )

        metadata = {
            "phase": 6,
            "random_state": args.random_state,
            "train_row_count": int(len(x_train)),
            "test_row_count": int(len(x_test)),
            "test_set_evaluated": True,
            "candidate_selection_basis": "Phase 5 training PR-AUC",
            "candidate_test_models": candidate_names,
            "selected_final_model": "logistic_regression",
            "selected_calibration_method": selected_calibration,
            "selected_threshold": threshold,
            "minimum_training_recall_objective": 0.65,
            "risk_categories": {
                "low": "probability < 0.30",
                "medium": "0.30 <= probability <= 0.60",
                "high": "probability > 0.60",
            },
            "held_out_metrics": final_metrics,
            "brier_score_raw_test": float(
                brier_score_loss(
                    y_test,
                    test_probabilities["logistic_regression"],
                )
            ),
            "brier_score_calibrated_test": float(
                brier_score_loss(y_test, final_probabilities)
            ),
            "sensitive_columns_excluded": SENSITIVE_COLUMNS,
            "interpretation": (
                "Historical attrition propensity; not a 30/60/90-day resignation forecast."
            ),
        }
        (reports_dir / "phase6_validation_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        write_phase6_report(
            reports_dir / "phase6_validation_report.md",
            candidate_test,
            calibration_summary,
            str(selected_calibration),
            threshold,
            employee_count_by_risk,
        )

        save_calibration_plot(
            y_test,
            {
                "Raw logistic": test_probabilities["logistic_regression"],
                "Calibrated logistic": final_probabilities,
            },
            figures_dir / "phase6_calibration_curves.png",
        )
        save_threshold_plot(
            oof_tradeoffs,
            threshold,
            figures_dir / "phase6_threshold_tradeoff.png",
        )
        save_test_curves(
            y_test,
            final_probabilities,
            figures_dir / "phase6_test_precision_recall.png",
        )
        save_test_confusion(
            y_test,
            final_predictions,
            figures_dir / "phase6_test_confusion_matrix.png",
        )

        logger.info(
            "Held-out test evaluated for %d records.",
            len(y_test),
        )
        logger.info(
            "Selected calibration=%s threshold=%.2f",
            selected_calibration,
            threshold,
        )
        logger.info(
            "Selected held-out PR-AUC=%.4f recall=%.4f",
            final_metrics["pr_auc"],
            final_metrics["recall"],
        )
        logger.info(
            "Saved employee-risk artifact: %s",
            artifacts_dir / "employee_risk_scores.csv",
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
