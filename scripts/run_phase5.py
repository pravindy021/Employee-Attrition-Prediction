"""Execute Phase 5 bounded hyperparameter optimization."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
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
    calculate_classification_metrics,
    select_promising_models,
)
from src.utils.logging_config import configure_logging


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
}


def build_search_registry(
    negative_count: int,
    positive_count: int,
) -> dict[str, tuple[object, dict[str, object]]]:
    """Build bounded RandomizedSearchCV specifications."""
    registry: dict[str, tuple[object, dict[str, object]]] = {
        "logistic_regression": (
            LogisticRegression(
                max_iter=3000,
                random_state=42,
                solver="liblinear",
            ),
            {
                "classifier__C": loguniform(0.01, 100),
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(
                random_state=42,
            ),
            {
                "classifier__learning_rate": uniform(0.03, 0.15),
                "classifier__max_iter": randint(80, 220),
                "classifier__max_leaf_nodes": randint(15, 45),
                "classifier__l2_regularization": uniform(0.0, 1.0),
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }

    try:
        from xgboost import XGBClassifier
    except (ImportError, OSError):
        return registry

    scale_pos_weight = (
        negative_count / positive_count if positive_count else 1.0
    )
    registry["xgboost"] = (
        XGBClassifier(
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
        {
            "classifier__n_estimators": randint(150, 351),
            "classifier__learning_rate": uniform(0.03, 0.12),
            "classifier__max_depth": randint(3, 8),
            "classifier__min_child_weight": randint(1, 5),
            "classifier__subsample": uniform(0.75, 0.25),
            "classifier__colsample_bytree": uniform(0.75, 0.25),
            "classifier__reg_alpha": uniform(0.0, 1.0),
            "classifier__reg_lambda": uniform(0.5, 2.0),
            "classifier__scale_pos_weight": [
                1.0,
                scale_pos_weight,
            ],
        },
    )
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 hyperparameter optimization."
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
        "--baseline-comparison",
        default="reports/model_comparison.csv",
    )
    parser.add_argument(
        "--models-dir",
        default="models/phase5",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_optimization_report(
    report_path: Path,
    tuned_comparison: pd.DataFrame,
    candidate_names: list[str],
    iterations: int,
) -> None:
    """Write the measured Phase 5 baseline-versus-tuned report."""
    best_overall = tuned_comparison.sort_values(
        ["tuned_pr_auc", "tuned_recall", "tuned_roc_auc"],
        ascending=False,
    ).iloc[0]

    lines = [
        "# Phase 5 Hyperparameter Optimization",
        "",
        "## Scope",
        "",
        "- Data partition: 1,176-record training partition only",
        "- Held-out test partition: untouched",
        "- Search method: RandomizedSearchCV",
        f"- Iterations per candidate: {iterations}",
        "- Cross-validation: 5-fold stratified",
        "- Primary optimization metric: PR-AUC",
        "- Secondary metrics: recall, ROC-AUC, F1, balanced accuracy",
        "",
        "## Candidate models",
        "",
    ]
    lines.extend([f"- {name}" for name in candidate_names])

    lines.extend(
        [
            "",
            "These candidates were selected from the Phase 4 PR-AUC ranking.",
            "",
            "## Baseline and tuned comparison",
            "",
            "| Model | Baseline PR-AUC | Tuned PR-AUC | Baseline recall | Tuned recall | Baseline ROC-AUC | Tuned ROC-AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in tuned_comparison.iterrows():
        lines.append(
            f"| {row['model']} | {row['baseline_pr_auc']:.4f} | "
            f"{row['tuned_pr_auc']:.4f} | {row['baseline_recall']:.4f} | "
            f"{row['tuned_recall']:.4f} | {row['baseline_roc_auc']:.4f} | "
            f"{row['tuned_roc_auc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Best tuned candidate",
            "",
            f"- Model: **{best_overall['model']}**",
            f"- Tuned PR-AUC: **{best_overall['tuned_pr_auc']:.4f}**",
            f"- Tuned recall: **{best_overall['tuned_recall']:.4f}**",
            f"- Baseline PR-AUC: **{best_overall['baseline_pr_auc']:.4f}**",
            f"- Best parameters: `{json.dumps(best_overall['best_parameters'], sort_keys=True)}`",
            "",
            "## Leakage controls",
            "",
            "- Search folds use only the training partition.",
            "- Feature engineering, imputation, scaling, and encoding are inside the pipeline.",
            "- The held-out test partition is not used for fitting or cross-validation.",
            "- Sensitive fields remain excluded from model inputs.",
            "- SMOTE remains disabled; class imbalance is handled by class weighting or XGBoost weighting.",
            "",
            "## Limitations",
            "",
            "- Randomized search samples a bounded subset of the possible parameter space.",
            "- Tuned cross-validation metrics are not held-out test metrics.",
            "- Probability calibration and threshold optimization occur in later phases.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = configure_logging()

    if args.iterations < 1 or args.folds < 2:
        logger.error("Iterations must be positive and folds must be at least 2.")
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

        baseline = pd.read_csv(args.baseline_comparison)
        promising = select_promising_models(
            baseline,
            limit=3,
        )
        candidate_names = promising["model"].tolist()
        baseline_by_name = baseline.set_index("model").to_dict(orient="index")

        class_counts = y_train.value_counts()
        search_registry = build_search_registry(
            negative_count=int(class_counts.get(0, 0)),
            positive_count=int(class_counts.get(1, 0)),
        )

        cv = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.random_state,
        )
        models_dir = Path(args.models_dir)
        reports_dir = Path(args.reports_dir)
        for directory in (models_dir, reports_dir):
            directory.mkdir(parents=True, exist_ok=True)

        tuned_rows = []
        all_search_results = []

        for name in candidate_names:
            if name not in search_registry:
                continue
            classifier, parameter_grid = search_registry[name]
            pipeline = Pipeline(
                steps=[
                    ("preprocessing", build_preprocessing_pipeline(spec)),
                    ("classifier", classifier),
                ]
            )
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=parameter_grid,
                n_iter=args.iterations,
                scoring=SCORING,
                refit="pr_auc",
                cv=cv,
                n_jobs=-1,
                random_state=args.random_state,
                return_train_score=False,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UndefinedMetricWarning,
                )
                search.fit(x_train, y_train)

            tuned_pipeline = search.best_estimator_
            y_pred = cross_val_predict(
                tuned_pipeline,
                x_train,
                y_train,
                cv=cv,
                method="predict",
                n_jobs=-1,
            )
            y_probability = cross_val_predict(
                tuned_pipeline,
                x_train,
                y_train,
                cv=cv,
                method="predict_proba",
                n_jobs=-1,
            )[:, 1]

            tuned_metrics = calculate_classification_metrics(
                y_train,
                y_pred,
                y_probability,
            )
            baseline_metrics = baseline_by_name[name]
            tuned_row = {
                "model": name,
                "baseline_pr_auc": float(baseline_metrics["pr_auc"]),
                "tuned_pr_auc": tuned_metrics["pr_auc"],
                "baseline_recall": float(baseline_metrics["recall"]),
                "tuned_recall": tuned_metrics["recall"],
                "baseline_roc_auc": float(baseline_metrics["roc_auc"]),
                "tuned_roc_auc": tuned_metrics["roc_auc"],
                "baseline_f1": float(baseline_metrics["f1"]),
                "tuned_f1": tuned_metrics["f1"],
                "baseline_accuracy": float(baseline_metrics["accuracy"]),
                "tuned_accuracy": tuned_metrics["accuracy"],
                "baseline_precision": float(baseline_metrics["precision"]),
                "tuned_precision": tuned_metrics["precision"],
                "baseline_balanced_accuracy": float(
                    baseline_metrics["balanced_accuracy"]
                ),
                "tuned_balanced_accuracy": tuned_metrics["balanced_accuracy"],
                "baseline_specificity": float(
                    baseline_metrics["specificity"]
                ),
                "tuned_specificity": tuned_metrics["specificity"],
                "best_parameters": search.best_params_,
            }
            tuned_rows.append(tuned_row)

            for parameter, value in search.best_params_.items():
                all_search_results.append(
                    {
                        "model": name,
                        "parameter": parameter,
                        "value": str(value),
                    }
                )

            tuned_pipeline.fit(x_train, y_train)
            joblib.dump(
                tuned_pipeline,
                models_dir / f"{name}_tuned.joblib",
            )

            logger.info(
                "%s: baseline PR-AUC=%.4f tuned PR-AUC=%.4f",
                name,
                baseline_metrics["pr_auc"],
                tuned_metrics["pr_auc"],
            )

        if not tuned_rows:
            logger.error("No Phase 4 candidates could be optimized.")
            return 1

        tuned_comparison = pd.DataFrame(tuned_rows)
        tuned_comparison.to_csv(
            reports_dir / "tuned_model_comparison.csv",
            index=False,
        )
        pd.DataFrame(all_search_results).to_csv(
            reports_dir / "hyperparameter_results.csv",
            index=False,
        )

        metadata = {
            "phase": 5,
            "random_state": args.random_state,
            "iterations_per_candidate": args.iterations,
            "cross_validation_folds": args.folds,
            "train_row_count": int(len(x_train)),
            "test_row_count": 294,
            "test_set_evaluated": False,
            "candidate_models": candidate_names,
            "primary_metric": "average_precision",
            "smote_applied": False,
            "sensitive_columns_excluded": SENSITIVE_COLUMNS,
            "models": tuned_rows,
        }
        (reports_dir / "phase5_optimization_results.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )
        write_optimization_report(
            reports_dir / "phase5_hyperparameter_report.md",
            tuned_comparison,
            candidate_names,
            args.iterations,
        )

        logger.info(
            "Saved tuned models under %s",
            models_dir,
        )
        logger.info(
            "Saved tuned model comparison: %s",
            reports_dir / "tuned_model_comparison.csv",
        )
        logger.info(
            "Saved optimization report: %s",
            reports_dir / "phase5_hyperparameter_report.md",
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
