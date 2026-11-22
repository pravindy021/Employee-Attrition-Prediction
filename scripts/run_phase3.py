"""Execute Phase 3 feature engineering and preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_csv
from src.features.feature_engineering import engineer_features
from src.features.preprocessing import (
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    resolve_preprocessing_spec,
)
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 preprocessing and artifact generation."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/Palo Alto Networks.csv",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
    )
    parser.add_argument(
        "--report",
        default="reports/preprocessing_report.md",
    )
    parser.add_argument(
        "--metadata",
        default="reports/preprocessing_metadata.json",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_report(
    report_path: Path,
    frame: pd.DataFrame,
    spec,
    x_train,
    x_test,
) -> None:
    """Write a human-readable Phase 3 decision report."""
    lines = [
        "# Feature Engineering and Preprocessing Report",
        "",
        "## Scope",
        "",
        "This report documents Phase 3 only. No model was trained and no dashboard was built.",
        "The original CSV was not modified.",
        "",
        "## Dataset split",
        "",
        f"- Source records: **{len(frame):,}**",
        f"- Training records: **{len(x_train):,}**",
        f"- Test records: **{len(x_test):,}**",
        "- Split method: **stratified train-test split**",
        "- Test fraction: **20%**",
        "- Random seed: **42**",
        "",
        "## Target handling",
        "",
        "- Target: `Attrition`",
        "- Class 0: no attrition",
        "- Class 1: attrition",
        "- Initial class-imbalance policy: **balanced class weights**",
        "- SMOTE: **not applied**",
        "",
        "SMOTE will be considered only later inside cross-validation folds, where it can be "
        "isolated from validation data. The EDA majority/minority ratio is 5.20:1.",
        "",
        "## Sensitive feature separation",
        "",
        "- Audit-only fields: `Age`, `Gender`, `MaritalStatus`",
        "- These fields are excluded from the model input matrix.",
        "- They may be used separately for fairness auditing where legally and ethically appropriate.",
        "- They must not be used to recommend adverse employment actions.",
        "",
        "## Excluded columns",
        "",
        f"- Identifier columns: **{', '.join(spec.identifier_columns) or 'none detected'}**",
        f"- Constant columns: **{', '.join(spec.constant_columns) or 'none detected'}**",
        f"- All excluded columns: **{', '.join(spec.excluded_columns)}**",
        "",
        "## Numeric preprocessing",
        "",
        "- Missing values: median imputation",
        "- Scaling: standardization with `StandardScaler`",
        "- Numeric features: `"
        + "`, `".join(spec.numeric_columns)
        + "`",
        "",
        "## Categorical preprocessing",
        "",
        "- Missing values: most-frequent imputation",
        "- Encoding: one-hot encoding",
        "- Unknown categories: ignored at transform time",
        "- Categorical features: `"
        + "`, `".join(spec.categorical_columns)
        + "`",
        "",
        "## Business-oriented engineered features",
        "",
        "| Feature | Definition | Business rationale | Limitation |",
        "|---|---|---|---|",
        "| IncomePerTotalWorkingYear | MonthlyIncome / (TotalWorkingYears + 1) | Approximate income relative to career length. | Ratios can be unstable for very short careers. |",
        "| PromotionDelayRatio | YearsSinceLastPromotion / (YearsAtCompany + 1) | Approximate share of tenure without promotion. | A delay is not necessarily the result of poor performance or unfair treatment. |",
        "| RoleTenureRatio | YearsInCurrentRole / (YearsAtCompany + 1) | Approximate proportion of tenure spent in the current role. | Organizational roles and promotion definitions vary. |",
        "| OvertimeFlag | 1 when OverTime is Yes, otherwise 0 | Overtime had a strong observed attrition association in EDA. | Association is not evidence of causation. |",
        "| EngagementComposite | Mean of five satisfaction/involvement ratings | Summarizes related experience measures. | The score assumes approximately comparable ordinal scales. |",
        "",
        "## Leakage prevention",
        "",
        "- The data is split before fitting the preprocessing pipeline.",
        "- The feature engineer and all imputers, scalers, and encoders are fit on training data only.",
        "- The test set is only transformed after fitting.",
        "- The target column is never included in model inputs.",
        "- Sensitive fields are withheld from model inputs and stored separately for auditing.",
        "",
        "## Reproducibility",
        "",
        "- Fixed random seed: 42",
        "- Fixed test fraction: 20%",
        "- Stratified split preserves target proportions.",
        "- The saved sklearn pipeline is serializable and reusable for unseen records.",
        "",
        "## Limitations",
        "",
        "- The engineered engagement score is exploratory rather than a clinically or scientifically validated measure.",
        "- Correlations do not establish causal retention drivers.",
        "- Protected attributes may reproduce organizational bias if used inappropriately.",
        "- No transformed output should be interpreted as an individual employment decision.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = configure_logging()

    if not 0.0 < args.test_size < 1.0:
        logger.error("Test size must be between 0 and 1.")
        return 1

    try:
        frame = load_csv(args.csv_path)
        if TARGET_COLUMN not in frame.columns:
            logger.error("Required target column '%s' is missing.", TARGET_COLUMN)
            return 1
        if frame[TARGET_COLUMN].isna().any():
            logger.error("Target column contains missing values.")
            return 1

        spec = resolve_preprocessing_spec(frame)
        model_columns = [
            column
            for column in frame.columns
            if column not in spec.excluded_columns
        ]

        x = frame[model_columns].copy()
        y = frame[TARGET_COLUMN].astype("int64").copy()
        indices = frame.index

        train_indices, test_indices = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        x_train = x.loc[train_indices].copy()
        x_test = x.loc[test_indices].copy()
        y_train = y.loc[train_indices].copy()
        y_test = y.loc[test_indices].copy()

        pipeline = build_preprocessing_pipeline(spec)
        pipeline.fit(x_train)
        x_train_transformed = pipeline.transform(x_train)
        x_test_transformed = pipeline.transform(x_test)

        processed_dir = Path(args.processed_dir)
        models_dir = Path(args.models_dir)
        report_path = Path(args.report)
        metadata_path = Path(args.metadata)
        for directory in (
            processed_dir,
            models_dir,
            report_path.parent,
            metadata_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, models_dir / "preprocessing_pipeline.joblib")
        joblib.dump(
            {
                "x_train": x_train_transformed,
                "x_test": x_test_transformed,
                "y_train": y_train,
                "y_test": y_test,
                "train_indices": train_indices,
                "test_indices": test_indices,
            },
            processed_dir / "phase3_split.joblib",
        )
        joblib.dump(
            {
                "train": frame.loc[train_indices, SENSITIVE_COLUMNS].copy(),
                "test": frame.loc[test_indices, SENSITIVE_COLUMNS].copy(),
                "train_indices": train_indices,
                "test_indices": test_indices,
            },
            processed_dir / "fairness_audit_features.joblib",
        )

        metadata = {
            "random_state": args.random_state,
            "test_size": args.test_size,
            "target_column": TARGET_COLUMN,
            "source_row_count": int(len(frame)),
            "train_row_count": int(len(x_train)),
            "test_row_count": int(len(x_test)),
            "train_target_distribution": y_train.value_counts().to_dict(),
            "test_target_distribution": y_test.value_counts().to_dict(),
            "numeric_columns": list(spec.numeric_columns),
            "categorical_columns": list(spec.categorical_columns),
            "excluded_columns": list(spec.excluded_columns),
            "identifier_columns": list(spec.identifier_columns),
            "constant_columns": list(spec.constant_columns),
            "sensitive_audit_columns": SENSITIVE_COLUMNS,
            "engineered_features": [
                "IncomePerTotalWorkingYear",
                "PromotionDelayRatio",
                "RoleTenureRatio",
                "OvertimeFlag",
                "EngagementComposite",
            ],
            "class_weight_strategy": "balanced",
            "smote_applied": False,
            "preprocessing_artifacts": [
                str(models_dir / "preprocessing_pipeline.joblib"),
                str(processed_dir / "phase3_split.joblib"),
                str(processed_dir / "fairness_audit_features.joblib"),
            ],
            "train_only_fitted": True,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        build_report(report_path, frame, spec, x_train, x_test)

        logger.info("Created stratified train/test split: %d/%d", len(x_train), len(x_test))
        logger.info("Saved preprocessing pipeline: %s", models_dir / "preprocessing_pipeline.joblib")
        logger.info("Saved processed split: %s", processed_dir / "phase3_split.joblib")
        logger.info("Saved fairness audit fields: %s", processed_dir / "fairness_audit_features.joblib")
        logger.info("Wrote preprocessing report: %s", report_path)
        logger.info("Wrote preprocessing metadata: %s", metadata_path)
        return 0
    except (OSError, ValueError, TypeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

