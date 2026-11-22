"""Execute Phase 7 SHAP explainability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_csv
from src.explainability.shap_analysis import (
    calculate_shap_values,
    clean_feature_name,
    dense_matrix,
    generate_individual_explanation,
    get_transformed_feature_names,
    global_importance,
    initialize_linear_explainer,
    protect_sensitive_features,
    transform_features,
)
from src.features.preprocessing import (
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
    resolve_preprocessing_spec,
)
from src.models.risk_scoring import classify_risk
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 7 SHAP explanations."
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
        "--calibrated-model",
        default="models/phase6_attrition_risk_model.joblib",
    )
    parser.add_argument(
        "--tuned-logistic-model",
        default="models/phase5/logistic_regression_tuned.joblib",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="reports/explainability",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures/phase7",
    )
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--individual-count", type=int, default=5)
    return parser.parse_args()


def save_shap_summary(
    shap_values: np.ndarray,
    transformed: np.ndarray,
    names: list[str],
    figure_path: Path,
) -> None:
    plt.close("all")
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=names,
        show=False,
        plot_type="dot",
        max_display=20,
    )
    plt.gcf().set_size_inches(11, 8)
    plt.title("SHAP Summary: Attrition Propensity")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def save_shap_bar(
    shap_values: np.ndarray,
    names: list[str],
    figure_path: Path,
) -> None:
    mean_absolute = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_absolute)[::-1][:20]
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.barh(
        [names[index] for index in order][::-1],
        [mean_absolute[index] for index in order][::-1],
        color="#2878B5",
    )
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_title("SHAP Feature Importance")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_waterfall(
    explainer,
    shap_values: np.ndarray,
    base_value: float,
    transformed: np.ndarray,
    names: list[str],
    record_index: int,
    figure_path: Path,
) -> None:
    explanation = shap.Explanation(
        values=shap_values[record_index],
        base_values=base_value,
        data=transformed[record_index],
        feature_names=names,
    )
    plt.close("all")
    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False,
    )
    plt.gcf().set_size_inches(11, 7)
    plt.title("Individual SHAP Explanation")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def write_report(
    report_path: Path,
    importance: pd.DataFrame,
    coefficient: pd.DataFrame,
    individual: list[dict[str, object]],
    risk_counts: pd.Series,
) -> None:
    top = importance.iloc[0]
    top_positive = importance.sort_values(
        "average_positive_shap",
        ascending=False,
    ).iloc[0]
    top_negative = importance.sort_values(
        "average_negative_shap",
        ascending=True,
    ).iloc[0]

    overtime_rows = importance[
        importance["feature"].str.lower().str.contains("overtime")
    ]
    tenure_rows = importance[
        importance["feature"].str.lower().str.contains(
            "tenure|years at company|years in current role"
        )
    ]
    satisfaction_rows = importance[
        importance["feature"].apply(
            lambda name: any(
                token in str(name).lower().replace(" ", "")
                for token in (
                    "jobsatisfaction",
                    "engagementcomposite",
                    "worklifebalance",
                )
            )
        )
    ]

    lines = [
        "# SHAP Explainability Report",
        "",
        "## Scope",
        "",
        "- Model: selected tuned Logistic Regression plus isotonic calibration",
        "- SHAP explainer: `shap.LinearExplainer`",
        "- SHAP output: underlying logistic score in log-odds space",
        "- Reported employee probabilities: isotonic-calibrated probability",
        "- Sensitive model fields: Age, Gender, and MaritalStatus excluded",
        "",
        "SHAP explains the fitted linear model score. The isotonic calibrator transforms "
        "that score into a final probability, so calibrated probability values must not be "
        "described as direct SHAP contributions.",
        "",
        "## Global feature findings",
        "",
        f"- Highest mean absolute SHAP feature: **{top['feature']}** "
        f"({top['mean_absolute_shap']:.6f}).",
        f"- Largest average positive contribution: **{top_positive['feature']}** "
        f"({top_positive['average_positive_shap']:.6f}).",
        f"- Largest average negative contribution: **{top_negative['feature']}** "
        f"({top_negative['average_negative_shap']:.6f}).",
        "",
        "These values describe average associations across the explanation data.",
        "They do not establish causal effects or a guaranteed retention action.",
        "",
        "### Overtime",
        "",
    ]
    if overtime_rows.empty:
        lines.append("- No overtime-specific transformed feature was available.")
    else:
        for _, row in overtime_rows.iterrows():
            lines.append(
                f"- **{row['feature']}** mean absolute SHAP: "
                f"{row['mean_absolute_shap']:.6f}; average positive: "
                f"{row['average_positive_shap']:.6f}; average negative: "
                f"{row['average_negative_shap']:.6f}. "
                "This direction is association, not proof that overtime causes attrition."
            )

    lines.extend(["", "### Tenure", ""])
    if tenure_rows.empty:
        lines.append("- No identifiable tenure feature was available.")
    else:
        for _, row in tenure_rows.iterrows():
            lines.append(
                f"- **{row['feature']}** mean absolute SHAP: "
                f"{row['mean_absolute_shap']:.6f}."
            )

    lines.extend(["", "### Job satisfaction", ""])
    if satisfaction_rows.empty:
        lines.append("- No identifiable satisfaction feature was available.")
    else:
        for _, row in satisfaction_rows.iterrows():
            lines.append(
                f"- **{row['feature']}** mean absolute SHAP: "
                f"{row['mean_absolute_shap']:.6f}."
            )

    lines.extend(
        [
            "",
            "## Individual high-risk contributors",
            "",
        ]
    )
    for item in individual:
        lines.append(
            f"- **{item['anonymous_record_id']}**: "
            f"{item['human_readable_explanation']}"
        )

    lines.extend(
        [
            "",
            "## Coefficient comparison",
            "",
            "| Feature | Logistic coefficient | Mean absolute SHAP |",
            "|---|---:|---:|",
        ]
    )
    comparison = coefficient.merge(
        importance,
        left_on="feature",
        right_on="feature",
        how="inner",
    ).sort_values("mean_absolute_shap", ascending=False)
    for _, row in comparison.head(20).iterrows():
        lines.append(
            f"| {row['feature']} | {row['coefficient']:.6f} | "
            f"{row['mean_absolute_shap']:.6f} |"
        )

    lines.extend(
        [
            "",
            "Coefficient and SHAP rankings can differ because coefficients measure "
            "directional model weight in standardized/transformed feature units, while "
            "SHAP mean absolute values measure the magnitude of observed contributions "
            "across the explanation population.",
            "",
            "## Risk category distribution",
            "",
        ]
    )
    for category in ("LOW", "MEDIUM", "HIGH"):
        lines.append(
            f"- {category}: **{int(risk_counts.get(category, 0))}**"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- SHAP associations do not prove causation.",
            "- SHAP is generated for the underlying linear score, not exact isotonic calibrated probabilities.",
            "- The model excludes Age, Gender, and MaritalStatus from operational explanations.",
            "- Encoded one-hot features represent categories and may fragment global importance.",
            "- The 294-record held-out set is limited and not proof of production performance.",
            "- Explanations should support voluntary, human-reviewed retention discussions.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = configure_logging()

    if args.background_size < 2:
        logger.error("Background size must be at least 2.")
        return 1
    if args.individual_count < 1:
        logger.error("Individual count must be at least 1.")
        return 1

    try:
        frame = load_csv(args.csv_path)
        split = joblib.load(args.split_artifact)
        calibrated_model = joblib.load(args.calibrated_model)
        tuned_logistic = joblib.load(args.tuned_logistic_model)

        spec = resolve_preprocessing_spec(frame)
        model_columns = [
            column
            for column in frame.columns
            if column not in spec.excluded_columns
        ]
        train_indices = split["train_indices"]
        test_indices = split["test_indices"]
        x_train = frame.loc[train_indices, model_columns].copy()
        x_test = frame.loc[test_indices, model_columns].copy()

        transformed_train = transform_features(tuned_logistic, x_train)
        transformed_test = transform_features(tuned_logistic, x_test)
        raw_names = get_transformed_feature_names(tuned_logistic)
        names = [clean_feature_name(name) for name in raw_names]

        safe_names = protect_sensitive_features(names)
        if len(safe_names) != len(names):
            raise ValueError("Sensitive feature exposed in transformed feature names.")

        background_size = min(args.background_size, len(transformed_train))
        background = transformed_train[:background_size]
        classifier = tuned_logistic.named_steps["classifier"]
        explainer = initialize_linear_explainer(classifier, background)
        explanation = calculate_shap_values(explainer, transformed_test)

        importance = global_importance(explanation.shap_values, names)
        coefficient = pd.DataFrame(
            {
                "feature": names,
                "coefficient": np.asarray(classifier.coef_).reshape(-1),
            }
        )

        reports_dir = Path(args.reports_dir)
        artifacts_dir = Path(args.artifacts_dir)
        figures_dir = Path(args.figures_dir)
        for directory in (reports_dir, artifacts_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        importance.to_csv(
            artifacts_dir / "shap_global_importance.csv",
            index=False,
        )
        coefficient.to_csv(
            artifacts_dir / "logistic_coefficients.csv",
            index=False,
        )

        save_shap_summary(
            explanation.shap_values,
            transformed_test,
            names,
            figures_dir / "phase7_shap_summary.png",
        )
        save_shap_bar(
            explanation.shap_values,
            names,
            figures_dir / "phase7_shap_feature_importance.png",
        )

        calibrated_probabilities = calibrated_model.predict_proba(x_test)[:, 1]
        risk_categories = classify_risk(calibrated_probabilities)

        individual_results = []
        high_positions = np.argsort(calibrated_probabilities)[::-1]
        selected_positions = high_positions[:args.individual_count]

        for local_index in selected_positions:
            record_id = f"record_{int(test_indices[local_index])}"
            threshold_prediction = int(
                calibrated_probabilities[local_index] >= 0.21
            )
            result = generate_individual_explanation(
                explanation.shap_values[local_index],
                transformed_test[local_index],
                names,
                float(calibrated_probabilities[local_index]),
                threshold_prediction,
                record_id,
            )
            individual_results.append(result)

            save_waterfall(
                explainer,
                explanation.shap_values,
                explanation.base_value,
                transformed_test,
                names,
                int(local_index),
                figures_dir / f"phase7_waterfall_{record_id}.png",
            )

        (artifacts_dir / "individual_explanations.json").write_text(
            json.dumps(individual_results, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(individual_results).to_csv(
            artifacts_dir / "individual_explanations.csv",
            index=False,
        )

        risk_counts = pd.Series(risk_categories).value_counts()
        write_report(
            reports_dir / "SHAP_EXPLAINABILITY_REPORT.md",
            importance,
            coefficient,
            individual_results,
            risk_counts,
        )

        metadata = {
            "phase": 7,
            "shap_explainer": "shap.LinearExplainer",
            "explained_model": "Underlying tuned logistic regression",
            "calibration_model": "Isotonic calibrated logistic regression",
            "calibration_explanation": (
                "SHAP explains the underlying logistic score; it does not "
                "exactly explain isotonic calibrated probability."
            ),
            "background_size": background_size,
            "held_out_explanation_rows": int(len(transformed_test)),
            "transformed_feature_count": int(len(names)),
            "shap_output_shape": list(explanation.shap_values.shape),
            "sensitive_features_excluded": list(SENSITIVE_COLUMNS),
            "individual_count": int(len(individual_results)),
            "risk_category_method": "0.30 / 0.60 thresholds",
        }
        (artifacts_dir / "phase7_shap_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Generated SHAP values with shape %s",
            explanation.shap_values.shape,
        )
        logger.info(
            "Generated %d individual explanations",
            len(individual_results),
        )
        logger.info(
            "Saved SHAP report: %s",
            reports_dir / "SHAP_EXPLAINABILITY_REPORT.md",
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
