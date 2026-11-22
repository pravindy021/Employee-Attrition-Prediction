"""Execute Phase 8 fairness and responsible-AI audit."""

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
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_csv
from src.fairness.fairness_metrics import (
    fairness_summary,
    group_metrics,
)
from src.features.preprocessing import TARGET_COLUMN
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 8 demographic subgroup audit."
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
        "--final-model",
        default="models/phase6_attrition_risk_model.joblib",
    )
    parser.add_argument(
        "--risk-config",
        default="data/processed/phase6_risk_scoring_config.joblib",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures/phase8",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )
    return parser.parse_args()


def age_group(age) -> str:
    """Convert an age value into a documented audit band."""
    if pd.isna(age):
        return "Missing"
    age = float(age)
    if age < 31:
        return "18-30"
    if age < 41:
        return "31-40"
    if age < 51:
        return "41-50"
    return "51+"


def save_group_chart(
    table: pd.DataFrame,
    audit_name: str,
    metric: str,
    title: str,
    figure_path: Path,
) -> None:
    """Save a single group-wise metric chart."""
    fig, axis = plt.subplots(figsize=(10, 6))
    data = table.sort_values(metric, ascending=False)
    sns.barplot(
        data=data,
        x="group",
        y=metric,
        hue="group",
        palette="viridis",
        legend=False,
        ax=axis,
    )
    axis.set_title(f"{audit_name}: {title}")
    axis.set_xlabel(audit_name)
    axis.set_ylabel(title)
    axis.tick_params(axis="x", rotation=30)
    axis.set_ylim(0, max(1.0, float(data[metric].max()) * 1.15))
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fairness_summary(
    summary_tables: dict[str, pd.DataFrame],
    figure_path: Path,
) -> None:
    """Save demographic parity and equal opportunity differences."""
    rows = []
    for audit_name, table in summary_tables.items():
        summary = fairness_summary(table)
        rows.append(
            {
                "audit": audit_name,
                "demographic_parity_difference": summary[
                    "demographic_parity_difference"
                ],
                "equal_opportunity_difference": summary[
                    "equal_opportunity_difference"
                ],
            }
        )
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(
        data=summary,
        x="audit",
        y="demographic_parity_difference",
        hue="audit",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Demographic Parity Difference")
    axes[0].set_ylabel("Maximum-minus-minimum selection rate")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=summary,
        x="audit",
        y="equal_opportunity_difference",
        hue="audit",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Equal Opportunity Difference")
    axes[1].set_ylabel("Maximum-minus-minimum recall")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_fairness_report(
    report_path: Path,
    audit_tables: dict[str, pd.DataFrame],
    threshold: float,
) -> None:
    """Write the measured Phase 8 fairness report."""
    lines = [
        "# Fairness Audit Report",
        "",
        "## Scope",
        "",
        "- Held-out test records: **294**",
        "- Operational model inputs: Age, Gender, and MaritalStatus excluded",
        "- Demographic fields: used only for subgroup audit",
        f"- Classification threshold: **{threshold:.2f}**",
        "- Outcome: historical Attrition target",
        "",
        "This audit describes model performance differences across observed groups.",
        "It does not prove discrimination or establish causal bias.",
        "",
        "## Metric definitions",
        "",
        "- Selection/prediction rate: positive predictions divided by subgroup size.",
        "- TPR/recall: positive actuals correctly predicted positive.",
        "- FPR: negative actuals incorrectly predicted positive.",
        "- FNR: positive actuals incorrectly predicted negative.",
        "- Precision: positive predictions that were actual positives.",
        "- Demographic parity difference: largest minus smallest selection rate.",
        "- Equal opportunity difference: largest minus smallest TPR/recall.",
        "",
        "Equal opportunity differences include only groups with at least 20 records "
        "and at least one actual positive in this implementation.",
        "",
    ]

    for audit_name, table in audit_tables.items():
        summary = fairness_summary(table)
        lines.extend(
            [
                f"## {audit_name}",
                "",
                "| Group | Size | Selection rate | Recall/TPR | FPR | FNR | Precision |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in table.sort_values("sample_size", ascending=False).iterrows():
            lines.append(
                f"| {row['group']} | {int(row['sample_size'])} | "
                f"{row['selection_rate']:.4f} | {row['tpr_recall']:.4f} | "
                f"{row['fpr']:.4f} | {row['fnr']:.4f} | "
                f"{row['precision']:.4f} |"
            )

        lines.extend(
            [
                "",
                f"- Demographic parity difference: "
                f"**{summary['demographic_parity_difference']:.4f}**",
                f"- Equal opportunity difference: "
                f"**{summary['equal_opportunity_difference']:.4f}**",
                f"- False-positive-rate difference: "
                f"**{summary['false_positive_rate_difference']:.4f}**",
                f"- Precision difference: "
                f"**{summary['precision_difference']:.4f}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Observed findings",
            "",
            "The findings above are descriptive only. They summarize the held-out "
            "subset and do not indicate whether group differences are statistically "
            "significant or caused by model bias.",
            "",
            "## Bias investigation",
            "",
            "- Compare false-positive and false-negative rates before considering action.",
            "- Small subgroups can produce unstable percentages.",
            "- Missing demographic values may indicate data-quality or reporting issues.",
            "- Protected attributes are excluded from the model; proxy variables were not "
            "formally audited for indirect discrimination.",
            "- Feature importance and group performance do not establish causal discrimination.",
            "",
            "## Responsible-AI recommendations",
            "",
            "- Do not automate termination or opportunity decisions.",
            "- Require human review and contextual employee information.",
            "- Use supportive and voluntary retention conversations.",
            "- Recalculate subgroup metrics for future datasets and update thresholds only after documented review.",
            "- Report uncertainty and small-sample limitations.",
            "- Continue to audit indirect proxy effects with appropriate legal and data-science review.",
            "",
            "## Limitations",
            "",
            "- The 294-record holdout is small for demographic subgroup analysis.",
            "- Age bands are post hoc categories and may conceal within-band differences.",
            "- Missing demographic categories should not be inferred.",
            "- Historical outcomes may reflect organizational bias.",
            "- A single threshold can distribute errors unevenly across groups.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = configure_logging()
    sns.set_theme(style="whitegrid")

    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        logger.error("Threshold must be between 0 and 1.")
        return 1

    try:
        frame = load_csv(args.csv_path)
        split = joblib.load(args.split_artifact)
        final_model = joblib.load(args.final_model)
        risk_config = joblib.load(args.risk_config)

        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else float(risk_config["threshold"])
        )
        test_indices = split["test_indices"]
        test_frame = frame.loc[test_indices].copy()
        y_test = test_frame[TARGET_COLUMN].astype("int64").to_numpy()

        spec = __import__(
            "src.features.preprocessing",
            fromlist=["resolve_preprocessing_spec"],
        ).resolve_preprocessing_spec(frame)
        model_columns = [
            column
            for column in frame.columns
            if column not in spec.excluded_columns
        ]
        x_test = test_frame[model_columns].copy()
        probabilities = final_model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= threshold).astype("int64")

        audit_definitions = {
            "AgeGroup": test_frame["Age"].apply(age_group),
            "Gender": test_frame["Gender"].fillna("Missing"),
            "MaritalStatus": test_frame["MaritalStatus"].fillna("Missing"),
        }

        audit_tables = {}
        combined_rows = []
        for audit_name, groups in audit_definitions.items():
            table = group_metrics(
                y_true=y_test,
                y_pred=predictions,
                probabilities=probabilities,
                groups=groups,
            )
            table.insert(0, "audit", audit_name)
            audit_tables[audit_name] = table

            summary = fairness_summary(table)
            for _, row in table.iterrows():
                combined_rows.append(
                    {
                        "audit": audit_name,
                        **row.to_dict(),
                        **summary,
                    }
                )

        reports_dir = Path(args.reports_dir)
        figures_dir = Path(args.figures_dir)
        for directory in (reports_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        demographic = pd.DataFrame(combined_rows)
        demographic.to_csv(
            reports_dir / "DEMOGRAPHIC_METRICS.csv",
            index=False,
        )

        for audit_name, table in audit_tables.items():
            save_group_chart(
                table,
                audit_name,
                "selection_rate",
                "Selection/prediction rate",
                figures_dir / f"phase8_{audit_name.lower()}_selection.png",
            )
            save_group_chart(
                table,
                audit_name,
                "fpr",
                "False-positive rate",
                figures_dir / f"phase8_{audit_name.lower()}_fpr.png",
            )
            save_group_chart(
                table,
                audit_name,
                "tpr_recall",
                "Recall/true-positive rate",
                figures_dir / f"phase8_{audit_name.lower()}_recall.png",
            )

        save_fairness_summary(
            audit_tables,
            figures_dir / "phase8_fairness_metric_summary.png",
        )
        write_fairness_report(
            reports_dir / "FAIRNESS_AUDIT_REPORT.md",
            audit_tables,
            threshold,
        )

        guidelines = """# Responsible AI Guidelines

## Prediction use

- Attrition scores are estimates of historical propensity, not facts.
- The model does not provide a 30/60/90-day resignation prediction.
- The system must not automate termination, demotion, or opportunity denial.
- Human reviewers must consider job context and alternative explanations.

## Supportive interventions

- Retention discussions must be voluntary and supportive.
- Managers must not use risk scores as proof of dissatisfaction or disloyalty.
- Organizational interventions should address workload, development, pay
  fairness, and role fit only where lawful and appropriate.

## Demographic data

- Age, Gender, and MaritalStatus are audit-only fields.
- They must not be exposed as operational explanation reasons.
- Demographic data must not be used to recommend adverse employment actions.
- Subgroup metrics require privacy-aware aggregation and secure handling.

## Governance

- Revalidate performance after data, policy, or workforce changes.
- Document threshold changes and their subgroup effects.
- Escalate unexplained subgroup differences to HR, legal, and Responsible AI review.
- Preserve human oversight and provide a route for employee questions.

## Monitoring

- Monitor data drift, probability drift, and outcome drift.
- Recalculate fairness metrics when outcomes become available.
- Small subgroup sizes require uncertainty and result suppression review.
- Do not interpret a fairness metric difference as proof of unlawful discrimination.
"""
        (reports_dir / "RESPONSIBLE_AI_GUIDELINES.md").write_text(
            guidelines,
            encoding="utf-8",
        )

        audit_metadata = {
            "phase": 8,
            "test_row_count": int(len(test_frame)),
            "threshold": threshold,
            "model_artifact": args.final_model,
            "sensitive_fields_audit_only": [
                "Age",
                "Gender",
                "MaritalStatus",
            ],
            "operational_sensitive_fields_excluded": True,
            "age_bands": ["18-30", "31-40", "41-50", "51+", "Missing"],
            "audit_scope": [
                "AgeGroup",
                "Gender",
                "MaritalStatus",
            ],
            "interpretation": (
                "Subgroup performance differences are descriptive; "
                "they do not establish discrimination."
            ),
        }
        (reports_dir / "phase8_fairness_metadata.json").write_text(
            json.dumps(audit_metadata, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Audited %d held-out records across Age, Gender, and MaritalStatus.",
            len(test_frame),
        )
        logger.info(
            "Wrote fairness report: %s",
            reports_dir / "FAIRNESS_AUDIT_REPORT.md",
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
