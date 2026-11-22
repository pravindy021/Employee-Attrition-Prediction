"""Execute Phase 2 exploratory data analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.eda import (
    attrition_rate_table,
    cramers_v,
    mann_whitney_result,
    numeric_outliers,
)
from src.data.load_data import load_csv
from src.utils.logging_config import configure_logging


PALETTE = {"stayed": "#2878B5", "left": "#D95F02"}
TARGET_COLORS = [PALETTE["stayed"], PALETTE["left"]]


def save_figure(figures_dir: Path, name: str) -> None:
    """Save and close the current chart consistently."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_univariate_charts(
    frame: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Create distribution charts for the target and major feature groups."""
    ax = sns.countplot(
        data=frame,
        x="Attrition",
        hue="Attrition",
        palette=TARGET_COLORS,
        legend=False,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stayed (0)", "Attrition (1)"])
    ax.set_title("Employee Attrition Class Distribution")
    ax.set_xlabel("Historical outcome")
    ax.set_ylabel("Employee count")
    for container in ax.containers:
        ax.bar_label(container)
    save_figure(figures_dir, "01_target_distribution.png")

    numerical = [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "TotalWorkingYears",
        "DistanceFromHome",
        "JobSatisfaction",
    ]
    frame[numerical].hist(
        bins=20,
        figsize=(14, 9),
        color="#2878B5",
        edgecolor="white",
    )
    plt.suptitle("Univariate Numerical Distributions", fontsize=16)
    save_figure(figures_dir, "02_numerical_distributions.png")

    categorical = [
        "Gender",
        "Department",
        "JobRole",
        "OverTime",
        "MaritalStatus",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    axes = axes.ravel()
    for axis, column in zip(axes, categorical):
        sns.countplot(
            data=frame,
            y=column,
            hue=column,
            order=frame[column].value_counts().index,
            legend=False,
            ax=axis,
        )
        axis.set_title(column)
        axis.set_xlabel("Employee count")
        axis.set_ylabel("")
    axes[-1].axis("off")
    fig.suptitle("Univariate Categorical Distributions", fontsize=16)
    save_figure(figures_dir, "03_categorical_distributions.png")


def generate_bivariate_charts(
    frame: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Create target comparison charts and calculated rate tables."""
    rate_tables = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(
        data=frame,
        x="Attrition",
        y="Age",
        hue="Attrition",
        palette=TARGET_COLORS,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Stayed", "Attrition"])
    axes[0].set_title("Age vs Attrition")
    sns.boxplot(
        data=frame,
        x="Attrition",
        y="MonthlyIncome",
        hue="Attrition",
        palette=TARGET_COLORS,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Stayed", "Attrition"])
    axes[1].set_title("Monthly Income vs Attrition")
    save_figure(figures_dir, "04_age_income_vs_attrition.png")

    categorical_columns = [
        "Gender",
        "Department",
        "JobRole",
        "JobLevel",
        "OverTime",
        "JobSatisfaction",
        "WorkLifeBalance",
    ]
    for index, column in enumerate(categorical_columns, start=1):
        table = attrition_rate_table(frame, column)
        rate_tables[column] = table
        fig, axis = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=table,
            x=column,
            y="attrition_rate_percent",
            hue=column,
            palette="viridis",
            legend=False,
            ax=axis,
        )
        axis.set_title(f"{column}: Attrition Rate")
        axis.set_ylabel("Attrition rate (%)")
        axis.set_xlabel(column)
        axis.tick_params(axis="x", rotation=35)
        for patch, value in zip(axis.patches, table["attrition_rate_percent"]):
            axis.annotate(
                f"{value:.1f}%",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        save_figure(figures_dir, f"{4 + index:02d}_{column.lower()}_attrition_rate.png")

    tenure_bins = pd.cut(
        frame["YearsAtCompany"],
        bins=[-1, 2, 5, 10, 20, frame["YearsAtCompany"].max()],
        labels=["0-2", "3-5", "6-10", "11-20", "21+"],
    )
    tenure_table = attrition_rate_table(
        frame.assign(YearsAtCompanyBand=tenure_bins),
        "YearsAtCompanyBand",
    )
    rate_tables["YearsAtCompanyBand"] = tenure_table
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=tenure_table,
        x="YearsAtCompanyBand",
        y="attrition_rate_percent",
        hue="YearsAtCompanyBand",
        palette="magma",
        legend=False,
        ax=axis,
    )
    axis.set_title("Years at Company: Attrition Rate")
    axis.set_xlabel("Years at company band")
    axis.set_ylabel("Attrition rate (%)")
    save_figure(figures_dir, "12_tenure_attrition_rate.png")

    return rate_tables


def generate_multivariate_charts(
    frame: pd.DataFrame,
    figures_dir: Path,
) -> pd.DataFrame:
    """Create correlation and grouped multivariate visualizations."""
    numeric = frame.select_dtypes(include="number")
    correlation = numeric.corr()

    fig, axis = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.3,
        ax=axis,
    )
    axis.set_title("Numerical Feature Correlation Matrix")
    save_figure(figures_dir, "13_correlation_heatmap.png")

    satisfaction_table = (
        frame.groupby(["JobSatisfaction", "WorkLifeBalance"], observed=True)["Attrition"]
        .agg(["size", "mean"])
        .rename(columns={"size": "employee_count", "mean": "attrition_rate"})
        .reset_index()
    )
    satisfaction_pivot = satisfaction_table.pivot(
        index="JobSatisfaction",
        columns="WorkLifeBalance",
        values="attrition_rate",
    )
    fig, axis = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        satisfaction_pivot,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        ax=axis,
    )
    axis.set_title("Attrition Rate by Job and Life Satisfaction")
    save_figure(figures_dir, "14_satisfaction_attrition_heatmap.png")

    fig, axis = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=frame,
        x="TotalWorkingYears",
        y="MonthlyIncome",
        hue="Attrition",
        alpha=0.55,
        palette=TARGET_COLORS,
        ax=axis,
    )
    axis.set_title("Experience, Income, and Historical Attrition")
    save_figure(figures_dir, "15_experience_income_attrition.png")

    department_role = (
        frame.groupby(["Department", "JobRole"], observed=True)["Attrition"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "attrition_count", "mean": "attrition_rate"})
    )
    department_role["attrition_rate_percent"] = department_role["attrition_rate"] * 100
    fig, axis = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=department_role.sort_values("attrition_rate_percent", ascending=False),
        x="attrition_rate_percent",
        y="JobRole",
        hue="Department",
        dodge=False,
        ax=axis,
    )
    axis.set_title("Department and Job-Role Attrition Rates")
    axis.set_xlabel("Attrition rate (%)")
    axis.set_ylabel("Job role")
    save_figure(figures_dir, "16_department_role_attrition.png")

    return correlation


def build_eda_report(
    frame: pd.DataFrame,
    report_path: Path,
    rate_tables: dict[str, pd.DataFrame],
    correlation: pd.DataFrame,
) -> None:
    """Generate an executive and analytical Markdown report."""
    attrition_count = int(frame["Attrition"].sum())
    stayed_count = int((frame["Attrition"] == 0).sum())
    attrition_rate = attrition_count / len(frame) * 100

    highest_department = attrition_rate_table(frame, "Department").iloc[0]
    highest_role = attrition_rate_table(frame, "JobRole").iloc[0]
    overtime_table = rate_tables["OverTime"]
    overtime_yes = overtime_table.loc[overtime_table["OverTime"] == "Yes"].iloc[0]
    overtime_no = overtime_table.loc[overtime_table["OverTime"] == "No"].iloc[0]

    target_correlations = (
        correlation["Attrition"]
        .drop(labels=["Attrition"])
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    top_correlations = target_correlations.head(8)

    outliers = numeric_outliers(frame)
    categorical_tests = []
    for column in ["Gender", "Department", "JobRole", "OverTime", "JobSatisfaction", "WorkLifeBalance"]:
        categorical_tests.append(cramers_v(frame, column))

    numerical_tests = []
    for column in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"]:
        numerical_tests.append(mann_whitney_result(frame, column))

    lines = [
        "# Exploratory Data Analysis Report",
        "",
        "## Scope and interpretation",
        "",
        "This report summarizes observed patterns in the supplied dataset. "
        "Associations are predictive and exploratory; they do not establish causation.",
        "",
        "## 1. Target and class imbalance",
        "",
        f"- Total employees: **{len(frame):,}**",
        f"- Historical attrition (1): **{attrition_count:,}**",
        f"- No attrition (0): **{stayed_count:,}**",
        f"- Historical attrition rate: **{attrition_rate:.2f}%**",
        f"- Majority/minority ratio: **{stayed_count / attrition_count:.2f}:1**",
        "",
        "The target is moderately imbalanced. Accuracy alone will be insufficient "
        "during later modeling; recall, PR-AUC, F1, balanced accuracy, and false-positive cost should be considered.",
        "",
        "## 2. Key business findings",
        "",
        f"- Employees with overtime have an observed attrition rate of "
        f"**{overtime_yes['attrition_rate_percent']:.2f}%**, compared with "
        f"**{overtime_no['attrition_rate_percent']:.2f}%** for employees without overtime.",
        f"- **{highest_department['Department']}** has the highest department-level "
        f"attrition rate in this data: **{highest_department['attrition_rate_percent']:.2f}%**.",
        f"- **{highest_role['JobRole']}** has the highest job-role attrition rate: "
        f"**{highest_role['attrition_rate_percent']:.2f}%**.",
        "- These findings describe historical association only and should not be used "
        "to make individual employment decisions.",
        "",
        "## 3. Numerical patterns",
        "",
    ]

    lines.append("| Feature | Left median | Stayed median | Mann-Whitney p-value |")
    lines.append("|---|---:|---:|---:|")
    for test in numerical_tests:
        lines.append(
            f"| {test['feature']} | {test['left_median']:.2f} | "
            f"{test['stayed_median']:.2f} | {test['p_value']:.6g} |"
        )

    lines.extend(
        [
            "",
            "The p-values indicate statistical evidence of distribution differences, "
            "but not practical importance or causation.",
            "",
            "## 4. Categorical patterns",
            "",
        ]
    )
    lines.append("| Feature | Cramer's V | Chi-square p-value |")
    lines.append("|---|---:|---:|")
    for test in sorted(categorical_tests, key=lambda item: item["cramers_v"], reverse=True):
        lines.append(
            f"| {test['feature']} | {test['cramers_v']:.4f} | "
            f"{test['p_value']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## 5. Attrition rates by selected features",
            "",
        ]
    )
    for column in ["Department", "JobRole", "JobLevel", "OverTime", "JobSatisfaction", "WorkLifeBalance", "YearsAtCompanyBand"]:
        table = rate_tables.get(column)
        if table is None:
            continue
        lines.append(f"### {column}")
        lines.append("")
        lines.append("| Group | Employees | Attrition count | Attrition rate |")
        lines.append("|---|---:|---:|---:|")
        for _, row in table.iterrows():
            lines.append(
                f"| {row.iloc[0]} | {int(row['employee_count'])} | "
                f"{int(row['attrition_count'])} | "
                f"{row['attrition_rate_percent']:.2f}% |"
            )
        lines.append("")

    lines.extend(
        [
            "## 6. Numerical correlations with Attrition",
            "",
        ]
    )
    lines.append("| Feature | Pearson correlation |")
    lines.append("|---|---:|")
    for feature, value in top_correlations.items():
        lines.append(f"| {feature} | {value:.4f} |")

    lines.extend(
        [
            "",
            "Correlation identifies linear association only. Several features are "
            "related to experience or seniority and may introduce redundant information.",
            "",
            "## 7. Outlier review",
            "",
            "| Feature | IQR lower bound | IQR upper bound | Outliers | Outlier percent |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in outliers.head(12).iterrows():
        lines.append(
            f"| {row['feature']} | {row['lower_bound']:.2f} | "
            f"{row['upper_bound']:.2f} | {int(row['outlier_count'])} | "
            f"{row['outlier_percent']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "IQR outliers are observations that fall outside the simple rule; "
            "they are not automatically errors. Domain review is needed before removal.",
            "",
            "## 8. Leakage, redundancy, and modeling exclusions",
            "",
            "- No obvious identifier or direct future-outcome column was found in the inspected schema.",
            "- Attrition is the target and must be excluded from model inputs.",
            "- MonthlyIncome is strongly connected to job level and experience; redundancy should be checked during feature engineering.",
            "- JobLevel, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, "
            "and YearsWithCurrManager may reflect overlapping career-history concepts.",
            "- Age, Gender, and MaritalStatus are protected or sensitive attributes. "
            "They should not be used to recommend adverse employment actions. "
            "Use them only for separate fairness auditing where ethically and legally appropriate.",
            "- Numeric satisfaction and rating fields should be handled carefully because "
            "their meaning depends on the survey or rating scale.",
            "",
            "## 9. Figure inventory",
            "",
            "- `01_target_distribution.png`: target class distribution",
            "- `02_numerical_distributions.png`: numerical univariate distributions",
            "- `03_categorical_distributions.png`: categorical univariate distributions",
            "- `04_age_income_vs_attrition.png`: age and income versus outcome",
            "- `05_gender_attrition_rate.png` through `11_worklifebalance_attrition_rate.png`: selected categorical rates",
            "- `12_tenure_attrition_rate.png`: tenure band rates",
            "- `13_correlation_heatmap.png`: numerical correlations",
            "- `14_satisfaction_attrition_heatmap.png`: satisfaction interactions",
            "- `15_experience_income_attrition.png`: experience and income relationship",
            "- `16_department_role_attrition.png`: department and role comparison",
            "",
            "## 10. Phase 2 limitations",
            "",
            "- The data are historical and may reproduce organizational bias.",
            "- Missingness is absent in the supplied file, but field quality still requires source validation.",
            "- Statistical significance does not prove a feature is a retention intervention.",
            "- Summary tables are descriptive; no assumptions of representativeness are made.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 exploratory data analysis.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/Palo Alto Networks.csv",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures",
    )
    parser.add_argument(
        "--report",
        default="reports/eda_report.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logging()
    sns.set_theme(style="whitegrid")

    try:
        frame = load_csv(args.csv_path)
        if "Attrition" not in frame.columns:
            raise ValueError("Required target column 'Attrition' is missing.")

        figures_dir = Path(args.figures_dir)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        generate_univariate_charts(frame, figures_dir)
        rate_tables = generate_bivariate_charts(frame, figures_dir)
        correlation = generate_multivariate_charts(frame, figures_dir)
        build_eda_report(frame, report_path, rate_tables, correlation)

        logger.info("Analyzed %d records and %d columns.", len(frame), len(frame.columns))
        logger.info("Saved charts under %s", figures_dir)
        logger.info("Wrote EDA report: %s", report_path)
        return 0
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
