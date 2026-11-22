"""Phase 1 dataset profiling command-line script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import DatasetLoadError, load_csv
from src.data.validate_data import DatasetValidationError, validate_dataset
from src.utils.logging_config import configure_logging


FIELD_DESCRIPTIONS: dict[str, str] = {
    "Age": "Employee age in years.",
    "Attrition": "Target indicator; 1 means attrition, 0 means no attrition.",
    "BusinessTravel": "Frequency of work-related travel.",
    "DailyRate": "Daily compensation rate.",
    "Department": "Employee department.",
    "DistanceFromHome": "Distance from home to work.",
    "Education": "Education level encoded numerically.",
    "EducationField": "Field of education.",
    "EnvironmentSatisfaction": "Environment satisfaction score.",
    "Gender": "Reported gender category.",
    "HourlyRate": "Hourly compensation rate.",
    "JobInvolvement": "Job involvement level.",
    "JobLevel": "Job level in the organizational hierarchy.",
    "JobRole": "Employee job role.",
    "JobSatisfaction": "Job satisfaction score.",
    "MaritalStatus": "Marital status.",
    "MonthlyIncome": "Monthly compensation.",
    "MonthlyRate": "Monthly rate.",
    "NumCompaniesWorked": "Number of previous companies worked for.",
    "OverTime": "Whether the employee works overtime.",
    "PercentSalaryHike": "Salary increase percentage.",
    "PerformanceRating": "Performance rating.",
    "RelationshipSatisfaction": "Relationship satisfaction score.",
    "StockOptionLevel": "Stock option eligibility level.",
    "TotalWorkingYears": "Total years of work experience.",
    "TrainingTimesLastYear": "Training sessions attended last year.",
    "WorkLifeBalance": "Work-life balance score.",
    "YearsAtCompany": "Years employed by the company.",
    "YearsInCurrentRole": "Years in the current role.",
    "YearsSinceLastPromotion": "Years since the last promotion.",
    "YearsWithCurrManager": "Years with the current manager.",
}


def build_data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a data dictionary from observed dataset columns."""
    rows = []
    for column in frame.columns:
        series = frame[column]
        rows.append(
            {
                "column": column,
                "observed_dtype": str(series.dtype),
                "role": (
                    "target"
                    if column == "Attrition"
                    else "numerical"
                    if pd.api.types.is_numeric_dtype(series)
                    else "categorical"
                ),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "description": FIELD_DESCRIPTIONS.get(
                    column, "Description pending review."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_report(
    report_path: Path,
    frame: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    """Write a human-readable Phase 1 profiling report."""
    target_counts = validation["target_counts"]
    attrition_count = target_counts.get("1", 0)
    non_attrition_count = target_counts.get("0", 0)
    total = attrition_count + non_attrition_count
    attrition_rate = (attrition_count / total * 100) if total else 0.0

    lines = [
        "# Employee Attrition Dataset Profiling Report",
        "",
        "## Scope",
        "",
        "This Phase 1 report describes the supplied CSV dataset. "
        "The dataset is treated as a company-inspired case study; "
        "no claim is made that it contains verified Palo Alto Networks employee data.",
        "",
        "## Dataset Summary",
        "",
        f"- Records: **{validation['row_count']}**",
        f"- Columns: **{validation['column_count']}**",
        f"- Duplicate rows: **{validation['duplicate_rows']}**",
        f"- Missing cells: **{validation['missing_cells']}**",
        f"- Attrition count (1): **{attrition_count}**",
        f"- Non-attrition count (0): **{non_attrition_count}**",
        f"- Historical attrition rate: **{attrition_rate:.2f}%**",
        f"- Validation status: **{'PASS' if validation['is_valid'] else 'REVIEW REQUIRED'}**",
        "",
        "## Validation Findings",
        "",
    ]

    if validation["issues"]:
        lines.extend([f"- {issue}" for issue in validation["issues"]])
    else:
        lines.append("- No validation issues detected.")

    lines.extend(["", "## Column Types", "", "| Column | Observed dtype | Role | Missing | Unique |", "|---|---|---|---:|---:|"])
    for column in frame.columns:
        dtype = str(frame[column].dtype)
        role = (
            "target"
            if column == "Attrition"
            else "numerical"
            if pd.api.types.is_numeric_dtype(frame[column])
            else "categorical"
        )
        lines.append(
            f"| {column} | {dtype} | {role} | "
            f"{int(frame[column].isna().sum())} | "
            f"{int(frame[column].nunique(dropna=True))} |"
        )

    lines.extend(
        [
            "",
            "## Categorical Cardinality",
            "",
            "| Column | Unique values |",
            "|---|---:|",
        ]
    )
    for column in frame.select_dtypes(exclude="number").columns:
        lines.append(
            f"| {column} | {int(frame[column].nunique(dropna=True))} |"
        )

    lines.extend(
        [
            "",
            "## Important Limitations",
            "",
            "- This is a profiling report, not an attrition analysis or causal study.",
            "- Numeric-looking category fields may require semantic review.",
            "- Attribute names and target values do not establish provenance or causality.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Profile and validate the employee dataset.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/Palo Alto Networks.csv",
        help="CSV dataset path (default: data/raw/Palo Alto Networks.csv)",
    )
    parser.add_argument(
        "--json-output",
        default="reports/dataset_validation.json",
        help="JSON validation report path.",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/dataset_profile.md",
        help="Markdown profiling report path.",
    )
    parser.add_argument(
        "--dictionary-output",
        default="reports/data_dictionary.csv",
        help="CSV data dictionary path.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Phase 1 data profiling workflow."""
    args = parse_args()
    logger = configure_logging()

    try:
        frame = load_csv(args.csv_path)
        validation = validate_dataset(frame)
        dictionary = build_data_dictionary(frame)

        json_path = Path(args.json_output)
        markdown_path = Path(args.markdown_output)
        dictionary_path = Path(args.dictionary_output)

        for path in (json_path, markdown_path, dictionary_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        json_path.write_text(
            json.dumps(validation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_markdown_report(markdown_path, frame, validation)
        dictionary.to_csv(dictionary_path, index=False)

        logger.info("Loaded %d records and %d columns.", len(frame), len(frame.columns))
        logger.info("Wrote JSON report: %s", json_path)
        logger.info("Wrote Markdown report: %s", markdown_path)
        logger.info("Wrote data dictionary: %s", dictionary_path)

        if not validation["is_valid"]:
            logger.warning("Dataset validation found issues: %s", validation["issues"])
            return 2

        logger.info("Dataset validation passed.")
        return 0
    except (DatasetLoadError, DatasetValidationError, OSError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

