"""Reusable exploratory data analysis calculations."""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def attrition_rate_table(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Return record counts and Attrition rates for a categorical or binned column."""
    table = (
        frame.groupby(column, observed=True, dropna=False)
        .agg(
            employee_count=("Attrition", "size"),
            attrition_count=("Attrition", "sum"),
            attrition_rate=("Attrition", "mean"),
        )
        .reset_index()
    )
    table["attrition_rate_percent"] = table["attrition_rate"] * 100
    return table.sort_values("attrition_rate_percent", ascending=False)


def cramers_v(frame: pd.DataFrame, categorical_column: str) -> dict[str, float | str]:
    """Calculate Cramer's V for one categorical feature versus Attrition."""
    contingency = pd.crosstab(frame[categorical_column], frame["Attrition"])
    if contingency.empty or 0 in contingency.shape:
        return {"feature": categorical_column, "cramers_v": 0.0, "p_value": 1.0}

    chi_square, p_value, _, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    phi_squared = chi_square / n
    r, k = contingency.shape
    corrected_phi = max(
        0.0,
        phi_squared - ((k - 1) * (r - 1)) / (n - 1),
    )
    corrected_r = r - ((r - 1) ** 2) / (n - 1)
    corrected_k = k - ((k - 1) ** 2) / (n - 1)
    denominator = min((corrected_k - 1), (corrected_r - 1))
    value = np.sqrt(corrected_phi / denominator) if denominator > 0 else 0.0
    return {
        "feature": categorical_column,
        "cramers_v": float(value),
        "p_value": float(p_value),
    }


def mann_whitney_result(
    frame: pd.DataFrame,
    numerical_column: str,
) -> dict[str, float | str]:
    """Compare a numerical feature for attrition and non-attrition groups."""
    stayed = frame.loc[frame["Attrition"] == 0, numerical_column].dropna()
    left = frame.loc[frame["Attrition"] == 1, numerical_column].dropna()
    statistic, p_value = mannwhitneyu(left, stayed, alternative="two-sided")
    rank_biserial = (
        2 * statistic / (len(left) * len(stayed)) - 1
        if len(left) and len(stayed)
        else 0.0
    )

    return {
        "feature": numerical_column,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "left_median": float(left.median()) if len(left) else 0.0,
        "stayed_median": float(stayed.median()) if len(stayed) else 0.0,
        "rank_biserial_approx": float(rank_biserial),
    }


def numeric_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    """Report IQR-based outlier counts for numeric columns."""
    rows = []
    for column in frame.select_dtypes(include="number").columns:
        if column == "Attrition":
            continue
        series = frame[column].dropna()
        quartile_1 = series.quantile(0.25)
        quartile_3 = series.quantile(0.75)
        iqr = quartile_3 - quartile_1
        lower = quartile_1 - 1.5 * iqr
        upper = quartile_3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "feature": column,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": outlier_count,
                "outlier_percent": float(outlier_count / len(series) * 100)
                if len(series)
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)
