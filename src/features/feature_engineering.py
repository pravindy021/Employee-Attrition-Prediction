"""Business-oriented feature engineering."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


BUSINESS_NUMERIC_FEATURES = [
    "IncomePerTotalWorkingYear",
    "PromotionDelayRatio",
    "RoleTenureRatio",
    "OvertimeFlag",
    "EngagementComposite",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create explainable features supported by available employee fields."""
    result = frame.copy()

    total_working_years = result["TotalWorkingYears"].fillna(0).clip(lower=0)
    years_at_company = result["YearsAtCompany"].fillna(0).clip(lower=0)
    years_since_promotion = (
        result["YearsSinceLastPromotion"].fillna(0).clip(lower=0)
    )

    result["IncomePerTotalWorkingYear"] = (
        result["MonthlyIncome"] / (total_working_years + 1)
    )
    result["PromotionDelayRatio"] = (
        years_since_promotion / (years_at_company + 1)
    )
    result["RoleTenureRatio"] = (
        result["YearsInCurrentRole"].fillna(0).clip(lower=0)
        / (years_at_company + 1)
    )
    result["OvertimeFlag"] = (
        result["OverTime"].eq("Yes").astype("int64")
    )

    engagement_columns = [
        "JobInvolvement",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    result["EngagementComposite"] = result[engagement_columns].mean(axis=1)

    return result


class EmployeeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible business feature transformer."""

    def fit(self, frame: pd.DataFrame, y=None):
        del y
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("EmployeeFeatureEngineer expects a pandas DataFrame.")
        required = {
            "MonthlyIncome",
            "TotalWorkingYears",
            "YearsSinceLastPromotion",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "OverTime",
            "JobInvolvement",
            "JobSatisfaction",
            "EnvironmentSatisfaction",
            "RelationshipSatisfaction",
            "WorkLifeBalance",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Missing required feature-engineering columns: {missing}"
            )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("EmployeeFeatureEngineer expects a pandas DataFrame.")
        return engineer_features(frame)
