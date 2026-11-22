"""Reproducible train-only data preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_engineering import (
    BUSINESS_NUMERIC_FEATURES,
    EmployeeFeatureEngineer,
)


TARGET_COLUMN = "Attrition"
SENSITIVE_COLUMNS = ["Age", "Gender", "MaritalStatus"]


@dataclass(frozen=True)
class PreprocessingSpec:
    """Resolved columns used for model and fairness audit data."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]


def detect_identifier_columns(frame: pd.DataFrame) -> list[str]:
    """Identify likely identifier columns from names and full-column uniqueness."""
    identifier_names = ("id", "employee_number", "employeenumber")
    return [
        column
        for column in frame.columns
        if column.lower() in identifier_names
        or (
            column.lower().endswith("id")
            and frame[column].nunique(dropna=False) == len(frame)
        )
    ]


def detect_constant_columns(frame: pd.DataFrame) -> list[str]:
    """Identify columns with no more than one observed value."""
    return [column for column in frame.columns if frame[column].nunique(dropna=False) <= 1]


def resolve_preprocessing_spec(
    frame: pd.DataFrame,
) -> PreprocessingSpec:
    """Separate model inputs, sensitive fields, and excluded columns."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    identifiers = detect_identifier_columns(frame)
    constants = detect_constant_columns(frame)

    excluded = {
        TARGET_COLUMN,
        *SENSITIVE_COLUMNS,
        *identifiers,
        *constants,
    }
    model_candidates = [column for column in frame.columns if column not in excluded]

    engineered_numeric = set(BUSINESS_NUMERIC_FEATURES)
    numeric_columns = [
        column
        for column in model_candidates
        if pd.api.types.is_numeric_dtype(frame[column]) or column in engineered_numeric
    ]
    categorical_columns = [
        column for column in model_candidates if column not in numeric_columns
    ]

    # Add engineered numeric features to the model feature list.
    numeric_columns.extend(
        column
        for column in BUSINESS_NUMERIC_FEATURES
        if column not in numeric_columns
    )

    return PreprocessingSpec(
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        excluded_columns=tuple(sorted(excluded)),
        identifier_columns=tuple(identifiers),
        constant_columns=tuple(constants),
    )


def build_preprocessing_pipeline(
    spec: PreprocessingSpec,
) -> Pipeline:
    """Build an sklearn pipeline that engineers features and transforms model inputs."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(spec.numeric_columns)),
            ("categorical", categorical_pipeline, list(spec.categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", EmployeeFeatureEngineer()),
            ("preprocessor", column_transformer),
        ]
    )
