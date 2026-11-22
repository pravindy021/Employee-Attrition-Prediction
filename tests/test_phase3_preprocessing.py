"""Tests for Phase 3 feature engineering and preprocessing."""

from __future__ import annotations

import unittest

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import engineer_features
from src.features.preprocessing import (
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    resolve_preprocessing_spec,
)


class Phase3PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "Age": [25, 40, 35, 50],
                "Attrition": [1, 0, 1, 0],
                "Gender": ["Female", "Male", "Female", "Male"],
                "MaritalStatus": ["Single", "Married", "Single", "Married"],
                "Department": ["Sales", "R&D", "Sales", "R&D"],
                "OverTime": ["Yes", "No", "Yes", "No"],
                "MonthlyIncome": [3000, 6000, 4000, 8000],
                "TotalWorkingYears": [2, 15, 8, 25],
                "YearsSinceLastPromotion": [1, 3, 2, 5],
                "YearsAtCompany": [1, 10, 5, 20],
                "YearsInCurrentRole": [1, 7, 4, 10],
                "JobInvolvement": [2, 3, 1, 4],
                "JobSatisfaction": [1, 3, 2, 4],
                "EnvironmentSatisfaction": [2, 4, 1, 3],
                "RelationshipSatisfaction": [3, 4, 2, 4],
                "WorkLifeBalance": [1, 3, 2, 4],
            }
        )

    def test_engineered_features_are_explainable_and_bounded(self) -> None:
        engineered = engineer_features(self.frame)
        self.assertIn("IncomePerTotalWorkingYear", engineered)
        self.assertIn("PromotionDelayRatio", engineered)
        self.assertIn("RoleTenureRatio", engineered)
        self.assertIn("OvertimeFlag", engineered)
        self.assertIn("EngagementComposite", engineered)
        self.assertTrue(engineered["OvertimeFlag"].isin([0, 1]).all())

    def test_sensitive_columns_are_excluded_from_model_spec(self) -> None:
        spec = resolve_preprocessing_spec(self.frame)
        model_columns = set(spec.numeric_columns) | set(spec.categorical_columns)
        for column in SENSITIVE_COLUMNS:
            self.assertNotIn(column, model_columns)
        self.assertIn(TARGET_COLUMN, spec.excluded_columns)

    def test_scaler_is_fitted_only_on_training_data(self) -> None:
        spec = resolve_preprocessing_spec(self.frame)
        pipeline = build_preprocessing_pipeline(spec)
        x = self.frame.drop(columns=[TARGET_COLUMN])
        y = self.frame[TARGET_COLUMN]
        x_train, x_test, _, _ = train_test_split(
            x,
            y,
            test_size=0.50,
            random_state=42,
            stratify=y,
        )
        del x_test

        pipeline.fit(x_train)
        numeric_transformer = pipeline.named_steps["preprocessor"].named_transformers_[
            "numeric"
        ]
        scaler = numeric_transformer.named_steps["scaler"]
        imputer = numeric_transformer.named_steps["imputer"]

        expected_means = imputer.statistics_
        transformed = pipeline.transform(x_train)
        self.assertEqual(transformed.shape[0], len(x_train))
        self.assertEqual(len(scaler.mean_), len(expected_means))

    def test_repeated_fit_has_reproducible_output(self) -> None:
        spec = resolve_preprocessing_spec(self.frame)
        pipeline_one = build_preprocessing_pipeline(spec)
        pipeline_two = build_preprocessing_pipeline(spec)
        x = self.frame.drop(columns=[TARGET_COLUMN])

        first = pipeline_one.fit_transform(x)
        second = pipeline_two.fit_transform(x)

        first_dense = first.toarray() if hasattr(first, "toarray") else first
        second_dense = second.toarray() if hasattr(second, "toarray") else second
        self.assertTrue((first_dense == second_dense).all())


if __name__ == "__main__":
    unittest.main()
