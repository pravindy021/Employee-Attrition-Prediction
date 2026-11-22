"""Tests for Phase 6 thresholds, probabilities, and risk summaries."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.risk_scoring import (
    build_department_risk_summary,
    build_employee_risk_table,
    classify_risk,
    optimize_threshold,
    threshold_predictions,
    validate_probabilities,
)


class Phase6RiskScoringTests(unittest.TestCase):
    def test_risk_categories_match_phase6_boundaries(self) -> None:
        categories = classify_risk([0.29, 0.30, 0.60, 0.61])
        self.assertEqual(categories.tolist(), ["LOW", "MEDIUM", "MEDIUM", "HIGH"])

    def test_invalid_probabilities_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_probabilities([0.2, 1.2])
        with self.assertRaises(ValueError):
            validate_probabilities([0.2, np.nan])

    def test_threshold_is_validated(self) -> None:
        predictions = threshold_predictions([0.1, 0.5, 0.8], 0.5)
        self.assertEqual(predictions.tolist(), [0, 1, 1])
        with self.assertRaises(ValueError):
            threshold_predictions([0.5], 1.2)

    def test_threshold_optimization_returns_valid_tradeoffs(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1])
        probabilities = np.array([0.1, 0.4, 0.7, 0.8, 0.2, 0.9])
        threshold, tradeoffs = optimize_threshold(
            y_true,
            probabilities,
            minimum_recall=0.65,
        )
        self.assertGreaterEqual(threshold, 0.05)
        self.assertLessEqual(threshold, 0.95)
        self.assertIn("precision", tradeoffs.columns)
        self.assertIn("recall", tradeoffs.columns)

    def test_employee_and_department_summaries_are_reproducible(self) -> None:
        test_frame = pd.DataFrame(
            {
                "Department": ["Sales", "R&D", "Sales"],
                "JobRole": ["Sales Executive", "Research Scientist", "Sales Executive"],
                "OverTime": ["Yes", "No", "Yes"],
            },
            index=[10, 11, 12],
        )
        y_true = np.array([1, 0, 1])
        probabilities = np.array([0.65, 0.25, 0.50])

        employee = build_employee_risk_table(
            test_frame,
            y_true,
            probabilities,
            threshold=0.50,
        )
        department = build_department_risk_summary(employee)

        self.assertEqual(len(employee), 3)
        self.assertEqual(employee["risk_category"].tolist(), ["HIGH", "LOW", "MEDIUM"])
        self.assertIn("high_risk_count", department.columns)
        self.assertEqual(employee["anonymous_record_id"].tolist()[0], "record_10")


if __name__ == "__main__":
    unittest.main()
