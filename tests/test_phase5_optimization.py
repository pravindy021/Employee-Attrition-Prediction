"""Tests for Phase 5 hyperparameter optimization helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.model_development import select_promising_models


class Phase5OptimizationTests(unittest.TestCase):
    def test_promising_models_are_ranked_by_pr_auc(self) -> None:
        comparison = pd.DataFrame(
            {
                "model": ["logistic", "xgboost", "dummy", "random_forest"],
                "pr_auc": [0.62, 0.56, 0.16, 0.55],
                "recall": [0.74, 0.43, 0.0, 0.46],
                "roc_auc": [0.83, 0.80, 0.50, 0.80],
            }
        )
        selected = select_promising_models(comparison, limit=3)
        self.assertEqual(selected["model"].iloc[0], "logistic")
        self.assertEqual(len(selected), 3)
        self.assertNotIn("dummy", set(selected["model"]))
        self.assertIn("xgboost", set(selected["model"]))


if __name__ == "__main__":
    unittest.main()
