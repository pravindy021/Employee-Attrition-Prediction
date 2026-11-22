"""Tests for Phase 4 model development helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.model_development import (
    build_model_registry,
    calculate_classification_metrics,
)


class Phase4ModelTests(unittest.TestCase):
    def test_model_registry_contains_required_models(self) -> None:
        registry = build_model_registry()
        names = set(registry)
        self.assertIn("dummy_baseline", names)
        self.assertIn("logistic_regression", names)
        self.assertIn("decision_tree", names)
        self.assertIn("random_forest", names)
        self.assertIn("gradient_boosting", names)

    def test_confusion_and_metrics_are_consistent(self) -> None:
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        y_probability = np.array([0.1, 0.2, 0.6, 0.3, 0.8, 0.7, 0.4, 0.9])

        metrics = calculate_classification_metrics(
            y_true,
            y_pred,
            y_probability,
        )

        self.assertEqual(
            metrics["true_negative"]
            + metrics["false_positive"]
            + metrics["false_negative"]
            + metrics["true_positive"],
            8,
        )
        self.assertEqual(metrics["true_negative"], 3)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_positive"], 3)


if __name__ == "__main__":
    unittest.main()
