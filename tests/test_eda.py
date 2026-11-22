"""Tests for Phase 2 EDA calculations."""

from __future__ import annotations

import unittest

import pandas as pd

from src.analysis.eda import attrition_rate_table, numeric_outliers


class EDATests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "Department": ["Sales", "Sales", "R&D", "R&D"],
                "Age": [25, 30, 45, 55],
                "Attrition": [1, 0, 0, 1],
            }
        )

    def test_attrition_rate_table(self) -> None:
        table = attrition_rate_table(self.frame, "Department")
        sales = table.loc[table["Department"] == "Sales"].iloc[0]
        self.assertEqual(int(sales["employee_count"]), 2)
        self.assertAlmostEqual(float(sales["attrition_rate"]), 0.5)

    def test_numeric_outliers_returns_feature_rows(self) -> None:
        table = numeric_outliers(self.frame)
        self.assertIn("Age", set(table["feature"]))
        self.assertIn("outlier_count", table.columns)


if __name__ == "__main__":
    unittest.main()
