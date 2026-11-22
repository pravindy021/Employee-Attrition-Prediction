"""Tests for Phase 1 loading and dataset validation behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.load_data import DatasetLoadError, load_csv
from src.data.validate_data import DatasetValidationError, validate_dataset


class DataValidationTests(unittest.TestCase):
    def test_validate_clean_binary_target(self) -> None:
        frame = pd.DataFrame(
            {
                "Age": [25, 40],
                "Department": ["Sales", "Research & Development"],
                "Attrition": [0, 1],
            }
        )

        result = validate_dataset(frame)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["missing_cells"], 0)
        self.assertEqual(result["duplicate_rows"], 0)

    def test_validate_reports_missing_duplicate_and_invalid_target(self) -> None:
        frame = pd.DataFrame(
            {
                "Age": [25, 25, None],
                "Attrition": [0, 0, 2],
            }
        )

        result = validate_dataset(frame)

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["missing_cells"], 1)
        self.assertGreaterEqual(result["duplicate_rows"], 0)
        self.assertIn("Missing values detected.", result["issues"])
        self.assertIn("The Attrition target contains values other than 0 or 1.", result["issues"])

    def test_missing_target_raises_validation_error(self) -> None:
        with self.assertRaises(DatasetValidationError):
            validate_dataset(pd.DataFrame({"Age": [25]}))

    def test_empty_path_raises_load_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(DatasetLoadError):
                load_csv(path)


if __name__ == "__main__":
    unittest.main()

