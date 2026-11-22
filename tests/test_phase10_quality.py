"""Tests for Phase 10 configuration, artifacts, and logging."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from src.config.settings import ProjectSettings
from src.utils.artifact_registry import inspect_artifacts, write_artifact_report
from src.utils.logging_config import configure_logging


def test_project_settings_preserve_phase_thresholds() -> None:
    settings = ProjectSettings.load()
    assert settings.random_state == 42
    assert settings.test_size == pytest.approx(0.20)
    assert settings.classification_threshold == pytest.approx(0.21)
    assert settings.low_risk_max == pytest.approx(0.30)
    assert settings.medium_risk_max == pytest.approx(0.60)
    assert settings.root.name == "Employee-Attrition-Prediction"


def test_missing_artifact_is_reported_without_crashing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checks = inspect_artifacts(root, ["missing.joblib"])
        assert len(checks) == 1
        assert checks[0].exists is False
        assert checks[0].sha256 is None


def test_artifact_report_is_written_for_existing_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        file_path = root / "artifact.txt"
        file_path.write_text("integrity-check", encoding="utf-8")
        report_path = root / "artifact_report.json"
        report = write_artifact_report(
            root,
            ["artifact.txt"],
            report_path,
        )
        assert report["all_required_artifacts_present"] is True
        assert report["artifacts"][0]["sha256"]
        assert report_path.exists()


def test_logger_returns_consistent_named_logger() -> None:
    logger = configure_logging(logging.INFO)
    assert logger.name == "employee_attrition"
    assert logger.level == logging.INFO
    assert logger.handlers
