"""Verify required Phase 1-9 artifact availability and integrity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import ProjectSettings
from src.utils.artifact_registry import write_artifact_report
from src.utils.logging_config import configure_logging


REQUIRED_ARTIFACTS = (
    "data/Palo Alto Networks.csv",
    "data/raw/Palo Alto Networks.csv",
    "reports/eda_report.md",
    "models/preprocessing_pipeline.joblib",
    "data/processed/phase3_split.joblib",
    "reports/model_comparison.csv",
    "reports/tuned_model_comparison.csv",
    "models/phase6_attrition_risk_model.joblib",
    "data/processed/phase6_risk_scoring_config.joblib",
    "reports/explainability/shap_global_importance.csv",
    "reports/DEMOGRAPHIC_METRICS.csv",
    "app.py",
)


def main() -> int:
    """Run an artifact integrity check and write a JSON report."""
    logger = configure_logging()
    settings = ProjectSettings.load()
    report = write_artifact_report(
        settings.root,
        REQUIRED_ARTIFACTS,
        settings.reports_dir / "artifact_integrity.json",
    )
    if not report["all_required_artifacts_present"]:
        missing = [
            item["relative_path"]
            for item in report["artifacts"]
            if not item["exists"]
        ]
        logger.error("Missing required artifacts: %s", missing)
        return 1
    logger.info(
        "Verified %d required project artifacts.",
        len(REQUIRED_ARTIFACTS),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
