"""Check deployment readiness for Streamlit Community Cloud."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import configure_logging


REQUIRED_FILES = (
    "app.py",
    "requirements.txt",
    "runtime.txt",
    ".streamlit/config.toml",
    "configs/project.json",
    "data/raw/Palo Alto Networks.csv",
    "data/processed/phase3_split.joblib",
    "data/processed/phase6_risk_scoring_config.joblib",
    "data/processed/employee_risk_scores.csv",
    "data/processed/department_risk_summary.csv",
    "models/preprocessing_pipeline.joblib",
    "models/phase5/logistic_regression_tuned.joblib",
    "models/phase6_attrition_risk_model.joblib",
    "reports/phase6_validation_metadata.json",
    "reports/DEMOGRAPHIC_METRICS.csv",
    "reports/explainability/shap_global_importance.csv",
    "reports/explainability/individual_explanations.json",
)


def main() -> int:
    """Write and log a deployment readiness report."""
    logger = configure_logging()
    missing = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not (PROJECT_ROOT / relative_path).is_file()
    ]

    excluded = [".venv", "__pycache__", ".pytest_cache"]
    files = [
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and not any(part in excluded for part in path.parts)
    ]
    total_bytes = sum(path.stat().st_size for path in files)
    largest = sorted(files, key=lambda path: path.stat().st_size, reverse=True)[:10]

    report = {
        "deployment_status": "PREPARATION_COMPLETE" if not missing else "BLOCKED",
        "missing_required_files": missing,
        "python_runtime": "python-3.12.10",
        "streamlit_entrypoint": "app.py",
        "repository_size_excluding_venv_bytes": total_bytes,
        "largest_files": [
            {
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
            }
            for path in largest
        ],
        "public_url": None,
    }
    output = PROJECT_ROOT / "reports" / "deployment_readiness.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if missing:
        logger.error("Deployment readiness blocked by missing files: %s", missing)
        return 1
    logger.info(
        "Deployment readiness passed: %d required files, %.2f MB excluding .venv.",
        len(REQUIRED_FILES),
        total_bytes / 1024 / 1024,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
