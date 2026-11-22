"""Measure model-loading and repeated-prediction responsiveness."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.preprocessing import SENSITIVE_COLUMNS, resolve_preprocessing_spec
from src.utils.logging_config import configure_logging


def main() -> int:
    """Execute Phase 11 performance smoke checks."""
    logger = configure_logging()
    frame = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "Palo Alto Networks.csv")
    spec = resolve_preprocessing_spec(frame)
    model_columns = [
        column
        for column in frame.columns
        if column not in spec.excluded_columns
    ]

    model_path = PROJECT_ROOT / "models" / "phase6_attrition_risk_model.joblib"
    start = time.perf_counter()
    model = joblib.load(model_path)
    model_load_seconds = time.perf_counter() - start

    profile = frame.iloc[[0]][model_columns].copy()
    start = time.perf_counter()
    first = model.predict_proba(profile)[:, 1]
    first_prediction_seconds = time.perf_counter() - start

    repetitions = 20
    start = time.perf_counter()
    probabilities = [float(model.predict_proba(profile)[:, 1][0]) for _ in range(repetitions)]
    repeated_seconds = time.perf_counter() - start

    if len(set(probabilities)) != 1:
        logger.error("Repeated predictions were not deterministic.")
        return 1
    if not all(0.0 <= value <= 1.0 for value in probabilities):
        logger.error("A repeated prediction produced an invalid probability.")
        return 1

    result = {
        "model_load_seconds": round(model_load_seconds, 6),
        "first_prediction_seconds": round(first_prediction_seconds, 6),
        "20_repeated_predictions_seconds": round(repeated_seconds, 6),
        "average_prediction_seconds": round(repeated_seconds / repetitions, 6),
        "probability_reproducible": True,
        "sensitive_features_excluded": list(SENSITIVE_COLUMNS),
        "sample_probability": float(first[0]),
    }
    output = PROJECT_ROOT / "reports" / "phase11_performance.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Model loaded in %.6f seconds.", model_load_seconds)
    logger.info("First prediction completed in %.6f seconds.", first_prediction_seconds)
    logger.info("20 repeated predictions completed in %.6f seconds.", repeated_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
