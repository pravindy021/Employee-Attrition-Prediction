# Test Execution Report

## Actual results

| Test command | Result |
|---|---|
| `python -m pytest tests -q` | 64 passed (final Phase 13 run) |
| `python -m pytest tests/test_phase11_integration.py tests/test_phase11_dashboard.py -q` | 10 passed |
| `python -m pytest tests -q` | 48 passed |
| `python -m unittest discover -s tests -p "test_*.py" -q` | 18 passed |
| `python scripts/verify_artifacts.py` | 12 required artifacts verified |
| `python scripts/run_phase11_quality.py` | Completed successfully |
| `python -m pip check` | No broken requirements |

No failing or skipped tests were recorded in the final test runs.

## Integration findings

- The saved Phase 6 calibrated model loads and generates probabilities in [0, 1].
- The 0.21 classification threshold remains consistent with Phase 6 metadata.
- Low/Medium/High risk boundaries remain 0.30 and 0.60.
- Train/test indices are disjoint: 1,176 training and 294 test records.
- Age, Gender, and MaritalStatus are excluded from operational model inputs.
- The original and raw-copy CSV hashes match.
- SHAP and fairness reports are readable.
- Streamlit AppTest starts the dashboard without script exceptions.

## Performance findings

- Model loading: **0.048907 seconds** in the latest measured run.
- First prediction: **0.308115 seconds**.
- 20 repeated predictions: **4.698003 seconds**.
- Average repeated prediction: **0.234900 seconds**.
- Repeated predictions returned the same probability.

Timings are local smoke measurements on this environment, not production benchmarks.
