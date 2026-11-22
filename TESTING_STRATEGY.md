# Testing Strategy

## Test layers

- **Unit tests** validate data, preprocessing, feature engineering, metrics,
  risk scoring, SHAP utilities, fairness calculations, configuration, and artifact integrity.
- **Integration tests** verify model loading, probability validity, the 0.21 threshold,
  train/test separation, SHAP and fairness report loading, and dashboard artifact access.
- **Dashboard tests** use Streamlit AppTest for startup and navigation elements.
- **Regression tests** run existing and new pytest and unittest suites.
- **Performance checks** measure model loading and repeated-prediction consistency.

## Data and leakage checks

- Original CSV SHA-256 is compared with the raw copy.
- Training and test indices must be disjoint.
- Age, Gender, and MaritalStatus are excluded from operational model inputs.
- Test data is not used for tuning or model selection.
- No fake employee records are substituted for dashboard production views.

## Boundary coverage

Tests cover missing values, empty data, invalid probabilities, threshold boundaries,
zero-division behavior, mismatched lengths, missing artifacts, small subgroups,
and repeated-prediction reproducibility.

## Reproducible commands

```powershell
python -m pytest tests -q
python -m unittest discover -s tests -p "test_*.py" -q
python scripts/verify_artifacts.py
python scripts/run_phase11_quality.py
python -m pip check
```
