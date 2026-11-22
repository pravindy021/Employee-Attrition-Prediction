# Testing Strategy

## Test layers

1. **Unit tests** cover validation, preprocessing, feature engineering, metrics,
   risk scoring, SHAP utilities, fairness calculations, configuration, and artifacts.
2. **Integration tests** verify the saved calibrated model, 1,176/294 split, probability
   output, 0.21 threshold, SHAP/fairness reports, and sensitive-feature separation.
3. **Dashboard tests** use Streamlit AppTest to verify startup, navigation elements,
   and absence of script exceptions.
4. **Regression tests** run both the existing unittest suite and complete pytest suite.
5. **Performance checks** measure model loading and repeated prediction consistency.

## Data-integrity checks

- Compare SHA-256 hashes for the original and raw-copy CSV.
- Verify train/test indices are disjoint.
- Verify the sensitive fields Age, Gender, and MaritalStatus are excluded.
- Verify no identifier column is added as a model input.
- Verify existing reports and model artifacts exist.

## Boundary coverage

Tests cover missing values, empty data, invalid probabilities, threshold boundaries,
zero-division fairness metrics, mismatched inputs, small subgroups, and duplicate
or repeated prediction results.

## Reproducible commands

```powershell
python -m pytest tests -q
python -m unittest discover -s tests -p "test_*.py" -q
python scripts/verify_artifacts.py
python scripts/run_phase11_quality.py
python -m pip check
```
