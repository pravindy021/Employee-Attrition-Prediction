# XAI Methodology

## Explainer selection

The selected Phase 6 model is a fitted isotonic calibrated Logistic Regression.
The underlying fitted classifier is a linear model, so SHAP
`LinearExplainer` was used with 100 training records as a background sample.

## Feature preparation

The model pipeline performs:

- Median imputation and standard scaling for numeric fields
- Most-frequent imputation and one-hot encoding for categorical fields
- Business feature engineering

SHAP receives the transformed 50-column feature matrix. Transformed names are
preserved by the fitted `ColumnTransformer`.

## Calibration distinction

The isotonic calibrator maps underlying logistic scores to probabilities.
SHAP values explain the uncalibrated linear score, not exact calibrated
probability contributions. Calibrated probabilities are reported by the Phase 6
model and are not described as SHAP values.

## Global analysis

Global importance is mean absolute SHAP value across the 294 held-out
explanation records. Average positive and negative SHAP values are also
reported to describe observed directions.

## Individual analysis

The five highest calibrated-probability records are selected for waterfall
plots and human-readable explanations. Individual values are ranked by absolute
contribution.

## Sensitive fields

Age, Gender, and MaritalStatus are excluded from model inputs and explanation
outputs. This prevents the SHAP interface from presenting protected attributes
as operational retention reasons.

## Limitations

- SHAP values are associative, not causal.
- One-hot categories fragment feature importance.
- Linear SHAP does not explain the isotonic calibration mapping exactly.
- Business features depend on definitions and survey scales.
- A held-out result is not evidence of real-world production performance.
