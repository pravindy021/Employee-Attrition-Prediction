# Model Card

## Intended use

The model supports historical workforce analytics and human-reviewed retention
research. It must not automate employment actions or make 30/60/90-day exit claims.

## Model

- Family: Logistic Regression
- Calibration: Isotonic
- Threshold: 0.21
- Sensitive attributes excluded: Age, Gender, MaritalStatus

## Training data

- 1,176 training records
- 294 held-out test records
- 31 source columns
- 50 transformed model features
- Stratified split

## Held-out performance

| Metric | Value |
|---|---:|
| Accuracy | 0.8469387755 |
| Precision | 0.5192307692 |
| Recall | 0.5744680851 |
| F1 | 0.5454545455 |
| ROC-AUC | 0.8035145146 |
| PR-AUC | 0.5653826702 |
| Balanced accuracy | 0.7366267551 |
| Specificity | 0.8987854251 |
| Calibrated Brier score | 0.1014904169 |

## Explainability

The underlying linear model uses SHAP `LinearExplainer`. SHAP does not exactly explain
the isotonic calibration layer.

## Fairness

Age, Gender, and MaritalStatus are audit-only fields. Relevant subgroup metrics are in
`reports/DEMOGRAPHIC_METRICS.csv` and `reports/FAIRNESS_AUDIT_REPORT.md`.

## Limitations

- Historical data may reproduce organizational bias.
- Held-out performance is not external validation.
- Associations are not causal.
- Fairness subgroup sizes are limited.
- Probabilities are not a guaranteed resignation event.
