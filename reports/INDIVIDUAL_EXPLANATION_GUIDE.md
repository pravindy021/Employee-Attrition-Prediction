# Individual Explanation Guide

## Purpose

Individual explanations support human review of a model-generated attrition
propensity. They do not determine employment decisions or establish causation.

## Required fields

- Anonymous record identifier
- Calibrated attrition probability
- Low, Medium, or High risk category
- Classification threshold prediction
- Features associated with increased propensity
- Features associated with decreased propensity
- Human-readable interpretation

## Risk categories

| Category | Calibrated probability |
|---|---|
| LOW | Below 30% |
| MEDIUM | 30% through 60% |
| HIGH | Above 60% |

The boundaries are configurable operational assumptions.

## Interpretation steps

1. Review the calibrated probability.
2. Review the positive and negative SHAP contributions.
3. Check the transformed feature values and encoded categories.
4. Conduct a voluntary, contextual HR discussion.
5. Do not use the output alone for punitive or adverse action.

Age, Gender, and MaritalStatus are deliberately excluded from operational
explanations. The output should not expose protected attributes as employee
retention reasons.

## SHAP-to-probability distinction

Linear SHAP values explain the underlying Logistic Regression score. The
isotonic calibrator is not directly explained by the SHAP linear explainer.
The calibrated probability is reported separately.
