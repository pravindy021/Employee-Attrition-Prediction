# Executive Summary

## Business problem

Organizations may face avoidable workforce disruption when employee resignations
are not anticipated early enough for supportive planning. This platform provides
HR analysts with a historical-pattern-based view of attrition propensity.

## Proposed solution

The Employee Attrition Intelligence Platform combines:

- historical workforce analysis;
- calibrated employee-level attrition propensity;
- department-level risk summaries;
- explainable feature associations;
- demographic fairness review;
- an interactive Streamlit dashboard.

The system is designed for human-reviewed research and retention planning. It is
not an automated employment decision system.

## Main analytical findings

The supplied data contains 1,470 records and 237 historical attrition records.
The historical attrition rate is 16.12%.

Observed EDA patterns include:

- overtime employees had 30.53% historical attrition versus 10.44% without overtime;
- Sales had the highest department-level historical attrition rate at 20.63%;
- Sales Representative had the highest role-level historical rate at 39.76%;
- the 0–2-year tenure band had 29.82% historical attrition.

These are associations in the supplied data, not causal effects.

## Model performance

The project selected a tuned Logistic Regression model with isotonic probability
calibration. On the 294-record held-out test set:

| Measure | Result |
|---|---:|
| PR-AUC | 0.5654 |
| Recall | 0.5745 |
| F1-score | 0.5455 |
| Accuracy | 0.8469 |
| ROC-AUC | 0.8035 |
| Calibrated Brier score | 0.1015 |

At the 0.21 classification threshold, 27 actual attrition records were correctly
identified and 20 actual attrition records were missed. The model also generated
25 false-positive alerts among records without attrition.

## Risk scoring

The dashboard uses three operational risk bands:

- Low: below 30%
- Medium: 30% through 60%
- High: above 60%

These boundaries are configurable assumptions. A risk score is a relative estimate,
not a certainty, and does not indicate that an employee will leave within 30, 60,
or 90 days.

## Explainability

SHAP was applied to the underlying Logistic Regression model. The highest mean
absolute SHAP feature in the held-out explanation set was RoleTenureRatio,
followed by StockOptionLevel and TotalWorkingYears. Overtime, tenure, satisfaction,
income, role, and experience fields also appear among influential transformed
features.

SHAP does not directly explain the isotonic probability calibration mapping.
Explanations should be presented as model associations rather than reasons.

## Fairness safeguards

Age, Gender, and MaritalStatus are excluded from operational model inputs and
SHAP employee explanations. They are used only for held-out subgroup auditing.

Measured subgroup differences include:

- Age equal-opportunity difference: 0.5000
- Gender equal-opportunity difference: 0.1714
- MaritalStatus equal-opportunity difference: 0.2500

Small subgroups can make these differences unstable. They do not prove
discrimination or bias and require legal, HR, and Responsible AI review.

## Potential business applications

- workforce-planning discussion;
- department-level historical trend review;
- voluntary employee-retention research;
- scenario analysis for human review;
- model monitoring and external validation planning.

## Limitations

- The supplied dataset is observational and not independently proven to represent
  a specific organization.
- No exact resignation timing outcome was supplied.
- The 294-record holdout is limited for external and subgroup conclusions.
- Model associations do not establish causal intervention effects.
- Risk categories require stakeholder and legal review before operational use.

## Recommended next steps

1. Validate against independently sourced, time-stamped workforce outcomes.
2. Establish prospective 30/60/90-day outcomes only through an appropriate study.
3. Monitor probability, feature, calibration, and subgroup drift.
4. Improve subgroup sample sizes and uncertainty reporting.
5. Use retention interventions only through voluntary, supportive human review.
6. Revalidate thresholds and model performance before any operational rollout.
