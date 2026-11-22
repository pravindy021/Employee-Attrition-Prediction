# SHAP Explainability Report

## Scope

- Model: selected tuned Logistic Regression plus isotonic calibration
- SHAP explainer: `shap.LinearExplainer`
- SHAP output: underlying logistic score in log-odds space
- Reported employee probabilities: isotonic-calibrated probability
- Sensitive model fields: Age, Gender, and MaritalStatus excluded

SHAP explains the fitted linear model score. The isotonic calibrator transforms that score into a final probability, so calibrated probability values must not be described as direct SHAP contributions.

## Global feature findings

- Highest mean absolute SHAP feature: **RoleTenureRatio** (0.509322).
- Largest average positive contribution: **RoleTenureRatio** (0.767726).
- Largest average negative contribution: **BusinessTravel Non-Travel** (-0.974485).

These values describe average associations across the explanation data.
They do not establish causal effects or a guaranteed retention action.

### Overtime

- **OvertimeFlag** mean absolute SHAP: 0.407226; average positive: 0.725417; average negative: -0.296297. This direction is association, not proof that overtime causes attrition.
- **OverTime No** mean absolute SHAP: 0.339357; average positive: 0.604517; average negative: -0.246916. This direction is association, not proof that overtime causes attrition.
- **OverTime Yes** mean absolute SHAP: 0.020280; average positive: 0.014756; average negative: -0.036127. This direction is association, not proof that overtime causes attrition.

### Tenure

- **RoleTenureRatio** mean absolute SHAP: 0.509322.

### Job satisfaction

- **EngagementComposite** mean absolute SHAP: 0.314209.
- **JobSatisfaction** mean absolute SHAP: 0.157090.
- **WorkLifeBalance** mean absolute SHAP: 0.123091.

## Individual high-risk contributors

- **record_357**: Record record_357 has a calibrated attrition propensity of 100.0%, categorized as HIGH risk. The largest increasing features are OvertimeFlag, JobRole Sales Representative, OverTime No. These are model associations, not causal reasons or a certainty of resignation.
- **record_911**: Record record_911 has a calibrated attrition propensity of 100.0%, categorized as HIGH risk. The largest increasing features are RoleTenureRatio, TotalWorkingYears, OvertimeFlag. The largest decreasing features are YearsInCurrentRole. These are model associations, not causal reasons or a certainty of resignation.
- **record_688**: Record record_688 has a calibrated attrition propensity of 100.0%, categorized as HIGH risk. The largest increasing features are RoleTenureRatio, TotalWorkingYears, OvertimeFlag. The largest decreasing features are EducationField Other, YearsInCurrentRole. These are model associations, not causal reasons or a certainty of resignation.
- **record_1021**: Record record_1021 has a calibrated attrition propensity of 97.1%, categorized as HIGH risk. The largest increasing features are EngagementComposite, JobRole Sales Representative, Department Research & Development. The largest decreasing features are YearsInCurrentRole. These are model associations, not causal reasons or a certainty of resignation.
- **record_829**: Record record_829 has a calibrated attrition propensity of 97.1%, categorized as HIGH risk. The largest increasing features are EngagementComposite, OvertimeFlag, IncomePerTotalWorkingYear. The largest decreasing features are NumCompaniesWorked. These are model associations, not causal reasons or a certainty of resignation.

## Coefficient comparison

| Feature | Logistic coefficient | Mean absolute SHAP |
|---|---:|---:|
| RoleTenureRatio | -0.620718 | 0.509322 |
| StockOptionLevel | -0.519587 | 0.435854 |
| TotalWorkingYears | -0.542397 | 0.427211 |
| YearsInCurrentRole | 0.458521 | 0.420218 |
| NumCompaniesWorked | 0.468887 | 0.407509 |
| OvertimeFlag | 0.463196 | 0.407226 |
| OverTime No | -0.851433 | 0.339357 |
| YearsSinceLastPromotion | 0.406289 | 0.327925 |
| Department Research & Development | -0.717369 | 0.322084 |
| EngagementComposite | -0.421535 | 0.314209 |
| MonthlyIncome | -0.357213 | 0.296249 |
| DistanceFromHome | 0.373441 | 0.288274 |
| JobRole Laboratory Technician | 0.918139 | 0.259640 |
| EnvironmentSatisfaction | -0.269580 | 0.242395 |
| IncomePerTotalWorkingYear | 0.292352 | 0.240782 |
| JobInvolvement | -0.278933 | 0.235295 |
| YearsWithCurrManager | -0.244880 | 0.211780 |
| BusinessTravel Non-Travel | -1.082761 | 0.187826 |
| EducationField Medical | -0.384591 | 0.172621 |
| DailyRate | -0.177667 | 0.159518 |

Coefficient and SHAP rankings can differ because coefficients measure directional model weight in standardized/transformed feature units, while SHAP mean absolute values measure the magnitude of observed contributions across the explanation population.

## Risk category distribution

- LOW: **255**
- MEDIUM: **19**
- HIGH: **20**

## Limitations

- SHAP associations do not prove causation.
- SHAP is generated for the underlying linear score, not exact isotonic calibrated probabilities.
- The model excludes Age, Gender, and MaritalStatus from operational explanations.
- Encoded one-hot features represent categories and may fragment global importance.
- The 294-record held-out set is limited and not proof of production performance.
- Explanations should support voluntary, human-reviewed retention discussions.
