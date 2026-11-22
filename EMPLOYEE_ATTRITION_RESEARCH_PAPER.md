# Employee Attrition Intelligence Platform: An Interpretable HR Analytics Study

## 1. Title

Employee Attrition Intelligence Platform: A Reproducible Machine Learning,
Explainability, Risk Scoring, and Fairness Case Study

## 2. Abstract

This paper documents an implemented employee attrition analytics platform built
from a supplied 1,470-record, 31-column dataset. The platform evaluates historical
attrition patterns, engineered workforce features, multiple classification
candidates, isotonic probability calibration, configurable risk scoring, SHAP
explainability, and demographic subgroup auditing.

The supplied target contains 237 attrition records and 1,233 non-attrition records.
An 80/20 stratified split produced 1,176 training records and 294 held-out test
records. Tuned cross-validation models were compared, and Logistic Regression was
selected as the candidate for final isotonic calibration.

On the held-out test set, the calibrated selected pipeline achieved PR-AUC 0.5654,
recall 0.5745, F1 0.5455, accuracy 0.8469, precision 0.5192, ROC-AUC 0.8035,
and calibrated Brier score 0.1015 under the implemented 0.21 classification
threshold.

The work is limited to a historical observational dataset and does not establish
causation or a future 30-, 60-, or 90-day resignation outcome.

## 3. Keywords

Employee attrition; workforce analytics; Logistic Regression; probability
calibration; explainable AI; SHAP; fairness auditing; risk scoring; Streamlit.

## 4. Introduction

Employee attrition can create workforce planning, replacement, knowledge-retention,
and operational challenges. This project investigates whether historical employee
records contain patterns associated with attrition propensity. It includes
descriptive analysis, model development, calibrated risk estimates, local and
global explanations, and responsible-AI subgroup review.

The implementation is decision-support software. It does not automatically decide
employment actions or provide a verified prediction of a specific resignation date.

## 5. Problem Statement

The binary prediction task is to estimate the probability that an employee record
belongs to the historical attrition class, `Attrition = 1`. The target is
imbalanced: 16.12% of records are in the positive class. Accuracy alone can
therefore provide a misleading view of performance.

The platform must also separate historical attrition outcomes from hypothetical
operational risk categories and avoid presenting associative explanations as
causal reasons.

## 6. Research Objectives

1. Profile and validate the supplied dataset.
2. Compare classification approaches using a reproducible methodology.
3. Optimize promising models using training data only.
4. Evaluate probability calibration and threshold trade-offs.
5. Produce employee and department-level analytical risk summaries.
6. Explain the underlying linear model with SHAP.
7. Audit outcomes across available demographic groups.
8. Provide a transparent dashboard and testing strategy.

## 7. Dataset Description

The dataset contains 1,470 rows and 31 columns. Phase 1 found 0 missing cells
and 0 duplicate rows.

| Target class | Count | Percentage |
|---|---:|---:|
| 0: No recorded attrition | 1,233 | 83.88% |
| 1: Recorded attrition | 237 | 16.12% |

The dataset provenance is not independently verified as actual employee data
from a named organization. It is presented only as a company-inspired case study.

The seven categorical fields are BusinessTravel, Department, EducationField,
Gender, JobRole, MaritalStatus, and OverTime. The remaining fields are numeric or
ordinal-like numeric fields.

## 8. Data Preprocessing

The 80/20 split is stratified and reproducible:

- Training records: 1,176
- Held-out test records: 294
- Random seed: 42

Numeric fields use median imputation and standard scaling. Categorical fields use
most-frequent imputation and one-hot encoding with unknown-category handling.
Preprocessing is embedded in the model pipeline so transformations are fitted
within training folds and not from the held-out test set.

No SMOTE was applied. The initial imbalance strategy uses balanced class weights
where applicable and scale weighting for XGBoost experiments.

## 9. Exploratory Data Analysis

EDA produced 16 charts and a Markdown findings report. Observed univariate and
bivariate patterns include:

- Overtime historical attrition rate: 30.53% versus 10.44% without overtime.
- Sales had the highest department-level attrition rate: 20.63%.
- Sales Representative had the highest reported job-role attrition rate: 39.76%.
- Employees in the 0–2-year tenure band had an observed attrition rate of 29.82%.
- Job Satisfaction and Work-Life Balance showed variation across category bands.

These findings are descriptive associations and do not establish causes or
intervention effects.

## 10. Feature Engineering

The implemented engineered features are:

| Feature | Definition |
|---|---|
| IncomePerTotalWorkingYear | MonthlyIncome divided by TotalWorkingYears plus one |
| PromotionDelayRatio | YearsSinceLastPromotion divided by YearsAtCompany plus one |
| RoleTenureRatio | YearsInCurrentRole divided by YearsAtCompany plus one |
| OvertimeFlag | 1 for overtime Yes, otherwise 0 |
| EngagementComposite | Mean of five involvement, satisfaction, environment, relationship, and work-life fields |

The Age, Gender, and MaritalStatus fields are excluded from operational model
inputs and retained for audit purposes only.

## 11. Machine Learning Methodology

The model development candidates were:

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Histogram Gradient Boosting
- XGBoost

Model comparison used five-fold stratified cross-validation on the 1,176-record
training partition. SMOTE was not applied.

## 12. Model Comparison

The Phase 4 out-of-fold training comparison is below.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7619 | 0.3797 | 0.7474 | 0.5035 | 0.8344 | 0.6263 | 0.7560 |
| Gradient Boosting | 0.8648 | 0.6240 | 0.4105 | 0.4952 | 0.8117 | 0.5685 | 0.6814 |
| XGBoost | 0.8622 | 0.6045 | 0.4263 | 0.5000 | 0.8027 | 0.5657 | 0.6863 |
| Random Forest | 0.8554 | 0.5641 | 0.4632 | 0.5087 | 0.8064 | 0.5630 | 0.6971 |
| Decision Tree | 0.8061 | 0.4000 | 0.4000 | 0.4000 | 0.6422 | 0.2569 | 0.6422 |
| Dummy | 0.8384 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1616 | 0.5000 |

These are cross-validation training-partition estimates, not held-out test results.

## 13. Hyperparameter Optimization

Phase 5 used bounded randomized search with five-fold stratified cross-validation
and 10 iterations per candidate. The measured tuned PR-AUC values were:

| Model | Baseline PR-AUC | Tuned PR-AUC | Tuned Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.6263 | 0.6553 | 0.4158 |
| Histogram Gradient Boosting | 0.5685 | 0.5868 | 0.5158 |
| XGBoost | 0.5657 | 0.5974 | 0.3316 |

Logistic Regression became the final selected candidate by tuned training PR-AUC.
The tuned Logistic Regression parameters were `C = 0.6071989493` and no model
class weighting.

## 14. Probability Calibration

The candidate held-out test Brier scores were:

- Logistic Regression: 0.101431
- XGBoost: 0.107622
- Gradient Boosting: 0.123742

Training-only out-of-fold calibration selection compared sigmoid and isotonic
calibration. Isotonic calibration had the lower training OOF Brier score:

- Sigmoid: 0.0912806
- Isotonic: 0.0883740

The final Logistic Regression pipeline therefore uses isotonic calibration.
This selection was made before held-out test evaluation.

## 15. Risk Scoring Framework

The classification threshold is 0.21. It differs from the following operational
risk-category boundaries:

- Low: probability below 30%
- Medium: probability 30% through 60%, inclusive
- High: probability above 60%

Threshold 0.21 was selected using training OOF data with a minimum-recall objective
of 65% and then the highest F1 among eligible thresholds. These values are
operational assumptions, not universal scientific boundaries.

## 16. Explainable AI Using SHAP

SHAP explanations use `LinearExplainer` on the underlying tuned Logistic
Regression pipeline. The global output contains 294 held-out rows and 50
transformed features.

The highest observed mean absolute SHAP feature was RoleTenureRatio (0.509322),
followed by StockOptionLevel (0.435854), TotalWorkingYears (0.427211),
YearsInCurrentRole (0.420218), NumCompaniesWorked (0.407509), and OvertimeFlag
(0.407226).

SHAP does not explain the isotonic calibration transformation exactly. It explains
the linear logistic score; calibrated probabilities are reported separately.
SHAP values are associative and cannot establish causality.

## 17. Fairness and Responsible AI

The 294-record held-out test set was audited across Age, Gender, and MaritalStatus.
Sensitive fields are excluded from operational model inputs and SHAP reasons.

Principal measured differences:

| Audit | Demographic parity difference | Equal opportunity difference |
|---|---:|---:|
| AgeGroup | 0.2143 | 0.5000 |
| Gender | 0.0211 | 0.1714 |
| MaritalStatus | 0.2500 | 0.2500 |

The 51+ age group had recall 0.1667 versus 0.6667 for 18–30 and 0.6429 for
31–40. The Single marital-status group had a false-positive rate of 0.2361,
compared with 0.0435 for Married and 0.0500 for Divorced.

These are descriptive held-out observations. Small subgroups and historical
organizational bias limit causal or legal interpretation.

## 18. Experimental Results

The final calibrated selected model's held-out results are:

| Metric | Result |
|---|---:|
| Accuracy | 0.8469387755 |
| Precision | 0.5192307692 |
| Recall | 0.5744680851 |
| F1 | 0.5454545455 |
| ROC-AUC | 0.8035145146 |
| PR-AUC | 0.5653826702 |
| Balanced Accuracy | 0.7366267551 |
| Specificity | 0.8987854251 |
| Calibrated Brier score | 0.1014904169 |

At the threshold 0.21, the held-out confusion matrix was:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 222 | 25 |
| Actual 1 | 20 | 27 |

## 19. Discussion

The tuned Logistic Regression approach improved training PR-AUC and preserved a
strong held-out ROC-AUC. The held-out test also retained a substantial number of
false positives and false negatives, especially when applying the 0.21 threshold.

The system is more useful as a prioritization aid than an employee-level truth
mechanism. Risk categories should trigger human-reviewed retention research rather
than punitive or automated action.

## 20. Limitations

- The 1,470-record dataset is small relative to many production workforces.
- Dataset provenance beyond the supplied CSV is not verified.
- No outcome horizon of 30, 60, or 90 days exists in the supplied target.
- The held-out set is one split, not external temporal validation.
- SHAP associations are not causal estimates.
- Calibration can degrade under data drift.
- Fairness subgroup samples are limited and may be unstable.
- Proxy variables were not formally counterfactually audited.

## 21. Ethical Considerations

The platform separates sensitive demographic fields from operational model inputs
and provides demographic audit metrics. Predictions must not be used to automate
termination or opportunity denial. Retention support, if used, should be voluntary,
supportive, and human-reviewed.

## 22. Future Work

Potential future work includes external dataset validation, time-stamped outcomes,
calibration monitoring, subgroup uncertainty estimation, proxy-feature analysis,
drift detection, model-version tracking, and prospective business impact evaluation.

## 23. Conclusion

The project delivers a complete, tested employee attrition analytics workflow.
The final calibrated Logistic Regression model achieved PR-AUC 0.5654 and recall
0.5745 on the held-out test set. These results support analytical decision support
but do not justify employment decisions or exact future exit forecasts.

## 24. References

No external scholarly results are claimed or formally cited. The project does not
invent causal findings or experimental outcomes.

Suggested technical and methodology resources for independent review:

- scikit-learn documentation for preprocessing, calibration, and cross-validation.
- XGBoost documentation for gradient-boosted tree implementation.
- SHAP documentation for model explanation methods.
- Streamlit documentation for application delivery.
- Joblib documentation for artifact serialization.

These are suggested implementation references, not evidence supporting project
model metrics or causal conclusions.
