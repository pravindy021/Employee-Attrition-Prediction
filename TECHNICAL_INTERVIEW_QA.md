# Technical Interview Q&A

## 1. Python and Pandas

**1. How did you structure the Python project?**
The project separates loading and validation, EDA, feature engineering,
preprocessing, model development, explainability, fairness, dashboard utilities,
tests, scripts, reports, and configuration.

**2. Why use Pandas for this project?**
Pandas provides DataFrame operations for loading, profiling, grouping, filtering,
data validation, and report tables.

**3. How did you preserve the original dataset?**
I loaded the CSV read-only and made a separate `data/raw/` copy. The original and
raw copy remained SHA-256 identical.

**4. How did you handle missing values?**
The current dataset had no missing cells. The pipeline still uses median
imputation for numeric fields and most-frequent imputation for categorical fields.

**5. How did you ensure reproducibility?**
Fixed random seed 42, a fixed 80/20 split, stratified sampling, pipeline
serialization, pinned dependencies, and documented commands.

## 2. Data preprocessing

**6. Why use ColumnTransformer?**
It applies appropriate transformations to numeric and categorical columns through
one sklearn preprocessing object.

**7. Why fit preprocessing only on training data?**
Fitting on test data leaks test distribution statistics into model training.

**8. Which scaling method was used?**
StandardScaler for numeric features.

**9. Why one-hot encode categorical features?**
The selected Logistic Regression and several tree or boosting implementations
require numeric inputs. One-hot encoding represents categories without assuming order.

**10. How are unknown categories handled?**
The OneHotEncoder uses `handle_unknown="ignore"`.

## 3. Exploratory Data Analysis

**11. What were the key EDA findings?**
Overtime had 30.53% historical attrition versus 10.44% without overtime; Sales
had the highest department rate at 20.63%; Sales Representative had 39.76%.

**12. Why are EDA findings not causal?**
EDA measures historical associations in an observational dataset and cannot
isolate intervention effects.

**13. How did you measure class imbalance?**
The target had 237 positive and 1,233 negative records, a 16.12% positive rate.

**14. Which plots answered business questions?**
Target counts, categorical attrition rates, tenure bands, satisfaction heatmaps,
correlation heatmaps, and department-role comparisons.

**15. How did you treat IQR outliers?**
They were reported and reviewed. The project did not automatically remove valid
historical values.

## 4. Feature engineering

**16. Which engineered features did you create?**
IncomePerTotalWorkingYear, PromotionDelayRatio, RoleTenureRatio, OvertimeFlag,
and EngagementComposite.

**17. Why not simply add arbitrary features?**
Each feature has a documented formula, business reasoning, and limitation.

**18. What is EngagementComposite?**
The mean of five employee involvement, satisfaction, environment, relationship,
and work-life fields.

**19. What are sensitive features?**
Age, Gender, and MaritalStatus are kept separately for fairness auditing.

**20. Did feature engineering use target information?**
No. The implementations use available employee fields and Attrition is excluded.

## 5. Classification algorithms

**21. Which algorithms did you compare?**
Dummy baseline, Logistic Regression, Decision Tree, Random Forest, Histogram
Gradient Boosting, and XGBoost.

**22. Why include a dummy baseline?**
It establishes minimum majority-class behavior for comparison.

**23. Why evaluate more than accuracy?**
The minority attrition class is 16.12%, so recall, precision, PR-AUC, and F1
are more informative than accuracy alone.

**24. Which model was initially strongest?**
Logistic Regression had the highest Phase 4 out-of-fold PR-AUC at 0.6263.

**25. Did you choose a model using the test set?**
No. Candidate and tuning decisions used the training partition; held-out test
evaluation happened in Phase 6.

## 6. Logistic Regression

**26. Why did you select Logistic Regression?**
It had the highest tuned training PR-AUC among candidates and is linear,
interpretable, and compatible with Linear SHAP.

**27. How does it produce probabilities?**
It applies a logistic transformation to a linear score.

**28. What does a coefficient mean?**
Its sign shows the directional association for one transformed feature unit,
holding other inputs constant in the model.

**29. Does coefficient sign imply causation?**
No.

**30. Why is scaling useful for comparison?**
StandardScaler puts numeric features on comparable standard-deviation scales.

## 7. Random Forest

**31. How does Random Forest reduce variance?**
It averages or votes across multiple decorrelated decision trees.

**32. What settings did you use?**
300 estimators, minimum leaf size 2, balanced class weights, and a fixed random
seed in the baseline.

**33. Why not necessarily select it?**
Its Phase 4 PR-AUC was 0.5630 and recall 0.4632, below Logistic Regression.

**34. Does Random Forest handle one-hot categories?**
Yes, after preprocessing converts categories to numeric indicators.

**35. What is its limitation here?**
It can be less transparent and may require more computation than Logistic Regression.

## 8. XGBoost and Gradient Boosting

**36. How does gradient boosting train?**
It adds sequential weak learners that correct previous residual errors.

**37. Which XGBoost weighting was used?**
The baseline used a scale-pos-weight based on negative/positive class counts.

**38. What did tuned XGBoost achieve in training?**
Tuned training PR-AUC 0.5974 and recall 0.3316.

**39. What did histogram gradient boosting achieve when tuned?**
Training PR-AUC 0.5868 and recall 0.5158.

**40. Why was XGBoost not final?**
Its tuned training PR-AUC remained below tuned Logistic Regression 0.6553.

## 9. Cross-validation

**41. Which cross-validation method was used?**
Five-fold StratifiedKFold with shuffle and random seed 42.

**42. Why stratified folds?**
They preserve attrition class proportions across folds.

**43. How are preprocessing and leakage controlled?**
Preprocessing is in the sklearn pipeline and re-fitted within each training fold.

**44. Which partition was used for tuning?**
Only the 1,176-record training partition.

**45. Which partition was held out?**
The 294-record test partition.

## 10. Hyperparameter tuning

**46. Why RandomizedSearchCV?**
It explores useful parameter combinations with bounded computation.

**47. How many iterations were used?**
10 randomized iterations per promising candidate.

**48. What was the primary metric?**
PR-AUC, followed by recall and ROC-AUC as supporting measures.

**49. What improved after tuning?**
Logistic training PR-AUC rose from 0.6263 to 0.6553.

**50. Why not tune on test data?**
That would leak test outcomes into model selection and invalidate final evaluation.

## 11. Class imbalance

**51. What is the positive-class prevalence?**
237/1,470 = 16.12%.

**52. How was imbalance addressed?**
Balanced class weights or equivalent weighting in candidate models.

**53. Was SMOTE used?**
No. It was intentionally not applied.

**54. Could SMOTE be considered later?**
Only inside training folds, never before splitting or as a final test transformation.

**55. Why not use accuracy alone?**
The dummy classifier has 83.84% accuracy but 0 recall and 0.1616 PR-AUC.

## 12. Metrics

**56. Why is recall important?**
It measures how many actual attrition records the system identifies.

**57. Why is precision important?**
It measures how many positive predictions were actual positive records.

**58. Why use PR-AUC?**
It is informative for performance on the minority attrition class.

**59. Why use ROC-AUC?**
It evaluates ranking across thresholds independent of one fixed threshold.

**60. What is F1?**
The harmonic mean of precision and recall.

## 13. Probability calibration

**61. Why calibrate probabilities?**
Raw model scores can have poor probability meaning and operational thresholds.

**62. Which calibration methods were compared?**
Sigmoid and isotonic.

**63. Why was isotonic selected?**
Its training OOF Brier score was 0.088374 versus 0.091281 for sigmoid.

**64. What is Brier score?**
Mean squared difference between predicted probability and the actual binary outcome.

**65. Did held-out calibration improve Brier substantially?**
Calibrated test Brier was 0.101490 versus raw logistic test Brier 0.101431.

## 14. Threshold and risk

**66. Why is the classification threshold 0.21?**
Training OOF trade-off selection used minimum recall 65%, then the highest F1
eligible threshold.

**67. What are Low/Medium/High boundaries?**
Below 30%, 30–60%, and above 60%.

**68. Are these scientific boundaries?**
No. They are configurable operational assumptions.

**69. Why separate threshold and risk category?**
0.21 determines positive prediction in the confusion matrix; 30/60 create user-facing
probability bands.

**70. What happens when threshold decreases?**
Recall generally increases while false positives can increase.

## 15. Risk scoring

**71. What is employee risk scoring?**
An anonymous record receives a calibrated propensity, category, and feature explanation.

**72. What did the held-out calibrated model produce?**
222 true negatives, 25 false positives, 20 false negatives, and 27 true positives.

**73. Is probability a certainty?**
No.

**74. Can the score identify a specific resignation date?**
No. No 30/60/90-day horizon exists.

**75. What is a responsible use?**
Human-reviewed analytical discussion and voluntary support planning.

## 16. SHAP

**76. Which SHAP explainer was used?**
LinearExplainer for the fitted underlying Logistic Regression.

**77. What shape did SHAP produce?**
294 held-out records by 50 transformed features.

**78. Which feature ranked highest?**
RoleTenureRatio with mean absolute SHAP 0.509322.

**79. Does SHAP explain isotonic calibration?**
No exact calibrated-probability explanation was implemented.

**80. Is SHAP causal?**
No.

## 17. Fairness

**81. Which demographic fields were audited?**
Age groups, Gender, and MaritalStatus.

**82. Which metrics were calculated?**
Selection rate, TPR/recall, FPR, FNR, precision, accuracy, demographic parity
difference, and equal opportunity difference.

**83. Which model fields are sensitive?**
Age, Gender, and MaritalStatus are separate from operational inputs.

**84. What subgroup differences were observed?**
Age equal-opportunity difference 0.5000, Gender 0.1714, and MaritalStatus 0.2500.

**85. Do these differences prove discrimination?**
No. They describe held-out model performance and require deeper review.

## 18. Data leakage

**86. How was test leakage prevented?**
The test set was not used for tuning or preprocessing fit.

**87. How was target leakage prevented?**
Attrition was excluded from model inputs.

**88. Which fields can proxy protected attributes?**
The project did not formally counterfactually audit proxies.

**89. How was preprocessing leakage prevented?**
The pipeline is fit inside cross-validation training folds.

**90. How is split leakage tested?**
Phase 11 verifies train and test indices are disjoint.

## 19. Streamlit and deployment

**91. What is the dashboard entry point?**
`app.py`.

**92. How many dashboard pages exist?**
Eight.

**93. How are artifacts loaded?**
Using project-relative pathlib, cached Streamlit loaders, and joblib/Pandas.

**94. Did the dashboard retrain models?**
No.

**95. Was public deployment verified?**
No. Only local smoke testing and deployment preparation were completed.

## 20. Model monitoring and limitations

**96. What monitoring would you add?**
Probability drift, feature drift, calibration drift, outcome monitoring, and
subgroup metrics after actual outcomes arrive.

**97. What is the main generalization limitation?**
The 294-record holdout is limited and not external organizational validation.

**98. What would improve the project?**
Time-stamped outcomes, external data, larger subgroup samples, and prospective
impact evaluation.

**99. What should happen when a high score appears?**
Human HR review, contextual discussion, and supportive optional action planning.

**100. How should the model never be used?**
It must not be used to automate termination or deny promotion, compensation, or opportunity.
