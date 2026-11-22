# Phase 5 Hyperparameter Optimization

## Scope

- Data partition: 1,176-record training partition only
- Held-out test partition: untouched
- Search method: RandomizedSearchCV
- Iterations per candidate: 10
- Cross-validation: 5-fold stratified
- Primary optimization metric: PR-AUC
- Secondary metrics: recall, ROC-AUC, F1, balanced accuracy

## Candidate models

- logistic_regression
- gradient_boosting
- xgboost

These candidates were selected from the Phase 4 PR-AUC ranking.

## Baseline and tuned comparison

| Model | Baseline PR-AUC | Tuned PR-AUC | Baseline recall | Tuned recall | Baseline ROC-AUC | Tuned ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.6263 | 0.6553 | 0.7474 | 0.4158 | 0.8344 | 0.8426 |
| gradient_boosting | 0.5685 | 0.5868 | 0.4105 | 0.5158 | 0.8117 | 0.8166 |
| xgboost | 0.5657 | 0.5974 | 0.4263 | 0.3316 | 0.8027 | 0.8159 |

## Best tuned candidate

- Model: **logistic_regression**
- Tuned PR-AUC: **0.6553**
- Tuned recall: **0.4158**
- Baseline PR-AUC: **0.6263**
- Best parameters: `{"classifier__C": 0.6071989493441299, "classifier__class_weight": null}`

## Leakage controls

- Search folds use only the training partition.
- Feature engineering, imputation, scaling, and encoding are inside the pipeline.
- The held-out test partition is not used for fitting or cross-validation.
- Sensitive fields remain excluded from model inputs.
- SMOTE remains disabled; class imbalance is handled by class weighting or XGBoost weighting.

## Limitations

- Randomized search samples a bounded subset of the possible parameter space.
- Tuned cross-validation metrics are not held-out test metrics.
- Probability calibration and threshold optimization occur in later phases.
