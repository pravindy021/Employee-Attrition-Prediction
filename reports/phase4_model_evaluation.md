# Phase 4 Machine Learning Model Development

## Scope

This report uses only the 1,176-record training partition. The 294-record test partition remains untouched for later final evaluation.

## Method

- Stratified cross-validation folds: **5**
- Random seed: **42**
- Preprocessing: existing feature engineering plus train-fold-only imputation, scaling, and encoding
- Class imbalance: initial balanced class weights or equivalent XGBoost weighting
- SMOTE: **not applied**
- XGBoost available: **yes**

## Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced accuracy | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.7619 | 0.3797 | 0.7474 | 0.5035 | 0.8344 | 0.6263 | 0.7560 | 0.7647 |
| gradient_boosting | 0.8648 | 0.6240 | 0.4105 | 0.4952 | 0.8117 | 0.5685 | 0.6814 | 0.9523 |
| xgboost | 0.8622 | 0.6045 | 0.4263 | 0.5000 | 0.8027 | 0.5657 | 0.6863 | 0.9462 |
| random_forest | 0.8554 | 0.5641 | 0.4632 | 0.5087 | 0.8064 | 0.5630 | 0.6971 | 0.9310 |
| decision_tree | 0.8061 | 0.4000 | 0.4000 | 0.4000 | 0.6422 | 0.2569 | 0.6422 | 0.8844 |
| dummy_baseline | 0.8384 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1616 | 0.5000 | 1.0000 |

## Selection

Using out-of-fold training results, the highest measured PR-AUC model is **logistic_regression** (PR-AUC **0.6263**, recall **0.7474**).

This is a Phase 4 candidate selection only. It has not been evaluated on the held-out test set. Accuracy was not used as the sole ranking criterion because attrition is imbalanced.

## Business interpretation

- Recall is important because missed at-risk employees are a high-cost false negative.
- Precision and specificity protect against excessive false alerts.
- PR-AUC is especially relevant when the positive class is minority.
- Final model suitability also requires calibration, threshold analysis, fairness auditing, and held-out test evaluation in later phases.

## Leakage controls

- The test partition was not loaded for model fitting or cross-validation.
- Preprocessing is embedded in a pipeline and is re-fitted inside each training fold.
- Sensitive fields remain excluded from model inputs.
- The target column is not an input feature.

## Limitations

- Cross-validation results estimate training-fold performance, not final test performance.
- Feature associations do not establish causation.
- Probability thresholds have not yet been optimized.
- SHAP, fairness, and final risk thresholds are separate later phases.
