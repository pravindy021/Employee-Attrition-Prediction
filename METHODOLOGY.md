# Machine Learning Methodology

## Dataset and target

The supplied dataset contains 1,470 records and 31 columns. The binary target is
`Attrition`: 1 for historical attrition and 0 for no recorded attrition.

## Split and cross-validation

- Training: 1,176 records
- Held-out test: 294 records
- Stratified 80/20 split
- Fixed random seed: 42
- Five-fold stratified cross-validation during model development and tuning

The held-out test set was not used for hyperparameter tuning or Phase 5 model selection.

## Preprocessing

- Numeric median imputation
- Numeric standard scaling
- Categorical most-frequent imputation
- Categorical one-hot encoding
- Unknown-category handling
- Train-fold-only fitting of preprocessing transformations

## Feature engineering

The implementation includes:

- `IncomePerTotalWorkingYear`
- `PromotionDelayRatio`
- `RoleTenureRatio`
- `OvertimeFlag`
- `EngagementComposite`

These are exploratory business features, not proven causal measures.

## Model development

Candidates included dummy baseline, Logistic Regression, Decision Tree,
Random Forest, Histogram Gradient Boosting, and XGBoost. Bounded randomized
search was used for the promising Logistic Regression, Histogram Gradient
Boosting, and XGBoost candidates.

## Calibration and threshold

The selected model is tuned Logistic Regression with isotonic calibration.
The classification threshold is 0.21.

The risk categories are:

- Low: probability below 0.30
- Medium: probability 0.30 through 0.60
- High: probability above 0.60
