# Feature Engineering and Preprocessing Report

## Scope

This report documents Phase 3 only. No model was trained and no dashboard was built.
The original CSV was not modified.

## Dataset split

- Source records: **1,470**
- Training records: **1,176**
- Test records: **294**
- Split method: **stratified train-test split**
- Test fraction: **20%**
- Random seed: **42**

## Target handling

- Target: `Attrition`
- Class 0: no attrition
- Class 1: attrition
- Initial class-imbalance policy: **balanced class weights**
- SMOTE: **not applied**

SMOTE will be considered only later inside cross-validation folds, where it can be isolated from validation data. The EDA majority/minority ratio is 5.20:1.

## Sensitive feature separation

- Audit-only fields: `Age`, `Gender`, `MaritalStatus`
- These fields are excluded from the model input matrix.
- They may be used separately for fairness auditing where legally and ethically appropriate.
- They must not be used to recommend adverse employment actions.

## Excluded columns

- Identifier columns: **none detected**
- Constant columns: **none detected**
- All excluded columns: **Age, Attrition, Gender, MaritalStatus**

## Numeric preprocessing

- Missing values: median imputation
- Scaling: standardization with `StandardScaler`
- Numeric features: `DailyRate`, `DistanceFromHome`, `Education`, `EnvironmentSatisfaction`, `HourlyRate`, `JobInvolvement`, `JobLevel`, `JobSatisfaction`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `PercentSalaryHike`, `PerformanceRating`, `RelationshipSatisfaction`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, `WorkLifeBalance`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`, `IncomePerTotalWorkingYear`, `PromotionDelayRatio`, `RoleTenureRatio`, `OvertimeFlag`, `EngagementComposite`

## Categorical preprocessing

- Missing values: most-frequent imputation
- Encoding: one-hot encoding
- Unknown categories: ignored at transform time
- Categorical features: `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `OverTime`

## Business-oriented engineered features

| Feature | Definition | Business rationale | Limitation |
|---|---|---|---|
| IncomePerTotalWorkingYear | MonthlyIncome / (TotalWorkingYears + 1) | Approximate income relative to career length. | Ratios can be unstable for very short careers. |
| PromotionDelayRatio | YearsSinceLastPromotion / (YearsAtCompany + 1) | Approximate share of tenure without promotion. | A delay is not necessarily the result of poor performance or unfair treatment. |
| RoleTenureRatio | YearsInCurrentRole / (YearsAtCompany + 1) | Approximate proportion of tenure spent in the current role. | Organizational roles and promotion definitions vary. |
| OvertimeFlag | 1 when OverTime is Yes, otherwise 0 | Overtime had a strong observed attrition association in EDA. | Association is not evidence of causation. |
| EngagementComposite | Mean of five satisfaction/involvement ratings | Summarizes related experience measures. | The score assumes approximately comparable ordinal scales. |

## Leakage prevention

- The data is split before fitting the preprocessing pipeline.
- The feature engineer and all imputers, scalers, and encoders are fit on training data only.
- The test set is only transformed after fitting.
- The target column is never included in model inputs.
- Sensitive fields are withheld from model inputs and stored separately for auditing.

## Reproducibility

- Fixed random seed: 42
- Fixed test fraction: 20%
- Stratified split preserves target proportions.
- The saved sklearn pipeline is serializable and reusable for unseen records.

## Limitations

- The engineered engagement score is exploratory rather than a clinically or scientifically validated measure.
- Correlations do not establish causal retention drivers.
- Protected attributes may reproduce organizational bias if used inappropriately.
- No transformed output should be interpreted as an individual employment decision.
