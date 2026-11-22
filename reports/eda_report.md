# Exploratory Data Analysis Report

## Scope and interpretation

This report summarizes observed patterns in the supplied dataset. Associations are predictive and exploratory; they do not establish causation.

## 1. Target and class imbalance

- Total employees: **1,470**
- Historical attrition (1): **237**
- No attrition (0): **1,233**
- Historical attrition rate: **16.12%**
- Majority/minority ratio: **5.20:1**

The target is moderately imbalanced. Accuracy alone will be insufficient during later modeling; recall, PR-AUC, F1, balanced accuracy, and false-positive cost should be considered.

## 2. Key business findings

- Employees with overtime have an observed attrition rate of **30.53%**, compared with **10.44%** for employees without overtime.
- **Sales** has the highest department-level attrition rate in this data: **20.63%**.
- **Sales Representative** has the highest job-role attrition rate: **39.76%**.
- These findings describe historical association only and should not be used to make individual employment decisions.

## 3. Numerical patterns

| Feature | Left median | Stayed median | Mann-Whitney p-value |
|---|---:|---:|---:|
| Age | 32.00 | 36.00 | 5.30434e-11 |
| MonthlyIncome | 3202.00 | 5204.00 | 2.95083e-14 |
| YearsAtCompany | 3.00 | 6.00 | 2.91619e-13 |
| TotalWorkingYears | 7.00 | 10.00 | 2.39957e-14 |

The p-values indicate statistical evidence of distribution differences, but not practical importance or causation.

## 4. Categorical patterns

| Feature | Cramer's V | Chi-square p-value |
|---|---:|---:|
| OverTime | 0.2427 | 8.15842e-21 |
| JobRole | 0.2307 | 2.75248e-15 |
| JobSatisfaction | 0.0994 | 0.0005563 |
| WorkLifeBalance | 0.0952 | 0.00097257 |
| Department | 0.0774 | 0.00452561 |
| Gender | 0.0089 | 0.290572 |

## 5. Attrition rates by selected features

### Department

| Group | Employees | Attrition count | Attrition rate |
|---|---:|---:|---:|
| Sales | 446 | 92 | 20.63% |
| Human Resources | 63 | 12 | 19.05% |
| Research & Development | 961 | 133 | 13.84% |

### JobRole

| Group | Employees | Attrition count | Attrition rate |
|---|---:|---:|---:|
| Sales Representative | 83 | 33 | 39.76% |
| Laboratory Technician | 259 | 62 | 23.94% |
| Human Resources | 52 | 12 | 23.08% |
| Sales Executive | 326 | 57 | 17.48% |
| Research Scientist | 292 | 47 | 16.10% |
| Manufacturing Director | 145 | 10 | 6.90% |
| Healthcare Representative | 131 | 9 | 6.87% |
| Manager | 102 | 5 | 4.90% |
| Research Director | 80 | 2 | 2.50% |

### JobLevel

| Group | Employees | Attrition count | Attrition rate |
|---|---:|---:|---:|
| 1.0 | 543 | 143 | 26.34% |
| 3.0 | 218 | 32 | 14.68% |
| 2.0 | 534 | 52 | 9.74% |
| 5.0 | 69 | 5 | 7.25% |
| 4.0 | 106 | 5 | 4.72% |

### OverTime

| Group | Employees | Attrition count | Attrition rate |
|---|---:|---:|---:|
| Yes | 416 | 127 | 30.53% |
| No | 1054 | 110 | 10.44% |

### JobSatisfaction

| Group | Employees | Attrition count | Attrition rate |
|---|---:|---:|---:|
| 1.0 | 289 | 66 | 22.84% |
| 3.0 | 442 | 73 | 16.52% |
| 2.0 | 280 | 46 | 16.43% |
| 4.0 | 459 | 52 | 11.33% |

### WorkLifeBalance

| Group | Employees | Attrition count | Attrition rate |
|---|---:|---:|---:|
| 1.0 | 80 | 25 | 31.25% |
| 4.0 | 153 | 27 | 17.65% |
| 2.0 | 344 | 58 | 16.86% |
| 3.0 | 893 | 127 | 14.22% |

### YearsAtCompanyBand

| Group | Employees | Attrition count | Attrition rate |
|---|---:|---:|---:|
| 0-2 | 342 | 102 | 29.82% |
| 3-5 | 434 | 60 | 13.82% |
| 6-10 | 448 | 55 | 12.28% |
| 21+ | 66 | 8 | 12.12% |
| 11-20 | 180 | 12 | 6.67% |

## 6. Numerical correlations with Attrition

| Feature | Pearson correlation |
|---|---:|
| TotalWorkingYears | -0.1711 |
| JobLevel | -0.1691 |
| YearsInCurrentRole | -0.1605 |
| MonthlyIncome | -0.1598 |
| Age | -0.1592 |
| YearsWithCurrManager | -0.1562 |
| StockOptionLevel | -0.1371 |
| YearsAtCompany | -0.1344 |

Correlation identifies linear association only. Several features are related to experience or seniority and may introduce redundant information.

## 7. Outlier review

| Feature | IQR lower bound | IQR upper bound | Outliers | Outlier percent |
|---|---:|---:|---:|---:|
| TrainingTimesLastYear | 0.50 | 4.50 | 238 | 16.19% |
| PerformanceRating | 3.00 | 3.00 | 226 | 15.37% |
| MonthlyIncome | -5291.00 | 16581.00 | 114 | 7.76% |
| YearsSinceLastPromotion | -4.50 | 7.50 | 107 | 7.28% |
| YearsAtCompany | -6.00 | 18.00 | 104 | 7.07% |
| StockOptionLevel | -1.50 | 2.50 | 85 | 5.78% |
| TotalWorkingYears | -7.50 | 28.50 | 63 | 4.29% |
| NumCompaniesWorked | -3.50 | 8.50 | 52 | 3.54% |
| YearsInCurrentRole | -5.50 | 14.50 | 21 | 1.43% |
| YearsWithCurrManager | -5.50 | 14.50 | 14 | 0.95% |
| EnvironmentSatisfaction | -1.00 | 7.00 | 0 | 0.00% |
| DistanceFromHome | -16.00 | 32.00 | 0 | 0.00% |

IQR outliers are observations that fall outside the simple rule; they are not automatically errors. Domain review is needed before removal.

## 8. Leakage, redundancy, and modeling exclusions

- No obvious identifier or direct future-outcome column was found in the inspected schema.
- Attrition is the target and must be excluded from model inputs.
- MonthlyIncome is strongly connected to job level and experience; redundancy should be checked during feature engineering.
- JobLevel, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, and YearsWithCurrManager may reflect overlapping career-history concepts.
- Age, Gender, and MaritalStatus are protected or sensitive attributes. They should not be used to recommend adverse employment actions. Use them only for separate fairness auditing where ethically and legally appropriate.
- Numeric satisfaction and rating fields should be handled carefully because their meaning depends on the survey or rating scale.

## 9. Figure inventory

- `01_target_distribution.png`: target class distribution
- `02_numerical_distributions.png`: numerical univariate distributions
- `03_categorical_distributions.png`: categorical univariate distributions
- `04_age_income_vs_attrition.png`: age and income versus outcome
- `05_gender_attrition_rate.png` through `11_worklifebalance_attrition_rate.png`: selected categorical rates
- `12_tenure_attrition_rate.png`: tenure band rates
- `13_correlation_heatmap.png`: numerical correlations
- `14_satisfaction_attrition_heatmap.png`: satisfaction interactions
- `15_experience_income_attrition.png`: experience and income relationship
- `16_department_role_attrition.png`: department and role comparison

## 10. Phase 2 limitations

- The data are historical and may reproduce organizational bias.
- Missingness is absent in the supplied file, but field quality still requires source validation.
- Statistical significance does not prove a feature is a retention intervention.
- Summary tables are descriptive; no assumptions of representativeness are made.
