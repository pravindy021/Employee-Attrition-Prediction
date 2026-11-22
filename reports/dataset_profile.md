# Employee Attrition Dataset Profiling Report

## Scope

This Phase 1 report describes the supplied CSV dataset. The dataset is treated as a company-inspired case study; no claim is made that it contains verified Palo Alto Networks employee data.

## Dataset Summary

- Records: **1470**
- Columns: **31**
- Duplicate rows: **0**
- Missing cells: **0**
- Attrition count (1): **237**
- Non-attrition count (0): **1233**
- Historical attrition rate: **16.12%**
- Validation status: **PASS**

## Validation Findings

- No validation issues detected.

## Column Types

| Column | Observed dtype | Role | Missing | Unique |
|---|---|---|---:|---:|
| Age | int64 | numerical | 0 | 43 |
| Attrition | int64 | target | 0 | 2 |
| BusinessTravel | str | categorical | 0 | 3 |
| DailyRate | int64 | numerical | 0 | 886 |
| Department | str | categorical | 0 | 3 |
| DistanceFromHome | int64 | numerical | 0 | 29 |
| Education | int64 | numerical | 0 | 5 |
| EducationField | str | categorical | 0 | 6 |
| EnvironmentSatisfaction | int64 | numerical | 0 | 4 |
| Gender | str | categorical | 0 | 2 |
| HourlyRate | int64 | numerical | 0 | 71 |
| JobInvolvement | int64 | numerical | 0 | 4 |
| JobLevel | int64 | numerical | 0 | 5 |
| JobRole | str | categorical | 0 | 9 |
| JobSatisfaction | int64 | numerical | 0 | 4 |
| MaritalStatus | str | categorical | 0 | 3 |
| MonthlyIncome | int64 | numerical | 0 | 1349 |
| MonthlyRate | int64 | numerical | 0 | 1427 |
| NumCompaniesWorked | int64 | numerical | 0 | 10 |
| OverTime | str | categorical | 0 | 2 |
| PercentSalaryHike | int64 | numerical | 0 | 15 |
| PerformanceRating | int64 | numerical | 0 | 2 |
| RelationshipSatisfaction | int64 | numerical | 0 | 4 |
| StockOptionLevel | int64 | numerical | 0 | 4 |
| TotalWorkingYears | int64 | numerical | 0 | 40 |
| TrainingTimesLastYear | int64 | numerical | 0 | 7 |
| WorkLifeBalance | int64 | numerical | 0 | 4 |
| YearsAtCompany | int64 | numerical | 0 | 37 |
| YearsInCurrentRole | int64 | numerical | 0 | 19 |
| YearsSinceLastPromotion | int64 | numerical | 0 | 16 |
| YearsWithCurrManager | int64 | numerical | 0 | 18 |

## Categorical Cardinality

| Column | Unique values |
|---|---:|
| BusinessTravel | 3 |
| Department | 3 |
| EducationField | 6 |
| Gender | 2 |
| JobRole | 9 |
| MaritalStatus | 3 |
| OverTime | 2 |

## Important Limitations

- This is a profiling report, not an attrition analysis or causal study.
- Numeric-looking category fields may require semantic review.
- Attribute names and target values do not establish provenance or causality.
