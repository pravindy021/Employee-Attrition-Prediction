# Data Dictionary

## Dataset summary

- Source file: `data/raw/Palo Alto Networks.csv`
- Original file: `data/Palo Alto Networks.csv`
- Records: 1,470
- Columns: 31
- Missing cells: 0
- Duplicate rows: 0

The supplied dataset is not represented as verified Palo Alto Networks employee data.

## Core fields

| Column | Type | Description |
|---|---|---|
| Age | Numeric | Employee age |
| Attrition | Binary target | 1 for historical attrition; 0 otherwise |
| BusinessTravel | Categorical | Travel frequency |
| Department | Categorical | Department |
| DistanceFromHome | Numeric | Distance from home |
| Education | Ordinal-like numeric | Education level code |
| EducationField | Categorical | Education field |
| EnvironmentSatisfaction | Ordinal-like numeric | Environment satisfaction score |
| Gender | Categorical audit-only | Reported gender |
| JobInvolvement | Ordinal-like numeric | Involvement score |
| JobLevel | Ordinal-like numeric | Job level |
| JobRole | Categorical | Job role |
| JobSatisfaction | Ordinal-like numeric | Satisfaction score |
| MaritalStatus | Categorical audit-only | Marital status |
| MonthlyIncome | Numeric | Monthly income |
| OverTime | Categorical | Whether overtime is recorded |
| TotalWorkingYears | Numeric | Total work experience |
| WorkLifeBalance | Ordinal-like numeric | Work-life balance score |
| YearsAtCompany | Numeric | Tenure in years |
| YearsInCurrentRole | Numeric | Time in current role |
| YearsSinceLastPromotion | Numeric | Time since promotion |
| YearsWithCurrManager | Numeric | Time with current manager |

## Target distribution

| Target | Count | Percentage |
|---|---:|---:|
| 0 | 1,233 | 83.88% |
| 1 | 237 | 16.12% |

The complete observed columns and unique counts are available in
`reports/data_dictionary.csv`.
