# Project Walkthrough

## 1. Introduction

Introduce the project as an end-to-end employee attrition analytics platform.
State clearly that it analyzes historical patterns and does not predict exact
future departures.

## 2. Launch the dashboard

From the project root in Windows PowerShell:

```powershell
.\\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Deployment preparation and the dashboard entry point are documented in
`DEPLOYMENT_GUIDE.md`.

## 3. Open Executive Overview

Begin with:

- 1,470 employee records
- 237 historical attrition records
- 16.12% historical attrition rate
- department distributions
- held-out predicted risk distribution

Explain the distinction between historical attrition and model-predicted risk.

## 4. Demonstrate Employee Risk Assessment

Use supported operational fields only:

- Department
- JobRole
- OverTime
- MonthlyIncome
- JobSatisfaction
- WorkLifeBalance
- YearsAtCompany
- TotalWorkingYears
- YearsInCurrentRole
- DistanceFromHome

Do not enter Age, Gender, or MaritalStatus as operational prediction inputs.
These fields are separated for fairness auditing.

After calculation, explain:

- calibrated propensity;
- Low/Medium/High category;
- 0.21 classification threshold prediction;
- underlying Logistic Regression SHAP associations;
- calibration limitation.

## 5. Demonstrate Department Analytics

Show department employee counts, historical attrition rates, and held-out
department predicted propensity. Use the department filter for comparison.

## 6. Demonstrate Workforce Insights

Use overtime, job role, tenure, satisfaction, and work-life patterns.
State that charts show associations rather than causal effects.

## 7. Demonstrate Explainability Center

Show:

- global SHAP importance;
- SHAP summary and bar plots;
- logistic coefficient comparison;
- individual waterfall examples;
- the distinction between logistic score SHAP and isotonic probabilities.

## 8. Demonstrate Fairness

Use Age, Gender, and MaritalStatus subgroup tables and visualizations.
Explain selection rate, recall, FPR, FNR, and small-subgroup warnings.
Emphasize that subgroup differences do not prove discrimination.

## 9. Demonstrate What-If

Change OverTime, JobRole, tenure, or satisfaction and compare baseline/scenario
propensity. Clearly label the change as hypothetical and non-causal.

## 10. Demonstrate Model Performance

Present the held-out metrics:

- PR-AUC 0.5654
- Recall 0.5745
- F1 0.5455
- Accuracy 0.8469
- ROC-AUC 0.8035
- Calibrated Brier score 0.1015

Explain that these are one held-out test split and not proof of future production
performance.

## 11. Close the demonstration

Summarize responsible-use limits, data provenance limitations, and next steps:
external validation, time-stamped outcomes, monitoring, and subgroup review.
