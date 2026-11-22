# Final Project Audit

## 1. Project overview

**Employee Attrition Intelligence Platform** is a completed 16-phase analytical
project covering data validation, EDA, preprocessing, feature engineering,
model comparison, tuning, calibration, held-out evaluation, risk scoring,
SHAP explainability, fairness auditing, Streamlit delivery, testing,
deployment preparation, documentation, and interview preparation.

This audit verifies the repository as it exists. It does not add a public
deployment claim or modify completed model/data outputs.

## 2. Phase completion summary

Phases 1–15 have implemented project work in the repository. Phase 16 is an
evidence audit only. No model retraining or original-data modification occurred
during this audit.

## 3. Dataset integrity

Verified properties:

| Check | Actual result |
|---|---|
| Records | 1,470 |
| Columns | 31 |
| Missing cells | 0 |
| Duplicate rows | 0 |
| Attrition 0 | 1,233 |
| Attrition 1 | 237 |
| Historical attrition rate | 16.12% |
| Training/test sizes | 1,176 / 294 |
| Train/test overlap | None |

The original and raw-copy CSV SHA-256 hashes match:

`389735e8909136a51fe487040df8f8662c2d222bb3505a83745c5a8c5bf20c7c`

The dataset has no verified actual Palo Alto Networks provenance.

## 4. Model validation

The final selected pipeline is a calibrated Logistic Regression model.

Verified held-out test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.8469387755 |
| Precision | 0.5192307692 |
| Recall | 0.5744680851 |
| F1 | 0.5454545455 |
| ROC-AUC | 0.8035145146 |
| PR-AUC | 0.5653826702 |
| Balanced accuracy | 0.7366267551 |
| Specificity | 0.8987854251 |
| Calibrated Brier score | 0.1014904169 |

The classification threshold is 0.21. Risk categories remain:

- Low: probability below 0.30
- Medium: probability from 0.30 through 0.60
- High: probability above 0.60

The 0.21 threshold is separate from 30/60 risk boundaries.

Isotonic calibration was selected over sigmoid using lower training OOF Brier
score. The test set was not used for tuning or preprocessing fit.

## 5. Explainability

Verified SHAP artifacts exist and describe the underlying tuned Logistic
Regression pipeline with:

- SHAP method: `LinearExplainer`
- Held-out explanation rows: 294
- Transformed feature count: 50
- SHAP output shape: 294 × 50

Global and individual explanation files load. Age, Gender, and MaritalStatus
are excluded from operational inputs and explanations.

The documentation correctly distinguishes underlying Logistic Regression SHAP
values from isotonic-calibrated probabilities.

## 6. Fairness audit

Age, Gender, and MaritalStatus subgroup metrics are present in
`reports/DEMOGRAPHIC_METRICS.csv`.

Key measured differences:

| Audit | Demographic parity difference | Equal opportunity difference |
|---|---:|---:|
| Age | 0.2143 | 0.5000 |
| Gender | 0.0211 | 0.1714 |
| MaritalStatus | 0.2361 | 0.2500 |

The 51+ age group had the lowest observed recall at 0.1667. The Single marital
group had the highest observed false-positive rate at 0.2361.

These are descriptive held-out results, not proof of discrimination or
causal organizational bias. Small subgroup sample sizes require caution.

## 7. Dashboard validation

Streamlit AppTest loaded `app.py` with no script exceptions. The navigation
radio and dashboard startup interface were detected.

The eight implemented pages are:

1. Executive Overview
2. Employee Risk Assessment
3. Department Analytics
4. Workforce Insights
5. Explainability Center
6. Fairness & Responsible AI
7. What-If Scenario Lab
8. Model Performance

The dashboard loads saved data, models, risk outputs, SHAP outputs, and fairness
reports. It does not retrain models during startup.

Local Streamlit HTTP smoke testing previously returned HTTP 200. No public URL
was verified.

## 8. Test execution summary

Final measured test results:

| Test type | Result |
|---|---|
| pytest full suite | 73 passed |
| unittest suite | 18 passed |
| Artifact verifier | 12 required artifacts verified |
| Deployment verifier | 17 required files verified |
| Dependency checker | No broken requirements |
| Streamlit AppTest | No script exceptions |
| Markdown link check | No broken links among audited documents |
| Secret pattern scan | No matches found |

The pytest and unittest counts are separate runner results and were not summed.

## 9. Security audit

- No credential or secret pattern matches were found in the audited source tree.
- `.venv`, caches, `.env`, secrets, and local sensitive artifact patterns are ignored.
- `.env.example` contains no real secret.
- The original employee CSV remains unchanged.
- Age, Gender, and MaritalStatus are restricted to audit use.
- No public employee identity fields or fake employee identities are used in
  production dashboard views.

## 10. Deployment readiness

The project is prepared for Streamlit Community Cloud:

- Runtime: Python 3.12.10
- Entry point: `app.py`
- Streamlit configuration: `.streamlit/config.toml`
- Required-file verification: passed
- Repository size excluding `.venv`: approximately 17.99 MB
- Largest required model artifact: approximately 6.34 MB
- Deployment status: preparation complete
- Public deployment status: pending account and GitHub access

No public URL is claimed.

## 11. Known limitations

- One historical dataset and one held-out split are not external validation.
- Data provenance is not independently verified.
- No exact 30/60/90-day outcome horizon exists.
- Held-out subgroup sample sizes are limited.
- SHAP explains the underlying linear model score, not exact isotonic calibration.
- Fairness differences do not prove discrimination or causation.
- AppTest is startup-focused rather than a complete browser interaction suite.
- Public deployment and real-account access were not available.

## 12. Remaining manual actions

1. Initialize or connect a GitHub remote repository.
2. Review the uncommitted Git status and commit the project.
3. Push to GitHub.
4. Create a Streamlit Community Cloud app.
5. Select `app.py` and verify the public URL.
6. Run a browser-level prediction, SHAP, and fairness smoke test.
7. Add outcome monitoring and external validation only after real data arrives.

## 13. Final GitHub readiness assessment

**GitHub readiness:** Ready for manual review and publishing.

Local Git metadata is initialized. The final Git status remains uncommitted,
as no commit or push was requested. Required data and model artifacts are present,
local caches and `.venv` are ignored, and no secrets or public URL were created.
