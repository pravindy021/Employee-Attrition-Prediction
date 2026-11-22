# Phase 6 Model Validation and Employee Risk Scoring

## Scope and data

- Training rows used for threshold and calibration decisions: **1,176**
- Held-out test rows evaluated for the first time: **294**
- Candidate models: top tuned candidates by Phase 5 training PR-AUC
- Historical target: `Attrition = 1`
- No 30/60/90-day outcome horizon is present in the dataset.

This is an estimate of historical attrition propensity, not a time-bound prediction of who will leave within 30, 60, or 90 days.

## Held-out candidate validation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8673 | 0.6429 | 0.3830 | 0.4800 | 0.8064 | 0.5679 | 0.1014 |
| xgboost | 0.8605 | 0.6667 | 0.2553 | 0.3692 | 0.7809 | 0.5147 | 0.1076 |
| gradient_boosting | 0.8367 | 0.4878 | 0.4255 | 0.4545 | 0.7567 | 0.4886 | 0.1237 |

The highest held-out PR-AUC candidate in this validation run is **logistic_regression** at **0.5679**.

## Calibration

| Method | Training OOF Brier score |
|---|---:|
| sigmoid | 0.0913 |
| isotonic | 0.0884 |

Selected calibration method: **isotonic**

Calibration was selected using training-only out-of-fold predictions.
The held-out test set was used only to report final validation performance.

## Threshold decision

- Selected threshold: **0.21**
- Selection objective: minimum recall 65%, then highest training OOF F1
- Lower thresholds generally increase recall and false positives.
- Higher thresholds generally increase precision and missed-risk count.

## Initial risk categories

- LOW: probability below 30%
- MEDIUM: probability from 30% through 60%
- HIGH: probability above 60%

These are configurable operational assumptions, not scientific boundaries.
A probability is a relative estimate, not certainty.

## Employee risk distribution

- LOW: **255**
- MEDIUM: **19**
- HIGH: **20**

## Business and ethical interpretation

- Risk scores support HR research and human review, not automated employment decisions.
- Historical patterns may reproduce organizational bias.
- Age, Gender, and MaritalStatus remain separate from model inputs.
- The reports do not establish that a feature causes attrition.
- A high score does not determine that an employee will resign.

## Limitations

- The held-out data is a single temporal-independent split, not an external production sample.
- No survey source, outcome horizon, or company provenance beyond the supplied file was verified.
- Risk categories require stakeholder and legal review before operational use.
- Calibration and threshold behavior may change under future data drift.
