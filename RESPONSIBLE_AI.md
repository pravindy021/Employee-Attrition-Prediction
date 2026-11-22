# Responsible AI

## Core policy

Employee attrition estimates are decision-support data, not facts about an employee.
The system must not automate termination, demotion, compensation, promotion, or
opportunity denial.

## Human review

All retention follow-up must include contextual employee information and human
review. Conversations should be voluntary and supportive.

## Sensitive features

Age, Gender, and MaritalStatus are retained only for fairness auditing. They are
excluded from model inputs and operational SHAP explanations.

## Fairness

The 294-record held-out set was audited across Age, Gender, and MaritalStatus.
Differences in selection, recall, false-positive, false-negative, and precision
metrics are descriptive. They do not by themselves prove discrimination.

## Data and model limitations

- The dataset has no verified actual company provenance.
- The target describes a historical outcome, not a fixed future time horizon.
- Historical outcomes may encode organizational bias.
- Small subgroup results can be unstable.
- SHAP associations are not causal.

## Governance

- Revalidate after workforce or policy changes.
- Monitor probability drift and actual outcomes.
- Recalculate fairness metrics when sufficient outcomes are available.
- Escalate unexplained subgroup differences to HR and legal review.
- Preserve a route for employee questions and correction.
