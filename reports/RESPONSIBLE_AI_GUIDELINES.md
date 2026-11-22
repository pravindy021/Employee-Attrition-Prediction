# Responsible AI Guidelines

## Prediction use

- Attrition scores are estimates of historical propensity, not facts.
- The model does not provide a 30/60/90-day resignation prediction.
- The system must not automate termination, demotion, or opportunity denial.
- Human reviewers must consider job context and alternative explanations.

## Supportive interventions

- Retention discussions must be voluntary and supportive.
- Managers must not use risk scores as proof of dissatisfaction or disloyalty.
- Organizational interventions should address workload, development, pay
  fairness, and role fit only where lawful and appropriate.

## Demographic data

- Age, Gender, and MaritalStatus are audit-only fields.
- They must not be exposed as operational explanation reasons.
- Demographic data must not be used to recommend adverse employment actions.
- Subgroup metrics require privacy-aware aggregation and secure handling.

## Governance

- Revalidate performance after data, policy, or workforce changes.
- Document threshold changes and their subgroup effects.
- Escalate unexplained subgroup differences to HR, legal, and Responsible AI review.
- Preserve human oversight and provide a route for employee questions.

## Monitoring

- Monitor data drift, probability drift, and outcome drift.
- Recalculate fairness metrics when outcomes become available.
- Small subgroup sizes require uncertainty and result suppression review.
- Do not interpret a fairness metric difference as proof of unlawful discrimination.
