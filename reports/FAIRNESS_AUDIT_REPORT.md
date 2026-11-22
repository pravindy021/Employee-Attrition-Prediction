# Fairness Audit Report

## Scope

- Held-out test records: **294**
- Operational model inputs: Age, Gender, and MaritalStatus excluded
- Demographic fields: used only for subgroup audit
- Classification threshold: **0.21**
- Outcome: historical Attrition target

This audit describes model performance differences across observed groups.
It does not prove discrimination or establish causal bias.

## Metric definitions

- Selection/prediction rate: positive predictions divided by subgroup size.
- TPR/recall: positive actuals correctly predicted positive.
- FPR: negative actuals incorrectly predicted positive.
- FNR: positive actuals incorrectly predicted negative.
- Precision: positive predictions that were actual positives.
- Demographic parity difference: largest minus smallest selection rate.
- Equal opportunity difference: largest minus smallest TPR/recall.

Equal opportunity differences include only groups with at least 20 records and at least one actual positive in this implementation.

## AgeGroup

| Group | Size | Selection rate | Recall/TPR | FPR | FNR | Precision |
|---|---:|---:|---:|---:|---:|---:|
| 31-40 | 126 | 0.1508 | 0.6429 | 0.0893 | 0.3571 | 0.4737 |
| 18-30 | 77 | 0.2857 | 0.6667 | 0.1695 | 0.3333 | 0.5455 |
| 41-50 | 63 | 0.1429 | 0.5556 | 0.0741 | 0.4444 | 0.5556 |
| 51+ | 28 | 0.0714 | 0.1667 | 0.0455 | 0.8333 | 0.5000 |

- Demographic parity difference: **0.2143**
- Equal opportunity difference: **0.5000**
- False-positive-rate difference: **0.1240**
- Precision difference: **0.0819**

## Gender

| Group | Size | Selection rate | Recall/TPR | FPR | FNR | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Male | 178 | 0.1685 | 0.5161 | 0.0952 | 0.4839 | 0.5333 |
| Female | 116 | 0.1897 | 0.6875 | 0.1100 | 0.3125 | 0.5000 |

- Demographic parity difference: **0.0211**
- Equal opportunity difference: **0.1714**
- False-positive-rate difference: **0.0148**
- Precision difference: **0.0333**

## MaritalStatus

| Group | Size | Selection rate | Recall/TPR | FPR | FNR | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Married | 133 | 0.1053 | 0.5000 | 0.0435 | 0.5000 | 0.6429 |
| Single | 97 | 0.3299 | 0.6000 | 0.2361 | 0.4000 | 0.4688 |
| Divorced | 64 | 0.0938 | 0.7500 | 0.0500 | 0.2500 | 0.5000 |

- Demographic parity difference: **0.2361**
- Equal opportunity difference: **0.2500**
- False-positive-rate difference: **0.1926**
- Precision difference: **0.1741**

## Observed findings

The findings above are descriptive only. They summarize the held-out subset and do not indicate whether group differences are statistically significant or caused by model bias.

## Bias investigation

- Compare false-positive and false-negative rates before considering action.
- Small subgroups can produce unstable percentages.
- Missing demographic values may indicate data-quality or reporting issues.
- Protected attributes are excluded from the model; proxy variables were not formally audited for indirect discrimination.
- Feature importance and group performance do not establish causal discrimination.

## Responsible-AI recommendations

- Do not automate termination or opportunity decisions.
- Require human review and contextual employee information.
- Use supportive and voluntary retention conversations.
- Recalculate subgroup metrics for future datasets and update thresholds only after documented review.
- Report uncertainty and small-sample limitations.
- Continue to audit indirect proxy effects with appropriate legal and data-science review.

## Limitations

- The 294-record holdout is small for demographic subgroup analysis.
- Age bands are post hoc categories and may conceal within-band differences.
- Missing demographic categories should not be inferred.
- Historical outcomes may reflect organizational bias.
- A single threshold can distribute errors unevenly across groups.
