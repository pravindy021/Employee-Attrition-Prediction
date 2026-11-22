# Business Case Study Scenarios

## 1. A department has increasing historical attrition

**Business problem:** Leaders need to distinguish a general historical trend from
one department's current workforce context.

**Data needed:** time-stamped department records, headcount, exits, hires,
roles, workload, tenure, pay, manager changes, and survey context.

**Approach:** compare historical rates over time and use cohort analysis rather
than an aggregate snapshot.

**Responsible action:** investigate workload, role clarity, development, manager
quality, and retention supports with department and HR representatives.

**Limitation:** correlation and historic attrition do not prove a cause.

## 2. A group has high predicted propensity

**Business problem:** The model highlights records that meet a high-risk threshold.

**Data needed:** recent model inputs, probability history, SHAP context, and
voluntary employee feedback.

**Approach:** review the calibrated probability, 0.21 classification threshold,
Low/Medium/High category, and individual feature contributions.

**Responsible action:** offer a voluntary, private support conversation and
validate the model with updated outcomes.

**Limitation:** a high probability does not establish that a person intends to leave.

## 3. High recall but low precision

**Business problem:** Analysts detect many actual attrition cases but generate
many false alerts.

**Data needed:** confusion-matrix outcomes, alert costs, reviewer capacity,
and actual follow-up results.

**Approach:** vary the threshold, inspect precision-recall trade-offs, and
document the cost of false positives and false negatives.

**Responsible action:** use the model as a prioritization aid, not an adverse
decision rule.

**Limitation:** lowering the threshold can increase support workload and employee anxiety.

## 4. A demographic subgroup has different error rates

**Business problem:** Accuracy or recall varies across observed Age, Gender,
or MaritalStatus subgroups.

**Data needed:** sufficient subgroup sample sizes, actual outcomes, proxy features,
collection quality, and model probabilities.

**Approach:** inspect selection rate, TPR, FPR, FNR, precision, demographic
parity difference, and equal opportunity difference.

**Responsible action:** escalate to HR, legal, and Responsible AI review;
increase data quality and consider model or policy changes only after documented evaluation.

**Limitation:** metric differences do not prove discrimination and small samples
can be unstable.

## 5. HR wants reasons behind risk scores

**Business problem:** A leader needs understandable model associations.

**Data needed:** transformed feature names, underlying SHAP values, calibrated
probability, and source employee context.

**Approach:** explain global and individual SHAP for the underlying Logistic
Regression score while reporting calibrated probability separately.

**Responsible action:** use the explanation to guide respectful questions and
contextual retention planning.

**Limitation:** SHAP does not explain the exact isotonic transformation and
does not prove causation.

## 6. A manager wants termination based on the model

**Business problem:** A manager may treat a risk score as an adverse performance
indicator.

**Data needed:** no additional model output is sufficient for an employment decision.
Job performance, policy, and local legal review require separate processes.

**Approach:** reject the requested use and explain that the model is a historical
propensity estimate at 0.21 threshold, not a fact or resignation forecast.

**Responsible action:** document and halt punitive use, provide human oversight,
and use supportive voluntary retention discussions only.

**Limitation:** using the model for automated or manager-driven termination can
create legal, ethical, privacy, and fairness harm.
