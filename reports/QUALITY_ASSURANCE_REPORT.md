# Quality Assurance Report

## Scope

Phase 11 added comprehensive integration, leakage, dashboard, and reliability
coverage without modifying trained model parameters, calibration, threshold,
or the original employee CSV.

## Integrity

- Original CSV SHA-256: `389735e8909136a51fe487040df8f8662c2d222bb3505a83745c5a8c5bf20c7c`
- Original and raw-copy hashes match.
- 12 required project artifacts were found and hashed.
- Train/test indices do not overlap.
- Sensitive attributes are audit-only.

## Quality improvements from testing

- Fixed Streamlit deprecation usage from `use_container_width` to `width="stretch"`.
- Fixed AppTest paths to use the project-relative absolute app entry point.
- Added strict model probability shape validation.
- Added repeatable model-loading and prediction smoke measurements.

## Known limitations

- Streamlit AppTest performs startup-level checks, not a complete browser interaction suite.
- Timing observations are machine-dependent.
- The small single holdout can be sensitive to subgroup sample size.
- Tests confirm technical consistency but do not prove real-world fairness or clinical validity.
- No external deployment or real employee source-system integration was tested.

## Remaining issues

- Real-browser automated interaction coverage is limited.
- Dashboard accessibility auditing is not yet included.
- Production monitoring and external data validation remain future work.
