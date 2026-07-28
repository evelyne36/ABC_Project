# Synthetic MetaGB Rigorous Smoke Report

## Scope

This smoke validates the experimental framework. It is explicitly
non-confirmatory and must not be used to claim statistical significance.

- Protocol: `smoke`
- Models: `xgb_cpu`, `current_metagb_pred0`, `current_metagb`
- Scenarios: 8 core plus 3 pressure variants
- Independent data seeds: 2
- Nested CV: 2 outer x 2 inner folds
- Candidate budget: 30 per model
- Search seed: 42
- Task workers: 4, estimator `n_jobs=1`

The smoke high-dimensional variant uses 10 nuisance features to keep the
framework check executable. The formal Random300 protocol retains 50 nuisance
features.

## Integrity Results

- Completed tasks: 132 / 132
- Candidate evaluation rows: 3,960 / 3,960
- Unique task IDs: 132
- Candidate fairness audit: passed
- Shared split and leakage audit: passed
- Fresh fixed-seed deterministic rerun: passed
- Resume audit: passed; rerun reported `pending_tasks=0`
- Candidate joint coverage: all 30 combinations of 10
  `pred_feature_fraction` values x 3 `metagb_score_agg` values
- Config hash:
  `e40c7f9ec21ea0d34a37080efc7589d065e322213041a29560347adb1efacc50`
- Candidate hash:
  `cf78df72c40791aa27a8757018bfe604894285c63a2355a1112600b2937d9232`
- Split manifest hash:
  `95423d77bdf8b1cf65cab47b9dbf84734b3c5af43c085a8c08acf0b8453cda65`

## Non-Confirmatory Observations

Against XGB, MetaGB had a positive mean paired advantage in 9 of 11 scenarios.
Against pred0, MetaGB also had a positive mean paired advantage in 9 of 11
scenarios.

The strongest observed advantage patterns were:

- `regime_slope` and `regime_slope__small_n`
- `multiplicative` and `multiplicative__high_noise`
- `regime_logit` and `regime_logit__high_dim`

Observed negative-control weaknesses were also visible:

- MetaGB lost to both references on `curved_control`.
- MetaGB lost to pred0 on `pure_step_control`.
- MetaGB was slightly below XGB on `xor_classification`.

With only two independent data seeds, Wilcoxon and Holm-adjusted results are
not informative; the generated statistics are correctly marked
`smoke_only_non_confirmatory_do_not_claim_significance`.

## Runtime

The complete smoke took 1,502.7 seconds (25.0 minutes) with 4 task workers:

- Model fits: 8,052
- Random300 fit-count-only lower bound on the same machine: about 6.6 days
- Random600 fit-count-only lower bound on the same machine: about 4.7 days

These are optimistic lower bounds. Formal protocols use larger datasets, and
Random300's 50-feature variants make MetaGB substantially more expensive.
Run the one-seed capacity pilot in `LINUX_RUN_GUIDE.md` before the complete
server experiments.

## Result Location

Final artifacts are in:

```text
synthetic_metagb/results/rigorous_smoke_20260614/
```
