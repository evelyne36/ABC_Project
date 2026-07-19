# Time-Series Tree Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing time-series benchmark so MetaGB can be compared fairly with XGBoost, LightGBM, and CatBoost on fixed chronological sequence forecasting with one shared `window_summary` feature policy.

**Architecture:** Keep the current `timeseries_extra_random300` runner as the orchestration surface. Extend model names and candidate generation while preserving the existing fixed chronological split, sequence target construction, train-only scaling, and result/audit outputs.

**Tech Stack:** Python, NumPy, pandas, scikit-learn, local XGBoost/MetaGB adapter, optional LightGBM, optional CatBoost, pytest.

---

## Experimental Contract

Main benchmark:

```text
split_protocol = fixed
seeds = 0 1 2
feature_mode = window_summary
target_mode = sequence
output_scope = target
seq_lens = 12 24 48
pred_lens = 1 6 24
models = current_metagb xgb_cpu lgbm_cpu catboost_cpu
```

`window_summary` feature dimension for a dataset with `C` numeric channels is:

```text
features = seq_len * C + 9 * C + 6 + calendar_features
calendar_features <= 4
features_max = (seq_len + 9) * C + 10
```

At the preregistered maximum `seq_len=48`, this is `57*C+10`. Current low/mid-dimensional main-table candidates are:

| Dataset | Channels | Max `window_summary` columns |
|---|---:|---:|
| ETTh1 / ETTh2 / ETTm1 / ETTm2 / ILI | 7 | 409 |
| exchange_rate | 8 | 466 |
| Weather | 21 | 1207 |

PEMS-style wide datasets are excluded from the main table unless explicitly capped or moved to a separate smoke/appendix protocol, because `window_summary` reaches tens of thousands of columns.

## File Structure

- Modify `random600/utils/candidates.py`: generate shared tree backbones for `lgbm_cpu` and `catboost_cpu`, mapping the same abstract search space used by XGBoost and MetaGB.
- Modify `timeseries_extra_random300/scripts/run_timeseries_benchmark.py`: accept the two new model names and construct optional LightGBM/CatBoost regressors.
- Modify `timeseries_extra_random300/tests/test_runner_fixed_protocol.py`: lock model choices, task expansion, and sequence output behavior.
- Add or modify `random600/tests/test_candidates.py` if an existing candidate test file is present; otherwise add targeted candidate assertions to the time-series test file.
- Optionally update `timeseries_extra_random300/启动指南.md`: document the fixed sequence full command after smoke passes.

## Task 1: Candidate Generation Contract

**Files:**
- Modify: `random600/utils/candidates.py`
- Test: `timeseries_extra_random300/tests/test_runner_fixed_protocol.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting that `candidate_sets(..., models=["xgb_cpu", "lgbm_cpu", "catboost_cpu", "current_metagb"])` returns all four models, that each non-MetaGB candidate count equals `max_candidates`, and that generated candidates retain `backbone_id` for auditability.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python3 -m pytest timeseries_extra_random300/tests/test_runner_fixed_protocol.py -q
```

Expected: failure because `lgbm_cpu` and `catboost_cpu` are not generated.

- [ ] **Step 3: Implement minimal candidate generation**

Add `generate_lgbm_candidates` and `generate_catboost_candidates` using the same `_generate_shared_backbones` helper. Map:

```text
lambda_l2 -> reg_lambda for LightGBM
lambda_l2 -> l2_leaf_reg for CatBoost
```

Preserve `candidate_id`, `backbone_id`, and `source`.

- [ ] **Step 4: Run test to verify GREEN**

Run the same pytest command. Expected: pass.

## Task 2: Model Adapter Contract

**Files:**
- Modify: `timeseries_extra_random300/scripts/run_timeseries_benchmark.py`
- Test: `timeseries_extra_random300/tests/test_runner_fixed_protocol.py`

- [ ] **Step 1: Write failing tests**

Add tests that parser accepts:

```text
--models xgb_cpu current_metagb lgbm_cpu catboost_cpu
```

and that fixed protocol task expansion counts all four models across one dataset, one seed, one sequence length, and one horizon.

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
python3 -m pytest timeseries_extra_random300/tests/test_runner_fixed_protocol.py -q
```

Expected: failure because parser choices reject `lgbm_cpu` and `catboost_cpu`.

- [ ] **Step 3: Implement optional model constructors**

In `build_timeseries_estimator`, keep XGBoost/MetaGB routed through `random600.utils.runner.build_estimator`. For `lgbm_cpu`, import `lightgbm` inside the factory and instantiate `LGBMRegressor`. For `catboost_cpu`, import `CatBoostRegressor` inside the factory and set `verbose=False`. If packages are missing, raise an actionable `ImportError` naming the missing package and model.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
python3 -m pytest timeseries_extra_random300/tests/test_runner_fixed_protocol.py -q
```

Expected: pass.

## Task 3: Sequence Window-Summary Smoke

**Files:**
- Modify only if tests expose a bug.

- [ ] **Step 1: Run existing leakage and fixed protocol tests**

Run:

```bash
python3 -m pytest timeseries_extra_random300/tests/test_window_no_leakage.py timeseries_extra_random300/tests/test_runner_fixed_protocol.py -q
```

Expected: pass.

- [ ] **Step 2: Run a no-new-dependency smoke**

Run XGBoost/MetaGB only, since local LightGBM/CatBoost may be absent:

```bash
python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode smoke \
  --split-protocol fixed \
  --datasets ETTh1 \
  --models xgb_cpu current_metagb \
  --feature-mode window_summary \
  --target-mode sequence \
  --output-scope target \
  --seeds 0 \
  --seq-lens 12 \
  --pred-lens 1 \
  --max-raw-rows 160 \
  --max-candidates 10 \
  --metagb-fixed-score-agg 0 \
  --search-space-json timeseries_extra_random300/configs/smoke_search_space.json \
  --task-workers 1 \
  --n-jobs 1 \
  --run-id smoke_sequence_window_summary_tree_contract
```

Expected: `summary.csv` contains two ok rows and `input_features=157` for ETTh1.

## Task 4: Documentation and Server Command

**Files:**
- Modify: `timeseries_extra_random300/启动指南.md`

- [ ] **Step 1: Document feature dimension policy**

Add the `window_summary` dimension formula and main-table dataset guidance.

- [ ] **Step 2: Document full server command**

Add a server command using the experimental contract:

```bash
python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode full \
  --split-protocol fixed \
  --datasets exchange_rate ETTh1 ETTh2 Weather ILI ETTm1 ETTm2 \
  --models current_metagb xgb_cpu lgbm_cpu catboost_cpu \
  --feature-mode window_summary \
  --target-mode sequence \
  --output-scope target \
  --seeds 0 1 2 \
  --seq-lens 12 24 48 \
  --pred-lens 1 6 24 \
  --max-candidates 300 \
  --task-workers 15 \
  --n-jobs 1 \
  --progress-every 50 \
  --run-id ts_tree_sequence_window_summary_full
```

- [ ] **Step 3: Run smoke after documentation**

Repeat Task 3 smoke and verify outputs still work.

## Self-Review

- Spec coverage: the plan covers fixed split, 3 seeds, `window_summary`, sequence multi-horizon, three-by-three `seq_len × pred_len`, four tree models, and smoke verification.
- Placeholder scan: no placeholder steps are left.
- Type consistency: model names are consistently `xgb_cpu`, `current_metagb`, `lgbm_cpu`, and `catboost_cpu`; feature mode is consistently `window_summary`.
