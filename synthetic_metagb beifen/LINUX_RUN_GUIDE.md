# Synthetic MetaGB Linux Server Run Guide

## 1. Environment Verification

Run from the repository root. Use the modified local XGBoost checkout, not the
PyPI `xgboost` package.

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=beforereview/xgboost-master/python-package python3 -c \
  "import xgboost; print(xgboost.__version__, xgboost.__file__)"
PYTHONPATH=beforereview/xgboost-master/python-package python3 -m pytest \
  synthetic_metagb/test_scenarios.py synthetic_metagb/test_protocol.py \
  tests/test_random_search_contract.py \
  beforereview/xgboost-master/tests/python/test_metagb.py -q
```

The printed XGBoost path must point inside this repository.

## 2. Protocols

All protocols use three models with equal candidate budgets:

- `xgb_cpu`
- `current_metagb_pred0`
- `current_metagb`

Randomness is separated into `search_seed`, `data_seed`, `cv_seed`, and
`model_seed`. Statistics use independent data seeds as the pairing unit.

### Smoke

The smoke high-dimensional scenario uses 10 nuisance features so that it
validates the high-dimensional code path without turning the framework check
into a formal-scale run. Random300 keeps the required 50 nuisance features.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=beforereview/xgboost-master/python-package python3 \
  synthetic_metagb/run_benchmark.py \
  --protocol smoke \
  --search-seed 42 \
  --task-workers 15 \
  --n-jobs 1 \
  --run-id smoke_server
```

### Random300 Confirmatory

```bash
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=beforereview/xgboost-master/python-package python3 \
  synthetic_metagb/run_benchmark.py \
  --protocol random300_confirmatory \
  --search-seed 42 \
  --task-workers 15 \
  --n-jobs 1 \
  --run-id random300_confirmatory_v1 \
  > synthetic_metagb/random300_confirmatory_v1.log 2>&1 &
```

### Random600 Confirmatory

```bash
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=beforereview/xgboost-master/python-package python3 \
  synthetic_metagb/run_benchmark.py \
  --protocol random600_confirmatory \
  --search-seed 42 \
  --task-workers 15 \
  --n-jobs 1 \
  --run-id random600_confirmatory_v1 \
  > synthetic_metagb/random600_confirmatory_v1.log 2>&1 &
```

For the requested 16-core configuration, use exactly `task_workers=15` and
`n_jobs=1`. Fifteen independent resumable tasks run concurrently, while each
XGBoost fit stays single-threaded. One core remains available for the
operating system and orchestration process. Do not run Random300 and Random600
concurrently.

## 3. Computation Units and Exact Fit Counts

The smallest resumable task is:

```text
scenario x data_seed x outer_fold x model
```

For one task, all candidates are selected using the inner folds, followed by
one final fit on the complete outer-training fold:

```text
fits_per_task = candidates_per_model x inner_folds + 1 final fit
```

| Protocol | Scenarios | Data seeds | Outer x inner | Candidates/model | Tasks | Candidate rows | Inner fits | Final fits | Total fits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke | 11 | 2 | 2 x 2 | 30 | 132 | 3,960 | 7,920 | 132 | 8,052 |
| Random300 | 17 | 10 | 5 x 4 | 300 | 2,550 | 765,000 | 3,060,000 | 2,550 | 3,062,550 |
| Random600 | 6 | 10 | 5 x 4 | 600 | 900 | 540,000 | 2,160,000 | 900 | 2,160,900 |

The model-level and data-level decomposition is:

| Unit | Random300 | Random600 |
| --- | ---: | ---: |
| One resumable task | 300 candidate rows, 1,201 fits | 600 candidate rows, 2,401 fits |
| One scenario x data seed, all models/folds | 15 tasks, 4,500 candidate rows, 18,015 fits | 15 tasks, 9,000 candidate rows, 36,015 fits |
| One complete scenario, 10 seeds | 150 tasks, 45,000 candidate rows, 180,150 fits | 150 tasks, 90,000 candidate rows, 360,150 fits |
| One model over the complete protocol | 850 tasks, 255,000 candidate rows, 1,020,850 fits | 300 tasks, 180,000 candidate rows, 720,300 fits |

The three models have equal candidate budgets. Random300 therefore evaluates
`300 candidates x 3 models x 17 scenarios x 10 seeds x 5 outer folds =
765,000` candidate rows. Every row requires 4 inner-fold fits.

Because a shard is written only after a complete task, interruption can lose
at most the currently running task: 1,201 fits for Random300 or 2,401 fits for
Random600. Completed task shards are preserved by `--resume`.

## 4. Measured Minimum-Unit Runtime

The completed local smoke used `n_jobs=1`. Its measurements provide
smoke-scale lower bounds for a single inner-fold model fit:

| Model | Normal scenarios mean | Normal median | Smoke high-dim mean | Smoke high-dim median |
| --- | ---: | ---: | ---: | ---: |
| `xgb_cpu` | 0.138 s | 0.109 s | 0.105 s | 0.098 s |
| `current_metagb_pred0` | 0.245 s | 0.186 s | 0.463 s | 0.395 s |
| `current_metagb` | 1.176 s | 0.710 s | 7.436 s | 4.109 s |

The table below reports the measured mean time of the smallest resumable task
for each smoke dataset and model. One smoke task contains 30 candidates x 2
inner folds plus one final fit. Times include fit and prediction work and were
measured with `n_jobs=1`.

| Scenario | `xgb_cpu` | `current_metagb_pred0` | `current_metagb` |
| --- | ---: | ---: | ---: |
| `global_linear` | 10.6 s | 18.1 s | 73.0 s |
| `regime_slope` | 9.8 s | 15.2 s | 72.4 s |
| `feature_switch` | 11.0 s | 19.0 s | 77.1 s |
| `multiplicative` | 9.5 s | 14.7 s | 69.2 s |
| `pure_step_control` | 9.6 s | 16.3 s | 77.4 s |
| `curved_control` | 10.1 s | 16.2 s | 70.6 s |
| `xor_classification` | 3.5 s | 11.7 s | 66.9 s |
| `regime_logit` | 4.0 s | 14.8 s | 77.9 s |
| `regime_slope__small_n` | 8.2 s | 13.7 s | 63.9 s |
| `multiplicative__high_noise` | 9.5 s | 14.0 s | 65.9 s |
| `regime_logit__high_dim` | 6.6 s | 29.1 s | 449.0 s |

This shows that `current_metagb`, especially high-dimensional MetaGB,
dominates wall time. A smoke high-dimensional MetaGB task is already about
6.2x slower than an ordinary MetaGB task.

One smoke task contains 61 fits. Random300 and Random600 tasks contain 1,201
and 2,401 fits, respectively. Holding dataset size and per-candidate
complexity fixed, candidate-count-only `n_jobs=1` task lower bounds are:

| Model/task type | Random300 task | Random600 task |
| --- | ---: | ---: |
| Ordinary `xgb_cpu` | 2.8 min | 5.6 min |
| Ordinary `current_metagb_pred0` | 5.0 min | 10.1 min |
| Ordinary `current_metagb` | 23.4 min | 46.8 min |
| Smoke-scale high-dimensional `current_metagb` | 2.46 h | not in Random600 |

The formal protocols use 3,000 samples instead of smoke's 600 samples.
Random300 high-dimensional variants also use 50 nuisance features instead of
the smoke variant's 10. Therefore the table above is a minimum-unit lower
bound, not a formal-runtime prediction.

## 5. Sixteen-Core, Fifteen-Task Parallel Runtime Planning

With `task_workers=15, n_jobs=1`, up to 15 independent tasks execute at once.
This avoids nested thread oversubscription and dynamically balances ordinary
tasks against slower high-dimensional MetaGB tasks.

Use:

```text
T_task ~= (candidate_count x inner_folds x t_inner_fit_1) + t_final_fit_1
T_protocol_wall ~= total_single_thread_task_work / effective_parallel_workers
```

The measured smoke contains 5,717 CPU-seconds of serial fit work. A
fit-count-only extrapolation, before accounting for larger formal datasets,
gives:

| Effective parallel workers | Random300 lower bound | Random600 lower bound |
| ---: | ---: | ---: |
| 9 workers, 60% utilization | 2.8 days | 2.0 days |
| 12 workers, 80% utilization | 2.1 days | 1.5 days |
| 15 workers, ideal balance | 1.7 days | 1.2 days |

These remain optimistic lower bounds. Random300's 50-feature high-dimensional
tasks can be substantially slower than the table suggests.

Measure the target server before the formal runs:

```bash
time env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=beforereview/xgboost-master/python-package python3 \
  synthetic_metagb/run_benchmark.py \
  --protocol smoke \
  --data-seeds 999 \
  --search-seed 42 \
  --task-workers 15 \
  --n-jobs 1 \
  --run-id smoke_15tasks_capacity
```

Compare its `metadata.json` elapsed time with the local one-seed preflight
reference: 951.6 seconds using 4 task workers with `n_jobs=1`. Then run a
formal-scale one-seed Random300 pilot; it is exactly 10% of the complete
Random300 protocol:

```bash
time env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=beforereview/xgboost-master/python-package python3 \
  synthetic_metagb/run_benchmark.py \
  --protocol random300_confirmatory \
  --data-seeds 1001 \
  --search-seed 42 \
  --task-workers 15 \
  --n-jobs 1 \
  --run-id random300_15tasks_capacity
```

The pilot contains 255 tasks, 76,500 candidate rows, and 306,255 fits. Its
elapsed time multiplied by approximately 10 is the best initial full
Random300 estimate on the target server.

## 6. Monitoring and Resume

```bash
tail -f synthetic_metagb/random300_confirmatory_v1.log
find synthetic_metagb/results/random300_confirmatory_v1/tasks -name '*.json' | wc -l
```

Resume after interruption with the exact same protocol and seeds:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=beforereview/xgboost-master/python-package python3 \
  synthetic_metagb/run_benchmark.py \
  --protocol random300_confirmatory \
  --search-seed 42 \
  --task-workers 15 \
  --n-jobs 1 \
  --run-id random300_confirmatory_v1 \
  --resume
```

Resume refuses to continue if the protocol configuration hash differs.

## 7. Outputs and Integrity Checks

Each run directory contains:

- `protocol.json`: frozen protocol and config hash.
- `candidate_configs.json`: complete model candidate universes.
- `candidate_fairness_audit.json`: equal-budget and paired-backbone audit.
- `parameter_audit.json`: searched, fixed, and default parameters.
- `split_manifest.json`: shared split hashes and leakage checks.
- `tasks/*.json`: resumable task shards.
- `candidates.csv`: all inner-CV candidate evaluations.
- `folds.csv`: selected-candidate outer-test results.
- `seed_results.csv`: outer-fold results averaged by independent data seed.
- `paired_statistics.csv`: bootstrap CI, Wilcoxon, Holm correction, win rate,
  relative improvement, and rank-biserial effect size.
- `completeness_audit.json`: expected versus actual task and candidate counts.
- `metadata.json`: final run hashes and completion status.

Check completion:

```bash
python3 -m json.tool synthetic_metagb/results/random300_confirmatory_v1/completeness_audit.json
python3 -m json.tool synthetic_metagb/results/random300_confirmatory_v1/metadata.json
```

Both must report successful completion and matching expected/actual counts.

## 8. Results to Return

After each formal run, return these files:

```text
protocol.json
candidate_fairness_audit.json
parameter_audit.json
split_manifest.json
completeness_audit.json
metadata.json
seed_results.csv
paired_statistics.csv
```

Smoke output is explicitly non-confirmatory. Only the Random300 and Random600
protocols are intended for formal advantage claims.
