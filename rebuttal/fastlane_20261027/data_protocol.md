# Fastlane data protocol, v2 fixed real test (2026-10-27)

`prepare_data.py` is independent of machine-specific root paths. Its default root is two parents above the script; `--root`, `--out`, and `--raw` override locations. It needs the existing NumPy and pandas runtime only. It never reads model scores. Old sparse generators, inputs, and preexperiments remain untouched.

## Reproduction

```sh
python rebuttal/fastlane_20261027/prepare_data.py --mode synthetic
python rebuttal/fastlane_20261027/prepare_data.py --mode real --download
python rebuttal/fastlane_20261027/prepare_data.py --mode cpu-stats
```

Archives already present in `local_sync/fastlane_data/raw` are reused. For portable transfer, copy this folder and the script; prebuilt NPZ files have no external path dependencies. Each NPZ has `X`, `y`, `train`, `val`, `test`, `task_type`, plus separate `y_train`, `y_val`, `y_test` members for worker isolation. Configuration search must read only `y_train` and `y_val`; combined `y` is retained for compatibility with capacity diagnostics. Index arrays select rows of the common X/y; preprocessing is already applied. JSON records source/archive/NPZ/array SHA-256, split sizes, preprocessing parameters, features, and grouping. `groups`, where present, are never inputs. `train_uncapped` is an audit array. For real regression, predictions in standardized target units convert to source units as `prediction * preprocessing.y_scale + preprocessing.y_mean`. Do not standardize these supplied inputs a second time.

Development seeds are **97001, 97002**. Confirmation seeds are **98001–98005**. Before generation, the script checks the prior `rebuttal/preexperiments_20261027` sources/configs and seed columns in CSVs for reuse. Decimal substrings in historical scores are not seed collisions. The executed check found no collisions. Confirmation seeds must remain excluded from development selection.

## Synthetic mechanisms and controls

All covariates are mutually independent standard normal. Observation noise is independent `N(0, .3²)`. Covariates are **not centered or scaled**: this preserves the specified zero-boundary slope mechanism. Every family/seed shares exactly the same X, noise realization, and split across sparse, dense, and offset controls. All controls have population signal variance 4; realized finite-sample variances need not equal exactly. The 60/20/20 split is a dedicated seeded random permutation, giving S1 144/48/48 and S2/S3 288/96/96. Generation uses versioned seed substreams. This is a new protocol, not pooled with old 75/25 sparse results.

| Family | n,d | Sparse signal (before noise) |
|---|---|---|
| S1 | 240,32 | `2 sign(x0) x1` (two regions) |
| S2 | 480,32 | `b[r] x[2+r]`, `r=2 I(x0>=0)+I(x1>=0)`, `b=[-2,+2,-2,+2]` |
| S3 | 480,16 | `2/sqrt(2) [sign(x0)x2 + sign(x1)x3]` (two additive sparse modules) |

Dense S1 replaces x1 by the normalized sum of all 31 nongate variables. Dense S2 partitions its 30 nongate variables into four disjoint blocks (8,8,7,7), replacing each region's predictive coordinate by that block's sum divided by the square root of its width. Dense S3 partitions its 14 nongate variables into two blocks of seven and does the same per module. Dense controls preserve each module's population slope norm and total signal variance.

Offset controls replace each normalized predictive scalar by 1 while retaining gates and coefficients. S1 is `2 sign(x0)`; S2 is `b[r]`; S3 is `2/sqrt(2)[sign(x0)+sign(x1)]`. These control whether region selection alone suffices. There are 3 families × 3 controls × 7 seeds = 63 datasets. `oracle_region`, `oracle_signal`, and `noise` are diagnostic arrays; JSON oracle support is descriptive. **Never append oracle arrays or support indicators to X.**

## Real datasets

For every dataset, fit numeric median imputation, feature mean/scale, and target mean/scale on the **capped training rows only**; then apply unchanged to validation/test. All methods receive the same representation. Centering changes the representable behavior of models whose leaves use slopes without a bias, so this preprocessing choice is explicitly part of the protocol. Training cap is 1800 uniformly sampled indices without reference to labels. Validation/test are uncapped. Validation/test label summaries and raw-label arrays are omitted. A capped NPZ intentionally leaves excluded training rows outside the three evaluation index arrays.

For non-gas real datasets, outer seed **96900** freezes one 20% test partition before development and confirmation. Each listed seed varies only the 75/25 train/validation partition within the remaining 80%, and the uniform train cap. Grouped datasets partition subjects or composition groups at both steps. Thus all development/confirmation repeats share a held-out test set; the five confirmation repeats are **not independent new datasets**. These v2 files replaced the original unfit v1 files before model fitting; the v1 per-seed real test partition must not be used.

| Dataset | Features and target | Split |
|---|---|---|
| CPU activity | existing local 21 numeric inputs; target usr | frozen outer row test; repeated train/validation in complement |
| Protein | existing `physiochemical_protein.csv`, 9 numeric inputs; target ResidualSize | frozen outer row test; repeated train/validation in complement |
| kin8nm | existing 8 inputs; target last column; **simulation provenance** | frozen outer row test; repeated train/validation in complement |
| UCI 551 Gas turbine | 9 original sensor inputs, target CO; exclude CO, NOX, year | train 2011–12, validation 2013, test 2014–15; uniform train cap only |
| UCI 189 Parkinson telemonitoring | exactly 16 voice inputs, total_UPDRS target; exclude subject ID, age, sex, test_time, motor_UPDRS | frozen outer **subject** test; repeated grouped train/validation in complement |
| UCI 464 Superconductivity | 81 extracted features, critical_temp target | frozen outer **composition group** test; repeated grouped train/validation in complement |

Superconductivity `train.csv` and `unique_m.csv` are checked for identical row counts and matching critical temperatures in row order. Groups use all elemental amounts in `unique_m.csv`, normalized by total atom count and represented to 12 significant digits. This groups reordered/equivalent and stoichiometric-multiple formulas; `material` strings and critical temperature are not grouping features. Group disjointness is asserted for train/validation/test. Gas turbine uses explicit year files; years are retained only for splitting and metadata.

Official public sources: [Gas turbine](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set), [Parkinson telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring), [Superconductivity](https://archive.ics.uci.edu/dataset/464/superconductivty+data). Download URLs and archive hashes are in each dataset's JSON. All three archives were retrieved from `archive.ics.uci.edu`; no alternate mirrors or credentials were used.

## CPU descriptive audit

The actual CSV has 8192 rows, 21 features, target usr (0–99), and SHA-256 `1025ce99015d397e0508a23595d9b3936a383179aae6fb367c42cda62c74ab86`. There are no constant or exactly duplicated feature columns, no feature pairs with absolute Pearson correlation ≥ .98, and maximum absolute correlation .939348 (fork, vflt). High skewness includes atch 21.542, lread 13.898, runqsz 11.717, and swrite 9.606. Full per-column ranges, moments, and scalar-denominator proxies are in `cpu_activity_descriptive_stats.json`.

These are descriptive scale/geometry observations. Whole-table `sum(x_j²)` under h=1 and standardized row-norm variation do not estimate actual leaf Hessian denominators, explain score aggregation causally, or establish that CPU activity should favor any method. Model preprocessing remains train-only; global standardization in the descriptive audit is not supplied to models.
