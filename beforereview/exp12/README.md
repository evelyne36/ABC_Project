# exp12: Integrated MetaGB vs XGBoost Benchmarks

This folder contains rewritten experiment scripts that use the MetaGB implementation embedded in `xgboost-master`.

The old standalone C/CFFI `metagb_utils.MetaGB` interface is not used.

## Layout

```text
exp12/
  scripts/
    exp1_ordinal.py
    exp1_drop.py
    exp2_ordinal.py
    exp2_onehot.py
    exp2_drop.py
  utils/
    benchmark_core.py
    hyperparams.py
  logs/
  results/
```

## Environment And Build

Use Python 3.10+ with `numpy`, `pandas`, `scipy`, and `scikit-learn`. The scripts do not depend on a specific conda environment name; they use whichever Python interpreter launches them.

`exp12` always imports XGBoost from the modified local checkout by default:

```text
./xgboost-master/python-package
```

This means `current_metagb` uses the MetaGB code embedded in `xgboost-master`, not a pip-installed upstream XGBoost wheel. You can override the path with `METAGB_XGBOOST_PYTHON_PACKAGE` or `--xgboost-python-package`.

Check the interpreter:

```bash
python --version
which python
```

Install Python runtime dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install numpy pandas scipy scikit-learn
```

Or install the project requirements from the repository root:

```bash
pip install -r requirements.txt
```

`scikit-learn` is recommended. The scripts include a small KFold fallback, but XGBoost itself still imports `scipy`.

### CMake Tools

Recommended with conda:

```bash
conda install -c conda-forge cmake ninja compilers -y
```

You can also use system packages if they provide a recent enough CMake and compiler.

### Build XGBoost On Linux

Yes, Linux servers must rebuild XGBoost locally. The macOS `.dylib` and Linux `.so` binaries are not interchangeable, and C++ extension binaries depend on the target OS, compiler, Python ABI, OpenMP runtime, and CPU architecture.

Build the modified XGBoost checkout on the target Linux machine from `beforereview/`:

```bash
cd xgboost-master
rm -rf build
mkdir build
cd build
cmake .. -DUSE_OPENMP=ON -GNinja
cmake --build . -j"$(nproc)"
```

If `pip install -e .` is not needed or you prefer not to install the package, the experiment scripts can still load the local package from:

```text
./xgboost-master/python-package
```

Recommended Linux verification after building:

```bash
cd ./beforereview
PYTHONPATH=xgboost-master/python-package python scripts/check_metagb_runtime.py
```

Expected output should point to `xgboost-master/python-package/xgboost/__init__.py`, with `enable_metagb: 1` and `status: OK`.

### Build XGBoost On macOS

From the repository root:

```bash
cd xgboost-master
rm -rf build
mkdir build
cd build
cmake .. -DUSE_OPENMP=ON -GNinja
cmake --build . -j$(sysctl -n hw.ncpu)
cd ../python-package
pip install -e .
```

### Build Notes

If `xgboost-master/build/CMakeCache.txt` was generated on another OS or another Python/compiler environment, delete the build directory before configuring again:

```bash
rm -rf xgboost-master/build
```

For Linux 16-core servers, use:

```bash
cmake --build xgboost-master/build -j16
```

The experiments require CPU hist mode. The scripts set:

```text
tree_method="hist"
device="cpu"
```

## Paths

The scripts avoid hard-coded absolute paths. Defaults are resolved from the project root:

```text
dataset dir            ./dataset/tabArena_regression
xgboost python package ./xgboost-master/python-package
output root            ./exp12
```

Optional overrides:

```text
METAGB_PROJECT_ROOT
METAGB_XGBOOST_PYTHON_PACKAGE
METAGB_DATASET_DIR
METAGB_OUTPUT_ROOT
```

## Outputs

Each run creates timestamped folders:

```text
exp12/logs/<script_name>_YYYYMMDD_HHMMSS/
exp12/results/<script_name>_YYYYMMDD_HHMMSS/
```

Each result folder contains:

```text
*_candidates.csv
*_summary.csv
*_aggregate.csv
*_metadata.json
```

## Smoke Runs

```bash
python exp12/scripts/exp1_ordinal.py --mode quick --datasets airfoil_self_noise.csv --max-candidates 2 --outer-folds 2 --inner-folds 2
python exp12/scripts/exp1_drop.py --mode quick --datasets airfoil_self_noise.csv --max-candidates 2 --outer-folds 2 --inner-folds 2
python exp12/scripts/exp2_ordinal.py --mode quick --datasets airfoil_self_noise.csv --max-candidates 2 --outer-folds 2 --inner-folds 2
python exp12/scripts/exp2_onehot.py --mode quick --datasets airfoil_self_noise.csv --max-candidates 2 --outer-folds 2 --inner-folds 2
python exp12/scripts/exp2_drop.py --mode quick --datasets airfoil_self_noise.csv --max-candidates 2 --outer-folds 2 --inner-folds 2
```

The sibling histogram subtraction optimization is part of the current embedded updater implementation. It is not exposed as a runtime command-line switch in this version.

## One-Command Full Run

Preferred cross-platform launcher for macOS and Linux:

```bash
python exp12/run_all_full.py start
```

It runs the scripts listed in `SCRIPT_NAMES` inside `exp12/run_all_full.py` serially. The current checked-in list is:

```text
1. exp1_ordinal
2. exp1_drop
3. exp2_ordinal
4. exp2_onehot
5. exp2_drop
```

Each script receives `--task-workers 16 --n-jobs 1` by default, so every script can fill a 16-core CPU with 16 independent one-core tasks while it is running. This avoids five full CV jobs fighting each other for cores.

Current CPU strategy:

```text
The five scripts run serially.
Inside each script, tasks run asynchronously at dataset * model * outer_fold granularity.
For 11 datasets, 2 models, 5 outer folds, this creates 110 tasks.
Each task uses model-level --n-jobs 1 by default.
The script-level task pool uses --task-workers 16 by default.
```

This avoids CPU oversubscription from nested parallelism: parallelism comes from independent tasks, not from each XGBoost model spawning many threads.

Useful commands:

```bash
python exp12/run_all_full.py status
python exp12/run_all_full.py stop
python exp12/run_all_full.py start --wait
python exp12/run_all_full.py start --task-workers 16 --n-jobs 1
python exp12/run_all_full.py start --continue-on-failure
```

Linux shell wrapper:

```bash
bash exp12/run_all_full_linux.sh start
bash exp12/run_all_full_linux.sh status
bash exp12/run_all_full_linux.sh stop
```

Each full-run launcher writes orchestration logs to:

```text
exp12/logs/run_all_full_YYYYMMDD_HHMMSS/
```

For a full run, each child experiment uses a deterministic run id tied to the supervisor run:

```text
exp12/logs/run_all_full_YYYYMMDD_HHMMSS/
  supervisor.stdout.log
  jobs.json
  run_config.json
  exp1_ordinal.stdout.log
  exp1_drop.stdout.log
  exp2_ordinal.stdout.log
  exp2_onehot.stdout.log
  exp2_drop.stdout.log

exp12/logs/run_all_full_YYYYMMDD_HHMMSS__exp1_ordinal/
  run_all_full_YYYYMMDD_HHMMSS__exp1_ordinal.log
  task_status.json
  tasks/
    <dataset>__<model>__out<fold>.log

exp12/results/run_all_full_YYYYMMDD_HHMMSS__exp1_ordinal/
  *_candidates.csv
  *_summary.csv
  *_aggregate.csv
  *_metadata.json
```

The same child log/result pattern is used for `exp1_drop`, `exp2_ordinal`, `exp2_onehot`, and `exp2_drop`.

Useful live-monitoring commands:

```bash
python exp12/run_all_full.py status
tail -f exp12/logs/<run_all_id>/supervisor.stdout.log
tail -f exp12/logs/<run_all_id>/<script_name>.stdout.log
tail -f exp12/logs/<run_all_id>__<script_name>/tasks/<dataset>__<model>__out<fold>.log
```

`task_status.json` is rewritten during execution and records every `dataset * model * outer_fold` subtask, its status, selected candidate, RMSE, elapsed time, and the exact task log path. The individual task logs record candidate-level progress, including candidate counts and periodic validation RMSE lines.

## Search Semantics

`exp1_*` and `exp2_*` intentionally use different hyperparameter search modes:

```text
exp1_*: random search, 200 candidates per model by default.
exp2_*: grid search from utils/hyperparams.py, not capped by --max-candidates by default.
```

`--max-candidates` is an Exp1 random-search setting. For Exp2 it is ignored unless you explicitly add:

```bash
--limit-grid-candidates
```

That flag is intended only for quick debugging.

## MetaGB Parameter Mapping

Old standalone MetaGB:

```text
pred_feature_fraction
```

New embedded XGBoost MetaGB:

```text
metagb_pred_colsample_bytree
```

Under the default experiment settings, this controls the per-tree sampling probability for predictive features. Bias is still appended internally by the embedded MetaGB updater.
