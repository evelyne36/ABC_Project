# Linux Deployment Guide For Random300 Baseline Models

This guide is for quickly moving `random300_baselinemodel` to a Linux server and
running CPU-only baseline experiments fairly against the existing MGB/XGBoost
Random300 protocol.

## 0. What Must Be Present

Required repository paths:

- `random300_baselinemodel/`
- `tabular_extra_random300/data/`
- `tabular_extra_random300/utils/`

The baseline runner reuses the existing tabular data files and metric/split
helpers from `tabular_extra_random300`.

## 1. Transfer The Code

Option A: use git on the server.

```bash
git clone <your-repo-url> check_codereview0701
cd check_codereview0701
```

Option B: use rsync from local machine.

```bash
rsync -av --progress \
  /Users/ancd/VSCodeProject/check_codereview0701/ \
  <user>@<server>:/path/to/check_codereview0701/
```

If bandwidth is tight, at minimum transfer:

```text
random300_baselinemodel/
tabular_extra_random300/data/
tabular_extra_random300/utils/
tabular_extra_random300/__init__.py
tests/test_random300_baselinemodel_framework.py
```

## 2. Create A Clean CPU Environment

Recommended Python: 3.10.

```bash
cd /path/to/check_codereview0701

conda create -n random300_cpu python=3.10 -y
conda activate random300_cpu
python -m pip install --upgrade pip setuptools wheel
```

Install the CPU-only full suite:

```bash
pip install -r random300_baselinemodel/requirements-linux-cpu.txt
```

If the server has SSL/index issues, retry with trusted hosts:

```bash
pip install \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  --trusted-host download.pytorch.org \
  -r random300_baselinemodel/requirements-linux-cpu.txt
```

## 3. Verify CPU-Only Fairness

Run:

```bash
python - <<'PY'
import torch
print({"torch": torch.__version__, "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
PY
```

Expected:

```text
cuda_version: None
cuda_available: False
```

For extra safety on shared GPU servers, export:

```bash
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="$PWD/random300_baselinemodel/tmp_mplconfig"
mkdir -p "$MPLCONFIGDIR"
```

The runner also passes `device="cpu"` for pytabkit RealMLP/TabM.

## 4. Verify Package Health

```bash
python -m pip check
python -m pytest tests/test_random300_baselinemodel_framework.py
python -m compileall -q random300_baselinemodel tests/test_random300_baselinemodel_framework.py
```

Expected:

- no broken requirements;
- framework tests pass;
- compileall exits 0.

## 5. Smoke Test Before Long Runs

Tree/linear smoke:

```bash
python random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode quick \
  --task-types regression \
  --models linear extra_trees random_forest lightgbm catboost \
  --categorical-handlings drop \
  --regression-datasets regression__Moneyball.csv \
  --max-candidates 1 \
  --outer-folds 2 \
  --inner-folds 2 \
  --outer-fold-index 0 \
  --task-workers 1 \
  --n-jobs 1 \
  --run-id smoke_linux_tree_cpu
```

Neural smoke:

```bash
python random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode quick \
  --task-types binary \
  --models torchmlp realmlp tabm \
  --categorical-handlings drop \
  --binary-datasets binary__haberman.csv \
  --max-candidates 1 \
  --outer-folds 2 \
  --inner-folds 2 \
  --outer-fold-index 0 \
  --task-workers 1 \
  --n-jobs 1 \
  --run-id smoke_linux_neural_cpu
```

Check:

```bash
cat random300_baselinemodel/results/smoke_linux_tree_cpu/run_metadata.json
cat random300_baselinemodel/results/smoke_linux_neural_cpu/run_metadata.json
```

Expected `failed_summary_rows` is 0.

## 6. Recommended Full Run Strategy

Full mode uses:

- 300 candidates per model;
- 5 outer folds;
- 4 inner folds;
- `drop` and `onehot`;
- regression, binary, and multiclass datasets.

Because `300 candidates * 5 outer folds * 4 inner folds` is expensive, run by
model group. This gives useful partial results quickly and avoids one heavy
model blocking everything.

### 6.1 Fast Baselines

```bash
RUN_ID=baseline_fast_random300_cpu_$(date +%Y%m%d_%H%M%S)

python random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode full \
  --models linear extra_trees random_forest \
  --categorical-handlings drop onehot \
  --max-candidates 300 \
  --task-workers 8 \
  --n-jobs 1 \
  --run-id "$RUN_ID"
```

### 6.2 Boosting Baselines

```bash
RUN_ID=baseline_boosting_random300_cpu_$(date +%Y%m%d_%H%M%S)

python random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode full \
  --models lightgbm catboost \
  --categorical-handlings drop onehot \
  --max-candidates 300 \
  --task-workers 8 \
  --n-jobs 1 \
  --run-id "$RUN_ID"
```

### 6.3 Neural Baselines

Start more conservatively because these are heavier:

```bash
RUN_ID=baseline_neural_random300_cpu_$(date +%Y%m%d_%H%M%S)

python random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode full \
  --models torchmlp realmlp tabm \
  --categorical-handlings drop onehot \
  --max-candidates 300 \
  --task-workers 2 \
  --n-jobs 1 \
  --run-id "$RUN_ID"
```

Increase `--task-workers` only after confirming CPU and RAM pressure.

## 7. Running Under tmux/nohup

Recommended tmux:

```bash
tmux new -s random300_baselines
conda activate random300_cpu
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="$PWD/random300_baselinemodel/tmp_mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Run one of the commands from section 6.
```

Detach with `Ctrl-b d`.

Nohup alternative:

```bash
mkdir -p random300_baselinemodel/nohup_logs
nohup bash random300_baselinemodel/run_linux_full.sh \
  > random300_baselinemodel/nohup_logs/full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

The all-in-one script runs every model and may take much longer than the grouped
strategy.

## 8. Monitoring

Task status:

```bash
cat random300_baselinemodel/logs/<run_id>/task_status.json
```

Live logs:

```bash
tail -f random300_baselinemodel/logs/<run_id>/tasks/*.log
```

Summaries:

```bash
head random300_baselinemodel/results/<run_id>/summary.csv
head random300_baselinemodel/results/<run_id>/aggregate.csv
```

Failed rows:

```bash
python - <<'PY'
import pandas as pd
run_id = "<run_id>"
df = pd.read_csv(f"random300_baselinemodel/results/{run_id}/summary.csv")
print(df[df["status"] != "ok"][["model", "categorical_handling", "task_type", "dataset", "outer_fold", "status"]])
PY
```

## 9. Result Files To Bring Back

For each run:

```text
random300_baselinemodel/results/<run_id>/
random300_baselinemodel/logs/<run_id>/task_status.json
```

Usually enough:

```text
summary.csv
aggregate.csv
candidates.csv
candidate_configs.json
parameter_audit.json
run_metadata.json
```

Rsync back:

```bash
rsync -av --progress \
  <user>@<server>:/path/to/check_codereview0701/random300_baselinemodel/results/<run_id>/ \
  /Users/ancd/VSCodeProject/check_codereview0701/random300_baselinemodel/results/<run_id>/
```

## 10. Common Issues

### PyTorch accidentally installs GPU build

Check:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

If CUDA appears, recreate the environment and reinstall from
`requirements-linux-cpu.txt`.

### Matplotlib cache warnings

Set:

```bash
export MPLCONFIGDIR="$PWD/random300_baselinemodel/tmp_mplconfig"
mkdir -p "$MPLCONFIGDIR"
```

### Too much CPU pressure

Lower:

```bash
--task-workers 1
--n-jobs 1
```

Keep `--n-jobs 1` when using multiple task workers to avoid oversubscription.

### pytabkit API mismatch

Run:

```bash
python - <<'PY'
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Regressor, TabM_D_Regressor
print(RealMLP_TD_Regressor)
print(TabM_D_Regressor)
PY
```

The current implementation was smoke-tested with pytabkit 1.7.3.

