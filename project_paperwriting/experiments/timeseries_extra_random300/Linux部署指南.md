# timeseries_extra_random300 Linux 部署指南

这份指南用于在一台新的 Linux 服务器上部署 TS 树模型实验：从编译本地 MetaGB/XGBoost fork，到 Python requirements 配置，再到分层全量实验启动与结果检查。

核心原则：**不要安装 PyPI 版 `xgboost` 作为主实验模型**。本实验里的 `current_metagb` 是修改过的本地 XGBoost fork，必须从：

```text
beforereview/xgboost-master/
```

编译并通过：

```text
PYTHONPATH=beforereview/xgboost-master/python-package
```

导入。

## 1. 服务器系统依赖

推荐 Ubuntu 22.04/24.04 或同类 Linux。CPU 版实验不需要 CUDA。

```bash
sudo apt-get update
sudo apt-get install -y \
  git \
  build-essential \
  cmake \
  ninja-build \
  python3 \
  python3-venv \
  python3-dev \
  pkg-config
```

检查版本：

```bash
gcc --version
cmake --version
ninja --version
python3 --version
```

建议：

- `cmake >= 3.18`
- GCC/Clang 支持 C++17
- Python 3.10-3.12 均可用于 `PYTHONPATH` 源码导入方式

如果你打算 `pip install` 本地 `python-package/`，该 fork 的 `pyproject.toml` 声明 `requires-python >=3.12`；但本实验脚本默认不需要安装本地 xgboost wheel，只需要编译共享库并设置 `PYTHONPATH`。

## 2. 准备代码与数据

进入工程根目录：

```bash
cd /path/to/check_codereview0701
```

确认本地 fork 存在：

```bash
test -d beforereview/xgboost-master
test -d beforereview/xgboost-master/python-package
```

如果服务器是重新 clone 的仓库，并且 `dmlc-core` 等子目录缺文件，执行：

```bash
cd beforereview/xgboost-master
git submodule update --init --recursive
cd ../..
```

确认 TS raw 数据已经在：

```bash
ls timeseries_extra_random300/data/raw
```

当前实验需要这些文件：

```text
ETTh1.csv
ETTh2.csv
ETTm1.csv
ETTm2.csv
exchange_rate.txt
national_illness.csv
weather.csv
PEMS03.npz
PEMS04.npz
PEMS07.npz
PEMS08.npz
```

缺数据时可先运行下载脚本：

```bash
python3 timeseries_extra_random300/scripts/download_datasets.py
```

## 3. 创建 Python 环境

推荐用项目内 venv：

```bash
python3 -m venv .venv-ts
source .venv-ts/bin/activate
python -m pip install -U pip setuptools wheel
```

安装 TS 实验运行依赖：

```bash
pip install -r timeseries_extra_random300/requirements-linux.txt
```

再次强调，不要执行：

```bash
pip install xgboost
```

如果环境里已经装过 PyPI `xgboost`，实验脚本仍会优先通过 `PYTHONPATH` 导入本地 fork；但为了减少歧义，建议卸载：

```bash
pip uninstall -y xgboost
```

## 4. 编译本地 MetaGB/XGBoost 共享库

CPU 版构建：

```bash
cd beforereview/xgboost-master
rm -rf build_linux
cmake -B build_linux -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF \
  -GNinja
cmake --build build_linux --parallel "$(nproc)"
cd ../..
```

构建完成后应看到 Linux 共享库：

```bash
ls -lh beforereview/xgboost-master/lib/libxgboost.so*
```

如果没有 `libxgboost.so`，说明 C++ 编译没有成功，先不要启动实验。

## 5. 配置本地 XGBoost 导入路径

每次运行实验前，在工程根目录设置：

```bash
export XGBOOST_PYTHON_PACKAGE="$PWD/beforereview/xgboost-master/python-package"
export PYTHONPATH="$XGBOOST_PYTHON_PACKAGE:${PYTHONPATH:-}"
```

检查 Python 实际导入路径：

```bash
python - <<'PY'
import xgboost as xgb
print("xgboost file:", xgb.__file__)
print("xgboost version:", getattr(xgb, "__version__", "unknown"))
PY
```

输出路径应该包含：

```text
beforereview/xgboost-master/python-package/xgboost
```

再检查 MetaGB 参数是否被本地 fork 接受：

```bash
python - <<'PY'
import numpy as np
from xgboost import XGBRegressor

X = np.random.default_rng(0).normal(size=(64, 8)).astype("float32")
y = X[:, 0] * 0.5 + X[:, 1] * -0.2
model = XGBRegressor(
    n_estimators=2,
    max_depth=2,
    tree_method="hist",
    device="cpu",
    enable_metagb=True,
    metagb_pred_colsample_bytree=0.5,
    metagb_score_agg=0,
    metagb_lambda_l2=1.0,
    verbosity=0,
)
model.fit(X, y)
print("metagb smoke ok", model.predict(X[:3]).shape)
PY
```

如果这里报 `Unknown parameter: enable_metagb` 或导入路径指向 site-packages 里的 PyPI `xgboost`，说明 `PYTHONPATH` 没配对。

## 6. 框架 smoke test

先跑最小 TS smoke，确认四个模型依赖都能导入：

```bash
MPLCONFIGDIR=/tmp/mpl_ts_smoke \
python timeseries_extra_random300/scripts/run_tiered_timeseries_benchmark.py \
  --dry-run \
  --tiers non_pems \
  --base-run-id dryrun_ts_tree_server
```

再跑一个真正的小训练 smoke：

```bash
MPLCONFIGDIR=/tmp/mpl_ts_smoke \
python timeseries_extra_random300/scripts/run_tiered_timeseries_benchmark.py \
  --tiers non_pems \
  --base-run-id smoke_ts_tree_server \
  --mode smoke \
  --models current_metagb xgb_cpu lgbm_cpu catboost_cpu \
  --seeds 42 \
  --seq-lens 24 \
  --pred-lens 1 \
  --max-raw-rows 180 \
  --max-features 32 \
  --max-candidates 30 \
  --search-space-json timeseries_extra_random300/configs/smoke_search_space.json \
  --task-workers 1 \
  --n-jobs 1 \
  --progress-every 30
```

检查结果：

```bash
ls timeseries_extra_random300/results/smoke_ts_tree_server
ls timeseries_extra_random300/results/smoke_ts_tree_server__seed42__non_pems
```

子目录里应包含：

```text
summary.csv
candidates.csv
aggregate.csv
metadata.json
split_manifest.json
parameter_audit.json
```

## 7. 全量实验启动

当前服务器正式默认参数：

```text
models = current_metagb xgb_cpu lgbm_cpu catboost_cpu
split_protocol = fixed
feature_mode = window_summary
target_mode = sequence
output_scope = target
seed = 42
seq_lens = 24 48
pred_lens = 1 6 12
max_candidates = 300
max_features = 64
```

默认分层顺序：

```text
non_pems:
  ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, ILI, Weather

pems:
  PEMS03, PEMS04, PEMS07, PEMS08
```

启动完整分层实验：

```bash
source .venv-ts/bin/activate
export XGBOOST_PYTHON_PACKAGE="$PWD/beforereview/xgboost-master/python-package"
export PYTHONPATH="$XGBOOST_PYTHON_PACKAGE:${PYTHONPATH:-}"

bash timeseries_extra_random300/run_linux_tiered_tree.sh
```

只跑非 PEMS 阶段：

```bash
TIERS="non_pems" \
bash timeseries_extra_random300/run_linux_tiered_tree.sh
```

指定多 seed 时，seed 是最外层循环：

```bash
SEEDS="42 43 44" \
bash timeseries_extra_random300/run_linux_tiered_tree.sh
```

调度顺序为：

```text
seed42/non_pems -> seed42/pems -> seed43/non_pems -> ...
```

脚本启动后会打印：

```text
base_run_id=...
pid=...
log=...
combined_results=...
```

## 8. 监控运行

查看日志：

```bash
tail -f timeseries_extra_random300/nohup_logs/<base_run_id>.log
```

查看进程：

```bash
ps -fp <pid>
```

查看某个 seed/tier 的任务状态：

```bash
cat timeseries_extra_random300/logs/<base_run_id>__seed42__non_pems/task_status.json
```

持续检查父目录合并结果：

```bash
ls timeseries_extra_random300/results/<base_run_id>
head -5 timeseries_extra_random300/results/<base_run_id>/summary.csv
head -20 timeseries_extra_random300/results/<base_run_id>/aggregate.csv
```

每个 seed/tier 子目录都会保留完整候选记录：

```text
timeseries_extra_random300/results/<base_run_id>__seed42__non_pems/candidates.csv
```

## 9. 结果目录结构

父目录：

```text
timeseries_extra_random300/results/<base_run_id>/
  summary.csv
  aggregate.csv
  smoke_report.md
  tier_manifest.json
```

子目录：

```text
timeseries_extra_random300/results/<base_run_id>__seed42__non_pems/
  summary.csv
  candidates.csv
  aggregate.csv
  metadata.json
  parameter_audit.json
  split_manifest.json
  candidate_configs.json
  smoke_report.md
```

父目录 `summary.csv` 会合并已经完成的 seed/tier；因此即使 PEMS 还没跑完，也可以先看 non_pems 的结果。

## 10. 常见问题

### Python 导入了错误的 xgboost

症状：

```text
Unknown parameter: enable_metagb
```

或：

```text
xgboost file: .../site-packages/xgboost/...
```

处理：

```bash
pip uninstall -y xgboost
export XGBOOST_PYTHON_PACKAGE="$PWD/beforereview/xgboost-master/python-package"
export PYTHONPATH="$XGBOOST_PYTHON_PACKAGE:${PYTHONPATH:-}"
```

### 找不到 libxgboost.so

症状：

```text
XGBoostLibraryNotFound
```

处理：

```bash
cd beforereview/xgboost-master
cmake -B build_linux -S . -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -GNinja
cmake --build build_linux --parallel "$(nproc)"
cd ../..
ls beforereview/xgboost-master/lib/libxgboost.so*
```

### LightGBM 或 CatBoost ImportError

处理：

```bash
pip install -r timeseries_extra_random300/requirements-linux.txt
python - <<'PY'
import lightgbm, catboost
print(lightgbm.__version__)
print(catboost.__version__)
PY
```

### 服务器 CPU 被过度订阅

默认：

```text
TASK_WORKERS=15
N_JOBS=1
```

含义是并行 15 个外层任务，每个模型内部单线程。若服务器核心少，可以降到：

```bash
TASK_WORKERS=8 N_JOBS=1 bash timeseries_extra_random300/run_linux_tiered_tree.sh
```

不要随意把 `TASK_WORKERS` 和 `N_JOBS` 同时调大。

### PEMS 太慢

默认 `MAX_FEATURES=64` 会把 PEMS 原始通道截到最多 64 个，保证可计算。如果还慢，可以先跑：

```bash
TIERS="non_pems" bash timeseries_extra_random300/run_linux_tiered_tree.sh
```

确认主结果后再追加：

```bash
TIERS="pems" bash timeseries_extra_random300/run_linux_tiered_tree.sh
```

## 11. 最小命令清单

从零开始的一组最短命令：

```bash
cd /path/to/check_codereview0701

python3 -m venv .venv-ts
source .venv-ts/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r timeseries_extra_random300/requirements-linux.txt
pip uninstall -y xgboost || true

cd beforereview/xgboost-master
git submodule update --init --recursive
cmake -B build_linux -S . -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -GNinja
cmake --build build_linux --parallel "$(nproc)"
cd ../..

export XGBOOST_PYTHON_PACKAGE="$PWD/beforereview/xgboost-master/python-package"
export PYTHONPATH="$XGBOOST_PYTHON_PACKAGE:${PYTHONPATH:-}"

python - <<'PY'
import xgboost as xgb
print(xgb.__file__)
PY

bash timeseries_extra_random300/run_linux_tiered_tree.sh
```
