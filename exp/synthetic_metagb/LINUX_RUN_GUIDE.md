# Synthetic MetaGB Linux 主实验运行指南

## 1. 当前主实验

当前冻结主实验见：

- `EXPERIMENT_DESIGN_ZH.md`
- `EXECUTION_PLAN_ZH.md`

主实验配置：

```text
57 cells
data_seed = 42
models = xgb_cpu, current_metagb
5 outer folds x 4 inner folds
300 candidates/model
task_workers = 15
n_jobs = 1
```

`current_metagb_pred0` 不进入主实验；它属于后续独立 ablation。

## 2. 环境验证

在仓库根目录执行：

```bash
python3 -m pip install -r requirements.txt

PYTHONPATH=beforereview/xgboost-master/python-package python3 -c \
  "import xgboost; print(xgboost.__version__, xgboost.__file__)"

PYTHONPATH=beforereview/xgboost-master/python-package python3 -m pytest \
  synthetic_metagb/test_scenarios.py \
  synthetic_metagb/test_protocol.py \
  tests/test_random_search_contract.py \
  beforereview/xgboost-master/tests/python/test_metagb.py -q
```

打印的 XGBoost 路径必须位于本仓库内。

## 3. 16 核并行策略

使用 15 个独立单线程任务：

```text
task_workers = 15
n_jobs = 1
```

第 16 核留给操作系统和主调度进程。限制底层库线程，避免每个 worker
再次创建线程：

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

## 4. 已完成 Framework Smoke

本地已完成每个基础 case 的 `dim10` 代表 smoke：

```text
protocol = synthetic_main_framework_smoke
19 cells
76 / 76 tasks
2,280 / 2,280 candidate rows
models = xgb_cpu, current_metagb
elapsed = 1,144.8 seconds
```

结果目录：

```text
synthetic_metagb/results/synthetic_main_framework_smoke_dim10_20260615/
```

三种维度 `10/50/100` 的准确性通过生成器契约测试验证；完整 57 cells
仅在 Linux Random300 主实验中运行。

## 5. 单 Seed Random300 主实验

计算量：

```text
57 cells x 1 seed x 5 outer folds x 2 models = 570 tasks
570 x 300 candidates = 171,000 candidate evaluations
171,000 x 4 inner folds + 570 final fits = 684,570 model fits
```

后台运行：

```bash
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=beforereview/xgboost-master/python-package python3 \
  synthetic_metagb/run_benchmark.py \
  --protocol synthetic_main_random300_single_seed \
  --search-seed 42 \
  --task-workers 15 \
  --n-jobs 1 \
  --run-id synthetic_main_random300_seed42 \
  > synthetic_metagb/synthetic_main_random300_seed42.log 2>&1 &
```

## 6. 监控与恢复

监控日志和已完成任务数：

```bash
tail -f synthetic_metagb/synthetic_main_random300_seed42.log

find synthetic_metagb/results/synthetic_main_random300_seed42/tasks \
  -name '*.json' | wc -l
```

完整运行应产生 570 个 task shards。

中断后使用完全相同配置恢复：

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=beforereview/xgboost-master/python-package python3 \
  synthetic_metagb/run_benchmark.py \
  --protocol synthetic_main_random300_single_seed \
  --search-seed 42 \
  --task-workers 15 \
  --n-jobs 1 \
  --run-id synthetic_main_random300_seed42 \
  --resume
```

恢复时会校验协议配置哈希，并跳过已完成 task。

## 7. 时间预估

Framework smoke 使用 15 个独立 worker，完成 76 个低维任务耗时约
19.1 分钟。

完整主实验包含：

- 570 个任务；
- 每任务候选数从 30 增加到 300；
- inner folds 从 2 增加到 4；
- 样本数从 180 增加到 3,000；
- 38 个 `dim50/dim100` cells，其中高维 MetaGB 是主要长尾。

因此不能用 smoke 时间简单线性外推。建议将服务器运行的前 30–60 个
已完成任务作为容量校准，并重点观察 `dim100 current_metagb` task。
单 seed 主实验预期以天为单位，而非小时。

## 8. 输出文件

运行目录包含：

- `protocol.json`：冻结协议与配置哈希。
- `candidate_configs.json`：两模型完整候选。
- `candidate_fairness_audit.json`：候选预算与 paired backbone 审计。
- `split_manifest.json`：共享 nested-CV split 与泄漏审计。
- `tasks/*.json`：可恢复任务分片。
- `candidates.csv`：inner-CV 候选评估。
- `folds.csv`：outer-test 结果。
- `seed_results.csv`：seed 级结果。
- `paired_statistics.csv`：MetaGB vs XGB cell 级比较。
- `case_summary.csv`：case 级维度汇总。
- `type_summary.csv`：A-E/type 级汇总。
- `selected_complexity.csv`：最佳深度、树数、预测特征比例和时间。
- `completeness_audit.json`：结果完整性。
- `metadata.json`：最终哈希、耗时和执行模式。

完成后检查：

```bash
python3 -m json.tool \
  synthetic_metagb/results/synthetic_main_random300_seed42/completeness_audit.json

python3 -m json.tool \
  synthetic_metagb/results/synthetic_main_random300_seed42/metadata.json
```

预期：

```text
task_count = 570
candidate_row_count = 171000
models = current_metagb, xgb_cpu
completed = true
```

## 9. Legacy 与 Ablation 协议

以下旧协议仍保留用于历史复现或后续 pred0 ablation，不属于当前主实验：

- `smoke`
- `random300_confirmatory`
- `random600_confirmatory`

不要使用这些协议替代 `synthetic_main_random300_single_seed`。
