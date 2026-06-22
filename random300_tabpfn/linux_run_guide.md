# random300_tabpfn Linux Run Guide

本文档用于在 Linux 服务器上正式运行 TabPFN 数据集版 Random300 实验。工程默认只跑 `dataset/TabPFN` 或 `dataset/tabpfn` 下的 18 个 CSV，模型默认为 `xgb_cpu current_metagb`。

## 1. 环境准备

在仓库根目录执行：

```bash
cd /path/to/check_codereview0618_ckpt_pfn

export XGB_PYTHON_PACKAGE="$PWD/beforereview/xgboost-master/python-package"
export PYTHONPATH="$XGB_PYTHON_PACKAGE:$PWD:${PYTHONPATH:-}"

python3 - <<'PY'
from pathlib import Path
root = Path.cwd()
for rel in ["random300_tabpfn", "beforereview/xgboost-master/python-package"]:
    path = root / rel
    print(rel, "OK" if path.exists() else "MISSING")
for rel in ["dataset/TabPFN", "dataset/tabpfn"]:
    path = root / rel
    if path.exists():
        print("tabpfn_data", path)
PY
```

Linux 文件系统通常大小写敏感。如果服务器上的数据目录是小写 `dataset/tabpfn`，脚本会在 `dataset/TabPFN` 不存在时自动使用它；也可以显式传 `--tabpfn-dir dataset/tabpfn`。

## 2. 正式 Random300 运行

默认正式命令：

```bash
nohup python3 random300_tabpfn/scripts/run_random300_full.py \
  --mode full \
  --models xgb_cpu current_metagb \
  --max-candidates 300 \
  --outer-folds 5 \
  --inner-folds 4 \
  --task-workers 16 \
  --n-jobs 1 \
  --run-id random300_tabpfn_full_$(date +%Y%m%d_%H%M%S) \
  > random300_tabpfn_full_launch.log 2>&1 &
```

如果数据目录是小写：

```bash
nohup python3 random300_tabpfn/scripts/run_random300_full.py \
  --tabpfn-dir dataset/tabpfn \
  --mode full \
  --models xgb_cpu current_metagb \
  --max-candidates 300 \
  --outer-folds 5 \
  --inner-folds 4 \
  --task-workers 16 \
  --n-jobs 1 \
  --run-id random300_tabpfn_full_$(date +%Y%m%d_%H%M%S) \
  > random300_tabpfn_full_launch.log 2>&1 &
```

默认会跑：

- regression: 6 datasets
- binary: 7 datasets
- multiclass: 5 datasets
- models: `xgb_cpu`, `current_metagb`
- outer folds: 5
- total tasks: `18 * 2 * 5 = 180`

本地 smoke 中 `multiclass__vehicle.csv/current_metagb` 的部分候选出现明显长尾；正式 run 建议保持 `--task-workers 16 --n-jobs 1`，并预留长尾任务时间。

## 3. 日志查看

启动日志：

```bash
tail -f random300_tabpfn_full_launch.log
```

run-level 日志位置会在启动日志中打印，默认形如：

```bash
random300_tabpfn/nohup_logs/<run_id>.log
```

任务状态：

```bash
RUN_ID=<your_run_id>
tail -f random300_tabpfn/nohup_logs/${RUN_ID}.log
```

更直接的状态检查：

```bash
RUN_ID=<your_run_id>
python3 - <<PY
import json
from pathlib import Path
path = Path("random300_tabpfn/logs") / "$RUN_ID" / "task_status.json"
payload = json.loads(path.read_text())
print("total", payload["total"])
print("completed", payload["completed"])
print("failed", payload["failed"])
for task, status in payload["tasks"].items():
    if str(status) != "ok" and str(status) != "pending":
        print(task, status)
PY
```

## 4. 结果检查

正式结果目录：

```bash
RUN_ID=<your_run_id>
ls random300_tabpfn/results/${RUN_ID}
```

应包含：

- `summary.csv`
- `candidates.csv`
- `aggregate.csv`
- `pred_fraction_analysis.csv`
- `candidate_configs.json`
- `parameter_audit.json`
- `run_metadata.json`

快速验收：

```bash
RUN_ID=<your_run_id>
python3 - <<PY
import json
import pandas as pd
from pathlib import Path
result_dir = Path("random300_tabpfn/results") / "$RUN_ID"
summary = pd.read_csv(result_dir / "summary.csv")
metadata = json.loads((result_dir / "run_metadata.json").read_text())
configs = json.loads((result_dir / "candidate_configs.json").read_text())
print("summary_rows", len(summary))
print("status_counts")
print(summary["status"].value_counts(dropna=False))
print("datasets", {k: len(v) for k, v in metadata["selected_datasets"].items()})
print("current_metagb_candidates", len(configs["current_metagb"]))
PY
```

预期：

- `summary_rows = 180`
- `status` 全部为 `ok`
- selected datasets 为 `regression=6, binary=7, multiclass=5`
- `current_metagb_candidates = 300`

## 5. 恢复与重跑建议

当前 launcher 不做断点续跑；如果某个正式 run 中断，建议保留原目录用于排查，然后使用新的 `--run-id` 重跑。

如需只重跑某一类任务：

```bash
python3 random300_tabpfn/scripts/run_random300_server.py \
  --mode full \
  --task-types binary \
  --models xgb_cpu current_metagb \
  --max-candidates 300 \
  --outer-folds 5 \
  --inner-folds 4 \
  --task-workers 16 \
  --n-jobs 1 \
  --run-id random300_tabpfn_binary_rerun_$(date +%Y%m%d_%H%M%S)
```

如需加入 pred0 消融：

```bash
python3 random300_tabpfn/scripts/run_random300_server.py \
  --mode full \
  --models xgb_cpu current_metagb current_metagb_pred0 \
  --max-candidates 300 \
  --outer-folds 5 \
  --inner-folds 4 \
  --task-workers 16 \
  --n-jobs 1 \
  --run-id random300_tabpfn_full_with_pred0_$(date +%Y%m%d_%H%M%S)
```
