# random300_tabpfn 执行清单

创建日期：2026-06-19

## 目标

- [x] 新建 `random300_tabpfn/` 工程，结构对齐 `random300/`。
- [x] 复制当前工作区版本的 `random300` 源码与配置，不复制旧 `results/`、`logs/`、`__pycache__/`。
- [x] 将包内引用改为 `random300_tabpfn.*`。
- [x] 默认数据源只指向 `dataset/TabPFN` 或 Linux 小写目录 `dataset/tabpfn`。
- [x] 默认正式模型集合为 `xgb_cpu current_metagb`。
- [x] 保留 `current_metagb_pred0` 作为可选 CLI 模型。

## TabPFN 数据清单

- [x] regression 6 个：`regression__kin8nm.csv`, `regression__energy_efficiency.csv`, `regression__naval_propulsion_plant.csv`, `regression__cpu_activity.csv`, `regression__space_ga.csv`, `regression__Moneyball.csv`
- [x] binary 7 个：`binary__banknote-authentication.csv`, `binary__credit-approval.csv`, `binary__breast-w.csv`, `binary__ilpd.csv`, `binary__haberman.csv`, `binary__kc1.csv`, `binary__electricity.csv`
- [x] multiclass 5 个：`multiclass__vehicle.csv`, `multiclass__segment.csv`, `multiclass__satimage.csv`, `multiclass__vowel.csv`, `multiclass__optdigits.csv`
- [x] 默认清单总数为 18 个 CSV。

## Smoke Test

参数：

```bash
python3 random300_tabpfn/scripts/run_random300_server.py \
  --mode quick \
  --single-outer-fold \
  --task-types regression binary multiclass \
  --models xgb_cpu current_metagb \
  --max-candidates 30 \
  --outer-folds 2 \
  --inner-folds 2 \
  --task-workers 1 \
  --n-jobs 1 \
  --regression-datasets regression__Moneyball.csv \
  --binary-datasets binary__haberman.csv \
  --multiclass-datasets multiclass__vehicle.csv \
  --run-id smoke_random300_tabpfn_20260619
```

验收：

- [x] 6 个任务全部 `status=ok`。
- [x] 生成 `summary.csv`。
- [x] 生成 `candidates.csv`。
- [x] 生成 `aggregate.csv`。
- [x] 生成 `run_metadata.json`。
- [x] 生成 `task_status.json`。
- [x] `run_metadata.json` 中 selected datasets 只包含 smoke 数据。
- [x] `candidate_configs.json` 中 `current_metagb` 候选数为 30。

结果：

- 2026-06-19 首次 `smoke_random300_tabpfn_20260619` 完成 5/6：`regression__Moneyball.csv` 的 `xgb_cpu/current_metagb`、`binary__haberman.csv` 的 `xgb_cpu/current_metagb`、`multiclass__vehicle.csv` 的 `xgb_cpu` 均为 `ok`。
- 首次 run 的最后一项 `multiclass__vehicle.csv/current_metagb` 因本地耗时过长被中断；随后用 `progress-every=1` 单独复跑为 `smoke_random300_tabpfn_vehicle_metagb_diag_20260619`。
- 诊断 run 中 `multiclass__vehicle.csv/current_metagb` 完成 30/30 candidates，最终 `status=ok`，生成 `summary.csv`、`candidates.csv`、`aggregate.csv`、`run_metadata.json` 和 `task_status.json`。
- 组合验收结果：6/6 smoke tasks 为 `ok`，`current_metagb` smoke 候选数为 30。
- 注意：本地 `multiclass__vehicle.csv/current_metagb` 个别候选非常慢，Linux 正式 random300 建议用 `task-workers=16` 并预留长尾时间。

## Linux 正式运行文档

- [x] 新增 `random300_tabpfn/linux_run_guide.md`。
- [x] 文档包含环境变量、正式命令、日志查看、结果检查、恢复/重跑建议。
- [x] 文档说明 Linux 大小写敏感，以及 `--tabpfn-dir dataset/tabpfn` 覆盖方式。
