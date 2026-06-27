# 新增数据综合实验 Linux 启动指南

日期：2026-06-28

## 数据与顺序

两组新增数据都已生成按计算压力升序的执行顺序：

- TS：`timeseries_extra_random300/execution_order_by_pressure.csv`
- Tabular：`tabular_extra_random300/execution_order_by_pressure.csv`

压力定义：

- Tabular：`rows * features * features`
- TS：`rows_or_timesteps * (seq_len * channels)^2`，其中清单默认 `seq_len=24`

当前检查结论：

- 原始清单顺序不是严格压力升序。
- 已补充压力排序清单。
- `tabular_extra_random300` runner 已改为按压力排序任务。
- TS 综合启动脚本显式按压力升序传入数据集。

## 已补齐的数据

TS 全部 14 个候选均已下载并可读取：

`ILI, exchange_rate, ETTh1, ETTh2, solar, ETTm1, ETTm2, Weather, PEMS08, PEMS04, PEMS03, electricity, PEMS07, traffic`

Tabular 全部 27 个候选均已落盘并可读取：

- regression: `Moneyball, energy_efficiency, boston, space_ga, friedman1, abalone, kin8nm, naval_propulsion_plant, cpu_activity`
- binary: `haberman, banknote-authentication, breast-w, ilpd, phoneme, credit-approval, mozilla4, pc1, kc1, electricity`
- multiclass: `balance-scale, vowel, vehicle, segment, mfeat-karhunen, satimage, mfeat-fourier, optdigits`

## 已完成 Smoke

TS 全量 14 个：

- run_id: `smoke_ts_all14_pressure_order_mgb_vs_xgb_20260628`
- summary rows: 28
- candidate rows: 840
- 对比表：`timeseries_extra_random300/results/smoke_ts_all14_pressure_order_mgb_vs_xgb_20260628/mgb_vs_xgb_comparison.csv`

Tabular 全量 27 个：

- run_id: `smoke_tabular_all27_mgb_vs_xgb_tiny_20260628`
- summary rows: 54
- candidate rows: 1620
- 对比表：`tabular_extra_random300/results/smoke_tabular_all27_mgb_vs_xgb_tiny_20260628/mgb_vs_xgb_comparison.csv`

这些 smoke 是轻量预览，不是 full 结论：

- TS smoke 使用 `seq_len=4, pred_len=1, max_raw_rows=240, max_features=32, max_candidates=30`
- Tabular smoke 使用 `max_rows_per_dataset=160, max_candidates=30, tiny_smoke_search_space.json`

## TS 综合实验启动

默认按压力升序运行全部 14 个 TS 数据：

```bash
bash timeseries_extra_random300/run_linux_comprehensive.sh
```

默认参数：

- `DATASETS="ILI exchange_rate ETTh1 ETTh2 solar ETTm1 ETTm2 Weather PEMS08 PEMS04 PEMS03 electricity PEMS07 traffic"`
- `SEQ_LEN=24`
- `PRED_LEN=1`
- `MAX_CANDIDATES=300`
- `TASK_WORKERS=15`
- `N_JOBS=1`
- `MAX_RAW_ROWS=0`
- `MAX_FEATURES=0`

建议先跑低压/可控版本：

```bash
DATASETS="ILI exchange_rate ETTh1 ETTh2 Weather" \
TASK_WORKERS=15 N_JOBS=1 \
bash timeseries_extra_random300/run_linux_comprehensive.sh
```

宽通道数据建议单独跑，先限制通道数：

```bash
DATASETS="PEMS08 PEMS04 PEMS03 electricity PEMS07 traffic" \
MAX_FEATURES=64 SEQ_LEN=12 PRED_LEN=1 \
TASK_WORKERS=15 N_JOBS=1 \
bash timeseries_extra_random300/run_linux_comprehensive.sh
```

## Tabular 综合实验启动

默认按压力排序运行全部 27 个 tabular 数据：

```bash
bash tabular_extra_random300/run_linux_comprehensive.sh
```

默认参数：

- `HANDLING=ordinal`
- `MAX_CANDIDATES=300`
- `TASK_WORKERS=15`
- `N_JOBS=1`
- `MAX_ROWS_PER_DATASET=0`

如需跑 drop / onehot：

```bash
TASK_WORKERS=15 N_JOBS=1 \
bash tabular_extra_random300/run_linux_comprehensive_drop.sh
```

```bash
TASK_WORKERS=15 N_JOBS=1 \
bash tabular_extra_random300/run_linux_comprehensive_onehot.sh
```

服务器首次建议先跑受限版本：

```bash
MAX_ROWS_PER_DATASET=2000 TASK_WORKERS=15 N_JOBS=1 \
bash tabular_extra_random300/run_linux_comprehensive.sh
```

## TabArena30 分组实验启动

`tabarena_30` 的 binary/multiclass 与 drop/onehot 需要分开跑，因为四个组合的旧 XGB backbone 来源和待跑数据集并不完全相同：

```bash
TASK_WORKERS=15 N_JOBS=1 bash tabarena_30/run_linux_binary_drop.sh
TASK_WORKERS=15 N_JOBS=1 bash tabarena_30/run_linux_binary_onehot.sh
TASK_WORKERS=15 N_JOBS=1 bash tabarena_30/run_linux_multiclass_drop.sh
TASK_WORKERS=15 N_JOBS=1 bash tabarena_30/run_linux_multiclass_onehot.sh
```

对应数据列表在：

- `tabarena_30/configs/binary_drop_datasets.txt`
- `tabarena_30/configs/binary_onehot_datasets.txt`
- `tabarena_30/configs/multiclass_drop_datasets.txt`
- `tabarena_30/configs/multiclass_onehot_datasets.txt`

`Food_Delivery_Time.csv` 是 regression onehot 剩余任务，但计算太慢，当前不再提供 regression onehot 启动入口。

## 监控

脚本启动后会打印：

- `run_id`
- `pid`
- `log`
- `results`

查看日志：

```bash
tail -f <log-path>
```

查看任务状态：

```bash
cat tabular_extra_random300/logs/<run_id>/task_status.json
```

TS 当前 runner 是顺序执行，`--task-workers` 仅用于接口一致；tabular runner 支持任务级并行。若 TS full 太慢，下一步再移植 tabular 的任务级并行。
