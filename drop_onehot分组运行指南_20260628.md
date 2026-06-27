# drop / onehot 分组运行指南

日期：2026-06-28

## 1. tabular_extra_random300

其余 tabular 数据需要分别跑 `drop` 与 `onehot` 两个版本。两者都使用 nested CV 外 5 内 4，模型为 `xgb_cpu current_metagb`，默认覆盖当前 27 个已落盘数据，并按 `rows * features * features` 压力升序调度。

drop：

```bash
TASK_WORKERS=15 N_JOBS=1 \
bash tabular_extra_random300/run_linux_comprehensive_drop.sh
```

onehot：

```bash
TASK_WORKERS=15 N_JOBS=1 \
bash tabular_extra_random300/run_linux_comprehensive_onehot.sh
```

如需先限制规模确认服务器环境：

```bash
MAX_ROWS_PER_DATASET=2000 TASK_WORKERS=15 N_JOBS=1 \
bash tabular_extra_random300/run_linux_comprehensive_drop.sh
```

```bash
MAX_ROWS_PER_DATASET=2000 TASK_WORKERS=15 N_JOBS=1 \
bash tabular_extra_random300/run_linux_comprehensive_onehot.sh
```

dry-run 检查命令展开：

```bash
DRY_RUN=1 bash tabular_extra_random300/run_linux_comprehensive_drop.sh
DRY_RUN=1 bash tabular_extra_random300/run_linux_comprehensive_onehot.sh
```

## 2. tabarena_30

`tabarena_30` 必须区分 `binary/multiclass` 和 `drop/onehot`。原因是旧 XGB backbone 来源目录不同，数据队列也不同：

- binary drop: `tabarena_30/configs/binary_drop_datasets.txt`
- binary onehot: `tabarena_30/configs/binary_onehot_datasets.txt`
- multiclass drop: `tabarena_30/configs/multiclass_drop_datasets.txt`
- multiclass onehot: `tabarena_30/configs/multiclass_onehot_datasets.txt`

四个入口分别启动：

```bash
TASK_WORKERS=15 N_JOBS=1 bash tabarena_30/run_linux_binary_drop.sh
```

```bash
TASK_WORKERS=15 N_JOBS=1 bash tabarena_30/run_linux_binary_onehot.sh
```

```bash
TASK_WORKERS=15 N_JOBS=1 bash tabarena_30/run_linux_multiclass_drop.sh
```

```bash
TASK_WORKERS=15 N_JOBS=1 bash tabarena_30/run_linux_multiclass_onehot.sh
```

dry-run 检查命令展开：

```bash
DRY_RUN=1 bash tabarena_30/run_linux_binary_drop.sh
DRY_RUN=1 bash tabarena_30/run_linux_binary_onehot.sh
DRY_RUN=1 bash tabarena_30/run_linux_multiclass_drop.sh
DRY_RUN=1 bash tabarena_30/run_linux_multiclass_onehot.sh
```

## 3. 已确认的区分证据

runner 的任务键为 `(task_type, handling, dataset, outer_fold)`，旧 backbone 加载时也从路径推断 `drop/onehot`，因此不会把 drop 与 onehot 的 XGB 参数混用。

程序级 dry-run 抽查结果：

- binary drop 使用 `datacollection0624beifen/binary_drop_0624/...`
- binary onehot 使用 `datacollection0624beifen/binary_onehot_0624/...`
- multiclass drop 使用 `datacollection0624beifen/multiclass_drop_0624/...`
- multiclass onehot 使用 `datacollection0624beifen/multiclass_onehot_0624/...`

## 4. 需要注意

- `in_vehicle_coupon_recommendation.csv` 已排除：严格缺失行删除后只剩 108 行。
- `kddcup09_appetency.csv` 没有放入实际启动列表：当前严格预处理后可用行数为 0。
- TabArena regression onehot 只剩 `Food_Delivery_Time.csv`，计算太慢，按当前要求放弃，不提供启动入口。
- 用户标记慢任务已放在各自队尾，例如 binary drop 的 `NATICUSdroid / taiwanese / Bioresponse`，multiclass drop 的 `hiva_agnostic`，multiclass onehot 的 `hiva_agnostic`。
