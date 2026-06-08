# Random500 Linux Server Run Commands

本文档用于把当前 `current_metagb` Random500 实验迁移到 Linux 服务器后直接启动。以下命令默认在项目根目录执行：

```bash
cd /path/to/check_codereview0604cc
```

## 1. 编译后验证

```bash
PYTHONPATH=./beforereview/xgboost-master/python-package \
python3 ./beforereview/scripts/check_metagb_runtime.py
```

期望看到：

```text
enable_metagb: 1
status: OK
```

## 2. 日志命名规则

推荐使用 `random500/scripts/run_random500_server.py` 启动。脚本会自动生成：

```text
run_id = <experiment-name>_<task-type-tag>_<YYYYMMDD_HHMMSS>
run_log = random500/nohup_logs/<run_id>.log
results = random500/results/<run_id>/
task_logs = random500/logs/<run_id>/tasks/
task_status = random500/logs/<run_id>/task_status.json
```

其中 task-type-tag 为：

```text
all
regression
binary
multiclass
regression-binary
...
```

即使外层使用 `nohup`，也建议让脚本自己写 run-level log。外层 `nohup` 文件只负责记录 `run_id` 和 `run_log` 路径。

## 3. Smoke Test

```bash
nohup python3 random500/scripts/run_random500_server.py \
  --experiment-name smoke_random500 \
  --mode quick \
  --task-types regression binary multiclass \
  --regression-datasets airfoil_self_noise.csv \
  --binary-datasets diabetes.csv \
  --multiclass-datasets anneal.csv \
  --models xgb_cpu current_metagb \
  --max-candidates 10 \
  --outer-folds 2 --inner-folds 2 \
  --single-outer-fold --outer-fold-index 0 \
  --task-workers 1 --n-jobs 1 \
  > smoke_random500_launcher.log 2>&1 &
```

查看外层启动日志：

```bash
tail -f smoke_random500_launcher.log
```

外层日志会打印真实 run log，例如：

```text
run_log: /path/to/check_codereview0604cc/random500/nohup_logs/smoke_random500_all_YYYYMMDD_HHMMSS.log
```

之后查看真实实验日志：

```bash
tail -f random500/nohup_logs/<run_id>.log
```

## 4. 主实验：current_metagb vs xgboost

主实验只跑：

```text
xgb_cpu
current_metagb
```

`current_metagb` 会自动使用：

```text
pred_feature_fraction = 0.1, 0.2, ..., 1.0
```

`--max-candidates 500` 表示每个 pred fraction 约 50 组候选。包含 `current_metagb` 时，`--max-candidates` 必须能被 10 整除。

16 核 CPU 推荐：

```text
--task-workers 16 --n-jobs 1
```

这是完整 nested-CV 主实验的默认推荐。脚本会按 `task_type x dataset x outer_fold x model` 生成大量独立 task，任务级并行比单模型内部多线程更适合作为主实验吞吐策略，也能避免多进程和 XGBoost/OpenMP 线程互相抢核。

不建议把完整主实验默认改成 `--task-workers 4 --n-jobs 4` 或 `--task-workers 1 --n-jobs 16`。`4x4` 只适合宽表专项计时或追求单任务延迟时单独测试；`1x16` 只有在任务数量太少、无法填满 CPU 时才有意义。

简化入口：

```bash
nohup python3 random500/scripts/run_random500_full.py \
  > random500_main_launcher.log 2>&1 &
```

等价完整入口：

```bash
nohup python3 random500/scripts/run_random500_server.py \
  --experiment-name random500_main_xgb_vs_current_metagb \
  --mode full \
  --task-types regression binary multiclass \
  --models xgb_cpu current_metagb \
  --max-candidates 500 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  > random500_main_launcher.log 2>&1 &
```

## 5. 分服务器按任务类型运行

只跑 regression：

```bash
nohup python3 random500/scripts/run_random500_server.py \
  --experiment-name random500_main_xgb_vs_current_metagb \
  --mode full \
  --task-types regression \
  --models xgb_cpu current_metagb \
  --max-candidates 500 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  > random500_main_regression_launcher.log 2>&1 &
```

只跑 binary：

```bash
nohup python3 random500/scripts/run_random500_server.py \
  --experiment-name random500_main_xgb_vs_current_metagb \
  --mode full \
  --task-types binary \
  --models xgb_cpu current_metagb \
  --max-candidates 500 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  > random500_main_binary_launcher.log 2>&1 &
```

只跑 multiclass：

```bash
nohup python3 random500/scripts/run_random500_server.py \
  --experiment-name random500_main_xgb_vs_current_metagb \
  --mode full \
  --task-types multiclass \
  --models xgb_cpu current_metagb \
  --max-candidates 500 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  > random500_main_multiclass_launcher.log 2>&1 &
```

多分类 `current_metagb` 默认使用 k-class scalar tree，即 `multi_strategy=one_output_per_tree`。主实验不要传 `multi_strategy="multi_output_tree"`。

## 5.1 速度口径说明

如果要公平比较原版 standalone `metagb/` 与嵌入版 `current_metagb` 的单模型训练速度，使用：

```text
n_jobs=1
```

原版 standalone MetaGB 不使用 XGBoost 的 `n_jobs`。因此 `current_metagb` 在 `n_jobs=4/16` 下得到的加速，属于嵌入版 XGBoost/OpenMP 内部并行收益，不能直接作为和原版 MetaGB 的公平单核速度比。

主实验只跑 `xgb_cpu vs current_metagb` 时，仍推荐：

```text
--task-workers 16 --n-jobs 1
```

这样既保证每个模型训练口径一致，又能最大化完整 nested-CV 批量实验的整体吞吐。

## 6. 分服务器按数据集运行

显式指定 binary 的一部分数据集：

```bash
nohup python3 random500/scripts/run_random500_server.py \
  --experiment-name random500_main_binary_part1 \
  --mode full \
  --task-types binary \
  --binary-datasets diabetes.csv credit-g.csv heloc.csv \
  --models xgb_cpu current_metagb \
  --max-candidates 500 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  > random500_main_binary_part1_launcher.log 2>&1 &
```

从默认 `all` 中跳过指定数据集：

```bash
nohup python3 random500/scripts/run_random500_server.py \
  --experiment-name random500_main_binary_without_large \
  --mode full \
  --task-types binary \
  --binary-datasets all \
  --skip-binary-datasets APSFailure.csv Diabetes130US.csv \
  --models xgb_cpu current_metagb \
  --max-candidates 500 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  > random500_main_binary_without_large_launcher.log 2>&1 &
```

跨任务类型统一跳过：

```bash
--skip-datasets dataset_a.csv dataset_b.csv
```

## 7. 后续副实验：pred0 ablation

主实验完成后再启动副实验。推荐副实验同 run 内带上 `xgb_cpu`，方便直接比较：

```bash
nohup python3 random500/scripts/run_random500_server.py \
  --experiment-name random500_ablation_pred0_vs_xgb \
  --mode full \
  --task-types regression binary multiclass \
  --models xgb_cpu current_metagb_pred0 \
  --max-candidates 500 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  > random500_ablation_pred0_launcher.log 2>&1 &
```

如果想避免重复跑 XGBoost，也可以只跑 pred0，然后离线和主实验的 `xgb_cpu` 对比：

```bash
nohup python3 random500/scripts/run_random500_server.py \
  --experiment-name random500_ablation_pred0_only \
  --mode full \
  --task-types regression binary multiclass \
  --models current_metagb_pred0 \
  --max-candidates 500 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  > random500_ablation_pred0_only_launcher.log 2>&1 &
```

## 8. 查看进度

查看外层启动日志：

```bash
tail -f *_launcher.log
```

查看真实 run-level 日志：

```bash
tail -f random500/nohup_logs/<run_id>.log
```

查看任务状态：

```bash
cat random500/logs/<run_id>/task_status.json
```

查看单个任务日志：

```bash
ls random500/logs/<run_id>/tasks/
tail -f random500/logs/<run_id>/tasks/<task_id>.log
```

查看结果：

```bash
ls random500/results/<run_id>/
head random500/results/<run_id>/summary.csv
head random500/results/<run_id>/aggregate.csv
```

## 9. 停止任务

先找到进程：

```bash
ps -ef | grep run_random500_server.py
ps -ef | grep exp1_random500_all.py
```

再按 PID 停止：

```bash
kill <PID>
```

如确实需要强制停止：

```bash
kill -9 <PID>
```
