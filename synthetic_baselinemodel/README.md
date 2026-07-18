# Synthetic Baseline Models

本工程是 `synthetic_metagb` 的 baseline 分支。它复用完全相同的合成数据
生成器与 case 设计，并使用 `random300_baselinemodel` 已冻结的模型实现和
超参数空间。baseline 协议只保留 10/20-feature 的 38 cells，不运行 50
features。

因此部署到服务器时应复制整个仓库；本目录有意依赖同级的
`synthetic_metagb`、`random300_baselinemodel` 和
`tabular_extra_random300`，以避免复制后协议悄悄漂移。

仅运行三个 CPU 树模型 baseline：

1. `lightgbm`
2. `catboost`
3. `random_forest`

任务调度顺序固定为：

```text
LightGBM: feature 10 → feature 20
CatBoost: feature 10 → feature 20
Random Forest: feature 10 → feature 20
```

即全局按 `model → feature dimension → scenario → outer fold` 排队，便于优先
得到最快模型和低维 cells 的完整结果。

正式协议：

```text
38 cells（19 cases × 10/20 维）
data_seed = 42
5 outer folds × 4 inner folds
300 candidates / model
3 models
570 outer-fold tasks
171,000 candidate evaluations
684,570 planned model fits
```

超参配置位于 `configs/default_search_space.json`。其中三个树模型的配置与
`random300_baselinemodel/configs/default_search_space.json` 对应部分逐项相同；
LightGBM 和 CatBoost 的学习率、树数、深度、L2 公共部分保持相同。

## 快速验证

```bash
bash synthetic_baselinemodel/smoke_linux.sh
```

默认 smoke 只跑第一个 scenario、2 个候选和全部三个树模型。缺少可选依赖时，
对应 task 会被记为失败，不会伪装成成功结果。

## 16 核 Linux 正式运行

```bash
bash synthetic_baselinemodel/run_linux_full.sh
```

脚本内部已经使用 `nohup` 后台启动并写入 PID 文件，命令返回后可以直接退出
SSH。默认使用 15 个 task worker，每个模型内部单线程，留 1 核给系统和
调度进程。

监控：

```bash
tail -f synthetic_baselinemodel/logs/synthetic_baseline_random300_seed42.log
cat synthetic_baselinemodel/logs/synthetic_baseline_random300_seed42.pid
```

## 恢复

正式脚本使用稳定的默认 run id。中断后执行：

```bash
bash synthetic_baselinemodel/run_linux_full.sh --resume
```

runner 会校验协议、搜索空间、候选集合和相关实现源码哈希，并跳过
`results/<run_id>/tasks/` 下哈希一致且成功完成的 shard；失败或损坏的 shard
会在 `--resume` 时自动重试。

如果当前版本只需要 CatBoost 与 Random Forest，在 32 核服务器使用：

```bash
bash synthetic_baselinemodel/run_linux_remaining_32core.sh
```

该脚本使用独立 run-id `synthetic_catboost_randomforest_random300_seed42`，
协议中只有 CatBoost 与 Random Forest，不读取或调度 LightGBM。首次执行
创建新实验；再次执行检测到该 run-id 后自动 `--resume`。

## 结果

每个 run 目录包含：

- `protocol.json`
- `candidate_configs.json`
- `parameter_audit.json`
- `split_manifest.json`
- `tasks/*.json`
- `candidates.csv`
- `folds.csv`
- `aggregate.csv`
- `model_summary.csv`
- `metadata.json`
- `completeness_audit.json`

详细服务器运行和总耗时评估见 `LINUX_RUN_GUIDE.md`；树模型的逐阶段实测与
慢点分析见 `TREE_MODEL_RUNTIME_PROFILE_ZH.md`。
