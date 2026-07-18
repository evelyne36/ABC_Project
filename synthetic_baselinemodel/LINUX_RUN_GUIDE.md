# Synthetic Baseline：16 核 Linux 运行指南

## 1. 环境

```bash
python3 -m venv .venv-synthetic-baseline
source .venv-synthetic-baseline/bin/activate
python -m pip install --upgrade pip
python -m pip install -r synthetic_baselinemodel/requirements-linux-cpu.txt
```

建议先验证：

```bash
python -m pytest \
  tests/test_synthetic_baselinemodel_framework.py \
  synthetic_metagb/test_scenarios.py -q

bash synthetic_baselinemodel/smoke_linux.sh
```

## 2. 正式运行

```bash
bash synthetic_baselinemodel/run_linux_full.sh
```

该脚本内部执行 `nohup ... &`，自动将输出追加到
`synthetic_baselinemodel/logs/synthetic_baseline_random300_seed42.log`，
并把后台进程号写入同名 `.pid` 文件。启动命令返回后可以安全退出 SSH。

16 核机器的默认并行配置：

```text
TASK_WORKERS=15
N_JOBS=1
OMP/BLAS/MKL/NUMEXPR threads=1
```

避免将 `N_JOBS` 调高；否则 15 个外层进程会与模型内部线程相乘，导致严重
超卖和内存压力。

队列固定为 model-first，并在每个模型内先 10 features、后 20 features：

```text
LightGBM(10 → 20) → CatBoost(10 → 20) → Random Forest(10 → 20)
```

### 32 核 CatBoost + Random Forest 独立版本

当前版本可以把 LightGBM 当作不存在，不依赖其原结果目录或 `protocol.json`：

```bash
bash synthetic_baselinemodel/run_linux_remaining_32core.sh
```

脚本创建独立的两模型协议，并以 30 个单线程 worker 运行：

```text
models = catboost, random_forest
run_id = synthetic_catboost_randomforest_random300_seed42
tasks = 380
candidate evaluations = 114,000
planned fits = 456,380
```

LightGBM 不会被读取或调度。首次执行创建新 run；再次执行会自动检测
`protocol.json` 并使用 `--resume`。日志位置：

```text
synthetic_baselinemodel/logs/
  synthetic_catboost_randomforest_random300_seed42_32core.log
```

剩余两个模型的历史投影约为：

```text
CatBoost       150.9 worker-hours
Random Forest  170.0 worker-hours
合计            320.9 worker-hours
```

30 workers 的理想墙钟约 10.7 小时；考虑深树长尾、并行效率和内存带宽，
预计 **13–18 小时**。如果服务器同时有其他任务或出现 swap，改用：

```bash
TASK_WORKERS=24 bash synthetic_baselinemodel/run_linux_remaining_32core.sh
```

24 workers 下建议预留约 **15–22 小时**。

合理且不改变实验协议的提速方式：

1. 保持 `N_JOBS=1`，优先并行 outer tasks，避免 30 个进程再各自开多线程；
2. 使用本地 SSD 保存 task shards，避免网络盘高频小文件延迟；
3. 监控 `free -h`、`vmstat` 和 `htop`；一旦发生 swap，将 workers 从 30
   降至 24；
4. 确认旧服务器进程已经停止，避免两个 runner 同时写同一 run-id；
5. 先观察前 20–30 个 CatBoost tasks 的完成速率，再校准剩余时间。

不建议只为 CatBoost/RF 减少 300 candidates、缩小深度/树数空间或使用提前
终止，因为 LightGBM 已按完整 Random300 完成；只修改剩余模型会破坏候选
预算与搜索协议的一致性。若要采用缩小搜索空间的快速版，应三个模型全部
重新运行。

## 3. 监控与恢复

```bash
tail -f synthetic_baselinemodel/logs/synthetic_baseline_random300_seed42.log

find synthetic_baselinemodel/results/synthetic_baseline_random300_seed42/tasks \
  -name '*.json' | wc -l
```

完整运行应产生 570 个 task shard。中断后恢复：

```bash
bash synthetic_baselinemodel/run_linux_full.sh --resume
```

## 4. 计算量

```text
38 scenarios × 1 seed × 5 outer folds × 3 models = 570 tasks
570 tasks × 300 candidates = 171,000 candidate evaluations
171,000 × 4 inner folds + 570 final fits = 684,570 planned fits
```

各模型分别承担：

```text
190 tasks
57,000 candidate evaluations
228,190 fits
```

## 5. 时间预估与校准

在 16 核 CPU Linux 服务器、15 个单线程 worker 下，建议为完整任务预留
约 **22–32 小时墙钟时间**，通常可在 **1–1.5 天**内完成。主要耗时来自
随机森林和 CatBoost；LightGBM 占比较小。

仓库中已有的同配置级别 Random300 完整 task 日志
`datacollection0702/datacollection_baselinemodel_random300/tasks/` 提供了可比
实测：筛选 2,500–3,500 行、回归/二分类、300 candidates、5×4 nested CV
的任务后，中位 task 时间及对本工程 190 tasks/model 的投影如下：

| 模型 | 中位 task 时间 | 190 tasks 投影 |
| --- | ---: | ---: |
| LightGBM | 299 秒 | 15.8 worker-hours |
| CatBoost | 2,859 秒 | 150.9 worker-hours |
| Random Forest | 3,221.5 秒 | 170.0 worker-hours |
| **合计** |  | **约 336.7 worker-hours** |

理想地除以 15 workers 是约 22.4 小时。考虑 CatBoost 深树长尾、
进程尾部不均衡、I/O、内存带宽竞争与服务器差异后，采用 22–32 小时作为
更稳妥范围，因此计划排期建议按 **1–1.5 天**。

上表可复现：

```bash
python synthetic_baselinemodel/scripts/estimate_runtime_from_logs.py
```

针对正式 synthetic 10/20 维数据的单次拟合剖析表明，三个树模型合计约为
285–337 worker-hours，15 workers 下理想墙钟约 19–23 小时；详细候选级
分析见 `TREE_MODEL_RUNTIME_PROFILE_ZH.md`。

该估计不是 SLA。服务器 CPU 代际、内存带宽和同时运行的其他作业都会造成
明显波动。最可靠的方法是在目标服务器先跑一个保持
`n_samples=3000` 的容量样本：

```bash
python synthetic_baselinemodel/run_benchmark.py \
  --protocol synthetic_baseline_random300_single_seed \
  --scenario-limit 1 \
  --max-candidates 10 \
  --task-workers 15 \
  --n-jobs 1 \
  --run-id synthetic_baseline_capacity10
```

容量样本覆盖 1/38 scenarios，且是正式单个 scenario 的 1/30 候选预算。
一阶全量外推因子是 `38 × 30 = 1,140`；考虑固定启动开销、候选复杂度与
20-feature scenarios 的差异，建议用实测墙钟时间乘 **1,140–1,500** 作为区间，
不要只乘 30。
查看：

```bash
cat synthetic_baselinemodel/results/synthetic_baseline_capacity10/model_summary.csv
cat synthetic_baselinemodel/results/synthetic_baseline_capacity10/metadata.json
```

如果完整运行已经启动，也可用已完成 shard 的实际速率动态估计：

```text
预计剩余墙钟时间
≈ (570 - 已完成 tasks) × 已用墙钟时间 / 已完成 tasks
```

前 30–60 个 task 的模型构成可能不均衡，建议至少完成 100 个 task 后再用
该公式。
