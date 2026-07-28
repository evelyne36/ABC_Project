# MetaGB Fixed-Backbone Ablation

这是一个与 `random300` 主流程解耦的专用消融工程。它不重新为 MetaGB
选择公共超参数，而是先仅用 XGBoost 的 inner-CV 结果选择公共 backbone，
随后在完全相同的 outer train/test split 上评估每个冻结的 MetaGB 机制配置。
真实数据的 ordinal/one-hot 编码器均在每个 inner-train/outer-train 内局部拟合，
validation/test 只做 transform。

## 文档入口

- [EXPERIMENT_DESIGN_ZH.md](EXPERIMENT_DESIGN_ZH.md)：多层级消融理念、假设、
  数据集、统计口径和图表设计。
- [IMPLEMENTATION_PLAN_ZH.md](IMPLEMENTATION_PLAN_ZH.md)：代码架构、运行阶段、
  可恢复执行、验证门和产物契约。
- [ARTIFACTS_ZH.md](ARTIFACTS_ZH.md)：每个输出文件的粒度、字段、用途和禁止误用方式。
- [VERIFICATION_REPORT_ZH.md](VERIFICATION_REPORT_ZH.md)：当前代码已执行的契约测试、
  synthetic/real smoke 和 resume 证据。
- [SYNTHETIC_CONCLUSION_TEMPLATE_ZH.md](SYNTHETIC_CONCLUSION_TEMPLATE_ZH.md)：
  Synthetic 完成后必须填写的结论模板，也是 Real 阶段的门禁输入。

## 冻结实验族

一次正式运行默认包含 16 个唯一配置：

```text
1 × XGBoost
1 × MetaGB-Bias (p=0, greedy)
5 × MetaGB-SplitOnly (p=0.2,0.4,0.6,0.8,1.0, greedy)
5 × MetaGB-Full (p=0.2,0.4,0.6,0.8,1.0, greedy)
2 × MetaGB-Full (p=1, mean/median)
2 × MetaGB-SplitOnly (p=1, mean/median)
```

主消融、pred sensitivity 和 `SplitOnly/Full × greedy/mean/median`
score aggregation 共用这些结果，不重复拟合。`--no-split-scoreagg` 仅保留为
工程调试用的 14 配置降级模式，不是正式论文协议。

## 最小命令

只生成计划和协议，不训练：

```bash
python3 -m metagb_ablation.run_ablation \
  --protocol smoke \
  --domains synthetic \
  --plan-only \
  --run-id plan_smoke
```

执行一个最小 synthetic smoke：

```bash
PYTHONPATH=beforereview/xgboost-master/python-package \
python3 -m metagb_ablation.run_ablation \
  --protocol smoke \
  --domains synthetic \
  --synthetic-cases regime_slope \
  --dimensions 10 \
  --data-seeds 101 \
  --outer-fold-index 0 \
  --xgb-candidates 1 \
  --run-id smoke_regime_slope
```

恢复中断运行：

```bash
python3 -m metagb_ablation.run_ablation \
  --protocol smoke \
  --domains synthetic \
  --synthetic-cases regime_slope \
  --dimensions 10 \
  --data-seeds 101 \
  --outer-fold-index 0 \
  --xgb-candidates 1 \
  --run-id smoke_regime_slope \
  --resume
```

## 已完成训练的离线分析恢复

如果训练已完成且 `tasks/*.json` 仍在，但原 runner 在汇总/分析阶段因 Python
环境、PyArrow 或 XGBoost identity 变化而报错，不要在新环境直接执行
`--resume`。`--resume` 会重新校验训练环境和 XGBoost hash，并可能补跑缺失
task；它不是纯分析入口。

使用离线恢复命令：

```bash
python3 -m metagb_ablation.analyze_existing \
  --result-dir /data/metagb_ablation/results/syn_formal_formal_v1 \
  --output-dir /data/metagb_ablation/reanalysis/syn_formal_formal_v1
```

该命令：

- 只读取历史 manifests 和 `tasks/*.json`；
- 不读取当前 XGBoost，不重新计算当前环境 identity，不执行训练；
- 从 shards 重建 fold、prediction 和 backbone，并重新执行完整性硬审计；
- 只有审计通过才重算 OOF unit metric 和六张统计表；
- 写入独立输出目录，不修改原始 run；
- 生成 `ANALYSIS_EVIDENCE.md`，汇总主效应、剂量响应、scoreagg、
  mechanism-aligned scenarios 与 negative controls 的证据计数。

默认输出到原 run 的同级 `<run_id>_reanalysis/`。默认不写
`predictions.parquet`，因此不需要 PyArrow；若需要保存重建后的 Parquet，
显式添加 `--write-predictions-parquet`，并安装 `pyarrow` 或 `fastparquet`。
输出目录非空时命令会拒绝覆盖；确认只覆盖恢复产物时可传入 `--overwrite`。

结果默认写入：

```text
metagb_ablation/results/<run_id>/
```

## 一键启动脚本

入口：

```bash
bash metagb_ablation/scripts/launch_ablation.sh start synthetic
bash metagb_ablation/scripts/launch_ablation.sh start real
```

- `synthetic`：只运行正式 synthetic；
- `real`：只有 Synthetic 审计通过、完整结论已批准后，才按样本量从小到大
  依次运行八个真实数据集；
- 不提供 `start all`，避免 Synthetic 结束后未经分析就自动进入 Real。

默认配置：

```text
TASK_WORKERS=15
N_JOBS=1
RUN_TAG=v1
CATEGORICAL_HANDLING=ordinal
RUN_PREFLIGHT=1
```

正式启动器固定运行 16 variants，不接受 14 配置降级。

正式 synthetic 的 14 个 `scenario × dimension` backbone cells 会并行选择，
因此该阶段最多占用 14 核；随后 210 个 outer tasks 使用 `TASK_WORKERS`
个独立 worker 进程并行。Real 每个数据集的 25 个 outer tasks 使用相同并行
规则。每个 estimator 固定 `N_JOBS=1`，避免嵌套线程过度订阅。

当前正式 synthetic 使用三个 evaluation seeds：

```text
1001, 1002, 1003
```

所以任务规模为 `7 scenarios × 2 dimensions × 3 seeds × 5 folds = 210`，
对应 `210 × 16 = 3,360` 条 fold-level variant rows。

脚本在后台启动，并自动为所有阶段传递 `--resume`。首次运行会正常创建结果；
中断后用相同命令和 `RUN_TAG` 重启，只补缺失 task shards。

状态与日志：

```bash
bash metagb_ablation/scripts/launch_ablation.sh status synthetic
bash metagb_ablation/scripts/launch_ablation.sh tail synthetic
```

只跑 synthetic 并使用独立数据盘：

```bash
OUTPUT_ROOT=/data/metagb_ablation/results \
LOG_DIR=/data/metagb_ablation/logs \
RUN_TAG=formal_v1 \
bash metagb_ablation/scripts/launch_ablation.sh start synthetic
```

Synthetic 完成后，先复制并填写结论模板：

```bash
cp metagb_ablation/SYNTHETIC_CONCLUSION_TEMPLATE_ZH.md \
  /data/metagb_ablation/results/syn_formal_formal_v1/SYNTHETIC_CONCLUSION.md
```

结论必须基于完整产物，删除全部占位符，并以以下标记结束：

```text
SYNTHETIC_ANALYSIS_STATUS: COMPLETE
REAL_STAGE_DECISION: GO
```

批准结论。批准文件会绑定 Synthetic 的 config/execution/variant hash 以及结论
文件 SHA256；批准后再改结论会使门禁失效：

```bash
OUTPUT_ROOT=/data/metagb_ablation/results \
RUN_TAG=formal_v1 \
bash metagb_ablation/scripts/launch_ablation.sh approve-synthetic
```

随后才能启动 real：

```bash
OUTPUT_ROOT=/data/metagb_ablation/results \
LOG_DIR=/data/metagb_ablation/logs \
RUN_TAG=formal_v1 \
bash metagb_ablation/scripts/launch_ablation.sh start real
```

32 核服务器建议保留 1–2 核给系统：

```bash
TASK_WORKERS=30 N_JOBS=1 RUN_TAG=formal_v1 \
bash metagb_ablation/scripts/launch_ablation.sh start synthetic
```

完整帮助：

```bash
bash metagb_ablation/scripts/launch_ablation.sh help
```

## 解释边界

- 这是 fixed-backbone 机制消融，不是四个方法各自调优后的最高性能比较。
- `p={0.2,0.4,0.6,0.8,1.0}` 的test结果全部保留，不允许从outer test选择最佳
  `p`；`p=0` 由Bias变体提供唯一锚点。
- XGBoost 只作为水平参照，不属于 `p` 剂量曲线。
- smoke 结果只验证工程和趋势，不能作为论文显著性结论。
- 所有尚未运行的数值均为 `TBD`，工程不会生成虚构结果。
