# MetaGB 消融产物文件说明

## 1. 为什么需要产物契约

消融运行会同时产生原始预测、fold 指标、聚合表和论文视图。如果只保留最终
均值，后期无法确认 backbone、split、配置和统计单位。本文档规定每个文件的
粒度与用途。

## 2. 根目录产物

### `protocol.json`

粒度：每个 run 一个。

包含：

- 协议名称与版本；
- confirmatory/smoke 标记；
- real/synthetic 数据选择；
- seeds、folds、repeats；
- XGB candidate budget；
- 路径和 CLI filters；
- canonical `config_hash`。

用途：判断两次运行是否可以合并或 resume。

禁止：不同 `config_hash` 的运行直接拼接。

### `variant_manifest.csv`

粒度：每个冻结 variant 一行，正式协议默认 16 行。

关键字段：

```text
variant_id
family
leaf_mode
pred_fraction
score_agg
score_agg_label
leaf_only_use_bias
in_main_table
in_pred_sensitivity
in_scoreagg_ablation
```

用途：代码、表格与论文命名的唯一映射。

### `task_plan.csv`

粒度：每个 outer evaluation task 一行。

任务单位：

```text
real: dataset × cv_repeat × outer_fold
synthetic: scenario × dimension × data_seed × outer_fold
```

用途：估算训练量、核对缺失任务、生成服务器队列。

### `split_manifest.json`

粒度：每个 task 的 outer/inner split identity。

包含 train/test index hash 与 inner split hash，不保存敏感原始数据。

用途：证明所有变体共享 split，并审计 outer-test 未进入 inner-CV。

注意：split hash 只证明索引隔离；真实数据编码器还必须分别在每个 inner-train
和 outer-train 上拟合。该行为由 `real_data.py` 与对应单元测试约束。

### `backbone_manifest.csv`

粒度：每个 task 一个 selected XGB backbone。

关键字段：

```text
task_id
selected_candidate_id
validation_primary_metric
outer_train_hash
outer_test_hash
inner_manifest_hash
search_space_hash
xgboost_source_hash
params_json
```

用途：审计公共参数来源；未来严格复用 backbone。

禁止：仅凭 dataset 名和 fold 编号宽松复用。

### `execution_manifest.json`

粒度：每个实际执行 run 一个。

绑定：

```text
protocol config hash
variant hash
XGB candidate hash
synthetic backbone hash
指定本地 XGBoost Python package 下全部 `.py` 源码的组合 SHA256
实际加载的 XGBoost 动态库 SHA256
Python/platform 与 NumPy/pandas/scikit-learn/SciPy/PyArrow 版本
n_jobs
```

用途：防止 resume 时混入不同 synthetic backbone、不同编译二进制或不同执行
参数。`task_hash` 必须包含 `execution_hash`。

## 3. 原始结果

### `tasks/*.json`

粒度：每个 task 一个原子 shard。

包含：

- selected backbone；
- 16 个 variant 的 fold-level 指标；
- 每个 test sample 的预测；
- status/error；
- task config hash。

每个 shard 顶层以及内部 fold/prediction/backbone 行都保存：

```text
run_id
config_hash
execution_hash
variant_hash
```

用途：断点恢复和失败定位。汇总文件可由 shards 重建。

### `fold_results.csv`

粒度：`task × variant`，成功任务默认 16 行。

关键字段：

```text
domain
dataset/scenario
dimension
data_seed/cv_repeat
outer_fold
variant_id
leaf_mode
pred_fraction
score_agg
test_primary_metric
test_rmse
test_logloss
test_auc
fit_sec
predict_sec
status
params_json
```

用途：检查 fold 级性能与运行时间。

禁止：把 outer folds 直接当作 iid 重复做普通 t-test。

### `predictions.parquet`

粒度：`task × variant × test_sample`。

关键字段：

```text
sample_id
y_true
y_pred
probabilities_json
variant_id
outer_fold
cv_repeat/data_seed
```

用途：

- 重算指标；
- 构造完整 OOF prediction；
- real 数据的 paired bootstrap；
- 分类校准与错误分析。

这是后期最重要的可复算产物，不能只保留 `fold_results.csv`。

## 4. 中间聚合

### `unit_results.csv`

统计单位：

```text
real: dataset × cv_repeat × variant
synthetic: scenario × dimension × data_seed × variant
```

先跨 outer folds 聚合，避免 folds 伪重复。

这里的 RMSE/logloss 必须从该 unit 的完整 OOF predictions 重算，禁止直接对
fold-level RMSE/logloss 做算术平均。

用途：paired bootstrap 和论文级 effect calculation。

### `completeness_audit.json`

包含：

```text
expected_tasks
actual_tasks
expected_fold_rows
actual_fold_rows
successful_fold_rows
expected_predictions
actual_predictions
backbone_rows
issues
passed
```

用途：在生成论文表格前执行硬门禁。

硬门禁还逐表验证 `run_id/config_hash/execution_hash/variant_hash` 与 manifest
完全一致；任一 shard 或行混入其他执行身份都会使 `passed=false`。
同时逐 task 校验 fold、prediction 和 shard 的 `backbone_hash` 均等于
`backbone_manifest.csv` 中该 task 的参数哈希。
prediction/fold/backbone 的 task metadata（`unit_id`、`domain`、`task_type`、
`dataset`、`scenario`、`dimension`、`data_seed`、`cv_repeat`、`outer_fold`）
也必须与 `task_plan.csv` 一致。
审计失败时，原始 shard/fold/prediction 仍保留用于排错，但所有派生统计表只写
固定表头，不输出可被误用的部分结果。

## 5. 三套主要结果表

### `main_ablation.csv`

只包含：

```text
xgb
bias_p00_sa0
split_p10_sa0
full_p10_sa0
```

字段包括均值、标准差、bootstrap CI、重复单位数量。用于四行主消融表。

### `main_effects.csv`

包含：

```text
split_vs_bias
leaf_vs_split
total_vs_bias
full_vs_xgb
```

报告 paired signed improvement、relative improvement 和 95% CI。

这是机制解释的主统计文件。

### `pred_sensitivity.csv`

粒度：`dataset/scenario × leaf_mode × p`。

包含 raw metric、CI、运行时间和候选覆盖率理论量。用于 SplitOnly/Full 两条
dose-response 曲线；相对 Bias 的 signed improvement 单独保存在
`pred_effects.csv`。

### `pred_effects.csv`

粒度：`dataset/scenario × p × effect_type`。

`effect_type`：

```text
split
leaf
total
```

这是三联机制 heatmap 的直接输入。

### `scoreagg_ablation.csv`

粒度：`dataset/scenario × leaf_mode × score_agg`，固定 `p=1`。

正式协议是
`leaf_mode ∈ {SplitOnly, Full} × score_agg ∈ {0,1,2}` 的完整六格设计，
用于隔离 aggregation 与 predictive leaf 的交互。

### `scoreagg_effects.csv`

包含：

```text
mean_vs_greedy
median_vs_greedy
```

报告 paired effect 和 CI。用于 forest plot 或附录小表。

## 6. 运行元数据

### `run_metadata.json`

包含：

```text
run/protocol/config/variant/execution identity
confirmatory 与 completed 状态
本次调用 wall-clock
task、variant、fold、prediction、unit 行数
environment identity
results directory
```

### `artifact_catalog.json`

机器可读的预期文件目录：文件名、用途、是否为完整运行所必需。

用途：服务器运行结束后解释“哪些文件应该存在”；实际是否完整由
`completeness_audit.json` 和 `run_metadata.json` 判断。

## 7. Synthetic 分析门控产物

### `SYNTHETIC_CONCLUSION.md`

位置：

```text
results/syn_formal_<RUN_TAG>/SYNTHETIC_CONCLUSION.md
```

从 `SYNTHETIC_CONCLUSION_TEMPLATE_ZH.md` 复制后填写。必须覆盖：

- 完整性与运行 identity；
- 四类主消融和三个配对机制效应；
- SplitOnly/Full 的 `pred_fraction` 剂量响应；
- scoreagg 六格；
- 机制匹配场景与负对照；
- 三个 seed 的方向一致性、不确定性和结论边界；
- 是否进入 Real 的明确决策。

该文件不是训练程序自动生成的“结论”；所有判断必须来自本次 Synthetic 的真实
产物。

### `synthetic_analysis_approval.json`

由 `launch_ablation.sh approve-synthetic` 原子写入，记录：

```text
synthetic run_id
config_hash
execution_hash
variant_hash
variant_count = 16
conclusion_sha256
approval timestamp
```

Real 启动前会再次核验所有字段。Synthetic 运行身份变化、使用 14 配置或批准后
修改结论都会使门控失败。

## 8. 后续绘图输入

预期 publication figure 不直接读取 shards，而读取：

```text
主表：main_ablation.csv + main_effects.csv
pred heatmap：pred_effects.csv
dose-response：pred_sensitivity.csv
scoreagg：scoreagg_ablation.csv + scoreagg_effects.csv
```

绘图脚本必须保留 metric direction、CI、重复单位数和 protocol hash。
