# 结果证据源注册表

原则：`datacollection0702` 是唯一 raw truth；`analysis_aaai_0702` 是 narrative snapshot；最终表图读取新生成的 canonical CSV。

## 1. 证据层级

| 层级 | 允许用途 | 禁止用途 |
| --- | --- | --- |
| raw logs / JSON / CSV | 解析指标、coverage、配置、时间 | 人工选择有利结果 |
| protocol / split / parameter audit | 写协议并核对公平性 | 替代实际 test metric |
| canonical CSV | 最终表图和正文数字 | 手工修改 |
| analysis Markdown | 叙事、交叉核验、发现异常 | 作为最终表格唯一数值输入 |
| historical/smoke results | 调试、背景 | 与正式协议合并 |

## 2. TabArena main

真实源：

- `datacollection0702/binary_drop_0624/**/tasks/*.log`
- `datacollection0702/binary_onehot_0624/**/tasks/*.log`
- `datacollection0702/multiclass_drop_0624/**/tasks/*.log`
- `datacollection0702/multiclass_onehot_0624/**/tasks/*.log`
- `datacollection0702/regression_drop_0621/**/tasks/*.log`
- `datacollection0702/regression_onehot_0624/**/tasks/*.log`

允许派生：paired fold、dataset-level mean、task/preprocessing W/T/L、runtime ratio、coverage。

协议标签必须保留：classification Random300，regression Random600；drop 与 one-hot 不得先合并再比较。

## 3. Extra tabular Random300

真实源：

- `datacollection0702/tabular_drop_20260628_030502/tasks/*.log`
- `datacollection0702/tabular_onehot_20260628_025442/tasks/*.log`
- 对应 `candidate_configs.json` 与 `parameter_audit.json`

允许派生：与 TabArena 分开的 dataset-level 和 group-level 表；只有在协议标签保留时才可放入同一汇总图。

## 4. TabArena30 fixed-backbone tail protocol

真实源：

- `datacollection0702/tabarena30_drop0702/tasks/*.log`
- `datacollection0702/tabarena30_onehot0702/tasks/*.log`
- `candidate_configs_by_task.json`
- `xgb_backbone_source.csv`
- `parameter_audit.json`

约束：该协议使用 XGB-best backbone，只搜索 MetaGB 的 10 个 `pred_feature_fraction` × 3 个 `score_agg`。不能与完整 Random300/600 称为相同 HPO budget。

## 5. Synthetic nested CV

真实源：

- `datacollection0702/synthetic_main1020full_random300/protocol.json`
- `tasks/*__dim10__seed42__*.json`
- `tasks/*__dim20__seed42__*.json`
- `split_manifest.json`
- `candidate_configs.json`
- `candidate_fairness_audit.json`
- `parameter_audit.json`

冻结协议：

- 19 个 base mechanisms；
- dimension 10 与 20；
- data seed 42；
- 5 outer folds；
- 4 inner folds；
- 300 candidates；
- `xgb_cpu` 与 `current_metagb`；
- search seed 42、CV seed 314159、model seed 271828。

有效 raw tasks：380 个 JSON。

注意：`protocol.json` 将完整 57-scenario 计划标记为 `confirmatory=false`。论文可以准确写 nested CV，但不要擅自称为 confirmatory study。

## 6. Strong baseline context

真实源：

- `datacollection0702/datacollection_baselinemodel_random300/tasks/*.log`
- 同目录 `candidate_configs.json`、`parameter_audit.json`

白名单成功记录：

| 模型 | 成功 folds | 处理 |
| --- | ---: | --- |
| Linear | 270 | 纳入 |
| ExtraTrees | 270 | 纳入 |
| RandomForest | 270 | 纳入 |
| LightGBM | 270 | 纳入 |
| CatBoost | 257 | 仅共同完整 cells，或补齐 13 folds |
| TorchMLP | 270 | 纳入 |
| RealMLP | 排除 | 用户决定放弃 |
| TabM | 无正式记录 | 用户决定放弃 |

禁止把 test 上每个数据集最优 baseline 的 oracle envelope 命名为独立算法；正式结果逐模型报告。

## 7. Time series：只用 `pred_len=1`

真实源：

- `datacollection0702/datacollection_ts_tree_seed42__non_pems/summary.csv`
- `candidates.csv`
- `candidate_configs.json`
- `split_manifest.json`
- `parameter_audit.json`

过滤条件：

```text
status == "ok" and pred_len == 1
```

过滤后：25 行、7 个数据集。LightGBM/XGBoost/CatBoost 各 7 行；MetaGB 仅 ETTh1、ETTh2、ILI、exchange_rate 4 行。ETTm1、ETTm2、Weather 的 MetaGB 必须显示为 NA。

协议是 seed42、fixed split、outer_folds=1、inner_folds=1、seq_len=24、window_summary、target scalar output。正文只能称为 targeted one-step evidence，不能称完整 LTSF benchmark。

## 8. 参数敏感性与 ablation

### 可用的真实敏感性数据

- `datacollection0702/regression_drop_0621/.../regression_drop_0621results/candidates.csv`：66,000 数据行。
- 同目录 `pred_fraction_analysis.csv`：1,650 数据行，即 11 datasets × 5 folds × 10 fractions × 3 score aggregations。
- Synthetic task JSON 中的完整 300-candidate inner-validation traces。

允许写：hyperparameter sensitivity、selection frequency、validation-performance/runtime tradeoff。

### 当前缺失的正式组件消融

`datacollection0702` 中 `current_metagb_pred0` 数量为 0；没有相同 seed42、5 outer、4 inner 协议下的 full-vs-component-off outer-test 结果。

因此不能把 sensitivity 命名为 component ablation。旧 `synthetic_metagb/results/`、`beforereview/exp12/` 的 pred0/score-agg smoke 与当前协议不同，不进入主证据。

## 9. 待生成的 provenance 文件

每张表/图保存一个记录：

- artifact ID；
- raw source glob；
- source SHA256；
- parser commit；
- 过滤条件；
- 聚合公式；
- 指标方向与 tie 定义；
- paired N / dataset N；
- canonical CSV；
- 生成脚本；
- LaTeX/figure 输出路径。

