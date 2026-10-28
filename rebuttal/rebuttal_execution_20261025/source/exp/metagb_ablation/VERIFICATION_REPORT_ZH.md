# MetaGB 消融工程验证报告

验证日期：2026-07-18

## 1. 验证范围

本报告验证工程契约与最小运行链路，不把 smoke 数值解释为论文结果。

已验证：

1. Python 语法；
2. 默认 16 个冻结变体及其复用关系；
3. 公共 backbone 与 lambda 映射；
4. 主表、pred sensitivity 和 scoreagg 统计分解；
5. plan-only；
6. synthetic development-seed backbone 路径；
7. real outer-train inner-CV backbone 路径；
8. task shard 与 `--resume`；
9. 逐样本 predictions 和 completeness audit；
10. real inner/outer fold-local categorical preprocessing；
11. fold/prediction/backbone/shard provenance identity；
12. Synthetic 结论 hash 门控 Real。

## 2. 单元测试

命令：

```bash
python3 -m pytest -q metagb_ablation/tests
```

结果：

```text
24 passed
```

覆盖：

- 默认配置恰为 16，显式降级模式恰为 14；
- 主表四项顺序固定；
- SplitOnly 与 Full 都覆盖五档 `p={0.2,0.4,0.6,0.8,1.0}`；
- scoreagg 映射为 `0/1/2 = greedy/mean/median`；
- Bias、SplitOnly、Full 的 leaf flag 正确；
- 14 配置降级模式只移除两个 SplitOnly scoreagg 配置；
- 公共 backbone 与 L2 映射一致；
- paired split/leaf/total effects 正确生成。
- 完整 OOF predictions 重算 RMSE/logloss；
- strict artifact audit；
- execution manifest 拒绝 backbone identity 改变；
- strict audit 拒绝被篡改的 execution identity；
- strict audit 拒绝 prediction 与 task plan 的 unit/domain/scenario metadata 不一致；
- unit aggregation 拒绝跨 fold 混合 identity；
- ordinal/one-hot 的未知类别不进入训练词表；
- 16 配置 scoreagg 六格汇总；
- zero-task 输出明确失败 audit，不在汇总前崩溃。
- Synthetic approval 拒绝 14 配置；
- 结论在批准后被修改会阻断 Real。

## 2.1 底层 `p=0` 编译与测试

为保证 Bias 不是概率意义上的“几乎为零”，而是数学上严格的空 predictive
feature set，`SampleFeatures()` 增加了 `fraction <= 0` 显式分支。

验证：

```text
独立 /tmp CMake configure       passed
xgboost target build            passed
targeted MetaGB Python tests     7 passed
```

新增测试会保存 `p=0` 模型并检查所有最终叶的 MetaGB feature ID 都等于 bias
feature ID。

## 3. Plan-only

命令：

```bash
PYTHONPATH=beforereview/xgboost-master/python-package \
python3 -m metagb_ablation.run_ablation \
  --protocol formal \
  --domains synthetic \
  --plan-only \
  --output-root /tmp/metagb_ablation_plan16 \
  --run-id formal_default16_check
```

验证结果：

```text
tasks = 210
variants = 16
expected_fold_rows = 3360
```

同一正式计划显式传入 `--no-split-scoreagg` 时为
`210 tasks / 14 variants / 2940 fold rows`，证明降级模式必须主动请求。

## 4. Synthetic 训练 Smoke

设置：

```text
scenario = regime_slope
dimension = 10
evaluation seed = 101
development backbone seed = 9001
outer fold = 1/2
XGB candidate budget = 1
```

验证结果：

```text
fold_results.csv        16 rows
predictions.parquet     3,200 rows
main_ablation.csv       4 rows
main_effects.csv        4 rows
pred_sensitivity.csv    10 rows
pred_effects.csv        15 rows
scoreagg_ablation.csv   6 rows
scoreagg_effects.csv    4 rows
```

`completeness_audit.json`：

```text
passed = true
successful_fold_rows = 16/16
duplicate_fold_keys = 0
actual_predictions = expected_predictions = 3,200
```

backbone 来源：

```text
synthetic_development_seed
```

新版 strict audit 还逐 task 验证：

```text
exact variant set
task × variant × sample uniqueness
y_true consistency
classification probability validity
split-manifest identity
one backbone per task
shared public params and L2 mapping
run/config/execution/variant identity
```

## 5. Real 训练 Smoke

数值数据 smoke 设置：

```text
dataset = airfoil_self_noise
preprocessing = ordinal
outer fold = 1/2
inner folds = 2
XGB candidate budget = 1
```

此前的 reduced-14 工程路径验证结果：

```text
fold_results.csv        14 rows
predictions.parquet     10,528 rows
completeness passed     true
```

backbone 来源：

```text
real_outer_train_inner_cv
```

这验证了 real 路径只在 outer-train 的 inner-CV 上选择 XGB backbone，再对同一
outer-test 评估全部冻结变体。

含类别特征的 `credit-g` fold-local ordinal路径此前已在旧24配置网格下验证：

```text
outer fold               1/2
inner folds              2
variant rows             24/24 successful
prediction rows          12,000
main/pred/scoreagg rows   4/20/3
completeness             true, issues=[]
```

outer encoder 仅在 outer-train 拟合；每个 inner encoder 仅在对应 inner-train
拟合。未知类别由 ordinal encoder 映射为 `-1`。相同 run-id 的 resume 没有
重复训练，审计仍通过。网格降采样只改变variant manifest，不改变该预处理路径；
该结果证明 Real 的 fold-local preprocessing 和 fixed-backbone 路径可运行，但不
作为当前正式 16 配置 Real 证据。按照新的阶段门控，正式 Real 必须等完整
Synthetic 运行与分析结论批准后再执行，因此本次没有提前重跑 Real。

## 6. Resume

对已完成 synthetic run 使用相同 `run_id` 和 `--resume`：

```text
没有重复训练 task
重新收集 shards
completeness 仍为 true
```

协议 hash 或 task hash 不一致时，代码会拒绝 resume。

execution hash 进一步绑定：

```text
synthetic backbone hash
XGB candidate hash
actual loaded XGBoost dynamic-library SHA256
all local XGBoost Python `.py` source hashes
Python/platform and scientific-package versions
n_jobs
direct Python/C++ source dependency hashes
```

## 7. 默认 16 配置、阶段门控与失败路径

默认训练 smoke：

```text
fold rows              16
prediction rows        3,200
scoreagg raw rows       6
scoreagg effect rows    4
completeness            true
```

六格为：

```text
SplitOnly × greedy/mean/median
Full      × greedy/mean/median
```

启动器不提供 `start all`。测试夹具验证：

```text
16-variant Synthetic + passed audit + completed conclusion
  → approve-synthetic 成功并绑定 config/execution/variant/conclusion hash
批准后修改 SYNTHETIC_CONCLUSION.md
  → start real 被 conclusion_sha256 门控拒绝
14-variant Synthetic
  → approve-synthetic 被拒绝
```

`--limit-tasks 0` 非 plan 运行会：

```text
退出码 2
completeness passed = false
issues = ["no_tasks_selected"]
所有派生 CSV 保留固定表头
```

## 8. 尚未作为论文证据验证

- formal 8 real datasets；
- formal 7 synthetic cases × 2 dimensions × 3 seeds；
- real 5×5 repeated CV；
- 逐树 realized predictive-candidate telemetry；
- publication heatmap renderer。

这些项目的状态保持 planned，不能从本 smoke 推导论文结论。
