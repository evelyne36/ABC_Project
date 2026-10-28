# MetaGB 消融工程实现方案

## 1. 实现原则

1. 文档先于代码：协议变化先修改设计文档与版本。
2. 新工程独立：不改变 Random300 的“搜索并选 winner”语义。
3. 有界复用：指标、estimator 和本地 XGBoost 导入复用 `random300`；真实数据
   preprocessing 在本工程内实现 fold-local fit/transform，避免词表泄漏。
4. 固定配置全评估：MetaGB 不做 inner-CV winner selection。
5. 每层可审计：协议、split、backbone、variant、预测、统计都有独立产物。
6. 可恢复：一个 task 对应一个原子 shard；完成的 shard 不重复训练。
7. 失败显式：失败配置保留状态和错误，不静默丢行。

## 2. 目录结构

```text
metagb_ablation/
├── README.md
├── EXPERIMENT_DESIGN_ZH.md
├── IMPLEMENTATION_PLAN_ZH.md
├── ARTIFACTS_ZH.md
├── SYNTHETIC_CONCLUSION_TEMPLATE_ZH.md
├── __init__.py
├── protocols.py
├── io_utils.py
├── real_data.py
├── modeling.py
├── statistics.py
├── run_ablation.py
├── configs/
│   └── xgb_backbone_search_space.json
├── tests/
│   ├── test_protocols.py
│   ├── test_real_data.py
│   ├── test_statistics.py
│   └── test_audit.py
└── results/
    └── <run_id>/
```

## 3. 模块职责

### `protocols.py`

- 定义 `VariantSpec`、real datasets、synthetic cases 和协议；
- 生成默认 16 个唯一变体；
- 冻结 main/pred/scoreagg 三个视图；
- 验证 ID 唯一、`p` 范围、aggregation 映射和主表完整性。

### `io_utils.py`

- canonical config hash；
- actual XGBoost binary/source identity 与 execution hash；
- 数据、split、index 和文件 SHA256；
- 原子 JSON/CSV/Parquet 写入；
- run-id 冲突与 resume config-hash 校验。

### `modeling.py`

- 从 XGB candidate 生成共享 MetaGB backbone；
- 仅用 inner-CV 选择 XGB backbone；
- 构建 XGB/Bias/SplitOnly/Full estimator；
- 保证 `reg_lambda == metagb_lambda_l2`；
- 统一 fit/predict/probability normalization。

### `real_data.py`

- 只加载并清洗原始行，不在 split 前编码特征；
- 每个 outer-train 独立拟合 encoder，再 transform outer-test；
- 每个 inner-train 再拟合独立 encoder，再 transform inner-validation；
- ordinal 未知类别固定为 `-1`，one-hot 未知类别忽略。

### `run_ablation.py`

- CLI 与 plan-only；
- 加载真实数据、生成 synthetic 数据；
- 创建 repeated outer/inner splits；
- 逐 outer task 选择 XGB backbone；
- 在 outer test 上评估全部冻结配置；
- 保存 task shard、预测、backbone 和 split manifest；
- resume、严格逐 task/variant/sample 完整性审计和派生统计。

### `statistics.py`

- fold → seed/repeat-level 聚合；
- paired bootstrap CI；
- 主四类表；
- `Δsplit/Δleaf/Δtotal`；
- pred sensitivity 长表；
- scoreagg 表与相对 greedy 的 paired effect。

## 4. 运行阶段

### Stage A：Plan

```text
输入：protocol + CLI filters
输出：protocol.json, variant_manifest.csv, task_plan.csv, split_manifest.json
训练：无
```

验证：

- 变体数恰为 16；
- 每个 task 的预期结果数为 16；
- 所有 ID 唯一；
- main 表四项存在；
- sensitivity 两条完整 `p` 网格存在。

### Stage B：XGB Backbone Selection

每个 outer task：

```text
raw outer train
  └── each inner train: fit preprocessing
       └── transform inner validation
            └── inner CV over XGB public candidates
                 └── selected public backbone
```

禁止：

- 读取 outer test 指标选 backbone；
- 为 MetaGB 重新选择公共参数；
- 从历史 summary 宽松匹配不带 split/data hash 的 backbone。

Synthetic 使用独立 development stage：每个 `scenario × dimension` 只在
development seed 9001 上选择一次 backbone，之后跨 evaluation seeds/folds
冻结复用；不为每个 synthetic outer fold 重复 HPO。不同
`scenario × dimension` cells 由 `task_workers` 并行选择，正式协议共有 14
个 cells，因此该阶段最多并行使用 14 核。

### Stage C：Frozen Outer Evaluation

```text
selected XGB backbone
  ├── xgb
  ├── bias_p00_sa0
  ├── split_p02_sa0 ... split_p10_sa0
  ├── full_p02_sa0  ... full_p10_sa0
  ├── full_p10_sa1
  ├── full_p10_sa2
  ├── split_p10_sa1
  └── split_p10_sa2
```

所有配置共享：

```text
outer train/test
public params
model seed
preprocessing
objective
```

### Stage D：Aggregation

1. 合并 task shards；
2. 校验每个成功 task 恰有 16 个 variant rows；
3. 写 `fold_results.csv` 和 `predictions.parquet`；
4. 聚合为 unit-level；
5. 生成三套论文友好表；
6. 写 completeness audit。

## 5. Resume 与防串跑

`protocol.json` 保存 canonical `config_hash`。若同一 `run_id` 已存在：

- 未指定 `--resume`：拒绝覆盖；
- 指定 `--resume` 且 hash 相同：仅运行缺失 shard；
- 指定 `--resume` 但 hash 不同：拒绝继续。

每个 shard 文件名包含：

```text
domain + dataset/scenario + dimension + data_seed/repeat + outer_fold
```

每个 shard 的 `task_hash` 还绑定 `execution_hash`，后者包含 synthetic
backbone、XGB candidates、实际动态库、Python/科学计算依赖版本和 `n_jobs`
的 identity。

## 6. Backbone Manifest 最低字段

```text
task_id
domain
dataset/scenario
data_seed/cv_repeat
outer_fold
outer_train_hash
outer_test_hash
inner_manifest_hash
search_space_hash
xgboost_source_hash
selected_candidate_id
validation_metric
params_json
```

只有全部 identity 字段匹配时，未来版本才允许复用历史 backbone。

## 7. 预期配置数量

正式默认：

```text
XGB                               1
Bias p0 greedy                    1
SplitOnly p0.2...1.0 greedy       5
Full p0.2...1.0 greedy            5
Full p1 mean/median               2
SplitOnly p1 mean/median          2
-----------------------------------
Total                            16
```

工程调试时可显式传入 `--no-split-scoreagg`，仅移除：

```text
split_p10_sa1
split_p10_sa2
Total = 14
```

该 14 配置模式不属于正式论文协议，正式启动脚本默认并要求 16 配置。

## 8. 测试门

### 配置级

- 16 个 ID 唯一；
- greedy/mean/median 映射为 0/1/2；
- Bias 为 `p=0, leaf_only_bias=true`；
- SplitOnly 全部 `leaf_only_bias=true`；
- Full 全部 `leaf_only_bias=false`；
- `p=1` 主表行可被三个实验视图复用。

### 公平性

- 全变体公共参数逐字段相同；
- lambda 映射一致；
- split hash 完全一致；
- XGB HPO 不读取 outer test；
- preprocessing vocabulary 不读取 outer validation/test；
- model seed 一致。

### 结果

- 每个成功 task 恰有 16 个 fold rows；
- 每个 test sample 恰有 16 组预测；
- 分类概率有限且每行和为 1；
- sensitivity 与 scoreagg 不做 outer-test winner selection；
- config hash 不同的 resume 被拒绝。
- fold/prediction/backbone/shard 的 run/config/execution/variant identity 必须
  唯一且等于 manifest 预期值。

### 最小训练 smoke

- `regime_slope`, dimension 10, one seed, one outer fold；
- XGB candidate budget 1；
- 16 个配置全部状态为 `ok`；
- 派生表与 completeness audit 均生成。

## 9. 分阶段实现状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| S0 | 文档、协议和产物契约 | implemented |
| S1 | 默认 16 变体生成与审计 | implemented |
| S2 | XGB inner-CV backbone + frozen outer runner | implemented |
| S3 | shard/resume/manifest | implemented |
| S4 | 主表、sensitivity、scoreagg 统计 | implemented |
| S5 | plan-only、14/16 synthetic、real、resume、zero-task | verified |
| S6 | Synthetic 结论 hash 门控 Real | implemented |
| S7 | 逐树 realized candidate telemetry | planned；底层尚未暴露 |
| S8 | publication heatmap renderer | planned；读取真实结果后实现 |

本地验证证据记录在 [VERIFICATION_REPORT_ZH.md](VERIFICATION_REPORT_ZH.md)。

## 10. Synthetic → Real 阶段门控

启动器不提供 `start all`。正式流程必须依次完成：

1. `start synthetic`；
2. `completeness_audit.json` 的 `passed=true`；
3. 基于主消融、pred sensitivity、scoreagg 六格、机制匹配场景、负对照和
   seed-level 不确定性填写 `SYNTHETIC_CONCLUSION.md`；
4. 结论无占位符，且含
   `SYNTHETIC_ANALYSIS_STATUS: COMPLETE` 与 `REAL_STAGE_DECISION: GO`；
5. `approve-synthetic` 写入 `synthetic_analysis_approval.json`；
6. `start real` 再次校验 audit、16 variants、config/execution/variant hash
   与结论 SHA256，全部一致才启动。

这样能防止“Synthetic 刚跑完就自动进入 Real”，也能防止批准后修改结论而未
重新审阅。
