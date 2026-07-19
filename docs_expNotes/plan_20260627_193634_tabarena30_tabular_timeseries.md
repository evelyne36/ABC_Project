# Plan: tabarena_30 + Tabular/Time-Series 扩展

生成时间：2026-06-27 19:36:34  
任务名：tabarena30_tabular_timeseries

## 1. 背景与原则

当前 `current_metagb` 是嵌入 XGBoost 的 CPU hist updater。它保留 MetaGB 的核心机制：

- conditional feature 负责切分样本；
- predictive feature 负责叶子响应；
- 叶输出为 `beta * x_j`；
- bias predictive feature 恒为 1，使模型可退化为常数叶。

因此，MetaGB 的优势更可能出现在存在“局部一维线性响应”“conditional/predictive 分工明显”“局部异质性”的数据中；同时，计算压力近似受 `processed_samples * processed_features^2` 支配。后续所有新增数据和 `tabarena_30` 队列都以此为主约束。

## 2. 总目标

1. 新建 `tabarena_30/`，将 TabArena binary/multiclass 中昂贵 pending 的 Random300 MetaGB 搜索改为：先复用或补跑 XGBoost 最优 shared backbone，再只搜索 MetaGB 专项的 `10 pred_fraction * 3 score_agg = 30` 个组合。
2. 新增 tabular 扩展工程时，先列出候选全集，再用计算经济性与 MetaGB 机制匹配度筛选。
3. 新增 time-series 扩展工程时，先列出可用候选全集，再筛选，且使用对应顶会/主流 benchmark 的时间顺序处理口径，避免与既有工程混杂。
4. 所有新工程独立建目录，避免影响 `datacollection0624/`、`random300_tabpfn/`、`timeseries_metagb_adapter/` 等旧工程。
5. `in_vehicle_coupon_recommendation.csv` 从 `tabarena_30` 队列排除，因为 onehot 严格删除缺失行后仅剩 108 行，作为正式补算证据价值不足。
6. `tabarena_30` 的 drop 口径调整为先删除任一特征缺失行，再丢弃离散特征列；这样 HR 等数据的 drop 行数与 onehot 行数口径一致。

## 3. `tabarena_30/` 方案

### 3.1 实验语义

对每个 `(task_type, preprocessing, dataset, outer_fold)`：

1. 找到对应 `xgb_cpu` Random300 的 selected candidate。
2. 抽取共享 backbone：
   - `learning_rate`
   - `max_depth`
   - `subsample`
   - `colsample_bytree`
   - `n_estimators`
   - `max_bin`
   - `lambda_l2` / `reg_lambda`
3. 将 `reg_lambda` 映射为 `metagb_lambda_l2`。
4. 固定以上 shared backbone，只生成 30 个 MetaGB candidates：
   - `pred_feature_fraction = 0.1, 0.2, ..., 1.0`
   - `metagb_score_agg = 0, 1, 2`
5. 在该 outer fold 的 outer-train 内继续做 inner CV，从 30 个 MetaGB candidate 中选最优。
6. 用 outer-train 重训 selected MetaGB candidate，在 outer-test 上报告结果。

说明：`score_agg=3` 不进入主路径，因为当前文档已将其标记为并行随机性/复现语义不稳。

### 3.2 输出格式

`tabarena_30/` 需要保持与 Random300/Random600 相近的输出，方便后续合表：

- `results/<run_id>/candidate_configs.json`
- `results/<run_id>/parameter_audit.json`
- `results/<run_id>/candidates.csv`
- `results/<run_id>/summary.csv`
- `results/<run_id>/aggregate.csv`
- `results/<run_id>/pred_fraction_analysis.csv`
- `results/<run_id>/run_metadata.json`
- `logs/<run_id>/task_status.json`
- `logs/<run_id>/tasks/*.log`

额外新增：

- `xgb_backbone_source.csv`：记录每个 outer fold 的 XGB selected candidate 来源路径、candidate_id、backbone_id、参数。
- `queue_priority.csv`：记录计算压力、用户慢速标记、最终执行顺序。

### 3.3 初始执行队列

排序规则：

1. 未被用户标记为慢的数据，按 `processed_rows * processed_features^2` 从小到大。
2. 用户已标记非常慢或明显慢的数据，强制放到队尾。
3. `in_vehicle_coupon_recommendation.csv` 排除。
4. `kddcup09_appetency.csv` 在当前预处理下无可用 rows，暂列为检查/排除项，不进入正式 MetaGB 运行。
5. 若 XGB 也 pending，则先补跑 XGB Random300，再跑 30-grid MetaGB。

| Order | Task | Enc | Dataset | Proc rows | Proc features | Pressure | XGB pending | MGB pending | Note |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| - | binary | onehot | `in_vehicle_coupon_recommendation.csv` | 108 | 59 | 375948 | 5 | 5 | excluded: only 108 rows after strict missing-row removal |
| - | binary | drop | `in_vehicle_coupon_recommendation.csv` | 108 | 7 | 5292 | 5 | 5 | excluded with onehot pair |
| 1 | binary | drop | `bank-marketing.csv` | 45211 | 5 | 1130275 | 5 | 5 |  |
| 4 | binary | drop | `online_shoppers_intention.csv` | 12330 | 14 | 2416680 | 5 | 5 |  |
| 5 | binary | drop | `jm1.csv` | 10880 | 21 | 4798080 | 5 | 5 |  |
| 6 | binary | onehot | `jm1.csv` | 10880 | 21 | 4798080 | 5 | 5 |  |
| 7 | binary | drop | `heloc.csv` | 10459 | 23 | 5532811 | 5 | 5 |  |
| 8 | binary | onehot | `heloc.csv` | 10459 | 23 | 5532811 | 5 | 5 |  |
| 9 | binary | onehot | `online_shoppers_intention.csv` | 12330 | 29 | 10369530 | 5 | 5 |  |
| 10 | binary | drop | `credit_card_clients_default.csv` | 30000 | 23 | 15870000 | 5 | 5 |  |
| 11 | binary | onehot | `credit_card_clients_default.csv` | 30000 | 23 | 15870000 | 5 | 5 |  |
| 12 | binary | drop | `APSFailure.csv` | 756 | 170 | 21848400 | 0 | 5 |  |
| 13 | binary | onehot | `APSFailure.csv` | 756 | 170 | 21848400 | 0 | 5 |  |
| 14 | binary | drop | `customer_satisfaction_in_airline.csv` | 129487 | 18 | 41953788 | 5 | 5 |  |
| 15 | binary | onehot | `NATICUSdroid.csv` | 7491 | 86 | 55403436 | 5 | 5 |  |
| 16 | binary | onehot | `taiwanese_bankruptcy_prediction.csv` | 6819 | 94 | 60252684 | 5 | 5 |  |
| 17 | binary | onehot | `bank-marketing.csv` | 45211 | 37 | 61893859 | 5 | 5 |  |
| 18 | binary | drop | `coil2000_insurance_policies.csv` | 9822 | 80 | 62860800 | 5 | 5 |  |
| 19 | binary | onehot | `customer_satisfaction_in_airline.csv` | 129487 | 25 | 80929375 | 5 | 5 |  |
| 20 | binary | onehot | `coil2000_insurance_policies.csv` | 9822 | 149 | 218058222 | 5 | 5 |  |
| 21 | binary | onehot | `HR_Analytics_Job_Change_of_Data_Scientists.csv` | 8955 | 177 | 280551195 | 0 | 5 |  |
| 22 | binary | onehot | `Bioresponse.csv` | 3751 | 1776 | 11831314176 | 0 | 5 | unmarked but high pressure |
| 23 | binary | drop | `kddcup09_appetency.csv` | - | - | - | 5 | 5 | preprocess check/exclude |
| 24 | binary | onehot | `kddcup09_appetency.csv` | - | - | - | 5 | 5 | preprocess check/exclude |
| 25 | multiclass | onehot | `anneal.csv` | 898 | 82 | 6038152 | 0 | 5 | USER_SLOW: 50epoch/46h |
| 26 | binary | onehot | `polish_companies_bankruptcy.csv` | 3031 | 64 | 12414976 | 0 | 5 | USER_SLOW: 50epoch/12h |
| 27 | binary | drop | `NATICUSdroid.csv` | 7491 | 86 | 55403436 | 0 | 5 | USER_SLOW: 50epoch/34h |
| 28 | binary | drop | `taiwanese_bankruptcy_prediction.csv` | 6819 | 94 | 60252684 | 0 | 5 | USER_SLOW: 50epoch/39h |
| 29 | multiclass | onehot | `splice.csv` | 3190 | 287 | 262757110 | 0 | 5 | USER_SLOW: 1epoch/6h |
| 30 | multiclass | onehot | `students_dropout_and_academic_success.csv` | 4424 | 255 | 287670600 | 5 | 5 | USER_RUNNING: student ing |
| 31 | binary | onehot | `Marketing_Campaign.csv` | 2216 | 697 | 1076552744 | 0 | 5 | USER_SLOW: 1epoch/2h |
| 32 | multiclass | onehot | `hiva_agnostic.csv` | 3845 | 1617 | 10053479205 | 0 | 5 | USER_SLOW: unknown, likely very high |
| 33 | binary | drop | `Bioresponse.csv` | 3751 | 1776 | 11831314176 | 0 | 5 | USER_SLOW: 1epoch/132h |
| 34 | multiclass | drop | `hiva_agnostic.csv` | 3845 | 1617 | 10053479205 | 0 | 5 | USER_VERY_SLOW: 1epoch/7d |

重点提醒：

- `Bioresponse`、`hiva_agnostic` 的 feature 数超过 1600，属于 MetaGB 压力极高项。
- 用户已标记 `hiva_agnostic drop` 为 `1epoch/7d`，必须放到最后。
- `Marketing_Campaign onehot` 虽然 pressure 小于 Bioresponse/hiva，但用户标记 `1epoch/2h`，也放入慢速尾部。
- `anneal onehot` 的 pressure 不大但用户标记 `50epoch/46h`，同样不应过早占用主队列。

## 4. Tabular 扩展工程

建议新建目录：`tabular_extra_random300/`

### 4.1 先列候选全集

先不要直接筛掉数据，而是建立 `candidate_inventory.csv`，至少包含：

- 当前 `dataset/TabPFN/tabpfn_openml_manifest.json` 中已有数据；
- `dataset/TabPFN/tabpfn_tree_model_candidates.md` 中提到但尚未下载的数据；
- 后续从 OpenML / PMLB / UCI / AutoML benchmark 找到的候选。

现有 TabPFN manifest 已有候选：

| Task | Datasets |
|---|---|
| regression | `kin8nm`, `energy_efficiency`, `naval_propulsion_plant`, `cpu_activity`, `space_ga`, `Moneyball` |
| binary | `banknote-authentication`, `credit-approval`, `breast-w`, `ilpd`, `haberman`, `kc1`, `electricity` |
| multiclass | `vehicle`, `segment`, `satimage`, `vowel`, `optdigits` |

现有候选文档中还提到的候选：

| Task | Datasets |
|---|---|
| regression | `abalone`, `boston`, `friedman1` |
| binary | `phoneme`, `mozilla4`, `pc1` |
| multiclass | `balance-scale`, `mfeat-fourier`, `mfeat-karhunen` |

### 4.2 再筛选

筛选分三层：

1. 可运行性过滤：
   - CSV 可下载、target 明确；
   - classification 类别数与 task type 一致；
   - 缺失处理后仍有有效 rows/features。
2. 计算经济性过滤：
   - 优先 `features <= 80`；
   - 优先 `pressure_proxy = rows * features^2` 较小；
   - 大样本少特征可保留，但宽表要谨慎。
3. MetaGB 机制匹配过滤：
   - 复用或迁移 `metagb_preflight_adapter` 的 mechanism-alignment scorer；
   - 优先 local linear leaf advantage、conditional/predictive decoupling、local heterogeneity 高的数据；
   - 输出 `mechanism_alignment_score`，但不把最终实验胜负用于训练筛选器。

### 4.3 产物

- `tabular_extra_random300/candidate_inventory.csv`
- `tabular_extra_random300/selected_manifest.csv`
- `tabular_extra_random300/configs/default_search_space.json`
- `tabular_extra_random300/scripts/download_datasets.py`
- `tabular_extra_random300/scripts/run_random300_server.py`
- `tabular_extra_random300/README.md`

## 5. Time-Series 扩展工程

建议新建目录：`timeseries_extra_random300/`

### 5.1 先列候选全集

先建立 `candidate_inventory.csv`，把可选时序数据全部列出，再筛选。现有 `timeseries_metagb_adapter` 已登记：

| Family | Datasets |
|---|---|
| ETDataset / Informer-style | `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2` |
| LSTNet multivariate | `exchange_rate`, `electricity`, `traffic`, `solar` |

后续候选池建议继续纳入但先不全部运行：

| Family | Candidate examples | 备注 |
|---|---|---|
| LSTF common benchmark | `Weather`, `ILI`, `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08` | 与 TFB / Informer / Autoformer / PatchTST 系列更容易对齐 |
| energy/electricity | electricity variants, ECL | 通常数值特征，适合 drop-only |
| traffic | PEMS / traffic occupancy | 多变量，需控制窗口后特征数 |
| economic/finance | exchange / FRED-style | 少特征或多序列，较经济 |

### 5.2 再筛选

筛选原则：

1. 可比性优先：
   - 使用对应 benchmark 的时间顺序 split；
   - 特征工程与顶会或主流 benchmark 口径保持一致；
   - 不做 onehot，默认 drop-only 或纯数值窗口化。
2. 防泄漏：
   - validation/test 必须晚于 train；
   - scaler 只 fit train；
   - rolling/window features 只使用过去；
   - 不使用 drop-last trick 改变 test 覆盖范围。
3. 计算经济性：
   - 控制 `seq_len * channels`，优先 `seq_len=24/48/96` 的小中规模；
   - 对高维 traffic/PEMS，先做 target-subset 或小 horizon smoke；
   - 先 `max_candidates=30` smoke，再决定是否 Random300。
4. MetaGB 匹配度：
   - 关注局部趋势、 regime switching、不同变量作为 conditional/predictive 的潜在分工；
   - 用时序版 mechanism audit 评估局部线性、局部异质性、变量贡献集中度。

### 5.3 产物

- `timeseries_extra_random300/candidate_inventory.csv`
- `timeseries_extra_random300/selected_manifest.csv`
- `timeseries_extra_random300/configs/dataset_registry.json`
- `timeseries_extra_random300/scripts/download_datasets.py`
- `timeseries_extra_random300/scripts/run_timeseries_benchmark.py`
- `timeseries_extra_random300/tests/test_time_splits.py`
- `timeseries_extra_random300/tests/test_window_no_leakage.py`
- `timeseries_extra_random300/README.md`

## 6. 执行步骤

### Phase A: 状态审计

1. 读取 `datacollection0624/**/task_status.json`。
2. 按最新 `updated_at` 合并同一 task_id 的状态。
3. 生成：
   - `tabarena_30/audits/task_status_latest.csv`
   - `tabarena_30/audits/pending_by_dataset.csv`
   - `tabarena_30/audits/queue_priority.csv`
   - `tabarena_30/audits/tabarena_missing_drop_rates.csv`
   - `tabarena_30/audits/audit_report.md`
4. 标记用户已知慢速项，并强制进入队尾。

### Phase B: `tabarena_30` 工程

1. 新建 `tabarena_30/`。
2. 复制并瘦身 `random300_tabpfn` / `random300` 的 runner 结构。
3. 增加 XGB best-backbone extractor。
4. 增加 fixed-backbone 30-grid MetaGB candidate generator。
5. 增加 dry-run 队列模式，先输出将运行哪些任务，不训练模型。
6. 做 smoke：
   - 选 1 个小数据；
   - 1 outer fold；
   - 2 inner folds；
   - 30 MetaGB candidates；
   - 验证输出 schema。
7. 再按队列顺序执行正式任务。

### Phase C: Tabular 扩展

1. 新建 `tabular_extra_random300/`。
2. 生成候选全集 inventory。
3. 下载/转换候选数据到独立 `data/raw` 与 `data/processed`。
4. 计算 rows/features/pressure_proxy。
5. 跑 mechanism-alignment scorer。
6. 生成 selected manifest。
7. smoke 后进入 Random300。

### Phase D: Time-Series 扩展

1. 新建 `timeseries_extra_random300/`。
2. 生成候选全集 inventory。
3. 复用现有时序 adapter 的防泄漏窗口化与 ordered split 思路。
4. 下载或登记可用数据。
5. 先跑 30-candidate smoke。
6. 只对通过 smoke 且压力可控的数据跑 Random300。

## 7. 验收标准

### `tabarena_30`

- `candidate_configs.json` 中每个 MetaGB task 恰好 30 个候选。
- 30 个候选覆盖完整 `10 * 3` grid。
- 每个 MetaGB candidate 的 shared backbone 与对应 XGB selected candidate 一致。
- `summary.csv` 能追溯到 XGB backbone 来源。
- `queue_priority.csv` 中慢速标记项位于队尾。
- smoke 至少完成 1 个 dataset、1 个 outer fold，并输出 `summary.csv`、`candidates.csv`、`aggregate.csv`。

### Tabular 扩展

- 先产出 candidate inventory，再产出 selected manifest。
- selected manifest 每行包含 rows/features/pressure_proxy/mechanism score/选择理由。
- 不直接丢弃未选数据，未选原因必须记录，方便结果不够时回捞。

### Time-Series 扩展

- 先产出 candidate inventory，再产出 selected manifest。
- split 和 scaler 测试通过。
- 不使用 onehot。
- 不使用未来信息构造窗口特征。
- smoke 输出完整。

## 8. 风险与处理

| 风险 | 处理 |
|---|---|
| XGB selected candidate 缺失 | 先补跑对应 XGB Random300 fold，再进入 MetaGB 30-grid |
| 旧结果目录 schema 不一致 | 写 extractor 时兼容 `summary.csv` 与 `candidates.csv` 两种来源 |
| 用户标记慢速但 pressure 不高 | 以实测慢速为准，强制队尾 |
| pressure 高但未标记慢 | 在 queue 中标 `high pressure`，优先 dry-run/smoke |
| time-series 特征过宽 | 先缩短 `seq_len` 或只预测 subset，再考虑 full |
| 新 tabular 数据太容易 | 保留在 inventory，但 selected manifest 降低优先级 |

## 9. 下一步

经审查确认后，按以下顺序实施：

1. 建 `tabarena_30/` 的审计脚本与 30-grid runner。
2. 跑 dry-run，确认队列与慢速标记。
3. 跑最小 smoke。
4. 建 `tabular_extra_random300/` inventory。
5. 建 `timeseries_extra_random300/` inventory。
6. 再决定正式任务批次。
