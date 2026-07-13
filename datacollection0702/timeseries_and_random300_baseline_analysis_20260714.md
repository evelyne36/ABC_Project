# datacollection0702: timeseries 与 random300 baseline 结果分析

生成时间：2026-07-14 00:59:22 CST (+0800)

## 1. 范围与证据边界

本报告分析两个目标结果块：

- `datacollection0702/datacollection_ts_tree_seed42__non_pems/`：timeseries 工程的非 PEMS 树模型实验结果，核心文件为 `summary.csv`、`candidates.csv`、`split_manifest.json` 与 `parameter_audit.json`。
- `datacollection0702/datacollection_baselinemodel_random300/`：random300 系列 tabular 主实验的 baseline 对比结果，核心证据来自 `tasks/*.log`、`task_status.json` 与 `random300_baselinemodel_full_20260708_154605/parameter_audit.json`。

重要边界：

- timeseries 块不是完整的 7 数据集 x 3 horizon x 4 模型矩阵。LightGBM、XGBoost、CatBoost 各有 21 个 cell；`current_metagb` 只完成 5 个 cell。
- baseline 块 `task_status.json` 显示 total=2160、completed=1662、failed=0。实际目录中有 1692 个日志文件，其中 1662 个含有 `finish ... status=ok`，30 个已启动但未完成。
- baseline 计划包含 8 个模型：linear、extra_trees、random_forest、lightgbm、catboost、torchmlp、realmlp、tabm。但当前日志中没有 `tabm`；`realmlp` 只启动 72 个任务并完成 55 个；其余常规模型基本完整。
- 本报告的 baseline 排名使用 dataset-level mean test metric：AUC 越高越好，logloss/RMSE 越低越好。与 XGBoost/MetaGB 对齐时，仅使用同一 dataset、同一预处理、同一任务类型且 5 folds 完整的 cells。

## 2. Timeseries 结果

### 2.1 Coverage

timeseries `summary.csv` 共 68 条记录，全部 `status=ok`。数据集为 7 个：ETTh1、ETTh2、ETTm1、ETTm2、ILI、Weather、exchange_rate。输入窗口固定为 `seq_len=24`，预测 horizon 为 `pred_len=1/6/12`。

| 模型 | 完成 cells | 说明 |
| --- | ---: | --- |
| lgbm_cpu | 21 | 7 datasets x 3 horizons 完整 |
| xgb_cpu | 21 | 7 datasets x 3 horizons 完整 |
| catboost_cpu | 21 | 7 datasets x 3 horizons 完整 |
| current_metagb | 5 | 仅 ETTh1 h=1、ETTh2 h=1、ILI h=1/6、exchange_rate h=1 |

### 2.2 树模型主结果

在 LightGBM、XGBoost、CatBoost 三个完整树模型之间，21 个 timeseries cells 的 RMSE 最优模型分布如下：

| 模型 | RMSE wins | median relative gap to best | median elapsed |
| --- | ---: | ---: | ---: |
| lgbm_cpu | 13/21 | 0.00% | 6559.8s |
| xgb_cpu | 7/21 | 0.23% | 9728.5s |
| catboost_cpu | 1/21 | 9.80% | 10092.7s |

结论：当前 timeseries 证据下，LightGBM 是最强的常规树 baseline。XGBoost 在 exchange_rate 的较长 horizon 和若干 ETTh/ETTm 长 horizon 上有竞争力，但在 Weather 上明显弱于 LightGBM。CatBoost 只在 ETTh1 h=12 的一个 cell 上最佳。

### 2.3 Dataset/horizon 级别表

| Dataset | pred_len | 可用模型数 | Best | Best RMSE | LGBM | XGB | CatBoost | MetaGB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 1 | 4 | lgbm_cpu | 0.071221 | 0.071221 | 0.071335 | 0.078201 | 0.073488 |
| ETTh1 | 6 | 3 | xgb_cpu | 0.123579 | 0.124364 | 0.123579 | 0.133783 | NA |
| ETTh1 | 12 | 3 | catboost_cpu | 0.161291 | 0.176800 | 0.164842 | 0.161291 | NA |
| ETTh2 | 1 | 4 | lgbm_cpu | 0.065521 | 0.065521 | 0.066790 | 0.092155 | 0.077891 |
| ETTh2 | 6 | 3 | lgbm_cpu | 0.195004 | 0.195004 | 0.200976 | 0.203930 | NA |
| ETTh2 | 12 | 3 | lgbm_cpu | 0.270749 | 0.270749 | 0.276898 | 0.280772 | NA |
| ETTm1 | 1 | 3 | lgbm_cpu | 0.037507 | 0.037507 | 0.037939 | 0.041266 | NA |
| ETTm1 | 6 | 3 | lgbm_cpu | 0.068280 | 0.068280 | 0.068439 | 0.072772 | NA |
| ETTm1 | 12 | 3 | xgb_cpu | 0.090940 | 0.091589 | 0.090940 | 0.094457 | NA |
| ETTm2 | 1 | 3 | lgbm_cpu | 0.027295 | 0.027295 | 0.038871 | 0.031610 | NA |
| ETTm2 | 6 | 3 | lgbm_cpu | 0.059794 | 0.059794 | 0.066006 | 0.072370 | NA |
| ETTm2 | 12 | 3 | xgb_cpu | 0.105096 | 0.110697 | 0.105096 | 0.117466 | NA |
| ILI | 1 | 4 | current_metagb | 0.540101 | 1.954420 | 1.778242 | 1.905599 | 0.540101 |
| ILI | 6 | 4 | current_metagb | 0.804421 | 1.879966 | 1.880682 | 1.895341 | 0.804421 |
| ILI | 12 | 3 | lgbm_cpu | 1.962220 | 1.962220 | 1.965300 | 1.973670 | NA |
| Weather | 1 | 3 | lgbm_cpu | 0.038201 | 0.038201 | 0.307432 | 0.106785 | NA |
| Weather | 6 | 3 | lgbm_cpu | 0.201545 | 0.201545 | 0.409040 | 0.355725 | NA |
| Weather | 12 | 3 | lgbm_cpu | 0.254950 | 0.254950 | 0.642008 | 0.255831 | NA |
| exchange_rate | 1 | 4 | current_metagb | 0.050732 | 0.179309 | 0.069022 | 0.277689 | 0.050732 |
| exchange_rate | 6 | 3 | xgb_cpu | 0.117778 | 0.170994 | 0.117778 | 0.228405 | NA |
| exchange_rate | 12 | 3 | xgb_cpu | 0.175770 | 0.188430 | 0.175770 | 0.424492 | NA |

### 2.4 MetaGB timeseries 证据

`current_metagb` 完成 5 个 cell，其中 3 个为全模型最优：

| Cell | MetaGB RMSE | XGB RMSE | LGBM RMSE | 相对 XGB 改善 | MetaGB/XGB elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| ILI, h=1 | 0.540101 | 1.778242 | 1.954420 | +69.63% | 74.54x |
| ILI, h=6 | 0.804421 | 1.880682 | 1.879966 | +57.23% | 58.94x |
| exchange_rate, h=1 | 0.050732 | 0.069022 | 0.179309 | +26.50% | 93.92x |
| ETTh1, h=1 | 0.073488 | 0.071335 | 0.071221 | -3.02% vs XGB | 92.58x |
| ETTh2, h=1 | 0.077891 | 0.066790 | 0.065521 | -16.62% vs XGB | 89.25x |

解释上应谨慎：MetaGB 在 ILI 和 exchange_rate 的短 horizon 上非常强，但 ETTh1/ETTh2 短 horizon 明确不如 LightGBM/XGBoost。由于 MetaGB coverage 只有 5 个 cell，timeseries 结果目前最适合写成“早期/部分 evidence：在若干非平稳或多变量目标上有显著收益，但覆盖不足且计算代价很高”，而不是完整 benchmark 结论。

### 2.5 Timeseries 论文写法建议

可支持的强表述：

- 在非 PEMS timeseries 树模型实验中，LightGBM 是当前最稳健的常规树 baseline。
- MetaGB 在 ILI 与 exchange_rate 短 horizon 上表现出大幅优势，说明 local predictive response 可能对某些时间序列结构有效。
- 当前 timeseries MetaGB 证据受到 coverage 和 runtime 限制；应作为 targeted positive evidence，而非整体胜率证据。

不建议写的表述：

- 不宜说 MetaGB 在 timeseries 上整体优于树模型；可用证据只有 5 个 MetaGB cells。
- 不宜把 Weather 上的 XGBoost 弱表现泛化为所有 boosting 方法弱，因为 LightGBM 在 Weather 上表现最好。

## 3. Tabular random300 baseline 结果

### 3.1 Coverage 与模型完成度

baseline 结果覆盖 27 个 tabular 数据集、2 种 categorical handling、5 outer folds。理论任务数为 2160，即 8 models x 2 preprocessing x 27 datasets x 5 folds。

| 模型 | 完成日志 | 已启动日志 | 备注 |
| --- | ---: | ---: | --- |
| linear | 270 | 270 | 完整 |
| extra_trees | 270 | 270 | 完整 |
| random_forest | 270 | 270 | 完整 |
| lightgbm | 270 | 270 | 完整 |
| catboost | 257 | 270 | 少量未完成，主要影响部分 multiclass |
| torchmlp | 270 | 270 | 完整 |
| realmlp | 55 | 72 | 只覆盖小部分 cells |
| tabm | 0 | 0 | 当前没有日志结果 |

按 dataset-level 完整 5 folds 聚合后，baseline 至少有一个完整模型的 cells 为 54 个，即 27 datasets x 2 preprocessing 全覆盖。每个 cell 可比较的 baseline 模型数在 5 到 7 之间，原因是 `realmlp` 和少量 `catboost` 不完整。

### 3.2 最佳 baseline 模型分布

| 预处理 | 任务 | Cells | 最佳 baseline 分布 |
| --- | --- | ---: | --- |
| drop | binary | 10 | extra_trees 3, catboost 2, random_forest 2, lightgbm 1, linear 1, realmlp 1 |
| drop | multiclass | 8 | catboost 5, torchmlp 2, lightgbm 1 |
| drop | regression | 9 | catboost 4, torchmlp 3, extra_trees 1, linear 1 |
| onehot | binary | 10 | extra_trees 4, catboost 2, random_forest 2, torchmlp 1, linear 1 |
| onehot | multiclass | 8 | catboost 4, lightgbm 2, torchmlp 2 |
| onehot | regression | 9 | catboost 4, torchmlp 3, extra_trees 1, linear 1 |

总体上，CatBoost 是最稳定的 baseline 强者：在其完成的 51 个 cells 中平均 rank 为 2.10，并且在 multiclass/regression 中尤其强。LightGBM、ExtraTrees、TorchMLP 也有明确 positive pockets。RandomForest 主要在 binary 上有竞争力。Linear 虽然偶尔在简单数据上胜出，但整体平均排名最低。

### 3.3 与 random300 主实验 XGBoost/MetaGB 对齐比较

对齐 `tabular_drop_20260628_030502` 与 `tabular_onehot_20260628_025442` 中的 XGBoost/MetaGB 主实验后，baseline 与 XGBoost 有 51 个可比较 cells，与 MetaGB 有 46 个可比较 cells。

| 对照对象 | Cells | Best baseline 胜 | Tie | Best baseline 负 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 51 | 41 | 2 | 8 |
| MetaGB | 46 | 41 | 3 | 2 |

分任务看：

| 对照对象 | 任务 | Cells | Best baseline 胜/Tie/负 |
| --- | --- | ---: | --- |
| XGBoost | binary | 20 | 15 / 2 / 3 |
| XGBoost | multiclass | 13 | 10 / 0 / 3 |
| XGBoost | regression | 18 | 16 / 0 / 2 |
| MetaGB | binary | 20 | 16 / 2 / 2 |
| MetaGB | multiclass | 8 | 8 / 0 / 0 |
| MetaGB | regression | 18 | 17 / 1 / 0 |

这说明：在这 27 个 random300 tabular 数据集上，XGBoost/MetaGB 不是 strongest possible baseline 的代表。若论文 baseline 表只比较 XGBoost 与 MetaGB，容易被审稿人质疑“缺少强 baseline”。当前 baseline 结果反而提供了一个更严谨的结论：MetaGB 的真实数据故事不是“打败所有强 tabular baseline”，而是“相对 XGBoost 的结构扩展；与 CatBoost/ExtraTrees/MLP 等强 baseline 仍需明确报告竞争关系”。

### 3.4 代表性 baseline 优势案例

Best baseline 相对 MetaGB 的最大优势集中在 multiclass 与部分 regression：

| 预处理 | 任务 | Dataset | Best baseline | Baseline | MetaGB | 改善 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| onehot | multiclass | balance-scale.csv | torchmlp | 0.063948 | 0.342915 | 81.35% logloss 降低 |
| drop | multiclass | balance-scale.csv | torchmlp | 0.063948 | 0.342915 | 81.35% logloss 降低 |
| onehot | multiclass | vowel.csv | catboost | 0.051427 | 0.127105 | 59.54% logloss 降低 |
| drop | multiclass | vowel.csv | catboost | 0.055863 | 0.132048 | 57.69% logloss 降低 |
| onehot | regression | cpu_activity.csv | catboost | 2.221180 | 3.934440 | 43.55% RMSE 降低 |
| drop | regression | cpu_activity.csv | catboost | 2.221180 | 3.934440 | 43.55% RMSE 降低 |
| drop | regression | kin8nm.csv | torchmlp | 0.071155 | 0.100273 | 29.04% RMSE 降低 |
| onehot | regression | kin8nm.csv | torchmlp | 0.071155 | 0.100273 | 29.04% RMSE 降低 |

相对 XGBoost 的强 baseline 优势也很明显：

| 预处理 | 任务 | Dataset | Best baseline | Baseline | XGBoost | 改善 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| onehot | multiclass | vowel.csv | catboost | 0.051427 | 0.225064 | 77.15% logloss 降低 |
| drop | multiclass | vowel.csv | catboost | 0.055863 | 0.228168 | 75.52% logloss 降低 |
| onehot | multiclass | balance-scale.csv | torchmlp | 0.063948 | 0.233793 | 72.65% logloss 降低 |
| drop | multiclass | balance-scale.csv | torchmlp | 0.063948 | 0.233793 | 72.65% logloss 降低 |
| drop | regression | kin8nm.csv | torchmlp | 0.071155 | 0.112534 | 36.77% RMSE 降低 |
| onehot | regression | kin8nm.csv | torchmlp | 0.071155 | 0.112534 | 36.77% RMSE 降低 |

### 3.5 MetaGB 相对 strongest baseline 的 positive pockets

MetaGB 相对 best baseline 只有 2 个明确获胜 cells，且幅度很小：

| 预处理 | 任务 | Dataset | Best baseline | Baseline | MetaGB | 差异 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| onehot | binary | ilpd.csv | extra_trees | 0.767457 | 0.769598 | MetaGB +0.002141 AUC |
| drop | binary | ilpd.csv | catboost | 0.759327 | 0.759708 | MetaGB +0.000381 AUC |

还有 3 个 tie：banknote-authentication drop/onehot 接近满分 AUC，abalone onehot 的 RMSE 差异约 0.03%。因此，在 random300 tabular baseline 对比中，MetaGB 对 strongest baseline 的优势目前不强；它的论文定位更适合作为 XGBoost family 的结构性扩展，而不是 broad tabular SOTA。

### 3.6 Categorical handling 的观察

在 baseline 模型中，drop 与 onehot 的最佳模型分布相当接近：

- binary：ExtraTrees/RandomForest/CatBoost 经常胜出；onehot 没有显著改变总体格局。
- multiclass：CatBoost 仍是主导 baseline；TorchMLP 在 balance-scale、vowel 等数据集上非常强。
- regression：CatBoost 与 TorchMLP 是主要强 baseline，onehot 与 drop 的 best-model 分布几乎一致。

因此，baseline 结果没有显示 onehot 是整体更优预处理；预处理效应更多是 dataset/model-specific。

## 4. 面向顶会论文的主结论

### 4.1 可防守的结论

1. Timeseries 非 PEMS 树模型实验中，LightGBM 是当前最强常规树 baseline；XGBoost 在部分 horizon 上有竞争力，CatBoost 整体较弱。
2. Timeseries MetaGB 在 ILI 与 exchange_rate 的短 horizon 上显示非常大的正向信号，但 coverage 只有 5 cells 且运行时间高，应作为 targeted evidence。
3. Tabular random300 baseline 对比显示，CatBoost、TorchMLP、ExtraTrees 等强 baseline 在大量 cells 上超过 XGBoost/MetaGB。这个结果应被纳入论文，以避免 baseline 不充分的风险。
4. MetaGB 在真实 tabular strongest-baseline 对比中不是 universal winner；它更适合被论述为 XGBoost 的 mechanism-aware extension，在 synthetic/机制匹配场景有强证据，在真实数据上表现为 selective advantage。

### 4.2 建议的论文写法

主文建议采用三层叙事：

- 方法机制层：使用已有 synthetic 结果支撑 local-linear-in-leaves / conditional predictive response 的归纳偏置。
- XGBoost family 层：报告 MetaGB vs XGBoost，说明结构扩展何时提升、何时不提升。
- 强 baseline 层：新增一张 compact baseline 表，明确 CatBoost/ExtraTrees/TorchMLP/LightGBM 等结果，承认 strongest tabular baseline 的竞争力。

对于 timeseries，建议只写成补充实验或 targeted extension：

- “On non-PEMS time-series tasks, tree baselines are competitive, with LightGBM strongest overall. MetaGB shows large gains on ILI and exchange-rate short-horizon cells, but its current coverage is partial and computationally expensive.”

### 4.3 不建议的论文写法

- 不建议宣称 MetaGB 在 random300 tabular 上击败 strong baselines；当前 aligned cells 中 best baseline 对 MetaGB 为 41/3/2。
- 不建议将 timeseries MetaGB 的 3 个强胜利外推为全局 timeseries 优势；目前只有 5 个 MetaGB cells。
- 不建议把 baseline folder 当作完整 2160 任务结果；`tabm` 缺失，`realmlp` 与部分 `catboost` 不完整。

## 5. 后续最小补证建议

若时间允许，最有价值的补证不是盲目补全全部任务，而是针对论文风险补最小集合：

1. Tabular：补齐 `tabm` 或明确从 baseline 表中移除 `tabm`，否则 protocol 中列出但没有结果会显得不完整。
2. Tabular：若要声称 strongest baseline 对比，优先补齐 `realmlp` 和 `catboost` 缺失 cells；如果不补，应在表注中说明 best baseline 是“available baselines”。
3. Timeseries：优先补 MetaGB 在 Weather、ETTm1、ETTm2 的 h=1，以及 ILI/exchange_rate 的 h=12。这比补所有 cells 更能检验当前 positive pockets 是否稳定。
4. Runtime：对 MetaGB timeseries 的 59x-94x XGB 耗时比，应在附录报告，避免审稿人通过复现实验成本提出质疑。

## 6. 证据文件

- `datacollection0702/datacollection_ts_tree_seed42__non_pems/summary.csv`
- `datacollection0702/datacollection_ts_tree_seed42__non_pems/candidates.csv`
- `datacollection0702/datacollection_ts_tree_seed42__non_pems/split_manifest.json`
- `datacollection0702/datacollection_ts_tree_seed42__non_pems/parameter_audit.json`
- `datacollection0702/datacollection_baselinemodel_random300/task_status.json`
- `datacollection0702/datacollection_baselinemodel_random300/tasks/*.log`
- `datacollection0702/datacollection_baselinemodel_random300/random300_baselinemodel_full_20260708_154605/parameter_audit.json`
- 对齐参考：`datacollection0702/tabular_drop_20260628_030502/tasks/*.log`
- 对齐参考：`datacollection0702/tabular_onehot_20260628_025442/tasks/*.log`
