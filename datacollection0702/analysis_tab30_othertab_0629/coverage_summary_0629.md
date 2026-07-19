# 0629 Data Collection Coverage Summary

统计口径：

- 完整结果：同一 `task_type + dataset + encoding/protocol` 下，`xgb_cpu` 5 个 outer folds 全部 `ok`，且 `current_metagb` 5 个 outer folds 全部 `ok`。
- TabArena 合并口径：`datacollection0624` 中的 random 主体结果加 `mgb_datacollection_by0629/tabarena30_*` 的 fixed-backbone MGB30 补跑结果，并集去重。
- TabArena random 主体中，binary/multiclass 是 random300，regression 是 random600。
- `No rows remain after preprocessing` 单独列为“预处理后无样本”，不放入完整或未完整。

## 总览

| 数据组 | encoding / 口径 | 原始总数 | 预处理后无样本 | 有效总数 | 完整 | 未完整 | 主要缺口 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| TabArena | drop | 44 | 0 | 44 | 38 | 6 | 主要剩 binary 长尾和 `hiva_agnostic.csv` |
| TabArena | onehot | 48 | 3 | 45 | 26 | 19 | TabArena30 onehot 补跑缺口多；3 个预处理后无样本 |
| Tabular other data | drop | 27 | 0 | 27 | 17 | 10 | multiclass 缺口最多 |
| Tabular other data | onehot | 27 | 0 | 27 | 12 | 15 | binary/multiclass/部分 regression 仍未完整 |
| Synthetic | mechanism / variant | 19 类机制 / 38 个 dim 变体 | 0 | 38 个变体 | 35 个变体；16 类机制双维度完整 | 3 个变体 | 3 个 regression `dim20` 变体缺完整结果 |

Dataset/task 去重补充：TabArena 原始并集 48 个，其中 2 个仅无样本、排除后有效 46 个；drop 或 onehot 任一完整 39 个，drop 和 onehot 都完整 25 个。Tabular other data 共 27 个，任一编码完整 17 个，两种编码都完整 12 个。Synthetic 19 类机制全部至少一个 dim 完整。

## TabArena

### Random 主体

| encoding | 原始总数 | 预处理后无样本 | 有效总数 | 完整 | 未完整 |
| --- | ---: | ---: | ---: | ---: | ---: |
| drop | 37 | 0 | 37 | 32 | 5 |
| onehot | 35 | 0 | 35 | 26 | 9 |

Random 主体 onehot 分任务类型：

| task_type | 总数 | 完整 | 未完整 |
| --- | ---: | ---: | ---: |
| binary | 18 | 13 | 5 |
| multiclass | 6 | 3 | 3 |
| regression | 11 | 10 | 1 |
| 合计 | 35 | 26 | 9 |

### Random + TabArena30 合并

| encoding | 原始总数 | 预处理后无样本 | 有效总数 | 完整 | 未完整 |
| --- | ---: | ---: | ---: | ---: | ---: |
| drop | 44 | 0 | 44 | 38 | 6 |
| onehot | 48 | 3 | 45 | 26 | 19 |

按任务类型细分：

| encoding | task_type | 有效总数 | 完整 | 未完整 |
| --- | --- | ---: | ---: | ---: |
| drop | binary | 28 | 23 | 5 |
| drop | multiclass | 5 | 4 | 1 |
| drop | regression | 11 | 11 | 0 |
| onehot | binary | 27 | 13 | 14 |
| onehot | multiclass | 7 | 3 | 4 |
| onehot | regression | 11 | 10 | 1 |

按 dataset/task 去重后：

| 口径 | 数量 |
| --- | ---: |
| 原始 dataset/task 并集 | 48 |
| 仅无样本、没有其他完整口径可用 | 2 |
| 排除仅无样本后的有效 dataset/task | 46 |
| drop 完整 | 38 |
| onehot 完整 | 26 |
| drop 或 onehot 任一完整 | 39 |
| drop 和 onehot 都完整 | 25 |
| 排除仅无样本后仍两种编码都不完整 | 7 |

预处理后无样本：

| task_type | dataset | 说明 |
| --- | --- | --- |
| binary | `Diabetes130US.csv` | tabarena30 onehot 无样本；random300 drop 已完整 |
| binary | `kddcup09_appetency.csv` | tabarena30 onehot 无样本，且没有其他完整口径 |
| multiclass | `MIC.csv` | tabarena30 onehot 无样本，且没有其他完整口径 |

排除仅无样本后，仍两种编码都不完整：

| task_type | dataset |
| --- | --- |
| binary | `APSFailure.csv` |
| binary | `Bioresponse.csv` |
| binary | `NATICUSdroid.csv` |
| binary | `customer_satisfaction_in_airline.csv` |
| binary | `taiwanese_bankruptcy_prediction.csv` |
| multiclass | `hiva_agnostic.csv` |
| multiclass | `splice.csv` |

## Tabular Other Data

| encoding | 原始总数 | 预处理后无样本 | 有效总数 | 完整 | 未完整 |
| --- | ---: | ---: | ---: | ---: | ---: |
| drop | 27 | 0 | 27 | 17 | 10 |
| onehot | 27 | 0 | 27 | 12 | 15 |

按任务类型细分：

| encoding | task_type | 有效总数 | 完整 | 未完整 |
| --- | --- | ---: | ---: | ---: |
| drop | binary | 10 | 8 | 2 |
| drop | multiclass | 8 | 2 | 6 |
| drop | regression | 9 | 7 | 2 |
| onehot | binary | 10 | 5 | 5 |
| onehot | multiclass | 8 | 1 | 7 |
| onehot | regression | 9 | 6 | 3 |

按 dataset/task 去重后：

| 口径 | 数量 |
| --- | ---: |
| dataset/task 总数 | 27 |
| drop 完整 | 17 |
| onehot 完整 | 12 |
| drop 或 onehot 任一完整 | 17 |
| drop 和 onehot 都完整 | 12 |
| 两种编码都不完整 | 10 |

两种编码都不完整：

| task_type | dataset |
| --- | --- |
| binary | `electricity.csv` |
| binary | `kc1.csv` |
| multiclass | `mfeat-fourier.csv` |
| multiclass | `mfeat-karhunen.csv` |
| multiclass | `optdigits.csv` |
| multiclass | `satimage.csv` |
| multiclass | `segment.csv` |
| multiclass | `vehicle.csv` |
| regression | `cpu_activity.csv` |
| regression | `naval_propulsion_plant.csv` |

## Synthetic

Synthetic 没有 drop/onehot 编码差异，统计按机制类和 `dim10/dim20` 变体两层看。

| 口径 | 总数 | 完整 | 未完整 |
| --- | ---: | ---: | ---: |
| 机制类 | 19 | 16 个 dim10+dim20 都完整；19 个至少一个 dim 完整 | 3 个机制缺 dim20 |
| dim 变体 | 38 | 35 | 3 |

按任务类型细分：

| task_type | 机制类总数 | dim 变体总数 | 完整 dim 变体 | dim10+dim20 都完整的机制类 |
| --- | ---: | ---: | ---: | ---: |
| binary | 3 | 6 | 6 | 3 |
| regression | 16 | 32 | 29 | 13 |

未完整变体：

| task_type | mechanism | missing variant |
| --- | --- | --- |
| regression | `additive_nonlinear_control` | `dim20` |
| regression | `curved_control` | `dim20` |
| regression | `noise_regime_slope` | `dim20` |

## 综合结论

1. TabArena 的 drop 覆盖最好。合并 random 主体和 TabArena30 后，drop 有效 44 个中 38 个完整；onehot 有效 45 个中 26 个完整。两者差距主要来自 TabArena30 onehot：计划面更大，但很多只有 XGB backbone 或 MGB30 未完成；另外有 3 个 onehot 预处理后无样本，需要从完整/未完整统计里剥离。
2. Tabular other data 目前是 27 个数据集，drop 完整 17 个，onehot 完整 12 个；任一编码完整仍是 17 个，说明新增完整覆盖主要由 drop 提供，onehot 暂时没有补出额外 dataset/task。
3. Synthetic 覆盖最干净：19 类机制全部至少有一个 dim 完整，35/38 个变体完整。缺口集中在 3 个 regression 机制的 `dim20`，不影响 binary synthetic 的完整性。
4. 如果论文或报告里要给“可比较 paired 结果”的总数，建议使用有效口径：TabArena 排除仅无样本后是 46 个 dataset/task，其中 39 个任一编码完整；Tabular other data 是 27 个中 17 个任一编码完整；Synthetic 是 38 个变体中 35 个完整，或 19 类机制中 16 类双维度完整。
