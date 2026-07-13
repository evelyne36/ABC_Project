# TabArena30 与 Random300 并集下的 Drop/OneHot 缺失统计

## 统计口径

本文档回答一个更严格的覆盖问题：将 `TabArena main Random300` 与
`TabArena30` 的 paired dataset-level comparison 取并集后，分别统计 drop 与
onehot 在该并集上的缺失情况。

主口径仍沿用 0702 总报告的 paired comparison 规则：只有当同一
dataset/task/encoding/protocol 下 `xgb_cpu` 与 `current_metagb` 都存在成功
paired folds 时，该数据集才计为该协议和预处理下的有效覆盖。

这里的并集定义为：

```text
Union(task) =
  datasets paired in TabArena main Random300
  OR datasets paired in TabArena30
```

然后分别统计：

```text
Drop covered(task) =
  datasets paired under drop in either Random300 or TabArena30

OneHot covered(task) =
  datasets paired under onehot in either Random300 or TabArena30
```

注意：TabArena30 是 fixed-backbone + MGB30 tail protocol；Random300 是完整
随机搜索协议。本文只统计覆盖，不把两个协议的性能结果混成同一个实验协议。

## Paired Comparison 主结果

| 任务 | 并集数据集数 | Drop 覆盖 | Drop 缺失 | OneHot 覆盖 | OneHot 缺失 |
| --- | ---: | ---: | ---: | ---: | ---: |
| binary | 27 | 27 | 0 | 13 | 14 |
| multiclass | 5 | 4 | 1 | 4 | 1 |

在 paired comparison 口径下：

- Binary：drop 在 TabArena30 + Random300 并集上覆盖完整；onehot 缺失 14 个。
- Multiclass：drop 与 onehot 都缺 1 个，但缺失的数据集不同。
- 当前并集内没有 regression 的 TabArena30 结果，因此 regression 不参与
  TabArena30 + Random300 并集统计。

## Binary：TabArena30 + Random300 并集

Binary 并集共有 27 个 dataset-level paired comparison 单元：

```text
APSFailure.csv
Amazon_employee_access.csv
Bank_Customer_Churn.csv
Diabetes130US.csv
E-CommereShippingData.csv
Fitness_Club.csv
GiveMeSomeCredit.csv
HR_Analytics_Job_Change_of_Data_Scientists.csv
Is-this-a-good-customer.csv
Marketing_Campaign.csv
NATICUSdroid.csv
bank-marketing.csv
blood-transfusion-service-center.csv
churn.csv
coil2000_insurance_policies.csv
credit-g.csv
credit_card_clients_default.csv
customer_satisfaction_in_airline.csv
diabetes.csv
hazelnut-spread-contaminant-detection.csv
heloc.csv
jm1.csv
online_shoppers_intention.csv
polish_companies_bankruptcy.csv
qsar-biodeg.csv
seismic-bumps.csv
taiwanese_bankruptcy_prediction.csv
```

### Drop 覆盖

Drop 覆盖 27/27，缺失 0 个。

| 来源 | Drop paired datasets |
| --- | ---: |
| TabArena main Random300 drop | 17 |
| TabArena30 drop | 10 |
| 两者重叠 | 0 |
| 两者并集 | 27 |

这意味着，在 binary 任务上，如果把 Random300 drop 与 TabArena30 drop 视为
互补覆盖来源，drop 已覆盖当前 paired 并集中的全部 27 个数据集。

### OneHot 覆盖与缺失

Onehot 覆盖 13/27，缺失 14 个。

| 来源 | OneHot paired datasets |
| --- | ---: |
| TabArena main Random300 onehot | 13 |
| TabArena30 onehot | 8 |
| 两者重叠 | 8 |
| 两者并集 | 13 |

Onehot 相较 binary 并集缺失的数据集为：

| OneHot 缺失数据集 |
| --- |
| `APSFailure.csv` |
| `Diabetes130US.csv` |
| `HR_Analytics_Job_Change_of_Data_Scientists.csv` |
| `Marketing_Campaign.csv` |
| `NATICUSdroid.csv` |
| `bank-marketing.csv` |
| `coil2000_insurance_policies.csv` |
| `credit_card_clients_default.csv` |
| `customer_satisfaction_in_airline.csv` |
| `heloc.csv` |
| `jm1.csv` |
| `online_shoppers_intention.csv` |
| `polish_companies_bankruptcy.csv` |
| `taiwanese_bankruptcy_prediction.csv` |

解释上需要注意：TabArena30 onehot 的 8 个 paired binary datasets 完全落在
Random300 onehot 已有的 13 个 paired datasets 内，因此它没有扩展 onehot 的
binary 并集覆盖。相反，TabArena30 drop 补上了一批 Random300 drop 中未形成
paired 的长尾数据，使 drop 的 binary 并集覆盖达到 27/27。

## Multiclass：TabArena30 + Random300 并集

Multiclass 并集共有 5 个 dataset-level paired comparison 单元：

```text
SDSS17.csv
anneal.csv
maternal_health_risk.csv
students_dropout_and_academic_success.csv
website_phishing.csv
```

### Drop 覆盖与缺失

Drop 覆盖 4/5，缺失 1 个。

| 来源 | Drop paired datasets |
| --- | ---: |
| TabArena main Random300 drop | 4 |
| TabArena30 drop | 0 |
| 两者重叠 | 0 |
| 两者并集 | 4 |

Drop 缺失的数据集为：

| Drop 缺失数据集 |
| --- |
| `website_phishing.csv` |

### OneHot 覆盖与缺失

Onehot 覆盖 4/5，缺失 1 个。

| 来源 | OneHot paired datasets |
| --- | ---: |
| TabArena main Random300 onehot | 3 |
| TabArena30 onehot | 3 |
| 两者重叠 | 2 |
| 两者并集 | 4 |

Onehot 缺失的数据集为：

| OneHot 缺失数据集 |
| --- |
| `students_dropout_and_academic_success.csv` |

Multiclass 的结论与 binary 不同：drop 与 onehot 的覆盖数量相同，都是 4/5，
但缺失对象不同。Drop 缺 `website_phishing.csv`，onehot 缺
`students_dropout_and_academic_success.csv`。

## Raw Coverage 参考口径

为了避免误解，下面给出 raw record coverage 参考。该口径只看是否存在解析到
的记录，不要求形成成功 paired comparison。

| 任务 | Raw 并集数据集数 | Drop raw 覆盖 | Drop raw 缺失 | OneHot raw 覆盖 | OneHot raw 缺失 |
| --- | ---: | ---: | ---: | ---: | ---: |
| binary | 28 | 28 | 0 | 27 | 1 |
| multiclass | 7 | 5 | 2 | 7 | 0 |

Raw 口径与 paired 口径差异较大，说明主要缺口并不是“完全没有文件”，而是
“没有形成 XGB 与 MetaGB 同 fold 的成功 paired comparison”。

Raw 口径下：

- Binary onehot 只完全缺少 `Diabetes130US.csv`，但 paired 口径下缺 14 个。
- Multiclass onehot raw 覆盖为 7/7，但 paired 口径只覆盖 4/5。
- Multiclass drop raw 缺少 `splice.csv` 和 `website_phishing.csv`；其中
  `splice.csv` 没有进入 paired 并集主口径。

## 可引用结论

如果以 TabArena30 与 Random300 的 paired 并集作为目标覆盖集合：

> Binary 任务上，drop 覆盖 27/27，onehot 覆盖 13/27，onehot 缺失 14 个
> paired datasets。Multiclass 任务上，drop 和 onehot 都覆盖 4/5，但 drop
> 缺 `website_phishing.csv`，onehot 缺
> `students_dropout_and_academic_success.csv`。

因此，从“TabArena30 + Random300 并集补表”的角度看，drop 覆盖明显更完整；
onehot 的主要问题不是 raw 文件完全缺失，而是大量数据集没有形成可纳入胜负
统计的 paired comparison。
