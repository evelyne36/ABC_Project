# OneHot 相较 Drop 的数据集缺口说明

## 统计口径

本文档只比较 `datacollection0702` 中 onehot 与 drop 两种预处理在同一
family/protocol/task 下的覆盖差异。主口径采用此前 0702 总报告一致的
dataset-level paired comparison：一个数据集只有在 onehot/drop 对应协议下
同时具备 `xgb_cpu` 与 `current_metagb` 的成功 paired folds 时，才视为该预处理
下的可比较数据集。

因此，本文中的“onehot 缺失”主要表示：

```text
drop 中已有 paired dataset-level comparison，
但 onehot 中尚未形成 paired dataset-level comparison。
```

这不一定等价于 onehot 完全没有任何日志或 status 记录。若 onehot 只有 XGB
完成、MetaGB 未完成，或者有 incomplete 记录但没有成功 paired folds，也会在
主口径下被视为缺失。

## 总览

| 数据族 | 协议 | 任务 | Drop paired datasets | OneHot paired datasets | OneHot 相较 Drop 缺失 | OneHot-only paired |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| other_tabular_0702 | random300 | binary | 10 | 10 | 0 | 0 |
| other_tabular_0702 | random300 | multiclass | 4 | 4 | 0 | 0 |
| other_tabular_0702 | random300 | regression | 9 | 9 | 0 | 0 |
| tabarena30_0702 | xgb_best_mgb30 | binary | 10 | 8 | 10 | 8 |
| tabarena_main_0702 | random300 | binary | 17 | 13 | 4 | 0 |
| tabarena_main_0702 | random300 | multiclass | 4 | 3 | 2 | 1 |
| tabarena_main_0702 | random600 | regression | 11 | 10 | 1 | 0 |

结论上，other-tabular 的 onehot 与 drop paired 覆盖完全一致。主要缺口集中在
TabArena main 和 TabArena30。尤其是 TabArena30，drop 与 onehot 当前形成
paired comparison 的数据集集合并不相同，因此不能简单理解为 onehot 在同一
集合上少跑了 2 个数据集；更准确地说，是 two preprocessing protocols 当前
完成的 paired subsets 不重合。

## TabArena Main：Random300 Binary

Drop 中有 17 个 paired binary datasets，onehot 中有 13 个。Onehot 相较 drop
缺少以下 4 个 paired comparison：

| 缺失数据集 | OneHot raw coverage 状态 | 解释 |
| --- | --- | --- |
| `Diabetes130US.csv` | onehot 无 coverage row | onehot 当前没有形成可解析的 paired 证据 |
| `HR_Analytics_Job_Change_of_Data_Scientists.csv` | XGB 5 folds ok，MetaGB 0 folds ok；status 为 `incomplete,ok` | 有 onehot 记录，但 MetaGB 未形成成功 paired folds |
| `Marketing_Campaign.csv` | XGB 5 folds ok，MetaGB 0 folds ok；status 为 `incomplete,ok` | 有 onehot 记录，但 MetaGB 未形成成功 paired folds |
| `polish_companies_bankruptcy.csv` | XGB 5 folds ok，MetaGB 0 folds ok；status 为 `incomplete,ok` | 有 onehot 记录，但 MetaGB 未形成成功 paired folds |

该组没有 onehot-only paired datasets。也就是说，TabArena main binary 的
onehot paired 集合是 drop paired 集合的子集。

## TabArena Main：Random300 Multiclass

Drop 中有 4 个 paired multiclass datasets，onehot 中有 3 个。Onehot 相较
drop 缺少以下 2 个 paired comparison：

| 缺失数据集 | OneHot raw coverage 状态 | 解释 |
| --- | --- | --- |
| `anneal.csv` | XGB 5 folds ok，MetaGB 0 folds ok；status 为 `incomplete,ok` | 有 onehot 记录，但未形成 paired comparison |
| `students_dropout_and_academic_success.csv` | onehot 无 coverage row | onehot 当前没有形成可解析的 paired 证据 |

同时，该组存在 1 个 onehot-only paired dataset：

| OneHot-only 数据集 | 说明 |
| --- | --- |
| `website_phishing.csv` | onehot 有 paired comparison，但 drop paired 集合中没有该 dataset |

因此，multiclass onehot 并非 drop 的严格子集，而是少了 `anneal.csv` 和
`students_dropout_and_academic_success.csv`，同时多了 `website_phishing.csv`。

## TabArena Main：Random600 Regression

Drop 中有 11 个 paired regression datasets，onehot 中有 10 个。Onehot 相较
drop 缺少以下 1 个 paired comparison：

| 缺失数据集 | OneHot raw coverage 状态 | 解释 |
| --- | --- | --- |
| `Food_Delivery_Time.csv` | XGB 5 folds ok，MetaGB 0 folds ok；status 为 `incomplete,ok` | 有 onehot 记录，但 MetaGB 未形成成功 paired folds |

该组没有 onehot-only paired datasets。也就是说，TabArena main regression 的
onehot paired 集合是 drop paired 集合的子集。

## TabArena30：XGB-best + MGB30 Binary

TabArena30 的差异需要单独解释。Drop 中有 10 个 paired binary datasets，
onehot 中有 8 个 paired binary datasets，但两者当前的 paired 集合完全不同。

Onehot 相较 drop 缺少以下 10 个 drop-paired datasets：

| Drop-paired 但 OneHot 未形成 paired 的数据集 |
| --- |
| `APSFailure.csv` |
| `NATICUSdroid.csv` |
| `bank-marketing.csv` |
| `coil2000_insurance_policies.csv` |
| `credit_card_clients_default.csv` |
| `customer_satisfaction_in_airline.csv` |
| `heloc.csv` |
| `jm1.csv` |
| `online_shoppers_intention.csv` |
| `taiwanese_bankruptcy_prediction.csv` |

同时，onehot 有以下 8 个 onehot-only paired datasets：

| OneHot-only paired 数据集 |
| --- |
| `Fitness_Club.csv` |
| `Is-this-a-good-customer.csv` |
| `blood-transfusion-service-center.csv` |
| `credit-g.csv` |
| `diabetes.csv` |
| `hazelnut-spread-contaminant-detection.csv` |
| `qsar-biodeg.csv` |
| `seismic-bumps.csv` |

从 raw record set 看，`tabarena30_onehot0702` 实际包含比 drop 更多的 binary
数据集记录；drop 中那 10 个数据集在 onehot raw record set 中并非全部缺失。
问题在于 onehot 当前大量记录未形成 `xgb_cpu` 与 `current_metagb` 的成功 paired
comparison。因此，TabArena30 的 onehot/drop 差异更准确地应表述为：

> TabArena30 onehot 当前 paired 子集与 drop paired 子集不重合；这主要是
> paired completion 缺口，而不是简单的数据集列表设计缺口。

## Other Tabular：Random300

Other-tabular 三个任务类型在 paired 口径下均没有 onehot 相较 drop 的数据集
缺口：

| 任务 | Drop paired datasets | OneHot paired datasets | 缺失说明 |
| --- | ---: | ---: | --- |
| binary | 10 | 10 | 无缺失 |
| multiclass | 4 | 4 | 无缺失 |
| regression | 9 | 9 | 无缺失 |

需要注意的是，other-tabular multiclass 在 raw coverage 层面仍有若干 XGB-only
数据集，如 `mfeat-fourier.csv`、`mfeat-karhunen.csv` 和 `satimage.csv`。这些
没有进入 paired comparison，但它们在 drop 与 onehot 两边的 raw record 集合
是一致的，因此不构成 onehot 相较 drop 的额外缺口。

## 可引用结论

如果在论文或汇报中需要一句话总结，可以写为：

> 在 0702 当前 paired-comparison 口径下，other-tabular 的 onehot/drop 覆盖
> 一致；TabArena main 中 onehot 相较 drop 缺少 4 个 binary、2 个 multiclass
> 和 1 个 regression paired datasets；TabArena30 的 onehot/drop paired 子集
> 当前基本不重合，应解释为长尾 fixed-backbone 协议下的 paired-completion
> 差异，而非完整同集合对照。

若需要更严格的主表对照，建议优先使用 TabArena main 中 onehot 是 drop 子集的
部分；TabArena30 则应单独作为 partial tail evidence 标注。
