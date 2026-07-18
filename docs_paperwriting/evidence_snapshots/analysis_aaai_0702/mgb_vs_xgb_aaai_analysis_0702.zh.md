# MetaGB vs. XGBoost：datacollection0702 分析

## 摘要

本报告在 0629 分析所采用的 paired-fold 统计纪律下，分析
`datacollection0702` 中所有可用实验结果。证据被刻意划分为四类协议：完整
TabArena Random300/Random600、TabArena30 fixed-backbone 长尾补充实验、额外
tabular Random300 实验，以及 synthetic 机制实验。按照已经确认的 0702 范围
规则过滤后，本快照包含 1959 条 model/fold 记录、815 个 paired folds，以及
163 个 dataset-level 比较单元。总体上，MetaGB 在 dataset level 取得 80 胜、
9 平、74 负。然而，这个总数本身并不是最准确的科学叙事：synthetic 机制实验
强烈支持预设的 local-linear-in-leaves 假设，而真实 tabular 数据仍具有明显
异质性，在 125 个真实数据单元上表现为 49 胜、8 平、68 负。

因此，核心结论应是条件性的，而不是普遍性的。MetaGB 不能被表述为全面替代
XGBoost 的方法。其最强证据来自机制层面：当目标函数包含 regime-dependent
slopes、conditional feature/predictive feature separation、稀疏交互或
multiplicative structure 时，MetaGB 相比常数叶 XGBoost 表现出稳定优势。在
真实数据上，模型展示出若干 positive pockets，同时也存在明确的 negative
pockets；尤其是在额外的局部线性参数化带来计算和统计负担、却并不匹配真实
数据生成结构时，MetaGB 会输给 XGBoost。

## 范围与协议决策

0702 快照整合了多个实验阶段。`0621` 和 `0624` 目录被视为 TabArena
Random300/Random600 主体证据。`tabarena30` 目录被单独视作长尾慢任务的折中
实验：它们使用 XGB-best backbone，并只搜索 MetaGB 特有的
`pred_feature_fraction` 与 `metagb_score_agg` 网格。这类运行对补齐长尾覆盖
有价值，但不能与完整 Random300/Random600 搜索空间混为一谈。

以下用户指定的取舍规则属于本分析协议的一部分：

- `tabarena30_onehot0702` 仍明显不完整，主要解释为 partial tail evidence。
- `tabarena30_drop0702` 中剩余的 `Bioresponse.csv` 任务被排除，不计为缺失
  证据。
- other-tabular 剩余 multiclass 任务在当前阶段不视为必须补跑；已完成子集
  作为当前可用证据分析。
- Synthetic 分析仅纳入 feature 维度 10 和 20。维度 50 的产物不进入主统计。

比较单位是 paired outer fold。只有当 `xgb_cpu` 和 `current_metagb` 在同一
fold 下都具有 `ok` 状态和有限 test metric 时，该 fold 才进入 test 比较。
Dataset-level 结果通过在每个 dataset/protocol/preprocessing/task cell 内对
paired outer-fold test metrics 求平均得到。

Tie 规则沿用 0629 分析：

- AUC：绝对差不超过 `1e-4`。
- RMSE/logloss：相对差不超过 `0.1%`。

## Dataset-Level 总览

| 范围 | Dataset units | Paired folds | MetaGB | Tie | XGB | 中位 runtime ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0702 全部可用证据 | 163 | 815 | 80 | 9 | 74 | 18.29x |
| 仅真实 tabular | 125 | 625 | 49 | 8 | 68 | 18.38x |
| 仅 synthetic | 38 | 190 | 31 | 1 | 6 | 17.90x |
| TabArena 主体 | 58 | 290 | 25 | 1 | 32 | 19.90x |
| TabArena30 | 21 | 105 | 8 | 3 | 10 | 4.20x |
| Other tabular | 46 | 230 | 16 | 4 | 26 | 20.17x |

全量聚合结果接近平衡，是因为其中混合了两个不同现象：synthetic 机制证据对
MetaGB 强烈有利，而真实数据证据在 dataset-level count 上是混合且略偏不利
的。这与一种“结构性优势而非分布无关优势”的模型判断一致。

## 分组结果

| 数据族 | 协议 | 预处理 | 任务 | Dataset units | Paired folds | MetaGB/Tie/XGB | Fold MetaGB/Tie/XGB | 中位 runtime |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Other tabular | Random300 | drop | binary | 10 | 50 | 3/2/5 | 20/4/26 | 14.19x |
| Other tabular | Random300 | drop | multiclass | 4 | 20 | 1/0/3 | 8/0/12 | 71.85x |
| Other tabular | Random300 | drop | regression | 9 | 45 | 4/0/5 | 22/1/22 | 19.83x |
| Other tabular | Random300 | onehot | binary | 10 | 50 | 3/2/5 | 21/4/25 | 27.05x |
| Other tabular | Random300 | onehot | multiclass | 4 | 20 | 1/0/3 | 8/0/12 | 94.27x |
| Other tabular | Random300 | onehot | regression | 9 | 45 | 4/0/5 | 23/1/21 | 20.00x |
| Synthetic | Random300 single seed | synthetic | binary | 6 | 30 | 6/0/0 | 26/1/3 | 46.40x |
| Synthetic | Random300 single seed | synthetic | regression | 32 | 160 | 25/1/6 | 129/0/31 | 16.39x |
| TabArena30 | XGB-best + MGB30 | drop | binary | 10 | 50 | 2/2/6 | 17/6/27 | 6.37x |
| TabArena30 | XGB-best + MGB30 | onehot | binary | 8 | 40 | 3/1/4 | 20/1/19 | 2.02x |
| TabArena30 | XGB-best + MGB30 | onehot | multiclass | 3 | 15 | 3/0/0 | 12/0/3 | 4.59x |
| TabArena main | Random300 | drop | binary | 17 | 85 | 8/0/9 | 40/1/44 | 22.01x |
| TabArena main | Random300 | drop | multiclass | 4 | 20 | 2/1/1 | 13/1/6 | 26.23x |
| TabArena main | Random300 | onehot | binary | 13 | 65 | 5/0/8 | 35/1/29 | 58.98x |
| TabArena main | Random300 | onehot | multiclass | 3 | 15 | 2/0/1 | 12/0/3 | 21.48x |
| TabArena main | Random600 | drop | regression | 11 | 55 | 3/0/8 | 18/4/33 | 10.30x |
| TabArena main | Random600 | onehot | regression | 10 | 50 | 5/0/5 | 27/2/21 | 13.81x |

## Synthetic 机制证据

Synthetic 实验是检验本文归纳偏置最干净的证据。仅纳入维度 10 和 20 后，
MetaGB 在 38 个 dataset-level cells 中取得 31 胜、1 平。该模式并非在所有
synthetic 函数上均匀出现；它恰好集中在 local linear leaves 应当有帮助的
机制上，并在 negative controls 上消失或反转。

| 机制类别 | Dataset units | Paired folds | MetaGB | Tie | XGB | 中位 improvement | 中位 runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Interaction / XOR / multiplicative | 8 | 40 | 8 | 0 | 0 | 0.2068 | 23.20x |
| Linear-in-leaves | 10 | 50 | 9 | 0 | 1 | 0.0524 | 16.62x |
| Conditional-predictive separation | 6 | 30 | 6 | 0 | 0 | 0.3619 | 18.60x |
| Sparse interaction | 6 | 30 | 6 | 0 | 0 | 0.4745 | 14.54x |
| Negative controls | 6 | 30 | 0 | 1 | 5 | -0.0390 | 15.62x |
| Noise pressure | 2 | 10 | 2 | 0 | 0 | 0.1254 | 16.24x |

最强 synthetic cases 的优势幅度较大且具有系统性，而非孤立偶然。例如，
`conditional_multiplicative__dim20` 将 RMSE 从 2.1523 降至 0.3551，
`conditional_multiplicative__dim10` 将 RMSE 从 1.5504 降至 0.3748。稀疏
和条件 regime 设置也呈现同样的定性行为：`sparse_multiplicative__dim20`
从 0.5578 降至 0.2494，`categorical_regime_slope__dim20` 从 0.3008 降至
0.1585。

这支持预设机制：XGBoost 的常数叶需要通过更多划分去近似 region 内部斜率，
而 MetaGB 在条件结构定义出相关 region 之后，可以直接表达局部线性响应。
Negative-control block 对可信度很重要。它说明 synthetic generator 并非天然
偏向 MetaGB；当信号不匹配 local linear leaves 时，XGBoost 通常更好。

## TabArena 主体证据

TabArena 主体运行提供了最相关的真实数据证据。其结果是混合的：

- Binary Random300 接近平衡但并非一致有利。Drop 为 8/0/9，onehot 为
  5/0/8。
- Multiclass Random300 在当前快照中更偏正向，drop 为 2/1/1，onehot 为
  2/0/1，但完成的数据集数量较少。
- Regression Random600 在 drop 下不利，为 3/0/8；在 onehot 下平衡，为
  5/0/5。

这支持一个克制的结论：真实 TabArena 中存在 MetaGB 的 positive pockets，
但真实数据平均表不应被解释为普遍 dominance。较强正向例子包括
`credit-g.csv`、`Marketing_Campaign.csv`、
`hazelnut-spread-contaminant-detection.csv`、`QSAR_fish_toxicity.csv`、
`miami_housing.csv` 和 `physiochemical_protein.csv`。较强负向例子包括
`polish_companies_bankruptcy.csv`、`seismic-bumps.csv`、`houses.csv` 和
`airfoil_self_noise.csv`。

预处理比较也需要谨慎解释。Onehot 有时通过显式展开类别结构帮助 MetaGB，但
也可能显著提高 runtime ratio，尤其是在列数扩张的 binary 数据集上。
`credit-g.csv` 的 onehot full Random300 结果在 AUC 上有利，但中位 runtime
ratio 很大，因此它适合作为 accuracy evidence，而不适合作为 practical
tradeoff 的强例子。

## TabArena30 长尾证据

TabArena30 最适合被理解为慢速长尾任务的协议适配。它通过复用 XGB-best
backbone，并只搜索 MetaGB-specific 30-grid 来降低搜索成本。这使其更可运行，
但所得表格不能像完整 Random300/Random600 那样被解释为相同搜索空间下的
直接比较。

排除 drop run 中放弃的 `Bioresponse.csv` 残余任务后，drop binary tail 有
10 个完整 dataset units，结果为 2/2/6。这不是一个正向聚合结果，主要是
coverage strategy。若干 loss 的 AUC 差距较小，但
`taiwanese_bankruptcy_prediction.csv` 和 `APSFailure.csv` 是明确负向案例。
`jm1.csv` 与 `online_shoppers_intention.csv` 为正向但幅度有限。

Onehot tail 不完整。在 paired units 中，binary 为 3/1/4，multiclass 为
3/0/0。后者令人鼓舞，但只有 3 个 dataset units，因此应视为 suggestive
evidence。`tabarena30_onehot0702` 的 status file 显示 355 个 total tasks、
240 completed、15 failed、100 pending，因此引用该 block 时必须附带 coverage
限定。

## Other Tabular 证据

Other-tabular block 有助于在 TabArena 之外寻找 positive 和 negative pockets。
它包含 46 个 paired dataset units，结果为 16/4/26。Binary 和 regression
子集均为混合结果；multiclass 当前每种预处理只有 4 个 paired datasets，除非
论文特别需要该轴，否则没有必要继续追跑全部剩余 multiclass jobs。

`vowel.csv` 是最强真实数据正向案例：onehot 下 logloss 从 0.2251 降至
0.1271，drop 下从 0.2282 降至 0.1320。Regression 正向例子包括
`kin8nm.csv`、`Moneyball.csv` 和 `naval_propulsion_plant.csv`。Binary 正向
例子包括 `phoneme.csv`、`kc1.csv` 和 `ilpd.csv`。负向例子包括
`cpu_activity.csv`、`balance-scale.csv`、`segment.csv`、`boston.csv` 和
`energy_efficiency.csv`。

因此，other-tabular 证据强化了与 TabArena 相同的科学结论：MetaGB 是一个
conditional advantage model，而非 uniform benchmark winner。它为分析哪些
数据结构使 local linear leaves 有帮助提供了可信基础。

## 运行时间与计算权衡

运行时间仍是主要实践成本。所有 paired folds 上，MetaGB/XGB elapsed-time
ratio 的中位数为 18.29x。仅真实 tabular 数据也相近，为 18.38x。TabArena30
由于使用 fixed-backbone MGB30 协议，将中位数降至 4.20x，但高压力数据集
仍然昂贵。

压力代理 `rows * features^2` 继续解释许多高成本案例。高压力长尾数据包括
`coil2000_insurance_policies.csv` 的 62.9M、
`taiwanese_bankruptcy_prediction.csv` 的 60.3M、`NATICUSdroid.csv` 的
55.4M，以及 onehot `diamonds.csv` 的 36.5M。这些并不都是 MetaGB 的好目标：
其中多个为 loss 或 tie。对于 AAAI-27 论文，这意味着应联合报告 accuracy
与 computational budget，而不是采用 speed-neutral 的表述。

## 面向论文的解释

最稳健、最可辩护的表述是：

1. MetaGB 为 gradient-boosted trees 提供了一种归纳偏置扩展：在常数叶基础
   上加入局部 predictive-feature response。
2. Synthetic mechanisms 高特异性地验证了预设假设：收益集中在 interactions、
   regime-dependent slopes、conditional predictive separation 和 sparse
   local-linear structures。
3. 真实数据体现的是 selective transfer，而非 universal dominance。如果论文
   将 MetaGB 定位为 structure-aware extension，而非 blanket SOTA replacement，
   这一点是可以接受且更诚实的。
4. Runtime 是一条一等实验轴。完整搜索下该方法更慢；TabArena30 应被呈现为
   coverage-oriented tail protocol。

这种叙事比宣称平均全面优越更学术上干净，也更难被反驳：正向主张绑定于
可识别的数据机制，negative controls 的行为符合预期，真实数据异质性也被
显式报告而非隐藏。

## 局限性

任何论文草稿都应明确以下局限：

- Synthetic 结果使用单一 data seed 和 5 个 outer folds；它们是强机制检验，
  但不能替代更广泛的 stochastic robustness 分析。
- TabArena30 是 fixed-backbone protocol，不是完整 Random300/Random600 search。
- `tabarena30_onehot0702` 不完整，不应被过度加权。
- 部分 status files 含有过期或内部不一致的任务计数；paired-fold 解析才是
  权威统计来源。
- Runtime ratios 使用日志中的 elapsed task time，受机器、调度和慢尾任务
  影响。

## 推荐报告策略

主论文中建议报告三张表，而不是一张过载总表：

- Synthetic mechanism table，并纳入 negative controls。
- Real-data table，按 TabArena main、TabArena30 和 other tabular 分开。
- Runtime/coverage table，展示中位 runtime ratios 和 incomplete blocks。

叙事上应强调 “mechanism-aligned advantage”，而不是 “global dominance”。
0702 数据足以支持前者，但过于混合，无法支持后者。
