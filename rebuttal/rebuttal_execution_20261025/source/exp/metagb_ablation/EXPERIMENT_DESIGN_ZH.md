# MetaGB 多层级消融实验设计

## 0. 文档定位

本文档是 MetaGB 消融实验的科学协议源。代码、结果表、热力图和论文文字均应
与本文件保持一致。任何改变数据集、指标、重复次数、主对比或超参数选择规则的
操作，都必须先修改本文件和协议版本，再创建新的 `run_id`。

该实验不是 pipeline 模块的 `w/o component` 消融，而是对 MetaGB 归纳偏置的
机制干预：

```text
公共 XGBoost backbone
        ↓
MetaGB updater / predictive-aware split / predictive leaf
        ↓
pred_fraction 剂量响应与 score aggregation 稳健性
```

## 1. 中心科学问题与有界结论

### 1.1 中心问题

在相同树结构预算、公共超参数、数据划分和随机种子下，MetaGB 的性能变化究竟
来自：

1. MetaGB updater 本身；
2. predictive features 对 split scoring 的影响；
3. predictive leaf 的表达能力；
4. predictive feature 覆盖率 `pred_fraction`；
5. score aggregation 策略。

### 1.2 可以支持的结论

若结果成立，可支持：

> 在固定的 XGBoost 优选 backbone 下，predictive-aware split 与 predictive
> leaf 对机制匹配任务产生可分解的增益，并且该增益随 predictive feature
> 覆盖率呈现可审计的剂量响应。

### 1.3 不能越界的结论

本实验不能单独证明：

- MetaGB 在所有表格数据上普遍优于 XGBoost；
- `p=1` 是所有数据集的最优部署参数；
- 四个变体在各自完全调优后仍保持相同排序；
- 少数 diagnostic datasets 可以代表全部 TabArena。

## 2. Claim–Evidence Matrix

| Claim | Reviewer question | Evidence | 数据 | 主指标 | 失败时的解释 |
| --- | --- | --- | --- | --- | --- |
| updater 与 XGB 不等价 | `p=0` 是否只是 XGB 改名 | XGB vs Bias | real + syn | RMSE/logloss | updater 本身没有稳定增益或存在代价 |
| predictive-aware split 有作用 | 增益是否只来自叶表达 | SplitOnly(p) vs Bias | aligned syn + real | paired signed improvement | split scoring 不是主要来源 |
| predictive leaf 有作用 | Full 是否优于相同候选集的 SplitOnly | Full(p) vs SplitOnly(p) | aligned syn + controls + real | paired signed improvement | 增益主要来自 split 或机制不匹配 |
| `pred_fraction` 是剂量变量 | 结果是否仅由单个手选 `p` 驱动 | `p={0,0.2,0.4,0.6,0.8,1.0}` 双曲线 | real + syn | dose-response | 机制脆弱、非单调或受维度影响 |
| greedy 选择不是偶然 | aggregation 是否决定结果 | greedy/mean/median at `p=1` | real + syn | paired effect vs greedy | 结论限定到具体 aggregation |
| 机制具有边界 | 是否只展示正例 | negative controls | pure-step/curved | paired signed improvement | 若 controls 也大幅提升，机制解释需重审 |

所有结果单元初始状态为 `TBD`；只有真实运行产物可以填入论文表格。

## 3. Level 0：公平性与公共 Backbone

### 3.1 选择规则

每个 outer fold 内：

1. 仅在 outer-train 上进行 XGBoost inner-CV；
2. 根据预注册 primary metric 选择一个 XGBoost backbone；
3. 将该 backbone 原样映射到所有 MetaGB 变体；
4. 所有冻结配置分别在相同 outer-train 上拟合，并在相同 outer-test 上评估；
5. 禁止使用 outer-test 选择 `p`、score aggregation 或 backbone。

真实数据从原始表开始划分，任何依赖数据分布或类别词表的 preprocessing 都必须
局部拟合：

```text
raw rows
├── outer-train: fit outer encoder
│   └── outer-test: transform only
└── each inner-train: fit a fresh inner encoder
    └── inner-validation: transform only
```

`ordinal` 的未知类别编码为 `-1`；`onehot` 忽略 validation/test 独有类别且不
扩充训练特征空间。禁止在完整数据上先 `factorize/get_dummies` 再切 fold。

### 3.2 公共参数

必须一致：

```text
learning_rate
max_depth
n_estimators
min_child_weight
subsample
colsample_bytree
max_bin
L2 regularization
objective
preprocessing
train/test split
model seed
```

其中：

```text
XGB.reg_lambda == MetaGB.metagb_lambda_l2
```

为减少 feature-sampling 混杂，消融专用搜索空间默认固定：

```text
subsample = 1
colsample_bytree = 1
```

### 3.3 解释边界

固定 XGB backbone 对 MetaGB 可能偏保守，因此主结论必须写成
“fixed-backbone mechanism effect”，不能写成“每种方法的最佳可达性能”。

## 4. Level 1：四类主消融

| Variant | `p` | `score_agg` | `leaf_only_use_bias` | 机制 |
| --- | ---: | ---: | --- | --- |
| `xgb` | — | — | — | 普通常数叶 XGBoost |
| `bias_p00_sa0` | 0 | 0 | true | MetaGB updater + bias-only |
| `split_p10_sa0` | 1 | 0 | true | predictive-aware split，最终叶只用 bias |
| `full_p10_sa0` | 1 | 0 | false | predictive-aware split + predictive leaf |

`p=1` 是主表中的强机制开关，不允许把测试集最优 `p` 回填到主表。

主表除四个原始分数外，还必须给出配对效应：

```text
SplitOnly − Bias
Full − SplitOnly
Full − Bias
Full − XGB
```

## 5. Level 2：`pred_fraction` 敏感性

固定 `score_agg=0`，同时运行：

```text
Bias anchor: p = 0
SplitOnly:  p ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
Full:       p ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
```

双曲线是必要的，因为只观察 Full 会把 split scoring 与 predictive leaf 两种机制
混在一起。

对于“数值越小越好”的 loss，定义正值代表改进：

```text
relative_improvement(A vs B) = (loss_B - loss_A) / |loss_B|
```

三个主效应：

```text
Δsplit(p) = SplitOnly(p) − Bias
Δleaf(p)  = Full(p) − SplitOnly(p)
Δtotal(p) = Full(p) − Bias
```

分类主指标优先使用 logloss；AUC 作为辅助指标并使用绝对百分点差，不与
RMSE/logloss 共用色标。

### 5.1 低维解释风险

每个预测特征独立以概率 `p` 进入候选集。若共有 `d` 个真实特征：

```text
E[candidates] = p × d
P(no predictive candidate) = (1 - p)^d
```

因此低维下 `p=0.2` 仍可能产生大量 bias-only trees。结果必须至少保存
`expected_pred_candidates` 与 `probability_no_pred_candidate`。实际候选数的
逐树 telemetry 在底层接口暴露后作为增强产物加入。

## 6. Level 3：Score Aggregation 消融

主设置：

```text
leaf mode = Full
p = 1
score_agg ∈ {
  0: greedy/max,
  1: mean,
  2: median
}
```

固定 `p=1` 是为了隔离 aggregation，避免候选数量随机变化。主要报告：

```text
mean − greedy
median − greedy
```

SplitOnly 同样运行 mean/median，形成正式的 `2×3` 小因子设计：

```text
SplitOnly × {greedy, mean, median}
Full      × {greedy, mean, median}
```

这两个额外 SplitOnly 单元使 aggregation 的解释不再与 predictive leaf
绑定，是默认 16 配置的一部分。

## 7. Level 4：数据层级

### 7.1 真实数据 diagnostic panel

按任务、规模、维度和特征类型预先冻结：

| 任务 | 数据集 |
| --- | --- |
| Regression | `airfoil_self_noise`, `miami_housing`, `physiochemical_protein` |
| Binary | `diabetes`, `credit-g`, `seismic-bumps` |
| Multiclass | `maternal_health_risk`, `anneal` |

这些数据已有历史结果可见，因此称为 predefined diagnostic panel，不宣称盲选。
主 preprocessing 与论文主实验保持一致，并按 inner/outer fold 局部拟合；
`credit-g` 和 `anneal` 的 one-hot 结果可作为单独 robustness block，不与主
preprocessing 合并统计。

### 7.2 合成数据

机制匹配：

```text
regime_slope
feature_switch
categorical_regime_slope
sparse_multiplicative
regime_logit
```

负对照：

```text
pure_step_control
curved_control
```

正式设置：

```text
dimensions = {10, 20}
independent data seeds = {1001, 1002, 1003}
XGB backbone development seed = 9001
```

Synthetic 不在每个 evaluation seed 上重新选择公共参数。每个
`scenario × dimension` 仅在独立 development seed 9001 上选择一次 XGB
backbone，再冻结到 3 个 evaluation seeds。这样既避免 evaluation seed
反向参与选参，也显著减少重复 HPO。

每个 seed 内所有变体共享数据、split 与 model seed。先跨 outer folds 聚合到
seed-level，再对 seed-level 配对差做 bootstrap。

## 8. Level 5：不确定性、重复与统计

### 8.1 Synthetic

- 当前资源约束协议使用 3 个独立 data seeds；
- 同一 seed 的全部变体严格配对；
- 以 seed-level paired difference 做 10,000 次 bootstrap；
- outer folds 只用于降低单 seed 估计方差，不视为独立重复。

三个 seed 适合快速获得完整机制曲线和发现大效应，但不足以单独支撑稳定的
尾部概率或精细显著性结论。论文中应展示三个 paired seed-level 点，并将
bootstrap CI 标为探索性；若效应接近零、跨 seed 方向不一致或审稿阶段需要更
强统计证据，再将同一冻结协议扩展到 10–20 seeds。

### 8.2 Real

- 最低成本 smoke：固定 2-fold OOF predictions，只用于验证工程链路，不给正式
  CI；
- 正式版本：5 次 repeated 5-fold CV，每个 repeat 先由完整 OOF predictions
  重算 metric，再对五个 repeat-level 配对差报告 bootstrap CI 和全部散点；
- 不把 5 个 outer folds 当作 5 个 iid 样本做普通 t-test；
- 图表优先展示配对差 CI，而不是四个模型各自的独立误差棒。

逐样本同步、分类分层的 hierarchical bootstrap 可作为增强统计，但必须与
repeat-level 主统计分开命名，不能把样本数当成独立数据划分重复数。

### 8.3 多重比较

- 不对十个 `p` 单元格逐个打未校正显著性星号；
- 主确认性对比预注册为 `p=1` 的三种机制效应；
- sensitivity 主要解释整体曲线、平台区和失效区；
- 若输出逐格检验，必须单独提供校正方法与原始 p-value。

## 9. Level 6：呈现层级

### 9.1 主表

只含四行：

```text
XGB | Bias p0 | SplitOnly p1 | Full p1
```

单元格为真实指标均值与 95% CI；附带 paired effect 表。

### 9.2 Pred sensitivity

三联 heatmap：

```text
Panel A: Δsplit(p)
Panel B: Δleaf(p)
Panel C: Δtotal(p)
```

- 行：dataset/scenario；
- 列：`p={0,0.2,0.4,0.6,0.8,1.0}`；
- 颜色：signed relative improvement，正值更好；
- `p=0` 使用 Bias 作为锚点；
- 另选 2–3 个代表 case 画 dose-response 折线和 95% CI；
- XGBoost 只作为水平参考线。

### 9.3 Score aggregation

使用三列表或 forest plot：

```text
greedy(0) | mean(1) | median(2)
```

同时展示 `mean−greedy`、`median−greedy` 的 paired CI。

## 10. 执行优先级

| Priority | 实验 | 目的 | Stop condition |
| --- | --- | --- | --- |
| P0 | plan-only + protocol tests | 验证配置数、ID、数据和 schema | 16 个配置且审计通过 |
| P0 | 1 aligned syn smoke | 验证 SplitOnly/Full/p/scoreagg 可训练 | 每任务 16 行、预测完整 |
| P0 | 1 negative-control smoke | 验证控制路径 | 无配置/统计错误 |
| P1 | 7 syn × 2 dimensions × 3 seeds | 机制与边界检查 | 210 tasks、3360 fold rows 完整 |
| Gate | Synthetic 完整分析与结论批准 | 防止未解释结果直接扩展到 Real | audit 通过、结论无占位符且 hash 批准 |
| P2 | 8 real diagnostic datasets | 外部效度 | 只在 Gate 通过后启动，OOF predictions 完整 |
| P2 | real repeated-CV | split 稳定性 | 正式协议 5 repeats |

执行顺序是协议的一部分：

```text
Formal Synthetic
  → completeness audit
  → 主消融/pred/scoreagg/负对照联合分析
  → 完整 Synthetic 结论与边界
  → analysis approval
  → Real datasets（由小到大）
```

禁止自动串联 Synthetic 与 Real。

## 11. Reviewer 风险登记

| 风险 | 防护 |
| --- | --- |
| 根据 MetaGB 结果选择数据集 | 冻结 diagnostic panel 并声明非盲选 |
| 根据 outer-test 选择 `p` | runner 对每个冻结配置独立 outer-test，不做 winner selection |
| XGB 与 MetaGB 公共参数不一致 | backbone manifest + 参数逐字段断言 |
| fold 被当成 iid 重复 | seed/repeat-level 配对统计 + OOF predictions |
| `p` 与其他 sampling 混杂 | 专用搜索空间固定 subsample/colsample 为 1 |
| `p=0` 边界泄漏 | 底层显式处理 `fraction<=0` |
| 低维 `p` 横轴含义不清 | 保存期望候选数与零候选概率 |
| smoke 被误写成正式结果 | protocol 标记 `confirmatory=false` |

## 12. No-Fabrication 状态

本文档没有生成任何实验数值。所有结果表必须由本工程真实运行产物填充；
缺失结果必须保持 `TBD` 或显式标记为未运行。
