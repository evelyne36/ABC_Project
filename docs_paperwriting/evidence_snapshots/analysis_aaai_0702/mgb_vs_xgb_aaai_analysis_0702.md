# MetaGB vs. XGBoost: datacollection0702 Analysis

## Abstract

This report analyzes the complete usable `datacollection0702` snapshot under the
same paired-fold discipline used in the 0629 analysis. The evidence is deliberately
partitioned into four protocols: full TabArena Random300/Random600, TabArena30
fixed-backbone tail runs, other tabular Random300 runs, and synthetic mechanism
runs. After applying the agreed 0702 scope rules, the snapshot contains 1959
model/fold records, 815 paired folds, and 163 dataset-level comparison units.
Overall, MetaGB records 80 wins, 9 ties, and 74 losses at dataset level. This
aggregate, however, is not the right scientific story by itself: synthetic
mechanism tests strongly support the intended local-linear-in-leaves hypothesis,
whereas real tabular data remain heterogeneous, with 49 wins, 8 ties, and 68
losses across 125 real-data units.

The central conclusion is therefore conditional rather than universal. MetaGB is
not an across-the-board replacement for XGBoost. Its strongest evidence is
mechanistic: when the target function contains regime-dependent slopes, conditional
feature/predictive feature separation, sparse interactions, or multiplicative
structure, MetaGB consistently outperforms constant-leaf XGBoost. On real datasets,
the model exposes positive pockets but also clear negative pockets, especially
where the additional local-linear parameterization appears to add computational
and statistical burden without matching the data-generating structure.

## Scope and Protocol Decisions

The 0702 snapshot consolidates several experimental generations. The `0621` and
`0624` directories are treated as the main TabArena Random300/Random600 evidence.
The `tabarena30` directories are treated separately as tail-compromise experiments:
they use an XGB-best backbone and search only the MetaGB-specific grid over
`pred_feature_fraction` and `metagb_score_agg`. These runs are useful for long-tail
coverage but are not interchangeable with full Random300/Random600 search.

The following user-specified decisions are part of the analysis protocol:

- `tabarena30_onehot0702` remains substantially incomplete and is interpreted
  mainly as partial tail evidence.
- The remaining `Bioresponse.csv` jobs in `tabarena30_drop0702` are excluded and
  not counted as missing evidence.
- The remaining other-tabular multiclass jobs are not treated as necessary for
  this stage; the completed subset is analyzed as the available evidence.
- Synthetic analysis includes dimensions 10 and 20 only. Dimension 50 artifacts
  are excluded from the main statistics.

The comparison unit is a paired outer fold. A fold enters the test comparison only
when both `xgb_cpu` and `current_metagb` have status `ok` and a finite test metric.
Dataset-level results are computed by averaging paired outer-fold test metrics
within each dataset/protocol/preprocessing/task cell.

Tie thresholds follow the 0629 analysis:

- AUC: absolute difference no larger than `1e-4`.
- RMSE/logloss: relative difference no larger than `0.1%`.

## Dataset-Level Summary

| Scope | Dataset units | Paired folds | MetaGB | Tie | XGB | Median runtime ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All usable 0702 evidence | 163 | 815 | 80 | 9 | 74 | 18.29x |
| Real tabular only | 125 | 625 | 49 | 8 | 68 | 18.38x |
| Synthetic only | 38 | 190 | 31 | 1 | 6 | 17.90x |
| TabArena main | 58 | 290 | 25 | 1 | 32 | 19.90x |
| TabArena30 | 21 | 105 | 8 | 3 | 10 | 4.20x |
| Other tabular | 46 | 230 | 16 | 4 | 26 | 20.17x |

The full aggregate is nearly balanced because two different phenomena are mixed:
synthetic mechanism evidence is strongly favorable to MetaGB, while real-data
evidence is mixed and slightly unfavorable in dataset-level count. This is
consistent with a model whose advantage is structural rather than distribution-free.

## Group-Level Results

| Family | Protocol | Encoding | Task | Dataset units | Paired folds | MetaGB/Tie/XGB | Fold MetaGB/Tie/XGB | Median runtime |
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

## Synthetic Mechanism Evidence

Synthetic experiments are the cleanest test of the proposed inductive bias. With
dimensions 10 and 20 only, MetaGB wins 31 of 38 dataset-level cells and ties 1.
The pattern is not uniform across all synthetic functions: it concentrates exactly
where local linear leaves should help and disappears or reverses on negative
controls.

| Mechanism class | Dataset units | Paired folds | MetaGB | Tie | XGB | Median improvement | Median runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Interaction / XOR / multiplicative | 8 | 40 | 8 | 0 | 0 | 0.2068 | 23.20x |
| Linear-in-leaves | 10 | 50 | 9 | 0 | 1 | 0.0524 | 16.62x |
| Conditional-predictive separation | 6 | 30 | 6 | 0 | 0 | 0.3619 | 18.60x |
| Sparse interaction | 6 | 30 | 6 | 0 | 0 | 0.4745 | 14.54x |
| Negative controls | 6 | 30 | 0 | 1 | 5 | -0.0390 | 15.62x |
| Noise pressure | 2 | 10 | 2 | 0 | 0 | 0.1254 | 16.24x |

The strongest synthetic cases are large and systematic rather than isolated.
For example, `conditional_multiplicative__dim20` improves RMSE from 2.1523 to
0.3551, and `conditional_multiplicative__dim10` improves RMSE from 1.5504 to
0.3748. Sparse and conditional-regime settings show the same qualitative behavior:
`sparse_multiplicative__dim20` improves from 0.5578 to 0.2494, while
`categorical_regime_slope__dim20` improves from 0.3008 to 0.1585.

This supports the intended mechanism: XGBoost's constant leaves must approximate
within-region slopes by additional partitions, while MetaGB can represent a
locally linear response once the conditional structure has defined the relevant
region. The negative-control block is important for credibility. It shows the
synthetic generator does not simply favor MetaGB by construction; when the signal
does not match local linear leaves, XGBoost is usually better.

## TabArena Main Evidence

The TabArena main runs provide the most relevant real-data evidence. They are
mixed:

- Binary Random300 is close to balanced but not uniformly favorable. Drop gives
  8/0/9 and onehot gives 5/0/8 at dataset level.
- Multiclass Random300 is more favorable in this snapshot, with drop 2/1/1 and
  onehot 2/0/1, but the number of completed dataset units is small.
- Regression Random600 is unfavorable under drop, 3/0/8, and balanced under
  onehot, 5/0/5.

This supports a measured claim: real TabArena contains positive pockets for
MetaGB, but the average real-data table should not be framed as a universal
dominance result. Strong positive examples include `credit-g.csv`, `Marketing_Campaign.csv`,
`hazelnut-spread-contaminant-detection.csv`, `QSAR_fish_toxicity.csv`,
`miami_housing.csv`, and `physiochemical_protein.csv`. Strong negative examples
include `polish_companies_bankruptcy.csv`, `seismic-bumps.csv`, `houses.csv`, and
`airfoil_self_noise.csv`.

The preprocessing comparison is also nuanced. Onehot sometimes helps by exposing
categorical structure to MetaGB, but it can also greatly increase the runtime
ratio, especially on binary datasets with many expanded columns. The `credit-g.csv`
onehot full Random300 result is favorable in AUC but has a very large median
runtime ratio, so it is scientifically useful as accuracy evidence but weak as a
practical tradeoff example.

## TabArena30 Tail Evidence

TabArena30 is best understood as a protocol adaptation for slow long-tail tasks.
It reduces search cost by reusing the XGB-best backbone and searching only the
MetaGB-specific 30-grid. This makes it more tractable, but the resulting table is
not directly comparable to full Random300/Random600 as if the hyperparameter
search spaces were identical.

After excluding the abandoned `Bioresponse.csv` residual in the drop run, the
drop binary tail has 10 complete dataset units and gives 2/2/6. This is not a
positive aggregate result; it is mainly a coverage strategy. Several losses are
small in AUC, but `taiwanese_bankruptcy_prediction.csv` and `APSFailure.csv` are
clear negative cases. `jm1.csv` and `online_shoppers_intention.csv` are positive
but modest.

The onehot tail is incomplete. Among paired units, binary gives 3/1/4 and
multiclass gives 3/0/0. The latter is encouraging but based on only three dataset
units; it should be treated as suggestive. The status file reports 355 total tasks,
240 completed, 15 failed, and 100 still pending for `tabarena30_onehot0702`, so
coverage language is necessary whenever this block is cited.

## Other Tabular Evidence

The other-tabular block is useful for detecting positive and negative pockets
outside TabArena. It contains 46 paired dataset units and gives 16/4/26. The
binary and regression subsets are mixed; multiclass currently contains only four
paired datasets per preprocessing mode and is not a reason to continue chasing
all remaining multiclass jobs unless the paper specifically needs that axis.

`vowel.csv` is the strongest real-data positive case: logloss improves from
0.2251 to 0.1271 under onehot and from 0.2282 to 0.1320 under drop. Regression
positives include `kin8nm.csv`, `Moneyball.csv`, and `naval_propulsion_plant.csv`.
Binary positives include `phoneme.csv`, `kc1.csv`, and `ilpd.csv`. Negative cases
include `cpu_activity.csv`, `balance-scale.csv`, `segment.csv`, `boston.csv`, and
`energy_efficiency.csv`.

The other-tabular evidence therefore reinforces the same scientific conclusion as
TabArena: MetaGB is a conditional advantage model, not a uniform benchmark winner.
It creates a credible basis for analyzing what dataset structures make local
linear leaves helpful.

## Runtime and Computational Tradeoff

Runtime remains the principal practical cost. Across all paired folds the median
MetaGB/XGB elapsed-time ratio is 18.29x. Real tabular data alone are similar at
18.38x. TabArena30 reduces the median to 4.20x because it uses the fixed-backbone
MGB30 protocol, but high-pressure datasets remain expensive.

The pressure proxy `rows * features^2` continues to explain many costly cases.
High-pressure tail datasets include `coil2000_insurance_policies.csv` at 62.9M,
`taiwanese_bankruptcy_prediction.csv` at 60.3M, `NATICUSdroid.csv` at 55.4M, and
`diamonds.csv` onehot at 36.5M. These are not uniformly good targets for MetaGB:
several are losses or ties. For the AAAI-27 paper, this motivates reporting
accuracy and computational budget jointly, rather than using speed-neutral
language.

## Paper-Facing Interpretation

The strongest defensible framing is:

1. MetaGB contributes an inductive-bias extension to gradient-boosted trees:
   constant leaves are augmented by local predictive-feature responses.
2. Synthetic mechanisms validate the intended hypothesis with high specificity:
   gains concentrate in interactions, regime-dependent slopes, conditional
   predictive separation, and sparse local-linear structures.
3. Real data show selective transfer, not universal dominance. This is acceptable
   if the paper positions MetaGB as a structure-aware extension rather than a
   blanket SOTA replacement.
4. Runtime is a first-class experimental axis. The method is slower under full
   search, and TabArena30 should be presented as a coverage-oriented tail protocol.

This framing is academically cleaner than claiming average superiority. It is
also harder to refute: the positive claim is tied to identifiable data mechanisms,
the negative controls behave as expected, and the real-data heterogeneity is
reported rather than hidden.

## Limitations

Several limitations should be explicit in any paper draft:

- Synthetic results use a single data seed and five outer folds; they are strong
  mechanism checks but not a substitute for broader stochastic robustness.
- TabArena30 is a fixed-backbone protocol, not full Random300/Random600 search.
- `tabarena30_onehot0702` is incomplete and should not be over-weighted.
- Some status files contain stale or internally inconsistent task counters; paired
  fold parsing is therefore the authoritative statistical source.
- Runtime ratios use elapsed task time from logs and are sensitive to machine,
  scheduling, and tail-task effects.

## Recommended Reporting Strategy

For the main paper, report three tables rather than one overloaded table:

- A synthetic mechanism table, with negative controls included.
- A real-data table split by TabArena main, TabArena30, and other tabular.
- A runtime/coverage table showing median runtime ratios and incomplete blocks.

For the narrative, emphasize "mechanism-aligned advantage" rather than "global
dominance." The 0702 data are strong enough to support the former and too mixed
to support the latter.
