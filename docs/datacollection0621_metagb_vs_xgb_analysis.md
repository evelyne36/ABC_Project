# datacollection0621: current_metagb vs xgb_cpu 实验结果分析

更新时间：2026-06-22  
数据来源：`datacollection0621/` 当前日志、`task_status.json`、synthetic task JSON  
比较对象：`current_metagb` vs `xgb_cpu`

## 摘要结论

当前可比日志支持一个更细的结论：`current_metagb` 不是在 TabArena 主实验中全面优于 `xgb_cpu`，而是在特定任务族上表现出优势。

- Regression: `drop` 下明显偏弱，`onehot` 下从 win/loss 数看略好，但受 `houses` 和 `airfoil_self_noise` 两个大负例拖累，平均相对提升仍为负。
- Binary: 整体接近持平，绝大多数 AUC 差异小于 0.5%；`drop` 为 8W/8L，`onehot` 为 3W/7L。
- Multiclass: 当前可比数据较少，但是最支持 MetaGB 的 TabArena 组；`drop` 为 2W/2L，`onehot` 为 2W/1L，平均相对提升约 +1.3%。
- Synthetic: 当前只有 `dim10` 的两个 XOR 类 case 同时完成两模型，二者均为 MetaGB win；这支持机制假设，但不能代表完整 synthetic protocol。
- Runtime: MetaGB 通常显著更慢，尤其在 wide/high-cardinality 或 multiclass 任务上；因此论文叙述应强调 quality/runtime tradeoff 和适用边界，而不是速度替代。

建议论文主线表述为：

> MetaGB provides a richer local linear leaf response that can improve selected multiclass and mechanism-aligned synthetic cases, while its gains on broad TabArena tasks are heterogeneous and must be reported with dataset-level win/tie/loss and runtime tradeoffs.

## 实验口径

| 数据组 | 编码 | 搜索规模 | 模型 | CV | 主指标 | 方向 |
|---|---|---:|---|---|---|---|
| Regression | Drop / OneHot | Random600 | `xgb_cpu`, `current_metagb` | outer 5, inner 4 | RMSE | 越低越好 |
| Binary | Drop / OneHot | Random300 | `xgb_cpu`, `current_metagb` | outer 5, inner 4 | AUC | 越高越好 |
| Multiclass | Drop / OneHot | Random300 | `xgb_cpu`, `current_metagb` | outer 5, inner 4 | Logloss | 越低越好 |
| Synthetic | protocol scenarios | Random300 single seed | `xgb_cpu`, `current_metagb` | shared nested CV | AUC/RMSE/Logloss | 按任务类型 |

判定规则：

- Dataset/case 级 W/T/L 使用 5 个 outer folds 的 test 主指标均值。
- AUC: `MetaGB - XGB > 1e-4` 记为 win；小于 `-1e-4` 记为 loss。
- RMSE/Logloss: `XGB - MetaGB > 1e-6` 记为 win；小于 `-1e-6` 记为 loss。
- 只纳入两模型在同一 dataset/case 上 5 个 outer folds 都完成的可比单元。

## 完整性审计

当前解析到 650 条已完成 fold/task 记录，共 55 个可比 dataset/case。

| Run group | ok | pending | failed | 可比情况 |
|---|---:|---:|---:|---|
| regression/drop | 110 | 0 | 0 | 完整，11 datasets |
| regression/onehot | 100 | 10 | 0 | 9 datasets 可比；`diamonds`, `Food_Delivery_Time` 缺 MetaGB |
| binary/drop initial | 105 | 195 | 0 | 需与 remaining run 合并解释 |
| binary/drop remaining | 80 | 115 | 0 | 合并后 16 datasets 可比 |
| binary/onehot initial | 65 | 225 | 10 | 合并后 10 datasets 可比 |
| binary/onehot remaining | 65 | 160 | 0 | 仍有大量 pending |
| multiclass/drop | 45 | 5 | 30 | 4 datasets 可比；失败多为预处理后无行/无特征 |
| multiclass/onehot | 35 | 35 | 10 | 3 datasets 可比 |
| synthetic | 35 task JSON | - | - | 仅 2 个 dim10 XOR case 两模型完整 |

## Overall Win/Tie/Loss

### Dataset-level summary

| Scope | Comparable units | W/T/L | Win rate | Mean relative improvement | Median relative improvement |
|---|---:|---:|---:|---:|---:|
| TabArena only | 53 | 23/0/30 | 43.4% | - | - |
| TabArena + current synthetic comparable | 55 | 25/0/30 | 45.5% | - | - |

### By task and encoding

| Group | N | W/T/L | Visual W/L bar | Mean rel. | Median rel. | Val/Test same direction |
|---|---:|---:|---|---:|---:|---:|
| Regression Drop | 11 | 3/0/8 | WWWLLLLLLLL | -2.80% | -0.86% | 10/11 |
| Regression OneHot | 9 | 5/0/4 | WWWWWLLLL | -2.89% | +0.21% | 7/9 |
| Binary Drop | 16 | 8/0/8 | WWWWWWWWLLLLLLLL | -0.05% | -0.01% | 10/16 |
| Binary OneHot | 10 | 3/0/7 | WWWLLLLLLL | -0.05% | -0.08% | 7/10 |
| Multiclass Drop | 4 | 2/0/2 | WWLL | +1.33% | +0.87% | 3/4 |
| Multiclass OneHot | 3 | 2/0/1 | WWL | +1.36% | +1.77% | 3/3 |
| Synthetic comparable | 2 | 2/0/0 | WW | +0.11% | +0.11% | 2/2 |

Interpretation:

- `regression/drop` 的 dataset-level 结果偏负，且 val/test 基本同向，说明不是纯 CV 噪声问题。
- `regression/onehot` 的中位数为正，但均值为负，主要因为 `houses` 和 `airfoil_self_noise` 两个大负例。
- `binary` 的平均与中位提升都接近 0，当前不宜声称稳定优势。
- `multiclass` 的 win 比例和平均提升较好，但样本数太小，应标为 promising rather than conclusive。
- `synthetic` 当前仅验证了已完成的 dim10 XOR 子集，不代表 full synthetic suite。

## Result Matrix

表中 `Val` 与 `Test` 均为 `xgb_cpu / current_metagb`。`Rel.` 为 MetaGB 相对 XGB 的 test improvement：AUC 使用 `(MGB-XGB)/XGB`；RMSE/Logloss 使用 `(XGB-MGB)/XGB`。

### Regression: Drop

| Dataset | Metric | Val | Test | Rel. | Result |
|---|---|---:|---:|---:|---|
| Another-Dataset-on-used-Fiat-500.csv | RMSE | 745.818 / 753.634 | 750.622 / 761.066 | -1.39% | Loss |
| Food_Delivery_Time.csv | RMSE | 7.53976 / 7.54993 | 7.52504 / 7.53497 | -0.13% | Loss |
| QSAR_fish_toxicity.csv | RMSE | 0.876813 / 0.875690 | 0.902940 / 0.891145 | +1.31% | Win |
| airfoil_self_noise.csv | RMSE | 1.59282 / 1.60652 | 1.40665 / 1.52087 | -8.12% | Loss |
| concrete_compressive_strength.csv | RMSE | 4.35163 / 4.34663 | 3.89667 / 3.90686 | -0.26% | Loss |
| diamonds.csv | RMSE | 1343.56 / 1349.27 | 1338.29 / 1355.73 | -1.30% | Loss |
| healthcare_insurance_expenses.csv | RMSE | 11522.2 / 11655.5 | 11508.0 / 11608.5 | -0.87% | Loss |
| houses.csv | RMSE | 0.217688 / 0.233491 | 0.212653 / 0.255970 | -20.37% | Loss |
| miami_housing.csv | RMSE | 85819.5 / 84506.5 | 83836.7 / 83450.9 | +0.46% | Win |
| physiochemical_protein.csv | RMSE | 3.51634 / 3.49365 | 3.43548 / 3.41086 | +0.72% | Win |
| wine_quality.csv | RMSE | 0.616296 / 0.621195 | 0.588047 / 0.593125 | -0.86% | Loss |

Regression Drop takeaway:

- Strong losses: `houses` (-20.37%), `airfoil_self_noise` (-8.12%).
- Reliable wins: `QSAR_fish_toxicity`, `physiochemical_protein`, `miami_housing`.
- The negative mean is not only one tiny numerical drift; it is driven by visible large losses.

### Regression: OneHot

| Dataset | Metric | Val | Test | Rel. | Result |
|---|---|---:|---:|---:|---|
| Another-Dataset-on-used-Fiat-500.csv | RMSE | 736.221 / 743.513 | 740.777 / 748.834 | -1.09% | Loss |
| QSAR_fish_toxicity.csv | RMSE | 0.876813 / 0.875690 | 0.902940 / 0.891145 | +1.31% | Win |
| airfoil_self_noise.csv | RMSE | 1.59282 / 1.60652 | 1.40665 / 1.52087 | -8.12% | Loss |
| concrete_compressive_strength.csv | RMSE | 4.35163 / 4.34663 | 3.89667 / 3.90686 | -0.26% | Loss |
| healthcare_insurance_expenses.csv | RMSE | 4561.94 / 4542.68 | 4546.94 / 4493.98 | +1.16% | Win |
| houses.csv | RMSE | 0.217688 / 0.233491 | 0.212653 / 0.255970 | -20.37% | Loss |
| miami_housing.csv | RMSE | 85819.5 / 84506.5 | 83836.7 / 83450.9 | +0.46% | Win |
| physiochemical_protein.csv | RMSE | 3.51634 / 3.49365 | 3.43548 / 3.41086 | +0.72% | Win |
| wine_quality.csv | RMSE | 0.616095 / 0.617923 | 0.590158 / 0.588900 | +0.21% | Win |

Regression OneHot takeaway:

- Win count improves to 5/9, and median effect becomes positive.
- However, the same two strong negative cases remain: `houses` and `airfoil_self_noise`.
- `healthcare_insurance_expenses` flips from drop loss to onehot win, suggesting categorical handling can materially change MetaGB's relative behavior.

### Binary: Drop

| Dataset | Metric | Val | Test | Rel. | Result |
|---|---|---:|---:|---:|---|
| Amazon_employee_access.csv | AUC | 0.844226 / 0.841539 | 0.859789 / 0.858608 | -0.14% | Loss |
| Bank_Customer_Churn.csv | AUC | 0.849637 / 0.849081 | 0.850385 / 0.848853 | -0.18% | Loss |
| Diabetes130US.csv | AUC | 0.598985 / 0.598616 | 0.599449 / 0.598479 | -0.16% | Loss |
| E-CommereShippingData.csv | AUC | 0.744750 / 0.745076 | 0.744538 / 0.745572 | +0.14% | Win |
| Fitness_Club.csv | AUC | 0.808570 / 0.809683 | 0.810147 / 0.811218 | +0.13% | Win |
| GiveMeSomeCredit.csv | AUC | 0.856168 / 0.855612 | 0.856136 / 0.855610 | -0.06% | Loss |
| HR_Analytics_Job_Change_of_Data_Scientists.csv | AUC | 0.723510 / 0.723378 | 0.724459 / 0.725224 | +0.11% | Win |
| Is-this-a-good-customer.csv | AUC | 0.659148 / 0.657710 | 0.626316 / 0.623842 | -0.40% | Loss |
| Marketing_Campaign.csv | AUC | 0.876584 / 0.876276 | 0.873715 / 0.877472 | +0.43% | Win |
| blood-transfusion-service-center.csv | AUC | 0.738189 / 0.734885 | 0.717556 / 0.720450 | +0.40% | Win |
| churn.csv | AUC | 0.926342 / 0.925424 | 0.922337 / 0.922750 | +0.04% | Win |
| credit-g.csv | AUC | 0.661604 / 0.665607 | 0.675262 / 0.684024 | +1.30% | Win |
| diabetes.csv | AUC | 0.837945 / 0.840331 | 0.828799 / 0.827769 | -0.12% | Loss |
| hazelnut-spread-contaminant-detection.csv | AUC | 0.975083 / 0.977994 | 0.979490 / 0.983219 | +0.38% | Win |
| qsar-biodeg.csv | AUC | 0.930329 / 0.932871 | 0.932492 / 0.931635 | -0.09% | Loss |
| seismic-bumps.csv | AUC | 0.769716 / 0.766906 | 0.781116 / 0.761576 | -2.50% | Loss |

Binary Drop takeaway:

- W/L count is exactly balanced.
- Most effects are tiny; the largest positive case is `credit-g` (+1.30%), and the largest negative case is `seismic-bumps` (-2.50%).
- Several cases show val/test direction mismatch, so binary conclusions should be conservative.

### Binary: OneHot

| Dataset | Metric | Val | Test | Rel. | Result |
|---|---|---:|---:|---:|---|
| Amazon_employee_access.csv | AUC | 0.844226 / 0.841539 | 0.859789 / 0.858608 | -0.14% | Loss |
| Bank_Customer_Churn.csv | AUC | 0.867917 / 0.869101 | 0.868422 / 0.870679 | +0.26% | Win |
| E-CommereShippingData.csv | AUC | 0.742863 / 0.741729 | 0.745365 / 0.744980 | -0.05% | Loss |
| Fitness_Club.csv | AUC | 0.809437 / 0.809129 | 0.810823 / 0.808593 | -0.28% | Loss |
| GiveMeSomeCredit.csv | AUC | 0.856168 / 0.855612 | 0.856136 / 0.855610 | -0.06% | Loss |
| Is-this-a-good-customer.csv | AUC | 0.754352 / 0.750595 | 0.734754 / 0.728662 | -0.83% | Loss |
| blood-transfusion-service-center.csv | AUC | 0.738189 / 0.734885 | 0.717556 / 0.720450 | +0.40% | Win |
| diabetes.csv | AUC | 0.837945 / 0.840331 | 0.828799 / 0.827769 | -0.12% | Loss |
| hazelnut-spread-contaminant-detection.csv | AUC | 0.975083 / 0.977994 | 0.979490 / 0.983219 | +0.38% | Win |
| qsar-biodeg.csv | AUC | 0.930329 / 0.932871 | 0.932492 / 0.931635 | -0.09% | Loss |

Binary OneHot takeaway:

- OneHot does not improve binary W/L for MetaGB in the current completed subset.
- Positive cases are real but small.
- This group remains incomplete; additional completed datasets may change the count.

### Multiclass: Drop

| Dataset | Metric | Val | Test | Rel. | Result |
|---|---|---:|---:|---:|---|
| SDSS17.csv | Logloss | 0.073686 / 0.074060 | 0.072628 / 0.072807 | -0.25% | Loss |
| anneal.csv | Logloss | 0.252588 / 0.247156 | 0.226567 / 0.217858 | +3.84% | Win |
| maternal_health_risk.csv | Logloss | 0.465313 / 0.461732 | 0.400450 / 0.393375 | +1.77% | Win |
| students_dropout_and_academic_success.csv | Logloss | 0.607888 / 0.607684 | 0.602868 / 0.603046 | -0.03% | Loss |

Multiclass Drop takeaway:

- Best positive case: `anneal` (+3.84% logloss improvement).
- Negative cases are small, especially `students_dropout_and_academic_success` (-0.03%).
- This is a promising group but has only 4 completed comparable datasets.

### Multiclass: OneHot

| Dataset | Metric | Val | Test | Rel. | Result |
|---|---|---:|---:|---:|---|
| SDSS17.csv | Logloss | 0.073686 / 0.074060 | 0.072628 / 0.072807 | -0.25% | Loss |
| maternal_health_risk.csv | Logloss | 0.465313 / 0.461732 | 0.400450 / 0.393375 | +1.77% | Win |
| website_phishing.csv | Logloss | 0.247706 / 0.242267 | 0.245462 / 0.239167 | +2.56% | Win |

Multiclass OneHot takeaway:

- 2/3 wins, both with non-trivial improvements.
- `SDSS17` remains a small but consistent loss under both encodings.
- This is currently the cleanest TabArena evidence for MetaGB benefit, but sample size is still small.

### Synthetic: currently comparable subset

| Scenario | Family | Metric | Val | Test | Rel. | Result |
|---|---|---|---:|---:|---:|---|
| xor_3way__dim10 | A_interaction | AUC | 0.992499 / 0.997131 | 0.995231 / 0.997176 | +0.20% | Win |
| xor_classification__dim10 | A_interaction | AUC | 0.997781 / 0.998236 | 0.997988 / 0.998156 | +0.02% | Win |

Synthetic takeaway:

- The completed synthetic subset aligns with the structural hypothesis: conditional interaction tasks favor MetaGB.
- However, the current file set contains many `dim50/dim100` XGB-only tasks. These must not be used for final full synthetic claims until MetaGB counterparts complete.

## Positive and Negative Cases

### Strongest positive cases

| Rank | Group | Dataset | Metric | Relative improvement | Interpretation |
|---:|---|---|---|---:|---|
| 1 | Multiclass Drop | anneal.csv | Logloss | +3.84% | Clearest current multiclass win |
| 2 | Multiclass OneHot | website_phishing.csv | Logloss | +2.56% | OneHot multiclass positive case |
| 3 | Multiclass Drop/OneHot | maternal_health_risk.csv | Logloss | +1.77% | Stable across encodings |
| 4 | Regression Drop/OneHot | QSAR_fish_toxicity.csv | RMSE | +1.31% | Stable regression positive case |
| 5 | Binary Drop | credit-g.csv | AUC | +1.30% | Best binary improvement |
| 6 | Regression OneHot | healthcare_insurance_expenses.csv | RMSE | +1.16% | Encoding-sensitive positive case |

### Strongest negative cases

| Rank | Group | Dataset | Metric | Relative improvement | Interpretation |
|---:|---|---|---|---:|---|
| 1 | Regression Drop/OneHot | houses.csv | RMSE | -20.37% | Dominant regression failure case |
| 2 | Regression Drop/OneHot | airfoil_self_noise.csv | RMSE | -8.12% | Consistent loss across encodings |
| 3 | Binary Drop | seismic-bumps.csv | AUC | -2.50% | Largest binary loss |
| 4 | Regression Drop | Another-Dataset-on-used-Fiat-500.csv | RMSE | -1.39% | Stable negative case |
| 5 | Regression Drop | diamonds.csv | RMSE | -1.30% | Drop-only completed negative case |

## Val/Test Selection Behavior

| Group | Same direction | Notes |
|---|---:|---|
| Regression Drop | 10/11 | Validation mostly predicts test direction |
| Regression OneHot | 7/9 | Mostly aligned, but some small flips |
| Binary Drop | 10/16 | Frequent val/test mismatch; binary conclusions are less stable |
| Binary OneHot | 7/10 | Better than drop but still mixed |
| Multiclass Drop | 3/4 | Mostly aligned |
| Multiclass OneHot | 3/3 | Fully aligned in completed subset |
| Synthetic comparable | 2/2 | Fully aligned |

Implication:

- For regression, negative cases are likely real under current hyperparameter protocol.
- For binary, many wins/losses are near noise scale and validation selection does not always transfer.
- For multiclass, current evidence is small but internally consistent.

## Runtime-quality Tradeoff

The parsed logs show MetaGB usually has materially higher final training time. Examples:

| Dataset | Group | XGB final train sec | MetaGB final train sec | Ratio |
|---|---|---:|---:|---:|
| Amazon_employee_access.csv | binary/drop | 2.63 | 61.53 | 23.4x |
| hazelnut-spread-contaminant-detection.csv | binary/drop | 0.92 | 50.22 | 54.8x |
| qsar-biodeg.csv | binary/drop | 0.53 | 86.57 | 163.3x |
| SDSS17.csv | multiclass/drop | 20.72 | 312.28 | 15.1x |
| physiochemical_protein.csv | regression/drop | 17.18 | 127.66 | 7.4x |
| houses.csv | regression/drop | 8.05 | 9.91 | 1.2x |

Interpretation:

- Runtime overhead is workload-dependent.
- In some cases, such as `houses`, MetaGB is only modestly slower but much less accurate.
- In other cases, such as `qsar-biodeg`, MetaGB is much slower without test gain.
- Therefore, final paper tables should report quality and runtime together, at least in an appendix or scatter plot.

## Recommended Paper Framing

### Main claim to support

1. MetaGB is a richer leaf-response model integrated into XGBoost's objective/CV pipeline.
2. It shows promising gains in multiclass and mechanism-aligned synthetic tasks.
3. Across broad TabArena tasks, gains are heterogeneous; the correct empirical story is applicability boundary, not universal dominance.

### Claims to avoid

- Do not claim MetaGB generally outperforms XGB across TabArena.
- Do not claim MetaGB is a speed replacement for XGB.
- Do not claim full synthetic superiority until `dim50/dim100` MetaGB tasks complete.

### Suggested final tables/figures

| Paper artifact | Content | Current readiness |
|---|---|---|
| Table: Main W/T/L | task x encoding W/T/L + mean/median rel. improvement | Ready as interim |
| Table: Dataset-level results | val/test xgb/mgb for each completed comparable dataset | Ready as interim |
| Figure: Relative improvement bars | one bar per dataset, grouped by task and encoding | Can be generated from this table |
| Figure: Runtime vs improvement | x-axis runtime ratio, y-axis relative improvement | Recommended next |
| Synthetic figure | scenario family x dimension heatmap | Not ready until more MetaGB synthetic tasks complete |

## Final interim conclusion

The current `datacollection0621` evidence is strongest for a nuanced, top-conference-safe conclusion:

> MetaGB is not a uniformly better replacement for XGBoost on general tabular benchmarks. Its current empirical value appears in selected multiclass tasks and synthetic interaction mechanisms, while regression and binary results reveal clear applicability boundaries and non-trivial runtime cost. A credible final paper should foreground win/tie/loss, relative improvement distributions, runtime-quality tradeoffs, and positive/negative case analysis rather than only average scores.

