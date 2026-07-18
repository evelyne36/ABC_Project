# Synthetic Baseline 树模型速度剖析

## 1. 测量设置

本次剖析直接使用正式工程的：

- synthetic 数据生成器；
- baseline 模型工厂；
- Random300 中前 5 个确定性候选；
- 单线程模型训练。

数据规模模拟正式 nested-CV 的一个 inner fit：

```text
n_samples = 3,000
train rows = 1,800
validation rows = 600
dimensions = 10, 20
scenarios = regime_slope, xor_classification
models = LightGBM, CatBoost, Random Forest
```

共完成 60 次拟合。原始结果位于运行时生成的
`results/tree_profile/<timestamp>/fits.csv`，复现命令：

```bash
python synthetic_baselinemodel/scripts/profile_tree_baselines.py \
  --candidate-count 5 \
  --dimensions 10 20 \
  --scenarios regime_slope xor_classification \
  --n-jobs 1
```

## 2. 总体结果

| 模型 | 平均 fit | 中位 fit | P90 fit | 平均构造 | 平均预测 | 20维/10维 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.130 s | 0.120 s | 0.238 s | 0.045 s* | 0.009 s | 1.79× |
| CatBoost | 2.386 s | 0.889 s | 6.525 s | 0.002 s | 0.001 s | 1.96× |
| Random Forest | 1.988 s | 1.578 s | 3.800 s | 0.007 s | 0.018 s | 1.93× |

`*` LightGBM 的平均构造时间包含第一次动态导入的冷启动；后续构造远低于
该值。三个模型真正的持续性瓶颈都在 `fit()`，不是对象构造或预测。

## 3. CatBoost 为什么慢

CatBoost 的候选长尾最明显：

| 候选 | Trees | Depth | 平均 fit |
| --- | ---: | ---: | ---: |
| default | 400 | 6 | 0.741 s |
| slow tail | 800 | 9 | 8.737 s |
| sampled | 400 | 5 | 0.446 s |
| sampled | 400 | 7 | 1.285 s |
| sampled | 400 | 6 | 0.721 s |

`800 trees × depth 9` 比 `400 trees × depth 6` 慢约 12 倍。CatBoost 使用
对称树，深度增加会快速增加每棵树的叶结构与分裂计算；再叠加两倍迭代数，
少数深候选便主导了 Random300 总时间。20 维又使平均时间接近翻倍。

结论：CatBoost 慢在高深度与高迭代数的联合长尾，而不是启动或预测。

## 4. Random Forest 为什么慢

Random Forest 每个候选都会从零训练数百棵独立树：

| 候选 | Trees | Depth | Min leaf | Max features | 平均 fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | 300 | unlimited | 1 | sqrt | 0.948 s |
| sampled | 400 | 8 | 2 | 0.662 | 1.890 s |
| sampled | 200 | 8 | 17 | 0.894 | 1.046 s |
| sampled | 600 | 8 | 6 | 0.687 | 2.731 s |
| slow tail | 400 | unlimited | 1 | 0.867 | 3.323 s |

主要慢点：

1. `n_estimators` 最高到 800，训练量近似随树数线性增长；
2. `max_depth=None + min_samples_leaf=1` 会生成较深、较大的树；
3. `max_features` 接近 1 时，每次分裂扫描更多特征；
4. nested CV 中不能复用已训练的森林，每个候选、每个 inner fold 都会完整
   重建。

结论：Random Forest 慢在“树多 + 不限深 + 每次分裂看很多特征”的组合。

## 5. LightGBM 为什么相对快

LightGBM 的 5 个候选平均 fit 仅 0.045–0.196 秒。即使候选达到 600 trees，
其直方图算法仍明显快于逐棵构建的大森林和深层 CatBoost 对称树。

LightGBM 的总耗时仍不可忽略，是因为每个 outer task 包含：

```text
300 candidates × 4 inner folds + 1 final fit = 1,201 fits
```

它慢在重复次数，而非单次训练。首次库导入会有一次性冷启动，但对完整实验
影响很小。

## 6. 对完整 10/20-feature 实验的影响

38 cells 下，每个模型有：

```text
190 outer tasks
57,000 candidate evaluations
228,190 planned fits
```

将本次单 fit 均值外推，三个树模型约需：

| 模型 | 微基准投影 |
| --- | ---: |
| LightGBM | 约 8.2 worker-hours |
| CatBoost | 约 151.2 worker-hours |
| Random Forest | 约 126.0 worker-hours |

历史完整 Random300 task 日志给出的更保守投影分别约为 15.8、150.9、
170.0 worker-hours。因此树模型合计建议按 **285–337 worker-hours** 估算，
15 workers 下理想墙钟约 **19–23 小时**。

结合历史日志和本次微基准，完整三树模型 10/20-feature 实验建议按
**22–32 小时（约 1–1.5 天）**排期。
