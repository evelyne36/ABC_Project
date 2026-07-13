# Synthetic MetaGB 数据体系与计算量摘要

完整 case 公式和分析问题见 `EXPERIMENT_DESIGN_ZH.md`。

## 数据体系

主实验包含：

| Type | 基础 cases | 每 case 维度 | Cells |
| --- | ---: | --- | ---: |
| A：XOR 与乘法交互 | 4 | 10 / 50 / 100 | 12 |
| B：Linear-in-Leaves | 5 | 10 / 50 / 100 | 15 |
| C：Conditional-Predictive Separation | 3 | 10 / 50 / 100 | 9 |
| D：High-Dimensional Sparse Interaction | 3 | 10 / 50 / 100 | 9 |
| E：负对照 | 3 | 10 / 50 / 100 | 9 |
| 独立高斯噪声压力 | 1 | 10 / 50 / 100 | 3 |
| **总计** | **19** |  | **57** |

所有补充无关特征均来自：

```text
Normal(0, 1)
```

所有目标扰动均为 Gaussian noise。噪声强度不在每个 case 内重复展开，
只设置一个独立高噪声 case。

## 主实验计算量

主实验仅比较：

- `xgb_cpu`
- `current_metagb`

配置：

```text
57 cells
data_seed = 42
5 outer x 4 inner folds
300 candidates/model
```

精确计算量：

```text
570 tasks
171,000 candidate evaluations
684,000 inner-CV fits
570 final outer-train fits
684,570 total model fits
```

最小恢复单元：

```text
scenario x data_seed x outer_fold x model
```

每个 Random300 task 包含：

```text
300 candidates x 4 inner folds + 1 final fit = 1,201 fits
```

## Linux 并行方式

16 核服务器使用：

```text
task_workers = 15
n_jobs = 1
```

即 15 个 task 分别在 15 个核上独立执行，每个模型拟合单线程。详细
命令、恢复方法和输出检查见 `LINUX_RUN_GUIDE.md`。
