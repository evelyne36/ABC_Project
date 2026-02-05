# MetaGB 实验项目结构说明

## 文件清单

### 核心脚本

| 文件名 | 说明 | 依赖 |
|---------|------|------|
| `run_experiments.py` | 主实验运行脚本，包含模块A/B/C的实现 | MetaGB编译库, 所有依赖包 |
| `analyze_results.py` | 结果分析和可视化脚本 | matplotlib, seaborn, pandas |
| `test_setup.py` | 环境测试脚本，验证配置是否正确 | 所有依赖包 |
| `run_all.bat` | Windows快速启动脚本（交互式菜单） | Windows命令行 |

### 工具模块 (utils/)

| 文件名 | 功能 |
|---------|------|
| `logger.py` | 日志记录，包含ExperimentLogger类用于主脚本 |
| `evaluator.py` | 评估指标计算（MSE, RMSE, MAE, R²） |
| `data_loader.py` | 数据加载和预处理 |
| `visualizer.py` | 可视化工具 |
| `model_trainer.py` | 模型训练封装 |

### MetaGB实现 (ABC_Project/metagb/)

| 文件名 | 说明 |
|---------|------|
| `metagb_utils.py` | MetaGB Python接口，定义MetaGB类 |
| `metagb.c` | MetaGB C语言实现 |
| `metagb.h` | MetaGB C头文件 |
| `build_metagb.py` | 编译脚本，生成libmetagb库 |
| `compare_metagb_xgboost_grid.py` | 参考示例：网格搜索对比 |
| `compare_metagb_xgboost_housing_cv.py` | 参考示例：交叉验证对比 |

### 数据集 (dataset/)

| 目录 | 说明 |
|------|------|
| `tabArena_regression/` | TabArena回归数据集（12个） |
| `artificial_nonlinear_regression/` | 合成数据集缓存目录 |

### 输出目录

| 目录 | 说明 | 创建时机 |
|------|------|---|
| `results/` | 实验结果JSON文件 | 首次运行实验时自动创建 |
| `logs/` | 实验日志文件 | 首次运行实验时自动创建 |
| `plots/` | 可视化图表PNG文件 | 首次分析结果时自动创建 |

### 文档

| 文件名 | 说明 |
|---------|------|
| `README_EXPERIMENTS.md` | 实验完整使用指南 |
| `当前实验计划_until0204.md` | 实验计划文档 |
| `PROJECT_STRUCTURE.md` | 本文件，项目结构说明 |

---

## 实验模块详细说明

### 模块A: 参数理解实验

**位置**: `run_experiments.py` 中的 `experiment_module_A_diabetes()` 函数

**输入**: 
- 糖尿病数据集（sklearn内置）

**输出**: 
- `results/module_A_diabetes/results.json`

**测试组合**:
- MetaGB: 90个参数组合 × 10折 = 900次训练
- XGBoost: 72个参数组合 × 10折 = 720次训练

### 模块B: 合成数据集测试

**位置**: `run_experiments.py` 中的 `experiment_module_B_synthetic()` 函数

**输入**:
- 4种合成数据类型 × 6种噪声水平

**输出**:
- `results/module_B_synthetic/results.json`

**测试组合**:
- 每个测试: 5种pred_fraction × 2模型 = 10次训练
- 总计: 24种数据场景 × 10次 = 240次训练

### 模块C: TabArena数据集测试

**位置**: `run_experiments.py` 中的 `experiment_module_C_tabarena()` 函数

**输入**:
- TabArena回归数据集（12个CSV文件）

**输出**:
- `results/module_C_tabarena/results.json`

**测试组合**:
- 每个数据集: 5种pred_fraction + 1次XGBoost = 6次训练
- 总计: 12数据集 × 6次 = 72次训练

---

## 数据流程图

```
数据准备
  │
  ├─ 模块A: 糖尿病数据集(sklearn.datasets.load_diabetes)
  │      │
  │      └─> 训练/验证/测试划分(60/20/20)
  │
  ├─ 模块B: 合成数据生成
  │      │
  │      ├─ generate_synthetic_regression_square()
  │      ├─ generate_synthetic_regression_sin()
  │      ├─ generate_synthetic_regression_square_sin()
  │      └─ generate_synthetic_regression_sigmoid()
  │           │
  │           └─> 添加高斯噪声 [0.0, 0.1, 0.3, 0.5, 0.6, 1.0]
  │
  └─ 模块C: TabArena数据集(dataset/tabArena_regression/*.csv)
         │
         └─> 最后一列为目标，其余为特征
              │
              └─> 训练/验证/测试划分(60/20/20)

模型训练
  │
  ├─ XGBoost: XGBRegressor(sklearn API)
  │      │
  │      └─> 参数: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
  │
  └─ MetaGB: MetaGB(metagb_utils.py)
         │
         └─> 参数: num_trees, max_depth, learning_rate, pred_feature_fraction,
              feature_fraction, bagging_fraction, min_data_in_leaf, max_bin, version

结果保存
  │
  ├─ JSON格式(results/**/results.json)
  │      │
  │      ├─ 参数配置
  │      ├─ 训练集指标(MSE, RMSE, MAE, R²)
  │      ├─ 验证集指标
  │      ├─ 测试集指标
  │      └─ 训练时间
  │
  └─ 日志文件(logs/experiment_*.log)
         │
         └─> 详细执行过程和错误信息

结果分析
  │
  └─ analyze_results.py
         │
         ├─ 参数敏感性分析(模块A)
         ├─ 噪声鲁棒性曲线(模块B)
         ├─ 数据类型性能对比(模块B)
         ├─ TabArena性能矩阵(模块C)
         └─ 汇总报告(summary_report.md)

可视化
  │
  └─ plots/*png
         ├─ 模块A: 参数敏感性图、模型对比图
         ├─ 模块B: 噪声鲁棒性图、pred_fraction影响图
         └─ 模块C: 性能对比图、改进率分布图
```

---

## 关键配置参数

### MetaGB版本选择

| version | 名称 | 特点 | 推荐场景 |
|---------|------|------|---------|
| 0 | 标准GBDT | 等同XGBoost（仅bias预测） | 基线对比 |
| 2 | 完整版 | 支持深度分离、先验特征 | 高级用法 |
| 3 | 简化版 | 推荐使用，参数少、易调优 | 大多数场景 |

### 预期运行时间（单核）

| 模块 | 组合数 | 预计时间 | 并行优化 |
|-----|--------|---------|---------|
| A | 1620 | 30-60分钟 | 可并行 |
| B | 240 | 20-40分钟 | 不可并行 |
| C | 72 | 2-4小时 | 可并行 |

**并行策略**: 
- 模块A: 可按参数组合或折数并行
- 模块B: 顺序执行（数据生成）
- 模块C: 可按数据集并行

---

## 修改建议

### 自定义参数网格

编辑 `run_experiments.py` 中的参数字典：

```python
# 模块A
metagb_param_grid = {
    'n_estimators': [100, 200],  # 修改此处
    'max_depth': [3, 5],          # 修改此处
    # ...
}

# 模块C
metagb_fixed_params = {
    'num_trees': 100,              # 修改此处
    'max_depth': 5,               # 修改此处
    # ...
}
```

### 添加新数据集

1. 将CSV文件放入 `dataset/tabArena_regression/`
2. 确保：特征列在前，最后一列为目标变量
3. 修改 `run_experiments.py` 中的 `datasets` 列表

### 添加新的可视化

编辑 `analyze_results.py`，添加新的绘图方法到 `ResultAnalyzer` 类。

---

## 故障排查流程

```
实验失败
  │
  ├─> 查看logs/experiment_*.log
  │      │
  │      ├─ ImportError → 安装缺失的包或编译MetaGB
  │      └─ 运行时错误 → 检查数据格式、参数范围
  │
  ├─> 运行test_setup.py
  │      │
  │      └─> 检查环境完整性
  │
  └─> 参考README_EXPERIMENTS.md的"常见问题"章节
```

---

## 版本历史

### v1.0 (2026-02-04)

- 实现完整的三模块实验框架
- 支持参数网格搜索和交叉验证
- 自动化结果分析和可视化
- 提供Windows快速启动脚本
- 包含环境测试工具

---

**维护者**: MetaGB实验团队
**最后更新**: 2026年2月4日
