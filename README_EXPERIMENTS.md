# MetaGB 实验完整流程指南

## 目录
1. [项目结构](#项目结构)
2. [环境配置](#环境配置)
3. [快速开始](#快速开始)
4. [实验模块说明](#实验模块说明)
5. [结果分析](#结果分析)
6. [常见问题](#常见问题)

---

## 项目结构

```
g:/VSCodeWork/d_MetaGB_ExperimentV2/
├── run_experiments.py          # 主实验运行脚本
├── analyze_results.py          # 结果分析和可视化脚本
├── ABC_Project/metagb/        # MetaGB实现
│   ├── metagb_utils.py        # MetaGB Python接口
│   ├── metagb.c             # MetaGB C实现
│   └── build_metagb.py      # MetaGB编译脚本
├── dataset/                  # 数据集目录
│   ├── tabArena_regression/   # TabArena回归数据集
│   └── artificial_nonlinear_regression/  # 合成数据集
├── utils/                   # 工具模块
│   ├── logger.py            # 日志记录
│   ├── evaluator.py         # 评估指标
│   ├── data_loader.py       # 数据加载
│   └── visualizer.py       # 可视化
├── results/                 # 实验结果保存目录
├── logs/                   # 实验日志保存目录
└── plots/                  # 图表保存目录
```

---

## 环境配置

### 1. 安装依赖

```bash
# 安装Python依赖
pip install -r ABC_Project/metagb/requirements.txt
```

所需依赖包括：
- numpy
- pandas
- xgboost
- scikit-learn
- cffi
- matplotlib
- seaborn

### 2. 编译MetaGB库

**重要：首次使用前必须编译MetaGB C库**

```bash
# 进入MetaGB目录
cd ABC_Project/metagb

# 编译MetaGB库
python build_metagb.py

# 返回项目根目录
cd ../../
```

编译完成后，会在当前目录生成 `libmetagb.dll`（Windows）或 `libmetagb.so`（Linux/Mac）文件。

---

## 快速开始

### 一键运行所有实验

```bash
# 运行所有实验模块（A + B + C）
python run_experiments.py --module all

# 或仅运行特定模块
python run_experiments.py --module A    # 仅参数理解实验
python run_experiments.py --module B    # 仅合成数据集测试
python run_experiments.py --module C    # 仅TabArena数据集测试
```

### 分析实验结果

```bash
# 自动分析所有结果并生成图表
python analyze_results.py
```

---

## 实验模块说明

### 模块A: 参数理解实验（糖尿病数据集）

**目标**: 理解MetaGB各超参数对模型性能的影响

**数据集**: Scikit-learn糖尿病数据集（442样本，10特征）

**方法**: 10-fold交叉验证 + 网格搜索

**测试参数**:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [3, 5, 7]
- `learning_rate`: [0.01, 0.1, 0.3]
- `pred_feature_fraction`: [0.0, 0.4, 0.8]
- `subsample`: [0.8, 1.0]

**对比模型**: MetaGB v3 vs XGBoost

**输出结果**:
- 每个参数组合的平均训练/验证MSE ± 标准差
- 参数重要性排序
- 最优参数配置

---

### 模块B: 合成数据集性能测试

**目标**: 在可控条件下测试MetaGB的鲁棒性和优势场景

#### B.1: 非线性模式测试

测试4种非线性模式（每种模式测试6种噪声水平）：

1. **平方项非线性**: `y = x1² + x2² + x3² + x4 + x5 + x6 + noise`
2. **正弦波非线性**: `y = 2*sin(x1*x2) + 0.5*sin(x3*x4) + noise`
3. **混合非线性**: `y = x1² + x2² + 2*sin(x3*x4) + noise`
4. **Sigmoid非线性**: `y = 1/(1+exp(-x1-x2)) + 0.5*x3 + noise`

**噪声水平**: [0.0, 0.1, 0.3, 0.5, 0.6, 1.0]

**样本数**: 4000，**特征数**: 10

#### B.2: 测试pred_feature_fraction影响

对每个数据集和噪声水平，测试5种pred_feature_fraction值：
- 0.0（仅bias，等同标准GBDT）
- 0.2
- 0.4
- 0.6
- 0.8

**对比**: MetaGB vs XGBoost

**输出结果**:
- 不同非线性模式下的性能对比表
- 噪声鲁棒性曲线（噪声水平 vs RMSE）
- pred_feature_fraction对性能的影响

---

### 模块C: TabArena真实数据集批量测试

**目标**: 验证MetaGB在真实数据集上的泛化能力

**数据集**（12个）:
- airfoil_self_noise.csv
- concrete_compressive_strength.csv
- diamonds.csv
- Food_Delivery_Time.csv
- healthcare_insurance_expenses.csv
- houses.csv
- miami_housing.csv
- physiochemical_protein.csv
- QSAR_fish_toxicity.csv
- QSAR-TID-11.csv
- superconductivity.csv
- wine_quality.csv

**方法**: 
- 5-fold交叉验证
- 测试不同pred_feature_fraction值

**输出结果**:
- 每个数据集的最佳参数配置
- 性能对比矩阵（数据集 × 模型 × 指标）
- MetaGB优于XGBoost的数据集统计

---

## 命令行参数

### run_experiments.py

```bash
python run_experiments.py [OPTIONS]

选项:
  --module {A,B,C,all}  运行哪个实验模块 (默认: all)
  --cv INT                交叉验证折数 (默认: 10)
  --results_dir PATH       结果保存目录 (默认: results)
  --logs_dir PATH         日志保存目录 (默认: logs)
  --plots_dir PATH        图表保存目录 (默认: plots)
  --n_jobs INT           并行任务数 (默认: 1)
```

### 使用示例

```bash
# 示例1: 运行模块A，使用5折交叉验证
python run_experiments.py --module A --cv 5

# 示例2: 运行模块C，指定输出目录
python run_experiments.py --module C --results_dir my_results --logs_dir my_logs

# 示例3: 运行所有模块，使用4核并行
python run_experiments.py --module all --n_jobs 4
```

---

## 结果分析

### 1. 结果文件结构

```
results/
├── module_A_diabetes/
│   └── results.json                    # 模块A结果
├── module_B_synthetic/
│   └── results.json                    # 模块B结果
├── module_C_tabarena/
│   └── results.json                    # 模块C结果
└── all_results_20260204_103045.json   # 完整结果汇总

logs/
└── experiment_20260204_103045.log     # 实验日志

plots/
├── module_A_*.png                     # 模块A图表
├── module_B_*.png                     # 模块B图表
├── module_C_*.png                     # 模块C图表
└── summary_report.md                   # 汇总报告
```

### 2. 结果JSON格式

#### 模块A结果示例:
```json
{
  "module": "A",
  "dataset": "diabetes",
  "n_folds": 10,
  "metagb": [
    {
      "params": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "pred_feature_fraction": 0.4,
        "subsample": 0.8
      },
      "train_mse_mean": 2500.1234,
      "train_mse_std": 120.5678,
      "val_mse_mean": 2800.2345,
      "val_mse_std": 150.6789,
      "avg_train_time": 2.34
    }
  ],
  "xgboost": [...]
}
```

#### 模块B结果示例:
```json
{
  "module": "B",
  "dataset_type": "synthetic",
  "n_samples": 4000,
  "n_features": 10,
  "results": [
    {
      "data_type": "square",
      "noise_level": 0.0,
      "metagb": {
        "pred_0.0": {"train_rmse": 0.1234, "test_rmse": 0.2345, "train_time": 1.23},
        "pred_0.4": {"train_rmse": 0.0987, "test_rmse": 0.1876, "train_time": 1.45},
        ...
      },
      "xgboost": {"train_rmse": 0.1356, "test_rmse": 0.2456, "train_time": 0.98}
    }
  ]
}
```

#### 模块C结果示例:
```json
{
  "module": "C",
  "dataset_type": "tabarena",
  "n_folds": 5,
  "datasets": {
    "airfoil_self_noise": {
      "shape": [1503, 5],
      "xgboost": {
        "train_rmse": 2.34,
        "val_rmse": 2.56,
        "test_rmse": 2.61,
        "train_time": 1.23
      },
      "metagb": [
        {
          "pred_feature_fraction": 0.0,
          "train_rmse": 2.31,
          "val_rmse": 2.54,
          "test_rmse": 2.59,
          "train_time": 1.45
        },
        ...
      ]
    }
  }
}
```

### 3. 可视化图表

运行 `python analyze_results.py` 会自动生成以下图表：

**模块A图表**:
- `module_A_metagb_*_sensitivity.png` - MetaGB参数敏感性分析
- `module_A_xgb_*_sensitivity.png` - XGBoost参数敏感性分析
- `module_A_model_comparison.png` - 最佳模型对比

**模块B图表**:
- `module_B_noise_robustness_pred{frac}.png` - 噪声鲁棒性曲线
- `module_B_data_type_comparison.png` - 数据类型性能对比
- `module_B_{data_type}_pred_fraction.png` - pred_fraction影响分析

**模块C图表**:
- `module_C_tabarena_performance.png` - TabArena数据集性能对比
- `module_C_improvement_distribution.png` - 改进率分布
- `module_C_best_pred_fraction.png` - 最佳pred_fraction分布

**汇总报告**:
- `summary_report.md` - 文字版汇总报告

---

## 常见问题

### Q1: 编译MetaGB时出现错误

**错误信息**: `cl: command not found` 或类似编译器错误

**解决方法**: 
- Windows: 安装Visual Studio Build Tools或MinGW
- Linux: `sudo apt-get install build-essential`
- Mac: `xcode-select --install`

### Q2: 找不到libmetagb模块

**错误信息**: `ModuleNotFoundError: No module named 'libmetagb'`

**解决方法**: 
1. 确保已经运行了 `python ABC_Project/metagb/build_metagb.py`
2. 检查当前目录下是否生成了 `libmetagb.so` 或 `libmetagb.dll`
3. 如果在Windows上，确保 `.dll` 文件在正确的位置

### Q3: 数据集加载失败

**错误信息**: `FileNotFoundError: dataset/tabArena_regression/xxx.csv`

**解决方法**:
1. 确保数据集文件已正确放置在 `dataset/tabArena_regression/` 目录
2. 检查文件名是否匹配（注意大小写）
3. 某些数据集可能需要手动下载

### Q4: 内存不足错误

**错误信息**: `MemoryError` 或程序崩溃

**解决方法**:
1. 减少交叉验证折数: `--cv 5`
2. 减少数据集规模
3. 减少参数组合数量（修改代码中的param_grid）
4. 关闭不必要的程序释放内存

### Q5: 实验运行时间过长

**优化建议**:
1. 仅运行感兴趣的模块: `--module A` 而不是 `--module all`
2. 使用并行处理: `--n_jobs 4`（根据CPU核心数调整）
3. 减少交叉验证折数: `--cv 5`
4. 对于模块C，先测试少量数据集，确认流程正常后再运行全部

### Q6: 如何继续中断的实验

**解决方法**:
1. 查看已保存的结果文件
2. 修改 `run_experiments.py` 中的参数网格，跳过已完成的组合
3. 或使用独立的实验脚本（如 `ABC_Project/metagb/compare_metagb_xgboost_grid.py`）

### Q7: 如何自定义参数网格

**解决方法**:
编辑 `run_experiments.py` 中的参数网格定义：
- 模块A: 修改 `metagb_param_grid` 和 `xgb_param_grid`
- 模块C: 修改 `fixed_xgb_params` 和 `metagb_fixed_params`

例如：
```python
metagb_param_grid = {
    'n_estimators': [100, 200, 300],  # 添加更多值
    'max_depth': [3, 5, 7, 9],        # 添加更多值
    'learning_rate': [0.05, 0.1],     # 减少值
    # ...
}
```

### Q8: 如何添加新的数据集

**解决方法**:
1. 将CSV文件放入 `dataset/` 或相应子目录
2. 确保CSV格式：所有特征列在前，最后一列为目标变量
3. 修改 `run_experiments.py` 中的 `datasets` 列表，添加新数据集文件名

---

## 预期运行时间

以下为估计时间（基于单核CPU，4核并行可缩短约3倍）：

- **模块A**: 约30-60分钟（270个参数组合 × 10折）
- **模块B**: 约20-40分钟（4数据类型 × 6噪声 × 5pred_frac × 2模型）
- **模块C**: 约2-4小时（12数据集 × 5 pred_frac × 5折）
- **总计**: 约3-6小时

**建议**: 分模块运行，每个模块完成后检查结果再进行下一个。

---

## 技术支持

遇到问题请检查：

1. **日志文件**: `logs/experiment_*.log` - 记录详细错误信息
2. **环境检查**: `pip list | grep -E "(numpy|xgboost|cffi)"` - 确认依赖版本
3. **编译产物**: 检查 `libmetagb.dll` 或 `libmetagb.so` 是否存在

---

## 参考资料

- MetaGB论文: [待添加]
- 实验计划: `当前实验计划_until0204.md`
- MetaGB文档: `ABC_Project/metagb/README.md`
- 原始对比脚本: 
  - `ABC_Project/metagb/compare_metagb_xgboost_grid.py`
  - `ABC_Project/metagb/compare_metagb_xgboost_housing_cv.py`

---

**最后更新**: 2026年2月2日
**版本**: 1.0
