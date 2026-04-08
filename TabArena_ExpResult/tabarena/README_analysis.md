# XGBoost嵌套CV结果分析 - 使用说明

## 概述

本分析脚本用于统计200个XGBoost配置在13个回归数据集上的Top 10模型表现，包括验证集和测试集的RMSE、MAE和R²指标。

## 快速开始

### 运行分析脚本

```bash
python analyze_xgb_top10_results.py
```

脚本会自动：
1. 扫描 `xgb_nested_cv_results_fast/` 目录下的所有数据集
2. 加载每个数据集的200个XGBoost配置结果
3. 计算完整的评估指标（RMSE, MAE, R²）
4. 选择每个数据集上的Top 10模型（基于验证集RMSE）
5. 生成多种格式的分析报告

### 预期运行时间

- 约 10-20 秒（13个数据集 × 200个模型）

## 输出文件

所有结果保存在 `analysis_results/` 目录下：

### 1. per_dataset_top10_detailed.csv
每个数据集的Top 10模型详细结果（130行 = 13数据集 × 10模型）

**列说明：**
- `dataset`: 数据集名称
- `model_name`: 模型配置名称（如 XGBoost_r145_BAG_L1）
- `rank`: 在该数据集上的排名（1-10）
- `val_rmse`: 验证集RMSE
- `val_mae`: 验证集MAE
- `val_r2`: 验证集R²
- `test_rmse`: 测试集RMSE
- `test_mae`: 测试集MAE
- `test_r2`: 测试集R²
- `time_train_s`: 训练时间（秒）
- `time_infer_s`: 推理时间（秒）

**示例：**
```
dataset,model_name,rank,val_rmse,val_mae,val_r2,test_rmse,test_mae,test_r2
airfoil_self_noise,XGBoost_r174_BAG_L1,1,1.438,0.898,0.955,1.114,0.787,0.976
airfoil_self_noise,XGBoost_r56_BAG_L1,2,1.441,0.894,0.955,1.073,0.756,0.978
...
```

### 2. summary_statistics.csv
每个数据集Top 10模型的统计摘要（78行 = 13数据集 × 6指标）

**列说明：**
- `dataset`: 数据集名称
- `metric`: 指标名称（val_rmse, val_mae, val_r2, test_rmse, test_mae, test_r2）
- `mean`: 平均值
- `std`: 标准差
- `min`: 最小值
- `max`: 最大值
- `median`: 中位数

**示例：**
```
dataset,metric,mean,std,min,max,median
airfoil_self_noise,val_rmse,1.456,0.015,1.438,1.486,1.458
airfoil_self_noise,test_rmse,1.122,0.024,1.073,1.151,1.133
...
```

### 3. top10_models_comparison.csv
跨数据集的Top模型比较（显示哪些模型最常出现在Top 10中）

**列说明：**
- `model_name`: 模型配置名称
- `top10_count`: 出现在Top 10中的次数（最多13次）
- `avg_rank`: 平均排名

**示例：**
```
model_name,top10_count,avg_rank
XGBoost_r29_BAG_L1,4,4.75
XGBoost_r21_BAG_L1,4,4.75
XGBoost_r108_BAG_L1,4,8.00
...
```

### 4. all_models_full_results.csv
所有模型在所有数据集上的完整结果（2613行 = 13数据集 × ~201模型）

包含所有200个配置在13个数据集上的完整指标，用于深度分析。

### 5. xgb_top10_analysis_report.xlsx
Excel格式的综合报告，包含3个工作表：
- `Top10_Detailed`: 详细结果
- `Summary_Statistics`: 汇总统计
- `Top_Models_Comparison`: 模型比较

## 关键发现

根据分析结果，以下模型最常出现在Top 10中：

1. **XGBoost_r29_BAG_L1** - 4个数据集，平均排名4.75
2. **XGBoost_r21_BAG_L1** - 4个数据集，平均排名4.75
3. **XGBoost_r108_BAG_L1** - 4个数据集，平均排名8.00
4. **XGBoost_r156_BAG_L1** - 3个数据集，平均排名3.67
5. **XGBoost_r1_BAG_L1** - 3个数据集，平均排名3.33

## 数据集列表

分析涵盖以下13个回归数据集：

1. airfoil_self_noise
2. Another-Dataset-on-used-Fiat-500
3. concrete_compressive_strength
4. diamonds
5. Food_Delivery_Time
6. healthcare_insurance_expenses
7. houses
8. miami_housing
9. physiochemical_protein
10. QSAR-TID-11
11. QSAR_fish_toxicity
12. superconductivity
13. wine_quality

## 技术细节

### 指标计算方法

所有指标从原始预测结果重新计算：

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAE
mae = mean_absolute_error(y_true, y_pred)

# R²
r2 = r2_score(y_true, y_pred)
```

### Top 10选择标准

- 基于**验证集RMSE**（越小越好）
- 每个数据集独立排序
- 选择前10个模型

### 数据来源

- 结果目录: `xgb_nested_cv_results_fast/`
- 结果文件: `experiments/data/XGBoost_r*_BAG_L1/*/0_4/results.pkl`
- 配置数量: 200个随机搜索配置 + 1个默认配置

## 自定义分析

### 修改Top K数量

编辑脚本中的 `select_top10_models` 调用：

```python
df_top10 = select_top10_models(df_dataset, top_k=20)  # 改为Top 20
```

### 更改排序指标

修改 `select_top10_models` 函数：

```python
def select_top10_models(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    # 改为按测试集RMSE排序
    df_sorted = df.sort_values('test_rmse', ascending=True).reset_index(drop=True)
    ...
```

### 添加其他指标

在 `calculate_all_metrics` 函数中添加：

```python
from sklearn.metrics import mean_absolute_percentage_error

def calculate_all_metrics(y_true, y_pred):
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),  # 新增
    }
```

## 故障排除

### 问题：找不到results.pkl文件

**解决方案：** 确保 `xgb_nested_cv_results_fast/` 目录存在且包含实验结果。

### 问题：内存不足

**解决方案：** 脚本使用增量加载，正常情况下不会超过2GB内存。如果仍有问题，可以分批处理数据集。

### 问题：某些数据集缺失

**解决方案：** 检查该数据集的实验是否成功完成。脚本会跳过缺失的数据集并继续处理其他数据集。

## 相关文档

- 实现方案: `docs/xgb_top10_analysis_plan.md`
- 原始实验脚本: `fast_run_XGB_nestedCV.py`
- 数据集分析工具: `tabarena/plot/dataset_analysis.py`

## 联系方式

如有问题或建议，请查看项目文档或提交Issue。
