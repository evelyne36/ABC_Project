# XGBoost嵌套CV结果分析方案

## 1. 任务目标

统计200个XGBoost配置中Top 10模型在13个回归数据集上的表现，包括：
- 验证集指标：RMSE, MAE, R²
- 测试集指标：RMSE, MAE, R²

## 2. 数据结构分析

### 2.1 目录结构
```
xgb_nested_cv_results_fast/
├── Another-Dataset-on-used-Fiat-500/
│   └── experiments/data/
│       ├── XGBoost_r1_BAG_L1/1065875806/0_4/results.pkl
│       ├── XGBoost_r2_BAG_L1/1065875806/0_4/results.pkl
│       └── ... (共200个配置)
├── airfoil_self_noise/
├── concrete_compressive_strength/
└── ... (共13个数据集)
```

### 2.2 results.pkl文件结构
每个results.pkl包含以下关键字段：
- `framework`: 模型名称 (如 "XGBoost_r100_BAG_L1")
- `metric_error`: 测试集RMSE
- `metric_error_val`: 验证集RMSE
- `simulation_artifacts`: 包含预测结果
  - `y_val`: 验证集真实标签
  - `y_test`: 测试集真实标签
  - `pred_proba_dict_val`: 验证集预测 (字典格式)
  - `pred_proba_dict_test`: 测试集预测 (字典格式)

### 2.3 数据集列表
1. Another-Dataset-on-used-Fiat-500
2. airfoil_self_noise
3. concrete_compressive_strength
4. diamonds
5. Food_Delivery_Time
6. healthcare_insurance_expenses
7. houses
8. miami_housing
9. physiochemical_protein
10. QSAR_fish_toxicity
11. QSAR-TID-11
12. superconductivity
13. wine_quality

## 3. 实现方案

### 3.1 核心步骤

1. **遍历数据集**
   - 扫描xgb_nested_cv_results_fast目录下的所有数据集文件夹

2. **加载模型结果**
   - 对每个数据集，加载所有200个XGBoost配置的results.pkl文件
   - 提取预测结果和真实标签

3. **计算完整指标**
   - 从simulation_artifacts中提取预测和标签
   - 使用sklearn计算：
     - RMSE: `sqrt(mean_squared_error(y_true, y_pred))`
     - MAE: `mean_absolute_error(y_true, y_pred)`
     - R²: `r2_score(y_true, y_pred)`

4. **选择Top 10模型**
   - 根据验证集RMSE排序（升序，越小越好）
   - 选择前10个模型

5. **生成结果报告**
   - 创建汇总表格（每个数据集的Top 10模型）
   - 保存为CSV和Excel格式
   - 生成可视化图表（可选）

### 3.2 输出格式

#### 详细结果表 (per_dataset_top10_detailed.csv)
| dataset | model_name | rank | val_rmse | val_mae | val_r2 | test_rmse | test_mae | test_r2 |
|---------|------------|------|----------|---------|--------|-----------|----------|---------|
| dataset1 | XGBoost_r45 | 1 | 0.123 | 0.098 | 0.95 | 0.134 | 0.102 | 0.94 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

#### 汇总统计表 (summary_statistics.csv)
| dataset | metric | mean | std | min | max |
|---------|--------|------|-----|-----|-----|
| dataset1 | val_rmse | 0.145 | 0.012 | 0.123 | 0.167 |
| ... | ... | ... | ... | ... | ... |

## 4. 技术实现

### 4.1 依赖库
- pandas: 数据处理
- numpy: 数值计算
- sklearn.metrics: 指标计算
- pickle: 加载结果文件
- pathlib: 路径处理

### 4.2 关键函数

1. `load_model_results(results_path)`: 加载单个模型的results.pkl
2. `calculate_all_metrics(y_true, y_pred)`: 计算RMSE, MAE, R²
3. `process_dataset(dataset_path)`: 处理单个数据集的所有模型
4. `select_top10_models(df_results)`: 选择Top 10模型
5. `generate_summary_report(all_results)`: 生成汇总报告

### 4.3 错误处理
- 处理缺失的results.pkl文件
- 处理损坏的pickle文件
- 处理指标计算异常（如R²为负值）

## 5. 使用方法

```bash
python analyze_xgb_top10_results.py \
    --results_dir ./xgb_nested_cv_results_fast \
    --output_dir ./analysis_results \
    --top_k 10
```

## 6. 预期输出

```
analysis_results/
├── per_dataset_top10_detailed.csv      # 每个数据集的Top 10详细结果
├── summary_statistics.csv              # 汇总统计
├── top10_models_comparison.csv         # Top 10模型跨数据集比较
└── figures/                            # 可视化图表（可选）
    ├── top10_performance_heatmap.png
    └── metric_distribution.png
```

## 7. 扩展功能（可选）

1. 生成LaTeX表格（参考dataset_analysis.py中的save_latex_table）
2. 绘制性能热力图
3. 分析Top 10模型的超参数分布
4. 计算模型间的相关性
