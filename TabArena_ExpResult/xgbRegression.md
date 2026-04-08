# XGBoost 回归数据集 Baseline 运行说明

## 修改内容

已将 `run_xgboost_regression_validation.py` 更新为运行完整的 13 个回归数据集：

### 新增的 3 个回归数据集
- `363615` - Another-Dataset-on-used-Fiat-500 (1538 样本, 8 特征)
- `363672` - Food_Delivery_Time (45451 样本, 10 特征)
- `363697` - QSAR-TID-11 (5742 样本, 1025 特征)

### 完整的 13 个回归数据集列表
1. 363612 - airfoil_self_noise (1503 样本, 6 特征)
2. 363615 - Another-Dataset-on-used-Fiat-500 (1538 样本, 8 特征)
3. 363625 - concrete_compressive_strength (1030 样本, 9 特征)
4. 363631 - diamonds (53940 样本, 10 特征)
5. 363672 - Food_Delivery_Time (45451 样本, 10 特征)
6. 363675 - healthcare_insurance_expenses (1338 样本, 7 特征)
7. 363678 - houses (20640 样本, 9 特征)
8. 363686 - miami_housing (13776 样本, 16 特征)
9. 363693 - physiochemical_protein (45730 样本, 10 特征)
10. 363697 - QSAR-TID-11 (5742 样本, 1025 特征)
11. 363698 - QSAR_fish_toxicity (907 样本, 7 特征)
12. 363705 - superconductivity (21263 样本, 82 特征)
13. 363708 - wine_quality (6497 样本, 13 特征)

## 运行方法

### 1. 环境准备

确保已安装 TabArena 及其依赖：

```bash
# 安装 AutoGluon (如果未安装)
pip install autogluon

# 安装 TabArena
cd TabArena
pip install -e .
```

### 2. 运行 XGBoost Baseline

```bash
cd TabArena
python run_xgboost_regression_validation.py
```

### 3. 配置说明

脚本使用以下论文标准配置：
- **Bagging 折数**: 8 折 (论文标准)
- **时间限制**: 3600 秒/任务 (1 小时)
- **超参数**: XGBoost 默认超参数
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1

### 4. 输出结果

结果将保存在 `xgboost_regression_output/` 目录下，包含：
- 每个数据集的验证误差和测试误差
- 模型预测结果
- 性能指标

### 5. 预计运行时间

根据数据集大小，总运行时间约为：
- 小数据集 (< 2000 样本): 5-15 分钟/数据集
- 中等数据集 (2000-20000 样本): 15-45 分钟/数据集
- 大数据集 (> 20000 样本): 45-60 分钟/数据集

**总预计时间**: 约 6-10 小时 (13 个数据集)

## 注意事项

1. 确保有足够的磁盘空间存储结果
2. 建议在有良好计算资源的机器上运行
3. 可以通过修改 `time_limit` 参数调整每个任务的时间限制
4. 如需快速测试，可以将 `num_bag_folds` 改为 3
