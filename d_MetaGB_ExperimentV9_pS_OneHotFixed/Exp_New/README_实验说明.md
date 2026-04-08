# MetaGB 实验 1 - 最佳参数配置说明

## 概述

本实验旨在通过**嵌套交叉验证**的方式，比较 MetaGB 和 XGBoost 在多个真实回归数据集上的性能。

### 实验特点
- **无早停停止**：使用固定的树数量进行训练，不考虑早停
- **网格搜索**：通过网格搜索寻找最佳超参数组合
- **并行加速**：使用多进程并行加速参数搜索
- **固定数据划分**：使用80/20划分（RANDOM_SEED=42）进行评估

---

## 实验流程

### 1. 数据划分
- 使用固定随机种子（RANDOM_SEED=42）将数据集分为 80% 训练集 / 20% 测试集
- 每个数据集只进行一次划分（不是交叉验证）

### 2. 内层交叉验证（4折）
- 在80%训练集上进行4折交叉验证
- 对每个参数组合进行评估
- 选择平均RMSE最低的参数组合

### 3. 模型训练
- 使用最佳参数在完整80%训练集上重新训练
- 不考虑早停，使用固定的树数量（`n_estimators`）

### 4. 性能评估
- 在20%测试集上评估模型性能
- 计算RMSE、MAE、R²等指标
- 直接输出单次评估结果（无标准差）

---

## 实验配置

### 数据集

实验使用TabArena回归数据集中的11个数据集：

| 数据集 | 样本数 | 特征数 | 说明 |
|--------|--------|--------|------|
| airfoil_self_noise | - | - | 翼型自噪声数据 |
| Another-Dataset-on-used-Fiat-500 | - | - | 二手菲亚特500数据 |
| concrete_compressive_strength | - | - | 混凝土抗压强度 |
| diamonds | - | - | 钻石价格数据 |
| Food_Delivery_Time | - | - | 餐饮配送时间 |
| healthcare_insurance_expenses | - | - | 医疗保险费用 |
| houses | - | - | 房屋价格数据 |
| miami_housing | - | - | 迈阿密住房数据 |
| physiochemical_protein | - | - | 生理化学蛋白质数据 |
| QSAR_fish_toxicity | - | - | 鱼类毒性QSAR |
| wine_quality | - | - | 葡萄酒质量 |

### 全局超参数

```python
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8  # 80%训练/20%测试（固定划分，非交叉验证）
INNER_CV_FOLDS = 4  # 内层4折交叉验证
CATEGORICAL_HANDLING = 'drop'  # 删除分类特征
```

---

## 超参数搜索空间

### 共享参数（所有模型）

```python
EXP1_SHARED_PARAMS_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],  # 学习率
    'max_depth': [3, 4, 6, 8],  # 树的最大深度
    'n_estimators': [200, 400, 600, 800],  # 树的数量
    'max_bin': [255],  # 最大分箱数
}
```

### 数据集特定参数

所有数据集使用相同的参数空间：

```python
DATASET_SPECIFIC_COMMON_GRIDS = {
    '所有数据集': {
        'subsample': [0.7, 0.8, 0.9, 1.0],  # 样本采样比例
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # 特征采样比例
    }
}
```

### XGBoost特有参数

```python
XGBOOST_PARAMS = {
    'min_child_weight': [1, 10],  # 子节点最小样本数
    'reg_lambda': [0.0, 1.0],  # L2正则化系数
}
```

### MetaGB特有参数

```python
METAGB_PARAMS = {
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 预测特征采样比例
    'version': [3],  # MetaGB版本
    'min_data_in_leaf': [1, 10],  # 叶子节点最小样本数
}
```

---

## 超参数说明

### 核心参数详解

#### 1. subsample（样本采样比例）
- **作用**：控制每棵树使用的样本比例
- **范围**：0.7, 0.8, 0.9, 1.0
- **含义**：
  - 1.0：使用所有样本
  - 0.8：随机选择80%的样本
- **目的**：防止过拟合，增加模型多样性

#### 2. colsample_bytree（特征采样比例）
- **作用**：控制每棵树使用的特征比例
- **范围**：0.7, 0.8, 0.9, 1.0
- **含义**：
  - 1.0：使用所有特征
  - 0.8：随机选择80%的特征
- **目的**：
  - 对于XGBoost：统一作用于所有特征
  - 对于MetaGB：只作用于**条件特征**（分裂决策特征）

#### 3. pred_feature_fraction（MetaGB预测特征采样比例）
- **作用**：控制MetaGB叶子节点预测时使用的特征比例
- **范围**：0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- **含义**：
  - 1.0：使用所有候选特征进行预测
  - 0.6：随机选择60%的特征作为候选预测特征
- **目的**：减少叶子节点的计算复杂度，提高预测效率

#### 4. learning_rate（学习率）
- **作用**：每棵树的预测结果的缩放系数
- **范围**：0.01, 0.05, 0.1
- **含义**：
  - 较小的学习率（0.01）需要更多的树才能收敛
  - 较大的学习率（0.1）可以更快收敛但可能过拟合
- **目的**：控制模型的收敛速度和最终性能

#### 5. max_depth（树的最大深度）
- **作用**：限制每棵树的最大深度
- **范围**：3, 4, 6, 8
- **含义**：
  - 较深的树（8）可以拟合更复杂的模式
  - 较浅的树（3）更简单，不易过拟合
- **目的**：控制模型的复杂度

#### 6. n_estimators（树的数量）
- **作用**：梯度提升中树的数量
- **范围**：200, 400, 600, 800
- **含义**：
  - 由于不考虑早停，这些是固定的树数量
  - 较多的树可以提升性能但增加训练时间
- **目的**：控制模型的容量

#### 7. min_child_weight / min_data_in_leaf（最小叶节点数）
- **作用**：限制叶子节点的最小样本数
- **XGBoost**：`min_child_weight` [1, 10]
- **MetaGB**：`min_data_in_leaf` [1, 10]
- **目的**：防止过拟合，避免树过于复杂

#### 8. max_bin（最大分箱数）
- **作用**：连续特征离散化的最大分箱数
- **范围**：255（固定）
- **目的**：提高训练速度，减少内存使用

#### 9. reg_lambda（L2正则化）
- **作用**：叶节点权重的L2正则化系数
- **范围**：[0.0, 1.0]
- **说明**：仅XGBoost使用此参数
- **目的**：防止叶节点权重过大

#### 10. version（MetaGB版本）
- **作用**：MetaGB的算法版本
- **范围**：[3]（固定）
- **说明**：仅MetaGB使用此参数

---

## 参数搜索策略

### 参数组合数量

对于每个数据集和每个模型：

#### XGBoost参数组合
```
EXP1_SHARED_PARAMS_GRID:
  - learning_rate: [0.01, 0.05, 0.1] → 3
  - max_depth: [3, 4, 6, 8] → 4
  - n_estimators: [200, 400, 600, 800] → 4
  - max_bin: [255] → 1

DATASET_SPECIFIC_COMMON_GRIDS:
  - subsample: [0.7, 0.8, 0.9, 1.0] → 4
  - colsample_bytree: [0.7, 0.8, 0.9, 1.0] → 4

DATASET_SPECIFIC_XGBOOST_GRIDS:
  - reg_lambda: [0.0, 1.0] → 2

总计: 3 × 4 × 4 × 1 × 4 × 4 × 2 = 1536 组合
```

#### MetaGB参数组合
```
EXP1_SHARED_PARAMS_GRID:
  - learning_rate: [0.01, 0.05, 0.1] → 3
  - max_depth: [3, 4, 6, 8] → 4
  - n_estimators: [200, 400, 600, 800] → 4
  - max_bin: [255] → 1

DATASET_SPECIFIC_COMMON_GRIDS:
  - subsample: [0.7, 0.8, 0.9, 1.0] → 4
  - colsample_bytree: [0.7, 0.8, 0.9, 1.0] → 4

DATASET_SPECIFIC_METAGB_GRIDS:
  - pred_feature_fraction: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] → 6
  - version: [3] → 1

总计: 3 × 4 × 4 × 1 × 4 × 4 × 6 × 1 = 4608 组合
```

**注意**：`min_data_in_leaf` 和 `min_child_weight` 固定为 `[10]`，不参与参数搜索

### 优化策略

1. **并行化**：使用 `N_JOBS = -1` 启用所有CPU核心进行并行搜索
2. **网格搜索**：采用完整的网格搜索，不使用随机搜索
3. **参数过滤**：在参数搜索时自动过滤不相关的参数
   - XGBoost：不包含 MetaGB 特有参数
   - MetaGB：不包含 XGBoost 特有参数

---

## 启动步骤

### 步骤1：安装依赖

```bash
# 安装Python依赖
pip install -r Exp_New/metagb/requirements.txt

# 安装编译工具（如果未安装）
sudo apt-get update
sudo apt-get install python3-dev gcc build-essential

# 安装CFFI（如果未安装）
pip install cffi
```

### 步骤2：编译MetaGB

```bash
# 进入metagb目录
cd Exp_New/metagb

# 运行编译脚本
python build_metagb.py

# 编译成功后会生成以下文件：
# - libmetagb.o (目标文件)
# - libmetagb.cpython-38-x86_64-linux-gnu.so (共享库)
```

### 步骤3：复制编译文件

```bash
# 将编译生成的.o和.so文件复制到Exp_New根目录
# 确保以下文件存在并复制：
cp libmetagb.o ../
cp libmetagb.cpython-38-x86_64-linux-gnu.so ../
```

### 步骤4：准备数据集

确保数据集已放置在以下位置：
```
Exp_New/dataset/tabArena_regression/
- airfoil_self_noise.csv
- Another-Dataset-on-used-Fiat-500.csv
- concrete_compressive_strength.csv
- ... (其他数据集)
```

### 步骤5：创建日志目录

```bash
# 创建日志目录（如果不存在）
mkdir -p Exp_New/logs
```

### 步骤6：启动实验

```bash
# 返回Exp_New根目录
cd Exp_New

# 在后台运行实验，并记录日志
nohup python exp1_bestparams_noEarly.py \
  > logs/exp1_bestparams_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看进程ID
echo $!

# 查看实时日志
tail -f logs/exp1_bestparams_$(date +%Y%m%d_%H%M%S).log
```

### 步骤7：监控实验进度

```bash
# 查看运行中的Python进程
ps aux | grep exp1_bestparams_noEarly.py

# 查看日志文件
ls -lht Exp_New/logs/exp1_bestparams_*.log

# 查看最新的日志
tail -100 Exp_New/logs/exp1_bestparams_*.log
```

---

## 输出结果

### 日志文件

实验日志保存在：
```
Exp_New/logs/exp1_bestparams_YYYYMMDD_HHMMSS.log
```

日志内容包括：
- 每个数据集的进度
- 参数搜索过程
- 每个参数组合的性能
- 最佳参数选择
- 训练和测试结果

### 结果文件

#### 1. 数据集级别结果
```
Exp_New/logs/exp1_bestparams/{dataset_name}/
- exp1_bestparams_metagb.json - MetaGB最佳参数和结果
- exp1_bestparams_xgboost.json - XGBoost最佳参数和结果
- exp1_bestparams_metagb.log - MetaGB详细日志
- exp1_bestparams_xgboost.log - XGBoost详细日志
```

#### 2. 汇总结果
```
Exp_New/results/exp1_bestparams/
- exp1_bestparams_summary.csv - 所有数据集的性能汇总
```

### JSON文件格式

每个数据集的结果JSON包含：
```json
{
  "dataset": "数据集名称",
  "model": "模型类型 (metagb/xgboost)",
  "best_params": {
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 400,
    ...
  },
  "test_metrics": {
    "rmse": 0.123,
    "mae": 0.098,
    "r2": 0.876
  },
  "cv_metrics": {
    "mean_rmse": 0.115,
    "std_rmse": 0.008
  }
}
```

---

## 元素说明

### MetaGB 核心概念

MetaGB将特征分为两类：
1. **条件特征（Cond Features）**：用于树的分裂决策
2. **预测特征（Pred Features）**：用于叶子节点的预测

这种设计使MetaGB能够：
- 捕捉更复杂的非线性关系
- 保持比传统GBDT更快的训练速度
- 在叶子节点使用线性模型（特征×权重）

### 参数映射

XGBoost参数到MetaGB参数的映射：
```python
'n_estimators' → 'num_trees'
'colsample_bytree' → 'feature_fraction'  # 仅影响条件特征
'subsample' → 'bagging_fraction'
```

---

## 常见问题

### Q1: 编译失败怎么办？
**A**: 请检查是否安装了必要的依赖：
```bash
sudo apt-get install python3-dev gcc build-essential
pip install cffi
```

### Q2: 如何调整超参数搜索空间？
**A**: 编辑 `Exp_New/utils/hyperparams.py` 文件，修改相应的参数网格

### Q3: 如何只运行部分数据集？
**A**: 修改 `hyperparams.py` 中的 `EXP1_DATASETS` 列表

### Q4: 如何调整并行度？
**A**: 修改 `exp1_bestparams_noEarly.py` 中的 `N_JOBS` 参数
- `-1`：使用所有CPU核心
- `1`：顺序执行
- `N`：使用N个CPU核心

### Q5: 实验运行时间过长怎么办？
**A**: 可以：
1. 减少数据集数量
2. 缩小参数搜索空间
3. 减少交叉验证折数
4. 增加并行度

---

## 实验监控

### 检查进度

```bash
# 查看日志文件大小（增长表示正在运行）
du -h Exp_New/logs/exp1_bestparams_*.log

# 查看当前处理的数据集
grep "正在处理数据集" Exp_New/logs/exp1_bestparams_*.log

# 查看当前处理的参数组合
grep "参数组合" Exp_New/logs/exp1_bestparams_*.log | tail -20
```

### 终止实验

```bash
# 查看进程ID
ps aux | grep exp1_bestparams_noEarly.py

# 终止进程
kill <进程ID>

# 强制终止（如果正常终止无效）
kill -9 <进程ID>
pkill -9 python
```
---

---

## 参考文献

- XGBoost: https://xgboost.readthedocs.io/
- MetaGB: 详见项目文档和论文

---

## 联系方式

如有问题，请查看：
- `Exp_New/metagb/README.md` - MetaGB使用说明
- `Exp_New/metagb/编译步骤.md` - 编译说明
- `Exp_New/utils/` - 工具模块说明

---

## 版本历史

- **v1.0** (2025-03-19): 初始版本，支持无早停的网格搜索实验
