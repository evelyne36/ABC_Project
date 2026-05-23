# MetaGB-XGBoost：广义梯度提升树

将 **MetaGB（Meta-Gradient Boosting）** 方法嵌入到 XGBoost 官方源码中，通过分离"条件特征"和"预测特征"实现比标准梯度提升更具表达力的树模型。当前嵌入版已支持 CPU hist 路径下的回归、二分类和多分类任务。

---

## 目录

- [背景与动机](#背景与动机)
- [核心思想](#核心思想)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [参数说明](#参数说明)
- [实验与基准](#实验与基准)
- [技术实现细节](#技术实现细节)
- [编译构建](#编译构建)
- [当前限制](#当前限制)
- [开发路线](#开发路线)

---

## 背景与动机

### 标准 XGBoost 的局限

在标准梯度提升树（如 XGBoost）中，每个叶节点输出一个**常数预测值** $w_j$：

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

其中 $g_i$ 和 $h_i$ 分别是损失函数的一阶和二阶梯度。这意味着叶节点的预测能力被限制为一个标量。

### 为什么需要"条件-预测分离"

考虑经典的 XOR 问题：

```
y-axis
  ^
  | x o
  | o x
  +-------> x-axis
```

无论用 x 轴还是 y 轴单独分裂，都无法有效分类。标准 XGBoost 需要两层树才能捕捉这种交互：

```
      x < 1
      /   \
   y<1     y<1
   / \     / \
  o   x   x   o
```

但如果允许叶节点使用特征值作为预测函数，**只需一层**：

```
    x < 1
    /   \
   y    1-y
```

这就是 MetaGB 的核心动机。

---

## 核心思想

MetaGB 将树节点中的变量分为两种角色：

| 角色 | 作用 | 示例 |
|------|------|------|
| **条件特征（Conditional）** | 用于决定节点分裂的阈值 | 类别型、低基数码值型 |
| **预测特征（Predictive）** | 用于计算叶节点的线性预测值 | 连续型、高基数码值型 |

### 广义叶节点预测

标准 XGBoost 的叶节点输出为常数 $w_j$，而 MetaGB 将其推广为**线性函数**：

$$f_t(x_i) = \beta \cdot x_{ij}$$

其中 $x_{ij}$ 是样本 $x_i$ 的第 $j$ 个特征值。最优系数为：

$$\beta_j^* = -\frac{\sum_{i \in I_j} g_i \cdot x_{ij}}{\sum_{i \in I_j} h_i \cdot x_{ij}^2 + \lambda}$$

### 向后兼容

通过在输入特征中追加一列 **bias 特征**（恒等于 1），当 MetaGB 选择 bias 作为预测特征时，叶节点预测自动退化为常数预测，即退化为标准 XGBoost 行为。

### 分裂搜索过程

对每个待分裂节点，MetaGB：

1. 采样一组条件特征和预测特征
2. 对每个三元组 `(条件特征 ci, 预测特征 pi, 分裂阈值 bin)` 计算增益：
   ```
   Gx  = Σ(g_i × x_i[pi])
   Hxx = Σ(h_i × x_i[pi]²)
   score = Gx² / (Hxx + λ + ε)
   ```
3. 选择增益最大的组合执行分裂
4. 叶节点预测公式：`prediction += learning_rate × x[pred_feature_idx] × leaf_weight`

---

## 项目结构

```
MyProject_MGBexplore/
├── README.md                              # 本文件
├── MetaGB嵌入XGBoost_CPU实践方案.md        # 详细技术方案文档
├── MetaGB分类扩展实践路线.md               # 分类扩展的设计路线与完成度
├── MetaGB分类任务支持说明.md               # 分类任务支持边界与方法机理说明
│
├── xgboost-master/                        # 修改后的 XGBoost 源码（核心）
│   ├── src/tree/
│   │   ├── param.h                        # 新增 MetaGB 参数定义
│   │   ├── updater_metagb_hist.cc         # MetaGB 专用直方图更新器（新增）
│   │   ├── tree_model.h / tree_model.cc   # 扩展叶节点元信息
│   │   └── ...
│   ├── src/predictor/cpu_predictor.cc     # 修改：支持 MetaGB 预测路径
│   ├── src/gbm/gbtree.cc                  # 修改：支持 MetaGB updater 选择
│   ├── tests/python/test_metagb.py        # MetaGB 专项测试
│   └── python-package/xgboost/            # Python API（参数透传）
│
├── Exp1_54CV_ordinal/                     # 54 组超参数交叉验证实验
│   ├── exp1_200params_ordinal.py          # 实验主脚本
│   ├── configs_xgboost.json               # 超参数配置
│   ├── metagb/                            # 独立 MetaGB 库（Python + C/CFFI）
│   │   ├── metagb.c / metagb.h            # MetaGB C 核心实现
│   │   ├── metagb_utils.py                # Python 封装
│   │   ├── build_metagb.py                # 编译脚本
│   │   └── README.md                      # 独立版详细文档
│   └── dataset/
│       ├── tabArena_regression/           # TabArena 回归数据集
│       ├── tabArena_binary/               # TabArena 二分类数据集
│       └── tabArena_multiclass/           # TabArena 多分类数据集
│
├── Exp_New/                               # 新实验框架
│   ├── exp1_200params_ordinal.py          # 批量实验脚本
│   ├── utils/                             # 实验工具
│   │   ├── hyperparams.py                 # 超参数管理
│   │   ├── trainer.py                     # 训练器
│   │   ├── data_generators.py             # 合成数据生成
│   │   └── visualizer.py                  # 可视化
│   ├── data/                              # 合成实验数据
│   └── metagb/                            # MetaGB 库副本
│
├── metagb_原/                             # MetaGB 原始独立实现（备份）
│
├── compare_current_metagb_xgbcpu_nested_speed.py  # 回归：MetaGB vs XGBoost 嵌套CV对比
├── compare_current_metagb_xgbcpu_classification_nested_speed.py  # 分类：MetaGB vs XGBoost 嵌套CV对比
│
├── results_current_metagb_vs_xgbcpu/      # 回归对比实验结果
└── results_current_metagb_vs_xgbcpu_classification/  # 分类对比实验结果
```

---

## 快速开始

### 环境要求

- **操作系统**: Linux（当前实现基于 Linux 编译测试）
- **编译器**: GCC（支持 C++17）
- **构建工具**: CMake ≥ 3.15
- **Python**: 3.8+
- **依赖**: NumPy, pandas, scikit-learn, pytest

### 编译与安装

```bash
# 1. 进入修改后的 XGBoost 目录
cd xgboost-master

# 2. 创建构建目录并编译
mkdir -p build && cd build
cmake .. -DUSE_OPENMP=ON
cmake --build . -j$(nproc)

# 3. 安装 Python 包
cd ../python-package
pip install -e .
```

### 验证安装

```bash
# 运行 MetaGB 专项测试
PYTHONPATH=python-package python -m pytest tests/python/test_metagb.py -v
```

预期输出：13 个测试全部通过，覆盖回归、二分类、多分类、概率预测和模型保存/加载一致性。

---

## 使用方法

### 最简示例

#### 回归

```python
import xgboost as xgb
import numpy as np

# 生成示例数据
X = np.random.randn(500, 6).astype(np.float32)
y = np.where(X[:, 0] > 0, X[:, 1], -X[:, 1]).astype(np.float32)

# 使用 MetaGB 训练
model = xgb.XGBRegressor(
    tree_method="hist",
    objective="reg:squarederror",
    enable_metagb=True,             # 启用 MetaGB
    metagb_score_agg=0,             # 分裂搜索策略
    metagb_lambda_l2=0.0,           # L2 正则化
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
)

model.fit(X, y)
predictions = model.predict(X)
```

#### 二分类

二分类沿用 XGBoost 的 raw margin 训练机制。MetaGB 叶节点输出仍是 `w * x[pred_feature]`，累加到 margin 后由 `binary:logistic` 转为概率。

```python
import xgboost as xgb
import numpy as np

X = np.random.randn(500, 6).astype(np.float32)
margin = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3]
y = (margin > np.median(margin)).astype(np.int32)

model = xgb.XGBClassifier(
    tree_method="hist",
    objective="binary:logistic",    # 也支持 binary:logitraw
    enable_metagb=True,
    metagb_score_agg=0,
    metagb_lambda_l2=0.0,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
)

model.fit(X, y)
proba = model.predict_proba(X)
pred = model.predict(X)
```

#### 多分类

多分类第一版采用 XGBoost 的 class-wise scalar-tree 路径：每个 class 独立训练一棵 MetaGB scalar tree，分别更新对应 class 的 margin。暂未实现 vector leaf。

```python
import xgboost as xgb
import numpy as np

X = np.random.randn(600, 8).astype(np.float32)
scores = np.column_stack([
    X[:, 0] + 0.5 * X[:, 1],
    -X[:, 0] + X[:, 2],
    X[:, 3] - 0.25 * X[:, 4],
])
y = np.argmax(scores, axis=1).astype(np.int32)

model = xgb.XGBClassifier(
    tree_method="hist",
    objective="multi:softprob",     # 也支持 multi:softmax
    num_class=3,
    enable_metagb=True,
    metagb_score_agg=0,
    metagb_lambda_l2=0.0,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
)

model.fit(X, y)
proba = model.predict_proba(X)
pred = model.predict(X)
```

### 模型保存与加载

```python
# 保存模型
model.save_model("metagb_model.json")

# 加载模型（自动识别 MetaGB 模式；分类任务可使用 XGBClassifier）
loaded_model = xgb.XGBRegressor()
loaded_model.load_model("metagb_model.json")
predictions = loaded_model.predict(X)
```

### 不启用 MetaGB（标准 XGBoost 行为）

```python
# enable_metagb 默认为 False，行为完全等同于标准 XGBoost
model = xgb.XGBRegressor(
    tree_method="hist",
    objective="reg:squarederror",
    # 不设置 enable_metagb 或设为 False
)
```

---

## 参数说明

### MetaGB 专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_metagb` | bool | `False` | 是否启用 MetaGB 分裂模式 |
| `metagb_pred_colsample_bytree` | float | `1.0` | 每棵树预测特征的子采样比例（0~1） |
| `metagb_cond_pred_share` | bool | `False` | 为 `True` 时，预测特征由条件特征派生并追加 bias |
| `metagb_leaf_only_use_bias` | bool | `False` | 为 `True` 时，叶节点仅使用 bias 特征（退化为常数叶预测） |
| `metagb_score_agg` | int | `0` | 分裂评分聚合策略，取值 0~3（见下表） |
| `metagb_lambda_l2` | float | `0.0` | 叶节点线性系数的 L2 正则化（注意：默认为 0，不同于 XGBoost 的 `reg_lambda=1`） |
| `metagb_min_gain_to_split` | float | `-1e10` | 分裂所需的最小增益 |

### score_agg 分裂搜索策略

| 值 | 策略 | 说明 |
|----|------|------|
| 0 | **全局最优** | 在所有 `(条件特征, 预测特征, bin)` 中取增益最大值，最贪心 |
| 1 | **预测取平均** | 对同一 `(条件特征, bin)` 的所有预测特征得分取平均，再选最佳 bin |
| 2 | **预测取中位数** | 对预测特征得分取中位数，更鲁棒 |
| 3 | **随机预测** | 随机选择一个预测特征的得分，增加多样性 |

---

## 实验与基准

### 数据集

本项目使用 TabArena 风格数据集进行评估。当前工程包含回归、二分类和多分类三类任务。

回归数据集位于：

```text
Exp1_54CV_ordinal/dataset/tabArena_regression/
```

| 数据集 | 领域 | 规模 |
|--------|------|------|
| airfoil_self_noise | 声学 | ~1500 |
| concrete_compressive_strength | 材料 | ~1000 |
| diamonds | 商业 | ~54000 |
| wine_quality | 食品 | ~6500 |
| healthcare_insurance | 医疗 | ~1300 |
| ... | ... | ... |

分类数据集位于：

```text
Exp1_54CV_ordinal/dataset/tabArena_binary/diabetes.csv
Exp1_54CV_ordinal/dataset/tabArena_multiclass/anneal.csv
```

### 运行对比实验

```bash
# 运行 MetaGB vs XGBoost CPU 嵌套交叉验证对比
python compare_current_metagb_xgbcpu_nested_speed.py
```

实验结果保存在 `results_current_metagb_vs_xgbcpu/` 目录中。

分类任务：

```bash
# 二分类 + 多分类，默认 5 折外层 / 4 折内层嵌套交叉验证
python compare_current_metagb_xgbcpu_classification_nested_speed.py --mode full

# 快速 smoke test：只跑一个外层折和少量候选参数
python compare_current_metagb_xgbcpu_classification_nested_speed.py \
  --mode quick \
  --datasets diabetes.csv anneal.csv \
  --max-candidates 5 \
  --outer-fold-index 0 \
  --n-jobs 1
```

分类脚本的指标口径：

| 任务 | 默认 objective | 主指标 | 选择方向 | 辅助指标 |
|------|----------------|--------|----------|----------|
| 二分类 | `binary:logistic` | `AUC` | 最大化 | `logloss`, `accuracy`, `balanced_accuracy` |
| 多分类 | `multi:softprob` | `logloss` | 最小化 | `accuracy`, `balanced_accuracy` |

输出保存在 `results_current_metagb_vs_xgbcpu_classification/`，主要文件包括：

```text
current_metagb_vs_xgbcpu_classification_nested_candidates.csv
current_metagb_vs_xgbcpu_classification_nested_summary.csv
current_metagb_vs_xgbcpu_classification_nested_aggregate.csv
current_metagb_vs_xgbcpu_classification_nested_metadata.json
```

其中 `summary.csv` 包含 `primary_metric`、`primary_metric_direction`、`val_primary_metric_mean` 和 `test_primary_metric`，便于统一读取 binary 的 AUC 与 multiclass 的 logloss。

### 合成数据实验

项目包含专门的合成数据生成器，用于验证 MetaGB 在以下场景中的优势：

- **XOR 型交互数据**：特征乘积关系
- **分段线性数据**：不同区域有不同斜率
- **条件-预测分离数据**：类别特征定义区域，连续特征定义预测
- **高维稀疏交互数据**：大量无关特征中的少量重要交互

---

## 技术实现细节

### XGBoost 源码修改清单

本项目通过修改 XGBoost 官方源码实现 MetaGB 集成，遵循**最小侵入**原则：

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| [`src/tree/param.h`](xgboost-master/src/tree/param.h) | 修改 | 新增 6 个 MetaGB 参数声明与注册 |
| [`src/tree/updater_metagb_hist.cc`](xgboost-master/src/tree/updater_metagb_hist.cc) | **新增** | MetaGB 专用 CPU 直方图树构建器（~640行） |
| [`include/xgboost/tree_model.h`](xgboost-master/include/xgboost/tree_model.h) | 修改 | 旁路数组 `metagb_leaf_features_` 存储预测特征索引 |
| [`src/tree/tree_model.cc`](xgboost-master/src/tree/tree_model.cc) | 修改 | 模型序列化/反序列化支持 MetaGB 元信息 |
| [`src/tree/tree_view.h`](xgboost-master/src/tree/tree_view.h) | 修改 | CPU tree view 暴露 MetaGB 叶节点信息 |
| [`src/predictor/cpu_predictor.cc`](xgboost-master/src/predictor/cpu_predictor.cc) | 修改 | 预测路径：`leaf_output = leaf_weight × x[pred_feature_idx]` |
| [`src/gbm/gbtree.cc`](xgboost-master/src/gbm/gbtree.cc) | 修改 | GBTree updater 选择逻辑，`enable_metagb=true` 时使用 `grow_metagb_histmaker` |

### 设计决策

1. **旁路数组而非修改 Node 结构**：MetaGB 叶节点元信息通过 `metagb_leaf_features_` 旁路数组存储，避免影响 XGBoost 原有 `RegTree::Node` 的 ABI 兼容性和 GPU 相关代码。

2. **独立 updater 而非修改现有 updater**：新增 `MetaGBHistMaker` 注册为 `grow_metagb_histmaker`，不影响标准 `grow_quantile_histmaker` 路径。

3. **默认关闭**：`enable_metagb` 默认为 `False`，不影响任何现有 XGBoost 功能。

### 数据流

```
训练流程:
  XGBRegressor.fit() / XGBClassifier.fit()
    → GBTree::Update()
      → MetaGBHistMaker::Update()
        → 采样条件/预测特征
        → 构建节点级直方图 (Gx, Hxx, count)
        → 枚举 (cond, pred, bin) 搜索最佳分裂
        → 应用分裂 + 设置叶节点 pred_feature_idx
        → UpdatePredictionCache: pred += lr × x[pred_feature] × weight

预测流程:
  XGBRegressor.predict() / XGBClassifier.predict_proba()
    → CPUPredictor
      → 遍历每棵树
        → 对每个样本: leaf_output = weight × x[pred_feature_idx]
```

分类任务不会在 MetaGB updater 中手写交叉熵，而是复用 XGBoost objective 已经计算好的 `grad/hess`。MetaGB 改变的是叶节点函数和 split 统计量：

```text
普通 XGBoost:  leaf = w
MetaGB:        leaf = w * x[pred_feature]
```

二分类仍是单输出 margin：

```text
margin += tree(x)
prob = sigmoid(margin)
```

多分类当前走 class-wise scalar tree：

```text
for class k in 0..K-1:
    使用第 k 类 grad/hess 建一棵 MetaGB tree
    只更新 margin_k
```

---

## 编译构建

### 前置依赖

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install build-essential cmake python3-dev

# Python 依赖
pip install numpy pandas scikit-learn pytest
```

### 构建步骤

```bash
cd xgboost-master

# 配置
mkdir -p build && cd build
cmake .. -DUSE_OPENMP=ON

# 编译（-j 后跟 CPU 核数）
cmake --build . -j2

# 回到项目根目录安装 Python 包
cd ../python-package
pip install -e .

# 验证
python -c "import xgboost; print(xgboost.__version__)"
```

### 运行测试

```bash
# 从 xgboost-master 根目录运行
PYTHONPATH=python-package python -m pytest tests/python/test_metagb.py -v
```

### 独立 MetaGB 库编译（可选）

如果需要使用独立版 MetaGB（不依赖 XGBoost）：

```bash
cd Exp1_54CV_ordinal/metagb   # 或 Exp_New/metagb
pip install cffi
python build_metagb.py
python -c "import libmetagb; print('编译成功')"
```

---

## 当前限制

当前实现的目标是**功能正确性优先**，存在以下限制：

| 限制项 | 说明 |
|--------|------|
| **仅 CPU** | 不支持 GPU hist 路径 |
| **任务范围** | 已支持 `reg:squarederror`、`binary:logistic`、`binary:logitraw`、`multi:softprob`、`multi:softmax`；暂不支持排序、生存分析、多标签、多输出回归等任务 |
| **多分类叶结构** | 当前为 class-wise scalar MetaGB tree；尚未实现 vector leaf 多分类 |
| **缺失值** | 缺失预测特征按 `0` 处理 |
| **采样粒度** | 每棵树采样一次条件/预测特征集合，暂未实现节点级重新采样 |
| **停止条件** | 使用 XGBoost 的 `min_child_weight` Hessian 阈值，未完全复刻原 MetaGB 的 `min_data_in_leaf` 计数语义 |
| **性能** | 训练瓶颈主要来自 `条件特征 × 预测特征 × bin` 的三维搜索；尚未做充分 SIMD/OpenMP/缓存优化 |

---

## 开发路线

- [x] **阶段 0**: 基线确认 — XGBoost 编译通过，MetaGB v3 独立运行
- [x] **阶段 1**: 参数透传 — MetaGB 参数到达 C++ updater
- [x] **阶段 2**: 模型元信息 — 叶节点 `pred_feature_idx` 的存储/加载/复制
- [x] **阶段 3**: score_agg=0 分裂搜索 — 基础 MetaGB 分裂逻辑
- [x] **阶段 4**: 预测路径 — CPU predictor + prediction cache
- [x] **阶段 5**: score_agg=1/2/3 — 补齐全部聚合策略
- [x] **阶段 6**: 分类任务支持 — 二分类与 class-wise scalar-tree 多分类
- [ ] **阶段 7**: 性能优化 — 直方图缓存、OpenMP 并行、SIMD
- [ ] **阶段 8**: vector leaf 多分类
- [ ] **阶段 9**: GPU 支持
- [ ] **阶段 10**: SHAP 解释支持

---

## 参考文献

- Tianqi Chen and Carlos Guestrin. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). KDD 2016.
- [TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2505.22252).

---

## 许可证

XGBoost 部分遵循 Apache-2.0 许可证。MetaGB 部分为本项目原创实现。
