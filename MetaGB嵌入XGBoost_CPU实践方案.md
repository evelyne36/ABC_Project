# MetaGB 嵌入 XGBoost CPU 版实践方案

日期：2026-05-20

目标工程：`/Users/ancd/VSCodeProject/MyProject_MGBexplore/xgboost-master`

MetaGB 源码参考：`/Users/ancd/VSCodeProject/MyProject_MGBexplore/Exp1_54CV_ordinal/metagb`

## 1. 目标与边界

本方案的第一目标是将现有 MetaGB 方法完整迁移到 CPU 版本 XGBoost 官方源码工程中，而不是继续修改原有 `metagb` 自建 C/CFFI 库。

第一版实现范围：

- 迁移 MetaGB `version=3`，即当前 `train_simple` / `build_tree_double_simple` 逻辑。
- 只支持回归平方误差任务，对应当前 MetaGB 的 `grad = yhat - y`、`hessian = 1` 语义。
- 第一阶段已完整实现 `score_agg=0/1/2/3`。
- 保持 `lambda_l2` 默认值为 `0`，不直接复用 XGBoost 默认 `reg_lambda=1`。
- 保留 bias 预测特征，使 MetaGB 能退化为常数叶预测。
- 第一版只做 CPU hist 路径，不涉及 GPU、R/JVM 包、多目标树、分类任务和 SHAP 解释。

未来扩展注释：

- 分类任务可在 logit 空间复用线性叶预测形式，但需要重新确认目标函数、梯度/Hessian、base score、概率变换和多分类叶结构。
- `version=2` 的 prior feature、depth separation、split sample 等逻辑可作为第二阶段后续扩展，不进入第一版。

## 2. MetaGB v3 语义摘要

当前 MetaGB v3 的核心入口在：

- `Exp1_54CV_ordinal/metagb/metagb_utils.py`
- `Exp1_54CV_ordinal/metagb/metagb.c`

训练时 Python 侧会给输入特征追加一列 bias 特征 `1`。C 侧每棵树采样条件特征和预测特征：

- `cond_feature_idx`：用于节点分裂。
- `pred_feature_idx`：用于叶节点预测。
- `cond_pred_share=0` 时，两者独立采样。
- `cond_pred_share=1` 时，预测特征由条件特征派生，并追加 bias。

在每个节点中：

1. 先在当前节点样本上计算每个预测特征作为叶预测时的得分。
2. 若达到最大深度或样本不足，则叶节点存储最佳 `pred_feature_idx` 和 `leaf_weight`。
3. 若继续分裂，则枚举 `(cond_feature, pred_feature, bin)`。
4. 对每个候选 split，按预测特征计算：

   ```text
   Gx = sum(g_i * x_i[pred_feature])
   Hxx = sum(h_i * x_i[pred_feature]^2)
   score = Gx^2 / (Hxx + lambda_l2 + eps)
   weight = -Gx / (Hxx + lambda_l2 + eps)
   ```

5. 选择最佳条件特征、预测特征和阈值。
6. 叶节点预测：

   ```text
   prediction += learning_rate * x[pred_feature_idx] * leaf_weight
   ```

这意味着完整迁移不能只给 XGBoost 增加“线性叶子”概念，还必须保留“分裂条件特征”和“叶预测特征”分离的搜索过程。

## 3. XGBoost 工程中需要关注的最小文件集合

为了避免在整体工程中迷失，第一版只关注以下文件或新增邻近文件。

### 3.1 树训练主链路

文件：

- `xgboost-master/src/tree/updater_quantile_hist.cc`

作用：

- CPU hist 单目标树构建主入口。
- `HistUpdater` 管理初始化、直方图构建、分裂评估、分裂应用、叶节点位置缓存。
- `QuantileHistMaker` 注册为 `grow_quantile_histmaker`。

建议：

- 不直接大改现有 `HistUpdater`。
- 新增独立 `MetaGBHistUpdater` 或 `updater_metagb_hist.cc`。
- 仅在 `enable_metagb=true` 时启用 MetaGB 路径，否则标准 XGBoost 行为不变。

### 3.2 分裂评估

文件：

- `xgboost-master/src/tree/hist/evaluate_splits.h`

作用：

- 当前 XGBoost hist 分裂搜索只枚举 split feature 和 bin，叶值为常数。

建议：

- 新增 `metagb_evaluate_splits.h`，实现 MetaGB 自己的 evaluator。
- 第一阶段实现 `score_agg=0`：

  ```text
  best = max over all (cond_feature, pred_feature, bin)
  ```

- 第二阶段补：

  ```text
  score_agg=1: 对 pred 取平均，再对 bin 取最大
  score_agg=2: 对 pred 取中位数，再对 bin 取最大
  score_agg=3: 随机 pred，再对 bin 取最大
  ```

### 3.3 树模型结构

文件：

- `xgboost-master/include/xgboost/tree_model.h`
- `xgboost-master/src/tree/tree_model.cc`

当前问题：

- `RegTree::Node` 的 leaf 只保存一个 `leaf_value`。
- MetaGB 叶子还需要保存 `pred_feature_idx`。

建议有两个选择：

方案 A：扩展 `RegTree::Node`

- 给叶子增加 `pred_feature_idx` 和 MetaGB 标记。
- 优点：模型结构直观，预测路径容易统一。
- 风险：会影响模型序列化、兼容性、节点大小、GPU/多目标等路径。

方案 B：在 `RegTree` 旁路增加 MetaGB 叶子元信息

- 新增 `metagb_pred_feature_idx_` 数组，大小与节点数一致。
- 普通 XGBoost 节点保持不变。
- 只有 `enable_metagb=true` 的树使用该数组。
- 优点：对 XGBoost 现有节点结构侵入较小。
- 风险：需要补模型保存、加载、复制、预测缓存。

建议第一版采用方案 B，符合奥卡姆剃刀原则：减少对 `Node` ABI 和 GPU 相关代码的影响。

### 3.4 参数

文件：

- `xgboost-master/src/tree/param.h`
- 必要时新增 `xgboost-master/src/tree/metagb_param.h`

建议新增参数：

```text
enable_metagb: bool, default=false
metagb_pred_colsample_bytree: float, default=1.0
metagb_cond_pred_share: bool, default=false
metagb_leaf_only_use_bias: bool, default=false
metagb_score_agg: int, default=0
metagb_lambda_l2: float, default=0.0
```

说明：

- `metagb_lambda_l2` 默认必须保持 `0`，与当前 MetaGB 源码一致。
- 条件特征采样第一版可复用 XGBoost `colsample_bytree`。
- 行采样第一版可复用 XGBoost `subsample`，但需要确认随机种子导致的结果差异。

### 3.5 Python API

文件：

- `xgboost-master/python-package/xgboost/sklearn.py`
- 必要时补 `python-package/xgboost/core.py` 文档注释或参数白名单。

建议：

- 在 `XGBRegressor` / `XGBModel` 参数链路中透传新增 MetaGB 参数。
- 用户接口示例：

```python
model = xgboost.XGBRegressor(
    tree_method="hist",
    objective="reg:squarederror",
    enable_metagb=True,
    metagb_pred_colsample_bytree=0.8,
    metagb_lambda_l2=0.0,
    metagb_score_agg=0,
)
```

## 4. 不建议第一版修改的范围

第一版不碰以下区域：

- `src/tree/gpu_hist/*`
- `src/tree/updater_gpu_hist.cu`
- `include/xgboost/multi_target_tree_model.h`
- `R-package`
- `jvm-packages`
- `plugin`
- SHAP、dump、plotting 的完整解释支持
- categorical split 的复杂处理路径

如果第一版模型保存、加载和预测已经闭环，后续再补 dump/解释输出。

## 5. 推荐实现路线

### 阶段 0：基线确认

完成度：已完成基础工程验证，原 MetaGB 对照实验待补

工作内容：

- 已确认 `xgboost-master` 可用 CMake 编译。
- 已跑通 MetaGB 专项 Python 测试。
- 原始 MetaGB v3 小数据集预测对照和 benchmark 尚待补。

验收：

- 标准 XGBoost CPU hist 可训练和预测。
- MetaGB v3 原实现可训练和预测。
- 明确测试数据、参数和随机种子。

### 阶段 1：MetaGB 参数与 Python 透传

完成度：已完成第一版

工作内容：

- 已新增 MetaGB 参数到 `src/tree/param.h`。
- Python API 当前通过 sklearn 包装的 `**kwargs` 透传新增 native 参数。
- 默认 `enable_metagb=false` 时，所有现有行为不变。

验收：

- 不启用 MetaGB 时，现有 XGBoost 测试不应受影响。
- 启用 MetaGB 时参数能到达 C++ updater。

### 阶段 2：模型叶子元信息

完成度：已完成第一版

工作内容：

- 已在 `RegTree` 侧添加旁路 `metagb_leaf_features_`。
- 已支持设置叶节点 `pred_feature_idx`。
- 已补保存、加载、复制和 CPU tree view 暴露。

验收：

- 模型训练后，每个 MetaGB 叶节点都有合法 `pred_feature_idx`。
- bias 特征可被表示。
- 普通 XGBoost 模型加载与预测不受影响。

### 阶段 3：score_agg=0 的 MetaGB 分裂搜索

完成度：已完成第一版

工作内容：

- 已新增 `src/tree/updater_metagb_hist.cc`，注册 `grow_metagb_histmaker`。
- 已对每个节点枚举 `(cond_feature, pred_feature, bin)`。
- 使用 XGBoost quantile hist 的 cut/bin 信息。
- 计算 `sum(g*x)`、`sum(h*x*x)`、左右子节点 gain 和 leaf weight。
- 应用最佳分裂，左右叶节点保存同一个最佳 `pred_feature_idx` 对应的权重。

第一版可接受实现：

- 使用清晰、正确的节点级直方图缓存。
- 先优先可读性和一致性，再做 SIMD。
- OpenMP 并行可在 `(cond_feature, pred_feature)` 组合上做线程局部 best，再归并。

验收：

- 在小型 synthetic 数据上能训练。
- `metagb_leaf_only_use_bias=true` 时退化为常数叶预测。
- 与原 MetaGB v3 在同一参数下结果趋势一致。

### 阶段 4：预测路径和预测缓存

完成度：已完成第一版

工作内容：

- 已修改 CPU predictor 的普通预测路径：

  ```text
  leaf_output = leaf_weight * feature_value[pred_feature_idx]
  ```

- bias 特征取值固定为 `1.0`。
- 缺失预测特征第一版按 `0` 处理。
- 已实现 MetaGB updater 的 `UpdatePredictionCache`，训练多轮迭代时按 `leaf_weight * x[pred_feature]` 更新缓存。

验收：

- Python `predict()` 与内部训练迭代增量预测一致。
- 单树、浅树、bias-only 场景可手算验证。

### 阶段 5：补齐 score_agg=1/2/3

完成度：已完成第一版

工作内容：

- `score_agg=1`：对同一 `(cond_feature, bin)` 的所有 pred score 取平均，再选最佳 bin。
- `score_agg=2`：对 pred score 取中位数。
- `score_agg=3`：随机选择 pred，再选最佳 bin。

注意：

- 当前实现按原 MetaGB C 源码历史语义迁移：`score_agg=1/2/3` 聚合选择条件特征和 bin，分裂后左右子叶权重使用第 0 个预测特征计算；叶节点停止生长时仍从所有候选预测特征中选择最优叶预测特征。
- 后续若要让 `score_agg=1/2/3` 同时选择更一致的预测特征，应作为新语义单独加参数或版本标记，避免破坏 v3 迁移一致性。

验收：

- 四种 `score_agg=0/1/2/3` 都可训练。
- 固定随机种子下结果可复现。

### 阶段 6：性能优化

完成度：待执行

优先级：

1. 节点级直方图缓存，避免重复扫描样本。
2. OpenMP 并行搜索 `(ci, pi)`。
3. 使用列式/连续内存访问优化预测特征取值。
4. 再考虑 SIMD。

预期效果：

- 当前原 MetaGB 在 4-5 万样本、9-12 特征任务上慢于 XGBoost 约 20 倍。
- 第一版正确迁移后，不保证立刻最快，但应明显优于 C/CFFI 原始重复扫描实现。
- 加入直方图缓存和并行后，合理目标是训练时间降至 XGBoost CPU hist 的约 2-5 倍。
- 进一步优化后目标是约 1.5-3 倍。

## 6. 测试计划

### 6.1 单元测试

建议新增：

- `tests/cpp/tree/test_metagb_param.cc`
- `tests/cpp/tree/test_metagb_split.cc`
- `tests/python/test_metagb_regressor.py`

测试点：

- 线性系数公式：

  ```text
  beta = -sum(g*x) / (sum(h*x*x) + lambda_l2 + eps)
  ```

- bias-only 叶子等价于常数叶子。
- `cond_pred_share` 的预测特征集合包含 bias。
- `metagb_lambda_l2` 默认是 `0`。
- `enable_metagb=false` 不改变普通 XGBoost 训练。

### 6.2 回归测试

数据集：

- 小型手造 piecewise linear 数据。
- XOR-like 回归数据。
- `Exp1_54CV_ordinal/dataset/tabArena_regression` 中 1-2 个小数据集。

指标：

- RMSE。
- 训练时间。
- 与原 MetaGB v3 的预测相关性和误差趋势。
- 与 XGBoost hist 的速度比。

## 7. 风险与处理

### 7.1 XGBoost hist 现有直方图只存 `sum(g), sum(h)`

MetaGB 需要 `sum(g*x_pred), sum(h*x_pred^2)`，不能直接复用普通 `GradientPair` 直方图。

处理：

- 新增 MetaGB 专用直方图结构。
- 每个条件特征、预测特征组合维护按条件 bin 聚合的 `Gx/Hxx/count`。

### 7.2 模型结构兼容性

直接改 `RegTree::Node` 风险较高。

处理：

- 优先用旁路数组保存 MetaGB 叶子预测特征。
- 只在 `enable_metagb=true` 时启用。

### 7.3 预测缓存需要真实特征值

XGBoost 当前 prediction cache 只根据叶子 id 加叶值，而 MetaGB 还需要样本的预测特征值。

处理：

- MetaGB updater 的 `UpdatePredictionCache` 已读取 DMatrix 行特征并按 MetaGB 叶预测公式更新训练缓存。
- CPU predictor 仍保留通用预测路径，模型加载后可正常 `predict()`。

### 7.4 缺失值和稀疏输入

当前 MetaGB 原实现主要按 dense 数组语义工作。

处理：

- 第一版明确支持 dense / 常规 QuantileDMatrix 场景。
- 稀疏缺失预测特征值的处理规则需要在实现时固定，并写入测试。

## 8. 执行顺序建议

1. 不改源码，先编译并跑通 `xgboost-master` 基线。
2. 增加参数和 Python 透传。
3. 增加 MetaGB 叶子元信息和保存加载。
4. 新增 MetaGB evaluator，实现 `score_agg=0`。
5. 接入 CPU hist updater。
6. 跑小数据 correctness 测试。
7. 补 `score_agg=1/2/3`。
8. 做性能优化和大数据集 benchmark。

## 9. 当前完成度

- 需求理解：已完成。
- MetaGB v3 源码阅读：已完成。
- XGBoost CPU hist 主链路定位：已完成。
- 实践方案文档：已完成。
- XGBoost 源码修改：已完成第一版，包含参数、树元信息、模型保存加载、CPU predictor、prediction cache、GBTree updater 选择、新 MetaGB updater。
- `score_agg=0/1/2/3`：已完成第一版。
- 编译验证：已通过 `cmake --build build -j2`。
- 单元测试：已新增并通过 MetaGB 专项 Python 测试。
- 性能 benchmark：未开始。
- 原 MetaGB v3 数值对照：未开始。

## 10. 本轮实现备注

已修改文件：

- `xgboost-master/src/tree/param.h`
- `xgboost-master/include/xgboost/tree_model.h`
- `xgboost-master/src/tree/tree_model.cc`
- `xgboost-master/src/tree/tree_view.h`
- `xgboost-master/src/tree/tree_view.cc`
- `xgboost-master/src/predictor/cpu_predictor.cc`
- `xgboost-master/src/gbm/gbtree.cc`
- `xgboost-master/src/tree/updater_metagb_hist.cc`
- `xgboost-master/tests/python/test_metagb.py`

当前实现限制：

- 只注册并接入 `grow_metagb_histmaker`。
- 只支持 `enable_metagb=true`、CPU、单目标回归、`score_agg=0/1/2/3`。
- 已实现训练 prediction cache；通用 CPU predictor 也支持 MetaGB 模型预测。
- 缺失预测特征第一版按 `0` 处理。
- 当前按每棵树采样一次条件/预测特征集合，暂未实现节点级重新采样。
- 当前分裂停止使用 XGBoost `min_child_weight` Hessian 阈值，未完全复刻原 MetaGB `min_data_in_leaf` 计数语义。
- 稀疏输入、external memory、多页 batch 尚未做系统验证。

本轮验证命令：

```bash
cmake --build build -j2
PYTHONPATH=python-package /opt/anaconda3/bin/python3.13 -m pytest tests/python/test_metagb.py -q
```

测试结果：

```text
7 passed
```
