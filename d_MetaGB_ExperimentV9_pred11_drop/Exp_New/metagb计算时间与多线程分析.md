# 计算时间解释与多核并行分析

## 摘要：
1）xgboost 16核多核运行，而metagb没有OpenMP等，故只能单核；
2）超参数组合1536 vs 4608，天然差3倍；
3）但Metagb可以多线程并行跑参数组合，而非仅并行计算单组合；

## 训练次数计算（每个数据集 × 每个模型）

计算公式：`总训练次数 = 参数组合数 × inner_k折数 + 1次重训练`

### XGBoost训练次数
```
参数组合数：1,536 组合
  - learning_rate (3) × max_depth (4) × n_estimators (4) × max_bin (1)
  - × subsample (4) × colsample_bytree (4)
  - × reg_lambda (2)
  = 3 × 4 × 4 × 1 × 4 × 4 × 2 = 1,536 组合

内层4折CV: 1,536 × 4 = 6,144 次训练
最终重训练: 1 次
────────────────────────────────────
总计: 6,145 次训练
```

### MetaGB训练次数
```
参数组合数：4,608 组合
  - learning_rate (3) × max_depth (4) × n_estimators (4) × max_bin (1)
  - × subsample (4) × colsample_bytree (4)
  - × pred_feature_fraction (6) × version (1)
  = 3 × 4 × 4 × 1 × 4 × 4 × 6 × 1 = 4,608 组合

内层4折CV: 4,608 × 4 = 18,432 次训练
最终重训练: 1 次
────────────────────────────────────
总计: 18,433 次训练
```

**注意**：
- `min_data_in_leaf` = `[10]`（固定值，不参与参数搜索）
- `min_child_weight` = `[10]`（固定值，不参与参数搜索）

## 预期运行时间

基于实际日志数据（Inner CV Time部分），各数据集的实际运行时间如下：

| 数据集 | XGBoost运行时间 | MetaGB运行时间 | MetaGB是XGBoost的倍数 | 总计（单个数据集）|
|--------|---------------|---------------|---------------------|-----------------|
| airfoil_self_noise | 44.29s (0.7分钟) | 778.70s (13.0分钟) | 17.6倍 | 822.99s (13.7分钟) |
| Another-Dataset-on-used-Fiat-500 | 63.97s (1.1分钟) | 826.62s (13.8分钟) | 12.9倍 | 890.59s (14.9分钟) |
| concrete_compressive_strength | 86.13s (1.4分钟) | 995.41s (16.6分钟) | 11.6倍 | 1081.54s (18.0分钟) |
| diamonds | 384.09s (6.4分钟) | 14413.37s (240.2分钟/4.0小时) | 37.5倍 | 14797.46s (246.6分钟/4.1小时) |
| Food_Delivery_Time | 372.52s (6.2分钟) | 13643.63s (227.4分钟/3.8小时) | 36.6倍 | 14016.15s (233.6分钟/3.9小时) |
| healthcare_insurance_expenses | 43.84s (0.7分钟) | 361.16s (6.0分钟) | 8.2倍 | 405.00s (6.8分钟) |
| houses | 355.80s (5.9分钟) | 8435.70s (140.6分钟/2.3小时) | 23.7倍 | 8791.50s (146.5分钟/2.4小时) |
| miami_housing | 409.78s (6.8分钟) | 13111.05s (218.5分钟/3.6小时) | 32.0倍 | 13520.83s (225.3分钟/3.8小时) |
| physiochemical_protein | 581.40s (9.7分钟) | 18283.65s (304.7分钟/5.1小时) | 31.4倍 | 18865.05s (314.4分钟/5.2小时) |
| QSAR_fish_toxicity | 58.36s (1.0分钟) | 588.31s (9.8分钟) | 10.1倍 | 646.67s (10.8分钟) |
| wine_quality | 175.57s (2.9分钟) | 5104.10s (85.1分钟/1.4小时) | 29.1倍 | 5279.67s (88.0分钟/1.5小时) |

**总计**：约 **22小时**（所有11个数据集 × 2个模型方法）

**数据说明**：
- 以上时间来自实际日志文件中的 `Inner CV Time` 记录
- **MetaGB训练时间约为XGBoost的24.0倍（平均值）**，范围在8.2-37.5倍之间
- MetaGB总耗时：76231.66秒（约1270.5分钟/21.2小时）
- XGBoost总耗时：3175.75秒（约52.9分钟/0.9小时）
- 参数组合数：XGBoost 1,536组 vs MetaGB 4,608组（MetaGB参数空间是XGBoost的3倍）
- 最耗时数据集：physiochemical_protein（5.2小时）、diamonds（4.1小时）、Food_Delivery_Time（3.9小时）
- 最快数据集：healthcare_insurance_expenses（6.8分钟）、airfoil_self_noise（13.7分钟）
- 测试时间不计入 `Inner CV Time`（仅包含内层4折CV的参数搜索时间）
- 单次训练时间差异：MetaGB单次训练平均时间约为XGBoost的8.0倍（24.0 / 3.0）

## MetaGB多核并行分析

### MetaGB是否支持多核并行？

**答案：MetaGB核心C代码目前不支持多核并行**

### 详细分析

#### 1. 代码分析（基于 [`metagb.c`](Exp_New/metagb/metagb.c:1) 和 [`metagb_utils.py`](Exp_New/metagb/metagb_utils.py:1)）

**C代码层面（[`metagb.c`](Exp_New/metagb/metagb.c:1)）**：

- 训练函数（如 [`train_custom`](Exp_New/metagb/metagb.c:1571)）使用标准的**单线程for循环**
  - 第1666-1714行：样本采样使用 `for (int di = 0; di < num_samples; ++di)`
  - 第1138-1153行：梯度计算使用嵌套单线程循环
  - 第1189-1219行：分裂点搜索使用单线程循环
- **未使用任何多线程库**：
  - ❌ 无 OpenMP（`#pragma omp` 指令）
  - ❌ 无 pthread 库调用
  - ❌ 无 C++11 thread
- **内存分配**：使用 `malloc`/`free`（单线程操作）
- **随机数生成**：使用标准库 `rand()`（非线程安全）
- **树构建**：递归调用 `build_tree_double_custom`，完全串行

**Python封装层（[`metagb_utils.py`](Exp_New/metagb/metagb_utils.py:1)）**：

- `MetaGB` 类的 `fit()` 方法直接调用C库函数
- Python层通过 `lib.train_custom` 等 `ffi` 接口调用C代码
- Python层面无并行处理（如 `multiprocessing`、`threading`）

#### 2. 与XGBoost的对比

| 特性 | MetaGB (当前实现) | XGBoost |
|------|-----------------|---------|
| **多核并行** | ❌ 不支持 | ✅ 支持（OpenMP） |
| **训练加速** | 单CPU核心 | 多CPU核心并行 |
| **特征并行** | ❌ 不支持 | ✅ 支持 |
| **数据并行** | ❌ 不支持 | ✅ 支持 |
| **分裂点搜索** | 单线程串行 | 多线程并行 |

**XGBoost优势**：
- 使用OpenMP实现多线程特征采样、分裂点搜索、梯度计算
- 可通过 `n_jobs` 参数指定线程数
- 在多核服务器上可显著加速训练

#### 3. 时间差异解释

MetaGB训练时间是XGBoost的**平均24倍**，主要原因：

1. **算法复杂度**：
   - MetaGB需要分别计算条件特征和预测特征（双特征空间）
   - 叶子节点预测使用线性回归（额外计算：特征×权重）
   - 双重循环计算 `[条件特征 × 预测特征 × 分桶]` 分数

2. **单线程限制**：
   - MetaGB仅使用单个CPU核心
   - XGBoost可使用多核心并行加速特征遍历和分裂点搜索

3. **参数空间**（次要因素）：
   - MetaGB参数组合是XGBoost的3倍
   - 但单次训练时间差异（约8倍）才是主要因素

4. **实现优化**：
   - XGBoost经过多年优化（缓存友好、向量化等）
   - MetaGB当前实现较为简洁，未针对性能优化

#### 4. 性能改进建议

如需提升MetaGB训练速度，可考虑：

**短期方案**：
- 在Python层使用 `multiprocessing` 并行参数搜索（实验脚本已实现）
- 增加数据并行（分批训练）

**长期方案**（修改C代码）：
- 使用OpenMP并行化关键循环
  - 特征分数计算（第1138-1153行）
  - 分裂点搜索（第1189-1219行）
- 实现特征并行
- 优化内存访问模式
- 使用线程安全的随机数生成（如 `drand48_r`）

#### 5. 实验中的并行性

**当前实验**（[`exp1_bestparams_noEarly.py`](Exp_New/exp1_bestparams_noEarly.py:1)）：
- Python层：使用 `joblib` 或 `multiprocessing` 并行参数搜索
- 每个CPU核心独立运行一个参数组合
- **每个模型训练仍是单线程**

**并行层次**：
```
多进程（Python层）→ 单线程训练（C层）→ 串行计算
   ↓                      ↓                  ↓
参数组合并行化          MetaGB单核          循环串行执行
```

---
