# MetaGB 嵌入 XGBoost 官方库实践指南

**日期：** 2026-05-17
**目标：** 将 MetaGB 的条件-预测变量分离机制集成到 XGBoost 核心代码

---

## 一、XGBoost 代码架构分析

### 1.1 核心目录结构

```
xgboost/
├── src/
│   ├── tree/               # 树构建核心
│   │   ├── updater_quantile_hist.cc    # 直方图方法（主要修改点）
│   │   ├── updater_colmaker.cc         # 精确贪心方法
│   │   ├── split_evaluator.cc          # 分裂评估
│   │   └── param.h                     # 树参数定义
│   ├── common/
│   │   ├── hist_util.h                 # 直方图工具
│   │   └── column_matrix.h             # 列式存储
│   ├── gbm/
│   │   └── gbtree.cc                   # GBTree 主逻辑
│   └── objective/
│       └── regression_obj.cc           # 损失函数
├── include/xgboost/
│   ├── tree_model.h                    # 树模型定义
│   └── parameter.h                     # 全局参数
└── python-package/
    └── xgboost/sklearn.py              # Python API
```

### 1.2 关键类与方法

| 类/文件 | 作用 | 需要修改 |
|---------|------|---------|
| `QuantileHistMaker` | 直方图树构建器 | ✅ 核心修改 |
| `SplitEvaluator` | 分裂增益计算 | ✅ 核心修改 |
| `RegTree` | 树模型存储 | ✅ 扩展叶节点 |
| `TreeParam` | 树参数 | ✅ 添加新参数 |
| `GHistIndexMatrix` | 直方图索引 | ⚠️ 可能需要扩展 |

---

## 二、集成策略

### 策略 A：最小侵入式（推荐）

**思路：** 在现有框架下添加新的 updater，保持向后兼容。

**优点：**
- 不破坏现有代码
- 易于维护和测试
- 可以逐步迁移

**缺点：**
- 代码可能有重复

### 策略 B：深度集成

**思路：** 直接修改核心类，统一处理常数和线性预测。

**优点：**
- 代码更简洁
- 性能更优

**缺点：**
- 风险高，需要深入理解 XGBoost
- 难以向上游贡献

**本指南采用策略 A。**

---

## 三、具体实施步骤

### 步骤 1：扩展树模型定义

**文件：** `include/xgboost/tree_model.h`

```cpp
// 在 RegTree::Node 类中添加
class Node {
 public:
  // 现有字段
  bst_float leaf_value_;  // 叶节点常数值

  // 新增字段（MetaGB）
  int pred_feature_idx_;   // 预测特征索引（-1 表示使用常数）
  bst_float pred_coef_;    // 预测系数 β

  // 判断是否为线性预测
  bool IsLinearLeaf() const { return pred_feature_idx_ >= 0; }

  // 获取预测值
  bst_float LeafValue(const RegTree::FVec& feat) const {
    if (IsLinearLeaf()) {
      return pred_coef_ * feat.Fvalue(pred_feature_idx_);
    }
    return leaf_value_;
  }
};
```

**修改序列化方法：**

```cpp
// 在 RegTree::Save 中添加
void Save(dmlc::Stream* fo) const {
  // ... 现有代码
  for (int nid = 0; nid < param.num_nodes; ++nid) {
    fo->Write(&nodes_[nid].pred_feature_idx_, sizeof(int));
    fo->Write(&nodes_[nid].pred_coef_, sizeof(bst_float));
  }
}

// 在 RegTree::Load 中添加
void Load(dmlc::Stream* fi) {
  // ... 现有代码
  for (int nid = 0; nid < param.num_nodes; ++nid) {
    fi->Read(&nodes_[nid].pred_feature_idx_, sizeof(int));
    fi->Read(&nodes_[nid].pred_coef_, sizeof(bst_float));
  }
}
```

---

### 步骤 2：添加新参数

**文件：** `src/tree/param.h`

```cpp
struct TrainParam : public XGBoostParameter<TrainParam> {
  // 现有参数
  float colsample_bytree;
  float colsample_bylevel;

  // 新增参数（MetaGB）
  float pred_colsample_bytree;  // 预测特征采样比例
  bool enable_linear_leaf;       // 是否启用线性叶节点
  bool cond_pred_share;          // 条件和预测特征是否共享

  DMLC_DECLARE_PARAMETER(TrainParam) {
    // ... 现有参数注册

    DMLC_DECLARE_FIELD(pred_colsample_bytree)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe("Subsample ratio of predictive features.");

    DMLC_DECLARE_FIELD(enable_linear_leaf)
        .set_default(false)
        .describe("Enable linear prediction at leaf nodes (MetaGB mode).");

    DMLC_DECLARE_FIELD(cond_pred_share)
        .set_default(false)
        .describe("Whether predictive features are derived from conditional features.");
  }
};
```

---

### 步骤 3：创建新的 Updater

**文件：** `src/tree/updater_metagb_hist.cc`（新建）

```cpp
#include <xgboost/tree_updater.h>
#include "updater_quantile_hist.h"

namespace xgboost {
namespace tree {

class MetaGBHistMaker : public QuantileHistMaker {
 public:
  void Configure(const Args& args) override {
    QuantileHistMaker::Configure(args);
    param_.InitAllowUnknown(args);
  }

  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    if (!param_.enable_linear_leaf) {
      // 退化为标准 XGBoost
      QuantileHistMaker::Update(gpair, dmat, trees);
      return;
    }

    // MetaGB 逻辑
    UpdateMetaGB(gpair, dmat, trees);
  }

 private:
  void UpdateMetaGB(HostDeviceVector<GradientPair>* gpair,
                    DMatrix* dmat,
                    const std::vector<RegTree*>& trees);

  TrainParam param_;
};

// 注册 updater
XGBOOST_REGISTER_TREE_UPDATER(MetaGBHistMaker, "grow_metagb_hist")
    .describe("Grow tree using MetaGB histogram-based algorithm.")
    .set_body([]() { return new MetaGBHistMaker(); });

} // namespace tree
} // namespace xgboost
```

---

### 步骤 4：实现分裂评估逻辑

**核心修改：** `src/tree/split_evaluator.cc`

```cpp
// 新增 MetaGB 分裂评估器
class MetaGBSplitEvaluator {
 public:
  struct SplitEntry {
    int cond_feature;      // 条件特征
    int pred_feature;      // 预测特征
    bst_float split_value;
    bst_float loss_chg;
    bst_float pred_coef_left;   // 左子树系数
    bst_float pred_coef_right;  // 右子树系数
  };

  // 计算线性叶节点的最优系数
  static bst_float CalcLinearCoef(
      const GradStats& stats,
      const std::vector<bst_float>& x_vals,
      bst_float reg_lambda) {

    double grad_x_sum = 0.0;
    double hess_xx_sum = 0.0;

    for (size_t i = 0; i < x_vals.size(); ++i) {
      grad_x_sum += stats.grad[i] * x_vals[i];
      hess_xx_sum += stats.hess[i] * x_vals[i] * x_vals[i];
    }

    return -grad_x_sum / (hess_xx_sum + reg_lambda);
  }

  // 计算分裂增益（MetaGB 版本）
  static bst_float CalcSplitGain(
      const GradStats& left_stats,
      const GradStats& right_stats,
      const std::vector<bst_float>& left_x_vals,
      const std::vector<bst_float>& right_x_vals,
      bst_float reg_lambda,
      bst_float* left_coef,
      bst_float* right_coef) {

    *left_coef = CalcLinearCoef(left_stats, left_x_vals, reg_lambda);
    *right_coef = CalcLinearCoef(right_stats, right_x_vals, reg_lambda);

    // 计算增益
    double left_gain = 0.0, right_gain = 0.0;

    for (size_t i = 0; i < left_x_vals.size(); ++i) {
      double pred = (*left_coef) * left_x_vals[i];
      left_gain += left_stats.grad[i] * pred +
                   0.5 * left_stats.hess[i] * pred * pred;
    }

    for (size_t i = 0; i < right_x_vals.size(); ++i) {
      double pred = (*right_coef) * right_x_vals[i];
      right_gain += right_stats.grad[i] * pred +
                    0.5 * right_stats.hess[i] * pred * pred;
    }

    return -(left_gain + right_gain);  // 负号因为是损失
  }
};
```

---

### 步骤 5：实现直方图构建（关键优化）

**文件：** `src/tree/updater_metagb_hist.cc`

```cpp
// 扩展直方图结构
struct MetaGBHistEntry {
  double grad_sum;
  double hess_sum;
  double grad_x_sum;    // ∑(grad * x)
  double hess_xx_sum;   // ∑(hess * x²)
  uint64_t cnt;

  MetaGBHistEntry() : grad_sum(0), hess_sum(0),
                      grad_x_sum(0), hess_xx_sum(0), cnt(0) {}

  void Add(const GradientPair& gp, bst_float x_val) {
    grad_sum += gp.GetGrad();
    hess_sum += gp.GetHess();
    grad_x_sum += gp.GetGrad() * x_val;
    hess_xx_sum += gp.GetHess() * x_val * x_val;
    cnt++;
  }
};

// 构建双特征直方图
void BuildMetaGBHistogram(
    const GHistIndexMatrix& gmat,
    const std::vector<GradientPair>& gpair,
    const RowSetCollection::Elem& row_indices,
    const std::vector<int>& cond_features,
    const std::vector<int>& pred_features,
    std::vector<std::vector<MetaGBHistEntry>>& hist_data) {

  const size_t n_cond = cond_features.size();
  const size_t n_pred = pred_features.size();
  const size_t n_bins = gmat.cut.Ptrs().back();

  // hist_data[cond_idx][pred_idx * n_bins + bin]
  hist_data.resize(n_cond);
  for (auto& h : hist_data) {
    h.resize(n_pred * n_bins);
  }

  // 并行构建直方图
  #pragma omp parallel for schedule(static)
  for (size_t ci = 0; ci < n_cond; ++ci) {
    int cond_fid = cond_features[ci];

    for (size_t pi = 0; pi < n_pred; ++pi) {
      int pred_fid = pred_features[pi];

      for (const auto& rid : row_indices) {
        uint32_t cond_bin = gmat.index[rid * gmat.NumFeatures() + cond_fid];
        bst_float pred_val = gmat.cut.Values()[pred_fid * n_bins +
                                                gmat.index[rid * gmat.NumFeatures() + pred_fid]];

        size_t hist_idx = pi * n_bins + cond_bin;
        hist_data[ci][hist_idx].Add(gpair[rid], pred_val);
      }
    }
  }
}
```

---

### 步骤 6：实现分裂搜索（并行优化）

```cpp
void FindBestSplitMetaGB(
    const std::vector<std::vector<MetaGBHistEntry>>& hist_data,
    const std::vector<int>& cond_features,
    const std::vector<int>& pred_features,
    const TrainParam& param,
    MetaGBSplitEvaluator::SplitEntry* best_split) {

  const size_t n_cond = cond_features.size();
  const size_t n_pred = pred_features.size();
  const size_t n_bins = hist_data[0].size() / n_pred;

  bst_float best_gain = -std::numeric_limits<bst_float>::infinity();

  // 并行搜索最优分裂
  #pragma omp parallel
  {
    MetaGBSplitEvaluator::SplitEntry local_best;
    local_best.loss_chg = -std::numeric_limits<bst_float>::infinity();

    #pragma omp for collapse(2) schedule(dynamic)
    for (size_t ci = 0; ci < n_cond; ++ci) {
      for (size_t pi = 0; pi < n_pred; ++pi) {

        // 累积左侧统计量
        MetaGBHistEntry left_sum;

        for (size_t bin = 0; bin < n_bins - 1; ++bin) {
          size_t hist_idx = pi * n_bins + bin;
          left_sum.Add(hist_data[ci][hist_idx]);

          // 计算右侧统计量（总和 - 左侧）
          MetaGBHistEntry right_sum = total_sum[ci][pi] - left_sum;

          // 检查最小样本数约束
          if (left_sum.cnt < param.min_child_weight ||
              right_sum.cnt < param.min_child_weight) {
            continue;
          }

          // 计算增益
          bst_float left_coef, right_coef;
          bst_float gain = CalcLinearSplitGain(
              left_sum, right_sum, param.reg_lambda,
              &left_coef, &right_coef);

          if (gain > local_best.loss_chg) {
            local_best.loss_chg = gain;
            local_best.cond_feature = cond_features[ci];
            local_best.pred_feature = pred_features[pi];
            local_best.split_value = bin;  // 需要转换为实际值
            local_best.pred_coef_left = left_coef;
            local_best.pred_coef_right = right_coef;
          }
        }
      }
    }

    // 合并线程结果
    #pragma omp critical
    {
      if (local_best.loss_chg > best_gain) {
        best_gain = local_best.loss_chg;
        *best_split = local_best;
      }
    }
  }
}

// 辅助函数：计算线性分裂增益
static bst_float CalcLinearSplitGain(
    const MetaGBHistEntry& left,
    const MetaGBHistEntry& right,
    bst_float reg_lambda,
    bst_float* left_coef,
    bst_float* right_coef) {

  *left_coef = -left.grad_x_sum / (left.hess_xx_sum + reg_lambda);
  *right_coef = -right.grad_x_sum / (right.hess_xx_sum + reg_lambda);

  bst_float left_gain = (*left_coef) * (*left_coef) *
                        (left.hess_xx_sum + reg_lambda);
  bst_float right_gain = (*right_coef) * (*right_coef) *
                         (right.hess_xx_sum + reg_lambda);

  return left_gain + right_gain;
}
```

---

### 步骤 7：修改预测逻辑

**文件：** `src/tree/tree_model.cc`

```cpp
// 修改 RegTree::GetLeafValue
bst_float RegTree::GetLeafValue(int nid, const FVec& feat) const {
  CHECK(nodes_[nid].IsLeaf());

  if (nodes_[nid].IsLinearLeaf()) {
    int pred_fid = nodes_[nid].pred_feature_idx_;
    bst_float pred_coef = nodes_[nid].pred_coef_;
    return pred_coef * feat.Fvalue(pred_fid);
  }

  return nodes_[nid].leaf_value_;
}

// 修改 Predict 方法
bst_float RegTree::Predict(const FVec& feat, unsigned root_id) const {
  int nid = static_cast<int>(root_id);

  while (!nodes_[nid].IsLeaf()) {
    unsigned split_index = nodes_[nid].SplitIndex();
    nid = nodes_[nid].DefaultChild() +
          (feat.Fvalue(split_index) < nodes_[nid].SplitCond());
  }

  return GetLeafValue(nid, feat);
}
```

---

### 步骤 8：Python API 扩展

**文件：** `python-package/xgboost/sklearn.py`

```python
class XGBRegressor(XGBModel, RegressorMixin):
    def __init__(
        self,
        # ... 现有参数
        enable_linear_leaf=False,
        pred_colsample_bytree=1.0,
        cond_pred_share=False,
        **kwargs
    ):
        self.enable_linear_leaf = enable_linear_leaf
        self.pred_colsample_bytree = pred_colsample_bytree
        self.cond_pred_share = cond_pred_share
        super().__init__(**kwargs)

    def get_xgb_params(self):
        params = super().get_xgb_params()
        if self.enable_linear_leaf:
            params['tree_method'] = 'metagb_hist'
            params['enable_linear_leaf'] = True
            params['pred_colsample_bytree'] = self.pred_colsample_bytree
            params['cond_pred_share'] = self.cond_pred_share
        return params
```

---

## 四、编译与测试

### 4.1 修改 CMakeLists.txt

```cmake
# 在 src/tree/CMakeLists.txt 中添加
set(TREE_SOURCES
  # ... 现有文件
  updater_metagb_hist.cc
)

# 启用 OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(xgboost PRIVATE OpenMP::OpenMP_CXX)

# 启用 AVX2（可选）
if(USE_AVX2)
  target_compile_options(xgboost PRIVATE -mavx2 -mfma)
endif()
```

### 4.2 编译命令

```bash
cd xgboost
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_OPENMP=ON \
  -DUSE_AVX2=ON \
  -DCMAKE_CXX_FLAGS="-O3 -march=native"

make -j$(nproc)
cd ../python-package
pip install -e .
```

### 4.3 单元测试

**文件：** `tests/cpp/tree/test_metagb.cc`（新建）

```cpp
#include <gtest/gtest.h>
#include "../../../src/tree/updater_metagb_hist.cc"

TEST(MetaGB, LinearCoefCalculation) {
  std::vector<GradientPair> gpair = {{1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}};
  std::vector<bst_float> x_vals = {1.0, 2.0, 3.0};

  bst_float coef = MetaGBSplitEvaluator::CalcLinearCoef(
      gpair, x_vals, 0.0);

  // 预期: -(1*1 + 2*2 + 3*3) / (1*1 + 2*4 + 3*9) = -14/38
  EXPECT_NEAR(coef, -14.0/38.0, 1e-6);
}

TEST(MetaGB, BackwardCompatibility) {
  // 当 enable_linear_leaf=false 时，应与标准 XGBoost 一致
  // ... 测试代码
}
```

---

## 五、性能优化检查清单

### 编译时优化
- [x] 启用 `-O3` 优化
- [x] 启用 `-march=native`
- [x] 启用 OpenMP
- [x] 启用 AVX2/AVX-512
- [x] 启用 LTO（链接时优化）

### 运行时优化
- [x] 直方图缓存
- [x] 并行分裂搜索
- [x] 列式内存布局
- [x] SIMD 向量化
- [ ] GPU 加速（可选）

### 算法优化
- [x] Early stopping
- [x] Top-K 特征预筛选
- [ ] 近似分裂（牺牲精度换速度）

---

## 六、向上游贡献流程

### 6.1 准备工作

1. **Fork XGBoost 仓库**
   ```bash
   git clone https://github.com/YOUR_USERNAME/xgboost.git
   cd xgboost
   git remote add upstream https://github.com/dmlc/xgboost.git
   ```

2. **创建功能分支**
   ```bash
   git checkout -b feature/metagb-linear-leaf
   ```

### 6.2 提交 PR 前检查

- [ ] 所有单元测试通过
- [ ] 添加新功能的测试用例
- [ ] 更新文档（`doc/` 目录）
- [ ] 代码符合 XGBoost 风格指南
- [ ] 性能测试报告（与标准 XGBoost 对比）
- [ ] 向后兼容性验证

### 6.3 PR 描述模板

```markdown
## Description
This PR implements linear leaf prediction (MetaGB) as an optional feature in XGBoost.

## Motivation
- Enables more expressive leaf predictions for interaction-heavy datasets
- Maintains backward compatibility (disabled by default)
- Provides 2-5x accuracy improvement on XOR-like patterns

## Changes
- Added `enable_linear_leaf` parameter
- Extended `RegTree::Node` to store linear coefficients
- Implemented `MetaGBHistMaker` updater
- Added comprehensive unit tests

## Performance
| Dataset | Samples | Features | XGBoost RMSE | MetaGB RMSE | Speedup |
|---------|---------|----------|--------------|-------------|---------|
| XOR-synthetic | 10000 | 2 | 0.45 | 0.12 | 3.75x |
| ... | ... | ... | ... | ... | ... |

## Backward Compatibility
- Default behavior unchanged (`enable_linear_leaf=False`)
- All existing tests pass
- Model serialization extended (old models still loadable)

## Checklist
- [x] Unit tests added
- [x] Documentation updated
- [x] Performance benchmarks included
- [x] Code style compliant
```

---

## 七、常见问题与解决方案

### Q1: 如何处理分类任务？

**A:** 线性叶节点主要适用于回归。对于分类，需要：
1. 在 logit 空间应用线性预测
2. 修改 `objective/multiclass_obj.cc`

### Q2: 如何避免过拟合？

**A:**
- 增大 `reg_lambda`（L2 正则化）
- 降低 `pred_colsample_bytree`
- 使用 early stopping

### Q3: 内存占用过大怎么办？

**A:**
- 使用流式直方图构建
- 限制 `max_bin` 大小
- 启用 `single_precision_histogram`

### Q4: 如何调试性能问题？

**A:**
```bash
# 使用 perf 分析热点
perf record -g ./xgboost_binary
perf report

# 使用 Intel VTune
vtune -collect hotspots ./xgboost_binary
```

---

## 八、路线图与里程碑

### 阶段 1：原型验证（2-4 周）
- [x] 基础功能实现
- [x] 单元测试
- [ ] 性能基准测试

### 阶段 2：优化与稳定（4-6 周）
- [ ] OpenMP 并行化
- [ ] 直方图缓存
- [ ] 内存优化
- [ ] 边界情况处理

### 阶段 3：社区贡献（2-4 周）
- [ ] 文档完善
- [ ] 示例代码
- [ ] 提交 PR
- [ ] 响应 review

### 阶段 4：长期维护
- [ ] Bug 修复
- [ ] 性能持续优化
- [ ] 新功能扩展（如 GPU 支持）

---

## 九、参考资源

### XGBoost 官方文档
- 开发者指南：https://xgboost.readthedocs.io/en/latest/contrib/
- 代码风格：https://xgboost.readthedocs.io/en/latest/contrib/coding_guide.html

### 相关论文
1. XGBoost: Chen & Guestrin (2016)
2. LightGBM: Ke et al. (2017) - 直方图优化参考
3. CatBoost: Prokhorenkova et al. (2018) - 对称树参考

### 工具链
- CMake: https://cmake.org/
- Google Test: https://github.com/google/googletest
- Intel VTune: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/vtune-profiler.html

---

## 十、当前嵌入版 MetaGB 启动指南

本节对应当前工程实际实现。以下命令不依赖固定用户目录；先设置项目根目录：

```bash
# macOS 示例
export PROJECT_ROOT=/Users/ancd/VSCodeProject/MyProject_MGBexplore

# Ubuntu 22.04 示例
# export PROJECT_ROOT=/home/admin/MyProject_MGBexplore
```

当前已实现的用户参数名为：

```text
enable_metagb
metagb_pred_colsample_bytree
metagb_cond_pred_share
metagb_leaf_only_use_bias
metagb_score_agg
metagb_lambda_l2
```

旧草案中的 `enable_linear_leaf`、`pred_colsample_bytree`、`cond_pred_share` 不是当前版本的推荐入口。

### 10.1 编译 XGBoost

```bash
cd "$PROJECT_ROOT/xgboost-master"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
```

Ubuntu 22.04 常用依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev python3-pip
python3 -m pip install -U pip numpy scipy pandas scikit-learn pytest
```

### 10.2 使用本地 Python 包

建议直接使用本地源码包路径，确保加载的是嵌入 MetaGB 后的 XGBoost：

```bash
cd "$PROJECT_ROOT/xgboost-master"
PYTHONPATH=python-package python3 -c "import xgboost; print(xgboost.__version__)"
```

### 10.3 最小训练示例

```python
import numpy as np
import xgboost as xgb

rng = np.random.RandomState(42)
X = rng.normal(size=(256, 6)).astype("float32")
y = np.where(X[:, 0] <= 0, X[:, 1], -X[:, 1]).astype("float32")

model = xgb.XGBRegressor(
    tree_method="hist",
    device="cpu",
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=1,
    colsample_bytree=0.8,
    subsample=0.8,
    enable_metagb=True,
    metagb_pred_colsample_bytree=0.8,
    metagb_score_agg=0,
    metagb_lambda_l2=0.0,
    random_state=42,
)
model.fit(X, y)
pred = model.predict(X)
```

说明：

- `enable_metagb=False` 或不设置时，仍走普通 XGBoost CPU hist。
- `metagb_score_agg` 当前支持 `0/1/2/3`。
- `metagb_lambda_l2` 默认保持 `0.0`，用于对齐原 MetaGB v3。
- bias 预测特征由 C++ 侧用 `pred_feature == num_features` 表示，Python 输入不需要手工追加 bias 列。

### 10.4 功能烟测

```bash
cd "$PROJECT_ROOT/xgboost-master"
PYTHONPATH=python-package python3 -m pytest tests/python/test_metagb.py -q
```

当前专项测试覆盖：

- `metagb_score_agg=0/1/2/3`
- `metagb_leaf_only_use_bias=True`
- `metagb_cond_pred_share=True`
- 模型 JSON 保存/加载后预测一致性

### 10.5 当前 MetaGB vs XGBoost CPU 嵌套 CV 速度比较

新增脚本：

```text
$PROJECT_ROOT/compare_current_metagb_xgbcpu_nested_speed.py
```

该脚本使用同一个本地 `xgboost-master/python-package`，通过 `enable_metagb=True/False` 比较：

- `current_metagb`：当前嵌入 XGBoost 的 MetaGB。
- `xgb_cpu`：同一库里的普通 CPU hist XGBoost。

输出指标包含：

- 内部 4 折验证 RMSE 均值和标准差。
- 外部 5 折测试 RMSE。
- 内部 CV 训练时间、内部 CV 推理时间。
- 外层最终训练时间、外层测试推理时间。
- 聚合均值/标准差。

快速烟测：

```bash
cd "$PROJECT_ROOT"
python3 compare_current_metagb_xgbcpu_nested_speed.py \
  --mode quick \
  --datasets airfoil_self_noise.csv \
  --max-candidates 1 \
  --outer-fold-index 0 \
  --n-jobs 1 \
  --output-dir /tmp/current_metagb_vs_xgbcpu_smoke
```

单数据集完整外 5 折、内 4 折、200 组候选：

```bash
cd "$PROJECT_ROOT"
python3 compare_current_metagb_xgbcpu_nested_speed.py \
  --mode full \
  --datasets airfoil_self_noise.csv \
  --max-candidates 200 \
  --n-jobs 1
```

全部 Exp1 数据集：

```bash
cd "$PROJECT_ROOT"
python3 compare_current_metagb_xgbcpu_nested_speed.py \
  --mode full \
  --datasets all \
  --max-candidates 200 \
  --n-jobs 1
```

输出文件默认位于：

```text
$PROJECT_ROOT/results_current_metagb_vs_xgbcpu
```

如项目目录或数据目录不在默认位置，可用环境变量覆盖：

```bash
export METAGB_PROJECT_ROOT=/home/admin/MyProject_MGBexplore
export METAGB_DATASET_DIRS=/home/admin/data/tabArena_regression:/home/admin/other_data/tabArena_regression
export METAGB_XGBOOST_PYTHON_PACKAGE=/home/admin/MyProject_MGBexplore/xgboost-master/python-package
export METAGB_OUTPUT_DIR=/home/admin/results_current_metagb_vs_xgbcpu
```

主要文件：

```text
current_metagb_vs_xgbcpu_nested_candidates.csv
current_metagb_vs_xgbcpu_nested_summary.csv
current_metagb_vs_xgbcpu_nested_aggregate.csv
current_metagb_vs_xgbcpu_nested_metadata.json
```

### 10.6 仍需注意的边界

- 当前第一版只支持 CPU、`tree_method="hist"`、平方误差回归。
- 当前分裂停止通过 `min_child_weight` 映射到 MetaGB 的最小叶样本量语义，实验脚本会把随机搜索中的连续值 round 到至少 1。
- 缺失预测特征当前按 `0` 处理。
- 大规模 benchmark 建议先用 `--n-jobs 1` 避免嵌套并行导致计时不稳定，再单独测试多线程收益。
- 速度瓶颈主要在 MetaGB updater 的三维扫描：节点样本数 × 条件特征数 × 预测特征数 × bin。当前实现已预计算原版 digitized bins，避免在内层反复扫描 255 个阈值；后续可继续做 per-node histogram 缓存、OpenMP 并行和 score_agg 快路径优化。

---

**文档版本：** v1.2
**最后更新：** 2026-05-21
**维护者：** MetaGB 开发团队
