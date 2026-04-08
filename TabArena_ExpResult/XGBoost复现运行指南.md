# TabArena XGBoost 模型复现运行指南

## 一、项目概述

TabArena 是一个用于表格数据机器学习的动态基准测试系统，实现了最佳实践以确保方法在其最佳状态下进行比较，包括交叉验证集成、强超参数搜索空间、早停、模型重拟合、并行 bagging、内存使用估计等。

### 1.1 XGBoost 在 TabArena 中的位置

- **模型名称**: XGBoost
- **论文配置名称**: `XGBoost_c1_BAG_L1`
- **简写**: XGB
- **类型**: CPU 模型（属于 GBDT 家族）

### 1.2 相关链接

- **在线 Leaderboard**: https://huggingface.co/spaces/TabArena/leaderboard
- **官网**: https://tabarena.ai/
- **论文**: https://tabarena.ai/paper-tabular-ml-iid-study

---

## 二、现成的 Benchmark 结果查看

### 2.1 在线查看（推荐）

直接访问 TabArena 的 [HuggingFace Leaderboard](https://huggingface.co/spaces/TabArena/leaderboard) 查看所有模型的性能排名，包括 XGBoost。

### 2.2 本地生成 Leaderboard

运行以下命令在本地生成完整的 Leaderboard：

```bash
cd g:/VSCodeWork/TabArena
python examples/plots/run_generate_main_leaderboard.py
```

生成的结果保存在 `output_leaderboard/` 目录下，包含：
- `verified/`: 经过验证的结果
- `unverified/`: 未验证的结果

### 2.3 查看已发布的 XGBoost 结果

XGBoost 的元数据定义在 [`tabarena/nips2025_utils/artifacts/_tabarena_method_metadata_2025_06_12.py`](tabarena/tabarena/nips2025_utils/artifacts/_tabarena_method_metadata_2025_06_12.py) 中，配置名称为 `XGBoost_c1_BAG_L1`。

---

## 三、环境安装

### 3.1 基础安装

```bash
# 1. 克隆仓库
git clone https://github.com/autogluon/tabarena.git
cd tabarena

# 2. 安装 UV（如果尚未安装）
# 参考: https://docs.astral.sh/uv/getting-started/installation/

# 3. 安装基础依赖
uv sync

# 4. 安装 Benchmark 额外依赖（运行模型需要）
uv sync --extra benchmark
```

### 3.2 开发者安装

```bash
# 创建虚拟环境
uv venv --seed --python 3.12 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate  # Linux/macOS
# 或 Windows: .\env\Scripts\activate

# 安装 AutoGluon（开发模式）
git clone https://github.com/autogluon/autogluon.git
cd autogluon
./full_install.sh

# 安装 TabArena
cd ..
git clone https://github.com/autogluon/tabarena.git
uv pip install --prerelease=allow -e "./tabarena/tabarena[benchmark]"
uv pip install --prerelease=allow -e "./tabarena[benchmark]"(cd /home/admin/MyProject5/TabArena)
```

### 3.3 验证安装

```python
python -c "from autogluon.tabular.models import XGBoostModel; print('XGBoostModel imported successfully')"
python -c "from tabarena.models.utils import get_configs_generator_from_name; print(get_configs_generator_from_name('XGBoost'))"
```

---

## 四、运行 XGBoost 的方法

### 方法一：使用 TabArena Benchmark 框架（推荐）

参考 [`examples/benchmarking/run_quickstart_tabarena_one_dataset.py`](examples/benchmarking/run_quickstart_tabarena_one_dataset.py)，创建运行 XGBoost 的脚本：

```python
"""
XGBoost Benchmark 复现脚本
"""
from __future__ import annotations
from pathlib import Path
from tabarena.benchmark.experiment import run_experiments_new
from tabarena.models.utils import get_configs_generator_from_name
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

if __name__ == "__main__":
    # 实验输出目录
    tabarena_dir = str(Path(__file__).parent / "experiments" / "xgboost_experiment")
    
    # 获取任务元数据
    # OpenML 任务 ID 示例: 363621 (Bank Marketing)
    # 更多任务可从 TabArena 的 51 个数据集选择
    tasks = [363621]
    metadata = load_task_metadata()
    
    # 获取 XGBoost 配置生成器
    config_generator = get_configs_generator_from_name("XGBoost")
    
    # 生成实验配置
    model_experiments = config_generator.generate_all_bag_experiments(
        num_random_configs=1,  # 默认配置数量
        fold_fitting_strategy="sequential_local",  # 顺序拟合，便于调试
    )
    
    # 运行实验
    results_lst = run_experiments_new(
        output_dir=tabarena_dir,
        model_experiments=model_experiments,
        tasks=tasks,
        repetitions_mode="TabArena-Lite",  # 使用 TabArena-Lite 模式
    )
    
    # 计算并展示结果
    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=metadata,
        cache=False,
        cache_raw=False,
    )
    end_to_end_results = end_to_end.to_results()
    df_results = end_to_end_results.get_results()
    
    print("XGBoost Benchmark 结果:")
    print(df_results)
```

### 方法二：使用 AutoGluon 直接运行

参考 [`examples/running_tabarena_models/run_default_model.py`](examples/running_tabarena_models/run_default_model.py)：

```python
"""
直接使用 AutoGluon 运行 XGBoost
"""
from autogluon.tabular.models import XGBoostModel
from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if __name__ == "__main__":
    # 获取 TabArena 任务元数据
    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata
    
    # 示例数据集
    datasets = ["anneal", "credit-g", "diabetes"]  # 可替换为其他数据集
    folds = [0]  # 第一个 fold
    
    # 创建 XGBoost 实验
    xgboost_exp = AGModelBagExperiment(
        name="XGBoost_c1_BAG_L1",           # 与论文一致的命名
        model_cls=XGBoostModel,             # AutoGluon 的 XGBoostModel
        model_hyperparameters={},            # 使用默认超参数
        num_bag_folds=8,                    # 论文使用 8 折 bagging
        time_limit=3600,                     # 每任务 1 小时时间限制
    )
    
    # 创建实验运行器
    exp_batch_runner = ExperimentBatchRunner(
        expname="xgboost_output",
        task_metadata=task_metadata
    )
    
    # 运行实验
    results_lst = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=[xgboost_exp],
        ignore_cache=False,  # True 强制重新运行
    )
    
    print(f"完成! 结果保存在 xgboost_output 目录")
```

### 方法三：自定义 XGBoost 超参数

```python
"""
使用自定义超参数运行 XGBoost
"""
from autogluon.tabular.models import XGBoostModel
from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if __name__ == "__main__":
    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata
    
    # 自定义 XGBoost 超参数
    custom_xgboost_exp = AGModelBagExperiment(
        name="XGBoost_Custom_BAG_L1",
        model_cls=XGBoostModel,
        model_hyperparameters={
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        num_bag_folds=8,
        time_limit=3600,
    )
    
    exp_batch_runner = ExperimentBatchRunner(
        expname="xgboost_custom_output",
        task_metadata=task_metadata
    )
    
    results = exp_batch_runner.run(
        datasets=["credit-g"],
        folds=[0],
        methods=[custom_xgboost_exp],
    )
```

---

## 五、XGBoost 超参数搜索空间

XGBoost 的完整超参数搜索空间定义在 [`tabarena/tabarena/models/xgboost/generate.py`](tabarena/tabarena/models/xgboost/generate.py)：

### 5.1 搜索空间详情

| 超参数 | 范围 | 类型 | 说明 |
|--------|------|------|------|
| learning_rate | [5e-3, 1e-1] | log | 学习率 |
| max_depth | [4, 10] | log | 最大树深度 |
| min_child_weight | [1e-3, 5.0] | log | 最小子权重 |
| subsample | [0.6, 1.0] | linear | 行采样比例 |
| colsample_bytree | [0.6, 1.0] | linear | 列采样比例（树级别）|
| colsample_bylevel | [0.6, 1.0] | linear | 列采样比例（层级）|
| colsample_bynode | [0.6, 1.0] | linear | 列采样比例（节点）|
| reg_alpha | [1e-4, 5.0] | linear | L1 正则化 |
| reg_lambda | [1e-4, 5.0] | linear | L2 正则化 |
| grow_policy | [depthwise, lossguide] | categorical | 树生长策略 |
| max_cat_to_onehot | [8, 100] | log | 类别特征阈值 |
| max_leaves | [8, 1024] | log | 最大叶子节点数 |

### 5.2 查看所有预定义配置

XGBoost 的预定义配置文件位于 [`data/configs/configs_xgboost.json`](data/configs/configs_xgboost.json)，包含约 200 个随机采样配置。

---

## 六、TabArena 51 个标准数据集

TabArena 使用 51 个经过人工筛选的表格数据集。获取完整列表：

```python
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

context = TabArenaContext()
print(context.task_metadata["name"].tolist())
```

### 6.1 常用数据集示例

| 数据集名称 | OpenML Task ID | 任务类型 |
|-----------|---------------|---------|
| anneal | 固体力学 | 二分类 |
| credit-g | 德国信贷数据 | 二分类 |
| diabetes | 皮马糖尿病 | 二分类 |
| kc1 | NASA KC1 软件 | 二分类 |
| kr-vs-kp | 国际象棋 | 二分类 |
| lymphoma | 基因表达 | 多分类 |
| pol | 选举预测 | 二分类 |
| wine-quality | 葡萄酒质量 | 多分类 |

---

## 七、关键文件索引

### 7.1 XGBoost 核心文件

| 文件路径 | 说明 |
|---------|------|
| [`tabarena/tabarena/models/xgboost/generate.py`](tabarena/tabarena/models/xgboost/generate.py) | XGBoost 配置生成器和搜索空间定义 |
| [`data/configs/configs_xgboost.json`](data/configs/configs_xgboost.json) | 预定义的 XGBoost 超参数配置 |
| [`tabarena/tabarena/models/utils.py`](tabarena/tabarena/models/utils.py:46) | `get_configs_generator_from_name("XGBoost")` 模型注册 |

### 7.2 元数据和结果文件

| 文件路径 | 说明 |
|---------|------|
| [`tabarena/tabarena/nips2025_utils/artifacts/_tabarena_method_metadata_2025_06_12.py`](tabarena/tabarena/nips2025_utils/artifacts/_tabarena_method_metadata_2025_06_12.py:64) | XGBoost 元数据定义（配置名、缩写等）|
| [`tabarena/tabarena/nips2025_utils/artifacts/_tabarena_method_metadata.py`](tabarena/tabarena/nips2025_utils/artifacts/_tabarena_method_metadata.py:116) | 所有方法元数据集合 |
| [`tabarena/tabarena/nips2025_utils/metadata/task_metadata_tabarena51.csv`](tabarena/tabarena/nips2025_utils/metadata/task_metadata_tabarena51.csv) | 51 个数据集的元数据 |

### 7.3 示例脚本

| 脚本路径 | 说明 |
|---------|------|
| [`examples/benchmarking/run_quickstart_tabarena.py`](examples/benchmarking/run_quickstart_tabarena.py) | 快速入门示例 |
| [`examples/benchmarking/run_quickstart_tabarena_one_dataset.py`](examples/benchmarking/run_quickstart_tabarena_one_dataset.py) | 单数据集运行示例 |
| [`examples/running_tabarena_models/run_default_model.py`](examples/running_tabarena_models/run_default_model.py) | 默认模型运行示例 |
| [`examples/plots/run_generate_main_leaderboard.py`](examples/plots/run_generate_main_leaderboard.py) | 生成 Leaderboard |

---

## 八、常见问题

### 8.1 如何加速 XGBoost 运行？

- 使用并行 bagging：`num_bag_folds=8` 时会自动并行训练
- 减少 `time_limit`：如果只需要快速验证
- 使用 `fold_fitting_strategy="sequential_local"` 便于调试

### 8.2 如何比较我的结果与论文结果？

运行 `examples/plots/run_generate_main_leaderboard.py` 后，查看 `XGBoost_c1_BAG_L1` 配置的性能指标，包括：
- ELO 分数
- Win-rate
- 平均排名
- 相对于最佳方法的改进率

### 8.3 如何使用自定义数据集？

参考 [`examples/benchmarking/run_quickstart_tabarena_custom_datasets.py`](examples/benchmarking/run_quickstart_tabarena_custom_datasets.py)，使用 `UserTask` 类定义自定义数据集。

### 8.4 内存不足怎么办？

TabArena 论文配置需要较大内存。可以通过以下方式减少资源使用：
- 减少 `num_bag_folds`（如设为 4 而非 8）
- 减少 `num_random_configs`
- 使用更小的数据集开始

---

## 九、快速参考命令

```bash
# 安装 TabArena
cd g:/VSCodeWork/TabArena
uv sync --extra benchmark

# 生成 Leaderboard
python examples/plots/run_generate_main_leaderboard.py

# 运行快速示例
cd examples/benchmarking
python run_quickstart_tabarena.py

# 运行单数据集示例（修改为使用 XGBoost）
python run_quickstart_tabarena_one_dataset.py

# 查看 XGBoost 配置
python -c "from tabarena.models.utils import get_configs_generator_from_name; print(get_configs_generator_from_name('XGBoost').generate_all_bag_experiments(1))"
```

---

## 十、引用

如果使用 TabArena 的 XGBoost 结果，请引用：

```bibtex
@article{tabarena2025,
  title={TabArena: A Living Benchmark for Machine Learning on Tabular Data},
  author={TabArena Team},
  year={2025},
  url={https://tabarena.ai/}
}
```
