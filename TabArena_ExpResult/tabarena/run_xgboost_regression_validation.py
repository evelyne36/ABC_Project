"""
TabArena XGBoost Regression 快速验证脚本
用于在 CPU 上快速验证 XGBoost 模型并对比论文结果

使用方法:
    python run_xgboost_regression_validation.py
"""

from __future__ import annotations
from pathlib import Path
import time

from autogluon.tabular.models import XGBoostModel
from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


# TabArena 51 中的 Regression 数据集列表（完整13个）
REGRESSION_DATASETS = [
    # (OpenML Task ID, 数据集名称, 样本数, 特征数)
    (363612, "airfoil_self_noise", 1503, 6),
    (363615, "Another-Dataset-on-used-Fiat-500", 1538, 8),
    (363625, "concrete_compressive_strength", 1030, 9),
    (363631, "diamonds", 53940, 10),
    (363672, "Food_Delivery_Time", 45451, 10),
    (363675, "healthcare_insurance_expenses", 1338, 7),
    (363678, "houses", 20640, 9),
    (363686, "miami_housing", 13776, 16),
    (363693, "physiochemical_protein", 45730, 10),
    (363697, "QSAR-TID-11", 5742, 1025),
    (363698, "QSAR_fish_toxicity", 907, 7),
    (363705, "superconductivity", 21263, 82),
    (363708, "wine_quality", 6497, 13),
]

# 用于快速验证的小数据集（更快完成）
QUICK_TEST_DATASETS = [
    (363612, "airfoil_self_noise", 1503, 6),        # ~1500 样本
    (363698, "QSAR_fish_toxicity", 907, 7),         # ~900 样本
    (363675, "healthcare_insurance_expenses", 1338, 7),  # ~1300 样本
]


def run_xgboost_regression_benchmark(
    task_ids: list,
    num_bag_folds: int = 3,  # 快速验证用 3 折（论文用 8 折）
    time_limit: int = 3600,   # 每任务 60 分钟（论文用 1 小时）
    output_dir: str = "xgboost_regression_output",
    use_default_hyperparameters: bool = True,
) -> dict:
    """
    运行 XGBoost Regression Benchmark
    
    参数:
        task_ids: OpenML Task ID 列表
        num_bag_folds: Bagging 折数（快速验证用 3，论文用 8）
        time_limit: 每任务时间限制（秒）
        output_dir: 输出目录
        use_default_hyperparameters: 是否使用默认超参数（True）或搜索空间（False）
    
    返回:
        结果字典
    """
    print("=" * 60)
    print("TabArena XGBoost Regression Benchmark")
    print("=" * 60)
    print(f"Task IDs: {task_ids}")
    print(f"Num bag folds: {num_bag_folds}")
    print(f"Time limit per task: {time_limit}s")
    print(f"Output directory: {output_dir}")
    print()
    
    # 获取 TabArena 任务元数据
    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata
    
    # 创建 XGBoost 实验配置
    if use_default_hyperparameters:
        # 使用默认超参数（最快）
        xgboost_exp = AGModelBagExperiment(
            name="XGBoost_c1_BAG_L1_QuickTest",
            model_cls=XGBoostModel,
            model_hyperparameters={
                # XGBoost 默认超参数
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
            num_bag_folds=num_bag_folds,
            time_limit=time_limit,
        )
    else:
        # 使用 TabArena 搜索空间（更慢但更全面）
        from tabarena.models.utils import get_configs_generator_from_name
        config_generator = get_configs_generator_from_name("XGBoost")
        
        return config_generator.generate_all_bag_experiments(
            num_random_configs=1,
            fold_fitting_strategy="sequential_local",
        )
    
    # 创建实验运行器
    exp_batch_runner = ExperimentBatchRunner(
        expname=output_dir,
        task_metadata=task_metadata
    )
    
    # 运行实验
    start_time = time.time()
    # 将 task_ids 转换为数据集名称
    dataset_names = [name for tid, name, *_ in REGRESSION_DATASETS if tid in task_ids]
    results_lst = exp_batch_runner.run(
        datasets=dataset_names,
        folds=[0],  # 只跑第一个 fold 用于快速验证
        methods=[xgboost_exp],
        ignore_cache=False,
    )
    elapsed_time = time.time() - start_time
    
    print()
    print("=" * 60)
    print(f"Benchmark 完成! 总耗时: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    print("=" * 60)
    
    return results_lst


def compare_with_paper_results(results: list, output_path: str = None) -> None:
    """
    将结果与论文中的 XGBoost 结果进行对比
    
    注意: 完整对比需要运行 examples/plots/run_generate_main_leaderboard.py
    生成完整 leaderboard 后查看 XGBoost_c1_BAG_L1 的排名和分数
    """
    print()
    print("=" * 60)
    print("对比论文结果")
    print("=" * 60)
    print()
    print("要查看完整的 XGBoost 论文结果对比, 请运行:")
    print("  python examples/plots/run_generate_main_leaderboard.py")
    print()
    print("或访问在线 Leaderboard:")
    print("  https://huggingface.co/spaces/TabArena/leaderboard")
    print()
    print("在 Leaderboard 中搜索 'XGBoost_c1_BAG_L1' 查看:")
    print("  - ELO 分数")
    print("  - Win-rate")
    print("  - 平均排名")
    print("  - 相对于最佳方法的改进率")
    print()


def main():
    """主函数: 运行 XGBoost Regression 完整13个数据集"""

    # 筛选 Regression 任务的 Task IDs
    regression_task_ids = [tid for tid, *_ in REGRESSION_DATASETS]

    print("TabArena 完整13个回归数据集列表:")
    print("-" * 60)
    for tid, name, n_samples, n_features in REGRESSION_DATASETS:
        print(f"  {tid}: {name} (样本={n_samples}, 特征={n_features})")
    print()

    # 运行完整的13个回归数据集
    results = run_xgboost_regression_benchmark(
        task_ids=regression_task_ids,
        num_bag_folds=4,     # 论文标准：8折
        time_limit=3600,     # 论文标准：1小时
        output_dir="xgboost_regression_output",
        use_default_hyperparameters=False, #是否使用默认参数
    )

    # 打印结果摘要
    print()
    print("结果摘要:")
    print("-" * 60)
    for r in results:
        task_name = r.get("task_name", "Unknown")
        metric_error = r.get("metric_error", "N/A")
        metric_error_val = r.get("metric_error_val", "N/A")
        print(f"  {task_name}:")
        print(f"    Validation Error: {metric_error_val}")
        print(f"    Test Error: {metric_error}")

    # 对比论文结果
    compare_with_paper_results(results)


if __name__ == "__main__":
    main()
