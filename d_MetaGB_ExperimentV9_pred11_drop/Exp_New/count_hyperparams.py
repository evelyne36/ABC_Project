"""
Hyperparameter Combination Counter
Calculates the number of hyperparameter combinations for each model in experiments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from utils.hyperparams import (
        get_xgb_param_grid, get_metagb_param_grid,
        get_exp2a_xgb_param_grid, get_exp2a_metagb_param_grid,
        get_exp2b_xgb_param_grid, get_exp2b_metagb_param_grid,
        EXP1_DATASETS, CV_FOLDS
    )
except ImportError:
    from hyperparams import (
        get_xgb_param_grid, get_metagb_param_grid,
        get_exp2a_xgb_param_grid, get_exp2a_metagb_param_grid,
        get_exp2b_xgb_param_grid, get_exp2b_metagb_param_grid,
        EXP1_DATASETS, CV_FOLDS
    )


def count_combinations(param_grid):
    """Calculate total combinations from parameter grid."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def print_grid_details(param_grid, model_name):
    """Print parameter grid details."""
    print(f"\n{model_name} 参数网格:")
    print("-" * 50)
    for param, values in param_grid.items():
        print(f"  {param:25s}: {len(values):2d} 个值 {values}")
    total = count_combinations(param_grid)
    print(f"\n  总组合数: {total:,}")
    return total


def main():
    print("=" * 80)
    print("超参数组合数量统计")
    print("=" * 80)

    # Experiment 1
    print("\n【Experiment 1: Real Datasets】")
    print(f"数据集数量: {len(EXP1_DATASETS)}")
    print(f"交叉验证折数: {CV_FOLDS}")

    dataset = EXP1_DATASETS[0]
    print(f"\n以 {dataset} 为例:")

    xgb_grid = get_xgb_param_grid(dataset)
    metagb_grid = get_metagb_param_grid(dataset)

    xgb_total = print_grid_details(xgb_grid, "XGBoost")
    metagb_total = print_grid_details(metagb_grid, "MetaGB")

    print(f"\n每个数据集的训练次数:")
    print(f"  XGBoost: {xgb_total:,} 组合 × {CV_FOLDS} 折 = {xgb_total * CV_FOLDS:,} 次训练")
    print(f"  MetaGB:  {metagb_total:,} 组合 × {CV_FOLDS} 折 = {metagb_total * CV_FOLDS:,} 次训练")

    print(f"\n所有数据集的总训练次数:")
    print(f"  XGBoost: {xgb_total * CV_FOLDS * len(EXP1_DATASETS):,} 次")
    print(f"  MetaGB:  {metagb_total * CV_FOLDS * len(EXP1_DATASETS):,} 次")

    # Experiment 2A
    print("\n" + "=" * 80)
    print("【Experiment 2A: Synthetic Nonlinear】")

    xgb_2a = get_exp2a_xgb_param_grid()
    metagb_2a = get_exp2a_metagb_param_grid()

    xgb_2a_total = print_grid_details(xgb_2a, "XGBoost")
    metagb_2a_total = print_grid_details(metagb_2a, "MetaGB")

    # Experiment 2B
    print("\n" + "=" * 80)
    print("【Experiment 2B: Synthetic Interaction】")

    xgb_2b = get_exp2b_xgb_param_grid()
    metagb_2b = get_exp2b_metagb_param_grid()

    xgb_2b_total = print_grid_details(xgb_2b, "XGBoost")
    metagb_2b_total = print_grid_details(metagb_2b, "MetaGB")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
