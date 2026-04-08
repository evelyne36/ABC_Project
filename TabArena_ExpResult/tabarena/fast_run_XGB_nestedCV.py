"""
XGBoost嵌套交叉验证脚本 - 快速优化版本
Nested CV: Outer 3 folds (repeated) with Inner 8 folds

优化版本改进:
1. 配置数量从200降到20 (速度提升10倍)
2. Bagging folds从8降到2 (速度提升4倍)
3. 使用TabArena-Lite模式只运行第一个fold (速度提升3倍)
4. 启用缓存机制避免重复运行
5. 支持进度显示和时间估算

综合速度提升: 约60-100倍
预估运行时间: 10-30分钟 (单数据集)
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from tabarena.benchmark.experiment import run_experiments_new
from tabarena.benchmark.task import UserTask
from tabarena.models.utils import get_configs_generator_from_name
from tabarena.nips2025_utils.end_to_end import EndToEnd


def create_nested_cv_splits(dataset, n_outer=5, n_inner=4, random_state=42):
    """创建嵌套CV分割: outer 5 folds, inner 4 folds"""
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    splits = {}
    outer_kf = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    for fold_i, (train_val_idx, test_idx) in enumerate(outer_kf.split(indices)):
        splits[0] = splits.get(0, {})
        splits[0][fold_i] = (indices[train_val_idx].tolist(), indices[test_idx].tolist())

    return splits


def run_xgboost_nested_cv_fast(
    dataset_path: str,
    target_column: str = None,
    output_dir: str = "./xgb_nested_cv_results_fast",
    task_name: str = "RegressionTask",
    num_random_configs: int = 200,
    use_lite_mode: bool = False,
):
    """
    快速版XGBoost嵌套交叉验证

    优化点:
    1. 配置数量从200降到20 (可调整)
    2. 使用TabArena-Lite模式只运行第一个fold (可关闭)
    3. 启用缓存机制避免重复运行

    参数:
        dataset_path: 数据集CSV文件路径
        target_column: 目标列名称,默认None则使用最后一列
        output_dir: 输出目录
        task_name: 任务名称
        num_random_configs: 随机搜索配置数量 (默认20,论文中为200)
        use_lite_mode: 是否使用Lite模式 (默认True,只运行第一个fold)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    task_cache_dir = output_path / "task_cache"
    task_cache_dir.mkdir(exist_ok=True)

    # 加载数据集
    print(f"加载数据集: {dataset_path}")
    dataset = pd.read_csv(dataset_path)
    print(f"数据集大小: {len(dataset)} 行 × {len(dataset.columns)} 列")

    # 如果未指定目标列,使用最后一列
    if target_column is None:
        target_column = dataset.columns[-1]
        print(f"使用最后一列作为目标: {target_column}")

    # 创建嵌套CV分割
    splits = create_nested_cv_splits(dataset, n_outer=5, n_inner=4)# 论文中是3外层folds和8内层folds,这里为匹配改为5外层和4内层

    # 创建UserTask
    user_task = UserTask(
        task_name=task_name,
        task_cache_path=task_cache_dir,
    )
    oml_task = user_task.create_local_openml_task(
        dataset=dataset,
        target_feature=target_column,
        problem_type="regression",
        splits=splits,
    )
    user_task.save_local_openml_task(oml_task)

    # 获取XGBoost配置生成器
    config_generator = get_configs_generator_from_name("XGBoost")
    model_experiments = config_generator.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        fold_fitting_strategy="sequential_local",
        
    )

    # 计算预期运行时间
    if use_lite_mode:
        num_folds = 1
        mode_str = "TabArena-Lite"
    else:
        num_folds = 5
        mode_str = "TabArena"

    # 估算总训练次数
    num_bag_folds = 4  # 默认bagging folds
    total_trainings = num_random_configs * num_bag_folds * num_folds

    print(f"\n{'='*60}")
    print(f"实验配置:")
    print(f"  配置数量: {num_random_configs}")
    print(f"  Bagging folds: {num_bag_folds}")
    print(f"  外层folds: {num_folds} ({mode_str})")
    print(f"  总训练次数: {total_trainings:,}")
    print(f"  缓存模式: 启用 (跳过已完成的实验)")
    print(f"{'='*60}\n")

    # 运行实验
    start_time = time.time()
    print(f"开始运行XGBoost嵌套CV实验 (快速版)...")

    repetitions_mode = "TabArena-Lite" if use_lite_mode else "matrix"

    if use_lite_mode:
        # Lite模式: 只运行第一个fold的第一个repeat
        results_lst = run_experiments_new(
            output_dir=str(output_path / "experiments"),
            model_experiments=model_experiments,
            tasks=[user_task],
            repetitions_mode="TabArena-Lite",
            cache_mode="default",  # 启用缓存
        )
    else:
        # 完整模式: 运行所有5个folds
        results_lst = run_experiments_new(
            output_dir=str(output_path / "experiments"),
            model_experiments=model_experiments,
            tasks=[user_task],
            repetitions_mode="matrix",
            repetitions_mode_args=([4], [0]),  # 1folds/5 , 1 repeat
            cache_mode="default",  # 启用缓存
        )

    elapsed_time = time.time() - start_time
    print(f"\n实验完成! 总耗时: {elapsed_time/60:.1f} 分钟")

    # 处理结果
    task_metadata = pd.DataFrame([[
        user_task.task_id, task_name, "regression", task_name,
        int(len(dataset) * 0.8), int(len(dataset) * 0.2)
    ]], columns=["tid", "name", "task_type", "dataset",
                "n_samples_train_per_fold", "n_samples_test_per_fold"])

    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=task_metadata,
        cache=False,
        cache_raw=False,
    )
    end_to_end_results = end_to_end.to_results()
    df_results = end_to_end_results.get_results()

    # 提取指标
    metrics_summary = extract_metrics(df_results)

    # 保存结果(带时间戳)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_summary.to_csv(output_path / f"xgb_metrics_summary_fast_{timestamp}.csv", index=False)
    df_results.to_csv(output_path / f"xgb_full_results_fast_{timestamp}.csv", index=False)

    print("\n=== XGBoost嵌套CV快速版结果 ===")
    print(metrics_summary.to_string(index=False))
    print(f"\n结果已保存至: {output_path}")

    return metrics_summary, df_results


def extract_metrics(df_results):
    """提取验证集和测试集的RMSE, MAE, R2指标"""
    metrics = []

    for _, row in df_results.iterrows():
        metrics.append({
            'model': row.get('framework', 'XGBoost'),
            'fold': row.get('fold', 0),
            'val_rmse': row.get('metric_error_val', np.nan),
            'val_mae': row.get('metric_error_val_mae', np.nan),
            'val_r2': row.get('metric_error_val_r2', np.nan),
            'test_rmse': row.get('metric_error', np.nan),
            'test_mae': row.get('metric_error_mae', np.nan),
            'test_r2': row.get('metric_error_r2', np.nan),
        })

    metrics_df = pd.DataFrame(metrics)

    # 计算平均值和标准差
    summary = pd.DataFrame([{
        'metric': 'mean',
        'val_rmse': metrics_df['val_rmse'].mean(),
        'val_mae': metrics_df['val_mae'].mean(),
        'val_r2': metrics_df['val_r2'].mean(),
        'test_rmse': metrics_df['test_rmse'].mean(),
        'test_mae': metrics_df['test_mae'].mean(),
        'test_r2': metrics_df['test_r2'].mean(),
    }, {
        'metric': 'std',
        'val_rmse': metrics_df['val_rmse'].std(),
        'val_mae': metrics_df['val_mae'].std(),
        'val_r2': metrics_df['val_r2'].std(),
        'test_rmse': metrics_df['test_rmse'].std(),
        'test_mae': metrics_df['test_mae'].std(),
        'test_r2': metrics_df['test_r2'].std(),
    }])

    return summary


if __name__ == "__main__":
    # 配置选项
    NUM_RANDOM_CONFIGS = 200  # 配置数量 (默认20,论文用200)
    USE_LITE_MODE = False  # 是否使用Lite模式 (默认True)
    RUN_SINGLE_DATASET = False  # 是否只运行第一个数据集

    # 批量处理本地回归数据集
    dataset_dir = Path("./tabArena_regression")
    output_base_dir = "./xgb_nested_cv_results_fast"

    csv_files = sorted(dataset_dir.glob("*.csv"))

    if RUN_SINGLE_DATASET:
        csv_files = csv_files[:1]  # 只运行第一个数据集

    print(f"{'='*60}")
    print(f"XGBoost嵌套交叉验证 - 快速优化版本")
    print(f"{'='*60}")
    print(f"找到 {len(csv_files)} 个数据集{' (快速模式: 只运行第一个)' if RUN_SINGLE_DATASET else ''}")
    print(f"优化配置:")
    print(f"  - 配置数量: {NUM_RANDOM_CONFIGS} (论文默认: 200)")
    print(f"  - Lite模式: {'启用' if USE_LITE_MODE else '禁用'} ({'只运行1个fold' if USE_LITE_MODE else '运行3个folds'})")
    print(f"{'='*60}\n")

    for csv_file in csv_files:
        dataset_name = csv_file.stem
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*60}")

        try:
            metrics, results = run_xgboost_nested_cv_fast(
                dataset_path=str(csv_file),
                target_column=None,  # 自动使用最后一列
                output_dir=f"{output_base_dir}/{dataset_name}",
                task_name=dataset_name,
                num_random_configs=NUM_RANDOM_CONFIGS,
                use_lite_mode=USE_LITE_MODE,
            )
        except Exception as e:
            print(f"错误: {dataset_name} 处理失败 - {e}")
            import traceback
            traceback.print_exc()
            continue
