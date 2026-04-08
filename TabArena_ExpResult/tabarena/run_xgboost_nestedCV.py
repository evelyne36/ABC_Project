"""
XGBoost嵌套交叉验证脚本 - 复刻TabArena论文结果
Nested CV: Outer 3 folds (repeated) with Inner 8 folds
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tabarena.benchmark.experiment import run_experiments_new
from tabarena.benchmark.task import UserTask
from tabarena.models.utils import get_configs_generator_from_name
from tabarena.nips2025_utils.end_to_end import EndToEnd

def create_nested_cv_splits(dataset, n_outer=3, n_inner=8, random_state=42):
    """创建嵌套CV分割: outer 3 folds, inner 8 folds"""
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

def run_xgboost_nested_cv(
    dataset_path: str,
    target_column: str = None,
    output_dir: str = "./xgb_nested_cv_results",
    task_name: str = "RegressionTask",
    num_random_configs: int = 200,
):
    """
    运行XGBoost嵌套交叉验证

    参数:
        dataset_path: 数据集CSV文件路径
        target_column: 目标列名称，默认None则使用最后一列
        output_dir: 输出目录
        task_name: 任务名称
        num_random_configs: 随机搜索配置数量(论文中为200)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    task_cache_dir = output_path / "task_cache"
    task_cache_dir.mkdir(exist_ok=True)

    # 加载数据集
    dataset = pd.read_csv(dataset_path)

    # 如果未指定目标列，使用最后一列
    if target_column is None:
        target_column = dataset.columns[-1]
        print(f"使用最后一列作为目标: {target_column}")

    # 创建嵌套CV分割
    splits = create_nested_cv_splits(dataset, n_outer=3, n_inner=8)

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

    # 运行实验
    print(f"开始运行XGBoost嵌套CV实验...")
    results_lst = run_experiments_new(
        output_dir=str(output_path / "experiments"),
        model_experiments=model_experiments,
        tasks=[user_task],
        repetitions_mode="matrix",
        repetitions_mode_args=([0, 1, 2], [0]),  # 3 folds, 1 repeat/3repeats
    )

    # 处理结果
    task_metadata = pd.DataFrame([
        [user_task.task_id, task_name, "regression", task_name,
         int(len(dataset) * 0.67), int(len(dataset) * 0.33)]
    ], columns=["tid", "name", "task_type", "dataset",
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

    # 保存结果（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_summary.to_csv(output_path / f"xgb_metrics_summary_{timestamp}.csv", index=False)
    df_results.to_csv(output_path / f"xgb_full_results_{timestamp}.csv", index=False)

    print("\n=== XGBoost嵌套CV结果 ===")
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
    # 批量处理本地回归数据集
    dataset_dir = Path("./tabArena_regression")
    output_base_dir = "./xgb_nested_cv_results"

    csv_files = list(dataset_dir.glob("*.csv"))
    print(f"找到 {len(csv_files)} 个数据集")

    for csv_file in csv_files:
        dataset_name = csv_file.stem
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*60}")

        try:
            metrics, results = run_xgboost_nested_cv(
                dataset_path=str(csv_file),
                target_column=None,  # 自动使用最后一列
                output_dir=f"{output_base_dir}/{dataset_name}",
                task_name=dataset_name,
                num_random_configs=200,
            )
        except Exception as e:
            print(f"错误: {dataset_name} 处理失败 - {e}")
            continue

