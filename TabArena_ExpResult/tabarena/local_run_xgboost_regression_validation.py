"""
XGBoost 回归 Baseline - 使用本地CSV数据集
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd

from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# 13个回归数据集
REGRESSION_DATASETS = [
    "airfoil_self_noise",
    "Another-Dataset-on-used-Fiat-500",
    "concrete_compressive_strength",
    "diamonds",
    "Food_Delivery_Time",
    "healthcare_insurance_expenses",
    "houses",
    "miami_housing",
    "physiochemical_protein",
    "QSAR-TID-11",
    "QSAR_fish_toxicity",
    "superconductivity",
    "wine_quality",
]

# 数据集目录
DATA_DIR = Path("/home/admin/MyProject5/TabArena/tabarena/tabArena_regression")


def run_xgboost_on_dataset(dataset_name: str, num_folds: int = 8, time_limit: int = 3600):
    """在单个数据集上运行XGBoost"""
    print(f"\n{'='*60}")
    print(f"运行数据集: {dataset_name}")
    print(f"{'='*60}")

    # 读取数据
    csv_path = DATA_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)

    print(f"数据形状: {df.shape}")

    # 假设最后一列是目标变量
    target_col = df.columns[-1]
    print(f"目标列: {target_col}")

    # K折交叉验证
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\n--- Fold {fold_idx + 1}/{num_folds} ---")

        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]

        # 创建预测器
        predictor = TabularPredictor(
            label=target_col,
            problem_type='regression',
            eval_metric='root_mean_squared_error',
            path=f"xgboost_output/{dataset_name}/fold_{fold_idx}"
        )

        # 训练XGBoost
        predictor.fit(
            train_data=train_data,
            hyperparameters={'XGB': {}},
            time_limit=time_limit // num_folds,
            verbosity=2
        )

        # 预测
        y_pred = predictor.predict(test_data.drop(columns=[target_col]))
        y_true = test_data[target_col]

        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        fold_results.append({
            'fold': fold_idx,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 汇总结果
    avg_rmse = np.mean([r['rmse'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])

    print(f"\n平均结果:")
    print(f"  RMSE: {avg_rmse:.4f}")
    print(f"  MAE: {avg_mae:.4f}")
    print(f"  R2: {avg_r2:.4f}")

    return {
        'dataset': dataset_name,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'fold_results': fold_results
    }


def main():
    """主函数"""
    print("XGBoost 回归 Baseline - 13个数据集")
    print("="*60)

    all_results = []
    start_time = time.time()

    for dataset_name in REGRESSION_DATASETS:
        try:
            result = run_xgboost_on_dataset(
                dataset_name=dataset_name,
                num_folds=8,
                time_limit=3600
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n错误: {dataset_name} - {e}")
            continue

    elapsed_time = time.time() - start_time

    # 打印总结
    print(f"\n\n{'='*60}")
    print("所有数据集结果总结")
    print(f"{'='*60}")
    print(f"总耗时: {elapsed_time/3600:.2f} 小时")
    print()

    for result in all_results:
        print(f"{result['dataset']}:")
        print(f"  RMSE: {result['avg_rmse']:.4f}")
        print(f"  MAE: {result['avg_mae']:.4f}")
        print(f"  R2: {result['avg_r2']:.4f}")
        print()


if __name__ == "__main__":
    main()
