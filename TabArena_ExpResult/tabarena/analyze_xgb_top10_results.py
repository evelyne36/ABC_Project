"""
XGBoost嵌套CV结果分析脚本 - Top 10模型统计

功能：
1. 加载13个回归数据集上200个XGBoost配置的结果
2. 计算每个模型的验证集和测试集指标（RMSE, MAE, R²）
3. 选择每个数据集上的Top 10模型（基于验证集RMSE）
4. 生成详细的结果报告和汇总统计

使用方法：
    python analyze_xgb_top10_results.py
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm


def load_model_results(results_path: Path) -> Dict:
    """加载单个模型的results.pkl文件"""
    try:
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load {results_path}: {e}")
        return None


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算RMSE, MAE, R²指标"""
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    except Exception as e:
        warnings.warn(f"Failed to calculate metrics: {e}")
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}


def extract_model_metrics(results_data: Dict) -> Dict[str, float]:
    """从results.pkl中提取所有指标"""
    if results_data is None:
        return None

    try:
        sim = results_data['simulation_artifacts']
        framework = results_data['framework']

        # 提取真实标签
        y_val = np.array(sim['y_val'])
        y_test = np.array(sim['y_test'])

        # 提取预测结果（从字典中取出）
        pred_val = list(sim['pred_proba_dict_val'].values())[0]
        pred_test = list(sim['pred_proba_dict_test'].values())[0]

        # 计算验证集指标
        val_metrics = calculate_all_metrics(y_val, pred_val)

        # 计算测试集指标
        test_metrics = calculate_all_metrics(y_test, pred_test)

        return {
            'model_name': framework,
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'time_train_s': results_data.get('time_train_s', np.nan),
            'time_infer_s': results_data.get('time_infer_s', np.nan),
        }
    except Exception as e:
        warnings.warn(f"Failed to extract metrics: {e}")
        return None


def process_dataset(dataset_path: Path, dataset_name: str) -> pd.DataFrame:
    """处理单个数据集的所有模型结果"""
    print(f"\n处理数据集: {dataset_name}")

    # 查找所有results.pkl文件
    experiments_dir = dataset_path / "experiments" / "data"
    if not experiments_dir.exists():
        warnings.warn(f"Experiments directory not found: {experiments_dir}")
        return pd.DataFrame()

    results_files = list(experiments_dir.glob("*/*/0_4/results.pkl"))
    print(f"  找到 {len(results_files)} 个模型结果文件")

    if len(results_files) == 0:
        warnings.warn(f"No results files found in {experiments_dir}")
        return pd.DataFrame()

    # 加载所有模型结果
    all_metrics = []
    for results_path in tqdm(results_files, desc=f"  加载{dataset_name}结果"):
        results_data = load_model_results(results_path)
        metrics = extract_model_metrics(results_data)
        if metrics is not None:
            metrics['dataset'] = dataset_name
            all_metrics.append(metrics)

    if len(all_metrics) == 0:
        warnings.warn(f"No valid metrics extracted for {dataset_name}")
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics)
    print(f"  成功加载 {len(df)} 个模型的指标")

    return df


def select_top10_models(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """根据验证集RMSE选择Top K模型"""
    # 按验证集RMSE升序排序（越小越好）
    df_sorted = df.sort_values('val_rmse', ascending=True).reset_index(drop=True)

    # 选择Top K
    df_top = df_sorted.head(top_k).copy()
    df_top['rank'] = range(1, len(df_top) + 1)

    return df_top


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """生成汇总统计信息"""
    metrics = ['val_rmse', 'val_mae', 'val_r2', 'test_rmse', 'test_mae', 'test_r2']

    summary_data = []
    for dataset in df['dataset'].unique():
        df_dataset = df[df['dataset'] == dataset]
        for metric in metrics:
            summary_data.append({
                'dataset': dataset,
                'metric': metric,
                'mean': df_dataset[metric].mean(),
                'std': df_dataset[metric].std(),
                'min': df_dataset[metric].min(),
                'max': df_dataset[metric].max(),
                'median': df_dataset[metric].median(),
            })

    return pd.DataFrame(summary_data)


def analyze_top_models_across_datasets(df_all_top10: pd.DataFrame) -> pd.DataFrame:
    """分析Top 10模型在不同数据集上的表现"""
    # 统计每个模型出现在Top 10中的次数
    model_counts = df_all_top10['model_name'].value_counts().reset_index()
    model_counts.columns = ['model_name', 'top10_count']

    # 计算每个模型的平均排名
    model_avg_rank = df_all_top10.groupby('model_name')['rank'].mean().reset_index()
    model_avg_rank.columns = ['model_name', 'avg_rank']

    # 合并统计信息
    model_stats = model_counts.merge(model_avg_rank, on='model_name')
    model_stats = model_stats.sort_values('top10_count', ascending=False)

    return model_stats


def main():
    """主函数"""
    # 配置路径
    results_base_dir = Path("./xgb_nested_cv_results_fast")
    output_dir = Path("./analysis_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("XGBoost嵌套CV结果分析 - Top 10模型统计")
    print("="*60)

    # 获取所有数据集文件夹
    dataset_dirs = [d for d in results_base_dir.iterdir() if d.is_dir()]
    print(f"\n找到 {len(dataset_dirs)} 个数据集")

    # 处理所有数据集
    all_results = []
    all_top10_results = []

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name

        # 处理单个数据集
        df_dataset = process_dataset(dataset_dir, dataset_name)

        if len(df_dataset) == 0:
            continue

        # 保存完整结果
        all_results.append(df_dataset)

        # 选择Top 10模型
        df_top10 = select_top10_models(df_dataset, top_k=10)
        all_top10_results.append(df_top10)

        # 显示Top 10模型
        print(f"\n  {dataset_name} - Top 10模型:")
        print(df_top10[['rank', 'model_name', 'val_rmse', 'val_mae', 'val_r2',
                        'test_rmse', 'test_mae', 'test_r2']].to_string(index=False))

    # 合并所有结果
    if len(all_results) == 0:
        print("\n错误: 没有成功处理任何数据集")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    df_all_top10 = pd.concat(all_top10_results, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"结果汇总:")
    print(f"  总数据集数: {len(dataset_dirs)}")
    print(f"  成功处理数据集数: {df_all['dataset'].nunique()}")
    print(f"  总模型数: {len(df_all)}")
    print(f"  Top 10模型总数: {len(df_all_top10)}")
    print(f"{'='*60}")

    # 生成汇总统计
    df_summary = generate_summary_statistics(df_all_top10)

    # 分析Top模型跨数据集表现
    df_top_models = analyze_top_models_across_datasets(df_all_top10)

    # 保存结果
    print(f"\n保存结果到: {output_dir}")

    # 1. 每个数据集的Top 10详细结果
    try:
        output_file = output_dir / "per_dataset_top10_detailed.csv"
        df_all_top10.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  [OK] {output_file.name}")
    except Exception as e:
        print(f"  [FAIL] Failed to save per_dataset_top10_detailed.csv: {e}")

    # 2. 汇总统计
    try:
        output_file = output_dir / "summary_statistics.csv"
        df_summary.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  [OK] {output_file.name}")
    except Exception as e:
        print(f"  [FAIL] Failed to save summary_statistics.csv: {e}")

    # 3. Top模型跨数据集比较
    try:
        output_file = output_dir / "top10_models_comparison.csv"
        df_top_models.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  [OK] {output_file.name}")
    except Exception as e:
        print(f"  [FAIL] Failed to save top10_models_comparison.csv: {e}")

    # 4. 所有模型的完整结果（可选）
    try:
        output_file = output_dir / "all_models_full_results.csv"
        df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  [OK] {output_file.name}")
    except Exception as e:
        print(f"  [FAIL] Failed to save all_models_full_results.csv: {e}")

    # 5. 保存为Excel（多个sheet）
    try:
        output_file = output_dir / "xgb_top10_analysis_report.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_all_top10.to_excel(writer, sheet_name='Top10_Detailed', index=False)
            df_summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            df_top_models.to_excel(writer, sheet_name='Top_Models_Comparison', index=False)
        print(f"  [OK] {output_file.name}")
    except Exception as e:
        print(f"  [FAIL] Failed to save Excel file: {e}")

    # 显示最常出现在Top 10中的模型
    print(f"\n{'='*60}")
    print("最常出现在Top 10中的模型:")
    print(f"{'='*60}")
    print(df_top_models.head(20).to_string(index=False))

    print(f"\n{'='*60}")
    print("分析完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
