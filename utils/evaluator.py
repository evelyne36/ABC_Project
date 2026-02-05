"""
评估模块
用于计算模型性能指标
"""

import numpy as np
import json
import os
from datetime import datetime


def calculate_metrics(y_true, y_pred):
    """
    计算多种回归评估指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
    
    返回:
        指标字典
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def save_results(results_dict, results_dir="results"):
    """
    保存实验结果到JSON文件
    
    参数:
        results_dict: 结果字典
        results_dir: 结果保存目录
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成文件名
    exp_type = results_dict.get('exp_type', 'unknown')
    dataset = results_dict.get('dataset', 'unknown')
    model = results_dict.get('model', 'unknown')
    seed = results_dict.get('seed', 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{exp_type}-{dataset}-{seed}-{model}-{timestamp}.json"
    save_path = os.path.join(results_dir, filename)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    return save_path


def summarize_grid_search_results(results_list, param_grid):
    """
    汇总网格搜索结果，分析参数影响
    
    参数:
        results_list: 结果列表
        param_grid: 参数网格
    
    返回:
        汇总字典
    """
    param_names = list(param_grid.keys())
    
    # 按参数分组计算平均MSE
    param_importance = {}
    for param_name in param_names:
        param_values = sorted(list(set([r['params'][param_name] for r in results_list])))
        
        mse_by_value = {}
        for value in param_values:
            # 找到所有使用该参数值的结果
            relevant_results = [r for r in results_list if r['params'][param_name] == value]
            # 计算该参数值的平均验证集MSE
            avg_val_mse = np.mean([r['val_mse'] for r in relevant_results])
            mse_by_value[value] = avg_val_mse
        
        # 计算该参数的方差（方差越大，说明该参数越重要）
        mse_values = list(mse_by_value.values())
        variance = np.var(mse_values)
        param_importance[param_name] = {
            'values': mse_by_value,
            'variance': float(variance),
            'best_value': min(mse_by_value, key=mse_by_value.get)
        }
    
    # 找出最重要的参数（方差最大）
    best_param = max(param_importance.keys(), 
                      key=lambda k: param_importance[k]['variance'])
    
    return {
        'param_importance': param_importance,
        'most_important_param': best_param,
        'best_param_value': param_importance[best_param]['best_value']
    }


def find_best_result(results_list, metric='val_mse'):
    """
    从结果列表中找到最佳结果
    
    参数:
        results_list: 结果列表
        metric: 评估指标（默认为验证集MSE）
    
    返回:
        最佳结果字典
    """
    if not results_list:
        return None
    
    best_idx = np.argmin([r[metric] for r in results_list])
    return results_list[best_idx]
