"""
主实验运行脚本 - MetaGB 完整实验流程

根据实验计划_until0204.md设计的全流程实验框架
包含模块A、B、C的所有实验
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ABC_Project', 'metagb'))

from ABC_Project.metagb.metagb_utils import MetaGB
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from itertools import product
import time
import pandas as pd
from utils.visualizer import plot_mse_loss


# 设置中文字体
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def print_banner():
    """打印实验横幅"""
    banner = """
    ================================================================
                MetaGB 完整实验流程 - 主运行脚本
    ================================================================
    """
    print(banner)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MetaGB完整实验流程')
    
    parser.add_argument('--module', type=str, default='all',
                      choices=['A', 'B', 'C', 'all'],
                      help='运行哪个实验模块 (A/B/C/all)')
    
    parser.add_argument('--cv', type=int, default=10,
                      help='交叉验证折数 (默认: 10)')
    
    parser.add_argument('--results_dir', type=str, default='results',
                      help='结果保存目录 (默认: results)')
    
    parser.add_argument('--logs_dir', type=str, default='logs',
                      help='日志保存目录 (默认: logs)')
    
    parser.add_argument('--plots_dir', type=str, default='plots',
                      help='图表保存目录 (默认: plots)')
    
    parser.add_argument('--n_jobs', type=int, default=1,
                      help='并行任务数 (默认: 1)')
    
    return parser.parse_args()


def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('Experiment')


# ============================================================================
# 训练曲线跟踪和分析函数
# ============================================================================

def train_xgboost_with_tracking(model, X_train, y_train, X_val, y_val, logger, eval_interval=10):
    """
    训练XGBoost并记录训练/验证MSE
    
    参数:
        model: XGBoost模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        logger: 日志记录器
        eval_interval: 评估间隔（树的数量）
    
    返回:
        train_losses: 训练集MSE列表
        val_losses: 验证集MSE列表
        final_model: 训练完成后的模型
    """
    n_estimators = model.get_params()['n_estimators']
    train_losses = []
    val_losses = []
    
    # 使用warm_start逐步增加树的数量
    for i in range(1, n_estimators + 1, eval_interval):
        current_estimators = min(i, n_estimators)
        model.set_params(n_estimators=current_estimators)
        
        # 重新拟合（XGBoost不支持warm_start，所以需要重新训练）
        temp_model = XGBRegressor(
            objective=model.get_params()['objective'],
            tree_method=model.get_params()['tree_method'],
            random_state=model.get_params()['random_state'],
            n_jobs=model.get_params()['n_jobs'],
            **{k: v for k, v in model.get_params().items() if k not in ['objective', 'tree_method', 'random_state', 'n_jobs']}
        )
        temp_model.set_params(n_estimators=current_estimators)
        temp_model.fit(X_train, y_train)
        
        # 计算MSE
        train_pred = temp_model.predict(X_train)
        val_pred = temp_model.predict(X_val)
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        train_losses.append(train_mse)
        val_losses.append(val_mse)
        
        if current_estimators % (eval_interval * 5) == 0 or current_estimators == n_estimators:
            logger.info(f"  XGBoost [{current_estimators}/{n_estimators} trees] - "
                       f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
    
    # 训练最终模型
    model.fit(X_train, y_train)
    
    return train_losses, val_losses, model


def train_metagb_with_tracking(model, X_train, y_train, X_val, y_val, logger, eval_interval=10):
    """
    训练MetaGB并记录训练/验证MSE
    
    参数:
        model: MetaGB模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        logger: 日志记录器
        eval_interval: 评估间隔（树的数量）
    
    返回:
        train_losses: 训练集MSE列表
        val_losses: 验证集MSE列表
        final_model: 训练完成后的模型
    """
    num_trees = model.num_trees
    train_losses = []
    val_losses = []
    
    # 逐步增加树的数量
    for i in range(1, num_trees + 1, eval_interval):
        current_trees = min(i, num_trees)
        
        # 训练当前数量的树
        temp_model = MetaGB(
            learning_rate=model.learning_rate,
            num_trees=current_trees,
            feature_fraction=model.feature_fraction,
            pred_feature_fraction=model.pred_feature_fraction,
            bagging_fraction=model.bagging_fraction,
            min_data_in_leaf=model.min_data_in_leaf,
            max_depth=model.max_depth,
            max_bin=model.max_bin,
            version=model.version,
            seed=model.seed,
        )
        temp_model.fit(X_train, y_train)
        
        # 计算MSE
        train_pred = temp_model.predict(X_train)
        val_pred = temp_model.predict(X_val)
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        train_losses.append(train_mse)
        val_losses.append(val_mse)
        
        if current_trees % (eval_interval * 5) == 0 or current_trees == num_trees:
            logger.info(f"  MetaGB [{current_trees}/{num_trees} trees] - "
                       f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
    
    # 训练最终模型
    model.fit(X_train, y_train)
    
    return train_losses, val_losses, model


def analyze_training_curves(train_losses, val_losses, logger):
    """
    分析训练曲线并记录结果
    
    分析内容:
    1. 最终训练/验证MSE
    2. 最佳验证集MSE及对应轮次
    3. 是否过拟合（验证集MSE增长）
    4. 收敛情况（MSE下降趋势）
    
    返回:
        analysis_dict: 分析结果字典
    """
    analysis_dict = {}
    
    final_train_mse = train_losses[-1]
    final_val_mse = val_losses[-1]
    best_val_mse = min(val_losses)
    best_val_epoch = val_losses.index(best_val_mse) + 1
    
    # 计算MSE改善比例
    initial_train = train_losses[0]
    initial_val = val_losses[0]
    train_improvement = (initial_train - final_train_mse) / initial_train * 100
    val_improvement = (initial_val - final_val_mse) / initial_val * 100
    
    # 检测过拟合
    overfitting_threshold = 0.05  # 5%增长即为过拟合
    is_overfitting = (final_val_mse - best_val_mse) / best_val_mse > overfitting_threshold
    
    # 检测收敛（最后20%轮次改善小于5%）
    last_20_pct = max(int(len(val_losses) * 0.2), 1)
    recent_min = min(val_losses[-last_20_pct:])
    is_converged = (final_val_mse - recent_min) / recent_min < 0.05
    
    analysis_dict = {
        'final_train_mse': final_train_mse,
        'final_val_mse': final_val_mse,
        'best_val_mse': best_val_mse,
        'best_val_epoch': best_val_epoch,
        'train_improvement_pct': train_improvement,
        'val_improvement_pct': val_improvement,
        'is_overfitting': is_overfitting,
        'is_converged': is_converged,
        'val_mse_increase_from_best': (final_val_mse - best_val_mse) / best_val_mse * 100
    }
    
    # 记录分析结果
    logger.info("="*60)
    logger.info("训练曲线分析结果")
    logger.info("="*60)
    logger.info(f"最终训练MSE: {final_train_mse:.6f}")
    logger.info(f"最终验证MSE: {final_val_mse:.6f}")
    logger.info(f"最佳验证MSE: {best_val_mse:.6f} (轮次: {best_val_epoch})")
    logger.info(f"训练MSE改善: {train_improvement:.2f}%")
    logger.info(f"验证MSE改善: {val_improvement:.2f}%")
    logger.info(f"验证集MSE从最佳增长: {analysis_dict['val_mse_increase_from_best']:.2f}%")
    logger.info(f"是否过拟合: {'是' if is_overfitting else '否'}")
    logger.info(f"是否收敛: {'是' if is_converged else '否'}")
    logger.info("="*60)
    
    return analysis_dict


def save_training_curves_and_analysis(train_losses, val_losses, analysis_dict,
                                     exp_type, dataset_name, model_name,
                                     n_estimators, plots_dir, logger):
    """
    保存训练曲线图和分析结果
    
    参数:
        train_losses, val_losses: 训练/验证MSE列表
        analysis_dict: 分析结果字典
        exp_type: 实验类型
        dataset_name: 数据集名称
        model_name: 模型名称
        n_estimators: 训练轮次
        plots_dir: 图表保存目录
        logger: 日志记录器
    """
    # 生成训练曲线图
    plot_path = plot_mse_loss(
        train_losses, val_losses, exp_type, dataset_name,
        42, model_name, n_estimators, plot_dir=plots_dir
    )
    logger.info(f"训练曲线图已保存: {plot_path}")
    
    return plot_path


# ============================================================================
# 模块A: 参数理解实验 (糖尿病数据集)
# ============================================================================

def experiment_module_A_diabetes(args, logger):
    """
    模块A: 使用糖尿病数据集进行参数理解实验
    参考: ABC_Project/metagb/compare_metagb_xgboost_grid.py
    """
    logger.info("="*80)
    logger.info("模块A: 参数理解实验 - 糖尿病数据集")
    logger.info("="*80)
    
    # 加载糖尿病数据集
    print("加载糖尿病数据集...")
    logger.info("加载糖尿病数据集...")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    print(f"数据集形状: {X.shape}")
    logger.info(f"数据集形状: {X.shape}")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    
    # 划分数据集
    print("划分数据集...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    # 标准化目标值y（用于得到可比较的MSE）
    y_scaler = StandardScaler()
    y_train_val_scaled = y_scaler.fit_transform(y_train_val.reshape(-1, 1)).flatten()
    y_train = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
    print(f"目标值标准化: 原始范围 [{y_train_val.min():.2f}, {y_train_val.max():.2f}], "
          f"标准化后范围 [{y_train.min():.2f}, {y_train.max():.2f}]")
    logger.info(f"数据划分完成 - 训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
    logger.info(f"目标值标准化 - y_mean: {y_scaler.mean_[0]:.4f}, y_scale: {y_scaler.scale_[0]:.4f}")
    
    # 参数网格定义
    xgb_param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'max_bin': [64],
        'min_child_weight': [20],
        'reg_lambda': [0.0],
    }
    
    metagb_param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'pred_feature_fraction': [0.0],  # bias only
        'version': [3],
        'max_bin': [64],
        'min_data_in_leaf': [20],
    }
    
    print(f"\nXGBoost参数组合: {len(list(product(*xgb_param_grid.values())))}")
    print(f"MetaGB参数组合: {len(list(product(*metagb_param_grid.values())))}")
    
    results = {
        'module': 'A',
        'dataset': 'diabetes',
        'xgboost': [],
        'metagb': []
    }
    
    # 记录最佳模型的训练曲线
    best_xgb_result = None
    best_xgb_analysis = None
    best_metagb_result = None
    best_metagb_analysis = None
    
    # XGBoost 网格搜索
    print("\n" + "-"*80)
    print("XGBoost Grid Search")
    print("-"*80)
    
    param_names = list(xgb_param_grid.keys())
    param_values = list(xgb_param_grid.values())
    
    for i, params in enumerate(product(*param_values)):
        param_dict = dict(zip(param_names, params))
        
        model = XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            **param_dict
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        result = {
            'params': param_dict,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_time': train_time
        }
        results['xgboost'].append(result)
        
        print(f"[{i+1}/{len(list(product(*param_values)))}] {param_dict}")
        print(f"    Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Test MSE: {test_mse:.4f} | Time: {train_time:.2f}s")
        logger.info(f"XGBoost [{i+1}/{len(list(product(*param_values)))}] params={param_dict}, "
                   f"train_mse={train_mse:.4f}, val_mse={val_mse:.4f}, test_mse={test_mse:.4f}")
    
    # MetaGB 网格搜索
    print("\n" + "-"*80)
    print("MetaGB Grid Search (Version 3)")
    print("-"*80)
    
    param_names = list(metagb_param_grid.keys())
    param_values = list(metagb_param_grid.values())
    
    for i, params in enumerate(product(*param_values)):
        param_dict = dict(zip(param_names, params))
        
        model = MetaGB(
            learning_rate=param_dict['learning_rate'],
            num_trees=param_dict['n_estimators'],
            feature_fraction=param_dict['colsample_bytree'],
            pred_feature_fraction=param_dict.get('pred_feature_fraction', param_dict['colsample_bytree']),
            bagging_fraction=param_dict['subsample'],
            min_data_in_leaf=param_dict.get('min_data_in_leaf', 20),
            max_depth=param_dict['max_depth'],
            max_bin=param_dict.get('max_bin', 64),
            version=param_dict.get('version', 3),
            seed=42,
        )
        
        start_time = time.time()
        model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
        train_time = time.time() - start_time
        
        train_pred = model.predict(X_train.astype(np.float64))
        val_pred = model.predict(X_val.astype(np.float64))
        test_pred = model.predict(X_test.astype(np.float64))
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        result = {
            'params': param_dict,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_time': train_time
        }
        results['metagb'].append(result)
        
        print(f"[{i+1}/{len(list(product(*param_values)))}] {param_dict}")
        print(f"    Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Test MSE: {test_mse:.4f} | Time: {train_time:.2f}s")
        logger.info(f"MetaGB [{i+1}/{len(list(product(*param_values)))}] params={param_dict}, "
                   f"train_mse={train_mse:.4f}, val_mse={val_mse:.4f}, test_mse={test_mse:.4f}")
    
    # 找到最佳模型并记录详细训练曲线
    print("\n" + "-"*80)
    print("分析最佳模型训练曲线")
    print("-"*80)
    
    # 找到XGBoost最佳模型
    best_xgb_idx = min(range(len(results['xgboost'])),
                      key=lambda i: results['xgboost'][i]['val_mse'])
    best_xgb = results['xgboost'][best_xgb_idx]
    
    # 训练最佳XGBoost模型的完整曲线
    print(f"\n最佳XGBoost模型（验证MSE={best_xgb['val_mse']:.4f}）:")
    logger.info(f"\n最佳XGBoost模型验证曲线分析，参数={best_xgb['params']}")
    best_xgb_model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        n_jobs=1,
        **best_xgb['params']
    )
    train_losses_xgb, val_losses_xgb, best_xgb_model = \
        train_xgboost_with_tracking(best_xgb_model, X_train, y_train,
                                  X_val, y_val, logger, eval_interval=5)
    
    # 分析XGBoost训练曲线
    best_xgb_analysis = analyze_training_curves(train_losses_xgb, val_losses_xgb, logger)
    
    # 保存XGBoost训练曲线图
    save_training_curves_and_analysis(
        train_losses_xgb, val_losses_xgb, best_xgb_analysis,
        'A', 'diabetes', 'XGBoost',
        best_xgb['params']['n_estimators'], args.plots_dir, logger
    )
    
    # 找到MetaGB最佳模型
    best_metagb_idx = min(range(len(results['metagb'])),
                         key=lambda i: results['metagb'][i]['val_mse'])
    best_metagb = results['metagb'][best_metagb_idx]
    
    # 训练最佳MetaGB模型的完整曲线
    print(f"\n最佳MetaGB模型（验证MSE={best_metagb['val_mse']:.4f}）:")
    logger.info(f"\n最佳MetaGB模型验证曲线分析，参数={best_metagb['params']}")
    best_metagb_model = MetaGB(
        learning_rate=best_metagb['params']['learning_rate'],
        num_trees=best_metagb['params']['n_estimators'],
        feature_fraction=best_metagb['params']['colsample_bytree'],
        pred_feature_fraction=best_metagb['params'].get('pred_feature_fraction',
                                                best_metagb['params']['colsample_bytree']),
        bagging_fraction=best_metagb['params']['subsample'],
        min_data_in_leaf=best_metagb['params'].get('min_data_in_leaf', 20),
        max_depth=best_metagb['params']['max_depth'],
        max_bin=best_metagb['params'].get('max_bin', 64),
        version=best_metagb['params'].get('version', 3),
        seed=42,
    )
    train_losses_metagb, val_losses_metagb, best_metagb_model = \
        train_metagb_with_tracking(best_metagb_model, X_train, y_train,
                                 X_val, y_val, logger, eval_interval=5)
    
    # 分析MetaGB训练曲线
    best_metagb_analysis = analyze_training_curves(train_losses_metagb, val_losses_metagb, logger)
    
    # 保存MetaGB训练曲线图
    save_training_curves_and_analysis(
        train_losses_metagb, val_losses_metagb, best_metagb_analysis,
        'A', 'diabetes', 'MetaGB_v3',
        best_metagb['params']['n_estimators'], args.plots_dir, logger
    )
    
    # 保存训练分析结果到results
    results['best_xgb_analysis'] = best_xgb_analysis
    results['best_xgb_params'] = best_xgb['params']
    results['best_metagb_analysis'] = best_metagb_analysis
    results['best_metagb_params'] = best_metagb['params']
    
    # 保存结果
    results_path = os.path.join(args.results_dir, 'module_A_diabetes', 'results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print(f"结果已保存到: {results_path}")
    print("="*80)
    logger.info(f"结果保存到: {results_path}")
    
    return results


# ============================================================================
# 模块B: 合成数据集性能测试
# ============================================================================

def generate_synthetic_regression_square(n_samples=4000, n_features=10, noise=0.0, random_state=42):
    """生成平方项非线性回归数据"""
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(n_samples, n_features))
    
    # 平方项非线性
    y = np.sum(x[:, :3]**2, axis=1) + np.sum(x[:, 3:6], axis=1)
    y += rng.normal(scale=noise, size=n_samples)
    
    return x.astype(np.float64), y.astype(np.float64)


def generate_synthetic_regression_sin(n_samples=4000, n_features=10, noise=0.0, random_state=42):
    """生成正弦波非线性回归数据"""
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(n_samples, n_features))
    
    # 正弦波非线性
    y = 2.0 * np.sin(x[:, 0] * x[:, 1]) + 0.5 * np.sin(x[:, 2] * x[:, 3])
    y += rng.normal(scale=noise, size=n_samples)
    
    return x.astype(np.float64), y.astype(np.float64)


def generate_synthetic_regression_square_sin(n_samples=4000, n_features=10, noise=0.0, random_state=42):
    """生成平方+正弦混合非线性回归数据"""
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(n_samples, n_features))
    
    # 混合非线性
    y = x[:, 0]**2 + x[:, 1]**2 + 2.0 * np.sin(x[:, 2] * x[:, 3])
    y += rng.normal(scale=noise, size=n_samples)
    
    return x.astype(np.float64), y.astype(np.float64)


def generate_synthetic_regression_sigmoid(n_samples=4000, n_features=10, noise=0.0, random_state=42):
    """生成Sigmoid非线性回归数据"""
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(n_samples, n_features))
    
    # Sigmoid非线性
    y = 1.0 / (1.0 + np.exp(-x[:, 0] - x[:, 1])) + 0.5 * x[:, 2]
    y += rng.normal(scale=noise, size=n_samples)
    
    return x.astype(np.float64), y.astype(np.float64)


def experiment_module_B_synthetic(args, logger):
    """
    模块B: 合成数据集性能测试
    参考: ABC_Project/metagb/compare_metagb_xgboost.py main_v3
    """
    logger.info("="*80)
    logger.info("模块B: 合成数据集性能测试")
    logger.info("="*80)
    
    # 记录分析结果的变量
    first_analysis_done = False
    
    # 数据生成器列表
    data_generators = {
        'square': generate_synthetic_regression_square,
        'sin': generate_synthetic_regression_sin,
        'square_sin': generate_synthetic_regression_square_sin,
        'sigmoid': generate_synthetic_regression_sigmoid
    }
    
    # 噪声水平
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.6, 1.0]
    
    # 统一参数（参考compare_metagb_xgboost.py）
    feature_fraction = 0.8
    pred_fractions = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    results = {
        'module': 'B',
        'dataset_type': 'synthetic',
        'n_samples': 4000,
        'n_features': 10,
        'results': []
    }
    
    total_tests = len(data_generators) * len(noise_levels)
    test_count = 0
    
    # 对每种数据类型和噪声水平进行测试
    for data_type, generator in data_generators.items():
        for noise_level in noise_levels:
            test_count += 1
            print(f"\n[{test_count}/{total_tests}] 测试: {data_type} 数据, 噪声水平: {noise_level}")
            print("-" * 60)
            logger.info(f"测试 [{test_count}/{total_tests}]: {data_type}, noise={noise_level}")
            
            # 生成数据
            x, y = generator(n_samples=4000, n_features=10, noise=noise_level, random_state=42)
            
            # 划分数据集（训练集再划分出验证集用于曲线记录）
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42
            )
            
            test_result = {
                'data_type': data_type,
                'noise_level': noise_level,
                'metagb': {},
                'xgboost': {}
            }
            
            # MetaGB 测试 - 参考compare_metagb_xgboost.py main_v3
            for pred_frac in pred_fractions:
                model = MetaGB(
                    learning_rate=0.1,
                    num_trees=150,
                    feature_fraction=feature_fraction,
                    pred_feature_fraction=pred_frac,
                    bagging_fraction=0.8,
                    min_data_in_leaf=20,
                    max_depth=5,
                    max_bin=64,
                    version=3,
                    seed=42,
                )
                
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                test_result['metagb'][f'pred_{pred_frac}'] = {
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'train_time': float(train_time)
                }
                
                print(f"    MetaGB pred_fraction={pred_frac:.1f}: "
                      f"Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")
            
            # XGBoost 测试
            model = XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                random_state=42,
                n_jobs=1,
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=20,
                reg_lambda=0.0,
            )
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            test_result['xgboost'] = {
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_time': float(train_time)
            }
            
            print(f"    XGBoost: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")
            
            results['results'].append(test_result)
            logger.info(f"完成: {data_type}, noise={noise_level}")
            
            # 对第一个数据集（square）和无噪声情况（noise=0.0）记录详细训练曲线
            if not first_analysis_done and data_type == 'square' and noise_level == 0.0:
                print("\n" + "-"*80)
                print("记录代表性训练曲线 (Square数据, 无噪声)")
                print("-"*80)
                logger.info(f"\n代表性训练曲线分析 - {data_type}, noise={noise_level}")
                
                # 记录XGBoost训练曲线
                xgb_model = XGBRegressor(
                    objective='reg:squarederror',
                    tree_method='hist',
                    random_state=42,
                    n_jobs=1,
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=20,
                    reg_lambda=0.0,
                )
                train_losses_xgb, val_losses_xgb, _ = \
                    train_xgboost_with_tracking(xgb_model, X_train, y_train,
                                          X_val, y_val, logger, eval_interval=10)
                xgb_analysis = analyze_training_curves(train_losses_xgb, val_losses_xgb, logger)
                save_training_curves_and_analysis(
                    train_losses_xgb, val_losses_xgb, xgb_analysis,
                    'B', f'{data_type}_noise{noise_level}', 'XGBoost',
                    150, args.plots_dir, logger
                )
                
                # 记录最佳MetaGB (pred=0.0) 训练曲线
                metagb_model = MetaGB(
                    learning_rate=0.1,
                    num_trees=150,
                    feature_fraction=feature_fraction,
                    pred_feature_fraction=0.0,
                    bagging_fraction=0.8,
                    min_data_in_leaf=20,
                    max_depth=5,
                    max_bin=64,
                    version=3,
                    seed=42,
                )
                train_losses_metagb, val_losses_metagb, _ = \
                    train_metagb_with_tracking(metagb_model, X_train, y_train,
                                           X_val, y_val, logger, eval_interval=10)
                metagb_analysis = analyze_training_curves(train_losses_metagb, val_losses_metagb, logger)
                save_training_curves_and_analysis(
                    train_losses_metagb, val_losses_metagb, metagb_analysis,
                    'B', f'{data_type}_noise{noise_level}', 'MetaGB_pred0.0',
                    150, args.plots_dir, logger
                )
                
                results['rep_xgb_analysis'] = xgb_analysis
                results['rep_metagb_analysis'] = metagb_analysis
                first_analysis_done = True
    
    # 保存结果
    results_path = os.path.join(args.results_dir, 'module_B_synthetic', 'results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print(f"结果已保存到: {results_path}")
    print("="*80)
    logger.info(f"结果保存到: {results_path}")
    
    return results


# ============================================================================
# 模块C: TabArena真实数据集批量测试
# ============================================================================

def load_tabular_dataset(dataset_path):
    """
    加载TabArena回归数据集
    """
    df = pd.read_csv(dataset_path)
    
    # 假设最后一列是目标变量
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 处理缺失值
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    # 标准化目标变量
    y_mean = y.mean()
    y_std = y.std()
    if y_std > 0:
        y = (y - y_mean) / y_std
    
    return X.astype(np.float64), y.astype(np.float64), y_mean, y_std


def experiment_module_C_tabarena(args, logger):
    """
    模块C: TabArena真实数据集批量测试
    参考: ABC_Project/metagb/compare_metagb_xgboost_grid.py
    """
    logger.info("="*80)
    logger.info("模块C: TabArena真实数据集批量测试")
    logger.info("="*80)
    
    # 记录第一个数据集的详细分析
    first_dataset_analyzed = False
    
    # TabArena回归数据集列表
    dataset_dir = 'dataset/tabArena_regression'
    datasets = [
        'airfoil_self_noise.csv',
        'concrete_compressive_strength.csv',
        'diamonds.csv',
        'Food_Delivery_Time.csv',
        'healthcare_insurance_expenses.csv',
        'houses.csv',
        'miami_housing.csv',
        'physiochemical_protein.csv',
        'QSAR_fish_toxicity.csv',
        'QSAR-TID-11.csv',
        'superconductivity.csv',
        'wine_quality.csv'
    ]
    
    # 参数网格
    xgb_param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    metagb_param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'pred_feature_fraction': [0.0, 0.4, 0.8],
        'version': [3],
        'max_bin': [64],
        'min_data_in_leaf': [20],
    }
    
    n_folds = 3  # 使用较少折数加快实验速度
    
    results = {
        'module': 'C',
        'dataset_type': 'tabarena',
        'n_folds': n_folds,
        'datasets': {}
    }
    
    dataset_count = 0
    for dataset_filename in datasets:
        dataset_count += 1
        dataset_name = dataset_filename.replace('.csv', '')
        dataset_path = os.path.join(dataset_dir, dataset_filename)
        
        if not os.path.exists(dataset_path):
            print(f"\n[{dataset_count}/{len(datasets)}] 跳过: {dataset_name} (文件不存在)")
            continue
        
        print(f"\n[{dataset_count}/{len(datasets)}] 处理数据集: {dataset_name}")
        print("-" * 60)
        logger.info(f"数据集: {dataset_name}")
        
        # 加载数据
        try:
            X, y, y_mean, y_std = load_tabular_dataset(dataset_path)
            print(f"  数据形状: {X.shape}")
        except Exception as e:
            print(f"  错误: 加载数据失败 - {e}")
            logger.error(f"加载数据失败: {dataset_name} - {e}")
            continue
        
        # 划分数据集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )
        
        print(f"  训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
        
        dataset_result = {
            'shape': list(X.shape),
            'xgboost': [],
            'metagb': []
        }
        
        # 简化测试：仅使用一个参数组合
        fixed_xgb_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 20,
            'reg_lambda': 0.0,
        }
        
        metagb_fixed_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'num_trees': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 20,
            'max_bin': 64,
            'version': 3,
            'seed': 42,
        }
        
        # XGBoost 测试
        print("  测试 XGBoost...")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        train_mses, val_mses, test_mses = [], [], []
        train_times = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            xgb_model = XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                random_state=42 + fold_idx,
                n_jobs=1,
                **fixed_xgb_params
            )
            
            start_time = time.time()
            xgb_model.fit(X_train_fold, y_train_fold)
            train_time = time.time() - start_time
            train_times.append(train_time)
            
            train_pred = xgb_model.predict(X_train_fold)
            val_pred = xgb_model.predict(X_val_fold)
            test_pred = xgb_model.predict(X_test)
            
            train_mses.append(mean_squared_error(y_train_fold, train_pred))
            val_mses.append(mean_squared_error(y_val_fold, val_pred))
            test_mses.append(mean_squared_error(y_test, test_pred))
        
        dataset_result['xgboost'].append({
            'params': fixed_xgb_params,
            'train_mse_mean': float(np.mean(train_mses)),
            'val_mse_mean': float(np.mean(val_mses)),
            'test_mse_mean': float(np.mean(test_mses)),
            'train_time_mean': float(np.mean(train_times))
        })
        
        print(f"    Train MSE: {np.mean(train_mses):.4f}, Val MSE: {np.mean(val_mses):.4f}, "
              f"Test MSE: {np.mean(test_mses):.4f}")
        
        # MetaGB 测试（测试不同的 pred_feature_fraction）
        print("  测试 MetaGB...")
        pred_fractions = [0.0, 0.4, 0.8]
        
        for pred_frac in pred_fractions:
            train_mses, val_mses, test_mses = [], [], []
            train_times = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                model = MetaGB(
                    pred_feature_fraction=pred_frac,
                    **metagb_fixed_params
                )
                
                start_time = time.time()
                model.fit(X_train_fold.astype(np.float64), y_train_fold.astype(np.float64))
                train_time = time.time() - start_time
                train_times.append(train_time)
                
                train_pred = model.predict(X_train_fold.astype(np.float64))
                val_pred = model.predict(X_val_fold.astype(np.float64))
                test_pred = model.predict(X_test.astype(np.float64))
                
                train_mses.append(mean_squared_error(y_train_fold, train_pred))
                val_mses.append(mean_squared_error(y_val_fold, val_pred))
                test_mses.append(mean_squared_error(y_test, test_pred))
            
            dataset_result['metagb'].append({
                'pred_feature_fraction': pred_frac,
                'train_mse_mean': float(np.mean(train_mses)),
                'val_mse_mean': float(np.mean(val_mses)),
                'test_mse_mean': float(np.mean(test_mses)),
                'train_time_mean': float(np.mean(train_times))
            })
            
            print(f"    pred_fraction={pred_frac:.1f}: "
                  f"Train MSE={np.mean(train_mses):.4f}, Val MSE={np.mean(val_mses):.4f}, "
                  f"Test MSE={np.mean(test_mses):.4f}")
        
        results['datasets'][dataset_name] = dataset_result
        logger.info(f"完成数据集: {dataset_name}")
        
        # 对第一个成功处理的数据集记录详细训练曲线
        if not first_dataset_analyzed:
            print("\n" + "-"*80)
            print(f"记录代表性训练曲线 - 数据集: {dataset_name}")
            print("-"*80)
            logger.info(f"\n代表性训练曲线分析 - {dataset_name}")
            
            # 获取第一个训练-验证折叠（用于训练曲线）
            train_idx, val_idx = list(kf.split(X))[0]
            X_train_curve, X_val_curve = X[train_idx], X[val_idx]
            y_train_curve, y_val_curve = y[train_idx], y[val_idx]
            
            # 记录XGBoost训练曲线
            xgb_model = XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                random_state=42,
                n_jobs=1,
                **fixed_xgb_params
            )
            train_losses_xgb, val_losses_xgb, _ = \
                train_xgboost_with_tracking(xgb_model, X_train_curve, y_train_curve,
                                      X_val_curve, y_val_curve, logger, eval_interval=10)
            xgb_analysis = analyze_training_curves(train_losses_xgb, val_losses_xgb, logger)
            save_training_curves_and_analysis(
                train_losses_xgb, val_losses_xgb, xgb_analysis,
                'C', dataset_name, 'XGBoost',
                fixed_xgb_params['n_estimators'], args.plots_dir, logger
            )
            
            # 记录MetaGB训练曲线（pred=0.0）
            metagb_model = MetaGB(
                pred_feature_fraction=0.0,
                **metagb_fixed_params
            )
            train_losses_metagb, val_losses_metagb, _ = \
                train_metagb_with_tracking(metagb_model, X_train_curve, y_train_curve,
                                       X_val_curve, y_val_curve, logger, eval_interval=10)
            metagb_analysis = analyze_training_curves(train_losses_metagb, val_losses_metagb, logger)
            save_training_curves_and_analysis(
                train_losses_metagb, val_losses_metagb, metagb_analysis,
                'C', dataset_name, 'MetaGB_pred0.0',
                metagb_fixed_params['num_trees'], args.plots_dir, logger
            )
            
            results['rep_xgb_analysis'] = xgb_analysis
            results['rep_metagb_analysis'] = metagb_analysis
            results['rep_dataset_name'] = dataset_name
            first_dataset_analyzed = True
    
    # 保存结果
    results_path = os.path.join(args.results_dir, 'module_C_tabarena', 'results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print(f"结果已保存到: {results_path}")
    print("="*80)
    logger.info(f"结果保存到: {results_path}")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print_banner()
    
    # 解析参数
    args = parse_arguments()
    
    # 创建目录
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # 初始化日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.logs_dir, f'experiment_{timestamp}.log')
    logger = setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("实验开始")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"参数: {vars(args)}")
    logger.info("="*80)
    
    all_results = {}
    
    # 运行模块A: 参数理解实验
    if args.module in ['A', 'all']:
        try:
            results_A = experiment_module_A_diabetes(args, logger)
            all_results['module_A'] = results_A
            logger.info("\n模块A完成!")
        except Exception as e:
            logger.error(f"\n模块A错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 运行模块B: 合成数据集测试
    if args.module in ['B', 'all']:
        try:
            results_B = experiment_module_B_synthetic(args, logger)
            all_results['module_B'] = results_B
            logger.info("\n模块B完成!")
        except Exception as e:
            logger.error(f"\n模块B错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 运行模块C: TabArena数据集测试
    if args.module in ['C', 'all']:
        try:
            results_C = experiment_module_C_tabarena(args, logger)
            all_results['module_C'] = results_C
            logger.info("\n模块C完成!")
        except Exception as e:
            logger.error(f"\n模块C错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 保存所有结果
    final_results_path = os.path.join(args.results_dir, f'all_results_{timestamp}.json')
    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info("\n" + "="*80)
    logger.info("实验完成!")
    logger.info(f"总结果文件: {final_results_path}")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    print(f"\n实验完成! 结果保存在: {args.results_dir}/")
    print(f"日志文件: {log_file}")


if __name__ == "__main__":
    main()
