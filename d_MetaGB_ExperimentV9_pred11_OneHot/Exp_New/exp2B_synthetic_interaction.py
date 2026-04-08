"""
Experiment 2B: Feature Interaction Pattern Test
MetaGB vs XGBoost on Synthetic Data with Various Interaction Patterns

This experiment tests model performance on different feature interaction patterns:
1. Piecewise Linear: Feature space divided into regions with different linear relationships
2. Conditional Prediction: Categorical feature selects prediction function
3. Sparse Interaction: High-dimensional data with few relevant interacting features

Key features:
- sparse_config as outer loop (from hyperparams.py)
- Hyperparameter grid search for both XGBoost and MetaGB (inner loop)
- Multiple repetitions (REPEATING_NUMS times) with fresh datasets each time
- Training curve visualization (Train RMSE vs Validation RMSE)
- Z-Score normalization applied to features

Models tested:
- XGBoost (with grid search)
- MetaGB (with grid search, varying pred_feature_fraction)
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from itertools import product

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metagb'))

from utils.hyperparams import (
    RANDOM_SEED, TRAIN_TEST_SPLIT, REPEATING_NUMS,
    get_exp2b_xgb_param_grid, get_exp2b_metagb_param_grid,
    get_sparse_configs, EARLY_STOPPING_ROUNDS, EARLY_STOPPING_MIN_DELTA
)
from utils.data_generators import (
    make_piecewise_linear,
    make_conditional_prediction,
    make_sparse_interaction,
    generate_exp2b_datasets
)
from utils.trainer import (
    train_single_model, evaluate_model, apply_zscore_normalization,
    get_feature_importance, calculate_feature_selection_precision,
    TargetScaler, calculate_metrics
)
from utils.logger import ExperimentLogger
from utils.visualizer import (
    plot_training_curve,
    plot_tree_depth_comparison,
    plot_model_comparison_bar,
    plot_feature_importance_comparison
)


def save_experiment_log(
    log_path: str,
    exp_name: str,
    dataset_name: str,
    dataset_type: str,
    seed: int,
    model_name: str,
    model_type: str,
    params: dict,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
    scaler_params: dict,
    data_metadata: dict,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    train_time: float,
    train_losses: list,
    val_losses: list,
    feature_precision: float = None
):
    """
    Save detailed experiment log for a single run.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Experiment 2B: {model_name} - {dataset_name}\n")
        f.write("=" * 80 + "\n\n")

        # Basic info
        f.write(f"[{timestamp}] Experiment Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Dataset Type: {dataset_type}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Train Samples: {n_train}\n")
        f.write(f"Validation Samples: {n_val}\n")
        f.write(f"Test Samples: {n_test}\n")
        f.write(f"Features: {n_features}\n\n")

        # Data generation info
        f.write(f"[{timestamp}] Data Generation\n")
        f.write("-" * 40 + "\n")
        f.write(f"Formula: {data_metadata.get('formula', 'N/A')}\n")
        f.write(f"Type: {data_metadata.get('type', 'N/A')}\n")
        if 'relevant_features' in data_metadata:
            f.write(f"Relevant Features: {data_metadata['relevant_features']}\n")
        if 'n_regions' in data_metadata:
            f.write(f"Number of Regions: {data_metadata['n_regions']}\n")
        if 'n_categories' in data_metadata:
            f.write(f"Number of Categories: {data_metadata['n_categories']}\n")
        f.write("\n")

        # Z-Score normalization info
        f.write(f"[{timestamp}] Z-Score Normalization\n")
        f.write("-" * 40 + "\n")
        f.write("Feature scaling applied: Z-Score (StandardScaler)\n")
        if scaler_params:
            f.write(f"Feature means (first 10): {scaler_params.get('mean', [])[:10]}\n")
            f.write(f"Feature stds (first 10): {scaler_params.get('std', [])[:10]}\n")
        f.write("\n")

        # Model parameters
        f.write(f"[{timestamp}] Model Parameters\n")
        f.write("-" * 40 + "\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        # Training metrics
        f.write(f"[{timestamp}] Training Metrics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train RMSE: {train_metrics['rmse']:.6f}\n")
        f.write(f"Train MAE: {train_metrics['mae']:.6f}\n")
        f.write(f"Train R²: {train_metrics['r2']:.6f}\n\n")

        # Validation metrics
        f.write(f"[{timestamp}] Validation Metrics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Val RMSE: {val_metrics['rmse']:.6f}\n")
        f.write(f"Val MAE: {val_metrics['mae']:.6f}\n")
        f.write(f"Val R²: {val_metrics['r2']:.6f}\n\n")

        # Test metrics
        f.write(f"[{timestamp}] Test Metrics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.6f}\n")
        f.write(f"Test R²: {test_metrics['r2']:.6f}\n")
        f.write(f"Test MSE: {test_metrics['mse']:.6f}\n\n")

        # Feature selection precision (for sparse datasets)
        if feature_precision is not None:
            f.write(f"[{timestamp}] Feature Selection\n")
            f.write("-" * 40 + "\n")
            f.write(f"Feature Selection Precision (Top-10): {feature_precision:.4f}\n\n")

        # Training time
        f.write(f"[{timestamp}] Training Time\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Time: {train_time:.2f} seconds\n\n")

        # Training curve summary
        if train_losses and val_losses:
            f.write(f"[{timestamp}] Training Curve Summary\n")
            f.write("-" * 40 + "\n")
            f.write(f"Initial Train MSE: {train_losses[0]:.6f}\n")
            f.write(f"Final Train MSE: {train_losses[-1]:.6f}\n")
            f.write(f"Initial Val MSE: {val_losses[0]:.6f}\n")
            f.write(f"Final Val MSE: {val_losses[-1]:.6f}\n")
            f.write(f"Best Val MSE: {min(val_losses):.6f} (at step {val_losses.index(min(val_losses)) + 1})\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Experiment completed successfully\n")
        f.write("=" * 80 + "\n")

    # Also save as JSON
    json_path = log_path.replace('.log', '.json')
    json_data = {
        'timestamp': timestamp,
        'experiment': 'exp2B',
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'model_name': model_name,
        'model_type': model_type,
        'seed': seed,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'n_features': n_features,
        'scaler_params': scaler_params,
        'data_metadata': {k: str(v) if isinstance(v, list) else v for k, v in data_metadata.items()},
        'params': {k: str(v) for k, v in params.items()},
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_precision': feature_precision,
        'train_time': train_time,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def run_single_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    model_type: str,
    params: dict,
    record_loss_curve: bool = True,
    eval_interval: int = 10
) -> dict:
    """
    Run a single model training and evaluation.
    All y values should be in normalized scale.
    """
    # Train model
    train_result = train_single_model(
        model_type=model_type,
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        record_loss_curve=record_loss_curve,
        eval_interval=eval_interval
    )

    # Evaluate on test set (in normalized scale)
    if model_type == 'xgboost':
        test_pred = train_result['model'].predict(X_test)
    else:
        test_pred = train_result['model'].predict(X_test.astype(np.float64))

    test_metrics = calculate_metrics(y_test, test_pred)

    # Get feature importance
    feature_importance = get_feature_importance(
        train_result['model'],
        model_type,
        X_train.shape[1]
    )

    return {
        'model': train_result['model'],
        'model_name': model_name,
        'model_type': model_type,
        'params': params,
        'train_time': train_result['train_time'],
        'train_losses': train_result['train_losses'],
        'val_losses': train_result['val_losses'],
        'train_metrics': train_result['train_metrics'],
        'val_metrics': train_result['val_metrics'],
        'test_metrics': test_metrics,
        'feature_importance': feature_importance,
        'early_stopped': train_result.get('early_stopped', False),
        'early_stopping_round': train_result.get('early_stopping_round', None),
        'best_n_trees': train_result.get('best_n_trees', params.get('n_estimators', 100))
    }


def run_grid_search_for_model(
    model_type: str,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    logger: ExperimentLogger
) -> dict:
    """
    Run grid search for a single model type.

    Returns:
        Dictionary with best model results
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = int(np.prod([len(v) for v in param_values]))

    logger.info(f"    Grid search: {total_combinations} combinations")

    best_val_rmse = float('inf')
    best_result = None
    best_params = None

    for i, params in enumerate(product(*param_values)):
        param_dict = dict(zip(param_names, params))
        param_dict['random_state'] = random_state

        # Train and evaluate
        result = run_single_experiment(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            model_name=model_type,
            model_type=model_type,
            params=param_dict,
            record_loss_curve=True,
            eval_interval=10
        )

        val_rmse = result['val_metrics']['rmse']

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_result = result
            best_params = param_dict.copy()

        if (i + 1) % 20 == 0:
            logger.info(f"      [{i+1}/{total_combinations}] Best Val RMSE so far: {best_val_rmse:.4f}")

    best_result['best_params'] = best_params
    return best_result


def run_experiment_for_sparse_config(
    sparse_config: tuple,
    base_seed: int,
    repeat_idx: int,
    logger: ExperimentLogger,
    logs_dir: str,
    plots_dir: str,
    data_dir: str
) -> list:
    """
    Run experiments for a single sparse_config across grid search.
    Each repetition generates a fresh dataset.

    Args:
        sparse_config: (n_total_features, n_relevant, interaction_order)
        base_seed: Base random seed
        repeat_idx: Repetition index (0 to REPEATING_NUMS-1)
        logger: Logger instance
        logs_dir: Directory for logs
        plots_dir: Directory for plots
        data_dir: Directory for saved datasets

    Returns:
        List of result dictionaries
    """
    n_total, n_rel, order = sparse_config
    dataset_name = f"sparse_{n_total}f_{n_rel}r_{order}order"

    # Use different seed for each repetition to generate different datasets
    seed = base_seed + repeat_idx * 1000

    logger.info(f"\n{'='*80}")
    logger.info(f"Sparse Config: {dataset_name} | Repetition: {repeat_idx + 1}/{REPEATING_NUMS}")
    logger.info(f"Dataset seed: {seed}")
    logger.info(f"{'='*80}")

    # Generate fresh data for this repetition
    X, y, metadata = make_sparse_interaction(
        n_samples=4000,
        n_total_features=n_total,
        n_relevant=n_rel,
        interaction_order=order,
        noise_std=0.1,
        random_state=seed
    )

    # Save generated dataset
    dataset_filename = f"{dataset_name}_rep{repeat_idx}_seed{seed}.npz"
    dataset_path = os.path.join(data_dir, dataset_filename)
    np.savez(dataset_path, X=X, y=y, metadata=str(metadata))
    logger.info(f"Dataset saved: {dataset_filename}")

    # Split data: 80% train (64% train + 16% val) / 20% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed
    )

    # Apply Z-Score normalization
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = apply_zscore_normalization(
        X_train, X_val=X_val, X_test=X_test
    )
    scaler_params = scaler.get_params()

    # Apply target (y) normalization
    y_scaler = TargetScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    logger.info(f"Samples - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Features: {X.shape[1]}, Relevant: {n_rel}")
    logger.info("Z-Score normalization applied (X and y)")

    # Get parameter grids
    xgb_param_grid = get_exp2b_xgb_param_grid()
    metagb_param_grid = get_exp2b_metagb_param_grid()

    all_results = []
    importance_by_model = {}

    # =========================================================================
    # XGBoost Grid Search
    # =========================================================================
    logger.info(f"Training XGBoost (grid search)...")
    try:
        xgb_result = run_grid_search_for_model(
            model_type='xgboost',
            param_grid=xgb_param_grid,
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_val=X_val_scaled,
            y_val=y_val_scaled,
            X_test=X_test_scaled,
            y_test=y_test_scaled,
            random_state=seed,
            logger=logger
        )

        # Calculate feature selection precision
        feature_precision = None
        if 'relevant_features' in metadata:
            feature_precision = calculate_feature_selection_precision(
                xgb_result['feature_importance'],
                metadata['relevant_features'],
                top_k=10
            )

        result_entry = {
            'Dataset_Type': 'sparse_interaction',
            'Variant': dataset_name,
            'Repetition': repeat_idx + 1,
            'Model': 'XGBoost',
            'Best_LR': xgb_result['best_params'].get('learning_rate', '-'),
            'Best_Depth': xgb_result['best_params'].get('max_depth', '-'),
            'Best_N_Est': xgb_result['best_params'].get('n_estimators', '-'),
            'Test_RMSE': xgb_result['test_metrics']['rmse'],
            'Test_MAE': xgb_result['test_metrics']['mae'],
            'Test_R2': xgb_result['test_metrics']['r2'],
            'Val_RMSE': xgb_result['val_metrics']['rmse'],
            'Avg_Tree_Depth': xgb_result['best_params'].get('max_depth', 5),
            'Train_Time_s': xgb_result['train_time'],
            'Feature_Selection_Precision': feature_precision if feature_precision else '-'
        }
        all_results.append(result_entry)

        importance_by_model['XGBoost'] = xgb_result['feature_importance']

        logger.info(f"  XGBoost - Test RMSE: {xgb_result['test_metrics']['rmse']:.4f}")
        if feature_precision is not None:
            logger.info(f"  Feature Selection Precision (Top-10): {feature_precision:.2f}")
        logger.info(f"  Time: {xgb_result['train_time']:.2f}s")

        # Plot training curve
        if xgb_result['train_losses']:
            plot_training_curve(
                train_losses=xgb_result['train_losses'],
                val_losses=xgb_result['val_losses'],
                exp_type='exp2B',
                dataset_name=f"{dataset_name}_rep{repeat_idx}",
                seed=seed,
                model_name='XGBoost',
                n_iter=xgb_result['best_params'].get('n_estimators', 100),
                plot_dir=plots_dir,
                test_rmse=xgb_result['test_metrics']['rmse'],
                use_rmse=True
            )

        # Save log
        log_filename = f"exp2B-{dataset_name}-rep{repeat_idx}-XGBoost.log"
        log_path = os.path.join(logs_dir, log_filename)
        save_experiment_log(
            log_path=log_path,
            exp_name='exp2B',
            dataset_name=dataset_name,
            dataset_type='sparse_interaction',
            seed=seed,
            model_name='XGBoost',
            model_type='xgboost',
            params=xgb_result['best_params'],
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
            n_features=X.shape[1],
            scaler_params=scaler_params,
            data_metadata=metadata,
            train_metrics=xgb_result['train_metrics'],
            val_metrics=xgb_result['val_metrics'],
            test_metrics=xgb_result['test_metrics'],
            train_time=xgb_result['train_time'],
            train_losses=xgb_result['train_losses'],
            val_losses=xgb_result['val_losses'],
            feature_precision=feature_precision
        )

    except Exception as e:
        logger.error(f"  XGBoost failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # =========================================================================
    # MetaGB Grid Search
    # =========================================================================
    logger.info(f"Training MetaGB (grid search)...")
    try:
        metagb_result = run_grid_search_for_model(
            model_type='metagb',
            param_grid=metagb_param_grid,
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_val=X_val_scaled,
            y_val=y_val_scaled,
            X_test=X_test_scaled,
            y_test=y_test_scaled,
            random_state=seed,
            logger=logger
        )

        pff = metagb_result['best_params'].get('pred_feature_fraction', 0.0)
        model_name = f'MetaGB_v{pff}'

        # Calculate feature selection precision
        feature_precision = None
        if 'relevant_features' in metadata:
            feature_precision = calculate_feature_selection_precision(
                metagb_result['feature_importance'],
                metadata['relevant_features'],
                top_k=10
            )

        result_entry = {
            'Dataset_Type': 'sparse_interaction',
            'Variant': dataset_name,
            'Repetition': repeat_idx + 1,
            'Model': model_name,
            'Best_LR': metagb_result['best_params'].get('learning_rate', '-'),
            'Best_Depth': metagb_result['best_params'].get('max_depth', '-'),
            'Best_N_Est': metagb_result['best_params'].get('n_estimators', '-'),
            'Best_Pred_Frac': pff,
            'Test_RMSE': metagb_result['test_metrics']['rmse'],
            'Test_MAE': metagb_result['test_metrics']['mae'],
            'Test_R2': metagb_result['test_metrics']['r2'],
            'Val_RMSE': metagb_result['val_metrics']['rmse'],
            'Avg_Tree_Depth': metagb_result['best_params'].get('max_depth', 5),
            'Train_Time_s': metagb_result['train_time'],
            'Feature_Selection_Precision': feature_precision if feature_precision else '-'
        }
        all_results.append(result_entry)

        importance_by_model[model_name] = metagb_result['feature_importance']

        logger.info(f"  {model_name} - Test RMSE: {metagb_result['test_metrics']['rmse']:.4f}")
        if feature_precision is not None:
            logger.info(f"  Feature Selection Precision (Top-10): {feature_precision:.2f}")
        logger.info(f"  Time: {metagb_result['train_time']:.2f}s")

        # Plot training curve
        if metagb_result['train_losses']:
            plot_training_curve(
                train_losses=metagb_result['train_losses'],
                val_losses=metagb_result['val_losses'],
                exp_type='exp2B',
                dataset_name=f"{dataset_name}_rep{repeat_idx}",
                seed=seed,
                model_name=model_name,
                n_iter=metagb_result['best_params'].get('n_estimators', 100),
                plot_dir=plots_dir,
                test_rmse=metagb_result['test_metrics']['rmse'],
                use_rmse=True
            )

        # Save log
        log_filename = f"exp2B-{dataset_name}-rep{repeat_idx}-MetaGB.log"
        log_path = os.path.join(logs_dir, log_filename)
        save_experiment_log(
            log_path=log_path,
            exp_name='exp2B',
            dataset_name=dataset_name,
            dataset_type='sparse_interaction',
            seed=seed,
            model_name=model_name,
            model_type='metagb',
            params=metagb_result['best_params'],
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
            n_features=X.shape[1],
            scaler_params=scaler_params,
            data_metadata=metadata,
            train_metrics=metagb_result['train_metrics'],
            val_metrics=metagb_result['val_metrics'],
            test_metrics=metagb_result['test_metrics'],
            train_time=metagb_result['train_time'],
            train_losses=metagb_result['train_losses'],
            val_losses=metagb_result['val_losses'],
            feature_precision=feature_precision
        )

    except Exception as e:
        logger.error(f"  MetaGB failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Plot feature importance comparison
    if 'relevant_features' in metadata and importance_by_model:
        try:
            plot_feature_importance_comparison(
                importance_by_model=importance_by_model,
                relevant_features=metadata['relevant_features'],
                dataset_name=f"{dataset_name}_rep{repeat_idx}",
                top_k=min(10, X.shape[1]),
                plot_dir=plots_dir
            )
        except Exception as e:
            logger.warning(f"Failed to plot feature importance: {e}")

    # Plot model comparison bar chart
    if all_results:
        try:
            plot_model_comparison_bar(
                results=[{'model': r['Model'], 'test_rmse': r['Test_RMSE']} for r in all_results],
                metric='test_rmse',
                dataset_name=f"{dataset_name}_rep{repeat_idx}",
                plot_dir=plots_dir
            )
        except Exception as e:
            logger.warning(f"Failed to plot model comparison: {e}")

    return all_results


def run_experiment2b(seed: int = RANDOM_SEED):
    """
    Run Experiment 2B: Feature interaction pattern test.
    Uses sparse_config as outer loop, runs REPEATING_NUMS times with fresh datasets.
    """
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results', 'exp2B')
    logs_dir = os.path.join(base_dir, 'logs', 'exp2B')
    plots_dir = os.path.join(base_dir, 'plots', 'exp2B')
    data_dir = os.path.join(base_dir, 'data', 'exp2B')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Setup logger
    logger = ExperimentLogger('exp2B', log_dir=logs_dir)

    logger.info("=" * 80)
    logger.info("Experiment 2B: Feature Interaction Pattern Test")
    logger.info("MetaGB vs XGBoost on Synthetic Interaction Data (with Grid Search)")
    logger.info("=" * 80)
    logger.info(f"Base random seed: {seed}")
    logger.info(f"Number of repetitions: {REPEATING_NUMS}")
    logger.info("Feature normalization: Z-Score (StandardScaler)")

    # Get sparse configs (outer loop)
    sparse_configs = get_sparse_configs()
    logger.info(f"\nSparse configurations (outer loop):")
    for cfg in sparse_configs:
        logger.info(f"  - {cfg[0]} features, {cfg[1]} relevant, {cfg[2]}-order interaction")

    # Print grid search info
    xgb_grid = get_exp2b_xgb_param_grid()
    metagb_grid = get_exp2b_metagb_param_grid()
    n_xgb_combinations = int(np.prod([len(v) for v in xgb_grid.values()]))
    n_metagb_combinations = int(np.prod([len(v) for v in metagb_grid.values()]))
    logger.info(f"\nXGBoost grid search: {n_xgb_combinations} combinations")
    logger.info(f"MetaGB grid search: {n_metagb_combinations} combinations")

    # Calculate total experiments
    n_total = len(sparse_configs) * REPEATING_NUMS * 2  # 2 models
    logger.info(f"\nTotal experiments: {n_total}")

    # Run experiments
    all_results = []

    # Outer loop: sparse_config
    for sparse_config in sparse_configs:
        n_total, n_rel, order = sparse_config
        config_name = f"sparse_{n_total}f_{n_rel}r_{order}order"

        logger.info(f"\n{'#'*80}")
        logger.info(f"# SPARSE CONFIG: {config_name}")
        logger.info(f"{'#'*80}")

        # Middle loop: repetitions
        for repeat_idx in range(REPEATING_NUMS):
            logger.info(f"\n--- Repetition {repeat_idx + 1}/{REPEATING_NUMS} ---")

            try:
                results = run_experiment_for_sparse_config(
                    sparse_config=sparse_config,
                    base_seed=seed,
                    repeat_idx=repeat_idx,
                    logger=logger,
                    logs_dir=logs_dir,
                    plots_dir=plots_dir,
                    data_dir=data_dir
                )
                all_results.extend(results)

                # Save intermediate results
                if all_results:
                    df = pd.DataFrame(all_results)
                    df.to_csv(os.path.join(results_dir, 'exp2B_summary_partial.csv'), index=False)

            except Exception as e:
                logger.error(f"Failed for {config_name} rep {repeat_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # Save final summary
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(results_dir, 'exp2B_summary.csv')
        df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to: {summary_path}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT 2B SUMMARY")
        logger.info("=" * 80)

        # Summary by sparse config
        logger.info("\n--- Results by Sparse Config ---")
        for variant in df['Variant'].unique():
            logger.info(f"\n{variant}:")
            variant_df = df[df['Variant'] == variant]

            for model in variant_df['Model'].unique():
                model_df = variant_df[variant_df['Model'] == model]
                avg_rmse = model_df['Test_RMSE'].mean()
                std_rmse = model_df['Test_RMSE'].std()
                logger.info(f"  {model}: Avg RMSE = {avg_rmse:.4f} (+/- {std_rmse:.4f})")

        # Best model per sparse config (averaged across repetitions)
        logger.info("\n--- Best Model per Sparse Config (averaged) ---")
        for variant in df['Variant'].unique():
            variant_df = df[df['Variant'] == variant]
            avg_by_model = variant_df.groupby('Model')['Test_RMSE'].mean()
            best_model = avg_by_model.idxmin()
            best_rmse = avg_by_model.min()
            logger.info(f"  {variant}: {best_model} (Avg RMSE={best_rmse:.4f})")

        # Feature selection precision summary
        logger.info("\n--- Feature Selection Precision (averaged) ---")
        for variant in df['Variant'].unique():
            logger.info(f"\n  {variant}:")
            variant_df = df[df['Variant'] == variant]
            for model in variant_df['Model'].unique():
                model_df = variant_df[variant_df['Model'] == model]
                prec_values = model_df['Feature_Selection_Precision']
                prec_values = prec_values[prec_values != '-'].astype(float)
                if len(prec_values) > 0:
                    logger.info(f"    {model}: {prec_values.mean():.2f} (+/- {prec_values.std():.2f})")

        # Summary statistics per repetition
        logger.info("\n--- Results by Repetition ---")
        for rep in df['Repetition'].unique():
            rep_df = df[df['Repetition'] == rep]
            logger.info(f"Repetition {rep}: Avg Test RMSE = {rep_df['Test_RMSE'].mean():.4f}")

        # Plot tree depth comparison by sparse config
        for variant in df['Variant'].unique():
            variant_df = df[df['Variant'] == variant]
            if len(variant_df) > 0:
                # Aggregate results for plotting
                agg_results = []
                for model in variant_df['Model'].unique():
                    model_df = variant_df[variant_df['Model'] == model]
                    agg_results.append({
                        'model': model,
                        'avg_tree_depth': model_df['Avg_Tree_Depth'].mean(),
                        'std_tree_depth': model_df['Avg_Tree_Depth'].std(),
                        'test_rmse': model_df['Test_RMSE'].mean()
                    })

                try:
                    plot_tree_depth_comparison(
                        results=agg_results,
                        dataset_type=variant,
                        plot_dir=plots_dir
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot tree depth comparison for {variant}: {e}")

    logger.info("\nExperiment 2B completed!")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 2B: Feature Interaction Pattern Test')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs for grid search. -1 uses all CPUs (default: 1)')

    args = parser.parse_args()

    # Set global n_jobs for parallel processing
    from utils.trainer import set_n_jobs
    set_n_jobs(args.n_jobs)

    run_experiment2b(seed=args.seed)
