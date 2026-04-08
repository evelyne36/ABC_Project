"""
Experiment 1 - Best Params Version: Nested Cross-Validation with Best Hyperparameters

Workflow:
1. Outer 5-fold CV: Split data into 80% train / 20% test (5 mutually exclusive experiments)
2. Inner 4-fold CV: On 80% data, perform hyperparameter search with early stopping
3. Select best hyperparameters based on average inner CV RMSE
4. Retrain on full 80% using average early stopping rounds from inner CV
5. Evaluate on outer 20% test set
6. Report mean and std of metrics across 5 outer folds

Features:
- No normalization (only categorical handling)
- Custom early stopping for both XGBoost and MetaGB
- GridSearch or Optuna optimization
- Parallel acceleration
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metagb'))

from utils.hyperparams import (
    RANDOM_SEED, EXP1_DATASETS, CATEGORICAL_HANDLING,
    get_xgb_param_grid, get_metagb_param_grid,
    EARLY_STOPPING_ROUNDS, EARLY_STOPPING_MIN_DELTA
)
from utils.trainer import calculate_metrics, EarlyStopping
from utils.logger import ExperimentLogger
from utils.visualizer import plot_training_curve
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# Number of parallel jobs for parameter search (-1 = all CPUs, 1 = sequential)
N_JOBS = -1


def filter_param_grid_for_model(param_grid, model_type):
    """
    Filter parameter grid to only include parameters relevant to the specified model.

    Args:
        param_grid: Full parameter grid
        model_type: 'xgboost' or 'metagb'

    Returns:
        Filtered parameter grid
    """
    filtered_grid = param_grid.copy()

    if model_type == 'xgboost':
        # Remove MetaGB-specific parameters
        metagb_only_params = ['pred_feature_fraction', 'min_data_in_leaf', 'version']
        for param in metagb_only_params:
            filtered_grid.pop(param, None)
    elif model_type == 'metagb':
        # Remove XGBoost-specific parameters
        xgb_only_params = ['min_child_weight', 'reg_lambda']
        for param in xgb_only_params:
            filtered_grid.pop(param, None)

    return filtered_grid


def map_params_to_metagb(params):
    """
    Map XGBoost-style parameter names to MetaGB parameter names.

    Args:
        params: Dictionary with XGBoost-style parameter names

    Returns:
        Dictionary with MetaGB parameter names
    """
    metagb_params = {}

    param_mapping = {
        'n_estimators': 'num_trees',
        'colsample_bytree': 'feature_fraction',
        'subsample': 'bagging_fraction'
    }

    for key, value in params.items():
        if key in param_mapping:
            metagb_params[param_mapping[key]] = value
        else:
            metagb_params[key] = value

    return metagb_params


def load_dataset_no_normalization(csv_path, categorical_handling='drop'):
    """Load dataset with categorical handling but NO normalization."""
    df = pd.read_csv(csv_path)

    if categorical_handling == 'drop':
        df = df.select_dtypes(include=[np.number]).dropna()
    elif categorical_handling == 'onehot':
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df = df.dropna()

    if len(df) == 0:
        raise ValueError(f"No valid data in {csv_path}")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y


def train_with_early_stopping(model_type, params, X_train, y_train, X_val, y_val,
                               patience, min_delta, logger=None):
    """Train model with validation-based early stopping."""
    if model_type == 'xgboost':
        import xgboost as xgb
        # Force CPU usage to avoid CUDA conflicts in parallel execution
        xgb_params = params.copy()
        xgb_params['tree_method'] = 'hist'
        xgb_params['device'] = 'cpu'
        model = xgb.XGBRegressor(**xgb_params, random_state=RANDOM_SEED, early_stopping_rounds=patience)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        n_estimators_used = model.best_iteration + 1 if hasattr(model, 'best_iteration') else params.get('n_estimators', 100)
        early_stopped = hasattr(model, 'best_iteration')

    elif model_type == 'metagb':
        from metagb_utils import MetaGB
        # Map XGBoost-style params to MetaGB params
        metagb_params = map_params_to_metagb(params)
        n_estimators = metagb_params.get('num_trees', 100)  # Read but don't remove
        metagb_params['seed'] = RANDOM_SEED

        # Train once with full n_estimators
        model = MetaGB(**metagb_params)
        model.fit(X_train, y_train)

        # Find optimal number of trees using validation set
        best_val_mse = float('inf')
        best_n_trees = n_estimators
        no_improve_count = 0

        for n_trees in range(1, n_estimators + 1):
            # Predict using first n_trees
            y_val_pred = model.predict(X_val, end_tree_idx=n_trees)
            val_mse = np.mean((y_val - y_val_pred) ** 2)

            if val_mse < best_val_mse - min_delta:
                best_val_mse = val_mse
                best_n_trees = n_trees
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

        n_estimators_used = best_n_trees
        early_stopped = (best_n_trees < n_estimators)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return {
        'model': model,
        'train_losses': [],
        'val_losses': [],
        'early_stopped': early_stopped,
        'early_stopping_round': n_estimators_used
    }


def _evaluate_single_param(param_vals, param_names, model_type, X_train_outer, y_train_outer,
                           inner_k, patience, min_delta):
    """Helper function to evaluate a single parameter combination (for parallel execution)."""
    params = dict(zip(param_names, param_vals))
    inner_kf = KFold(n_splits=inner_k, shuffle=True, random_state=RANDOM_SEED)

    fold_rmses = []
    fold_early_stops = []
    param_start_time = time.time()

    for train_idx, val_idx in inner_kf.split(X_train_outer):
        X_train_inner = X_train_outer[train_idx]
        y_train_inner = y_train_outer[train_idx]
        X_val_inner = X_train_outer[val_idx]
        y_val_inner = y_train_outer[val_idx]

        result = train_with_early_stopping(
            model_type, params, X_train_inner, y_train_inner,
            X_val_inner, y_val_inner, patience, min_delta
        )

        val_pred = result['model'].predict(X_val_inner)
        val_rmse = np.sqrt(np.mean((y_val_inner - val_pred) ** 2))

        fold_rmses.append(val_rmse)
        fold_early_stops.append(result['early_stopping_round'])

    param_train_time = time.time() - param_start_time
    avg_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    avg_early_stop = int(np.mean(fold_early_stops))

    return {
        'params': params,
        'mean_cv_rmse': avg_rmse,
        'std_cv_rmse': std_rmse,
        'avg_early_stop': avg_early_stop,
        'train_time': param_train_time
    }


def inner_cv_with_hyperparameter_search(model_type, X_train_outer, y_train_outer,
                                        param_grid, inner_k, search_method,
                                        patience, min_delta, logger):
    """
    Perform inner K-fold CV for hyperparameter search with parallel processing.

    Returns:
        best_params: Best hyperparameters
        avg_early_stop_rounds: Average early stopping rounds
        best_cv_rmse: Best CV RMSE
        all_cv_results: List of all parameter combination results
    """
    from itertools import product

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    logger.info(f"  Inner CV: {inner_k} folds, {len(param_combinations)} param combinations")
    logger.info(f"  Starting hyperparameter search (n_jobs={N_JOBS})...")

    # Parallel execution
    all_cv_results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(_evaluate_single_param)(
            param_vals, param_names, model_type, X_train_outer, y_train_outer,
            inner_k, patience, min_delta
        )
        for param_vals in param_combinations
    )

    # Find best parameters
    best_params = None
    best_cv_rmse = float('inf')
    best_avg_early_stop = None

    for idx, result in enumerate(all_cv_results):
        # Print progress: first 3, then every 10, and last
        if (idx + 1) <= 3 or (idx + 1) % 10 == 0 or (idx + 1) == len(all_cv_results):
            logger.info(f"    [{idx + 1}/{len(all_cv_results)}] CV RMSE: {result['mean_cv_rmse']:.4f} (±{result['std_cv_rmse']:.4f})")

        if result['mean_cv_rmse'] < best_cv_rmse:
            best_cv_rmse = result['mean_cv_rmse']
            best_params = result['params'].copy()
            best_avg_early_stop = result['avg_early_stop']

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV RMSE: {best_cv_rmse:.4f}")
    logger.info(f"  Avg early stop rounds: {best_avg_early_stop}")

    return best_params, best_avg_early_stop, best_cv_rmse, all_cv_results


def save_detailed_fold_log(log_path, dataset_name, model_type, outer_fold, outer_k, inner_k,
                           n_train, n_test, n_features, param_grid, best_params,
                           avg_early_stop, best_cv_rmse, all_cv_results, test_metrics,
                           inner_cv_time, final_train_time, test_time):
    """Save detailed log for a single outer fold."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Exp1 BestParams: {model_type.upper()} - {dataset_name} - Fold {outer_fold}/{outer_k}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"[{timestamp}] Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Outer Fold: {outer_fold}/{outer_k}\n")
        f.write(f"Inner CV Folds: {inner_k}\n")
        f.write(f"Train Samples: {n_train}\n")
        f.write(f"Test Samples: {n_test}\n")
        f.write(f"Features: {n_features}\n")
        f.write(f"Normalization: None\n\n")

        f.write(f"[{timestamp}] Parameter Grid\n")
        f.write("-" * 40 + "\n")
        for k, v in param_grid.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"Total combinations: {len(all_cv_results)}\n\n")

        f.write(f"[{timestamp}] Best Parameters (Inner CV)\n")
        f.write("-" * 40 + "\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"Best CV RMSE: {best_cv_rmse:.6f}\n")
        f.write(f"Avg Early Stop Rounds: {avg_early_stop}\n\n")

        f.write(f"[{timestamp}] Test Results (Outer Fold)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.6f}\n")
        f.write(f"Test R²: {test_metrics['r2']:.6f}\n\n")

        f.write(f"[{timestamp}] Time Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Inner CV Time: {inner_cv_time:.2f}s\n")
        f.write(f"Final Train Time: {final_train_time:.2f}s\n")
        f.write(f"Test Time: {test_time:.4f}s\n")
        f.write(f"Total Time: {inner_cv_time + final_train_time:.2f}s\n\n")

        f.write(f"[{timestamp}] Top 10 Inner CV Results\n")
        f.write("-" * 40 + "\n")
        sorted_results = sorted(all_cv_results, key=lambda x: x['mean_cv_rmse'])[:10]
        for i, res in enumerate(sorted_results):
            f.write(f"\n#{i+1} CV RMSE: {res['mean_cv_rmse']:.6f} (±{res['std_cv_rmse']:.6f})\n")
            f.write(f"   Params: {res['params']}\n")
            f.write(f"   Avg Early Stop: {res['avg_early_stop']}\n")
            f.write(f"   Train Time: {res['train_time']:.2f}s\n")

        f.write("\n" + "=" * 80 + "\n")


def run_nested_cv_for_dataset(dataset_name, csv_path, outer_k, inner_k,
                               search_method, logger, results_dir, logs_dir):
    """Run nested CV for a single dataset with 80/20 split."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"{'='*80}")

    # Load data (no normalization)
    X, y = load_dataset_no_normalization(csv_path, CATEGORICAL_HANDLING)
    logger.info(f"Total samples: {len(X)}, Features: {X.shape[1]}")

    # Simple 80/20 train-test split
    X_train_outer, X_test_outer, y_train_outer, y_test_outer = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    logger.info(f"Train: {len(X_train_outer)}, Test: {len(X_test_outer)}")

    all_results = []

    for model_type in ['xgboost', 'metagb']:
        logger.info(f"\n--- {model_type.upper()} ---")

        param_grid = get_xgb_param_grid(dataset_name) if model_type == 'xgboost' else get_metagb_param_grid(dataset_name)
        param_grid = filter_param_grid_for_model(param_grid, model_type)

        # Inner CV for hyperparameter search
        inner_cv_start = time.time()
        best_params, avg_early_stop, best_cv_rmse, all_cv_results = inner_cv_with_hyperparameter_search(
            model_type, X_train_outer, y_train_outer, param_grid, inner_k,
            search_method, EARLY_STOPPING_ROUNDS, EARLY_STOPPING_MIN_DELTA, logger
        )
        inner_cv_time = time.time() - inner_cv_start

        # Retrain on full train set with avg early stop rounds
        logger.info(f"  Retraining on full train set (n_estimators={avg_early_stop})...")
        final_params = best_params.copy()
        final_params['n_estimators'] = avg_early_stop

        if model_type == 'xgboost':
            import xgboost as xgb
            # Force CPU usage to avoid CUDA conflicts
            final_params['tree_method'] = 'hist'
            final_params['device'] = 'cpu'
            final_model = xgb.XGBRegressor(**final_params, random_state=RANDOM_SEED)
        else:
            from metagb_utils import MetaGB
            # Map XGBoost-style params to MetaGB params
            metagb_final_params = map_params_to_metagb(final_params)
            metagb_final_params['seed'] = RANDOM_SEED
            final_model = MetaGB(**metagb_final_params)

        final_train_start = time.time()
        final_model.fit(X_train_outer, y_train_outer)
        final_train_time = time.time() - final_train_start

        # Evaluate on test set
        test_start = time.time()
        y_pred = final_model.predict(X_test_outer)
        test_time = time.time() - test_start
        test_metrics = calculate_metrics(y_test_outer, y_pred)

        total_time = inner_cv_time + final_train_time

        logger.info(f"  Test RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
        logger.info(f"  Time - Inner CV: {inner_cv_time:.1f}s, Final train: {final_train_time:.1f}s, Test: {test_time:.3f}s")

        # Create dataset-specific subdirectories
        dataset_logs_dir = os.path.join(logs_dir, dataset_name)
        os.makedirs(dataset_logs_dir, exist_ok=True)

        # Save detailed log
        log_path = os.path.join(dataset_logs_dir, f"exp1_bestparams_{model_type}.log")
        save_detailed_fold_log(
            log_path, dataset_name, model_type, 1, 1, inner_k,
            len(X_train_outer), len(X_test_outer), X.shape[1], param_grid,
            best_params, avg_early_stop, best_cv_rmse, all_cv_results,
            test_metrics, inner_cv_time, final_train_time, test_time
        )

        result_summary = {
            'Dataset': dataset_name,
            'Model': model_type.upper(),
            'Inner_K': inner_k,
            'Test_RMSE': test_metrics['rmse'],
            'Test_MAE': test_metrics['mae'],
            'Test_R2': test_metrics['r2'],
            'Train_Time': total_time
        }

        all_results.append(result_summary)

        logger.info(f"\n{model_type.upper()} Summary:")
        logger.info(f"  Test RMSE: {result_summary['Test_RMSE']:.4f}")
        logger.info(f"  Test R²: {result_summary['Test_R2']:.4f}")

        # Save detailed JSON log
        log_data = {
            'dataset': dataset_name,
            'model': model_type,
            'inner_k': inner_k,
            'best_params': best_params,
            'cv_rmse': best_cv_rmse,
            'summary': result_summary
        }
        json_log_path = os.path.join(dataset_logs_dir, f"exp1_bestparams_{model_type}.json")
        with open(json_log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    return all_results


def run_exp1_bestparams(outer_k=5, inner_k=4, search_method='grid', dataset_filter=None):
    """
    Run Experiment 1 - Best Params Version.

    Args:
        outer_k: Number of outer CV folds (default: 5)
        inner_k: Number of inner CV folds (default: 4)
        search_method: 'grid' or 'optuna'
        dataset_filter: List of dataset names to run
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results', 'exp1_bestparams')
    logs_dir = os.path.join(base_dir, 'logs', 'exp1_bestparams')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    logger = ExperimentLogger('exp1_bestparams', log_dir=logs_dir)

    logger.info("="*80)
    logger.info("Experiment 1 - Best Params: Nested Cross-Validation")
    logger.info("="*80)
    logger.info(f"Outer CV: {outer_k} folds")
    logger.info(f"Inner CV: {inner_k} folds")
    logger.info(f"Search method: {search_method}")
    logger.info(f"No normalization applied")

    # Get datasets
    data_dir = os.path.join(base_dir, '..', 'dataset', 'tabArena_regression')
    datasets = dataset_filter if dataset_filter else EXP1_DATASETS

    all_results = []

    for ds in datasets:
        name = os.path.splitext(ds)[0]
        path = os.path.join(data_dir, ds)

        if not os.path.exists(path):
            logger.warning(f"Dataset not found: {path}")
            continue

        try:
            results = run_nested_cv_for_dataset(
                name, path, outer_k, inner_k, search_method,
                logger, results_dir, logs_dir
            )
            all_results.extend(results)

            # Save intermediate results
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(results_dir, 'exp1_bestparams_summary.csv'), index=False)

        except Exception as e:
            logger.error(f"Failed on {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Final summary
    if all_results:
        df = pd.DataFrame(all_results)
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY")
        logger.info("="*80)
        logger.info("\n" + df.to_string(index=False))

    logger.info("\nExperiment completed!")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Exp1 BestParams: Nested CV')
    parser.add_argument('--outer_k', type=int, default=5, help='Outer CV folds')
    parser.add_argument('--inner_k', type=int, default=4, help='Inner CV folds')
    parser.add_argument('--search', type=str, default='grid', choices=['grid', 'optuna'])
    parser.add_argument('--dataset', type=str, nargs='*', help='Specific datasets')

    args = parser.parse_args()

    run_exp1_bestparams(
        outer_k=args.outer_k,
        inner_k=args.inner_k,
        search_method=args.search,
        dataset_filter=args.dataset
    )
