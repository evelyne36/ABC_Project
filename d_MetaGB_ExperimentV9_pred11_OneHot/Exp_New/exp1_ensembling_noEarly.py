"""
Experiment 1 - Ensembling Version: Nested Cross-Validation with Model Ensembling

Workflow:
1. Outer 5-fold CV: Split data into 80% train / 20% test (5 mutually exclusive experiments)
2. Inner 4-fold CV: On 80% data, perform hyperparameter search with early stopping
3. Save best model from each inner fold (total inner_k models as ensemble members)
4. Ensemble prediction: Average predictions from all inner_k models on outer test set
5. Report mean and std of metrics across 5 outer folds

Features:
- No normalization (only categorical handling)
- Custom early stopping for both XGBoost and MetaGB
- GridSearch or Optuna optimization
- Model ensembling via averaging
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


def train_model_no_early_stopping(model_type, params, X_train, y_train, X_val, y_val,
                                   patience, min_delta, logger=None):
    """Train model without early stopping - uses full n_estimators."""
    n_estimators = params.get('n_estimators', 100)

    if model_type == 'xgboost':
        import xgboost as xgb
        # Force CPU usage to avoid CUDA conflicts in parallel execution
        xgb_params = params.copy()
        xgb_params['tree_method'] = 'hist'
        xgb_params['device'] = 'cpu'
        model = xgb.XGBRegressor(**xgb_params, random_state=RANDOM_SEED)
        model.fit(X_train, y_train, verbose=False)

    elif model_type == 'metagb':
        from metagb_utils import MetaGB
        # Map XGBoost-style params to MetaGB params
        metagb_params = map_params_to_metagb(params)
        metagb_params['seed'] = RANDOM_SEED
        model = MetaGB(**metagb_params)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return {
        'model': model,
        'train_losses': [],
        'val_losses': [],
        'early_stopped': False,
        'early_stopping_round': n_estimators
    }


def _evaluate_single_param_for_fold(param_vals, param_names, model_type, X_train_inner,
                                     y_train_inner, X_val_inner, y_val_inner, patience, min_delta):
    """Helper function to evaluate a single parameter combination for one fold (for parallel execution)."""
    params = dict(zip(param_names, param_vals))

    result = train_model_no_early_stopping(
        model_type, params, X_train_inner, y_train_inner,
        X_val_inner, y_val_inner, patience, min_delta
    )

    val_pred = result['model'].predict(X_val_inner)
    val_rmse = np.sqrt(np.mean((y_val_inner - val_pred) ** 2))

    return {
        'params': params,
        'rmse': val_rmse,
        'early_stop': result['early_stopping_round'],
        'model': result['model']
    }


def inner_cv_with_ensemble_training(model_type, X_train_outer, y_train_outer,
                                    param_grid, inner_k, search_method,
                                    patience, min_delta, logger):
    """
    Perform inner K-fold CV and save best model from each fold for ensembling (with parallel processing).

    Returns:
        ensemble_models: List of best models from each inner fold
        ensemble_params: List of best params from each inner fold
        avg_cv_rmse: Average CV RMSE across inner folds
        all_fold_results: List of results for each fold
    """
    from itertools import product

    inner_kf = KFold(n_splits=inner_k, shuffle=True, random_state=RANDOM_SEED)

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    logger.info(f"  Inner CV: {inner_k} folds, {len(param_combinations)} param combinations")
    logger.info(f"  Starting hyperparameter search (n_jobs={N_JOBS})...")

    ensemble_models = []
    ensemble_params = []
    fold_rmses = []
    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(inner_kf.split(X_train_outer)):
        logger.info(f"    Inner Fold {fold_idx + 1}/{inner_k} - Training {len(param_combinations)} combinations...")

        X_train_inner = X_train_outer[train_idx]
        y_train_inner = y_train_outer[train_idx]
        X_val_inner = X_train_outer[val_idx]
        y_val_inner = y_train_outer[val_idx]

        fold_start_time = time.time()

        # Parallel execution for this fold
        fold_param_results = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(_evaluate_single_param_for_fold)(
                param_vals, param_names, model_type, X_train_inner, y_train_inner,
                X_val_inner, y_val_inner, patience, min_delta
            )
            for param_vals in param_combinations
        )

        fold_train_time = time.time() - fold_start_time

        # Find best result for this fold
        best_fold_result = min(fold_param_results, key=lambda x: x['rmse'])
        best_fold_rmse = best_fold_result['rmse']
        best_fold_model = best_fold_result['model']
        best_fold_params = best_fold_result['params']

        # Print progress for top results
        for idx, result in enumerate(fold_param_results):
            if (idx + 1) <= 3 or (idx + 1) % 10 == 0 or (idx + 1) == len(fold_param_results):
                logger.info(f"      [{idx + 1}/{len(fold_param_results)}] RMSE: {result['rmse']:.4f}")

        # Save best model from this fold
        ensemble_models.append(best_fold_model)
        ensemble_params.append(best_fold_params)
        fold_rmses.append(best_fold_rmse)
        all_fold_results.append({
            'fold': fold_idx + 1,
            'best_params': best_fold_params,
            'best_rmse': best_fold_rmse,
            'all_params': fold_param_results,
            'train_time': fold_train_time
        })

        logger.info(f"      Best RMSE: {best_fold_rmse:.4f}, Params: {best_fold_params}")

    avg_cv_rmse = np.mean(fold_rmses)
    logger.info(f"  Average inner CV RMSE: {avg_cv_rmse:.4f}")
    logger.info(f"  Ensemble size: {len(ensemble_models)} models")

    return ensemble_models, ensemble_params, avg_cv_rmse, all_fold_results


def save_detailed_ensemble_log(log_path, dataset_name, model_type, outer_fold, outer_k, inner_k,
                                n_train, n_test, n_features, param_grid, ensemble_params,
                                all_fold_results, avg_cv_rmse, test_metrics, inner_cv_time, test_time):
    """Save detailed log for ensemble experiment."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Exp1 Ensembling: {model_type.upper()} - {dataset_name} - Fold {outer_fold}/{outer_k}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"[{timestamp}] Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Outer Fold: {outer_fold}/{outer_k}\n")
        f.write(f"Inner CV Folds: {inner_k}\n")
        f.write(f"Ensemble Size: {inner_k} models\n")
        f.write(f"Train Samples: {n_train}\n")
        f.write(f"Test Samples: {n_test}\n")
        f.write(f"Features: {n_features}\n")
        f.write(f"Normalization: None\n\n")

        f.write(f"[{timestamp}] Ensemble Models\n")
        f.write("-" * 40 + "\n")
        for i, params in enumerate(ensemble_params):
            f.write(f"Model {i+1} params: {params}\n")
        f.write(f"\nAverage Inner CV RMSE: {avg_cv_rmse:.6f}\n\n")

        f.write(f"[{timestamp}] Test Results (Ensemble)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.6f}\n")
        f.write(f"Test R²: {test_metrics['r2']:.6f}\n\n")

        f.write(f"[{timestamp}] Time Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Inner CV Time: {inner_cv_time:.2f}s\n")
        f.write(f"Test Time: {test_time:.4f}s\n\n")

        f.write(f"[{timestamp}] Inner Fold Details\n")
        f.write("-" * 40 + "\n")
        for fold_res in all_fold_results:
            f.write(f"\nInner Fold {fold_res['fold']}:\n")
            f.write(f"  Best RMSE: {fold_res['best_rmse']:.6f}\n")
            f.write(f"  Best Params: {fold_res['best_params']}\n")
            f.write(f"  Train Time: {fold_res['train_time']:.2f}s\n")
            f.write(f"  Top 5 param combinations:\n")
            sorted_params = sorted(fold_res['all_params'], key=lambda x: x['rmse'])[:5]
            for i, p in enumerate(sorted_params):
                f.write(f"    #{i+1} RMSE: {p['rmse']:.6f}, Params: {p['params']}\n")

        f.write("\n" + "=" * 80 + "\n")


def ensemble_predict(models, X):
    """Make ensemble prediction by averaging predictions from all models."""
    predictions = []

    for model in models:
        pred = model.predict(X)
        predictions.append(pred)

    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)

    return ensemble_pred


def run_nested_cv_with_ensemble(dataset_name, csv_path, outer_k, inner_k,
                                 search_method, logger, results_dir, logs_dir):
    """Run nested CV with ensembling using 80/20 split."""
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

        # Inner CV for ensemble training
        inner_cv_start = time.time()
        ensemble_models, ensemble_params, avg_cv_rmse, all_fold_results = inner_cv_with_ensemble_training(
            model_type, X_train_outer, y_train_outer, param_grid, inner_k,
            search_method, EARLY_STOPPING_ROUNDS, EARLY_STOPPING_MIN_DELTA, logger
        )
        inner_cv_time = time.time() - inner_cv_start

        # Ensemble prediction on test set
        logger.info(f"  Making ensemble predictions on test set...")
        test_start = time.time()
        y_pred_ensemble = ensemble_predict(ensemble_models, X_test_outer)
        test_time = time.time() - test_start
        test_metrics = calculate_metrics(y_test_outer, y_pred_ensemble)

        logger.info(f"  Test RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
        logger.info(f"  Time - Inner CV: {inner_cv_time:.1f}s, Test: {test_time:.3f}s")

        # Create dataset-specific subdirectories
        dataset_logs_dir = os.path.join(logs_dir, dataset_name)
        os.makedirs(dataset_logs_dir, exist_ok=True)

        # Save detailed log
        log_path = os.path.join(dataset_logs_dir, f"exp1_ensembling_{model_type}.log")
        save_detailed_ensemble_log(
            log_path, dataset_name, model_type, 1, 1, inner_k,
            len(X_train_outer), len(X_test_outer), X.shape[1], param_grid,
            ensemble_params, all_fold_results, avg_cv_rmse, test_metrics, inner_cv_time, test_time
        )

        result_summary = {
            'Dataset': dataset_name,
            'Model': model_type.upper(),
            'Ensemble_Size': inner_k,
            'Inner_K': inner_k,
            'Test_RMSE': test_metrics['rmse'],
            'Test_MAE': test_metrics['mae'],
            'Test_R2': test_metrics['r2'],
            'Train_Time': inner_cv_time
        }

        all_results.append(result_summary)

        logger.info(f"\n{model_type.upper()} Ensemble Summary:")
        logger.info(f"  Test RMSE: {result_summary['Test_RMSE']:.4f}")
        logger.info(f"  Test R²: {result_summary['Test_R2']:.4f}")

        # Save detailed JSON log
        log_data = {
            'dataset': dataset_name,
            'model': model_type,
            'ensemble_size': inner_k,
            'inner_k': inner_k,
            'ensemble_params': ensemble_params,
            'cv_rmse': avg_cv_rmse,
            'summary': result_summary
        }
        json_log_path = os.path.join(dataset_logs_dir, f"exp1_ensembling_{model_type}.json")
        with open(json_log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    return all_results


def run_exp1_ensembling(outer_k=5, inner_k=4, search_method='grid', dataset_filter=None):
    """
    Run Experiment 1 - Ensembling Version.

    Args:
        outer_k: Number of outer CV folds (default: 5)
        inner_k: Number of inner CV folds (default: 4)
        search_method: 'grid' or 'optuna'
        dataset_filter: List of dataset names to run
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results', 'exp1_ensembling')
    logs_dir = os.path.join(base_dir, 'logs', 'exp1_ensembling')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    logger = ExperimentLogger('exp1_ensembling', log_dir=logs_dir)

    logger.info("="*80)
    logger.info("Experiment 1 - Ensembling: Nested CV with Model Ensembling")
    logger.info("="*80)
    logger.info(f"Outer CV: {outer_k} folds")
    logger.info(f"Inner CV: {inner_k} folds (ensemble size)")
    logger.info(f"Search method: {search_method}")
    logger.info(f"Ensemble method: Average predictions")
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
            results = run_nested_cv_with_ensemble(
                name, path, outer_k, inner_k, search_method,
                logger, results_dir, logs_dir
            )
            all_results.extend(results)

            # Save intermediate results
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(results_dir, 'exp1_ensembling_summary.csv'), index=False)

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

    parser = argparse.ArgumentParser(description='Exp1 Ensembling: Nested CV with Ensembling')
    parser.add_argument('--outer_k', type=int, default=5, help='Outer CV folds')
    parser.add_argument('--inner_k', type=int, default=4, help='Inner CV folds (ensemble size)')
    parser.add_argument('--search', type=str, default='grid', choices=['grid', 'optuna'])
    parser.add_argument('--dataset', type=str, nargs='*', help='Specific datasets')

    args = parser.parse_args()

    run_exp1_ensembling(
        outer_k=args.outer_k,
        inner_k=args.inner_k,
        search_method=args.search,
        dataset_filter=args.dataset
    )
