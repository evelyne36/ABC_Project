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
from sklearn.model_selection import KFold

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metagb'))

from utils.hyperparams import (
    RANDOM_SEED, EXP1_DATASETS, CATEGORICAL_HANDLING,
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
XGB_INNER_N_JOBS = 1


def get_param_parallel_jobs(model_type):
    """Return hyperparameter-level parallel jobs by model type."""
    return N_JOBS


def build_outer_fold_suffix(outer_fold_num):
    """Build filename suffix for outer fold tracking."""
    return f"_out{outer_fold_num}"


def _sample_log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _sample_log_uniform_int(rng, low, high):
    sampled = np.exp(rng.uniform(np.log(low), np.log(high)))
    return int(np.clip(np.round(sampled), low, high))


def _attach_n_estimators(candidates, rng):
    n_estimators_values = [200, 400, 600, 800]
    assigned = np.tile(
        n_estimators_values,
        int(np.ceil(len(candidates) / len(n_estimators_values)))
    )[:len(candidates)]
    rng.shuffle(assigned)
    for params, n_estimators in zip(candidates, assigned):
        params['n_estimators'] = int(n_estimators)


def load_xgb_random_candidates(config_path, num_candidates=200):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    random_keys = [f"XGBoost_r{i}" for i in range(1, num_candidates + 1)]
    missing_keys = [key for key in random_keys if key not in config_data]
    if missing_keys:
        raise ValueError(f"Missing XGBoost configs in {config_path}: {missing_keys[:5]} ...")

    candidates = []
    for key in random_keys:
        raw_params = config_data[key].get('hyperparameters', {})
        params = {k: v for k, v in raw_params.items() if k != 'ag_args'}
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        if 'max_cat_to_onehot' in params:
            params['max_cat_to_onehot'] = int(params['max_cat_to_onehot'])
        if 'max_leaves' in params:
            params['max_leaves'] = int(params['max_leaves'])
        candidates.append(params)

    attach_rng = np.random.default_rng(RANDOM_SEED + 2000)
    _attach_n_estimators(candidates, attach_rng)
    return candidates


def _sample_metagb_candidate(rng, pred_feature_fraction):
    return {
        'pred_feature_fraction': float(pred_feature_fraction),
        'learning_rate': _sample_log_uniform(rng, 0.005, 0.1),
        'max_depth': _sample_log_uniform_int(rng, 4, 10),
        'min_child_weight': _sample_log_uniform(rng, 0.001, 5.0),
        'subsample': float(rng.uniform(0.6, 1.0)),
        'colsample': float(rng.uniform(0.6, 1.0)),
        'version': 3,
        'max_bin': 255,
    }


def generate_metagb_random_candidates(num_candidates=200):
    if num_candidates < 11 * 18:
        raise ValueError("num_candidates must be at least 198 for class-balanced sampling")

    rng = np.random.default_rng(RANDOM_SEED + 3000)
    pred_feature_classes = [round(i * 0.1, 1) for i in range(11)]
    candidates = []

    for pred_feature_fraction in pred_feature_classes:
        for _ in range(18):
            candidates.append(_sample_metagb_candidate(rng, pred_feature_fraction))

    remaining = num_candidates - len(candidates)
    for _ in range(remaining):
        pred_feature_fraction = float(rng.choice(pred_feature_classes))
        candidates.append(_sample_metagb_candidate(rng, pred_feature_fraction))

    attach_rng = np.random.default_rng(RANDOM_SEED + 4000)
    _attach_n_estimators(candidates, attach_rng)
    return candidates


def get_random_search_candidates(model_type):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_type == 'xgboost':
        config_path = os.path.join(base_dir, 'configs_xgboost.json')
        candidates = load_xgb_random_candidates(config_path=config_path, num_candidates=200)
        search_info = {
            'search_type': 'random',
            'candidate_source': 'configs_xgboost.json:XGBoost_r1~XGBoost_r200',
            'num_candidates': len(candidates),
            'n_estimators': [200, 400, 600, 800],
        }
        return candidates, search_info

    if model_type == 'metagb':
        candidates = generate_metagb_random_candidates(num_candidates=200)
        search_info = {
            'search_type': 'random',
            'candidate_source': 'generated_metagb_200_points',
            'pred_feature_fraction_sampling': '0.0~1.0 step 0.1, each class 18 + random 2',
            'num_candidates': len(candidates),
            'n_estimators': [200, 400, 600, 800],
        }
        return candidates, search_info

    raise ValueError(f"Unknown model type: {model_type}")


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
        'colsample': 'feature_fraction',
        'subsample': 'bagging_fraction',
        'min_child_weight': 'min_data_in_leaf'
    }

    for key, value in params.items():
        if key == 'min_child_weight':
            metagb_params['min_data_in_leaf'] = max(1, int(round(value)))
        elif key in param_mapping:
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
    elif categorical_handling == 'ordinal':
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                df[col], _ = pd.factorize(df[col], sort=True)
        df = df.dropna()
    else:
        raise ValueError(
            f"Unsupported categorical_handling: {categorical_handling}. "
            "Expected one of ['drop', 'onehot', 'ordinal']."
        )

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
        xgb_params['n_jobs'] = XGB_INNER_N_JOBS
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


def _evaluate_single_param(params, model_type, X_train_outer, y_train_outer,
                           inner_k, patience, min_delta):
    """Helper function to evaluate a single parameter combination (for parallel execution)."""
    inner_kf = KFold(n_splits=inner_k, shuffle=True, random_state=RANDOM_SEED)

    fold_rmses = []
    fold_early_stops = []
    param_start_time = time.time()

    for train_idx, val_idx in inner_kf.split(X_train_outer):
        X_train_inner = X_train_outer[train_idx]
        y_train_inner = y_train_outer[train_idx]
        X_val_inner = X_train_outer[val_idx]
        y_val_inner = y_train_outer[val_idx]

        result = train_model_no_early_stopping(
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
                                        param_candidates, inner_k, search_method,
                                        patience, min_delta, logger):
    """
    Perform inner K-fold CV for hyperparameter search with parallel processing.

    Returns:
        best_params: Best hyperparameters
        avg_early_stop_rounds: Average early stopping rounds
        best_cv_rmse: Best CV RMSE
        all_cv_results: List of all parameter combination results
    """
    logger.info(f"  Inner CV: {inner_k} folds, {len(param_candidates)} param candidates")
    param_parallel_jobs = get_param_parallel_jobs(model_type)
    logger.info(
        f"  Starting hyperparameter search: "
        f"xgboost->param-level(n_jobs={param_parallel_jobs}, model_n_jobs={XGB_INNER_N_JOBS}), "
        f"metagb->param-level(n_jobs={param_parallel_jobs})"
    )

    all_cv_results = Parallel(n_jobs=param_parallel_jobs, verbose=0)(
        delayed(_evaluate_single_param)(
            params, model_type, X_train_outer, y_train_outer,
            inner_k, patience, min_delta
        )
        for params in param_candidates
    )

    # Find best parameters
    best_params = None
    best_cv_rmse = float('inf')
    best_avg_early_stop = None

    for idx, result in enumerate(all_cv_results):
        # Print progress: first 3, then every 10, and last
        if (idx + 1) <= 3 or (idx + 1) % 10 == 0 or (idx + 1) == len(all_cv_results):
            logger.info(f"    [{idx + 1}/{len(all_cv_results)}] CV RMSE: {result['mean_cv_rmse']:.4f} (卤{result['std_cv_rmse']:.4f})")

        if result['mean_cv_rmse'] < best_cv_rmse:
            best_cv_rmse = result['mean_cv_rmse']
            best_params = result['params'].copy()
            best_avg_early_stop = result['avg_early_stop']

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV RMSE: {best_cv_rmse:.4f}")
    logger.info(f"  Avg early stop rounds: {best_avg_early_stop}")

    return best_params, best_avg_early_stop, best_cv_rmse, all_cv_results


def save_detailed_fold_log(log_path, dataset_name, model_type, outer_fold, outer_k, inner_k,
                           n_train, n_test, n_features, param_search_info, best_params,
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
        for k, v in param_search_info.items():
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
        f.write(f"Test R虏: {test_metrics['r2']:.6f}\n\n")

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
            f.write(f"\n#{i+1} CV RMSE: {res['mean_cv_rmse']:.6f} (卤{res['std_cv_rmse']:.6f})\n")
            f.write(f"   Params: {res['params']}\n")
            f.write(f"   Avg Early Stop: {res['avg_early_stop']}\n")
            f.write(f"   Train Time: {res['train_time']:.2f}s\n")

        f.write("\n" + "=" * 80 + "\n")


def run_nested_cv_for_dataset(dataset_name, csv_path, outer_k, inner_k,
                               search_method, logger, results_dir, logs_dir):
    """Run nested CV for a single dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"{'='*80}")

    X, y = load_dataset_no_normalization(csv_path, CATEGORICAL_HANDLING)
    logger.info(f"Total samples: {len(X)}, Features: {X.shape[1]}")

    all_results = []
    dataset_logs_dir = os.path.join(logs_dir, dataset_name)
    os.makedirs(dataset_logs_dir, exist_ok=True)
    outer_kf = KFold(n_splits=outer_k, shuffle=True, random_state=RANDOM_SEED)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_kf.split(X), start=1):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        logger.info(f"\n[Outer Fold {outer_fold}/{outer_k}] Train: {len(X_train_outer)}, Test: {len(X_test_outer)}")

        for model_type in ['xgboost', 'metagb']:
            logger.info(f"\n--- {model_type.upper()} | Outer Fold {outer_fold}/{outer_k} ---")

            param_candidates, param_search_info = get_random_search_candidates(model_type)

            inner_cv_start = time.time()
            best_params, avg_early_stop, best_cv_rmse, all_cv_results = inner_cv_with_hyperparameter_search(
                model_type, X_train_outer, y_train_outer, param_candidates, inner_k,
                search_method, EARLY_STOPPING_ROUNDS, EARLY_STOPPING_MIN_DELTA, logger
            )
            inner_cv_time = time.time() - inner_cv_start

            logger.info(f"  Retraining on full train set (n_estimators={avg_early_stop})...")
            final_params = best_params.copy()
            final_params['n_estimators'] = avg_early_stop

            if model_type == 'xgboost':
                import xgboost as xgb
                final_params['tree_method'] = 'hist'
                final_params['device'] = 'cpu'
                final_params['n_jobs'] = XGB_INNER_N_JOBS
                final_model = xgb.XGBRegressor(**final_params, random_state=RANDOM_SEED)
            else:
                from metagb_utils import MetaGB
                metagb_final_params = map_params_to_metagb(final_params)
                metagb_final_params['seed'] = RANDOM_SEED
                final_model = MetaGB(**metagb_final_params)

            final_train_start = time.time()
            final_model.fit(X_train_outer, y_train_outer)
            final_train_time = time.time() - final_train_start

            test_start = time.time()
            y_pred = final_model.predict(X_test_outer)
            test_time = time.time() - test_start
            test_metrics = calculate_metrics(y_test_outer, y_pred)
            total_time = inner_cv_time + final_train_time

            logger.info(f"  Test RMSE: {test_metrics['rmse']:.4f}, R2: {test_metrics['r2']:.4f}")
            logger.info(f"  Time - Inner CV: {inner_cv_time:.1f}s, Final train: {final_train_time:.1f}s, Test: {test_time:.3f}s")

            fold_suffix = build_outer_fold_suffix(outer_fold)
            log_path = os.path.join(dataset_logs_dir, f"exp1_bestparams_{model_type}{fold_suffix}.log")
            save_detailed_fold_log(
                log_path, dataset_name, model_type, outer_fold, outer_k, inner_k,
                len(X_train_outer), len(X_test_outer), X.shape[1], param_search_info,
                best_params, avg_early_stop, best_cv_rmse, all_cv_results,
                test_metrics, inner_cv_time, final_train_time, test_time
            )

            result_summary = {
                'Dataset': dataset_name,
                'Model': model_type.upper(),
                'Outer_Fold': outer_fold,
                'Outer_K': outer_k,
                'Inner_K': inner_k,
                'CV_RMSE': best_cv_rmse,
                'Test_RMSE': test_metrics['rmse'],
                'Test_MAE': test_metrics['mae'],
                'Test_R2': test_metrics['r2'],
                'Train_Time': total_time
            }
            all_results.append(result_summary)

            logger.info(f"\n{model_type.upper()} Fold Summary:")
            logger.info(f"  Test RMSE: {result_summary['Test_RMSE']:.4f}")
            logger.info(f"  Test R2: {result_summary['Test_R2']:.4f}")

            log_data = {
                'dataset': dataset_name,
                'model': model_type,
                'outer_fold': outer_fold,
                'outer_k': outer_k,
                'inner_k': inner_k,
                'best_params': best_params,
                'cv_rmse': best_cv_rmse,
                'summary': result_summary
            }
            json_log_path = os.path.join(dataset_logs_dir, f"exp1_bestparams_{model_type}{fold_suffix}.json")
            with open(json_log_path, 'w') as f:
                json.dump(log_data, f, indent=2)

    for model_name in ['XGBOOST', 'METAGB']:
        model_results = [result for result in all_results if result['Model'] == model_name]
        if not model_results:
            continue
        rmse_values = [result['Test_RMSE'] for result in model_results]
        r2_values = [result['Test_R2'] for result in model_results]
        logger.info(
            f"\n{model_name} Outer-CV Summary: "
            f"RMSE = {np.mean(rmse_values):.4f} 卤 {np.std(rmse_values):.4f}, "
            f"R2 = {np.mean(r2_values):.4f} 卤 {np.std(r2_values):.4f}"
        )

    return all_results


def save_outer_cv_aggregate(all_results, results_dir, outer_k):
    """Save outer-fold aggregated mean/std results."""
    if not all_results:
        return

    df = pd.DataFrame(all_results)
    group_cols = ['Dataset', 'Model', 'Outer_K', 'Inner_K']
    metric_cols = ['CV_RMSE', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Train_Time']

    agg_df = df.groupby(group_cols, as_index=False)[metric_cols].agg(['mean', 'std'])
    agg_df.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in agg_df.columns.to_flat_index()
    ]

    outer_suffix = build_outer_fold_suffix(outer_k)
    csv_path = os.path.join(results_dir, f'exp1_bestparams_outercv_aggregate{outer_suffix}.csv')
    json_path = os.path.join(results_dir, f'exp1_bestparams_outercv_aggregate{outer_suffix}.json')
    agg_df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(agg_df.to_dict(orient='records'), f, indent=2)

def run_exp1_bestparams(outer_k=5, inner_k=4, search_method='random', dataset_filter=None):
    """
    Run Experiment 1 - Best Params Version.

    Args:
        outer_k: Number of outer CV folds (default: 5)
        inner_k: Number of inner CV folds (default: 4)
        search_method: random search over predefined 200 candidates
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

    data_dir = os.path.join(base_dir, '..', 'dataset', 'tabArena_regression')
    datasets = dataset_filter if dataset_filter else EXP1_DATASETS

    all_results = []
    outer_suffix = build_outer_fold_suffix(outer_k)

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

            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(results_dir, f'exp1_bestparams_summary{outer_suffix}.csv'), index=False)
                save_outer_cv_aggregate(all_results, results_dir, outer_k)

        except Exception as e:
            logger.error(f"Failed on {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if all_results:
        df = pd.DataFrame(all_results)
        save_outer_cv_aggregate(all_results, results_dir, outer_k)
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
    parser.add_argument('--search', type=str, default='random', choices=['random', 'grid', 'optuna'])
    parser.add_argument('--dataset', type=str, nargs='*', help='Specific datasets')

    args = parser.parse_args()

    run_exp1_bestparams(
        outer_k=args.outer_k,
        inner_k=args.inner_k,
        search_method=args.search,
        dataset_filter=args.dataset
    )

