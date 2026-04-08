"""
Experiment: Study the impact of pred_feature_fraction on MetaGB performance

For each hyperparameter combination (fixed except n_estimators):
- Train 1 XGBoost model
- Train 11 MetaGB models with pred_feature_fraction from 0.0 to 1.0 (step 0.1)
- Only optimize n_estimators (simulating early stopping)

All hyperparameters are configured in this file independently.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metagb'))

from utils.trainer import calculate_metrics
from utils.logger import ExperimentLogger
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration (All hyperparameters defined here)
# =============================================================================

RANDOM_SEED = 42
N_JOBS = -1

# Datasets to test
EXP1_DATASETS = [
    #"airfoil_self_noise.csv",
    #"Another-Dataset-on-used-Fiat-500.csv",
    #"concrete_compressive_strength.csv",
    #"diamonds.csv",
    #"Food_Delivery_Time.csv",#有问题，意外中断
    "healthcare_insurance_expenses.csv",
    "houses.csv",
    "miami_housing.csv",
    "physiochemical_protein.csv",
    "QSAR_fish_toxicity.csv",
    "wine_quality.csv"
]

# Categorical handling: 'drop' or 'onehot'
CATEGORICAL_HANDLING = 'onehot'

# Pred feature fraction values to test (0.0 to 1.0, step 0.1)
PRED_FRACTIONS = [round(x * 0.1, 1) for x in range(11)]

# Fixed hyperparameter grid (excluding n_estimators)
# These parameters are shared between XGBoost and MetaGB

#总组合=2*3*2*2*（1个XGBoost+11个MetaGB）=288个实验
FIXED_PARAM_GRID = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}


# n_estimators search space (only this parameter is optimized)
# 模拟早停，范围较大，仅用于寻找最佳n_estimators，最终结果只保留最佳n_estimators对应的模型性能
N_ESTIMATORS_GRID = [ 200, 400, 600, 800]

# # smallTest_OK
# FIXED_PARAM_GRID = {
#     'learning_rate': [ 0.1],
#     'max_depth': [4],
#     'subsample': [0.8 ],
#     'colsample_bytree': [0.8],
# }
# # n_estimators search space (only this parameter is optimized)
# N_ESTIMATORS_GRID = [50]

# XGBoost-specific fixed parameters (with defaults exposed)
XGBOOST_FIXED_PARAMS = {
    'max_bin': 255,
    'min_child_weight': 10,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
    'gamma': 0.0,
    'colsample_bylevel': 1.0,#default 1.0, not commonly tuned
    'colsample_bynode': 1.0,#default 1.0, not commonly tuned
    'tree_method': 'hist',
    'device': 'cpu',
}

# MetaGB-specific fixed parameters (with defaults exposed)
METAGB_FIXED_PARAMS = {
    'max_bin': 255,
    'min_data_in_leaf': 10,
    'version': 3,
}


def load_dataset(csv_path):
    """Load dataset with categorical handling but NO normalization."""
    df = pd.read_csv(csv_path)

    if CATEGORICAL_HANDLING == 'drop':
        df = df.select_dtypes(include=[np.number]).dropna()
    elif CATEGORICAL_HANDLING == 'onehot':
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df = df.dropna()

    if len(df) == 0:
        raise ValueError(f"No valid data in {csv_path}")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y


def train_and_evaluate_xgboost(params, X_train, y_train, X_val, y_val):
    """Train XGBoost with given params and find best n_estimators."""
    import xgboost as xgb

    best_val_rmse = float('inf')
    best_n_est = N_ESTIMATORS_GRID[0]

    for n_est in N_ESTIMATORS_GRID:
        xgb_params = params.copy()
        xgb_params.update(XGBOOST_FIXED_PARAMS)
        xgb_params['n_estimators'] = n_est

        model = xgb.XGBRegressor(**xgb_params, random_state=RANDOM_SEED)
        model.fit(X_train, y_train, verbose=False)

        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_n_est = n_est

    # Retrain with best n_estimators
    xgb_params = params.copy()
    xgb_params.update(XGBOOST_FIXED_PARAMS)
    xgb_params['n_estimators'] = best_n_est

    model = xgb.XGBRegressor(**xgb_params, random_state=RANDOM_SEED)
    model.fit(X_train, y_train, verbose=False)

    return model, best_n_est, best_val_rmse


def train_and_evaluate_metagb(params, pred_frac, X_train, y_train, X_val, y_val):
    """Train MetaGB with given params and pred_feature_fraction, find best n_estimators."""
    from metagb_utils import MetaGB

    best_val_rmse = float('inf')
    best_n_est = N_ESTIMATORS_GRID[0]

    for n_est in N_ESTIMATORS_GRID:
        metagb_params = {
            'learning_rate': params['learning_rate'],
            'num_trees': n_est,
            'feature_fraction': params['colsample_bytree'],
            'bagging_fraction': params['subsample'],
            'max_depth': params['max_depth'],
            'pred_feature_fraction': pred_frac,
            'seed': RANDOM_SEED
        }
        metagb_params.update(METAGB_FIXED_PARAMS)

        model = MetaGB(**metagb_params)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_n_est = n_est

    # Retrain with best n_estimators
    metagb_params = {
        'learning_rate': params['learning_rate'],
        'num_trees': best_n_est,
        'feature_fraction': params['colsample_bytree'],
        'bagging_fraction': params['subsample'],
        'max_depth': params['max_depth'],
        'pred_feature_fraction': pred_frac,
        'seed': RANDOM_SEED
    }
    metagb_params.update(METAGB_FIXED_PARAMS)

    model = MetaGB(**metagb_params)
    model.fit(X_train, y_train)

    return model, best_n_est, best_val_rmse


def _evaluate_single_config(param_vals, param_names, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate one hyperparameter configuration with XGBoost and all pred_fractions."""
    params = dict(zip(param_names, param_vals))
    results = []

    # Train XGBoost
    xgb_model, xgb_n_est, xgb_val_rmse = train_and_evaluate_xgboost(
        params, X_train, y_train, X_val, y_val
    )
    y_test_pred = xgb_model.predict(X_test)
    xgb_metrics = calculate_metrics(y_test, y_test_pred)

    results.append({
        'model_type': 'XGBoost',
        'pred_feature_fraction': None,
        'params': params.copy(),
        'best_n_estimators': xgb_n_est,
        'val_rmse': xgb_val_rmse,
        'test_rmse': xgb_metrics['rmse'],
        'test_mae': xgb_metrics['mae'],
        'test_r2': xgb_metrics['r2']
    })

    # Train MetaGB with different pred_feature_fractions
    for pred_frac in PRED_FRACTIONS:
        mgb_model, mgb_n_est, mgb_val_rmse = train_and_evaluate_metagb(
            params, pred_frac, X_train, y_train, X_val, y_val
        )
        y_test_pred = mgb_model.predict(X_test)
        mgb_metrics = calculate_metrics(y_test, y_test_pred)

        results.append({
            'model_type': 'MetaGB',
            'pred_feature_fraction': pred_frac,
            'params': params.copy(),
            'best_n_estimators': mgb_n_est,
            'val_rmse': mgb_val_rmse,
            'test_rmse': mgb_metrics['rmse'],
            'test_mae': mgb_metrics['mae'],
            'test_r2': mgb_metrics['r2']
        })

    return results


def run_pred_fraction_study(dataset_name, csv_path, logger, results_dir):
    """Run pred_feature_fraction study for a single dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"{'='*80}")

    # Load data
    X, y = load_dataset(csv_path)
    logger.info(f"Total samples: {len(X)}, Features: {X.shape[1]}")

    # Split: 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_SEED
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Generate all parameter combinations
    param_names = list(FIXED_PARAM_GRID.keys())
    param_values = list(FIXED_PARAM_GRID.values())
    param_combinations = list(product(*param_values))

    logger.info(f"Testing {len(param_combinations)} hyperparameter combinations")
    logger.info(f"Each with 1 XGBoost + {len(PRED_FRACTIONS)} MetaGB variants")

    # Parallel execution
    all_results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_evaluate_single_config)(
            param_vals, param_names, X_train, y_train, X_val, y_val, X_test, y_test
        )
        for param_vals in param_combinations
    )

    # Flatten results
    flat_results = []
    for config_results in all_results:
        flat_results.extend(config_results)

    # Save results
    df = pd.DataFrame(flat_results)
    csv_path = os.path.join(results_dir, f'{dataset_name}_pred_fraction_study.csv')
    df.to_csv(csv_path, index=False)

    logger.info(f"\nResults saved to: {csv_path}")
    logger.info(f"Total experiments: {len(flat_results)}")

    # Summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("Summary Statistics")
    logger.info(f"{'='*80}")

    xgb_results = df[df['model_type'] == 'XGBoost']
    logger.info(f"\nXGBoost: Mean Test RMSE = {xgb_results['test_rmse'].mean():.4f}")

    for pred_frac in PRED_FRACTIONS:
        mgb_results = df[(df['model_type'] == 'MetaGB') & (df['pred_feature_fraction'] == pred_frac)]
        logger.info(f"MetaGB (pred={pred_frac:.1f}): Mean Test RMSE = {mgb_results['test_rmse'].mean():.4f}")

    return df


def main(dataset_filter=None):
    """Main function to run pred_feature_fraction study."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results', 'pred_fraction_study')
    logs_dir = os.path.join(base_dir, 'logs', 'pred_fraction_study')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    logger = ExperimentLogger('pred_fraction_study', log_dir=logs_dir)

    logger.info("="*80)
    logger.info("Experiment: pred_feature_fraction Impact Study")
    logger.info("="*80)
    logger.info(f"Fixed params (except n_estimators): {FIXED_PARAM_GRID}")
    logger.info(f"n_estimators search space: {N_ESTIMATORS_GRID}")
    logger.info(f"pred_feature_fraction values: {PRED_FRACTIONS}")

    # Get datasets
    data_dir = os.path.join(base_dir, '..', 'dataset', 'tabArena_regression')
    datasets = dataset_filter if dataset_filter else EXP1_DATASETS

    for ds in datasets:
        name = os.path.splitext(ds)[0]
        path = os.path.join(data_dir, ds)

        if not os.path.exists(path):
            logger.warning(f"Dataset not found: {path}")
            continue

        try:
            run_pred_fraction_study(name, path, logger, results_dir)
        except Exception as e:
            logger.error(f"Failed on {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("\nExperiment completed!")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Study pred_feature_fraction impact')
    parser.add_argument('--dataset', type=str, nargs='*', help='Specific datasets')

    args = parser.parse_args()

    main(dataset_filter=args.dataset)
