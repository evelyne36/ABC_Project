"""
Comparison of MetaGB and XGBoost with Cross-Validation on California Housing Dataset.

This script performs grid search over hyperparameters for both models,
using k-fold cross-validation to evaluate performance.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from metagb_utils import MetaGB
from itertools import product
import time


def cross_validate_xgboost(X, y, param_dict, n_folds=5, random_state=42):
    """
    Perform k-fold cross-validation for XGBoost.
    Returns train and validation MSE for each fold.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_mses = []
    val_mses = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            random_state=random_state,
            n_jobs=1,
            min_child_weight=20,
            reg_lambda=0,
            **param_dict
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_mses.append(mean_squared_error(y_train, train_pred))
        val_mses.append(mean_squared_error(y_val, val_pred))

    return np.mean(train_mses), np.std(train_mses), np.mean(val_mses), np.std(val_mses)


def cross_validate_metagb(X, y, param_dict, n_folds=5, random_state=42):
    """
    Perform k-fold cross-validation for MetaGB.
    Returns train and validation MSE for each fold.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_mses = []
    val_mses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = MetaGB(
            learning_rate=param_dict['learning_rate'],
            num_trees=param_dict['n_estimators'],
            feature_fraction=param_dict['colsample_bytree'],
            pred_feature_fraction=param_dict.get('pred_feature_fraction', param_dict['colsample_bytree']),
            bagging_fraction=param_dict['subsample'],
            min_data_in_leaf=param_dict.get('min_data_in_leaf', 20),
            max_depth=param_dict['max_depth'],
            max_bin=param_dict.get('max_bin', 64),
            lambda_l2=param_dict.get('lambda_l2', 0.0),
            version=param_dict.get('version', 3),
            seed=random_state + fold_idx,
        )

        model.fit(X_train.astype(np.float64), y_train.astype(np.float64))

        train_pred = model.predict(X_train.astype(np.float64))
        val_pred = model.predict(X_val.astype(np.float64))

        train_mses.append(mean_squared_error(y_train, train_pred))
        val_mses.append(mean_squared_error(y_val, val_pred))

    return np.mean(train_mses), np.std(train_mses), np.mean(val_mses), np.std(val_mses)


def grid_search_xgboost_cv(X, y, param_grid, n_folds=5):
    """
    Perform grid search for XGBoost with cross-validation.
    """
    results = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total_combinations = np.prod([len(v) for v in param_values])
    print(f"\nXGBoost Grid Search ({n_folds}-fold CV): {total_combinations} combinations")
    print("=" * 80)

    for i, params in enumerate(product(*param_values)):
        param_dict = dict(zip(param_names, params))

        start_time = time.time()
        train_mean, train_std, val_mean, val_std = cross_validate_xgboost(
            X, y, param_dict, n_folds=n_folds
        )
        cv_time = time.time() - start_time

        result = {
            'params': param_dict,
            'train_mse_mean': train_mean,
            'train_mse_std': train_std,
            'val_mse_mean': val_mean,
            'val_mse_std': val_std,
            'cv_time': cv_time
        }
        results.append(result)

        print(f"[{i+1}/{total_combinations}] {param_dict}")
        print(f"    Train MSE: {train_mean:.4f} ± {train_std:.4f} | Val MSE: {val_mean:.4f} ± {val_std:.4f} | Time: {cv_time:.2f}s")

    return results


def grid_search_metagb_cv(X, y, param_grid, n_folds=5):
    """
    Perform grid search for MetaGB with cross-validation.
    """
    results = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total_combinations = np.prod([len(v) for v in param_values])
    print(f"\nMetaGB Grid Search ({n_folds}-fold CV): {total_combinations} combinations")
    print("=" * 80)

    for i, params in enumerate(product(*param_values)):
        param_dict = dict(zip(param_names, params))

        start_time = time.time()
        train_mean, train_std, val_mean, val_std = cross_validate_metagb(
            X, y, param_dict, n_folds=n_folds
        )
        cv_time = time.time() - start_time

        result = {
            'params': param_dict,
            'train_mse_mean': train_mean,
            'train_mse_std': train_std,
            'val_mse_mean': val_mean,
            'val_mse_std': val_std,
            'cv_time': cv_time
        }
        results.append(result)

        print(f"[{i+1}/{total_combinations}] {param_dict}")
        print(f"    Train MSE: {train_mean:.4f} ± {train_std:.4f} | Val MSE: {val_mean:.4f} ± {val_std:.4f} | Time: {cv_time:.2f}s")

    return results


def find_best_model(results, metric='val_mse_mean'):
    """Find the best model based on the specified metric."""
    best_idx = np.argmin([r[metric] for r in results])
    return results[best_idx]


def print_summary(xgb_results, metagb_results, metagb_v0_results):
    """Print summary comparison of best models."""
    print("\n" + "=" * 80)
    print("SUMMARY: Best Models by Cross-Validation MSE")
    print("=" * 80)

    best_xgb = find_best_model(xgb_results, 'val_mse_mean')
    best_metagb = find_best_model(metagb_results, 'val_mse_mean')
    best_metagb_v0 = find_best_model(metagb_v0_results, 'val_mse_mean')

    print("\nBest XGBoost:")
    print(f"  Parameters: {best_xgb['params']}")
    print(f"  Train MSE:  {best_xgb['train_mse_mean']:.4f} ± {best_xgb['train_mse_std']:.4f}")
    print(f"  Val MSE:    {best_xgb['val_mse_mean']:.4f} ± {best_xgb['val_mse_std']:.4f}")

    print("\nBest MetaGB (v3):")
    print(f"  Parameters: {best_metagb['params']}")
    print(f"  Train MSE:  {best_metagb['train_mse_mean']:.4f} ± {best_metagb['train_mse_std']:.4f}")
    print(f"  Val MSE:    {best_metagb['val_mse_mean']:.4f} ± {best_metagb['val_mse_std']:.4f}")

    print("\nBest MetaGB (v0):")
    print(f"  Parameters: {best_metagb_v0['params']}")
    print(f"  Train MSE:  {best_metagb_v0['train_mse_mean']:.4f} ± {best_metagb_v0['train_mse_std']:.4f}")
    print(f"  Val MSE:    {best_metagb_v0['val_mse_mean']:.4f} ± {best_metagb_v0['val_mse_std']:.4f}")

    print("\n" + "-" * 80)
    print("Comparison (Best Models):")
    print(f"  Val MSE - XGBoost: {best_xgb['val_mse_mean']:.4f} vs MetaGB(v3): {best_metagb['val_mse_mean']:.4f} vs MetaGB(v0): {best_metagb_v0['val_mse_mean']:.4f}")


def main():
    # Load the California Housing dataset from OpenML
    print("Loading California Housing dataset from OpenML...")
    housing = fetch_openml(name='California_Housing', version=1, as_frame=True)

    # Keep only numeric columns
    df = housing.data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    if non_numeric_cols:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        df = df[numeric_cols]

    X = df.values
    X = np.nan_to_num(X, nan=0.0)
    y = housing.target.values.astype(np.float64)

    # Standardize y
    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / y_std
    print(f"Target standardized: mean={y_mean:.4f}, std={y_std:.4f}")

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(df.columns)}")

    # Number of folds for cross-validation
    n_folds = 5

    # XGBoost parameter grid
    xgb_param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'max_bin': [64],
    }

    # MetaGB parameter grid (base XGBoost params + MetaGB-specific params) - version 3
    metagb_param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        # MetaGB-specific parameters
        'pred_feature_fraction': [0.0, 0.4, 0.8],  # 0.0 = bias only (like standard GB)
        'version': [3],  # Use version 3 (simplified, recommended)
        'max_bin': [64],
        'min_data_in_leaf': [20],
    }

    # MetaGB parameter grid - version 0
    metagb_v0_param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        # MetaGB-specific parameters
        'pred_feature_fraction': [0.0, 0.4, 0.8],
        'version': [0],  # Use version 0
        'max_bin': [64],
        'min_data_in_leaf': [20],
    }

    # Run grid searches with cross-validation
    xgb_results = grid_search_xgboost_cv(X, y, xgb_param_grid, n_folds=n_folds)

    print("\n--- MetaGB Version 3 ---")
    metagb_results = grid_search_metagb_cv(X, y, metagb_param_grid, n_folds=n_folds)

    print("\n--- MetaGB Version 0 ---")
    metagb_v0_results = grid_search_metagb_cv(X, y, metagb_v0_param_grid, n_folds=n_folds)

    # Print summary
    print_summary(xgb_results, metagb_results, metagb_v0_results)

    return xgb_results, metagb_results, metagb_v0_results


if __name__ == "__main__":
    main()