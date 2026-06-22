"""
Comparison of MetaGB and XGBoost with Grid Search on Diabetes Dataset.

This script performs grid search over hyperparameters for both models,
evaluating train, validation, and test MSE for each parameter combination.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from metagb_utils import MetaGB
from itertools import product
import time


def evaluate_model(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred):
    """Calculate MSE for train, validation, and test sets."""
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    return train_mse, val_mse, test_mse


def grid_search_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, param_grid):
    """
    Perform grid search for XGBoost and return results for all parameter combinations.
    """
    results = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total_combinations = np.prod([len(v) for v in param_values])
    print(f"\nXGBoost Grid Search: {total_combinations} combinations")
    print("=" * 80)

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

        train_mse, val_mse, test_mse = evaluate_model(
            y_train, train_pred, y_val, val_pred, y_test, test_pred
        )

        result = {
            'params': param_dict,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_time': train_time
        }
        results.append(result)

        print(f"[{i+1}/{total_combinations}] {param_dict}")
        print(f"    Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Test MSE: {test_mse:.4f} | Time: {train_time:.2f}s")

    return results


def grid_search_metagb(X_train, y_train, X_val, y_val, X_test, y_test, param_grid):
    """
    Perform grid search for MetaGB and return results for all parameter combinations.
    """
    results = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total_combinations = np.prod([len(v) for v in param_values])
    print(f"\nMetaGB Grid Search: {total_combinations} combinations")
    print("=" * 80)

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
            lambda_l2=param_dict.get('lambda_l2', 0.0),
            version=param_dict.get('version', 3),
            seed=42,
        )

        start_time = time.time()
        model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
        train_time = time.time() - start_time

        train_pred = model.predict(X_train.astype(np.float64))
        val_pred = model.predict(X_val.astype(np.float64))
        test_pred = model.predict(X_test.astype(np.float64))

        train_mse, val_mse, test_mse = evaluate_model(
            y_train, train_pred, y_val, val_pred, y_test, test_pred
        )

        result = {
            'params': param_dict,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_time': train_time
        }
        results.append(result)

        print(f"[{i+1}/{total_combinations}] {param_dict}")
        print(f"    Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Test MSE: {test_mse:.4f} | Time: {train_time:.2f}s")

    return results


def find_best_model(results, metric='val_mse'):
    """Find the best model based on the specified metric."""
    best_idx = np.argmin([r[metric] for r in results])
    return results[best_idx]


def print_summary(xgb_results, metagb_results, metagb_v0_results):
    """Print summary comparison of best models."""
    print("\n" + "=" * 80)
    print("SUMMARY: Best Models by Validation MSE")
    print("=" * 80)

    best_xgb = find_best_model(xgb_results, 'val_mse')
    best_metagb = find_best_model(metagb_results, 'val_mse')
    best_metagb_v0 = find_best_model(metagb_v0_results, 'val_mse')

    print("\nBest XGBoost:")
    print(f"  Parameters: {best_xgb['params']}")
    print(f"  Train MSE:  {best_xgb['train_mse']:.4f}")
    print(f"  Val MSE:    {best_xgb['val_mse']:.4f}")
    print(f"  Test MSE:   {best_xgb['test_mse']:.4f}")

    print("\nBest MetaGB (v3):")
    print(f"  Parameters: {best_metagb['params']}")
    print(f"  Train MSE:  {best_metagb['train_mse']:.4f}")
    print(f"  Val MSE:    {best_metagb['val_mse']:.4f}")
    print(f"  Test MSE:   {best_metagb['test_mse']:.4f}")

    print("\nBest MetaGB (v0):")
    print(f"  Parameters: {best_metagb_v0['params']}")
    print(f"  Train MSE:  {best_metagb_v0['train_mse']:.4f}")
    print(f"  Val MSE:    {best_metagb_v0['val_mse']:.4f}")
    print(f"  Test MSE:   {best_metagb_v0['test_mse']:.4f}")

    print("\n" + "-" * 80)
    print("Comparison (Best Models):")
    print(f"  Val MSE  - XGBoost: {best_xgb['val_mse']:.4f} vs MetaGB(v3): {best_metagb['val_mse']:.4f} vs MetaGB(v0): {best_metagb_v0['val_mse']:.4f}")
    print(f"  Test MSE - XGBoost: {best_xgb['test_mse']:.4f} vs MetaGB(v3): {best_metagb['test_mse']:.4f} vs MetaGB(v0): {best_metagb_v0['test_mse']:.4f}")


def main():
    # Load the Diabetes dataset
    print("Loading Diabetes dataset...")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {diabetes.feature_names}")

    # Split data into train, validation, and test sets (60/20/20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
    )

    print(f"Train size: {X_train.shape[0]}")
    print(f"Validation size: {X_val.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")

    # XGBoost parameter grid
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

    # MetaGB parameter grid (base XGBoost params + MetaGB-specific params) - version 3
    metagb_param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        # MetaGB-specific parameters
        # 'pred_feature_fraction': [0.0, 0.1, 0.2, 0.4],  # 0.0 = bias only (like standard GB)
        'pred_feature_fraction': [0.0],  # 0.0 = bias only (like standard GB)
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
        'pred_feature_fraction': [0.0],
        'version': [0],  # Use version 0
        'max_bin': [64],
        'min_data_in_leaf': [20],
    }

    # Run grid searches
    xgb_results = grid_search_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test, xgb_param_grid
    )

    print("\n--- MetaGB Version 3 ---")
    metagb_results = grid_search_metagb(
        X_train, y_train, X_val, y_val, X_test, y_test, metagb_param_grid
    )

    print("\n--- MetaGB Version 0 ---")
    metagb_v0_results = grid_search_metagb(
        X_train, y_train, X_val, y_val, X_test, y_test, metagb_v0_param_grid
    )

    # Print summary
    print_summary(xgb_results, metagb_results, metagb_v0_results)

    return xgb_results, metagb_results, metagb_v0_results


if __name__ == "__main__":
    main()
