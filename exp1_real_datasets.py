"""
Experiment 1: Real Datasets Performance Comparison
MetaGB vs XGBoost on TabArena Regression Datasets

This script performs grid search hyperparameter optimization and evaluates
both models on multiple real-world regression datasets.

Workflow for each dataset:
1. Data split: 80% train / 20% test (fixed seed)
2. Z-Score normalization on features (fit on train, transform all)
3. Hyperparameter optimization: GridSearchCV (K=5) on train set
4. Final evaluation: Train with best params, evaluate on test set
5. Save results: logs, plots, and summary CSV
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ABC_Project', 'metagb'))

from utils.hyperparams import (
    RANDOM_SEED, TRAIN_TEST_SPLIT, CV_FOLDS,
    get_xgb_param_grid, get_metagb_param_grid
)
from utils.trainer import (
    train_with_grid_search, train_single_model,
    evaluate_model, calculate_metrics, apply_zscore_normalization,
    TargetScaler
)
from utils.logger import ExperimentLogger
from utils.visualizer import plot_training_curve


def get_tabarena_datasets():
    """
    Get list of TabArena regression dataset paths.

    Returns:
        List of (dataset_name, csv_path) tuples
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'tabArena_regression')

    datasets = [
        "airfoil_self_noise.csv",#ok
        "Another-Dataset-on-used-Fiat-500.csv",#ok
        "concrete_compressive_strength.csv", #ok
        # "diamonds.csv", #训练过慢
        # "Food_Delivery_Time.csv", #训练过慢
        "healthcare_insurance_expenses.csv",#feature_fraction is too small, randomly select 1 feature
        #"houses.csv",#Slow
        #"miami_housing.csv",#Slow
        #"physiochemical_protein.csv",#Slow
        "QSAR_fish_toxicity.csv",#OK
        # "QSAR-TID-11.csv",#训练过慢，Feature1024个
        # "superconductivity.csv",#训练过慢，Feature81个
        "wine_quality.csv"#OK
    ]

    result = []
    for ds in datasets:
        name = os.path.splitext(ds)[0]
        path = os.path.join(data_dir, ds)
        if os.path.exists(path):
            result.append((name, path))
        else:
            print(f"Warning: Dataset not found: {path}")

    return result


def load_dataset(csv_path, test_size=0.2, random_state=42):
    """
    Load a CSV dataset and split into train/test.

    Args:
        csv_path: Path to CSV file
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)

    # Keep only numeric columns and drop NaN
    df = df.select_dtypes(include=[np.number]).dropna()

    if len(df) == 0:
        raise ValueError(f"No valid numeric data in {csv_path}")

    # Last column is target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def save_detailed_experiment_log(
    log_path: str,
    dataset_name: str,
    seed: int,
    model_name: str,
    n_train: int,
    n_test: int,
    n_features: int,
    scaler_params: dict,
    param_grid: dict,
    n_combinations: int,
    cv_folds: int,
    best_params: dict,
    best_cv_score: float,
    cv_results: list,
    test_metrics: dict,
    train_time: float
):
    """
    Save detailed experiment log to file.

    Args:
        log_path: Path to save log file
        dataset_name: Name of the dataset
        seed: Random seed
        model_name: Name of the model
        n_train: Number of training samples
        n_test: Number of test samples
        n_features: Number of features
        scaler_params: Z-Score scaler parameters
        param_grid: Parameter grid used
        n_combinations: Number of parameter combinations
        cv_folds: Number of CV folds
        best_params: Best parameters found
        best_cv_score: Best CV score
        cv_results: All CV results
        test_metrics: Test set metrics
        train_time: Total training time
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Experiment 1: {model_name} on {dataset_name}\n")
        f.write("=" * 80 + "\n\n")

        # Basic info
        f.write(f"[{timestamp}] Experiment Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Train Samples: {n_train}\n")
        f.write(f"Test Samples: {n_test}\n")
        f.write(f"Features: {n_features}\n\n")

        # Z-Score normalization info
        f.write(f"[{timestamp}] Z-Score Normalization\n")
        f.write("-" * 40 + "\n")
        f.write("Feature scaling applied: Z-Score (StandardScaler)\n")
        if scaler_params:
            f.write(f"Feature means: {scaler_params.get('mean', 'N/A')}\n")
            f.write(f"Feature stds: {scaler_params.get('std', 'N/A')}\n")
        f.write("\n")

        # Grid search info
        f.write(f"[{timestamp}] Grid Search Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"CV Folds: {cv_folds}\n")
        f.write(f"Total Combinations: {n_combinations}\n")
        f.write("Parameter Grid:\n")
        for k, v in param_grid.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        # Best parameters
        f.write(f"[{timestamp}] Best Parameters Found\n")
        f.write("-" * 40 + "\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nBest CV RMSE: {best_cv_score:.6f}\n\n")

        # Test results
        f.write(f"[{timestamp}] Final Test Results\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.6f}\n")
        f.write(f"Test R²: {test_metrics['r2']:.6f}\n")
        f.write(f"Test MSE: {test_metrics['mse']:.6f}\n\n")

        # Training time
        f.write(f"[{timestamp}] Training Time\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Time: {train_time:.2f} seconds\n\n")

        # Top 10 CV results
        f.write(f"[{timestamp}] Top 10 CV Results\n")
        f.write("-" * 40 + "\n")
        sorted_cv = sorted(cv_results, key=lambda x: x['mean_cv_rmse'])[:10]
        for i, result in enumerate(sorted_cv):
            f.write(f"\n#{i+1} CV RMSE: {result['mean_cv_rmse']:.6f} (+/- {result['std_cv_rmse']:.6f})\n")
            f.write(f"   Params: {result['params']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Experiment completed successfully\n")
        f.write("=" * 80 + "\n")

    # Also save as JSON for programmatic access
    json_path = log_path.replace('.log', '.json')
    json_data = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'n_train': n_train,
        'n_test': n_test,
        'n_features': n_features,
        'scaler_params': scaler_params,
        'param_grid': {k: [str(v) for v in vals] for k, vals in param_grid.items()},
        'n_combinations': n_combinations,
        'cv_folds': cv_folds,
        'best_params': {k: str(v) for k, v in best_params.items()},
        'best_cv_score': best_cv_score,
        'test_metrics': test_metrics,
        'train_time': train_time
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def run_experiment_for_dataset(
    dataset_name: str,
    csv_path: str,
    seed: int,
    logger: ExperimentLogger,
    results_dir: str,
    logs_dir: str,
    plots_dir: str
):
    """
    Run full experiment for a single dataset.

    Args:
        dataset_name: Name of the dataset
        csv_path: Path to CSV file
        seed: Random seed
        logger: Logger instance
        results_dir: Directory for results
        logs_dir: Directory for logs
        plots_dir: Directory for plots

    Returns:
        List of result dictionaries (one per model)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*80}")

    # Load data
    try:
        X_train, X_test, y_train, y_test = load_dataset(
            csv_path, test_size=1-TRAIN_TEST_SPLIT, random_state=seed
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return []

    n_features = X_train.shape[1]
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Features: {n_features}")

    # Apply Z-Score normalization
    logger.info("Applying Z-Score normalization...")
    X_train_scaled, _, X_test_scaled, scaler = apply_zscore_normalization(
        X_train, X_val=None, X_test=X_test
    )
    scaler_params = scaler.get_params()
    logger.info("Z-Score normalization applied (fit on train, transform all)")

    # Apply target (y) normalization
    logger.info("Applying target (y) normalization...")
    y_scaler = TargetScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    y_scaler_params = y_scaler.get_params()
    logger.info(f"Target normalization: mean={y_scaler_params['mean']:.4f}, std={y_scaler_params['std']:.4f}")

    results = []

    # =========================================================================
    # XGBoost with Grid Search
    # =========================================================================
    logger.info(f"\n--- XGBoost Grid Search ---")

    xgb_param_grid = get_xgb_param_grid()
    n_xgb_combinations = int(np.prod([len(v) for v in xgb_param_grid.values()]))
    logger.info(f"Parameter combinations: {n_xgb_combinations}")

    try:
        start_time = time.time()

        best_xgb_model, best_xgb_params, xgb_cv_results = train_with_grid_search(
            model_type='xgboost',
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            param_grid=xgb_param_grid,
            cv=CV_FOLDS,
            random_state=seed,
            logger=logger
        )

        xgb_train_time = time.time() - start_time

        # Evaluate on test set (in normalized scale)
        xgb_pred_scaled = best_xgb_model.predict(X_test_scaled)
        xgb_test_metrics = calculate_metrics(y_test_scaled, xgb_pred_scaled)

        # Get best CV score
        best_cv_idx = np.argmin([r['mean_cv_rmse'] for r in xgb_cv_results])
        best_cv_rmse = xgb_cv_results[best_cv_idx]['mean_cv_rmse']

        xgb_result = {
            'Dataset': dataset_name,
            'Model': 'XGBoost',
            'Best_LR': best_xgb_params.get('learning_rate', '-'),
            'Best_Depth': best_xgb_params.get('max_depth', '-'),
            'Best_N_Est': best_xgb_params.get('n_estimators', '-'),
            'Best_Pred_Frac': '-',
            'CV_RMSE': best_cv_rmse,
            'Test_RMSE': xgb_test_metrics['rmse'],
            'Test_MAE': xgb_test_metrics['mae'],
            'Test_R2': xgb_test_metrics['r2'],
            'Train_Time_s': xgb_train_time
        }
        results.append(xgb_result)

        logger.info(f"Best params: {best_xgb_params}")
        logger.info(f"Best CV RMSE: {best_cv_rmse:.4f}")
        logger.info(f"Test RMSE: {xgb_test_metrics['rmse']:.4f}")
        logger.info(f"Test MAE: {xgb_test_metrics['mae']:.4f}")
        logger.info(f"Test R²: {xgb_test_metrics['r2']:.4f}")
        logger.info(f"Total time: {xgb_train_time:.1f}s")

        # Save detailed log
        log_path = os.path.join(logs_dir, f"exp1-{dataset_name}-seed{seed}-XGBoost.log")
        save_detailed_experiment_log(
            log_path=log_path,
            dataset_name=dataset_name,
            seed=seed,
            model_name='XGBoost',
            n_train=len(X_train),
            n_test=len(X_test),
            n_features=n_features,
            scaler_params=scaler_params,
            param_grid=xgb_param_grid,
            n_combinations=n_xgb_combinations,
            cv_folds=CV_FOLDS,
            best_params=best_xgb_params,
            best_cv_score=best_cv_rmse,
            cv_results=xgb_cv_results,
            test_metrics=xgb_test_metrics,
            train_time=xgb_train_time
        )

    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # =========================================================================
    # MetaGB with Grid Search
    # =========================================================================
    logger.info(f"\n--- MetaGB Grid Search ---")

    metagb_param_grid = get_metagb_param_grid()
    n_metagb_combinations = int(np.prod([len(v) for v in metagb_param_grid.values()]))
    logger.info(f"Parameter combinations: {n_metagb_combinations}")

    try:
        start_time = time.time()

        best_metagb_model, best_metagb_params, metagb_cv_results = train_with_grid_search(
            model_type='metagb',
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            param_grid=metagb_param_grid,
            cv=CV_FOLDS,
            random_state=seed,
            logger=logger
        )

        metagb_train_time = time.time() - start_time

        # Evaluate on test set (in normalized scale)
        metagb_pred_scaled = best_metagb_model.predict(X_test_scaled.astype(np.float64))
        metagb_test_metrics = calculate_metrics(y_test_scaled, metagb_pred_scaled)

        # Get best CV score
        best_cv_idx = np.argmin([r['mean_cv_rmse'] for r in metagb_cv_results])
        best_cv_rmse = metagb_cv_results[best_cv_idx]['mean_cv_rmse']

        metagb_result = {
            'Dataset': dataset_name,
            'Model': 'MetaGB',
            'Best_LR': best_metagb_params.get('learning_rate', '-'),
            'Best_Depth': best_metagb_params.get('max_depth', '-'),
            'Best_N_Est': best_metagb_params.get('n_estimators', '-'),
            'Best_Pred_Frac': best_metagb_params.get('pred_feature_fraction', '-'),
            'CV_RMSE': best_cv_rmse,
            'Test_RMSE': metagb_test_metrics['rmse'],
            'Test_MAE': metagb_test_metrics['mae'],
            'Test_R2': metagb_test_metrics['r2'],
            'Train_Time_s': metagb_train_time
        }
        results.append(metagb_result)

        logger.info(f"Best params: {best_metagb_params}")
        logger.info(f"Best CV RMSE: {best_cv_rmse:.4f}")
        logger.info(f"Test RMSE: {metagb_test_metrics['rmse']:.4f}")
        logger.info(f"Test MAE: {metagb_test_metrics['mae']:.4f}")
        logger.info(f"Test R²: {metagb_test_metrics['r2']:.4f}")
        logger.info(f"Total time: {metagb_train_time:.1f}s")

        # Save detailed log
        log_path = os.path.join(logs_dir, f"exp1-{dataset_name}-seed{seed}-MetaGB.log")
        save_detailed_experiment_log(
            log_path=log_path,
            dataset_name=dataset_name,
            seed=seed,
            model_name='MetaGB',
            n_train=len(X_train),
            n_test=len(X_test),
            n_features=n_features,
            scaler_params=scaler_params,
            param_grid=metagb_param_grid,
            n_combinations=n_metagb_combinations,
            cv_folds=CV_FOLDS,
            best_params=best_metagb_params,
            best_cv_score=best_cv_rmse,
            cv_results=metagb_cv_results,
            test_metrics=metagb_test_metrics,
            train_time=metagb_train_time
        )

    except Exception as e:
        logger.error(f"MetaGB training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return results


def run_experiment1(seed: int = RANDOM_SEED):
    """
    Run Experiment 1: Real datasets performance comparison.

    Args:
        seed: Random seed for reproducibility
    """
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results', 'exp1')
    logs_dir = os.path.join(base_dir, 'logs', 'exp1')
    plots_dir = os.path.join(base_dir, 'plots', 'exp1')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Setup logger
    logger = ExperimentLogger('exp1', log_dir=logs_dir)

    logger.info("=" * 80)
    logger.info("Experiment 1: Real Datasets Performance Comparison")
    logger.info("MetaGB vs XGBoost on TabArena Regression Datasets")
    logger.info("=" * 80)
    logger.info(f"Random seed: {seed}")
    logger.info(f"Train/Test split: {TRAIN_TEST_SPLIT}/{1-TRAIN_TEST_SPLIT}")
    logger.info(f"CV folds: {CV_FOLDS}")
    logger.info("Feature normalization: Z-Score (StandardScaler)")

    # Get datasets
    datasets = get_tabarena_datasets()
    logger.info(f"Found {len(datasets)} datasets")

    # Run experiments
    all_results = []

    for dataset_name, csv_path in datasets:
        try:
            results = run_experiment_for_dataset(
                dataset_name=dataset_name,
                csv_path=csv_path,
                seed=seed,
                logger=logger,
                results_dir=results_dir,
                logs_dir=logs_dir,
                plots_dir=plots_dir
            )
            all_results.extend(results)

            # Save intermediate results
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(results_dir, 'exp1_summary_partial.csv'), index=False)

        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Save final summary
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(results_dir, 'exp1_summary.csv')
        df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to: {summary_path}")

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT 1 SUMMARY")
        logger.info("=" * 80)
        print(df.to_string(index=False))

        # Calculate aggregate statistics
        logger.info("\n--- Aggregate Statistics ---")
        for model in ['XGBoost', 'MetaGB']:
            model_df = df[df['Model'] == model]
            if len(model_df) > 0:
                logger.info(f"\n{model}:")
                logger.info(f"  Avg Test RMSE: {model_df['Test_RMSE'].mean():.4f}")
                logger.info(f"  Avg Test MAE: {model_df['Test_MAE'].mean():.4f}")
                logger.info(f"  Avg Test R²: {model_df['Test_R2'].mean():.4f}")
                logger.info(f"  Avg Train Time: {model_df['Train_Time_s'].mean():.1f}s")

    logger.info("\nExperiment 1 completed!")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 1: Real Datasets Comparison')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')

    args = parser.parse_args()

    run_experiment1(seed=args.seed)
