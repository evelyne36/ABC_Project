"""
Experiment 2A: Synthetic Nonlinear Robustness Test
MetaGB vs XGBoost on Synthetic Data with Various Nonlinear Patterns

This experiment tests model robustness under different:
- Nonlinear patterns: square, sin, square+sin, sigmoid
- Noise levels: 0, 0.1, 0.3, 0.5, 0.6, 1.0, 10.0

Key features:
- Hyperparameter grid search for both XGBoost and MetaGB
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
    RANDOM_SEED, TRAIN_TEST_SPLIT, NOISE_LEVELS, REPEATING_NUMS,
    get_exp2a_xgb_param_grid, get_exp2a_metagb_param_grid,
    EARLY_STOPPING_ROUNDS, EARLY_STOPPING_MIN_DELTA
)
from utils.data_generators import (
    make_regression_square_noise,
    make_regression_sin_noise,
    make_regression_square_sin_noise,
    make_regression_sigmoid_noise,
    get_nonlinear_generators
)
from utils.trainer import (
    train_single_model, evaluate_model, apply_zscore_normalization,
    TargetScaler, calculate_metrics
)
from utils.logger import ExperimentLogger
from utils.visualizer import (
    plot_training_curve,
    plot_noise_robustness,
    plot_training_curves_comparison
)


def save_experiment_log(
    log_path: str,
    exp_name: str,
    nonlinear_type: str,
    noise_std: float,
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
    val_losses: list
):
    """
    Save detailed experiment log for a single run.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Experiment 2A: {model_name} - {nonlinear_type} (noise={noise_std})\n")
        f.write("=" * 80 + "\n\n")

        # Basic info
        f.write(f"[{timestamp}] Experiment Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"Nonlinear Type: {nonlinear_type}\n")
        f.write(f"Noise Level: {noise_std}\n")
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
        f.write(f"Type: {data_metadata.get('type', 'N/A')}\n\n")

        # Z-Score normalization info
        f.write(f"[{timestamp}] Z-Score Normalization\n")
        f.write("-" * 40 + "\n")
        f.write("Feature scaling applied: Z-Score (StandardScaler)\n")
        if scaler_params:
            f.write(f"Feature means: {scaler_params.get('mean', 'N/A')}\n")
            f.write(f"Feature stds: {scaler_params.get('std', 'N/A')}\n")
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
        'experiment': 'exp2A',
        'nonlinear_type': nonlinear_type,
        'noise_std': noise_std,
        'model_name': model_name,
        'model_type': model_type,
        'seed': seed,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'n_features': n_features,
        'scaler_params': scaler_params,
        'data_metadata': data_metadata,
        'params': {k: str(v) for k, v in params.items()},
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
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

    return {
        'model_name': model_name,
        'model_type': model_type,
        'params': params,
        'train_time': train_result['train_time'],
        'train_losses': train_result['train_losses'],
        'val_losses': train_result['val_losses'],
        'train_metrics': train_result['train_metrics'],
        'val_metrics': train_result['val_metrics'],
        'test_metrics': test_metrics,
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


def run_experiment_for_nonlinear_type(
    nonlinear_type: str,
    generator_func,
    noise_levels: list,
    base_seed: int,
    repeat_idx: int,
    logger: ExperimentLogger,
    logs_dir: str,
    plots_dir: str,
    data_dir: str
) -> list:
    """
    Run experiments for a single nonlinear type across all noise levels.
    Each repetition generates a fresh dataset.
    """
    # Use different seed for each repetition to generate different datasets
    seed = base_seed + repeat_idx * 1000

    logger.info(f"\n{'='*80}")
    logger.info(f"Nonlinear Type: {nonlinear_type} | Repetition: {repeat_idx + 1}/{REPEATING_NUMS}")
    logger.info(f"Dataset seed: {seed}")
    logger.info(f"{'='*80}")

    # Get parameter grids
    xgb_param_grid = get_exp2a_xgb_param_grid()
    metagb_param_grid = get_exp2a_metagb_param_grid()

    all_results = []
    results_by_noise = {}

    for noise_std in noise_levels:
        logger.info(f"\n--- Noise Level: {noise_std} ---")

        # Generate fresh data for this repetition
        X, y, metadata = generator_func(
            n_samples=4000,
            n_features=10,
            noise_std=noise_std,
            random_state=seed
        )

        # Save generated dataset
        dataset_filename = f"{nonlinear_type}_noise{noise_std}_rep{repeat_idx}_seed{seed}.npz"
        dataset_path = os.path.join(data_dir, dataset_filename)
        np.savez(dataset_path, X=X, y=y, metadata=str(metadata))
        logger.info(f"  Dataset saved: {dataset_filename}")

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

        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info("  Z-Score normalization applied (X and y)")

        noise_results = {}
        training_curves = {}

        # =====================================================================
        # XGBoost Grid Search
        # =====================================================================
        logger.info(f"  Training XGBoost (grid search)...")
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

            result_entry = {
                'Nonlinear_Type': nonlinear_type,
                'Noise_Std': noise_std,
                'Repetition': repeat_idx + 1,
                'Model': 'XGBoost',
                'Best_LR': xgb_result['best_params'].get('learning_rate', '-'),
                'Best_Depth': xgb_result['best_params'].get('max_depth', '-'),
                'Best_N_Est': xgb_result['best_params'].get('n_estimators', '-'),
                'Test_RMSE': xgb_result['test_metrics']['rmse'],
                'Test_MAE': xgb_result['test_metrics']['mae'],
                'Test_R2': xgb_result['test_metrics']['r2'],
                'Val_RMSE': xgb_result['val_metrics']['rmse'],
                'Train_Time_s': xgb_result['train_time']
            }
            all_results.append(result_entry)

            noise_results['XGBoost'] = xgb_result['test_metrics']['rmse']
            training_curves['XGBoost'] = (xgb_result['train_losses'], xgb_result['val_losses'])

            logger.info(f"    XGBoost - Test RMSE: {xgb_result['test_metrics']['rmse']:.4f}, "
                       f"Time: {xgb_result['train_time']:.2f}s")

            # Plot training curve
            if xgb_result['train_losses']:
                plot_training_curve(
                    train_losses=xgb_result['train_losses'],
                    val_losses=xgb_result['val_losses'],
                    exp_type='exp2A',
                    dataset_name=f"{nonlinear_type}_noise{noise_std}_rep{repeat_idx}",
                    seed=seed,
                    model_name='XGBoost',
                    n_iter=xgb_result['best_params'].get('n_estimators', 100),
                    plot_dir=plots_dir,
                    test_rmse=xgb_result['test_metrics']['rmse'],
                    use_rmse=True
                )

            # Save log
            log_filename = f"exp2A-{nonlinear_type}-noise{noise_std}-rep{repeat_idx}-XGBoost.log"
            log_path = os.path.join(logs_dir, log_filename)
            save_experiment_log(
                log_path=log_path,
                exp_name='exp2A',
                nonlinear_type=nonlinear_type,
                noise_std=noise_std,
                seed=seed,
                model_name='XGBoost',
                model_type='xgboost',
                params=xgb_result['best_params'],
                n_train=len(X_train),
                n_val=len(X_val),
                n_test=len(X_test),
                n_features=X_train.shape[1],
                scaler_params=scaler_params,
                data_metadata=metadata,
                train_metrics=xgb_result['train_metrics'],
                val_metrics=xgb_result['val_metrics'],
                test_metrics=xgb_result['test_metrics'],
                train_time=xgb_result['train_time'],
                train_losses=xgb_result['train_losses'],
                val_losses=xgb_result['val_losses']
            )

        except Exception as e:
            logger.error(f"    XGBoost failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # =====================================================================
        # MetaGB Grid Search
        # =====================================================================
        logger.info(f"  Training MetaGB (grid search)...")
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

            result_entry = {
                'Nonlinear_Type': nonlinear_type,
                'Noise_Std': noise_std,
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
                'Train_Time_s': metagb_result['train_time']
            }
            all_results.append(result_entry)

            noise_results[model_name] = metagb_result['test_metrics']['rmse']
            training_curves[model_name] = (metagb_result['train_losses'], metagb_result['val_losses'])

            logger.info(f"    {model_name} - Test RMSE: {metagb_result['test_metrics']['rmse']:.4f}, "
                       f"Time: {metagb_result['train_time']:.2f}s")

            # Plot training curve
            if metagb_result['train_losses']:
                plot_training_curve(
                    train_losses=metagb_result['train_losses'],
                    val_losses=metagb_result['val_losses'],
                    exp_type='exp2A',
                    dataset_name=f"{nonlinear_type}_noise{noise_std}_rep{repeat_idx}",
                    seed=seed,
                    model_name=model_name,
                    n_iter=metagb_result['best_params'].get('n_estimators', 100),
                    plot_dir=plots_dir,
                    test_rmse=metagb_result['test_metrics']['rmse'],
                    use_rmse=True
                )

            # Save log
            log_filename = f"exp2A-{nonlinear_type}-noise{noise_std}-rep{repeat_idx}-MetaGB.log"
            log_path = os.path.join(logs_dir, log_filename)
            save_experiment_log(
                log_path=log_path,
                exp_name='exp2A',
                nonlinear_type=nonlinear_type,
                noise_std=noise_std,
                seed=seed,
                model_name=model_name,
                model_type='metagb',
                params=metagb_result['best_params'],
                n_train=len(X_train),
                n_val=len(X_val),
                n_test=len(X_test),
                n_features=X_train.shape[1],
                scaler_params=scaler_params,
                data_metadata=metadata,
                train_metrics=metagb_result['train_metrics'],
                val_metrics=metagb_result['val_metrics'],
                test_metrics=metagb_result['test_metrics'],
                train_time=metagb_result['train_time'],
                train_losses=metagb_result['train_losses'],
                val_losses=metagb_result['val_losses']
            )

        except Exception as e:
            logger.error(f"    MetaGB failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Store results for robustness curve
        results_by_noise[noise_std] = noise_results

        # Plot training curves comparison for this noise level
        if training_curves:
            plot_training_curves_comparison(
                results_by_model=training_curves,
                nonlinear_type=nonlinear_type,
                noise_std=noise_std,
                n_iter=200,
                plot_dir=plots_dir
            )

    return all_results


def run_experiment2a(seed: int = RANDOM_SEED):
    """
    Run Experiment 2A: Synthetic nonlinear robustness test.
    Runs REPEATING_NUMS times with fresh datasets each time.
    """
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results', 'exp2A')
    logs_dir = os.path.join(base_dir, 'logs', 'exp2A')
    plots_dir = os.path.join(base_dir, 'plots', 'exp2A')
    data_dir = os.path.join(base_dir, 'data', 'exp2A')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Setup logger
    logger = ExperimentLogger('exp2A', log_dir=logs_dir)

    logger.info("=" * 80)
    logger.info("Experiment 2A: Synthetic Nonlinear Robustness Test")
    logger.info("MetaGB vs XGBoost on Synthetic Data (with Grid Search)")
    logger.info("=" * 80)
    logger.info(f"Base random seed: {seed}")
    logger.info(f"Number of repetitions: {REPEATING_NUMS}")
    logger.info(f"Noise levels: {NOISE_LEVELS}")
    logger.info("Feature normalization: Z-Score (StandardScaler)")

    # Get nonlinear generators
    generators = get_nonlinear_generators()
    logger.info(f"Nonlinear types: {list(generators.keys())}")

    # Print grid search info
    xgb_grid = get_exp2a_xgb_param_grid()
    metagb_grid = get_exp2a_metagb_param_grid()
    n_xgb_combinations = int(np.prod([len(v) for v in xgb_grid.values()]))
    n_metagb_combinations = int(np.prod([len(v) for v in metagb_grid.values()]))
    logger.info(f"XGBoost grid search: {n_xgb_combinations} combinations")
    logger.info(f"MetaGB grid search: {n_metagb_combinations} combinations")

    # Calculate total experiments
    n_total = len(generators) * len(NOISE_LEVELS) * REPEATING_NUMS * 2  # 2 models
    logger.info(f"Total experiments: {n_total}")

    # Run experiments
    all_results = []

    for repeat_idx in range(REPEATING_NUMS):
        logger.info(f"\n{'#'*80}")
        logger.info(f"# REPETITION {repeat_idx + 1}/{REPEATING_NUMS}")
        logger.info(f"{'#'*80}")

        for nonlinear_type, generator_func in generators.items():
            try:
                results = run_experiment_for_nonlinear_type(
                    nonlinear_type=nonlinear_type,
                    generator_func=generator_func,
                    noise_levels=NOISE_LEVELS,
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
                    df.to_csv(os.path.join(results_dir, 'exp2A_summary_partial.csv'), index=False)

            except Exception as e:
                logger.error(f"Failed for {nonlinear_type} rep {repeat_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # Save final summary
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(results_dir, 'exp2A_summary.csv')
        df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to: {summary_path}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT 2A SUMMARY")
        logger.info("=" * 80)

        # Aggregate by model across all repetitions
        logger.info("\n--- Average Test RMSE by Model (across all repetitions) ---")
        for model in df['Model'].unique():
            model_df = df[df['Model'] == model]
            logger.info(f"{model}: {model_df['Test_RMSE'].mean():.4f} "
                       f"(+/- {model_df['Test_RMSE'].std():.4f})")

        # Aggregate by nonlinear type
        logger.info("\n--- Average Test RMSE by Nonlinear Type ---")
        for nl_type in df['Nonlinear_Type'].unique():
            type_df = df[df['Nonlinear_Type'] == nl_type]
            logger.info(f"{nl_type}: {type_df['Test_RMSE'].mean():.4f}")

        # Summary statistics per repetition
        logger.info("\n--- Results by Repetition ---")
        for rep in df['Repetition'].unique():
            rep_df = df[df['Repetition'] == rep]
            logger.info(f"Repetition {rep}: Avg Test RMSE = {rep_df['Test_RMSE'].mean():.4f}")

        # Best model per nonlinear type (averaged across repetitions)
        logger.info("\n--- Best Model per Nonlinear Type (averaged) ---")
        for nl_type in df['Nonlinear_Type'].unique():
            type_df = df[df['Nonlinear_Type'] == nl_type]
            avg_by_model = type_df.groupby('Model')['Test_RMSE'].mean()
            best_model = avg_by_model.idxmin()
            best_rmse = avg_by_model.min()
            logger.info(f"{nl_type}: {best_model} (Avg RMSE={best_rmse:.4f})")

    logger.info("\nExperiment 2A completed!")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 2A: Synthetic Nonlinear Robustness Test')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs for grid search. -1 uses all CPUs (default: 1)')

    args = parser.parse_args()

    # Set global n_jobs for parallel processing
    from utils.trainer import set_n_jobs
    set_n_jobs(args.n_jobs)

    run_experiment2a(seed=args.seed)
