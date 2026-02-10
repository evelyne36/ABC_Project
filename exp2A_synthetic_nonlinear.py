"""
Experiment 2A: Synthetic Nonlinear Robustness Test
MetaGB vs XGBoost on Synthetic Data with Various Nonlinear Patterns

This experiment tests model robustness under different:
- Nonlinear patterns: square, sin, square+sin, sigmoid
- Noise levels: 0, 0.1, 0.3, 0.5, 0.6, 1.0, 10.0

Key differences from Experiment 1:
- Fixed hyperparameters (no grid search)
- Only pred_feature_fraction varies for MetaGB
- Focus on comparing noise robustness across models
- Z-Score normalization applied to features

Models tested:
- XGBoost (baseline)
- MetaGB_v0.0 (pred_feature_fraction=0.0, like standard GB)
- MetaGB_v0.4 (pred_feature_fraction=0.4)
- MetaGB_v0.8 (pred_feature_fraction=0.8)
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
    RANDOM_SEED, TRAIN_TEST_SPLIT, NOISE_LEVELS,
    get_fixed_xgb_params, get_fixed_metagb_params, get_model_variants_exp2
)
from utils.data_generators import (
    make_regression_square_noise,
    make_regression_sin_noise,
    make_regression_square_sin_noise,
    make_regression_sigmoid_noise,
    get_nonlinear_generators
)
from utils.trainer import train_single_model, evaluate_model, apply_zscore_normalization, TargetScaler, calculate_metrics
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
        'test_metrics': test_metrics
    }


def run_experiment_for_nonlinear_type(
    nonlinear_type: str,
    generator_func,
    noise_levels: list,
    seed: int,
    logger: ExperimentLogger,
    logs_dir: str,
    plots_dir: str
) -> list:
    """
    Run experiments for a single nonlinear type across all noise levels.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Nonlinear Type: {nonlinear_type}")
    logger.info(f"{'='*80}")

    # Get model variants
    model_variants = get_model_variants_exp2()

    all_results = []
    results_by_noise = {}

    for noise_std in noise_levels:
        logger.info(f"\n--- Noise Level: {noise_std} ---")

        # Generate data
        X, y, metadata = generator_func(
            n_samples=4000,
            n_features=10,
            noise_std=noise_std,
            random_state=seed
        )

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

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info("Z-Score normalization applied (X and y)")

        noise_results = {}
        training_curves = {}

        for model_name, model_type, params in model_variants:
            logger.info(f"  Training {model_name}...")

            try:
                result = run_single_experiment(
                    X_train=X_train_scaled,
                    y_train=y_train_scaled,
                    X_val=X_val_scaled,
                    y_val=y_val_scaled,
                    X_test=X_test_scaled,
                    y_test=y_test_scaled,
                    model_name=model_name,
                    model_type=model_type,
                    params=params,
                    record_loss_curve=True,
                    eval_interval=10
                )

                # Store result
                result_entry = {
                    'Nonlinear_Type': nonlinear_type,
                    'Noise_Std': noise_std,
                    'Model': model_name,
                    'Test_RMSE': result['test_metrics']['rmse'],
                    'Test_MAE': result['test_metrics']['mae'],
                    'Test_R2': result['test_metrics']['r2'],
                    'N_Trees_Used': params.get('n_estimators', 200),
                    'Train_Time_s': result['train_time']
                }
                all_results.append(result_entry)

                noise_results[model_name] = result['test_metrics']['rmse']
                training_curves[model_name] = (result['train_losses'], result['val_losses'])

                logger.info(f"    Test RMSE: {result['test_metrics']['rmse']:.4f}, "
                           f"Time: {result['train_time']:.2f}s")

                # Save detailed log for this experiment
                log_filename = f"exp2A-{nonlinear_type}-noise{noise_std}-seed{seed}-{model_name}.log"
                log_path = os.path.join(logs_dir, log_filename)
                save_experiment_log(
                    log_path=log_path,
                    exp_name='exp2A',
                    nonlinear_type=nonlinear_type,
                    noise_std=noise_std,
                    seed=seed,
                    model_name=model_name,
                    model_type=model_type,
                    params=params,
                    n_train=len(X_train),
                    n_val=len(X_val),
                    n_test=len(X_test),
                    n_features=X_train.shape[1],
                    scaler_params=scaler_params,
                    data_metadata=metadata,
                    train_metrics=result['train_metrics'],
                    val_metrics=result['val_metrics'],
                    test_metrics=result['test_metrics'],
                    train_time=result['train_time'],
                    train_losses=result['train_losses'],
                    val_losses=result['val_losses']
                )

                # Plot individual training curve
                if result['train_losses']:
                    plot_training_curve(
                        train_losses=result['train_losses'],
                        val_losses=result['val_losses'],
                        exp_type='exp2A',
                        dataset_name=f"{nonlinear_type}_noise{noise_std}",
                        seed=seed,
                        model_name=model_name,
                        n_iter=params.get('n_estimators', 200),
                        plot_dir=plots_dir,
                        test_rmse=result['test_metrics']['rmse']
                    )

            except Exception as e:
                logger.error(f"    Failed: {e}")
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

    # Plot noise robustness curve
    if results_by_noise:
        results_by_model = {}
        for model_name, _, _ in model_variants:
            rmse_values = []
            for noise_std in noise_levels:
                if noise_std in results_by_noise and model_name in results_by_noise[noise_std]:
                    rmse_values.append(results_by_noise[noise_std][model_name])
                else:
                    rmse_values.append(np.nan)
            results_by_model[model_name] = rmse_values

        plot_noise_robustness(
            noise_levels=noise_levels,
            results_by_model=results_by_model,
            nonlinear_type=nonlinear_type,
            plot_dir=plots_dir
        )

    return all_results


def run_experiment2a(seed: int = RANDOM_SEED):
    """
    Run Experiment 2A: Synthetic nonlinear robustness test.
    """
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results', 'exp2A')
    logs_dir = os.path.join(base_dir, 'logs', 'exp2A')
    plots_dir = os.path.join(base_dir, 'plots', 'exp2A')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Setup logger
    logger = ExperimentLogger('exp2A', log_dir=logs_dir)

    logger.info("=" * 80)
    logger.info("Experiment 2A: Synthetic Nonlinear Robustness Test")
    logger.info("MetaGB vs XGBoost on Synthetic Data")
    logger.info("=" * 80)
    logger.info(f"Random seed: {seed}")
    logger.info(f"Noise levels: {NOISE_LEVELS}")
    logger.info("Feature normalization: Z-Score (StandardScaler)")

    # Get nonlinear generators
    generators = get_nonlinear_generators()
    logger.info(f"Nonlinear types: {list(generators.keys())}")

    # Print model variants
    model_variants = get_model_variants_exp2()
    logger.info(f"Models to test:")
    for name, mtype, params in model_variants:
        logger.info(f"  - {name} ({mtype})")

    # Calculate total experiments
    n_total = len(generators) * len(NOISE_LEVELS) * len(model_variants)
    logger.info(f"Total experiments: {n_total}")

    # Run experiments
    all_results = []

    for nonlinear_type, generator_func in generators.items():
        try:
            results = run_experiment_for_nonlinear_type(
                nonlinear_type=nonlinear_type,
                generator_func=generator_func,
                noise_levels=NOISE_LEVELS,
                seed=seed,
                logger=logger,
                logs_dir=logs_dir,
                plots_dir=plots_dir
            )
            all_results.extend(results)

            # Save intermediate results
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(results_dir, 'exp2A_summary_partial.csv'), index=False)

        except Exception as e:
            logger.error(f"Failed for {nonlinear_type}: {e}")
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

        # Aggregate by model
        logger.info("\n--- Average Test RMSE by Model ---")
        for model in df['Model'].unique():
            model_df = df[df['Model'] == model]
            logger.info(f"{model}: {model_df['Test_RMSE'].mean():.4f} "
                       f"(+/- {model_df['Test_RMSE'].std():.4f})")

        # Aggregate by nonlinear type
        logger.info("\n--- Average Test RMSE by Nonlinear Type ---")
        for nl_type in df['Nonlinear_Type'].unique():
            type_df = df[df['Nonlinear_Type'] == nl_type]
            logger.info(f"{nl_type}: {type_df['Test_RMSE'].mean():.4f}")

        # Best model per nonlinear type and noise level
        logger.info("\n--- Best Model per Configuration ---")
        for nl_type in df['Nonlinear_Type'].unique():
            for noise in df['Noise_Std'].unique():
                subset = df[(df['Nonlinear_Type'] == nl_type) & (df['Noise_Std'] == noise)]
                if len(subset) > 0:
                    best_idx = subset['Test_RMSE'].idxmin()
                    best_model = subset.loc[best_idx, 'Model']
                    best_rmse = subset.loc[best_idx, 'Test_RMSE']
                    logger.debug(f"{nl_type}, noise={noise}: {best_model} (RMSE={best_rmse:.4f})")

    logger.info("\nExperiment 2A completed!")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 2A: Synthetic Nonlinear Robustness Test')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')

    args = parser.parse_args()

    run_experiment2a(seed=args.seed)
