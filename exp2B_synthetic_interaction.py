"""
Experiment 2B: Feature Interaction Pattern Test
MetaGB vs XGBoost on Synthetic Data with Various Interaction Patterns

This experiment tests model performance on different feature interaction patterns:
1. Piecewise Linear: Feature space divided into regions with different linear relationships
2. Conditional Prediction: Categorical feature selects prediction function
3. Sparse Interaction: High-dimensional data with few relevant interacting features

Key metrics:
- Test RMSE, MAE, R²
- Tree depth comparison
- Feature selection precision (for sparse interaction datasets)
- Z-Score normalization applied to features

Models tested:
- XGBoost (baseline)
- MetaGB_v0.0 (pred_feature_fraction=0.0)
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
    RANDOM_SEED, TRAIN_TEST_SPLIT,
    get_fixed_xgb_params, get_fixed_metagb_params, get_model_variants_exp2
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
        'feature_importance': feature_importance
    }


def run_experiment_for_dataset(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict,
    seed: int,
    logger: ExperimentLogger,
    logs_dir: str,
    plots_dir: str
) -> list:
    """
    Run experiments for a single dataset with all model variants.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Type: {metadata.get('type', 'unknown')}")
    logger.info(f"Formula: {metadata.get('formula', 'N/A')}")
    logger.info(f"{'='*80}")

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
    logger.info(f"Features: {X.shape[1]}")
    logger.info("Z-Score normalization applied (X and y)")

    # Get model variants
    model_variants = get_model_variants_exp2()

    results = []
    importance_by_model = {}

    for model_name, model_type, params in model_variants:
        logger.info(f"\n  Training {model_name}...")

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

            # Calculate feature selection precision for sparse interaction datasets
            feature_precision = None
            if 'relevant_features' in metadata:
                feature_precision = calculate_feature_selection_precision(
                    result['feature_importance'],
                    metadata['relevant_features'],
                    top_k=10
                )

            # Store result
            result_entry = {
                'Dataset_Type': metadata.get('type', 'unknown'),
                'Variant': dataset_name,
                'Model': model_name,
                'Test_RMSE': result['test_metrics']['rmse'],
                'Test_MAE': result['test_metrics']['mae'],
                'Test_R2': result['test_metrics']['r2'],
                'Avg_Tree_Depth': params.get('max_depth', 5),
                'N_Estimators': params.get('n_estimators', 200),
                'Train_Time_s': result['train_time'],
                'Feature_Selection_Precision': feature_precision if feature_precision else '-'
            }
            results.append(result_entry)

            # Store feature importance for plotting
            importance_by_model[model_name] = result['feature_importance']

            logger.info(f"    Test RMSE: {result['test_metrics']['rmse']:.4f}")
            logger.info(f"    Test MAE: {result['test_metrics']['mae']:.4f}")
            logger.info(f"    Test R²: {result['test_metrics']['r2']:.4f}")
            if feature_precision is not None:
                logger.info(f"    Feature Selection Precision (Top-10): {feature_precision:.2f}")
            logger.info(f"    Time: {result['train_time']:.2f}s")

            # Save detailed log for this experiment
            log_filename = f"exp2B-{dataset_name}-seed{seed}-{model_name}.log"
            log_path = os.path.join(logs_dir, log_filename)
            save_experiment_log(
                log_path=log_path,
                exp_name='exp2B',
                dataset_name=dataset_name,
                dataset_type=metadata.get('type', 'unknown'),
                seed=seed,
                model_name=model_name,
                model_type=model_type,
                params=params,
                n_train=len(X_train),
                n_val=len(X_val),
                n_test=len(X_test),
                n_features=X.shape[1],
                scaler_params=scaler_params,
                data_metadata=metadata,
                train_metrics=result['train_metrics'],
                val_metrics=result['val_metrics'],
                test_metrics=result['test_metrics'],
                train_time=result['train_time'],
                train_losses=result['train_losses'],
                val_losses=result['val_losses'],
                feature_precision=feature_precision
            )

            # Plot training curve
            if result['train_losses']:
                plot_training_curve(
                    train_losses=result['train_losses'],
                    val_losses=result['val_losses'],
                    exp_type='exp2B',
                    dataset_name=dataset_name,
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

    # Plot feature importance comparison for sparse interaction datasets
    if 'relevant_features' in metadata and importance_by_model:
        try:
            plot_feature_importance_comparison(
                importance_by_model=importance_by_model,
                relevant_features=metadata['relevant_features'],
                dataset_name=dataset_name,
                top_k=min(10, X.shape[1]),
                plot_dir=plots_dir
            )
        except Exception as e:
            logger.warning(f"Failed to plot feature importance: {e}")

    # Plot model comparison bar chart
    if results:
        try:
            plot_model_comparison_bar(
                results=[{'model': r['Model'], 'test_rmse': r['Test_RMSE']} for r in results],
                metric='test_rmse',
                dataset_name=dataset_name,
                plot_dir=plots_dir
            )
        except Exception as e:
            logger.warning(f"Failed to plot model comparison: {e}")

    return results


def run_experiment2b(seed: int = RANDOM_SEED):
    """
    Run Experiment 2B: Feature interaction pattern test.
    """
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results', 'exp2B')
    logs_dir = os.path.join(base_dir, 'logs', 'exp2B')
    plots_dir = os.path.join(base_dir, 'plots', 'exp2B')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Setup logger
    logger = ExperimentLogger('exp2B', log_dir=logs_dir)

    logger.info("=" * 80)
    logger.info("Experiment 2B: Feature Interaction Pattern Test")
    logger.info("MetaGB vs XGBoost on Synthetic Interaction Data")
    logger.info("=" * 80)
    logger.info(f"Random seed: {seed}")
    logger.info("Feature normalization: Z-Score (StandardScaler)")

    # Generate all datasets
    logger.info("\nGenerating datasets...")
    datasets = generate_exp2b_datasets(random_state=seed)
    logger.info(f"Generated {len(datasets)} datasets:")
    for name, X, y, meta in datasets:
        logger.info(f"  - {name}: {X.shape[0]} samples, {X.shape[1]} features")

    # Print model variants
    model_variants = get_model_variants_exp2()
    logger.info(f"\nModels to test:")
    for name, mtype, params in model_variants:
        pff = params.get('pred_feature_fraction', '-')
        logger.info(f"  - {name} (pred_feature_fraction={pff})")

    # Calculate total experiments
    n_total = len(datasets) * len(model_variants)
    logger.info(f"\nTotal experiments: {n_total}")

    # Run experiments
    all_results = []

    for dataset_name, X, y, metadata in datasets:
        try:
            results = run_experiment_for_dataset(
                dataset_name=dataset_name,
                X=X,
                y=y,
                metadata=metadata,
                seed=seed,
                logger=logger,
                logs_dir=logs_dir,
                plots_dir=plots_dir
            )
            all_results.extend(results)

            # Save intermediate results
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(results_dir, 'exp2B_summary_partial.csv'), index=False)

        except Exception as e:
            logger.error(f"Failed for {dataset_name}: {e}")
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

        # Summary by dataset type
        logger.info("\n--- Results by Dataset Type ---")
        for dtype in df['Dataset_Type'].unique():
            logger.info(f"\n{dtype.upper()}:")
            type_df = df[df['Dataset_Type'] == dtype]

            for model in type_df['Model'].unique():
                model_df = type_df[type_df['Model'] == model]
                avg_rmse = model_df['Test_RMSE'].mean()
                logger.info(f"  {model}: Avg RMSE = {avg_rmse:.4f}")

        # Best model per dataset
        logger.info("\n--- Best Model per Dataset ---")
        for variant in df['Variant'].unique():
            subset = df[df['Variant'] == variant]
            if len(subset) > 0:
                best_idx = subset['Test_RMSE'].idxmin()
                best_model = subset.loc[best_idx, 'Model']
                best_rmse = subset.loc[best_idx, 'Test_RMSE']
                logger.info(f"  {variant}: {best_model} (RMSE={best_rmse:.4f})")

        # Feature selection precision for sparse datasets
        sparse_df = df[df['Dataset_Type'] == 'sparse_interaction']
        if len(sparse_df) > 0:
            logger.info("\n--- Feature Selection Precision (Sparse Interaction) ---")
            for variant in sparse_df['Variant'].unique():
                logger.info(f"\n  {variant}:")
                variant_df = sparse_df[sparse_df['Variant'] == variant]
                for _, row in variant_df.iterrows():
                    prec = row['Feature_Selection_Precision']
                    if prec != '-':
                        logger.info(f"    {row['Model']}: {prec:.2f}")

        # Plot tree depth comparison by dataset type
        for dtype in df['Dataset_Type'].unique():
            type_df = df[df['Dataset_Type'] == dtype]
            if len(type_df) > 0:
                # Aggregate results for plotting
                agg_results = []
                for model in type_df['Model'].unique():
                    model_df = type_df[type_df['Model'] == model]
                    agg_results.append({
                        'model': model,
                        'avg_tree_depth': model_df['Avg_Tree_Depth'].mean(),
                        'std_tree_depth': model_df['Avg_Tree_Depth'].std(),
                        'test_rmse': model_df['Test_RMSE'].mean()
                    })

                try:
                    plot_tree_depth_comparison(
                        results=agg_results,
                        dataset_type=dtype,
                        plot_dir=plots_dir
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot tree depth comparison for {dtype}: {e}")

    logger.info("\nExperiment 2B completed!")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 2B: Feature Interaction Pattern Test')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')

    args = parser.parse_args()

    run_experiment2b(seed=args.seed)
