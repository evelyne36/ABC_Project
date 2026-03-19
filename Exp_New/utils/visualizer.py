"""
Visualizer Module
Handles all plotting and visualization for experiments.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for background execution
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional, Tuple


def plot_training_curve(
    train_losses: List[float],
    val_losses: List[float],
    exp_type: str,
    dataset_name: str,
    seed: int,
    model_name: str,
    n_iter: int,
    plot_dir: str = "plots",
    test_rmse: Optional[float] = None,
    early_stopping_round: Optional[int] = None,
    use_rmse: bool = True
) -> str:
    """
    Plot training and validation MSE/RMSE curves.

    File naming format: exp1-{dataset_name}-seed{seed}-{model_name}-iter{n_iter}-mse.png

    Args:
        train_losses: List of training MSE values
        val_losses: List of validation MSE values
        exp_type: Experiment type (exp1, exp2A, exp2B)
        dataset_name: Name of the dataset
        seed: Random seed
        model_name: Model name
        n_iter: Number of iterations
        plot_dir: Directory to save plots
        test_rmse: Final test RMSE to annotate (optional)
        early_stopping_round: Round where early stopping occurred (optional)
        use_rmse: If True, convert MSE to RMSE for plotting

    Returns:
        Path to saved plot
    """
    os.makedirs(plot_dir, exist_ok=True)

    metric_name = "rmse" if use_rmse else "mse"
    filename = f"{exp_type}-{dataset_name}-seed{seed}-{model_name}-iter{n_iter}-{metric_name}.png"
    save_path = os.path.join(plot_dir, filename)

    fig, ax = plt.subplots(figsize=(10, 6))

    # X-axis: iteration/boosting rounds
    if len(train_losses) > 0:
        x_vals = np.linspace(1, n_iter, len(train_losses))
    else:
        x_vals = []

    # Convert to RMSE if requested
    if use_rmse:
        train_values = [np.sqrt(mse) for mse in train_losses] if train_losses else []
        val_values = [np.sqrt(mse) for mse in val_losses] if val_losses else []
        ylabel = 'RMSE'
    else:
        train_values = train_losses
        val_values = val_losses
        ylabel = 'MSE Loss'

    # Plot curves
    if len(train_values) > 0:
        ax.plot(x_vals, train_values, 'b-', label=f'Train {ylabel}', linewidth=2)
    if len(val_values) > 0:
        ax.plot(x_vals, val_values, 'r-', label=f'Validation {ylabel}', linewidth=2)

        # Mark best validation point
        best_idx = np.argmin(val_values)
        best_val = val_values[best_idx]
        best_iter = x_vals[best_idx]
        ax.scatter([best_iter], [best_val], color='red', s=100, zorder=5,
                   marker='*', label=f'Best Val {ylabel}: {best_val:.4f}')

        # Mark early stopping point if applicable
        if early_stopping_round is not None:
            ax.axvline(x=early_stopping_round, color='green', linestyle='--',
                      linewidth=2, label=f'Early Stop @ {early_stopping_round}')

    ax.set_xlabel('Boosting Rounds', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    title = f'Training Curve: {model_name}\n{dataset_name} (seed={seed})'
    if test_rmse is not None:
        title += f'\nFinal Test RMSE: {test_rmse:.4f}'
    if early_stopping_round is not None:
        title += f' (Early Stopped)'
    ax.set_title(title, fontsize=14)

    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_noise_robustness(
    noise_levels: List[float],
    results_by_model: Dict[str, List[float]],
    nonlinear_type: str,
    plot_dir: str = "plots"
) -> str:
    """
    Plot noise robustness curves for multiple models.

    File naming format: exp2A-{nonlinear_type}-robustness_curve.png

    Args:
        noise_levels: List of noise standard deviations
        results_by_model: Dictionary mapping model name to list of RMSE values
        nonlinear_type: Type of nonlinearity (square, sin, etc.)
        plot_dir: Directory to save plots

    Returns:
        Path to saved plot
    """
    os.makedirs(plot_dir, exist_ok=True)

    filename = f"exp2A-{nonlinear_type}-robustness_curve.png"
    save_path = os.path.join(plot_dir, filename)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for different models
    styles = {
        'XGBoost': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'MetaGB_v0.0': {'color': 'green', 'marker': 's', 'linestyle': '--'},
        'MetaGB_v0.4': {'color': 'orange', 'marker': '^', 'linestyle': '-.'},
        'MetaGB_v0.8': {'color': 'red', 'marker': 'd', 'linestyle': ':'},
    }

    for model_name, rmse_values in results_by_model.items():
        style = styles.get(model_name, {'color': 'gray', 'marker': 'x', 'linestyle': '-'})
        ax.plot(noise_levels, rmse_values,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=2,
                markersize=8,
                label=model_name)

    ax.set_xlabel('Noise Level (std)', fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title(f'Noise Robustness: {nonlinear_type} nonlinearity', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Use log scale for x-axis if noise levels span multiple orders of magnitude
    if max(noise_levels) / (min([n for n in noise_levels if n > 0]) if any(n > 0 for n in noise_levels) else 1) > 10:
        ax.set_xscale('symlog', linthresh=0.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_training_curves_comparison(
    results_by_model: Dict[str, Tuple[List[float], List[float]]],
    nonlinear_type: str,
    noise_std: float,
    n_iter: int,
    plot_dir: str = "plots"
) -> str:
    """
    Plot training curves comparison for multiple models (4 subplots).

    File naming format: exp2A-{nonlinear_type}-noise{std}-training_curves.png

    Args:
        results_by_model: Dict mapping model name to (train_losses, val_losses)
        nonlinear_type: Type of nonlinearity
        noise_std: Noise standard deviation
        n_iter: Number of iterations
        plot_dir: Directory to save plots

    Returns:
        Path to saved plot
    """
    os.makedirs(plot_dir, exist_ok=True)

    filename = f"exp2A-{nonlinear_type}-noise{noise_std}-training_curves.png"
    save_path = os.path.join(plot_dir, filename)

    n_models = len(results_by_model)
    n_cols = 2
    n_rows = (n_models + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    model_names = list(results_by_model.keys())

    for idx, model_name in enumerate(model_names):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        train_losses, val_losses = results_by_model[model_name]

        if len(train_losses) > 0:
            x_vals = np.linspace(1, n_iter, len(train_losses))
            ax.plot(x_vals, train_losses, 'b-', label='Train MSE', linewidth=1.5)
            ax.plot(x_vals, val_losses, 'r-', label='Validation MSE', linewidth=1.5)

        ax.set_xlabel('Boosting Rounds', fontsize=10)
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_title(f'{model_name}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'Training Curves: {nonlinear_type} (noise={noise_std})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_tree_depth_comparison(
    results: List[Dict],
    dataset_type: str,
    plot_dir: str = "plots"
) -> str:
    """
    Plot tree depth comparison bar chart for Experiment 2B.

    File naming format: exp2B-{dataset_type}-tree_depth_comparison.png

    Args:
        results: List of result dictionaries with 'model', 'avg_tree_depth', 'std_tree_depth'
        dataset_type: Type of dataset (piecewise, conditional, sparse)
        plot_dir: Directory to save plots

    Returns:
        Path to saved plot
    """
    os.makedirs(plot_dir, exist_ok=True)

    filename = f"exp2B-{dataset_type}-tree_depth_comparison.png"
    save_path = os.path.join(plot_dir, filename)

    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = [r['model'] for r in results]
    avg_depths = [r.get('avg_tree_depth', r.get('max_depth', 5)) for r in results]
    std_depths = [r.get('std_tree_depth', 0) for r in results]

    x_pos = np.arange(len(model_names))
    colors = ['blue', 'green', 'orange', 'red'][:len(model_names)]

    bars = ax.bar(x_pos, avg_depths, yerr=std_depths, capsize=5,
                  color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Average Tree Depth', fontsize=12)
    ax.set_title(f'Tree Depth Comparison: {dataset_type}', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, avg_depths):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_model_comparison_bar(
    results: List[Dict],
    metric: str,
    dataset_name: str,
    plot_dir: str = "plots"
) -> str:
    """
    Plot bar chart comparing models on a specific metric.

    Args:
        results: List of result dictionaries
        metric: Metric to compare (test_rmse, test_mae, test_r2)
        dataset_name: Name of the dataset
        plot_dir: Directory to save plots

    Returns:
        Path to saved plot
    """
    os.makedirs(plot_dir, exist_ok=True)

    filename = f"comparison-{dataset_name}-{metric}.png"
    save_path = os.path.join(plot_dir, filename)

    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = [r['model'] for r in results]
    metric_values = [r[metric] for r in results]

    x_pos = np.arange(len(model_names))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    bars = ax.bar(x_pos, metric_values, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.upper().replace('_', ' '), fontsize=12)
    ax.set_title(f'{metric.upper().replace("_", " ")} Comparison: {dataset_name}', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_feature_importance_comparison(
    importance_by_model: Dict[str, np.ndarray],
    relevant_features: List[int],
    dataset_name: str,
    top_k: int = 10,
    plot_dir: str = "plots"
) -> str:
    """
    Plot feature importance comparison highlighting relevant features.

    Args:
        importance_by_model: Dict mapping model name to importance array
        relevant_features: List of truly relevant feature indices
        dataset_name: Name of the dataset
        top_k: Number of top features to show
        plot_dir: Directory to save plots

    Returns:
        Path to saved plot
    """
    os.makedirs(plot_dir, exist_ok=True)

    filename = f"exp2B-{dataset_name}-feature_importance.png"
    save_path = os.path.join(plot_dir, filename)

    n_models = len(importance_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    relevant_set = set(relevant_features)

    for ax, (model_name, importance) in zip(axes, importance_by_model.items()):
        # Get top-k features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_importance = importance[top_indices]

        # Color bars based on whether feature is relevant
        colors = ['green' if idx in relevant_set else 'gray' for idx in top_indices]

        y_pos = np.arange(top_k)
        ax.barh(y_pos, top_importance, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Feature {idx}' for idx in top_indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name}')
        ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Relevant'),
        Patch(facecolor='gray', alpha=0.7, label='Noise')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    fig.suptitle(f'Feature Importance: {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def create_summary_table_plot(
    summary_df,
    exp_type: str,
    plot_dir: str = "plots"
) -> str:
    """
    Create a visual table from summary DataFrame.

    Args:
        summary_df: Pandas DataFrame with summary results
        exp_type: Experiment type
        plot_dir: Directory to save plots

    Returns:
        Path to saved plot
    """
    os.makedirs(plot_dir, exist_ok=True)

    filename = f"{exp_type}_summary_table.png"
    save_path = os.path.join(plot_dir, filename)

    fig, ax = plt.subplots(figsize=(14, len(summary_df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    plt.title(f'{exp_type} Summary Results', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path
