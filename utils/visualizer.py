"""
Visualization Module
For plotting MSE Loss curves during training
"""

import matplotlib.pyplot as plt
import os


def plot_mse_loss(train_losses, val_losses, exp_type, dataset_name, seed,
                  model_name, epoch, plot_dir="plots"):
    """
    Plot and save MSE Loss curves during training
    
    Args:
        train_losses: Training set MSE list
        val_losses: Validation set MSE list
        exp_type: Experiment type
        dataset_name: Dataset name
        seed: Random seed
        model_name: Model name
        epoch: Number of training epochs
        plot_dir: Plot save directory
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    filename = f"{exp_type}-{dataset_name}-{seed}-{model_name}-{epoch}-mse_loss.png"
    save_path = os.path.join(plot_dir, filename)
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training MSE', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation MSE', linewidth=2)
    
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(f'MSE Loss During Training\n{exp_type} - {dataset_name} - {model_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_noise_robustness(noise_levels, xgb_rmse_list, metagb_rmse_list,
                          dataset_name, plot_dir="plots"):
    """
    Plot noise robustness curves
    
    Args:
        noise_levels: Noise level list
        xgb_rmse_list: XGBoost RMSE list
        metagb_rmse_list: MetaGB RMSE list
        dataset_name: Dataset name
        plot_dir: Plot save directory
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    filename = f"B-{dataset_name}-noise_robustness.png"
    save_path = os.path.join(plot_dir, filename)
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, xgb_rmse_list, 'b-o', label='XGBoost', linewidth=2, markersize=8)
    plt.plot(noise_levels, metagb_rmse_list, 'r-s', label='MetaGB', linewidth=2, markersize=8)
    
    plt.xlabel('Noise Level', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt_title = f'Noise Robustness - {dataset_name}'
    plt.title(plt_title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_parameter_sensitivity(param_values, avg_mse_list, param_name,
                              dataset_name, plot_dir="plots"):
    """
    Plot parameter sensitivity curves
    
    Args:
        param_values: Parameter value list
        avg_mse_list: Average MSE list
        param_name: Parameter name
        dataset_name: Dataset name
        plot_dir: Plot save directory
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    filename = f"A-{dataset_name}-{param_name}-sensitivity.png"
    save_path = os.path.join(plot_dir, filename)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, avg_mse_list, 'g-o', linewidth=2, markersize=8)
    
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Average Validation MSE', fontsize=12)
    plt.title(f'Parameter Sensitivity Analysis - {param_name} - {dataset_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path
