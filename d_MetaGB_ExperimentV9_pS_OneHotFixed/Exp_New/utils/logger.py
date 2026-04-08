"""
Logger Module
Handles experiment logging with file and console output.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any


def setup_experiment_logger(
    exp_type: str,
    dataset_name: str,
    seed: int,
    model_name: str,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up experiment logger with file and console handlers.

    Args:
        exp_type: Experiment type (exp1, exp2A, exp2B)
        dataset_name: Name of the dataset
        seed: Random seed
        model_name: Name of the model
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{exp_type}-{dataset_name}-seed{seed}-{model_name}-{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Create unique logger name to avoid conflicts
    logger_name = f"{exp_type}_{dataset_name}_{seed}_{model_name}_{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - info level only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Log file: {log_path}")

    return logger


class ExperimentLogger:
    """
    Experiment logger class for managing experiment logs.
    Provides a simplified interface for logging experiment progress.
    """

    def __init__(
        self,
        exp_type: str,
        log_dir: str = "logs",
        console_level: int = logging.INFO
    ):
        """
        Initialize experiment logger.

        Args:
            exp_type: Experiment type (exp1, exp2A, exp2B)
            log_dir: Directory for log files
            console_level: Logging level for console output
        """
        self.exp_type = exp_type
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{exp_type}-main-{timestamp}.log"
        self.log_path = os.path.join(log_dir, self.log_filename)

        # Set up logger
        self.logger = logging.getLogger(f"{exp_type}_main_{timestamp}")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def log_experiment_start(
        self,
        dataset_name: str,
        n_train: int,
        n_test: int,
        model_name: str,
        params: Dict[str, Any]
    ):
        """
        Log experiment start information.

        Args:
            dataset_name: Name of the dataset
            n_train: Number of training samples
            n_test: Number of test samples
            model_name: Name of the model
            params: Model parameters
        """
        self.info("=" * 80)
        self.info(f"Dataset: {dataset_name}")
        self.info(f"Train samples: {n_train}, Test samples: {n_test}")
        self.info(f"Model: {model_name}")
        self.info("Parameters:")
        for k, v in params.items():
            self.debug(f"  {k}: {v}")
        self.info("=" * 80)

    def log_grid_search_start(self, n_combinations: int):
        """Log grid search start."""
        self.info(f"Starting GridSearchCV with {n_combinations} combinations...")

    def log_grid_search_result(self, best_params: Dict, best_score: float):
        """Log grid search result."""
        self.info(f"Best params: {best_params}")
        self.info(f"Best CV score: {best_score:.4f} (RMSE)")

    def log_training_start(self):
        """Log training start."""
        self.info("Training final model...")

    def log_final_results(
        self,
        test_rmse: float,
        test_mae: float,
        test_r2: float,
        train_time: float
    ):
        """
        Log final test results.

        Args:
            test_rmse: Test RMSE
            test_mae: Test MAE
            test_r2: Test R-squared
            train_time: Training time in seconds
        """
        self.info(f"Final test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        self.info(f"Training time: {train_time:.1f}s")

    def log_separator(self):
        """Log a separator line."""
        self.info("-" * 80)

    def close(self):
        """Close all handlers."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


def create_detailed_log(
    log_path: str,
    dataset_name: str,
    seed: int,
    model_name: str,
    n_train: int,
    n_test: int,
    best_params: Dict,
    best_cv_score: float,
    test_rmse: float,
    test_mae: float,
    test_r2: float,
    train_time: float
):
    """
    Create a detailed log file for a single experiment run.

    Args:
        log_path: Path to save the log file
        dataset_name: Name of the dataset
        seed: Random seed
        model_name: Name of the model
        n_train: Number of training samples
        n_test: Number of test samples
        best_params: Best parameters from grid search
        best_cv_score: Best cross-validation score
        test_rmse: Test RMSE
        test_mae: Test MAE
        test_r2: Test R-squared
        train_time: Training time in seconds
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"[{timestamp}] Dataset: {dataset_name}, Seed: {seed}, Model: {model_name}\n")
        f.write(f"[{timestamp}] Train samples: {n_train}, Test samples: {n_test}\n")

        if best_params:
            n_combinations = "N/A"  # Would need to be passed in
            f.write(f"[{timestamp}] Starting GridSearchCV...\n")
            f.write(f"[{timestamp}] Best params: {best_params}\n")
            f.write(f"[{timestamp}] Best CV score: {best_cv_score:.4f} (RMSE)\n")

        f.write(f"[{timestamp}] Training final model...\n")
        f.write(f"[{timestamp}] Final test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}\n")
        f.write(f"[{timestamp}] Training time: {train_time:.1f}s\n")
