"""
Model Trainer Module
Handles training, evaluation, and grid search for XGBoost and MetaGB models.
Includes early stopping functionality.
"""

import sys
import os
import numpy as np
import time
from itertools import product
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from functools import partial

from xgboost import XGBRegressor

# Global setting for parallel jobs
N_JOBS = 1  # Default: single thread


def set_n_jobs(n_jobs: int):
    """
    Set the number of parallel jobs globally.

    Args:
        n_jobs: Number of parallel jobs. -1 means use all CPUs.
    """
    global N_JOBS
    N_JOBS = n_jobs


def get_n_jobs() -> int:
    """Get the current number of parallel jobs."""
    return N_JOBS

# Import MetaGB with proper path handling
# The metagb module requires libmetagb (compiled CFFI) to be importable
_current_dir = os.path.dirname(os.path.abspath(__file__))
_metagb_path = os.path.normpath(os.path.join(_current_dir, '..',  'metagb'))

# Save original state
_original_cwd = os.getcwd()

try:
    # Change to metagb directory so libmetagb can be found
    os.chdir(_metagb_path)
    if _metagb_path not in sys.path:
        sys.path.insert(0, _metagb_path)

    from metagb_utils import MetaGB
finally:
    # Restore original working directory
    os.chdir(_original_cwd)


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    Monitors validation loss and stops training when no improvement is seen.
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-6, restore_best: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of rounds to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best: Whether to restore the best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_score = float('inf')
        self.best_round = 0
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_round: int, val_score: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_round: Current training round
            val_score: Current validation score (lower is better)

        Returns:
            True if training should stop, False otherwise
        """
        if val_score < self.best_score - self.min_delta:
            self.best_score = val_score
            self.best_round = current_round
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False

    def get_best_round(self) -> int:
        """Get the round with the best validation score."""
        return self.best_round

    def reset(self):
        """Reset the early stopping state."""
        self.best_score = float('inf')
        self.best_round = 0
        self.counter = 0
        self.should_stop = False


def create_xgboost_model(params: Dict) -> XGBRegressor:
    """
    Create an XGBoost model with given parameters.

    Args:
        params: Parameter dictionary

    Returns:
        XGBRegressor instance
    """
    model_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'n_jobs': 1,
        'random_state': params.get('random_state', 42),
        'n_estimators': params.get('n_estimators', 100),
        'max_depth': params.get('max_depth', 5),
        'learning_rate': params.get('learning_rate', 0.1),
        'subsample': params.get('subsample', 1.0),
        'colsample_bytree': params.get('colsample_bytree', 0.8),
        'min_child_weight': params.get('min_child_weight', 20),
        'reg_lambda': params.get('reg_lambda', 0.0),
        'max_bin': params.get('max_bin', 64),
    }
    return XGBRegressor(**model_params)


def create_metagb_model(params: Dict) -> MetaGB:
    """
    Create a MetaGB model with given parameters.

    Args:
        params: Parameter dictionary

    Returns:
        MetaGB instance
    """
    model = MetaGB(
        learning_rate=params.get('learning_rate', 0.1),
        num_trees=params.get('n_estimators', 100),
        feature_fraction=params.get('colsample_bytree', 0.8),
        pred_feature_fraction=params.get('pred_feature_fraction',
                                         params.get('colsample_bytree', 0.8)),
        bagging_fraction=params.get('subsample', 1.0),
        min_data_in_leaf=params.get('min_data_in_leaf', 20),
        max_depth=params.get('max_depth', 5),
        max_bin=params.get('max_bin', 64),
        lambda_l2=params.get('reg_lambda', 0.0),
        version=params.get('version', 3),
        seed=params.get('random_state', 42)
    )
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary with mse, rmse, mae, r2
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


class ZScoreScaler:
    """
    Z-Score normalization scaler for features.
    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'ZScoreScaler':
        """
        Fit the scaler on training data.

        Args:
            X: Training features

        Returns:
            self
        """
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Training features

        Returns:
            Transformed features
        """
        self.fit(X)
        return self.transform(X)

    def get_params(self) -> Dict[str, Any]:
        """
        Get scaler parameters for logging.

        Returns:
            Dictionary with mean and std for each feature
        """
        if not self.is_fitted:
            return {}
        return {
            'mean': self.scaler.mean_.tolist(),
            'std': self.scaler.scale_.tolist()
        }


class TargetScaler:
    """
    Z-Score normalization scaler for target variable (y).
    Fits on training target and can inverse_transform predictions back to original scale.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False

    def fit(self, y: np.ndarray) -> 'TargetScaler':
        """
        Fit the scaler on training target.

        Args:
            y: Training target values

        Returns:
            self
        """
        self.mean_ = np.mean(y)
        self.std_ = np.std(y)
        if self.std_ == 0:
            self.std_ = 1.0  # Avoid division by zero
        self.is_fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform target using fitted scaler.

        Args:
            y: Target values to transform

        Returns:
            Transformed target values
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        return (y - self.mean_) / self.std_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            y: Training target values

        Returns:
            Transformed target values
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale.

        Args:
            y_scaled: Scaled predictions

        Returns:
            Predictions in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        return y_scaled * self.std_ + self.mean_

    def get_params(self) -> Dict[str, Any]:
        """
        Get scaler parameters for logging.

        Returns:
            Dictionary with mean and std
        """
        if not self.is_fitted:
            return {}
        return {
            'mean': float(self.mean_),
            'std': float(self.std_)
        }


def apply_zscore_normalization(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], ZScoreScaler]:
    """
    Apply Z-Score normalization to features.

    Fits scaler on training data and transforms all datasets.

    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = ZScoreScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_single_model(
    model_type: str,
    params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    record_loss_curve: bool = False,
    eval_interval: int = 10,
    early_stopping_rounds: Optional[int] = None,
    early_stopping_min_delta: float = 1e-6
) -> Dict[str, Any]:
    """
    Train a single model and return results.

    Args:
        model_type: 'xgboost' or 'metagb'
        params: Model parameters
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        record_loss_curve: Whether to record training loss curve
        eval_interval: Interval for recording loss (only if record_loss_curve=True)
        early_stopping_rounds: Number of rounds for early stopping (None to disable)
        early_stopping_min_delta: Minimum improvement for early stopping

    Returns:
        Dictionary with model, predictions, metrics, and training time
    """
    start_time = time.time()

    train_losses = []
    val_losses = []
    best_n_trees = params.get('n_estimators', 100)
    early_stopped = False
    early_stopping_round = None

    # Initialize early stopping if enabled
    early_stopper = None
    if early_stopping_rounds is not None and X_val is not None:
        early_stopper = EarlyStopping(
            patience=early_stopping_rounds,
            min_delta=early_stopping_min_delta
        )

    if model_type == 'xgboost':
        if record_loss_curve and X_val is not None:
            # Train incrementally to record loss curve
            n_estimators = params.get('n_estimators', 100)
            for n_trees in range(eval_interval, n_estimators + 1, eval_interval):
                iter_params = params.copy()
                iter_params['n_estimators'] = n_trees
                model = create_xgboost_model(iter_params)
                model.fit(X_train, y_train)

                train_pred = model.predict(X_train)
                train_mse = np.mean((y_train - train_pred) ** 2)
                train_losses.append(train_mse)

                if X_val is not None:
                    val_pred = model.predict(X_val)
                    val_mse = np.mean((y_val - val_pred) ** 2)
                    val_losses.append(val_mse)

                    # Check early stopping
                    if early_stopper is not None:
                        if early_stopper(n_trees, val_mse):
                            early_stopped = True
                            early_stopping_round = early_stopper.get_best_round()
                            best_n_trees = early_stopping_round
                            break

            # If early stopped, retrain with best number of trees
            if early_stopped and early_stopper.restore_best:
                iter_params = params.copy()
                iter_params['n_estimators'] = best_n_trees
                model = create_xgboost_model(iter_params)
                model.fit(X_train, y_train)
        else:
            model = create_xgboost_model(params)
            model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val) if X_val is not None else None

    else:  # metagb
        if record_loss_curve and X_val is not None:
            # Train incrementally to record loss curve
            n_estimators = params.get('n_estimators', 100)
            for n_trees in range(eval_interval, n_estimators + 1, eval_interval):
                iter_params = params.copy()
                iter_params['n_estimators'] = n_trees
                model = create_metagb_model(iter_params)
                model.fit(X_train.astype(np.float64), y_train.astype(np.float64))

                train_pred = model.predict(X_train.astype(np.float64))
                train_mse = np.mean((y_train - train_pred) ** 2)
                train_losses.append(train_mse)

                if X_val is not None:
                    val_pred = model.predict(X_val.astype(np.float64))
                    val_mse = np.mean((y_val - val_pred) ** 2)
                    val_losses.append(val_mse)

                    # Check early stopping
                    if early_stopper is not None:
                        if early_stopper(n_trees, val_mse):
                            early_stopped = True
                            early_stopping_round = early_stopper.get_best_round()
                            best_n_trees = early_stopping_round
                            break

            # If early stopped, retrain with best number of trees
            if early_stopped and early_stopper.restore_best:
                iter_params = params.copy()
                iter_params['n_estimators'] = best_n_trees
                model = create_metagb_model(iter_params)
                model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
        else:
            model = create_metagb_model(params)
            model.fit(X_train.astype(np.float64), y_train.astype(np.float64))

        train_pred = model.predict(X_train.astype(np.float64))
        val_pred = model.predict(X_val.astype(np.float64)) if X_val is not None else None

    train_time = time.time() - start_time

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, train_pred)
    val_metrics = calculate_metrics(y_val, val_pred) if val_pred is not None else None

    result = {
        'model': model,
        'train_pred': train_pred,
        'val_pred': val_pred,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'early_stopped': early_stopped,
        'early_stopping_round': early_stopping_round,
        'best_n_trees': best_n_trees
    }

    return result


def evaluate_model(
    model,
    model_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model
        model_type: 'xgboost' or 'metagb'
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary with predictions and metrics
    """
    if model_type == 'xgboost':
        test_pred = model.predict(X_test)
    else:
        test_pred = model.predict(X_test.astype(np.float64))

    metrics = calculate_metrics(y_test, test_pred)

    return {
        'test_pred': test_pred,
        'test_metrics': metrics
    }


def _evaluate_single_param_combination(
    param_dict: Dict,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    kfold,
) -> Dict:
    """
    Evaluate a single parameter combination with cross-validation.
    Helper function for parallel grid search.
    """
    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Apply Z-Score normalization within each fold
        X_fold_train_scaled, X_fold_val_scaled, _, _ = apply_zscore_normalization(
            X_fold_train, X_val=X_fold_val, X_test=None
        )

        result = train_single_model(
            model_type=model_type,
            params=param_dict,
            X_train=X_fold_train_scaled,
            y_train=y_fold_train,
            X_val=X_fold_val_scaled,
            y_val=y_fold_val,
            record_loss_curve=False
        )

        fold_scores.append(result['val_metrics']['rmse'])

    mean_cv_rmse = np.mean(fold_scores)
    std_cv_rmse = np.std(fold_scores)

    return {
        'params': param_dict.copy(),
        'mean_cv_rmse': mean_cv_rmse,
        'std_cv_rmse': std_cv_rmse,
        'fold_scores': fold_scores
    }


def train_with_grid_search(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List],
    cv: int = 5,
    random_state: int = 42,
    logger=None,
    n_jobs: int = None
) -> Tuple[Any, Dict, List[Dict]]:
    """
    Perform grid search with cross-validation (supports parallel execution).

    Args:
        model_type: 'xgboost' or 'metagb'
        X_train: Training features
        y_train: Training targets
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        random_state: Random seed
        logger: Logger instance (optional)
        n_jobs: Number of parallel jobs (None uses global N_JOBS setting)

    Returns:
        best_model: Model trained with best parameters on full training data
        best_params: Best parameter combination
        cv_results: List of all CV results
    """
    if n_jobs is None:
        n_jobs = get_n_jobs()

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = int(np.prod([len(v) for v in param_values]))

    if logger:
        logger.info(f"Starting GridSearchCV for {model_type}")
        logger.info(f"Total parameter combinations: {total_combinations}")
        logger.info(f"CV folds: {cv}")
        logger.info(f"Parallel jobs: {n_jobs}")

    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Generate all parameter combinations
    all_param_dicts = []
    for params in product(*param_values):
        param_dict = dict(zip(param_names, params))
        param_dict['random_state'] = random_state
        all_param_dicts.append(param_dict)

    # Parallel or sequential execution
    if n_jobs != 1 and model_type == 'xgboost':
        # Parallel execution for XGBoost (MetaGB may have issues with parallel)
        if logger:
            logger.info(f"Running parallel grid search...")

        cv_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_evaluate_single_param_combination)(
                param_dict, model_type, X_train, y_train, kfold
            )
            for param_dict in all_param_dicts
        )
    else:
        # Sequential execution
        cv_results = []
        for i, param_dict in enumerate(all_param_dicts):
            cv_result = _evaluate_single_param_combination(
                param_dict, model_type, X_train, y_train, kfold
            )
            cv_results.append(cv_result)

            if logger and (i + 1) % 10 == 0:
                logger.info(f"[{i+1}/{total_combinations}] CV RMSE: {cv_result['mean_cv_rmse']:.4f} (+/- {cv_result['std_cv_rmse']:.4f})")

    # Find best parameters
    best_cv_score = float('inf')
    best_params = None
    for cv_result in cv_results:
        if cv_result['mean_cv_rmse'] < best_cv_score:
            best_cv_score = cv_result['mean_cv_rmse']
            best_params = cv_result['params'].copy()

    if logger:
        logger.info(f"Best CV RMSE: {best_cv_score:.4f}")
        logger.info(f"Best params: {best_params}")

    # Train final model with best params on full training data
    best_model_result = train_single_model(
        model_type=model_type,
        params=best_params,
        X_train=X_train,
        y_train=y_train,
        record_loss_curve=False
    )

    return best_model_result['model'], best_params, cv_results


def evaluate_and_log(
    model,
    model_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict,
    train_time: float,
    train_losses: List[float],
    val_losses: List[float],
    dataset_name: str,
    model_name: str,
    seed: int,
    logger=None,
    log_dir: str = "logs",
    plot_dir: str = "plots"
) -> Dict[str, Any]:
    """
    Evaluate model and save logs/plots.

    Args:
        model: Trained model
        model_type: 'xgboost' or 'metagb'
        X_test: Test features
        y_test: Test targets
        params: Model parameters
        train_time: Training time in seconds
        train_losses: Training loss history
        val_losses: Validation loss history
        dataset_name: Name of the dataset
        model_name: Name of the model
        seed: Random seed used
        logger: Logger instance
        log_dir: Directory for logs
        plot_dir: Directory for plots

    Returns:
        Dictionary with all evaluation results
    """
    # Evaluate on test set
    eval_result = evaluate_model(model, model_type, X_test, y_test)

    results = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'params': params,
        'test_rmse': eval_result['test_metrics']['rmse'],
        'test_mae': eval_result['test_metrics']['mae'],
        'test_r2': eval_result['test_metrics']['r2'],
        'train_time': train_time,
        'n_estimators': params.get('n_estimators', 100)
    }

    if logger:
        logger.info(f"Test RMSE: {results['test_rmse']:.4f}")
        logger.info(f"Test MAE: {results['test_mae']:.4f}")
        logger.info(f"Test R²: {results['test_r2']:.4f}")
        logger.info(f"Training time: {train_time:.2f}s")

    return results


def get_feature_importance(model, model_type: str, n_features: int) -> np.ndarray:
    """
    Get feature importance from a trained model.

    Args:
        model: Trained model
        model_type: 'xgboost' or 'metagb'
        n_features: Number of features

    Returns:
        Array of feature importances
    """
    if model_type == 'xgboost':
        return model.feature_importances_
    else:
        # MetaGB doesn't have built-in feature importance
        # Return uniform importance as placeholder
        return np.ones(n_features) / n_features


def calculate_feature_selection_precision(
    importance: np.ndarray,
    relevant_features: List[int],
    top_k: int = 10
) -> float:
    """
    Calculate precision of feature selection.

    Args:
        importance: Feature importance array
        relevant_features: List of truly relevant feature indices
        top_k: Number of top features to consider

    Returns:
        Precision score (0-1)
    """
    top_k_features = np.argsort(importance)[-top_k:][::-1]
    relevant_set = set(relevant_features)
    selected_relevant = sum(1 for f in top_k_features if f in relevant_set)
    precision = selected_relevant / min(top_k, len(relevant_features))
    return precision
