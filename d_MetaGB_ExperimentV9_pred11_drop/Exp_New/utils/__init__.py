"""
Exp_New Utility Modules
MetaGB vs XGBoost Comparison Experiments

Note: Imports are done lazily to avoid circular import issues.
Import specific modules directly when needed:
    from utils.data_generators import make_regression_square_noise
    from utils.trainer import train_single_model
"""

# Only import modules that don't have external dependencies
from .data_generators import (
    make_regression_square_noise,
    make_regression_sin_noise,
    make_regression_square_sin_noise,
    make_regression_sigmoid_noise,
    make_piecewise_linear,
    make_conditional_prediction,
    make_sparse_interaction
)

from .hyperparams import (
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
    CV_FOLDS,
    COMMON_PARAMS_GRID,
    METAGB_SPECIFIC_PARAMS,
    XGBOOST_SPECIFIC_PARAMS,
    FIXED_COMMON_PARAMS,
    FIXED_METAGB_PARAMS,
    get_xgb_param_grid,
    get_metagb_param_grid
)

from .logger import (
    setup_experiment_logger,
    ExperimentLogger
)

from .visualizer import (
    plot_training_curve,
    plot_noise_robustness,
    plot_training_curves_comparison,
    plot_tree_depth_comparison
)

# Trainer imports are deferred to avoid import issues with MetaGB
# Import directly when needed: from utils.trainer import train_single_model
def __getattr__(name):
    """Lazy import for trainer module functions."""
    trainer_functions = [
        'train_with_grid_search',
        'train_single_model',
        'evaluate_model',
        'evaluate_and_log'
    ]
    if name in trainer_functions:
        from . import trainer
        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
