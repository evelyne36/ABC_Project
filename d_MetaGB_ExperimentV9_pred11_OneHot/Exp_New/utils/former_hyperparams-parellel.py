"""
Hyperparameter Configuration Module
Defines all hyperparameter search spaces and fixed configurations for experiments.
"""

# =============================================================================
# Global Configuration
# =============================================================================

RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
CV_FOLDS = 3  # <=v6=5,v7=3; K-fold cross-validation folds (Experiment 1 only)

# =============================================================================
# Experiment 1: Dataset Configuration
# =============================================================================

# TabArena regression datasets to use in Experiment 1
EXP1_DATASETS = [
    "airfoil_self_noise.csv",
    "Another-Dataset-on-used-Fiat-500.csv",
    "concrete_compressive_strength.csv",
    "healthcare_insurance_expenses.csv",
    "QSAR_fish_toxicity.csv",
    "wine_quality.csv"
]

# =============================================================================
# Experiment Repetition Configuration
# =============================================================================

REPEATING_NUMS = 2  # Number of times to repeat experiments 2A and 2B

# =============================================================================
# Early Stopping Configuration
# =============================================================================

EARLY_STOPPING_ROUNDS = 50  # Stop if no improvement for this many rounds
EARLY_STOPPING_MIN_DELTA = 1e-6  # Minimum improvement to be considered as progress

# =============================================================================
# Experiment 1: Grid Search Parameter Spaces
# =============================================================================

# Common parameters shared by XGBoost and MetaGB
# COMMON_PARAMS_GRID = {
#     'learning_rate': [0.01, 0.05, 0.1],      # 3 values
#     'max_depth': [3, 5, 10],                   # 3 values
#     'n_estimators': [20, 50, 100],               # 2 values, reduced for faster grid search
#     'subsample': [0.7, 1.0],                  # 2 values
# }
# COMMON_PARAMS_GRID = {
#     'learning_rate': [0.01, 0.05],      # 3 values
#     'max_depth': [3, 6, 10],                   # 3 values
#     'n_estimators': [20, 60, 100],               # 2 values, reduced for faster grid search
#     'subsample': [0.7, 0.9],                  # 2 values
# }

COMMON_PARAMS_GRID = {
    'learning_rate': [0.005, 0.01, 0.05],      # 3 values
    'max_depth': [3,4,5,6,7],                   # 3 values 3-5-10 12去掉
    'n_estimators': [200, 800],       # 2 values, reduced for faster grid search
    'subsample': [0.6,0.8,1],                  # 2 values; 0.8-1.0
}

# MetaGB-specific parameters
METAGB_SPECIFIC_PARAMS = {
    'pred_feature_fraction': [0.0, 0.2, 0.4,0.6,0.8,1.0],  # 5 values - key MetaGB parameter1],  # 5 values - key MetaGB parameter
    'version': [3],                             # Fixed: use version 3
    'max_bin': [255],                            # Fixed
    'min_data_in_leaf': [20],                   # Fixed 默认20
    'colsample_bytree': [0.6,0.8,1.0],            # 细化 2 values 0.8-1.0 - feature_fraction in MetaGB; 过小则会加大单棵树选不到任何特征的选不中的概率（1-col)^n_features，过大则会增加过拟合风险
}

# XGBoost-specific parameters
XGBOOST_SPECIFIC_PARAMS = {
    'max_bin': [255],                            # Fixed for histogram method
    'min_child_weight': [10],                   # Equivalent to min_data_in_leaf 默认20
    'reg_lambda': [0.0,1.0],                        # L2 regularization
    'colsample_bytree': [0.6,0.8,1.0],            # 2 values 0.8-1.0 - feature_fraction in XGBoost; 过小则会加大单棵树选不到任何特征的选不中的概率（1-col)^n_features，过大则会增加过拟合风险
}

# =============================================================================
# Experiment 2: Fixed Parameters (No Grid Search)暂时弃用
# =============================================================================

FIXED_COMMON_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 8,
    'n_estimators': 400,
    'subsample': 0.8, #1.0
    'colsample_bytree': 0.8,#0.8
}

FIXED_METAGB_PARAMS = {
    'version': 3,
    'max_bin': 64,#64
    'min_data_in_leaf': 20,#20
}

FIXED_XGBOOST_PARAMS = {
    'max_bin': 64,#64
    'min_child_weight': 20,#20
    'reg_lambda': 1.0,#0
}

# Experiment 2: pred_feature_fraction variants to test
#PRED_FEATURE_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8]
PRED_FEATURE_FRACTIONS = [0.0,0.1,0.2,0.3, 0.4,0.6,0.8,1.0]

# =============================================================================
# Noise Levels for Experiment 2A
# =============================================================================

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0, 5.0]

# =============================================================================
# Experiment 2A: Hyperparameter Grid for Grid Search
# =============================================================================

EXP2A_XGBOOST_PARAM_GRID = {
    'learning_rate': [0.005, 0.01, 0.05],
    'max_depth': [3,4,5,6,7,8],
    'n_estimators': [ 100, 200, 400,800],
    'subsample': [0.8,0.9, 1.0],
    'colsample_bytree': [0.8,0.9,1.0],
    'max_bin': [64],
    'min_child_weight': [20],
    'reg_lambda': [0.0, 1.0],
}

EXP2A_METAGB_PARAM_GRID = {
    'learning_rate': [0.005, 0.01, 0.05],
    'max_depth': [3,4,5,6,7,8],
    'n_estimators': [ 100, 200, 400,800],
    'subsample': [0.8,0.9, 1.0],
    'colsample_bytree': [0.8,0.9,1.0],
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'version': [3],
    'max_bin': [64],
    'min_data_in_leaf': [20],
}

# =============================================================================
# Experiment 2B: Sparse Interaction Configurations (Outer Loop)
# =============================================================================

# sparse_config: (n_total_features, n_relevant, interaction_order)
SPARSE_CONFIGS = [
    (20, 5, 2),   # 20 features, 5 relevant, 2-order interaction
    (20, 10, 2),  # 20 features, 10 relevant, 2-order interaction
    (50, 5, 2),   # 50 features, 5 relevant, 2-order interaction
    (50, 10, 2),  # 50 features, 10 relevant, 2-order interaction
]

# =============================================================================
# Experiment 2B: Hyperparameter Grid for Grid Search
# =============================================================================

EXP2B_XGBOOST_PARAM_GRID = {
    'learning_rate': [0.005, 0.01, 0.05],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'n_estimators': [ 100, 200, 400,800],
    'subsample': [0.8,0.9, 1.0],
    'colsample_bytree': [0.8,0.9,1.0],
    'max_bin': [64],
    'min_child_weight': [20],
    'reg_lambda': [0.0, 1.0],
}

EXP2B_METAGB_PARAM_GRID = {
    'learning_rate': [0.001, 0.005, 0.01],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'n_estimators': [ 100, 200, 400,800],
    'subsample': [0.8,0.9, 1.0],
    'colsample_bytree': [0.8,0.9,1.0],
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'version': [3],
    'max_bin': [64],
    'min_data_in_leaf': [20],
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_xgb_param_grid():
    """
    Get the full XGBoost parameter grid for grid search.

    Returns:
        dict: Parameter grid for XGBoost
    """
    param_grid = {}
    param_grid.update(COMMON_PARAMS_GRID)
    param_grid.update(XGBOOST_SPECIFIC_PARAMS)
    return param_grid


def get_metagb_param_grid():
    """
    Get the full MetaGB parameter grid for grid search.

    Returns:
        dict: Parameter grid for MetaGB
    """
    param_grid = {}
    param_grid.update(COMMON_PARAMS_GRID)
    param_grid.update(METAGB_SPECIFIC_PARAMS)
    return param_grid


def get_fixed_xgb_params():
    """
    Get fixed XGBoost parameters for Experiment 2.

    Returns:
        dict: Fixed parameters for XGBoost
    """
    params = {}
    params.update(FIXED_COMMON_PARAMS)
    params.update(FIXED_XGBOOST_PARAMS)
    return params


def get_fixed_metagb_params(pred_feature_fraction=0.0):
    """
    Get fixed MetaGB parameters for Experiment 2.

    Args:
        pred_feature_fraction: The pred_feature_fraction value to use

    Returns:
        dict: Fixed parameters for MetaGB
    """
    params = {}
    params.update(FIXED_COMMON_PARAMS)
    params.update(FIXED_METAGB_PARAMS)
    params['pred_feature_fraction'] = pred_feature_fraction
    return params


def get_model_variants_exp2():
    """
    Get all model variants for Experiment 2.

    Returns:
        list: List of (model_name, model_type, params) tuples
    """
    variants = []

    # XGBoost
    variants.append(('XGBoost', 'xgboost', get_fixed_xgb_params()))

    # MetaGB variants with different pred_feature_fraction
    for pff in PRED_FEATURE_FRACTIONS:
        name = f'MetaGB_v{pff}'
        variants.append((name, 'metagb_v3', get_fixed_metagb_params(pff)))

    return variants


def get_exp2a_xgb_param_grid():
    """
    Get XGBoost parameter grid for Experiment 2A grid search.

    Returns:
        dict: Parameter grid for XGBoost in Exp2A
    """
    return EXP2A_XGBOOST_PARAM_GRID.copy()


def get_exp2a_metagb_param_grid():
    """
    Get MetaGB parameter grid for Experiment 2A grid search.

    Returns:
        dict: Parameter grid for MetaGB in Exp2A
    """
    return EXP2A_METAGB_PARAM_GRID.copy()


def get_exp2b_xgb_param_grid():
    """
    Get XGBoost parameter grid for Experiment 2B grid search.

    Returns:
        dict: Parameter grid for XGBoost in Exp2B
    """
    return EXP2B_XGBOOST_PARAM_GRID.copy()


def get_exp2b_metagb_param_grid():
    """
    Get MetaGB parameter grid for Experiment 2B grid search.

    Returns:
        dict: Parameter grid for MetaGB in Exp2B
    """
    return EXP2B_METAGB_PARAM_GRID.copy()


def get_sparse_configs():
    """
    Get sparse interaction configurations for Experiment 2B outer loop.

    Returns:
        list: List of (n_total_features, n_relevant, interaction_order) tuples
    """
    return SPARSE_CONFIGS.copy()


def print_param_grid_info():
    """Print information about parameter grid sizes."""
    xgb_grid = get_xgb_param_grid()
    metagb_grid = get_metagb_param_grid()

    xgb_combinations = 1
    for values in xgb_grid.values():
        xgb_combinations *= len(values)

    metagb_combinations = 1
    for values in metagb_grid.values():
        metagb_combinations *= len(values)

    print("=" * 60)
    print("Parameter Grid Information")
    print("=" * 60)
    print(f"\nXGBoost Grid Search:")
    for k, v in xgb_grid.items():
        print(f"  {k}: {v}")
    print(f"  Total combinations: {xgb_combinations}")

    print(f"\nMetaGB Grid Search:")
    for k, v in metagb_grid.items():
        print(f"  {k}: {v}")
    print(f"  Total combinations: {metagb_combinations}")
    print("=" * 60)


if __name__ == "__main__":
    print_param_grid_info()
