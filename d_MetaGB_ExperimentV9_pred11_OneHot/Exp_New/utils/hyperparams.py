"""
Hyperparameter Configuration Module
Defines all hyperparameter search spaces and fixed configurations for experiments.
"""

# =============================================================================
# Global Configuration
# =============================================================================

RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

# K-fold cross-validation folds (Experiment 1 only)
CV_FOLDS = 3  # version<=6 is5; version>=7 is 3; edit at 260306

# Categorical variable handling
# Options: 'drop' (delete categorical columns) or 'onehot' (one-hot encoding)
CATEGORICAL_HANDLING = 'onehot'  # Change to 'drop' to remove categorical features

# =============================================================================
# Experiment 1: Dataset Configuration
# =============================================================================

# TabArena regression datasets to use in Experiment 1
## your-dataset-name.csv should be replaced with the actual dataset file names (with .csv extension)
EXP1_DATASETS = [
    ###——————TabArena_Dataset——————###
    #OK
    #"airfoil_self_noise.csv",#ok
    #"Another-Dataset-on-used-Fiat-500.csv",#ok
    #"concrete_compressive_strength.csv", #ok
    #"diamonds.csv", #训练过慢
    #OneHot存在Bug，异常中断
    #"Food_Delivery_Time.csv", #!
    "healthcare_insurance_expenses.csv",#feature_fraction is too small, randomly select 1 feature
    "houses.csv",#OK
    "miami_housing.csv",#OK
    "physiochemical_protein.csv",#OK
    "QSAR_fish_toxicity.csv",#OK
    # "QSAR-TID-11.csv",#训练过慢，Feature1024个
    # "superconductivity.csv",#训练过慢，Feature81个
    "wine_quality.csv"#OK
    
    ###——————Other_Dataset——————###
    
    
    
]

# =============================================================================
# Experiment Repetition Configuration
# =============================================================================

REPEATING_NUMS = 2  # Number of times to repeat experiments 2A and 2B

# =============================================================================
# Early Stopping Configuration
# =============================================================================

EARLY_STOPPING_ROUNDS = 20  # Stop if no improvement for this many rounds
EARLY_STOPPING_MIN_DELTA = 1e-6  # Minimum improvement to be considered as progress

# =============================================================================
# Experiment 1: Grid Search Parameter Spaces
# =============================================================================

###参考###
# Default common parameters (used if no dataset-specific grid is defined)
COMMON_PARAMS_GRID = {
    'learning_rate': [ 0.01, 0.05, 0.1],
    'max_depth': [3, 4,  6,  8],
    'n_estimators': [200,400,600,800],
    'subsample': [ 0.7, 0.8,0.9, 1],
}
# Default MetaGB-specific parameters
METAGB_SPECIFIC_PARAMS = {
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'version': [3],
    'max_bin': [255],
    'min_data_in_leaf': [10],
    'colsample_bytree': [ 0.7, 0.8, 0.9, 1.0],
}
# Default XGBoost-specific parameters
XGBOOST_SPECIFIC_PARAMS = {
    'max_bin': [255],
    'min_child_weight': [10],
    'reg_lambda': [0.0, 1.0],
    'colsample_bytree': [ 0.7, 0.8, 0.9, 1.0],
}


# =============================================================================
# Experiment 1: Dataset-Specific Parameter Grids
# =============================================================================
##实际##
# Shared parameters for all Exp1 datasets
EXP1_SHARED_PARAMS_GRID = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4, 6, 8],
    'n_estimators': [200, 400, 800],
    'max_bin': [255],
}
min_child = [10] #[1,10]; metagb和xgboost的min_child_weight/min_data_in_leaf都固定为10，减少组合数量；

# Dataset-specific parameters (only parameters that vary by dataset)
DATASET_SPECIFIC_COMMON_GRIDS = {
    'airfoil_self_noise': {
        'subsample': [ 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [ 0.7, 0.8, 0.9, 1.0],
    },
    'Another-Dataset-on-used-Fiat-500': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'concrete_compressive_strength': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'diamonds': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'Food_Delivery_Time': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'healthcare_insurance_expenses': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'houses': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'miami_housing': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'physiochemical_protein': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'QSAR_fish_toxicity': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'wine_quality': {
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
}


# XGBoost-specific parameters (only parameters unique to XGBoost)

DATASET_SPECIFIC_XGBOOST_GRIDS = {
    'airfoil_self_noise': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
        
    },
    'Another-Dataset-on-used-Fiat-500': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
    'concrete_compressive_strength': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
    'diamonds': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
    'Food_Delivery_Time': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
    'healthcare_insurance_expenses': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
    'houses': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
    'miami_housing': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
    'physiochemical_protein': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
    'QSAR_fish_toxicity': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
    'wine_quality': {
        'reg_lambda': [0.0, 1.0],
        'min_child_weight': min_child,
    },
}

# MetaGB pred_feature_fraction values to search over
PRED_FEATURE_FRACTION_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# MetaGB-specific parameters (only parameters unique to MetaGB, excluding pred_feature_fraction)
DATASET_SPECIFIC_METAGB_GRIDS = {
    'airfoil_self_noise': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'Another-Dataset-on-used-Fiat-500': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'concrete_compressive_strength': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'diamonds': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'Food_Delivery_Time': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'healthcare_insurance_expenses': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'houses': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'miami_housing': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'physiochemical_protein': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'QSAR_fish_toxicity': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
    'wine_quality': {
        'version': [3],
        'min_data_in_leaf': min_child,
    },
}

# =============================================================================
# Experiment 2: Fixed Parameters
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
PRED_FEATURE_FRACTIONS = [0.0,0.1,0.2,0.4,0.6,0.8,1.0]

# =============================================================================
# Noise Levels for Experiment 2A
# =============================================================================

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0, 5.0]

# =============================================================================
# Experiment 2A: Hyperparameter Grid for Grid Search
# =============================================================================

EXP2A_XGBOOST_PARAM_GRID = {
    'learning_rate': [0.01, 0.05,0.1],
    'max_depth': [3,4,5,6,7],
    'n_estimators': [ 200, 800],
    'subsample': [0.6,0.8,1.0],
    'colsample_bytree': [0.6,0.8,1.0],
    'max_bin': [255],
    'min_child_weight': [10],
    'reg_lambda': [0.0, 1.0],
}

EXP2A_METAGB_PARAM_GRID = {
    'learning_rate': [0.01, 0.05,0.1],
    'max_depth': [3,4,5,6,7],
    'n_estimators': [200, 800],
    'subsample': [0.6,0.8,1.0],
    'colsample_bytree': [0.6,0.8,1.0],
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'version': [3],
    'max_bin': [255],
    'min_data_in_leaf': [10],
}

# =============================================================================
# Experiment 2B: Sparse Interaction Configurations (Outer Loop)
# =============================================================================

# sparse_config: (n_total_features, n_relevant, interaction_order)
#2-order is fixed;
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
    'learning_rate': [ 0.01, 0.05,0.1],
    'max_depth': [3, 4, 5, 6, 7],
    'n_estimators': [ 200,800],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8,1.0],
    'max_bin': [255],
    'min_child_weight': [10],
    'reg_lambda': [0.0, 1.0],
}

EXP2B_METAGB_PARAM_GRID = {
    'learning_rate': [ 0.01,0.05,0.1],
    'max_depth': [3, 4, 5, 6, 7],
    'n_estimators': [ 200,800],
    'subsample': [0.6,0.8, 1.0],
    'colsample_bytree': [0.6, 0.8,1.0],
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'version': [3],
    'max_bin': [255],
    'min_data_in_leaf': [10],
}

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# =============================================================================
# Helper Functions
# =============================================================================
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

def get_xgb_param_grid(dataset_name=None):
    """
    Get the full XGBoost parameter grid for grid search.

    Args:
        dataset_name: Optional dataset name (with or without .csv extension).
                     If provided and dataset-specific grid exists, returns that grid.
                     Otherwise returns default grid.

    Returns:
        dict: Parameter grid for XGBoost
    """
    param_grid = {}

    # Extract dataset name without .csv extension
    if dataset_name:
        dataset_key = dataset_name.replace('.csv', '')

        # Check if dataset-specific grids exist
        if dataset_key in DATASET_SPECIFIC_COMMON_GRIDS:
            # Use shared Exp1 grid + dataset-specific grid + XGBoost-specific grid
            param_grid.update(EXP1_SHARED_PARAMS_GRID)
            param_grid.update(DATASET_SPECIFIC_COMMON_GRIDS[dataset_key])
            param_grid.update(DATASET_SPECIFIC_XGBOOST_GRIDS[dataset_key])
            return param_grid

    # Return default grid if no dataset-specific grid found
    param_grid.update(COMMON_PARAMS_GRID)
    param_grid.update(XGBOOST_SPECIFIC_PARAMS)
    return param_grid


def get_metagb_param_grid(dataset_name=None):
    """
    Get the full MetaGB parameter grid for grid search.

    Args:
        dataset_name: Optional dataset name (with or without .csv extension).
                     If provided and dataset-specific grid exists, returns that grid.
                     Otherwise returns default grid.

    Returns:
        dict: Parameter grid for MetaGB
    """
    param_grid = {}

    # Extract dataset name without .csv extension
    if dataset_name:
        dataset_key = dataset_name.replace('.csv', '')

        # Check if dataset-specific grids exist
        if dataset_key in DATASET_SPECIFIC_COMMON_GRIDS:
            # Use shared Exp1 grid + dataset-specific grid + MetaGB-specific grid
            param_grid.update(EXP1_SHARED_PARAMS_GRID)
            param_grid.update(DATASET_SPECIFIC_COMMON_GRIDS[dataset_key])
            param_grid.update(DATASET_SPECIFIC_METAGB_GRIDS[dataset_key])
            return param_grid

    # Return default grid if no dataset-specific grid found
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


def print_param_grid_info(dataset_name=None):
    """
    Print information about parameter grid sizes.

    Args:
        dataset_name: Optional dataset name to show dataset-specific grid.
                     If None, shows all dataset-specific grids.
    """
    print("=" * 80)
    print("Parameter Grid Information (Experiment 1)")
    print("=" * 80)

    if dataset_name:
        # Show specific dataset grid
        xgb_grid = get_xgb_param_grid(dataset_name)
        metagb_grid = get_metagb_param_grid(dataset_name)

        xgb_combinations = 1
        for values in xgb_grid.values():
            xgb_combinations *= len(values)

        metagb_combinations = 1
        for values in metagb_grid.values():
            metagb_combinations *= len(values)

        print(f"\nDataset: {dataset_name}")
        print(f"\nXGBoost Grid Search ({xgb_combinations} combinations):")
        for k, v in xgb_grid.items():
            print(f"  {k}: {v}")

        print(f"\nMetaGB Grid Search ({metagb_combinations} combinations):")
        for k, v in metagb_grid.items():
            print(f"  {k}: {v}")
    else:
        # Show all dataset-specific grids
        print("\nDataset-Specific Grids:")
        print("-" * 80)

        total_xgb = 0
        total_metagb = 0

        for dataset in EXP1_DATASETS:
            dataset_key = dataset.replace('.csv', '')
            xgb_grid = get_xgb_param_grid(dataset)
            metagb_grid = get_metagb_param_grid(dataset)

            xgb_combinations = 1
            for values in xgb_grid.values():
                xgb_combinations *= len(values)

            metagb_combinations = 1
            for values in metagb_grid.values():
                metagb_combinations *= len(values)

            total_xgb += xgb_combinations
            total_metagb += metagb_combinations

            print(f"\n{dataset_key}:")
            print(f"  XGBoost: {xgb_combinations} combinations")
            print(f"  MetaGB: {metagb_combinations} combinations")

        print("\n" + "-" * 80)
        print(f"Total across all datasets:")
        print(f"  XGBoost: {total_xgb} combinations")
        print(f"  MetaGB: {total_metagb} combinations")
        print(f"  With CV_FOLDS={CV_FOLDS}:")
        print(f"    XGBoost: {total_xgb * CV_FOLDS} trainings")
        print(f"    MetaGB: {total_metagb * CV_FOLDS} trainings")

    print("=" * 80)


if __name__ == "__main__":
    print_param_grid_info()
