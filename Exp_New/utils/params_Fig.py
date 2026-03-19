###############################################################################
# Default common parameters (used if no dataset-specific grid is defined)
COMMON_PARAMS_GRID = {
    'learning_rate': [0.005, 0.01, 0.05],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'n_estimators': [200, 400, 600, 800],
    'subsample': [0.6, 0.7,  0.8,  0.9,  1],
}
# Default MetaGB-specific parameters
METAGB_SPECIFIC_PARAMS = {
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'version': [3],
    'max_bin': [64],
    'min_data_in_leaf': [20],
    'colsample_bytree': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
}
# Default XGBoost-specific parameters
XGBOOST_SPECIFIC_PARAMS = {
    'max_bin': [64],
    'min_child_weight': [20],
    'reg_lambda': [0.0, 1.0],
    'colsample_bytree': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
}

###############################################################################
# Dataset-specific grids for Exp1 (key = dataset name without .csv extension)
# 放到公共集合
DATASET_SPECIFIC_COMMON_GRIDS = {
    'airfoil_self_noise': {
        'learning_rate': [ 0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [200, 400, 600, 800],
        'subsample': [0.8, 0.9, 1.0],
        #补上'max_bin': [63,255],
        #补上'colsample_bytree': [0.8, 0.9, 1.0],
    },
    }

###############################################################################
DATASET_SPECIFIC_XGBOOST_GRIDS = {
    'airfoil_self_noise': {
        'max_bin': [63,255],#去掉，因为放到前面了
        'min_child_weight': [5,10],#此处去掉，需要放到前面
        'reg_lambda': [0.0, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],#此处去掉，因为放到前面了
        
    },}

DATASET_SPECIFIC_METAGB_GRIDS = {
    'airfoil_self_noise': {
        'pred_feature_fraction': [0.0, 0.3, 0.6, 1.0],
        'version': [3],
        'max_bin': [64],#此处去掉，因为放到前面了
        'min_data_in_leaf': [5,10],
        'colsample_bytree': [0.8, 0.9, 1.0],#此处去掉，因为放到前面了
    
    },}
    