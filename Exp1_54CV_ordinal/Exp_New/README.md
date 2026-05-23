# MetaGB vs XGBoost Comparison Experiments

This project implements a comprehensive comparison between MetaGB and XGBoost on both real and synthetic datasets.

## Project Structure

```
Exp_New_Normalization/
├── run_all_experiments.py      # Main experiment runner
├── run_experiments.bat         # Windows batch launcher
├── exp1_real_datasets.py       # Experiment 1: Real datasets comparison (with Early Stopping)
├── exp2A_synthetic_nonlinear.py # Experiment 2A: Nonlinear robustness test (Grid Search + Repetitions)
├── exp2B_synthetic_interaction.py # Experiment 2B: Feature interaction test (Grid Search + Repetitions)
├── utils/
│   ├── __init__.py
│   ├── hyperparams.py          # Hyperparameter configurations (all grids defined here)
│   ├── data_generators.py      # Synthetic data generators
│   ├── trainer.py              # Model training utilities (with Early Stopping)
│   ├── logger.py               # Logging utilities
│   └── visualizer.py           # Plotting utilities (Train/Val RMSE curves)
├── data/                       # Generated datasets (saved for reproducibility)
│   ├── exp2A/                  # Datasets for Experiment 2A
│   └── exp2B/                  # Datasets for Experiment 2B
├── results/                    # Experiment results (CSV files)
├── logs/                       # Experiment logs
└── plots/                      # Generated plots (including training curves)
```

## Key Features

### Experiment 1: Early Stopping
- Prevents overfitting by monitoring validation loss
- Configurable patience and minimum delta in `hyperparams.py`
- Training curves saved showing Train RMSE vs Validation RMSE

### Experiments 2A & 2B: Grid Search + Multiple Repetitions
- All hyperparameters defined centrally in `hyperparams.py`
- `sparse_config` as outer loop for Experiment 2B
- XGBoost and MetaGB hyperparameter grids as inner loop
- `REPEATING_NUMS` repetitions with fresh random datasets each time
- Datasets saved to `data/` directory for reproducibility

### Visualization
- Real-time Train RMSE vs Validation RMSE curves for all experiments
- Early stopping point marked on curves (Experiment 1)
- Helps identify overfitting visually

## Experiments Overview

### Experiment 1: Real Datasets Performance Comparison

- **Data Source**: `dataset/tabArena_regression/` (13 regression datasets)
- **Method**: GridSearchCV (K=5) for hyperparameter optimization
- **Early Stopping**: Enabled (patience=50 rounds)
- **Split**: 80% train (64% train + 16% val) / 20% test
- **Output**: `results/exp1/exp1_summary.csv`
- **Plots**: Training curves with early stopping markers

### Experiment 2A: Synthetic Nonlinear Robustness Test

- **Data Types**: Square, Sine, Square+Sine, Sigmoid nonlinearities
- **Noise Levels**: 0, 0.1, 0.3, 0.5, 0.8, 1.0, 10.0
- **Method**: Grid search for both XGBoost and MetaGB
- **Repetitions**: `REPEATING_NUMS` times (default: 5)
- **Dataset Generation**: Fresh random dataset for each repetition
- **Output**: `results/exp2A/exp2A_summary.csv`
- **Plots**: Training curves for each model/noise/repetition

### Experiment 2B: Feature Interaction Pattern Test

- **Outer Loop**: `sparse_config` from `hyperparams.py`
  - (20, 5, 2): 20 features, 5 relevant, 2-order interaction
  - (20, 10, 2): 20 features, 10 relevant, 2-order interaction
  - (50, 5, 2): 50 features, 5 relevant, 2-order interaction
  - (50, 10, 2): 50 features, 10 relevant, 2-order interaction
- **Inner Loop**: Hyperparameter grid search for XGBoost and MetaGB
- **Repetitions**: `REPEATING_NUMS` times (default: 5)
- **Dataset Generation**: Fresh random dataset for each repetition
- **Output**: `results/exp2B/exp2B_summary.csv`
- **Plots**: Training curves, feature importance comparison

## Hyperparameter Configuration

All hyperparameters are centralized in `utils/hyperparams.py`:

```python
# Global settings
REPEATING_NUMS = 5              # Number of repetitions for Exp 2A/2B
EARLY_STOPPING_ROUNDS = 50      # Early stopping patience
EARLY_STOPPING_MIN_DELTA = 1e-6 # Minimum improvement threshold

# Experiment 2A Grid Search
EXP2A_XGBOOST_PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    ...
}

EXP2A_METAGB_PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ...
}

# Experiment 2B Sparse Configurations (Outer Loop)
SPARSE_CONFIGS = [
    (20, 5, 2),   # (n_total_features, n_relevant, interaction_order)
    (20, 10, 2),
    (50, 5, 2),
    (50, 10, 2),
]

# Experiment 2B Grid Search (Inner Loop)
EXP2B_XGBOOST_PARAM_GRID = {...}
EXP2B_METAGB_PARAM_GRID = {...}
```

## Usage

### Run All Experiments

```bash
python run_all_experiments.py --exp all --seed 42
```

### Run Specific Experiments

```bash
# Experiment 1 only
python run_all_experiments.py --exp 1

# Experiment 2A only
python run_all_experiments.py --exp 2a

# Experiment 2B only
python run_all_experiments.py --exp 2b

# Both synthetic experiments
python run_all_experiments.py --exp 2

# With custom seed
python run_all_experiments.py --exp all --seed 123
```

### Windows Batch Script

```batch
run_experiments.bat --exp all --seed 42
```

## Models Compared

| Model | Description |
|-------|-------------|
| XGBoost | Standard XGBoost with histogram method (grid search optimized) |
| MetaGB | MetaGB with grid search optimized pred_feature_fraction |

## Output Files

### Summary CSV Format

**Experiment 1 (`exp1_summary.csv`):**
```
Dataset, Model, Best_LR, Best_Depth, Best_N_Est, Best_Pred_Frac, CV_RMSE, Test_RMSE, Test_MAE, Test_R2, Train_Time_s, Early_Stopped, Early_Stop_Round
```

**Experiment 2A (`exp2A_summary.csv`):**
```
Nonlinear_Type, Noise_Std, Repetition, Model, Best_LR, Best_Depth, Best_N_Est, Test_RMSE, Test_MAE, Test_R2, Val_RMSE, Train_Time_s
```

**Experiment 2B (`exp2B_summary.csv`):**
```
Dataset_Type, Variant, Repetition, Model, Best_LR, Best_Depth, Best_N_Est, Best_Pred_Frac, Test_RMSE, Test_MAE, Test_R2, Val_RMSE, Avg_Tree_Depth, Train_Time_s, Feature_Selection_Precision
```

### Plot Naming Convention

- Training curves: `{exp}-{dataset}-seed{seed}-{model}-iter{n}-rmse.png`
- Early stopping marker shown on Experiment 1 training curves
- Robustness curves: `exp2A-{nonlinear_type}-robustness_curve.png`
- Comparison plots: `exp2B-{dataset_type}-tree_depth_comparison.png`
- Feature importance: `exp2B-{dataset_name}-feature_importance.png`

### Saved Datasets

Generated datasets are saved for reproducibility:
- `data/exp2A/{nonlinear_type}_noise{noise}_rep{rep}_seed{seed}.npz`
- `data/exp2B/{sparse_config}_rep{rep}_seed{seed}.npz`

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- cffi (for building MetaGB)
- MetaGB (from ABC_Project/metagb)

## Prerequisites: Build MetaGB

Before running experiments, you must build the MetaGB CFFI module:

```bash
# Navigate to the MetaGB directory
cd ABC_Project/metagb

# Build the CFFI module (compiles C code into Python extension)
python build_metagb.py

# This creates libmetagb.cpython-*.so (Linux) or libmetagb.*.pyd (Windows)
```

After building, verify the setup:

```bash
cd Exp_New_Normalization
python test_setup.py
```

## Notes

1. MetaGB requires data to be float64 format
2. The `pred_feature_fraction` parameter is the key differentiator for MetaGB
3. When `pred_feature_fraction=0.0`, MetaGB behaves like standard gradient boosting
4. Higher `pred_feature_fraction` values enable linear leaf predictions
5. Early stopping in Experiment 1 helps prevent overfitting
6. Training curves (Train RMSE vs Val RMSE) help visualize overfitting
7. Multiple repetitions in Exp 2A/2B provide statistical robustness
8. All hyperparameters are centralized in `utils/hyperparams.py` for easy modification
