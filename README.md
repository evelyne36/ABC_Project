# MetaGB vs XGBoost Comparison Experiments

This project implements a comprehensive comparison between MetaGB and XGBoost on both real and synthetic datasets.

## Project Structure

```
Exp_New/
├── run_all_experiments.py      # Main experiment runner
├── run_experiments.bat         # Windows batch launcher
├── exp1_real_datasets.py       # Experiment 1: Real datasets comparison
├── exp2A_synthetic_nonlinear.py # Experiment 2A: Nonlinear robustness test
├── exp2B_synthetic_interaction.py # Experiment 2B: Feature interaction test
├── utils/
│   ├── __init__.py
│   ├── hyperparams.py          # Hyperparameter configurations
│   ├── data_generators.py      # Synthetic data generators
│   ├── trainer.py              # Model training utilities
│   ├── logger.py               # Logging utilities
│   └── visualizer.py           # Plotting utilities
├── results/                    # Experiment results (CSV files)
├── logs/                       # Experiment logs
└── plots/                      # Generated plots
```

## Experiments Overview

### Experiment 1: Real Datasets Performance Comparison

- **Data Source**: `dataset/tabArena_regression/` (13 regression datasets)
- **Method**: GridSearchCV (K=5) for hyperparameter optimization, choose the average of each hyperparameter combination after CV. 
- **Split**: 80% train / 20% test
- **Output**: `results/exp1/exp1_summary.csv`

### Experiment 2A: Synthetic Nonlinear Robustness Test

- **Data Types**: Square, Sine, Square+Sine, Sigmoid nonlinearities
- **Noise Levels**: 0, 0.1, 0.3, 0.5, 0.6, 1.0, 10.0
- **Method**: Fixed hyperparameters, varying `pred_feature_fraction`
- **Output**: `results/exp2A/exp2A_summary.csv`

### Experiment 2B: Feature Interaction Pattern Test

- **Data Types**:
  - Piecewise Linear (2, 3, 5 regions)
  - Conditional Prediction (5, 7, 10 categories)
  - Sparse Interaction (20,100 features, 5/10 relevant,2-order)
- **Method**: Fixed hyperparameters, varying `pred_feature_fraction`
- **Output**: `results/exp2B/exp2B_summary.csv`

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
| XGBoost | Standard XGBoost with histogram method |
| MetaGB_v0.0 | MetaGB with pred_feature_fraction=0.0 (like standard GB) |
| MetaGB_v0.4 | MetaGB with pred_feature_fraction=0.4 |
| MetaGB_v0.8 | MetaGB with pred_feature_fraction=0.8 |

## Hyperparameter Search Space

### Experiment 1 (Grid Search)

**Common Parameters:**
- learning_rate: [0.01, 0.05, 0.1]
- max_depth: [3, 5, 7]
- n_estimators: [100, 200]
- subsample: [0.8, 1.0]

**MetaGB Specific:**
- pred_feature_fraction: [0.0, 0.4, 0.8]
- version: 3
- max_bin: 64
- min_data_in_leaf: 20

**XGBoost Specific:**
- colsample_bytree: [0.8, 1.0]
- max_bin: 64
- min_child_weight: 20

### Experiment 2 (Fixed Parameters)

```python
FIXED_COMMON_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 200,
    'subsample': 1.0,
    'colsample_bytree': 0.8,
}
```

## Output Files

### Summary CSV Format

**Experiment 1 (`exp1_summary.csv`):**
```
Dataset, Model, Best_LR, Best_Depth, Best_N_Est, Best_Pred_Frac, Test_RMSE, Test_MAE, Test_R2, Train_Time_s
```

**Experiment 2A (`exp2A_summary.csv`):**
```
Nonlinear_Type, Noise_Std, Model, Test_RMSE, Test_MAE, Test_R2, N_Trees_Used, Train_Time_s
```

**Experiment 2B (`exp2B_summary.csv`):**
```
Dataset_Type, Variant, Model, Test_RMSE, Test_MAE, Test_R2, Avg_Tree_Depth, Feature_Selection_Precision
```

### Plot Naming Convention

- Training curves: `{exp}-{dataset}-seed{seed}-{model}-iter{n}-mse.png`
- Robustness curves: `exp2A-{nonlinear_type}-robustness_curve.png`
- Comparison plots: `exp2B-{dataset_type}-tree_depth_comparison.png`

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
cd Exp_New
python test_setup.py
```

## Notes

1. MetaGB requires data to be float64 format
2. The `pred_feature_fraction` parameter is the key differentiator for MetaGB
3. When `pred_feature_fraction=0.0`, MetaGB behaves like standard gradient boosting
4. Higher `pred_feature_fraction` values enable linear leaf predictions
