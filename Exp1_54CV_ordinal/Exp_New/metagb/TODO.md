# ABC_Project

A Generalized Gradient Boosting Library that extends XGBoost by decoupling conditional (splitting) and predictive (leaf) features.

## Verification Plan: MetaGB vs XGBoost

This document outlines a systematic plan to verify that the generalized gradient boosting (MetaGB) can outperform standard XGBoost on appropriate tasks.

---

### Phase 1: Artificial Datasets

**Goal**: Design synthetic datasets where MetaGB's architectural advantages should theoretically help.

#### 1.1 XOR-Style Interaction Datasets
- **Rationale**: MetaGB can learn `β·x` at leaves instead of constants, potentially handling feature interactions more efficiently
- **Dataset**: 2D/3D XOR patterns where classification depends on feature products
- **Hypothesis**: MetaGB with appropriate `pred_feature_fraction` should require shallower trees

#### 1.2 Linear-in-Leaves Datasets
- **Rationale**: When true model has linear functions at decision boundaries
- **Dataset**: Piecewise linear regression where each region has a different slope
- **Example**: `y = x1 * x2` when `x1 > 0`, else `y = -x1 * x2`
- **Hypothesis**: MetaGB should achieve lower error with fewer trees

#### 1.3 Conditional-Predictive Separation Datasets
- **Rationale**: When some features are better for splitting vs prediction
- **Dataset**: Generate data where categorical features define regions, continuous features define slopes within regions
- **Example**: Day-of-week determines regime, numeric features determine prediction magnitude
- **Hypothesis**: MetaGB with explicit feature separation should excel

#### 1.4 High-Dimensional Sparse Interaction Datasets
- **Rationale**: Many irrelevant features, few important interactions
- **Dataset**: 100+ features, only 5-10 relevant, with multiplicative interactions
- **Hypothesis**: MetaGB's separate feature sampling may find interactions more efficiently

#### 1.5 Noise Robustness Tests
- **Dataset**: Clean signal + varying levels of noise features
- **Hypothesis**: MetaGB's linear leaves may regularize differently than constant leaves

---

### Phase 2: Real Datasets from TabArena

**Goal**: Evaluate on diverse, realistic tabular datasets from the TabArena benchmark.

**Source**: [TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2505.22252)

#### 2.1 Dataset Selection Criteria
Select datasets with properties where MetaGB may have advantages:
- **Mixed feature types**: Categorical (for conditioning) + continuous (for prediction)
- **Known feature interactions**: Domains where interactions matter (e.g., finance, healthcare)
- **Large feature sets**: Where feature selection at different tree components helps
- **Regression tasks**: Where linear leaf predictions are meaningful

#### 2.2 Candidate Datasets
| Dataset | Domain | Size | Features | Task | Why Relevant |
|---------|--------|------|----------|------|--------------|
| California Housing | Real Estate | 20K | 8 | Regression | Baseline, existing scripts |
| Adult/Census Income | Demographics | 48K | 14 | Classification | Mixed types |
| Covertype | Ecology | 581K | 54 | Multi-class | High-dim, interactions |
| Higgs | Physics | 11M | 28 | Binary | Feature interactions |
| NYC Taxi | Transportation | Large | Mixed | Regression | Temporal conditioning |
| Medical datasets | Healthcare | Varies | Mixed | Various | Domain knowledge separation |

#### 2.3 Evaluation Protocol
- Use official train/test splits when available
- Report mean ± std across multiple random seeds
- Track both accuracy/RMSE and computational cost

#### 2.4 Data Preprocessing: Normalization

**Important**: For realistic datasets, feature scale matters significantly for MetaGB's generalized leaf predictions.

**Why normalization matters**:
- MetaGB computes leaf predictions as `β·x` where `x` is a feature value
- If features have vastly different scales (e.g., income in thousands vs age in decades), the learned `β` coefficients will be dominated by large-scale features
- This can hurt both optimization stability and predictive performance

**Recommended preprocessing**:
```python
# Option 1: StandardScaler (zero mean, unit variance)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Option 2: MinMaxScaler (scale to [0, 1])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

# Option 3: RobustScaler (robust to outliers)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Best practices**:
- Always examine feature distributions before training (`df.describe()`, histograms)
- Fit scaler on training data only, then transform validation/test sets
- Consider log-transform for heavily skewed features before scaling
- Document which normalization was used for reproducibility
- Compare results with and without normalization to quantify impact

#### 2.5 Feature Type Analysis: Conditional vs Predictive

**Important**: For real datasets, understanding which features should be used for conditioning (splitting) vs prediction (leaf values) is crucial for MetaGB's effectiveness.

**Feature type guidelines**:

| Feature Type | Better For | Rationale |
|--------------|------------|-----------|
| **Categorical / Discrete** | Conditional | Natural for splitting; creates distinct regimes |
| **Continuous / Numeric** | Predictive | Linear relationships captured by `β·x` |
| **Low cardinality** | Conditional | Few unique values → clean splits |
| **High cardinality** | Predictive | Many values → better as linear predictors |
| **Domain "regime" indicators** | Conditional | E.g., region, category, time period |
| **Domain "magnitude" indicators** | Predictive | E.g., amount, count, measurement |

**Examples by domain**:

| Domain | Conditional Features | Predictive Features |
|--------|---------------------|---------------------|
| **E-commerce** | Product category, user segment, device type | Price, quantity, time on page |
| **Finance** | Account type, region, risk tier | Transaction amount, balance, income |
| **Healthcare** | Diagnosis code, department, age group | Lab values, vitals, dosage |
| **Time series** | Day of week, month, holiday flag | Lag values, rolling averages |

**Analysis workflow**:
```python
import pandas as pd

# 1. Examine feature types and cardinality
for col in df.columns:
    n_unique = df[col].nunique()
    dtype = df[col].dtype
    print(f"{col}: {dtype}, {n_unique} unique values")

# 2. Categorize features
conditional_candidates = []  # Low cardinality, categorical
predictive_candidates = []   # High cardinality, continuous

for col in df.columns:
    if df[col].dtype == 'object' or df[col].nunique() < 20:
        conditional_candidates.append(col)
    else:
        predictive_candidates.append(col)

# 3. Use domain knowledge to refine
# - Move "regime" features to conditional
# - Move "magnitude" features to predictive
```

**Experimentation strategy**:
- Start with automatic categorization based on cardinality/type
- Use `cond_pred_share=0` to allow independent sampling of both sets
- Experiment with Version 2's `cond_prior_features` to explicitly specify conditional features
- Compare against `cond_pred_share=1` (predictive derived from conditional) as ablation

---

### Phase 3: Systematic Experiments

**Goal**: Fair, reproducible comparison with controlled and extensive hyperparameter tuning.

#### 3.0 Prerequisite: Hyperparameter Audit

**Critical**: Before running experiments, thoroughly understand ALL hyperparameters in both XGBoost and MetaGB. Do not take defaults or assumptions for granted.

**XGBoost hyperparameters to audit**:
| Parameter | Default | Impact | Notes |
|-----------|---------|--------|-------|
| `max_depth` | 6 | Tree complexity | Deeper trees = more capacity, more overfitting risk |
| `learning_rate` (eta) | 0.3 | Step size | Lower = more trees needed, often better generalization |
| `n_estimators` | 100 | Number of trees | More trees with lower LR often helps |
| `subsample` | 1.0 | Row sampling | <1.0 adds regularization |
| `colsample_bytree` | 1.0 | Column sampling per tree | Feature randomization |
| `colsample_bylevel` | 1.0 | Column sampling per level | Often overlooked |
| `colsample_bynode` | 1.0 | Column sampling per split | Most granular |
| `reg_lambda` (L2) | 1.0 | L2 regularization on weights | **XGBoost default is 1.0, not 0!** |
| `reg_alpha` (L1) | 0.0 | L1 regularization | Sparsity inducing |
| `min_child_weight` | 1.0 | Minimum sum of hessian in child | Controls split creation |
| `gamma` | 0.0 | Minimum loss reduction for split | Pruning parameter |
| `max_delta_step` | 0 | Maximum delta step | Useful for imbalanced data |
| `tree_method` | 'auto' | Algorithm | 'hist' vs 'exact' can differ |
| `grow_policy` | 'depthwise' | How to grow trees | 'lossguide' is leaf-wise |

**MetaGB hyperparameters to audit**:
| Parameter | Default | Impact | Notes |
|-----------|---------|--------|-------|
| `max_depth` | ? | Tree complexity | Verify same meaning as XGBoost |
| `learning_rate` | ? | Step size | Check if applied same way |
| `n_estimators` | ? | Number of trees | - |
| `subsample` | ? | Row sampling | Verify implementation matches XGBoost |
| `feature_fraction` | ? | Column sampling | **Is this per tree, level, or node?** |
| `reg_lambda` | ? | L2 regularization | **Check default! XGBoost uses 1.0** |
| `min_child_weight` | ? | Minimum hessian sum | Verify formula matches |
| `pred_feature_fraction` | ? | Predictive feature sampling | MetaGB-specific |
| `cond_pred_share` | ? | Cond/pred feature relationship | MetaGB-specific |
| `leaf_only_use_bias` | ? | Bias-only leaves | MetaGB-specific |
| `n_bins` | ? | Histogram bins | Check default and impact |

**Verification checklist**:
- [ ] Read XGBoost documentation for each parameter
- [ ] Read MetaGB source code to verify parameter meanings
- [ ] **Check default values** - are MetaGB defaults aligned with XGBoost?
- [ ] **Check parameter interactions** - does `feature_fraction` interact with `pred_feature_fraction`?
- [ ] Run ablation: vary one parameter at a time, observe impact
- [ ] Document any discrepancies between XGBoost and MetaGB parameter behavior

**Common pitfalls to avoid**:
1. **Lambda default mismatch**: XGBoost defaults `reg_lambda=1.0`, many implementations default to 0
2. **Column sampling scope**: XGBoost has 3 levels (bytree, bylevel, bynode); which does MetaGB use?
3. **Histogram binning**: Different bin counts can affect results
4. **Tree growth strategy**: Depth-wise vs leaf-wise can produce very different trees
5. **Gradient/Hessian computation**: Verify same formulas are used

**Recommended audit process**:
```python
# 1. Print all defaults
import xgboost as xgb
print(xgb.XGBRegressor().get_params())

# 2. Compare with MetaGB defaults
# Read metagb_utils.py and metagb.c for default values

# 3. Sanity check: identical params should give identical results
# Train XGBoost and MetaGB v0 with EXACT same params
# Compare predictions - should be very close if implementations match
```

#### 3.1 Approach A: Controlled Hyperparameter Comparison

Fix shared hyperparameters, vary only method-specific ones:

**Fixed Parameters** (same for all methods):
```python
shared_params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [0.0, 1.0],
    'min_child_weight': [1, 5, 10]
}
```

**Method-Specific Parameters**:
```python
# XGBoost: No additional parameters

# MetaGB Version 0 (baseline): No additional parameters

# MetaGB Version 3 (generalized):
metagb_v3_params = {
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'cond_pred_share': [0, 1],
    'leaf_only_use_bias': [0, 1]  # 1 reduces to standard GB
}
```

#### 3.2 Approach B: Extensive Tuning via Cross-Validation

**Protocol**:
1. 5-fold or 10-fold cross-validation on training set
2. Grid search or Bayesian optimization (Optuna)
3. Select best hyperparameters per method independently
4. Evaluate on held-out test set

**Search Space**:
- XGBoost: ~1000 configurations
- MetaGB v0: ~1000 configurations (same as XGBoost)
- MetaGB v3: ~3000 configurations (additional pred_feature_fraction variants)

**Comparison Metrics**:
| Metric | Purpose |
|--------|---------|
| Best CV score | In-sample performance |
| Test score | Generalization |
| Training time | Computational cost |
| Number of trees at convergence | Model complexity |

#### 3.3 Statistical Significance
- Paired t-tests or Wilcoxon signed-rank tests across datasets
- Report win/tie/loss counts
- Use critical difference diagrams for multi-dataset comparison

---

### Phase 4: Baselines

**Goal**: Ensure fair comparison by using both XGBoost and MetaGB v0 as baselines.

#### 4.1 Baseline Methods

| Method | Implementation | Notes |
|--------|----------------|-------|
| **XGBoost** | `xgboost` package | Industry standard |
| **MetaGB v0** | `metagb.train_original()` | Our XGBoost-equivalent implementation |
| **MetaGB v3** | `metagb.train_simple()` | Generalized version |

#### 4.2 Why Two Baselines?

1. **XGBoost baseline**: Ensures we compare against the optimized, widely-used implementation
2. **MetaGB v0 baseline**: Controls for implementation differences (C code, tree structure, etc.)

If MetaGB v3 beats MetaGB v0 but not XGBoost, it suggests the generalization helps but implementation needs optimization.
If MetaGB v3 beats both, the generalization provides genuine algorithmic improvement.

#### 4.3 Validation of Baseline Equivalence
- First verify MetaGB v0 ≈ XGBoost on standard datasets
- Document any performance gaps and their causes
- Use MetaGB v0 as the "fair" baseline for ablation studies

---

### Experiment Execution Checklist

- [ ] **Phase 1**: Create synthetic dataset generators
  - [ ] XOR interaction datasets
  - [ ] Piecewise linear datasets
  - [ ] Conditional-predictive separation datasets
  - [ ] High-dimensional sparse datasets
  - [ ] Noise robustness datasets

- [ ] **Phase 2**: Set up TabArena datasets
  - [ ] Download and preprocess selected datasets
  - [ ] Implement data loading utilities
  - [ ] Verify train/test splits

- [ ] **Phase 3**: Implement experiment framework
  - [ ] Controlled hyperparameter comparison script
  - [ ] Cross-validation grid search script
  - [ ] Optuna Bayesian optimization integration
  - [ ] Results logging and visualization

- [ ] **Phase 4**: Baseline validation
  - [ ] Verify MetaGB v0 ≈ XGBoost equivalence
  - [ ] Document any discrepancies
  - [ ] Set up three-way comparison pipeline

---

### Expected Deliverables

1. **Synthetic dataset analysis**: Identify scenarios where MetaGB theoretically and empirically wins
2. **TabArena benchmark results**: Performance comparison table across real datasets
3. **Hyperparameter sensitivity analysis**: How does `pred_feature_fraction` affect performance?
4. **Statistical summary**: Win/tie/loss counts with significance tests
5. **Recommendations**: Guidelines for when to use MetaGB vs XGBoost