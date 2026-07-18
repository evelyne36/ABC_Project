# Strong-baseline scope

Retained models: Linear, ExtraTrees, RandomForest, LightGBM, CatBoost and TorchMLP.

RealMLP and TabM are excluded from the migrated runner, dependencies, raw artifacts and paper tables. CatBoost is not rerun: it has 257 successful folds, so comparisons must use common complete dataset cells and state the resulting N.

The “best available baseline” chosen after seeing each test result is an oracle envelope, not an algorithm, and must not be reported as a standalone baseline row.

