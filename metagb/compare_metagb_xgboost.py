import numpy as np
from metagb_utils import MetaGB
import xgboost as xgb


def generate_synthetic_regression(
    n_samples=4000, n_features=10, noise=0.15, random_state=42
):
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(n_samples, n_features))
    coefs = rng.uniform(-2.0, 2.0, size=n_features)
    non_linear = (
        2.0 * np.sin(x[:, 0] * x[:, 1])
        + 0.5 * np.square(x[:, 2])
        - 1.5 * np.sin(x[:, 3])
    )
    y = x @ coefs + non_linear + rng.normal(scale=noise, size=n_samples)
    return x.astype(np.float64), y.astype(np.float64)


def evaluate_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_test_split_np(x, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    split = int((1.0 - test_size) * x.shape[0])
    train_idx, test_idx = idx[:split], idx[split:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def main():
    x, y = generate_synthetic_regression()
    x_train, x_test, y_train, y_test = train_test_split_np(
        x, y, test_size=0.2, random_state=42
    )

    metagb = MetaGB(
        learning_rate=0.1,
        num_trees=150,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        min_data_in_leaf=20,
        max_depth=5,
        max_bin=64,
        version=0,
        seed=42,
    )
    metagb.fit(x_train, y_train)
    metagb_pred = metagb.predict(x_test)
    metagb_rmse = evaluate_rmse(y_test, metagb_pred)

    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.0,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=42,
    )
    xgb_model.fit(x_train, y_train)
    xgb_pred = xgb_model.predict(x_test)
    xgb_rmse = evaluate_rmse(y_test, xgb_pred)

    print("MetaGB RMSE:", round(metagb_rmse, 4))
    print("XGBoost RMSE:", round(xgb_rmse, 4))


def main_v3():
    """Version 3 example: vary pred_fraction from 0 to 0.8"""
    x, y = generate_synthetic_regression()
    x_train, x_test, y_train, y_test = train_test_split_np(
        x, y, test_size=0.2, random_state=42
    )

    feature_fraction = 0.8
    pred_fractions = [0.0, 0.2, 0.4, 0.6, 0.8]

    print("\n=== Version 3: Varying pred_fraction ===")
    print(f"feature_fraction (cond) = {feature_fraction}")
    print("-" * 40)

    for pred_frac in pred_fractions:
        metagb = MetaGB(
            learning_rate=0.1,
            num_trees=150,
            feature_fraction=feature_fraction,
            pred_feature_fraction=pred_frac,
            bagging_fraction=0.8,
            min_data_in_leaf=20,
            max_depth=5,
            max_bin=64,
            version=3,
            seed=42,
        )
        metagb.fit(x_train, y_train)
        metagb_pred = metagb.predict(x_test)
        rmse = evaluate_rmse(y_test, metagb_pred)
        print(f"pred_fraction={pred_frac:.1f}  RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
    main_v3()

