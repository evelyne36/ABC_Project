from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "metagb_preflight_adapter" / "outputs" / "preflight_dataset_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "metagb_preflight_adapter" / "outputs"

BASE_FEATURES = [
    "raw_rows",
    "raw_features",
    "raw_numeric_fraction",
    "raw_categorical_fraction",
    "raw_missing_fraction",
    "processed_rows",
    "processed_features",
    "onehot_expansion_ratio",
    "feature_row_ratio",
    "zero_fraction_processed",
    "feature_scale_cv",
    "n_classes",
    "class_imbalance",
    "global_abs_corr_max",
    "global_abs_corr_top5_mean",
    "local_abs_corr_max_mean",
    "local_linearity_lift",
    "decoupled_leaf_fraction",
    "tree_leaf_count",
    "projected_gradient_excess_mean",
    "projected_gradient_lift_mean",
    "predictive_score_concentration_mean",
]

CATEGORICAL_FEATURES = ["task_type", "categorical_handling"]


def median_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def safe_metric(fn: Any, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_pred))
    except ValueError:
        return math.nan


def safe_divide(num: float, den: float, default: float = math.nan) -> float:
    if den == 0 or not math.isfinite(den):
        return default
    return num / den


def design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available = [col for col in BASE_FEATURES if col in df.columns]
    X = df[available].copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].fillna("missing"), prefix=col, dtype=float)
            X = pd.concat([X, dummies], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, list(X.columns)


def constant_probability(train_y: np.ndarray) -> float:
    if train_y.size == 0:
        return 0.5
    return float(np.clip(np.mean(train_y), 1e-6, 1.0 - 1e-6))


def fit_predict_classifier(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, c_value: float) -> tuple[np.ndarray, str]:
    if len(np.unique(y_train)) < 2:
        return np.full(X_test.shape[0], constant_probability(y_train)), "class-prior"
    model = Pipeline(
        steps=[
            ("imputer", median_imputer()),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1], "logistic-l2"


def fit_predict_regressor(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, alpha: float) -> tuple[np.ndarray, str]:
    if y_train.size == 0:
        return np.zeros(X_test.shape[0], dtype=float), "zero-prior"
    if np.nanstd(y_train) <= 1e-12:
        return np.full(X_test.shape[0], float(np.nanmean(y_train))), "mean-prior"
    model = Pipeline(
        steps=[
            ("imputer", median_imputer()),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=alpha)),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_test), "ridge"


def grouped_leave_one_dataset_out(df: pd.DataFrame, X: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_values = df["dataset"].astype(str).unique()
    target_class = df["mgb_worth_it"].astype(bool).to_numpy()
    target_gain = df["mgb_relative_improvement"].astype(float).to_numpy()
    target_runtime = np.log(df["runtime_ratio"].astype(float).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    for dataset in group_values:
        test_mask = df["dataset"].astype(str).eq(dataset).to_numpy()
        train_mask = ~test_mask
        prob, clf_kind = fit_predict_classifier(X.loc[train_mask], target_class[train_mask].astype(int), X.loc[test_mask], args.classifier_c)
        gain_pred, gain_kind = fit_predict_regressor(X.loc[train_mask], target_gain[train_mask], X.loc[test_mask], args.ridge_alpha)
        runtime_pred, runtime_kind = fit_predict_regressor(X.loc[train_mask], target_runtime[train_mask], X.loc[test_mask], args.ridge_alpha)
        utility_pred = gain_pred - args.alpha_runtime_penalty * runtime_pred

        for offset, idx in enumerate(np.flatnonzero(test_mask)):
            rows.append(
                {
                    "task_type": df.iloc[idx]["task_type"],
                    "dataset": df.iloc[idx]["dataset"],
                    "categorical_handling": df.iloc[idx]["categorical_handling"],
                    "observed_relative_gain": target_gain[idx],
                    "observed_runtime_ratio": df.iloc[idx]["runtime_ratio"],
                    "observed_utility": df.iloc[idx]["mgb_utility"],
                    "observed_worth_it": bool(target_class[idx]),
                    "predicted_worth_it_probability": float(prob[offset]),
                    "predicted_relative_gain": float(gain_pred[offset]),
                    "predicted_log_runtime_ratio": float(runtime_pred[offset]),
                    "predicted_utility": float(utility_pred[offset]),
                    "decision_threshold": args.decision_threshold,
                    "policy_decision": "TRY_MGB" if prob[offset] >= args.decision_threshold else "USE_XGB",
                    "classifier_kind": clf_kind,
                    "gain_model_kind": gain_kind,
                    "runtime_model_kind": runtime_kind,
                }
            )
    return pd.DataFrame(rows)


def fit_final_logistic_coefficients(X: pd.DataFrame, y: np.ndarray, c_value: float, feature_names: list[str]) -> pd.DataFrame:
    if len(np.unique(y)) < 2:
        return pd.DataFrame({"feature": feature_names, "coefficient": np.nan, "abs_coefficient": np.nan})
    model = Pipeline(
        steps=[
            ("imputer", median_imputer()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c_value, penalty="l2", solver="liblinear", class_weight="balanced", random_state=0)),
        ]
    )
    model.fit(X, y.astype(int))
    coefs = model.named_steps["clf"].coef_[0]
    return pd.DataFrame({"feature": feature_names, "coefficient": coefs, "abs_coefficient": np.abs(coefs)}).sort_values(
        "abs_coefficient", ascending=False
    )


def utility_of_policy(predictions: pd.DataFrame) -> dict[str, float]:
    observed = predictions["observed_utility"].astype(float)
    try_mask = predictions["policy_decision"].eq("TRY_MGB")
    policy_utility = np.where(try_mask, observed, 0.0)
    always_mgb = observed
    oracle = np.maximum(observed, 0.0)
    return {
        "policy_mean_utility": float(np.nanmean(policy_utility)),
        "always_mgb_mean_utility": float(np.nanmean(always_mgb)),
        "oracle_mean_utility": float(np.nanmean(oracle)),
        "policy_regret_vs_oracle": float(np.nanmean(oracle - policy_utility)),
        "try_rate": float(np.nanmean(try_mask.astype(float))),
    }


def threshold_sweep(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    y_true = predictions["observed_worth_it"].astype(bool).to_numpy()
    y_prob = predictions["predicted_worth_it_probability"].astype(float).to_numpy()
    observed = predictions["observed_utility"].astype(float).to_numpy()
    oracle = np.maximum(observed, 0.0)
    for threshold in np.linspace(0.05, 0.95, 19):
        try_mask = y_prob >= threshold
        policy_utility = np.where(try_mask, observed, 0.0)
        rows.append(
            {
                "threshold": float(threshold),
                "try_rate": float(np.mean(try_mask)),
                "accuracy": safe_metric(accuracy_score, y_true, try_mask),
                "policy_mean_utility": float(np.nanmean(policy_utility)),
                "policy_regret_vs_oracle": float(np.nanmean(oracle - policy_utility)),
                "retained_positive_rate": safe_divide(float(np.sum(try_mask & y_true)), float(np.sum(y_true)), default=math.nan),
                "rejected_loss_rate": safe_divide(float(np.sum((~try_mask) & (~y_true))), float(np.sum(~y_true)), default=math.nan),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    feature_names: list[str],
    sweep: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    y_true = predictions["observed_worth_it"].astype(bool).to_numpy()
    y_prob = predictions["predicted_worth_it_probability"].astype(float).to_numpy()
    y_decision = predictions["policy_decision"].eq("TRY_MGB").to_numpy()
    utility = utility_of_policy(predictions)

    lines = [
        "# Trainable MetaGB Preflight Policy Report",
        "",
        "## Protocol",
        "",
        "- Unit of evaluation: dataset.",
        "- Split: grouped leave-one-dataset-out.",
        "- Classifier: L2-regularized logistic regression when both classes are present; class-prior fallback otherwise.",
        "- Regression heads: ridge regressors for relative gain and log runtime ratio.",
        "- Decision target: `mgb_worth_it`, derived from quality/runtime utility.",
        "",
        "## Configuration",
        "",
        f"- input: `{args.input}`",
        f"- rows: {len(predictions)}",
        f"- features: {len(feature_names)}",
        f"- alpha_runtime_penalty: {args.alpha_runtime_penalty}",
        f"- decision_threshold: {args.decision_threshold}",
        f"- classifier_c: {args.classifier_c}",
        f"- ridge_alpha: {args.ridge_alpha}",
        "",
        "## Cross-Validated Classification",
        "",
        f"- worth-it positives: {int(np.sum(y_true))}/{len(y_true)}",
        f"- accuracy: {safe_metric(accuracy_score, y_true, y_decision):.4f}",
        f"- ROC-AUC: {safe_metric(roc_auc_score, y_true, y_prob):.4f}",
        f"- average precision: {safe_metric(average_precision_score, y_true, y_prob):.4f}",
        "",
        "## Utility Evaluation",
        "",
    ]
    for key, value in utility.items():
        lines.append(f"- {key}: {value:.6f}")

    best = sweep.sort_values(["policy_mean_utility", "accuracy"], ascending=False).head(1)
    if not best.empty:
        best_row = best.iloc[0]
        lines.extend(
            [
                "",
                "## Threshold Tuning Diagnostic",
                "",
                "- This sweep is computed on cross-validated predictions.",
                "- It is useful for choosing candidate thresholds, but final threshold selection still needs historical-only validation.",
                f"- best_threshold_by_cv_utility: {best_row['threshold']:.2f}",
                f"- best_threshold_policy_mean_utility: {best_row['policy_mean_utility']:.6f}",
                f"- best_threshold_try_rate: {best_row['try_rate']:.4f}",
                f"- best_threshold_retained_positive_rate: {best_row['retained_positive_rate']:.4f}",
                f"- best_threshold_rejected_loss_rate: {best_row['rejected_loss_rate']:.4f}",
            ]
        )

    confusion = pd.crosstab(predictions["policy_decision"], predictions["observed_worth_it"], margins=True)
    lines.extend(["", "## Policy-vs-Observed Confusion", "", confusion.to_markdown()])

    gain_mae = mean_absolute_error(predictions["observed_relative_gain"], predictions["predicted_relative_gain"])
    lines.extend(
        [
            "",
            "## Regression Heads",
            "",
            f"- relative gain MAE: {gain_mae:.6f}",
            "",
            "## Top Final Logistic Coefficients",
            "",
        ]
    )
    if coefficients["coefficient"].notna().any():
        lines.append(coefficients.head(12).to_markdown(index=False))
    else:
        lines.append("Final logistic coefficients are unavailable because the label has one class.")

    lines.extend(
        [
            "",
            "## Academic Caveats",
            "",
            "- This report is a protocol scaffold, not final evidence: the current labeled set is small.",
            "- The correct validation unit is a dataset/preprocessing group; candidate-level splits would leak information.",
            "- Thresholds and runtime penalty must be selected on historical datasets only before evaluating new datasets.",
            "- The projected-gradient features are theory-aligned proxies for MetaGB's second-order leaf score, not a full MetaGB training run.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a trainable MetaGB preflight meta-policy.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha-runtime-penalty", type=float, default=0.002)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--classifier-c", type=float, default=0.5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {"dataset", "mgb_worth_it", "mgb_relative_improvement", "runtime_ratio", "mgb_utility"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input table is missing required columns: {sorted(missing)}")
    df = df[df["mgb_relative_improvement"].notna() & df["runtime_ratio"].notna()].copy()
    X, feature_names = design_matrix(df)
    predictions = grouped_leave_one_dataset_out(df, X, args)
    coefficients = fit_final_logistic_coefficients(X, df["mgb_worth_it"].astype(bool).to_numpy(), args.classifier_c, feature_names)
    sweep = threshold_sweep(predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "meta_policy_cv_predictions.csv", index=False)
    coefficients.to_csv(args.output_dir / "meta_policy_final_coefficients.csv", index=False)
    sweep.to_csv(args.output_dir / "meta_policy_threshold_sweep.csv", index=False)
    (args.output_dir / "meta_policy_report.md").write_text(build_report(predictions, coefficients, feature_names, sweep, args), encoding="utf-8")

    print(f"wrote {args.output_dir / 'meta_policy_cv_predictions.csv'}")
    print(f"wrote {args.output_dir / 'meta_policy_final_coefficients.csv'}")
    print(f"wrote {args.output_dir / 'meta_policy_threshold_sweep.csv'}")
    print(f"wrote {args.output_dir / 'meta_policy_report.md'}")


if __name__ == "__main__":
    main()
