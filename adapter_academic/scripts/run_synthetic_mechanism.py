from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mechanism_score import ScoreConfig, score_arrays


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "adapter_academic" / "outputs"


def noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return rng.normal(0.0, scale, size=n)


def piecewise_constant(rng: np.random.Generator, n: int, d: int, noise_scale: float) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    X = rng.normal(size=(n, d))
    region = X[:, 0] > 0
    y = np.where(region, 1.0, -1.0) + noise(rng, n, noise_scale)
    return X, y, {0}, set()


def local_linear(rng: np.random.Generator, n: int, d: int, noise_scale: float) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    X = rng.normal(size=(n, d))
    y = np.where(X[:, 0] > 0, 2.0 * X[:, 3], -1.5 * X[:, 3]) + noise(rng, n, noise_scale)
    return X, y, {0}, {3}


def decoupled_switch(rng: np.random.Generator, n: int, d: int, noise_scale: float) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    X = rng.normal(size=(n, d))
    y = np.where(X[:, 0] > 0, 2.0 * X[:, 3], -2.0 * X[:, 5]) + noise(rng, n, noise_scale)
    return X, y, {0}, {3, 5}


def high_dim_sparse(rng: np.random.Generator, n: int, d: int, noise_scale: float) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    X = rng.normal(size=(n, d))
    y = np.where(X[:, 0] > 0, 1.8 * X[:, 3], -1.8 * X[:, 7]) + noise(rng, n, noise_scale)
    return X, y, {0}, {3, 7}


SCENARIOS = {
    "piecewise_constant": (piecewise_constant, 0),
    "local_linear_response": (local_linear, 1),
    "conditional_predictive_decoupling": (decoupled_switch, 1),
    "high_dim_sparse_noise": (high_dim_sparse, 1),
}


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int = 1000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_report(scores: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# Synthetic Mechanism Recovery Report",
        "",
        "## Contract",
        "",
        "- Scores are computed without outcome labels.",
        "- Synthetic labels indicate whether the data generator contains a MetaGB-aligned mechanism.",
        "- Feature-hit rate uses known true predictive features.",
        "",
        "## Scenario Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- `piecewise_constant` should score low because constant leaves already match the generator.",
        "- `local_linear_response` and `conditional_predictive_decoupling` should score high.",
        "- `high_dim_sparse_noise` should remain high if the projected-gradient signal is robust to irrelevant features.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic mechanism recovery for MetaGB preflight scoring.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScoreConfig(seed=42)
    rows: list[dict[str, Any]] = []
    dims_by_scenario = {
        "piecewise_constant": [10, 50],
        "local_linear_response": [10, 50],
        "conditional_predictive_decoupling": [10, 50],
        "high_dim_sparse_noise": [20, 100, 300],
    }
    for scenario, (generator, expected) in SCENARIOS.items():
        for d in dims_by_scenario[scenario]:
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                X, y, true_cond, true_pred = generator(rng, args.n, d, args.noise)
                scored = score_arrays(X, y, config, true_predictive_features=true_pred)
                rows.append(
                    {
                        "scenario": scenario,
                        "dimension": d,
                        "seed": seed,
                        "expected_mechanism_match": expected,
                        "true_conditional_features": ";".join(map(str, sorted(true_cond))),
                        "true_predictive_features": ";".join(map(str, sorted(true_pred))),
                        **scored,
                    }
                )
    scores = pd.DataFrame(rows)
    summary_rows = []
    for (scenario, d), group in scores.groupby(["scenario", "dimension"], sort=False):
        lo, hi = bootstrap_ci(group["mechanism_alignment_score"].to_numpy(dtype=float), seed=123)
        hit_lo, hit_hi = bootstrap_ci(group["feature_hit_rate"].to_numpy(dtype=float), seed=456)
        summary_rows.append(
            {
                "scenario": scenario,
                "dimension": d,
                "expected_mechanism_match": int(group["expected_mechanism_match"].iloc[0]),
                "score_mean": float(group["mechanism_alignment_score"].mean()),
                "score_ci95_low": lo,
                "score_ci95_high": hi,
                "feature_hit_rate_mean": float(group["feature_hit_rate"].mean(skipna=True)),
                "feature_hit_ci95_low": hit_lo,
                "feature_hit_ci95_high": hit_hi,
            }
        )
    summary = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "synthetic_scores.csv", index=False)
    summary.to_csv(args.output_dir / "synthetic_summary.csv", index=False)
    (args.output_dir / "synthetic_mechanism_report.md").write_text(build_report(scores, summary), encoding="utf-8")
    print(f"wrote {args.output_dir / 'synthetic_scores.csv'}")
    print(f"wrote {args.output_dir / 'synthetic_summary.csv'}")
    print(f"wrote {args.output_dir / 'synthetic_mechanism_report.md'}")


if __name__ == "__main__":
    main()
