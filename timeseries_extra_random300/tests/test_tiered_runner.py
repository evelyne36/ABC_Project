from __future__ import annotations

from pathlib import Path

from timeseries_extra_random300.scripts.run_tiered_timeseries_benchmark import (
    build_parser,
    benchmark_argv,
    load_tier_config,
    run_tiered,
    select_tiers,
)


def test_tier_config_groups_datasets_in_execution_order() -> None:
    tiers = load_tier_config(Path("timeseries_extra_random300/configs/tiered_tree_benchmark.json"))

    assert [tier["name"] for tier in tiers] == [
        "tier1_fast_core",
        "tier2_slow_favorable",
        "tier3_slow_unfavorable",
    ]
    assert tiers[0]["datasets"] == ["exchange_rate", "ETTm1", "ETTm2", "ETTh2", "ETTh1", "ILI"]
    assert tiers[1]["datasets"] == ["PEMS03", "PEMS07"]
    assert tiers[2]["datasets"] == ["Weather", "PEMS04", "PEMS08"]


def test_select_tiers_preserves_requested_order() -> None:
    tiers = load_tier_config(Path("timeseries_extra_random300/configs/tiered_tree_benchmark.json"))
    selected = select_tiers(tiers, ["tier2_slow_favorable", "tier1_fast_core"])

    assert [tier["name"] for tier in selected] == ["tier2_slow_favorable", "tier1_fast_core"]


def test_benchmark_argv_contains_registered_protocol() -> None:
    args = build_parser().parse_args(
        [
            "--base-run-id",
            "unit_tiered",
            "--task-workers",
            "2",
        ]
    )
    tier = {"name": "tier1_fast_core", "datasets": ["exchange_rate", "ETTm1"]}

    argv = benchmark_argv(args, tier, "unit_tiered__tier1_fast_core")

    assert argv[argv.index("--split-protocol") + 1] == "fixed"
    assert argv[argv.index("--feature-mode") + 1] == "window_summary"
    assert argv[argv.index("--target-mode") + 1] == "sequence"
    assert argv[argv.index("--output-scope") + 1] == "target"
    assert argv[argv.index("--seeds") + 1 : argv.index("--seq-lens")] == ["42"]
    assert argv[argv.index("--seq-lens") + 1 : argv.index("--pred-lens")] == ["24", "48"]
    assert argv[argv.index("--pred-lens") + 1 : argv.index("--max-candidates")] == ["1", "6", "12"]
    assert argv[argv.index("--max-candidates") + 1] == "300"
    assert argv[argv.index("--datasets") + 1 : argv.index("--models")] == ["exchange_rate", "ETTm1"]


def test_benchmark_argv_can_pin_one_seed_for_outer_seed_loop() -> None:
    args = build_parser().parse_args(["--base-run-id", "unit_tiered", "--seeds", "42", "43"])
    tier = {"name": "tier1_fast_core", "datasets": ["exchange_rate"]}

    argv = benchmark_argv(args, tier, "unit_tiered__seed43__tier1_fast_core", seeds=["43"])

    assert argv[argv.index("--seeds") + 1 : argv.index("--seq-lens")] == ["43"]


def test_tiered_dry_run_writes_manifest_without_training(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--base-run-id",
            "unit_tiered_dry_run",
            "--tiers",
            "tier1_fast_core",
            "--dry-run",
            "--output-root",
            str(tmp_path),
        ]
    )

    result = run_tiered(args)

    assert result["completed_tiers"] == 0
    assert result["completed_runs"] == 0
    manifest_path = tmp_path / "results" / "unit_tiered_dry_run" / "tier_manifest.json"
    assert manifest_path.exists()
