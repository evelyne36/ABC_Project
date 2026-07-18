from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "artifacts" / "raw" / "datacollection0702"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    report: dict[str, object] = {}

    synthetic_tasks = sorted((RAW / "synthetic_main1020full_random300" / "tasks").glob("*.json"))
    dim_counts = Counter("dim10" if "__dim10__" in path.name else "dim20" if "__dim20__" in path.name else "other" for path in synthetic_tasks)
    model_counts = Counter("current_metagb" if path.name.endswith("__current_metagb.json") else "xgb_cpu" if path.name.endswith("__xgb_cpu.json") else "other" for path in synthetic_tasks)
    require(len(synthetic_tasks) == 380, f"expected 380 synthetic tasks, found {len(synthetic_tasks)}")
    require(dim_counts == {"dim10": 190, "dim20": 190}, f"unexpected synthetic dimensions: {dim_counts}")
    require(model_counts == {"xgb_cpu": 190, "current_metagb": 190}, f"unexpected synthetic models: {model_counts}")
    require(not any("dim50" in path.name for path in synthetic_tasks), "dim50 task migrated")
    report["synthetic"] = {"tasks": len(synthetic_tasks), "dimensions": dim_counts, "models": model_counts}

    baseline_tasks = sorted((RAW / "datacollection_baselinemodel_random300" / "tasks").glob("*.log"))
    baseline_counts = Counter(path.name.split("__", 1)[0] for path in baseline_tasks)
    finish_counts: Counter[str] = Counter()
    for path in baseline_tasks:
        model = path.name.split("__", 1)[0]
        if re.search(r"finish .*status=ok", path.read_text(encoding="utf-8", errors="ignore")):
            finish_counts[model] += 1
    expected_baseline = {model: 270 for model in ["linear", "extra_trees", "random_forest", "lightgbm", "catboost", "torchmlp"]}
    require(baseline_counts == expected_baseline, f"unexpected baseline logs: {baseline_counts}")
    require(finish_counts["catboost"] == 257, f"unexpected CatBoost success count: {finish_counts['catboost']}")
    require(sum(finish_counts.values()) == 1607, f"unexpected baseline success total: {sum(finish_counts.values())}")
    require(not any("realmlp" in path.name or "tabm" in path.name for path in baseline_tasks), "excluded baseline migrated")
    report["baselines"] = {"logs": len(baseline_tasks), "by_model": baseline_counts, "finish_ok": finish_counts}

    ts_dir = RAW / "datacollection_ts_tree_seed42__non_pems"
    ts_files = sorted(path.name for path in ts_dir.iterdir() if path.is_file())
    require(len(ts_files) == 5, f"expected 5 time-series raw files, found {ts_files}")
    with (ts_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        ts_rows = list(csv.DictReader(handle))
    pred1 = [row for row in ts_rows if row["status"] == "ok" and row["pred_len"] == "1"]
    pred1_models = Counter(row["model"] for row in pred1)
    require(len(ts_rows) == 68, f"expected 68 time-series rows, found {len(ts_rows)}")
    require(len(pred1) == 25, f"expected 25 pred_len=1 rows, found {len(pred1)}")
    require(pred1_models == {"lgbm_cpu": 7, "xgb_cpu": 7, "catboost_cpu": 7, "current_metagb": 4}, f"unexpected pred_len=1 coverage: {pred1_models}")
    report["timeseries"] = {"raw_files": ts_files, "summary_rows": len(ts_rows), "pred_len_1_rows": len(pred1), "models": pred1_models}

    forbidden = [path for path in RAW.rglob("*") if path.is_file() and (path.name == ".DS_Store" or path.name in {"mcls_onehot_appendix", "only-bioresponse-left"})]
    require(not forbidden, f"forbidden marker files migrated: {forbidden}")

    output = PROJECT / "artifacts" / "provenance" / "migration_validation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=dict) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=dict))
    print(f"validation=ok output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

