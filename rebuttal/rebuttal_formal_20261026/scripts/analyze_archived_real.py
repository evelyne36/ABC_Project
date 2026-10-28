#!/usr/bin/env python3
"""Recompute descriptive facts from archived records; never fit a model."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_formal_20261026/results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    audit = ROOT / "rebuttal_plan_20261025/已有结果复用审计_数据.json"
    candidates = ROOT / ("aaai_FullCodeProject-check_codereview0701/datacollection0702/"
        "tabular_onehot_20260628_025442/default-tabular_comprehensive_onehot_20260628_025442/candidate_configs.json")
    data = json.loads(audit.read_text())["real_panel"]
    configs = json.loads(candidates.read_text())
    selected, progress, summary = [], [], []
    for dataset in ["cpu_activity.csv", "kin8nm.csv", "energy_efficiency.csv"]:
        models = data[dataset]
        pair = {}
        for model in ["xgb_cpu", "current_metagb"]:
            records = sorted(models[model]["records"], key=lambda r: r["outer_fold"])
            assert len(records) == 5
            pair[model] = records
            for record in records:
                index = int(record["selected_candidate_index"])
                params = configs[model][index - 1]
                assert params["candidate_id"] == record["selected_candidate_id"]
                path = ROOT / record["path"]
                selected.append({"dataset": dataset, "model": model, "outer_fold": record["outer_fold"],
                    "test_rmse": float(record["test_rmse"]), "selected_val_rmse": float(record["selected_val_rmse"]),
                    "selected_candidate_index": index, "selected_candidate_id": params["candidate_id"],
                    "final_train_elapsed_sec": float(record["final_train_sec"]),
                    "search_and_refit_elapsed_sec": float(record["elapsed_sec"]),
                    "params_json": json.dumps(params, sort_keys=True),
                    "log": record["path"], "log_sha256": digest(path)})
                for match in re.finditer(r"candidate (\d+)/(\d+) rmse=([^ ]+) status=(\w+)", path.read_text()):
                    progress.append({"dataset": dataset, "model": model, "outer_fold": record["outer_fold"],
                        "candidate_index": int(match[1]), "total_candidates": int(match[2]),
                        "validation_rmse": float(match[3]), "status": match[4],
                        "scope": "sparsely printed progress; not complete candidate trace"})
        xgb, cgb = pair["xgb_cpu"], pair["current_metagb"]
        mx = statistics.mean(float(r["test_rmse"]) for r in xgb)
        mc = statistics.mean(float(r["test_rmse"]) for r in cgb)
        summary.append({"dataset": dataset, "folds": 5, "xgb_mean_test_rmse": mx, "cgb_mean_test_rmse": mc,
            "cgb_relative_rmse_increase_pct": 100 * (mc / mx - 1),
            "cgb_better_fold_count": sum(float(c["test_rmse"]) < float(x["test_rmse"]) for x, c in zip(xgb, cgb)),
            "search_elapsed_ratio_of_means": statistics.mean(float(r["elapsed_sec"]) for r in cgb) / statistics.mean(float(r["elapsed_sec"]) for r in xgb),
            "search_elapsed_median_of_paired_fold_ratios": statistics.median(float(c["elapsed_sec"]) / float(x["elapsed_sec"]) for x, c in zip(xgb, cgb)),
            "final_train_elapsed_ratio_of_means": statistics.mean(float(r["final_train_sec"]) for r in cgb) / statistics.mean(float(r["final_train_sec"]) for r in xgb),
            "final_train_elapsed_median_of_paired_fold_ratios": statistics.median(float(c["final_train_sec"]) / float(x["final_train_sec"]) for x, c in zip(xgb, cgb)),
            "timing_scope": "archived elapsed; not same-host equal-budget comparison"})
    write_csv("N4_selected_folds.csv", selected)
    write_csv("N4_logged_progress_ONLY.csv", progress)
    write_csv("N4_archived_summary.csv", summary)
    (OUT / "N4_reanalysis_manifest.json").write_text(json.dumps({"model_fit_calls": 0,
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in [audit, candidates]],
        "selected_records": len(selected), "progress_records": len(progress),
        "limitations": ["No causal mechanism inferred from these correlations", "Progress is sparse; no proportion of unstable candidates can be estimated", "Historical elapsed is not CPU time or equal-compute evidence"]}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
