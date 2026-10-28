#!/usr/bin/env python3
"""Read-only N5 archive audit. Writes only the three adjacent audit artifacts."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("capacity", ROOT / "rebuttal/fastlane_20261027/capacity.py")
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)
SPLITS = capacity.SPLITS
VARIANTS = capacity.VARIANTS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label, atol=1e-10):
    if not np.allclose(actual, expected, rtol=1e-12, atol=atol):
        raise AssertionError(f"{label}: maximum difference {np.max(np.abs(np.asarray(actual) - expected))}")


def stats(values):
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def cell(value, digits=4):
    return f"{value['mean']:.{digits}f} ± {value['sample_sd']:.{digits}f}" if value["sample_sd"] is not None else f"{value['mean']:.{digits}f}"


def main():
    paths = sorted((ROOT / "local_sync/followup_results").rglob("capacity.json"))
    paths += sorted((ROOT / "local_sync/fastlane_results").rglob("capacity.json"))
    assert len(paths) == 36, len(paths)
    audits, rows, seen = [], [], set()
    for path in paths:
        report = json.loads(path.read_text())
        case, seed = path.parts[-4], int(path.parts[-2][1:])
        assert (case, seed) not in seen
        seen.add((case, seed))
        host_root = Path(*path.parts[:path.parts.index("results")])
        model_dir = host_root / "results" / report["model_dir"].split("/results/", 1)[1]
        model_path = model_dir / "model.json"
        data_path = ROOT / "local_sync/fastlane_data" / f"{case}__s{seed}.npz"
        metadata_path = data_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        scale = metadata["preprocessing"]["y_scale"] if case in ("cpu_activity", "protein") else 1.
        assert np.isfinite(scale) and scale > 0
        assert metadata["seed"] == seed
        assert metadata["npz_sha256"] == sha(data_path)
        learner = json.loads(model_path.read_text())["learner"]
        trees = learner["gradient_booster"]["model"]["trees"]
        base = capacity.scalar(learner["learner_model_param"]["base_score"])
        assert learner["objective"]["name"] == "reg:squarederror"
        assert report["route_frozen"] is True
        assert report["trees"] == len(trees) and report["base_score"] == base
        assert report["schema_version"] in (1, 2)
        assert report["replay"]["tolerance_abs"] == 1e-5
        with np.load(data_path, allow_pickle=False) as data:
            X, y, splits = capacity.validate_arrays(data["X"], data["y"], {s: data[s] for s in SPLITS})
        assert report["num_features"] == X.shape[1]
        assert report["split_sizes"] == {s: len(splits[s]) for s in SPLITS}
        result, ridge, eta, _, sources = capacity.load_saved_metadata(model_dir)
        assert report["ridge_lambda_l2"] == ridge and report["learning_rate"] == eta
        assert report["parameter_sources"] == sources
        metrics, gate, diagnostic, canonical = {}, {}, {}, {}
        max_prediction_difference = 0.
        constant_entries = capacity.anchor_entries(trees, X.shape[1])
        with np.load(path.with_name("predictions.npz"), allow_pickle=False) as predictions, \
                np.load(model_dir / "predictions.npz", allow_pickle=False) as native_archive:
            for split in SPLITS:
                np.testing.assert_array_equal(predictions[f"rows__{split}"], splits[split])
                np.testing.assert_array_equal(native_archive[f"{split}_indices"], splits[split])
                routes = capacity.crossgb_leaf_routes(X[splits[split]], trees)
                native = capacity.predict_stored_native(X[splits[split]], routes, trees, base)
                np.testing.assert_array_equal(native, native_archive[split])
                np.testing.assert_array_equal(native, predictions[f"native_anchor__{split}"])
                float64 = capacity.predict_frozen(X[splits[split]], routes, constant_entries, base)
                diagnostic[split] = float(np.max(np.abs(float64 - native.astype(np.float64))))
                canonical[split] = float(np.sqrt(np.mean((native.astype(np.float64) - y[splits[split]]) ** 2)))
                for variant in VARIANTS:
                    saved = predictions[f"{variant}__{split}"]
                    assert saved.shape == (len(splits[split]),) and np.isfinite(saved).all()
                    fitted = report["leaf_models"][variant]
                    assert len(fitted) == len(trees)
                    for index, entries in enumerate(fitted):
                        assert {e["leaf"] for e in entries} == set(np.flatnonzero(np.asarray(trees[index]["left_children"]) == -1))
                        for entry in entries:
                            anchor = next(e for e in constant_entries[index] if e["leaf"] == entry["leaf"])
                            if anchor["kind"] == "constant":
                                assert entry["kind"] == "constant" and entry["theta"] == anchor["theta"]
                            if variant != "stored_anchor" and split == "train":
                                assert entry["n_train"] == int(np.sum(routes[:, index] == entry["leaf"]))
                                if anchor["kind"] != "constant":
                                    assert entry["features"][0] == anchor["features"][0]
                                    assert len(entry["features"]) == entry["k"] <= int(variant[-1])
                                    assert entry["affine"] == variant.startswith("affine")
                                    assert len(entry["theta"]) == entry["k"] + int(entry["affine"])
                    if case == "S1_offset":
                        np.testing.assert_array_equal(saved, predictions[f"stored_anchor__{split}"])
                    replay = native if variant == "stored_anchor" and report["schema_version"] == 2 else capacity.predict_frozen(X[splits[split]], routes, fitted, base)
                    close(saved, replay, f"{case}/{seed}/{variant}/{split} route replay")
                    max_prediction_difference = max(max_prediction_difference, float(np.max(np.abs(saved - replay))))
                    rmse = float(np.sqrt(np.mean((saved.astype(np.float64) - y[splits[split]]) ** 2)))
                    close(rmse, report["metrics"][variant]["rmse"][split], "RMSE")
                    close(report["metrics"][variant]["delta_rmse_vs_stored_anchor"][split], rmse - report["metrics"]["stored_anchor"]["rmse"][split], "stored delta")
                    metrics.setdefault(variant, {})[split] = rmse
                stored = predictions[f"stored_anchor__{split}"]
                gate[split] = float(np.max(np.abs(stored.astype(np.float64) - native.astype(np.float64))))
                close(gate[split], report["replay"]["max_abs_difference"][split], "native anchor gate", atol=1e-14)
                assert gate[split] <= 1e-5
                if report["schema_version"] == 2:
                    assert gate[split] == 0
                    close(diagnostic[split], report["replay"]["float64_max_abs_difference"][split], "float64 diagnostic", atol=1e-14)
        for variant in VARIANTS:
            assert capacity.parameter_counts(report["leaf_models"][variant]) == report["metrics"][variant]["counts"]
            metric = report["metrics"][variant]
            for timing in (metric["fit_timing"], metric["prediction_timing_all_splits"], *metric["prediction_timing"].values()):
                for field in ("wall_seconds", "cpu_seconds"):
                    assert timing[field] is None or (np.isfinite(timing[field]) and timing[field] >= 0)
            counts = metric["counts"]
            degenerate = counts["predictive_leaf_count"] == 0
            assert degenerate == (case == "S1_offset")
            bias = "affine" if variant.startswith("affine") else "slope" if variant.startswith("slope") else "stored"
            row = {"case": case, "seed": seed, "variant": variant, "bias": bias,
                   "requested_k": int(variant[-1]) if bias != "stored" else 1,
                   "eligible": not degenerate, "schema_version": report["schema_version"],
                   "y_scale": scale, "rmse_unit": "original_target" if case in ("cpu_activity", "protein") else "synthetic_target",
                   "trees": len(trees), "source": str(path.relative_to(ROOT)),
                   "fit_wall_seconds": metric["fit_timing"]["wall_seconds"], "fit_cpu_seconds": metric["fit_timing"]["cpu_seconds"],
                   "prediction_wall_seconds": metric["prediction_timing_all_splits"]["wall_seconds"],
                   "prediction_cpu_seconds": metric["prediction_timing_all_splits"]["cpu_seconds"],
                   "scalar_parameters": counts["total_scalar_parameters_including_base"],
                   "actual_k": counts["actual_k_mean_predictive_leaves"],
                   "predictive_leaves": counts["predictive_leaf_count"], "constant_leaves": counts["constant_leaf_count"],
                   "refit_leaves": counts["leaves_refit"], "fallbacks": counts["no_training_rows_fallbacks"]}
            for split in SPLITS:
                row[f"raw_saved_{split}_rmse"] = metrics[variant][split]
                # Harmonize stored anchors across schema versions; do not alter raw observations.
                canonical_rmse = canonical[split] if variant == "stored_anchor" else metrics[variant][split]
                row[f"{split}_rmse"] = canonical_rmse * scale
                row[f"native_gate_{split}_max_abs"] = gate[split]
                row[f"float64_diagnostic_{split}_max_abs"] = diagnostic[split]
            rows.append(row)
        audits.append({"case": case, "seed": seed, "schema_version": report["schema_version"], "source": str(path.relative_to(ROOT)),
                       "capacity_sha256": sha(path), "predictions_sha256": sha(path.with_name("predictions.npz")),
                       "model_sha256": sha(model_path), "data_sha256": sha(data_path), "metadata_sha256": sha(metadata_path),
                       "y_scale": scale, "rmse_recomputed": True, "native_predictions_exact": True,
                       "routes_and_leaf_models_replayed": True, "max_probe_prediction_replay_difference": max_prediction_difference,
                       "counts_recomputed": True, "constant_leaves_frozen": True, "native_gate_max_abs": gate,
                       "float64_stored_diagnostic_max_abs": diagnostic})
    cases = sorted({r["case"] for r in rows if r["eligible"]})
    assert len(cases) == 7
    for case in cases:
        assert {r["seed"] for r in rows if r["case"] == case} == set(range(98001, 98006))
    index = {(r["case"], r["seed"], r["variant"]): r for r in rows}
    for row in rows:
        if row["bias"] != "stored":
            baseline = index[row["case"], row["seed"], row["bias"] + "_k1"]
            row["test_improvement_vs_same_bias_k1_pct"] = 100 * (baseline["test_rmse"] - row["test_rmse"]) / baseline["test_rmse"]
            row["fit_cpu_ratio_vs_same_bias_k1"] = row["fit_cpu_seconds"] / baseline["fit_cpu_seconds"]
            row["prediction_cpu_ratio_vs_same_bias_k1"] = row["prediction_cpu_seconds"] / baseline["prediction_cpu_seconds"]
            row["parameter_ratio_vs_same_bias_k1"] = row["scalar_parameters"] / baseline["scalar_parameters"]
        if row["bias"] == "affine":
            slope = index[row["case"], row["seed"], f"slope_k{row['requested_k']}"]
            row["test_improvement_vs_same_k_slope_pct"] = 100 * (slope["test_rmse"] - row["test_rmse"]) / slope["test_rmse"]
    numeric = ["train_rmse", "val_rmse", "test_rmse", "actual_k", "scalar_parameters", "fit_cpu_seconds", "fit_wall_seconds", "prediction_cpu_seconds", "prediction_wall_seconds", "predictive_leaves", "constant_leaves", "fallbacks", "test_improvement_vs_same_bias_k1_pct", "fit_cpu_ratio_vs_same_bias_k1", "prediction_cpu_ratio_vs_same_bias_k1", "parameter_ratio_vs_same_bias_k1", "test_improvement_vs_same_k_slope_pct"]
    aggregates = []
    for case in cases:
        for variant in VARIANTS:
            selected = [r for r in rows if r["case"] == case and r["variant"] == variant]
            aggregate = {"case": case, "variant": variant, "n": len(selected), **{field: stats([r[field] for r in selected]) for field in numeric if field in selected[0] and selected[0][field] is not None}}
            if "test_improvement_vs_same_bias_k1_pct" in selected[0]:
                aggregate["test_improved_seeds_vs_same_bias_k1"] = sum(r["test_improvement_vs_same_bias_k1_pct"] > 0 for r in selected)
            aggregates.append(aggregate)
    rows.sort(key=lambda r: (r["case"], r["seed"], VARIANTS.index(r["variant"])))
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (OUT / "N5完整结果_20261028.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    verification = {"status": "passed_local_archive_audit", "capacity_files": len(paths), "followup_files": 26, "prior_files": 10,
                    "effective_runs": 35, "effective_cases": cases, "rmse_values_recomputed": len(paths) * len(VARIANTS) * len(SPLITS),
                    "native_split_predictions_verified": len(paths) * len(SPLITS),
                    "excluded_degenerate": [{"case": "S1_offset", "seed": 98001, "reason": "pure-step control; all 4 leaves are frozen constants; no predictive leaves, all seven variants identical"}],
                    "schema_counts": {str(s): sum(a["schema_version"] == s for a in audits) for s in (1, 2)},
                    "statistics": "mean and sample SD (ddof=1), paired per-seed improvement and cost ratios; no inference of universal optimum",
                    "unit_policy": "CPU/protein RMSE multiplied by each seed's preprocessing.y_scale; synthetic RMSE unchanged; time seconds and parameter counts unscaled",
                    "schema_policy": "schema1 saved stored_anchor is float64, tolerance<=1e-5; schema2 native float32 exactly matches native_anchor, diagnostic float64 residual retained. Summaries harmonize stored baseline to native_anchor without changing saved raw metrics.",
                    "provenance": {"audit_script_sha256": sha(Path(__file__)), "prediction_replay_helper_sha256": sha(ROOT / "rebuttal/fastlane_20261027/capacity.py")},
                    "audits": audits, "aggregates": aggregates}
    (OUT / "followup_capacity_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_report(verification, rows)
    print(json.dumps({"files": len(paths), "effective_runs": 35, "status": "passed", "max_replay_difference": max(a["max_probe_prediction_replay_difference"] for a in audits)}))


def write_report(verification, rows):
    aggregates = verification["aggregates"]
    lookup = {(r["case"], r["variant"]): r for r in aggregates}
    lines = ["# N5 完整容量结果与独立归档验收（2026-10-28）", "",
             "验收通过：原始归档共 36 份容量结果，新增 24 份 followup、2 份 S3 repair，加旧 10 份。S1_offset/98001 是纯 step 对照（2 sign(x0)+eps），其四个叶均为冻结常数，七个变体预测完全相同；它不能用于检验 slope+bias 的拟合收益，作为退化对照保留。有效分析为 7 cases × 5 seeds = 35 次；S1_affine 才是含区域斜率与截距的对照。逐份从预测 NPZ 重算全部 756 个 train/val/test RMSE，以模型 JSON 重建路由并重放全部叶模型，核对拆分、参数来源、叶数与参数数、原生预测及门限；全部通过。没有重新训练。", "",
             "本报告是固定 k1 树结构与路由的容量探针，不是完整原生 k>1 CrossGB 算法比较。每个变体按自身训练残差逐树重拟合，stored feature 优先，新增特征由训练集贪心选择；ridge 与 eta 沿用已保存配置。常数叶对所有变体冻结；训练未占据叶保留存储斜率，affine 加零截距。affine 的 k 仅计算非偏置输入，另有不受惩罚的截距。", "",
             "CPU activity/protein 按各 seed 的数据侧文件 y_scale 将 RMSE 还原为原目标单位；合成任务保持原单位。表中均为五次均值 ± 样本标准差（ddof=1）。真实任务的五次训练/验证重采样共享同一 held-out test，不是五个独立数据集；合成重复也不足以证明普适最佳容量。成本是现有探针执行记录，单位秒；拟合不含原生建树与共用路由，预测耗时合并 train/val/test。每次 k>1 与同 bias 的 k1 在同 seed、同执行归档中作配对比较；CPU/protein/S1 的 seed98001 来自旧批次，跨批绝对耗时不能视为严格统一硬件基准。", "",
             "schema1（34 份）的 stored_anchor 使用 float64 存储叶重放，原生差异均 ≤1e-5；schema2（修复的 S3/98004、98005）使用 float32 叶乘法与逐树累加，原生差异为 0。审计另行核实所有 36 份的 float32 重放与原生预测逐值相等。float64 诊断完整保留：S3/98004 最大 1.0535789363608217e-5，S3/98005 最大 1.058922388974537e-5，属于运算精度差异；没有放宽门限，也不将混合 schema 的微小存储锚点差异判为失败。下表 stored_anchor 汇总统一用 NPZ 的 native_anchor；CSV 同时保留原报告 RMSE，六个 refit 变体保持原预测。", "", "## 精度、容量与成本", "",
             "提升百分比为逐 seed 的 100×(同 bias k1 RMSE−当前 RMSE)/同 bias k1 RMSE，再求均值与 SD；正值表示改善。参数和时间比同样先按 seed 配对再汇总。stored_anchor 不存在重新拟合成本，避免将其视作 k1 refit 成本。", ""]
    insertion = lines.index("## 精度、容量与成本")
    lines[insertion:insertion] = [
        "## 主要结论", "",
        "结果支持按任务权衡容量，而非普遍增大 k。S1_sparse 的 slope k1 test RMSE 为 0.3325±0.0232，k3 为 0.4028±0.0452，五次全变差；affine k1→k3 也五次全变差。相对地，S1_dense 的 slope k3 五次均改善，配对平均改善 2.323%±1.503%，但拟合 CPU 成本为 k1 的 35.908±0.457 倍，参数为 2.824±0.029 倍。", "",
        "bias 的收益依赖结构：S1_affine 中 affine k2 test 均值 0.4747±0.1014，优于 affine k1 的 0.5093±0.0774（4/5 次改善），但 S1_sparse 的 affine 反而在每个 k 上均更差。S2_sparse 的 slope k2 比 k1 配对改善 2.238%±2.652%（4/5 次改善），k3 收益接近零；S3_sparse 的 slope k1 test 均值低于 k2/k3，k>1 各仅 2/5 次改善，不能据部分 seed 宣称更大容量稳定受益。", "",
        "protein 的改善小而较一致：slope k3、affine k2/k3 各 5/5 次优于各自 k1，但 slope k3 的配对收益仅 0.980%±0.957%，拟合 CPU 为 6.452±0.068 倍；affine k3 为 0.552%±0.368%，成本 5.799±0.377 倍。CPU activity 则提供明确反例：较大 k 的训练误差下降，test 均值与波动却增加；slope k1→k3 为 2.9224±0.1394→3.5232±1.5446，affine 为 2.9181±0.1835→5.0884±3.6771。seed98005 的 affine k3 相对 affine k1 test RMSE 恶化 275.82%，seed98002 也恶化 98.82%；不能沿用单个 seed 的乐观判断。", "",
        "注意，配对百分比的均值与先求 RMSE 均值再算百分比权重不同，可能出现符号不同。例如 S3 slope k2 的配对百分比均值为 +0.894%，但 test RMSE 均值从 0.5742 升至 0.5759；应同时报告两者，不能将前者解释为均值 RMSE 降低。", "",
    ]
    for case in verification["effective_cases"]:
        lines.extend([f"### {case}", "", "| 变体 | train RMSE | val RMSE | test RMSE | test提升% vs 同bias k1 | 实际k | 参数比 | 拟合CPU比 | 预测CPU比 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for variant in VARIANTS:
            r = lookup[case, variant]
            ratio = lambda field: cell(r[field], 3) if field in r else "—"
            lines.append(f"| {variant} | {cell(r['train_rmse'])} | {cell(r['val_rmse'])} | {cell(r['test_rmse'])} | {ratio('test_improvement_vs_same_bias_k1_pct')} | {cell(r['actual_k'], 3)} | {ratio('parameter_ratio_vs_same_bias_k1')} | {ratio('fit_cpu_ratio_vs_same_bias_k1')} | {ratio('prediction_cpu_ratio_vs_same_bias_k1')} |")
        lines.append("")
    lines.extend(["## 原始成本与冻结常数", "", "时间为 CPU 秒均值 ± SD；参数包含 base、所有存储常数、非偏置斜率与 affine 截距。训练/验证/测试共用的原生路由成本另存原报告，未加进此处预测时间。", "", "| case | 变体 | 拟合CPU秒 | 预测CPU秒（3拆分） | 标量参数 | 冻结常数叶 | 训练空叶fallback |", "|---|---|---:|---:|---:|---:|---:|"])
    for r in aggregates:
        lines.append(f"| {r['case']} | {r['variant']} | {cell(r['fit_cpu_seconds'],3) if r['variant']!='stored_anchor' else '未测原训练'} | {cell(r['prediction_cpu_seconds'],3)} | {cell(r['scalar_parameters'],1)} | {cell(r['constant_leaves'],1)} | {cell(r['fallbacks'],1)} |")
    lines.extend(["", "## 主要方向与反例", ""])
    for case in verification["effective_cases"]:
        candidates = [lookup[case, v] for v in VARIANTS[1:]]
        best = min(candidates, key=lambda r: r["test_rmse"]["mean"])
        lines.append(f"- {case}：六个 refit 变体中，test 均值最低为 {best['variant']}（{cell(best['test_rmse'])}）；这是本次五次重复的描述性结果，未按 test 重新选配置。")
    lines.extend(["", "下列 bias 收益按相同 k、相同 seed 与 slope 比较，正值表示 affine 较好：", "", "| case | affine k1收益% | affine k2收益% | affine k3收益% |", "|---|---:|---:|---:|"])
    for case in verification["effective_cases"]:
        lines.append(f"| {case} | " + " | ".join(cell(lookup[case,f'affine_k{k}']["test_improvement_vs_same_k_slope_pct"],3) for k in (1,2,3)) + " |")
    lines.extend(["", "更大的 k 与额外 bias 是否值得，需要同时看 held-out 误差、配对变异、实际 k 和成本；训练 RMSE 降低本身不构成泛化改善证据。固定路由、冻结常数及存储特征优先都限制本结论的外推范围。", "", "## 复现与证据", "", "`followup_audit.py` 仅读取本地归档，输出本报告、逐 seed/variant 的 CSV（包含原报告 RMSE、统一单位 RMSE、收益与成本）以及完整验证 JSON。JSON 记录 36 份输入、模型、预测、数据与元数据哈希、schema 分布、原生门限和 float64 诊断。审计重放复用已验证的容量预测辅助函数，不执行岭回归或训练；由预测 NPZ 独立重算误差并核对全部叶模型。所有收益是描述性比较，没有额外显著性检验或普适最优断言。", ""])
    (OUT / "N5完整结果_20261028.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
