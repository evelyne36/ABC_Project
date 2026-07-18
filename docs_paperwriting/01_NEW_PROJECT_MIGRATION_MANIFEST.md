# 新工程迁移清单

目标：把本项目迁移为可复现、可审计、适合 AAAI 投稿的代码--数据--论文工程。

## 1. 推荐的新工程结构

```text
new_project/
  vendor/xgboost-metagb/
  experiments/
    tabarena_random300/
    tabarena_random600/
    tabular_extra_random300/
    synthetic_metagb/
    strong_baselines/
    timeseries_predlen1/
  datasets/
    manifests/
    downloaders/
  artifacts/
    raw/datacollection0702/
    canonical/
    provenance/
  analysis/
    scripts/
    tables/
    figures/
  paper/
  tests/
  environment/
```

## 2. 必须迁移：方法实现

### 推荐方案

迁移 `beforereview/xgboost-master/` 的完整 tracked 源码树，目标为 `vendor/xgboost-metagb/`。

原因：当前缺少准确的 XGBoost upstream base commit；只复制 MetaGB 改动文件无法保证新工程能在相同基线上编译和运行。

迁移时排除：

- `build/`、编译产物、动态库缓存；
- `__pycache__/`、`.pytest_cache/`、`.DS_Store`；
- 本地生成的临时结果。

### 最小 MetaGB 补丁面

在未来补齐 upstream base SHA 后，可将以下文件维护为 patch series：

- `src/tree/updater_metagb_hist.cc`
- `src/tree/param.h`
- `src/gbm/gbtree.cc`
- `include/xgboost/tree_model.h`
- `src/tree/tree_model.cc`
- `src/tree/tree_view.h`
- `src/tree/tree_view.cc`
- `src/predictor/cpu_predictor.cc`
- `python-package/xgboost/sklearn.py`
- `tests/python/test_metagb.py`

## 3. 必须迁移：实验 runner 与测试

### TabArena classification：`random300/`

迁移：

- `random300/configs/`
- `random300/scripts/`
- `random300/utils/`
- `random300/__init__.py`

排除 `random300/results/`、`random300/logs/`、cache。

### TabArena regression：`random600/`

迁移：

- `random600/configs/default_search_space.json`
- `random600/scripts/`
- `random600/utils/`
- `random600/__init__.py`

`random600/configs/configs_xgboost.json` 不作为当前唯一搜索空间来源；迁移时放入 legacy 或删除引用。

### Extra tabular：`tabular_extra_random300/`

迁移代码、配置和数据 manifest：

- `configs/`
- `scripts/`
- `utils/`
- `selected_manifest.csv`
- `candidate_inventory.csv`
- `openml_download_manifest.json`

原始 CSV 约束：先核对数据许可；优先迁移下载脚本、OpenML ID、版本和 SHA256，而不是把 `data/` 无条件当源码复制。

### Synthetic：`synthetic_metagb/`

迁移：

- `protocols.py`
- `run_benchmark.py`
- `scenarios.py`
- `statistics.py`
- `test_protocol.py`
- `test_scenarios.py`
- `__init__.py`

实际依赖以源码为准：当前 `synthetic_metagb/run_benchmark.py` 复用 `random300.utils.*`，不是 `random600`。

设计报告只迁入 docs/reference；`synthetic_metagb/results/` 不混入代码，正式结果只从 `datacollection0702` artifact 区迁移。

### Strong baselines：`random300_baselinemodel/`

迁移：

- `configs/default_search_space.json`
- `scripts/`
- `utils/`
- `requirements-core.txt`
- `requirements-tree.txt`
- `requirements-deep.txt`（仅 TorchMLP 所需部分）
- `requirements-linux-cpu.txt`
- runner shell scripts

迁移后从模型 registry、默认计划和依赖中删除 RealMLP、TabM；保留 Linear、ExtraTrees、RandomForest、LightGBM、CatBoost、TorchMLP。

### Time series：`timeseries_extra_random300/`

迁移：

- `configs/`
- `scripts/`
- `utils/`
- `tests/`
- `requirements-linux.txt`
- `timeseries_download_manifest.json`
- `selected_manifest.csv`
- `candidate_inventory.csv`

它直接依赖 `random600.utils.candidates`、`random600.utils.parameter_audit` 和 `random600.utils.runner`，因此不能只迁移 TS 目录。

论文运行配置冻结为 `pred_len=1`；代码可保留多 horizon 能力，但论文 artifact generator 必须过滤 `pred_len == 1`。

## 4. 必须迁移：真实结果 artifact

唯一权威根：`datacollection0702/`。

### 主 TabArena

- `binary_drop_0624/`
- `binary_onehot_0624/`
- `multiclass_drop_0624/`
- `multiclass_onehot_0624/`
- `regression_drop_0621/`
- `regression_onehot_0624/`

### Extra tabular 与 tail protocol

- `tabular_drop_20260628_030502/`
- `tabular_onehot_20260628_025442/`
- `tabarena30_drop0702/`
- `tabarena30_onehot0702/`

### Synthetic 白名单

迁移 metadata：

- `synthetic_main1020full_random300/protocol.json`
- `candidate_configs.json`
- `candidate_fairness_audit.json`
- `parameter_audit.json`
- `split_manifest.json`

迁移 tasks：只迁移 dim10/dim20、seed42、`xgb_cpu` 与 `current_metagb` 的 380 个 JSON。

排除 dim50 的 15 个 XGB-only JSON。

### Strong baseline 白名单

迁移 metadata 与以下日志：

- `candidate_configs.json`
- `parameter_audit.json`
- `linear__*.log`
- `extra_trees__*.log`
- `random_forest__*.log`
- `lightgbm__*.log`
- `catboost__*.log`
- `torchmlp__*.log`

成功记录应从 `finish ... status=ok` 重建：五个模型各 270，CatBoost 257；共 1607 条成功白名单日志。

不迁移 `realmlp__*.log`；TabM 无正式日志。

### Time series 白名单

迁移完整原始文件：

- `summary.csv`
- `candidates.csv`
- `candidate_configs.json`
- `parameter_audit.json`
- `split_manifest.json`

canonical 层只保留 `pred_len=1` 的 25 条成功记录。

## 5. 建议迁移：写作与审计材料

- 本 `docs_paperwriting/` 全目录。
- `datacollection0702/analysis_aaai_0702/` 作为叙述快照，不作为数字源。
- `docs/current_metagb_paper_notes.md`
- `docs/score_agg0612.md`
- `docs/超参数搜索空间.md`
- `docs/数据集规模与预处理口径.md`
- `docs/0629数据集描述与规模补充.md`
- `docs/MetaGB论文叙事框架_强baseline压力下.md`

这些文件应进入 `paper/notes/source_materials/`，并标注“待源码/原始记录复核”。

## 6. 不迁移或隔离

- `datacollection/`：与 `datacollection0702/` 重复，避免双重计数。
- 删除后的 `analysis_neurips_0702/`：已废弃。
- `paper_writing/` 中与根目录重复的工程副本：不作为 canonical 代码。
- `metagb/`：旧 standalone 实现，若保留放 `legacy/standalone_metagb/`。
- `beforereview/exp12/`：历史诊断、速度或局部消融，放 legacy，不进主证据。
- `metagb_preflight_adapter/`：可选分析工具，不是主训练依赖。
- smoke、dryrun、nohup logs、cache、临时图、build 目录。
- RealMLP、TabM 代码路径和结果。
- `metagb_pred_value_type`、vector branch、`score_agg=3` 的论文声称。

## 7. 迁移后仍需补齐的 provenance

P0（开始重跑前必须补）：

- XGBoost upstream URL 与准确 base SHA/tag；
- MetaGB 相对 upstream 的可重放 patch；
- 编译器、CMake、OpenMP、Release/Debug、CPU 架构和构建命令；
- 实际加载 `libxgboost` 的路径与 SHA256；
- Python/OS/CPU/RAM/BLAS/OpenMP runtime；
- 带版本锁的环境文件；
- 完整运行命令、`n_jobs`、`task_workers`、线程环境变量；
- dataset URL、版本、license、原始文件 SHA256；
- run commit SHA、配置副本和 artifact manifest。

P1（投稿复现包必须补）：

- 每个表图的生成脚本和 plotting-input CSV；
- failed/pending/excluded 的可机读 coverage 表；
- reproducibility checklist 的逐项映射。

