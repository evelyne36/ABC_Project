# 工程与文档迁移完成记录

迁移日期：2026-07-14  
源仓库快照：`af730a29827c79c22ad8836c802fa7cf8db91372`

## 冻结决策

- AAAI-27 Main Technical Track。
- drop 为主协议，one-hot 同时保留为 secondary/robustness。
- Ablation 只预留位置，等待作者后续提供真实记录。
- RealMLP 与 TabM 完全退出 migrated runner 和结果日志。
- CatBoost 不补跑，后续只在共同完整单元上报告。
- Time series 只写 `pred_len=1`。

## 工程目录

目标：`project_paperwriting/`

已迁移：

- XGBoost/MetaGB vendor source；
- Random300、Random600、extra tabular、synthetic、六模型 baseline、time-series runners；
- runner tests、配置、requirements 和数据 snapshot；
- `datacollection0702` 白名单原始 artifacts；
- source/dataset/vendor/experiment SHA256 manifests；
- `ccfa.yaml`、提交检查占位和 canonical artifact 空目录。

编译目录和预编译 `lib` 已从迁移工程排除；新工程需要重新构建 XGBoost。

## Raw artifact 验证

- source manifest：3720 个文件；
- Synthetic：380 个 tasks，dim10/dim20 各 190，XGB/MetaGB 各 190；
- Baseline：六模型各 270 个日志，共 1620；成功 1607；CatBoost 257，其余各 270；
- RealMLP logs：0；TabM logs：0；
- Time series：5 个 raw 文件，summary 68 行；`pred_len=1` 为 25 行；
- marker、`.DS_Store`、dim50 synthetic tasks：0。

验证文件：

- `project_paperwriting/artifacts/provenance/source_manifest.csv`
- `dataset_snapshot_manifest.csv`
- `vendor_source_manifest.csv`
- `experiment_source_manifest.csv`
- `migration_validation.json`
- `excluded_source_hashes.csv`

## 工程测试

使用工作区已有 Anaconda Python 执行：

```text
38 passed in 1.99s
```

覆盖：Random300/600 candidate contract、六模型 baseline candidate/schedule、synthetic protocol/scenarios、time-series split 与 leakage tests。

当前活动环境没有在迁移目录内重新编译 XGBoost，也没有重新安装 LightGBM/CatBoost/Torch；这三类 runtime smoke 留给环境冻结与构建阶段。

## 文档目录

目标：`docs_paperwriting/`

- `source_materials/`：方法、score aggregation、搜索空间、数据口径、机制叙事和系统优化参考。
- `evidence_snapshots/analysis_aaai_0702/`：重命名后的 AAAI 0702 叙述快照，仅作交叉核验。
- `protocol_notes/`：ablation 空位、六模型 baseline 范围、`pred_len=1` TS 范围。
- 根目录计划文件：范围、迁移、证据、缺口、表图和后续写作顺序。

## 下一阶段

生成 `artifacts/canonical/`：raw parser、paired fold、dataset level、coverage、baseline common-cell、TS pred_len=1 和 sensitivity CSV。完成后才开始 Method/Setup/Results 写作；Introduction 仍最后写。
