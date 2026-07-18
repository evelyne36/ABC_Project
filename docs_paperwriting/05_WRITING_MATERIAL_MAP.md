# 现有材料提炼映射

本表说明哪些旧材料值得迁入写作层，以及如何使用。它们不是最终数据源。

| 现有材料 | 后续目标 | 提炼内容 | 限制 |
| --- | --- | --- | --- |
| `docs/current_metagb_paper_notes.md` | `methodology_source_notes.md` | 叶函数、二阶 score、split gain、训练/分类路径、实现 | 必须逐句与当前源码核对 |
| `docs/score_agg0612.md` | `methodology_source_notes.md` | `score_agg={0,1,2}` 与最终叶重选 | 不纳入 `score_agg=3` 主协议 |
| `docs/超参数搜索空间.md` | `experiment_setup.md` | shared backbone、Random300/600、联合搜索 | pred0 合同不能自动视为 0702 正式结果 |
| `docs/数据集规模与预处理口径.md` | `experiment_setup.md` | TabArena 数据、drop/one-hot | 需写明预处理不是 fold-local |
| `docs/0629数据集描述与规模补充.md` | `experiment_setup.md` | extra tabular、TS 数据范围 | 仅用 0702 raw 复核后的字段 |
| `docs/MetaGB论文叙事框架_强baseline压力下.md` | `story_and_limitations.md` | 机制相关优势、非 universal SOTA、结果顺序 | 空表和旧数字不迁移 |
| `docs/METAGB_HIST_BUILD_OPTIMIZATION_PLAN.md` | `implementation_notes.md` | histogram/system 优化设计 | 只写有源码和实际 runtime 支持的部分 |
| `docs_paperwriting/evidence_snapshots/analysis_aaai_0702/*.md` | `results_crosscheck_notes.md` | 当前 0702 叙述、coverage 与正负案例交叉核验 | 已清理复制品命名；仍是 Markdown-only cross-check |
| `docs_paperwriting/protocol_notes/strong_baseline_scope.md` | `baseline_setup_notes.md` | 六模型白名单和 CatBoost common-cell 规则 | 结果以 raw logs 为准 |
| `docs_paperwriting/protocol_notes/timeseries_predlen1_scope.md` | `timeseries_predlen1_notes.md` | one-step scalar forecasting 与真实 coverage | 不外推到多 horizon |

## 不进入写作材料的历史文档

- `docs/实验填表清单.md`：状态过期。
- `docs/答疑0605.md`：含候选独立生成、baseline 状态等历史冲突。
- `docs/datacollection0621_metagb_vs_xgb_analysis.md`：仅保留历史参考，最终结果以 0702 raw 重建。
- `docs/论文写作叙事.md`：旧范围、显著性与篇幅计划已被 AAAI-27 计划替代。
- `random300_baselinemodel/IMPLEMENTATION_RECORD.md`：含已放弃模型，不复制；六模型范围由新的 protocol note 管理。
- `docs/timeseries_window_horizon_protocol.md`：多 horizon 口径不复制；只使用新的 `pred_len=1` note。
- `synthetic_metagb/REPORT.md` 与旧 `results/`：协议不同；不能替代 0702 seed42 Random300 nested CV。
- `paper_writing/` 内重复工程与 smoke 报告：不作为新工程 canonical source。
- 服务器启动指南、nohup 命令、临时 smoke 输出：只在必要时提取运行环境信息。

## 后续写作文档顺序

确认计划后逐步生成：

1. `methodology_source_notes.md`
2. `experiment_setup.md`
3. `results_evidence_notes.md`
4. `discussion_and_limitations.md`
5. `related_work_and_bibtex.md`
6. `introduction_outline.md`
7. AAAI LaTeX sections

Introduction 和 Abstract 只在前五项与最终表图冻结后生成。
