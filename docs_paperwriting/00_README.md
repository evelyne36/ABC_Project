# MetaGB AAAI-27 写作材料入口

本目录是论文写作与新工程迁移的**审计层**。范围与迁移已冻结并执行；尚未开始撰写 Introduction，也未生成最终论文表格或图片。

## 当前冻结范围

- 目标：AAAI-27 Main Technical Track，双栏，主内容最多 7 页，总长度最多 9 页；第 8--9 页仅用于参考文献。
- 论文主线：MetaGB 是 XGBoost CPU-hist 中的局部线性叶结构扩展，优势是机制相关而非普遍 SOTA。
- 主实现：`current_metagb`，仅写已验证的 CPU hist、回归、二分类和 scalar multiclass。
- 权威结果根：`datacollection0702/`。
- `datacollection0702/analysis_aaai_0702/` 只是当前叙述快照，不是最终数值源。
- 排除 RealMLP、TabM、synthetic dim50、vector multiclass、未完成的多 horizon 时序结论。
- Synthetic：采用 seed42、5 outer folds、4 inner folds、Random300；只纳入 dim10/dim20。
- Time series：只写 `pred_len=1` 的已有结果。
- 不做显著性检验；可做描述性均值、标准差、W/T/L、coverage 和效果方向，但不得使用 “statistically significant”。
- Introduction 与 Abstract 最后写。
- Ablation 只预留位置，等待作者后续提供同协议记录。
- CatBoost 不补跑；后续使用共同完整单元并明确 N。

## 文件导航

- `00_AAAI27_SCOPE_AND_EXECUTION_PLAN.md`：执行顺序、阶段门和 7 页预算。
- `01_NEW_PROJECT_MIGRATION_MANIFEST.md`：新工程必须迁移、建议迁移和排除的文件。
- `02_EVIDENCE_SOURCE_REGISTRY.md`：每类实验的真实数据源及允许的派生表。
- `03_PROTOCOL_CLARIFICATIONS_AND_MISSING_ITEMS.md`：协议已知项、仍缺项和需要确认的分叉。
- `04_TABLE_FIGURE_PLAN.md`：AAAI 主文表图与真实数据输入要求。
- `05_WRITING_MATERIAL_MAP.md`：现有文档到后续写作材料的提炼映射。
- `06_MIGRATION_COMPLETION_RECORD.md`：实际迁移目录、计数、哈希和验证记录。

## 证据纪律

最终表图必须遵循：

```text
raw log / JSON / CSV
  -> source registry + SHA256
  -> deterministic parser
  -> canonical CSV
  -> table/plot script
  -> LaTeX table / figure
  -> manuscript statement
```

禁止从叙述 Markdown 手工抄数后直接制作最终表格；Markdown 只用于交叉核验和解释。
