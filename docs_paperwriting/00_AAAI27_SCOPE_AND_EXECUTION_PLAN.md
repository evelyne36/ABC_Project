# AAAI-27 范围与执行计划

状态：**范围已确认，Phase 1 迁移完成**  
审计日期：2026-07-14  
目标 track：AAAI-27 Main Technical Track

## 1. 官方约束

AAAI-27 Main Technical Track 官方页面明确：

- 主内容最多 7 页；
- 总长度最多 9 页；
- 超过第 7 页的页面只能用于参考文献；
- supplementary 不保证被 reviewer 阅读；
- 投稿需要完成 reproducibility checklist。

官方来源：<https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/>

因此，方法定义、核心协议、主要结果、适用边界必须在 7 页正文内闭环；补充材料只承载完整数据表、额外图和复现细节。

## 2. 冻结后的论文范围

### 主文纳入

1. `current_metagb` 的 conditional split 与 predictive leaf。
2. XGBoost CPU-hist 集成及 scalar 回归/分类路径。
3. `datacollection0702` 中的真实 tabular MetaGB--XGB 结果。
4. Synthetic dim10/dim20 的 19 个机制、seed42、5×4 nested CV、Random300。
5. 不含 RealMLP/TabM 的强 baseline context。
6. 现有 `pred_len=1` time-series targeted evidence。
7. 参数敏感性；若作者同意补跑，再加入正式组件消融。
8. runtime 与失败案例，明确 MetaGB 的计算代价和选择性优势。

### 主文排除

- RealMLP、TabM。
- Synthetic dim50 的 XGB-only 残余记录。
- 多 horizon time-series 结论。
- vector-leaf / vector multiclass。
- GPU、external memory、原生 categorical 等未验证能力。
- `metagb_pred_value_type`，因为底层实现证据不足。
- universal tabular SOTA、严格算法等价于 XGBoost、速度替代 XGBoost等表述。
- 统计显著性检验和显著性声称。

## 3. 执行阶段与阶段门

### Phase 0：确认范围与迁移策略

交付：本目录中的计划、迁移清单、证据索引和缺口清单。

阶段门已通过：AAAI-27 Main Track；ablation 预留等待作者记录；drop 为主协议且 one-hot 保留；CatBoost 不补跑并使用共同完整子集。

### Phase 1：新工程迁移与证据冻结（已完成）

1. 建立新工程目录：`vendor/`、`experiments/`、`artifacts/raw/`、`artifacts/canonical/`、`paper/`。
2. 迁移完整的 tracked XGBoost fork、实验 runner、测试和配置。
3. 只迁移 `datacollection0702` 白名单证据；排除重复 `datacollection/`。
4. 为所有 raw artifacts 生成 source manifest 和 SHA256。
5. 记录迁移前仓库 commit `af730a29827c79c22ad8836c802fa7cf8db91372`；另行补齐 XGBoost upstream base SHA。

阶段门：每个结果族均能从 manifest 定位到原始记录，且无重复计数。

### Phase 2：重建 canonical 结果数据

实现并验证一个确定性生成器，输出至少：

- `fold_results.csv`
- `paired_fold_results.csv`
- `dataset_level_results.csv`
- `group_summary.csv`
- `coverage_status.csv`
- `baseline_fold_results.csv`
- `baseline_dataset_results.csv`
- `timeseries_predlen1_results.csv`
- `hyperparameter_sensitivity.csv`
- `source_manifest.csv`

每一行至少保存：协议、任务、数据集、预处理、模型、seed、outer fold、指标、方向、source path、source SHA256。

阶段门：canonical 数据重算出的 coverage 与 raw 成功记录一致；表中每个数字可回溯到行级来源。

### Phase 3：Methodology 基础稿

1. 冻结符号、叶函数、二阶目标、split score 和 `score_agg`。
2. 写 conditional/predictive feature 解耦、bias candidate、最终叶重选。
3. 写 CPU-hist 集成、训练/预测差异和支持范围。
4. 写可复现伪代码和保守复杂度说明。
5. 画 Fig. 1 方法示意图。

阶段门：每个算法性句子都有源码行或测试支撑；不声称 pred0 与 XGBoost 训练算法完全等价。

### Phase 4：Experimental Setup

1. 数据集族与实际 coverage。
2. drop 主协议与 one-hot secondary 协议。
3. Random300/600、shared backbone、nested CV 和指标。
4. Synthetic 的 seed、维度、机制、fold、候选预算。
5. Baseline 白名单和共同完成集合。
6. Time series `pred_len=1` fixed-split targeted protocol。
7. runtime 定义分协议说明，不跨协议混合。

阶段门：协议描述与 raw `protocol.json`、`parameter_audit.json`、runner 源码一致。

### Phase 5：Results、表格与图

写作顺序：Synthetic mechanism → real tabular heterogeneity → strong-baseline context → sensitivity/ablation → runtime/failure → pred_len=1 TS。

阶段门：所有表图只读取 canonical CSV；LaTeX 数字与 canonical CSV 自动比对。

### Phase 6：Discussion、Related Work、Introduction

1. 先写适用条件、负例、代价与局限。
2. 完成 closest-work 与 BibTeX 核验。
3. 最后根据已冻结证据写 Introduction 和 Abstract。

阶段门：Introduction 的每个 contribution 都能映射到正文方法或真实证据。

### Phase 7：AAAI 排版与提交审计

1. 使用 AAAI-27 Author Kit 建立 LaTeX 工程。
2. 编译检查双栏、字体、匿名、7 页正文、最多 9 页总长度。
3. 完成 reproducibility checklist、引用核验、数字一致性检查。

## 4. 7 页正文预算

页数包含图表占用，不把关键内容留给 supplement。

| 内容 | 页数预算 | 视觉对象 |
| --- | ---: | --- |
| Abstract | 0.18 | 无 |
| Introduction（最后写） | 0.77 | 无 |
| Related Work | 0.40 | 无 |
| Methodology | 1.60 | Fig. 1，约 0.33 页 |
| Experimental Setup | 0.75 | 紧凑文字，无完整 dataset 表 |
| Results and Analysis | 2.50 | 2--3 张表 + 1 张双 panel 图 |
| Discussion and Limitations | 0.45 | 无 |
| Conclusion | 0.15 | 无 |
| 排版缓冲 | 0.20 | 浮动体与断栏 |
| 合计 | 7.00 | 约 2 页图表 |

## 5. 当前停止点

Phase 1 已在 `project_paperwriting/` 完成并通过迁移验证。下一阶段是 Phase 2：从 raw artifacts 生成 canonical CSV；仍不先写 Introduction。
