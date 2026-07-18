# AAAI 主文表图计划

目标：7 页正文中最多保留 5 个视觉对象；所有对象都绑定 canonical CSV 和 provenance。

## Fig. 1：MetaGB 方法示意图

内容：conditional feature 路由样本；左右叶各自选择 predictive feature 和系数；bias candidate 表示常数叶；标出训练 histogram value 与推理 raw value。

证据：源码，不使用实验数字。

预计占用：单栏约 0.33 页。

## Table 1：Synthetic mechanism evidence

建议列：mechanism family、dim10/dim20、paired folds、XGB metric、MetaGB metric、improvement direction、runtime ratio。

真实输入：synthetic dim10/dim20 的 380 个 task JSON，经 `paired_fold_results.csv` 和 `dataset_level_results.csv` 聚合。

要求：

- 保留 negative controls；
- 不纳入 dim50；
- 标注 seed42、5×4 nested CV、Random300；
- 不写显著性符号。

预计占用：双栏约 0.42 页。

## Table 2：Real tabular heterogeneity

建议行：TabArena main / extra tabular / TabArena30 × task × drop/one-hot。

建议列：dataset N、paired fold N、M/T/X、median improvement、median runtime ratio。

真实输入：tabular raw logs → canonical dataset/group CSV。

要求：TabArena30 单独标 fixed-backbone tail protocol；不得与 full Random300/600 混称统一预算。

预计占用：双栏约 0.50 页。

## Table 3：Strong baseline context

模型：Linear、ExtraTrees、RandomForest、LightGBM、CatBoost、TorchMLP、XGBoost、MetaGB。

真实输入：baseline 白名单 logs 与 MetaGB/XGB canonical common cells。

要求：

- RealMLP/TabM 不出现；
- 逐模型报告；
- CatBoost 使用共同完整集合或补齐；
- 不使用 test-oracle “best baseline” 作为一行模型。

预计占用：单栏或紧凑双栏约 0.30 页。

## Fig. 2：Sensitivity 与 runtime--quality 双 panel

Panel A：`pred_feature_fraction × score_agg` inner-validation sensitivity heatmap。  
Panel B：dataset-level runtime ratio vs performance difference，按 protocol 分面/编码。

真实输入：

- Panel A：`pred_fraction_analysis.csv` 或 synthetic candidate traces；
- Panel B：canonical dataset-level performance 与 protocol-specific runtime。

命名限制：若未补正式 component-off outer-test 结果，Panel A 只能称 sensitivity，不能称 ablation。

预计占用：双栏约 0.48 页。

## Time series 的处理

不单独占主表。正文用一个短段或 Table 2 的 targeted-evidence 子行说明：

- `pred_len=1`，25 个成功结果行；
- 常规树 7/7，MetaGB 4/7；
- 其余 MetaGB cell 显示 NA；
- fixed split，不外推到 long-horizon forecasting。

完整 7-dataset × model 表放 supplement，输入为 `timeseries_predlen1_results.csv`。

## Supplement 建议

- 完整 dataset-level table。
- 完整 coverage matrix 与排除原因。
- Synthetic 19 mechanism × 2 dimension 表。
- baseline common-cell 明细。
- pred_len=1 TS 明细。
- 参数选择频率与额外 sensitivity。
- environment、命令、hash、reproducibility checklist。

Supplement 不承载理解主 claim 所必需的信息。

