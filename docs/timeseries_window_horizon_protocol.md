# Time-Series Window and Horizon Protocol

本文档记录 `timeseries_extra_random300` 中时间序列窗口长度
(`seq_len`) 与预测距离 (`pred_len`) 的实验口径。目标是让树模型的
tabularized time-series 实验在论文中可解释、可复现，并避免把
one-step forecasting 误表述为 long-term sequence forecasting。

## Current Final TS Design Summary

本节是当前 TS chapter 的收束版实验设计，可作为论文实验部分和全量运行脚本的
直接依据。

### Research Question

本章关注的问题是：在标准 long-term time-series forecasting 风格的
chronological split 和 sequence-output horizon 下，将时间序列窗口表格化后，
MetaGB 是否能在相同输入、相同候选预算、相同验证规则下优于 XGBoost。

本章不主张复现 Traffic4cast 式 domain-specific feature engineering，也不把
当前树模型表述为 Transformer/seq2seq 模型。若目标是严格对齐顶会 LTSF
结果表，主实验口径应为：

```text
multivariate-input sequence-output LTSF
with tabularized lag-window features
```

当前代码中的 direct H-step scalar forecasting 不应作为严格 LTSF 主表口径；
它可保留为 tree-native direct forecasting appendix/ablation。

### Dataset Scope

主实验使用 14 个标准 TS/LTSF 数据集：

```text
ETTh1, ETTh2, ETTm1, ETTm2,
exchange_rate, electricity, traffic, solar,
Weather, ILI,
PEMS03, PEMS04, PEMS07, PEMS08
```

严格 LTSF 主表应优先采用与 TSLib `features=M` 更接近的多变量输出设置：
输入包含所有变量，输出为未来 `pred_len` 个时间步上的所有变量，并在未来
序列和变量维度上聚合 MSE/MAE。若出于 MetaGB 适配成本暂时只预测一个目标列，
必须在正文中明确写成 `MS`/single-target variant，而不能声称完全等同
multivariate-output LTSF。

### Feature Construction

严格 LTSF 主协议中，每个 supervised row 由过去 `seq_len` 个时间步展开为
输入，输出为未来长度 `pred_len` 的序列：

```text
X_t = flatten(values[t : t + seq_len, all_channels]) + optional_calendar_features
Y_t = values[t + seq_len : t + seq_len + pred_len, output_channels]
metric = mean MSE/MAE over horizon steps and output channels
```

其中 `pred_len=H` 表示未来序列长度为 H，而不是只取第 H 个点。

现有 direct scalar 协议的定义是：

```text
X_t = flatten(values[t : t + seq_len, all_channels]) + optional_calendar_features
y_t = target[t + seq_len + pred_len - 1]
```

该定义可用于 appendix 中的 direct H-step-ahead scalar target，但不作为
严格 LTSF 主表。

当前框架只使用通用 lag-window flattening 和可选 calendar features。诸如
Traffic4cast 中的 similarity/static/node-flow/segment features 属于交通图
数据的细粒度领域特征，需要 OSM、road graph、loop counter、segment metadata
等额外信息；这不是本章主实验的公平比较范围。

### Split And Selection

TS 主实验采用固定 chronological train/validation/test：

```text
train / validation / test = 60% / 20% / 20%
```

Train 用于拟合模型；validation 用于选择 Random300 候选。`seq_len` 是显式
外层实验维度，不在 validation 内部自动选择；test 只在选择完成后报告一次。
Nested expanding-window CV 保留为 appendix robustness check，不作为 TS 主表。

### Main Grid

严格 LTSF 主表的 `pred_len` 是未来序列长度，使用顶会常见 horizon：

```text
pred_len = {96, 192, 336, 720}
```

`seq_len` 是主实验显式外层维度。考虑 MetaGB 的计算压力，按窗口分批运行，
先完成一个 `seq_len` 的完整表，再进入下一个 `seq_len`：

```text
primary seq_len batches: {24, 48, 96}
first table default:     seq_len = 48
```

主表按 `seq_len x dataset x model x pred_len` 汇报每个 seed 的 test 结果。
为了尽快产出可用表格，先跑 `seed=42, seq_len=48` 下所有数据集、两个模型和
四个 `pred_len`；随后补 `seq_len=24/96` 和更多 seeds。

如果算力无法承受 `{96,192,336,720}` 的完整 sequence-output protocol，可采用
两层汇报：

- Main-LTSF：优先跑 `{96,192,336,720}`，使用较小 seeds 或数据分批完成；
- Appendix-Direct：保留 `{1,24,48,96}` 的 direct scalar horizon，明确标注
  为 tree-native direct forecasting，不与 LTSF 主表混写。

### Model And Search Budget

比较模型为：

```text
xgb_cpu
current_metagb
```

超参候选保持 Random300 原样。`seq_len`、`pred_len`、split protocol、seed
是实验设计维度，不改变 Random300 的候选分布。

### Seeds And Task Count

主实验 seed 作为外层可调维度。当前第一轮默认：

```text
seeds = {42}
```

算力允许后再追加 `{0,1,2}` 或其他 seed 组合报告 mean/std。Fixed runner 的
外层任务单位是：

```text
seed x seq_len x pred_len x dataset x model
```

因此 1 seed、1 seq_len、14 datasets、2 models、4 horizons 的第一张表为：

```text
1 x 1 x 4 x 14 x 2 = 112 reported test rows
```

若后续补齐 1 seed、3 seq_len，则为：

```text
1 x 3 x 4 x 14 x 2 = 336 reported test rows
```

sequence-output 对树模型的内部训练成本显著更高。若用 horizon-wise direct
models 实现，一个报告任务内部近似需要：

```text
Random300 candidates x pred_len x output_channels
```

如果只做 single-target sequence output，则去掉 `output_channels` 倍数；如果
做 full multivariate output，则宽数据集计算量会非常高，需要分批、降 seed、
或先用 XGB profiling 确认可运行范围。

runner 已按 `seed -> seq_len` 分批提交 fixed tasks：一个 seed 下的一个
`seq_len` batch 完成后再进入下一个 `seq_len`，便于尽快先出一个窗口的表。

### Reporting Contract

主表报告：

```text
dataset, model, seed, pred_len, selected_seq_len,
RMSE, MAE, MSE, train time, predict time
```

论文正文按 horizon 分开报告，并以多个 seed 的 mean/std 总结。严格 LTSF
主表的 RMSE/MAE/MSE 必须基于未来整段 `pred_len` 序列，而不是单点
H-step-ahead target。不同 `pred_len` 的分数不混成唯一 headline，除非额外
给出清楚标注的 secondary aggregate。

## 背景

当前实现不是 Transformer/seq2seq 时序模型，而是把历史窗口展开成一行
表格特征后交给 XGBoost/MetaGB：

```text
X_t = flatten(values[t : t + seq_len, all_channels]) + optional_calendar_features
y_t = target[t + seq_len + pred_len - 1]
```

因此当前 `pred_len=H` 表示 direct H-step-ahead single-target
forecasting，即预测未来第 H 个点，而不是预测未来长度为 H 的整段序列。
这点必须在论文中明确写出。

窗口展开后的特征数近似为：

```text
num_features = seq_len * num_channels + optional_calendar_features
```

所以 `seq_len` 对树模型的运行时间和内存压力有直接线性影响；对宽数据集
如 `electricity`、`traffic`、PEMS，长窗口会迅速变成超高维表格问题。

## 文献依据

主流 long-term forecasting 论文和开源实验脚本通常不会只使用一个
固定 horizon：

- Autoformer 在 ETTh1 脚本中使用 `seq_len=96,label_len=48`，
  并测试 `pred_len=24,48,168,336,720`。
  Source: https://raw.githubusercontent.com/thuml/Autoformer/main/scripts/ETT_script/Autoformer_ETTh1.sh
- Autoformer 在 Electricity 和 Traffic 脚本中使用 `seq_len=96`，
  并测试 `pred_len=96,192,336,720`。
  Sources:
  https://raw.githubusercontent.com/thuml/Autoformer/main/scripts/ECL_script/Autoformer.sh,
  https://raw.githubusercontent.com/thuml/Autoformer/main/scripts/Traffic_script/Autoformer.sh
- Time-Series-Library/TimesNet 的 ETTh1 long-term forecasting 脚本使用
  `seq_len=96,label_len=48`，并测试 `pred_len=96,192,336,720`。
  Source: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
- PatchTST 使用 patching 和 channel independence 承载更长历史，在 ETTh1、
  Electricity、Traffic 脚本中使用 `seq_len=336`，并测试
  `pred_len=96,192,336,720`。
  Sources:
  https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/etth1.sh,
  https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/electricity.sh,
  https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/traffic.sh
- Informer、Autoformer、PatchTST 等论文均把 long-term forecasting 作为
  多预测长度问题来评估，而不是只报告 one-step。
  Papers:
  https://arxiv.org/abs/2012.07436,
  https://arxiv.org/abs/2106.13008,
  https://arxiv.org/abs/2211.14730

这些协议给出的启发是：NeurIPS 级别实验不能只报告
`seq_len=24,pred_len=1`。但上述深度模型的 `pred_len` 往往是
multi-output future sequence length；当前树模型实现里的 `pred_len`
是 single-output horizon distance，两者不能混淆。

## 推荐实验口径

### Current Implementation Status

当前 `timeseries_extra_random300` 是多变量输入、单目标输出的 direct
forecasting：

- 每个数据集在 `configs/dataset_registry.json` 中只有一个
  `target_column`。
- `csv_with_date` 数据集使用显式目标列，例如 ETT/Weather/ILI 的 `OT`。
- `plain_numeric_csv` 和 `npz_array` 数据集当前使用 `target_column=-1`，
  即最后一个通道作为目标。
- `utils/data.py::_make_xy` 生成的一维 `y` 只取
  `target[t + seq_len + pred_len - 1, target_index]`。

因此目前代码不是 multi-target forecasting，也不是直接预测未来整段序列。
如果论文目标是严格对齐顶会 LTSF 表格，这一实现状态只能作为 appendix
direct-forecasting variant，主实验需要新增 sequence-output LTSF protocol。

This is a required paper caveat: some LTSF benchmarks report multivariate
sequence forecasting, while the current MetaGB adapter evaluates one scalar
target per row. The experimental section must state the target definition before
comparing against time-series literature or using datasets that are commonly
used in multi-target settings.

### Main Protocol: Literature-Aligned Sequence-Output LTSF

若目标是严格对齐 Autoformer/TimesNet/PatchTST 等顶会 LTSF 表格，主协议
必须把 `pred_len=H` 定义为未来序列长度 H：

```text
X_t = flatten(values[t : t + seq_len, all_channels]) + optional_calendar_features
Y_t = values[t + seq_len : t + seq_len + pred_len, output_channels]
metric = mean MSE/MAE over future horizon steps and output channels
pred_len in {96, 192, 336, 720}
```

首选 output setting 是 `features=M` 风格的 multivariate-output forecasting。
如果 MetaGB 暂时难以承受 full multivariate output，可先实现
single-target sequence output，并在论文中明确标注为 `MS/single-target
variant`，不能声称完全复现 `features=M` 主表。

报告方式：

- 每个 `pred_len` 单独报告 RMSE/MAE/MSE。
- 不把不同 horizon 的分数简单平均成一个主结论。
- 明确称为 sequence-output long-term forecasting。
- `seq_len` 只允许在 validation 上选择；test 只能用于最终评估。

基于 MetaGB 计算压力，`seq_len` 仍采用 width-aware lookback candidates：

```text
small/medium datasets: seq_len in {12, 24, 48, 96}
wide datasets:         seq_len in {6, 12, 24, 48}
pred_len:              {96, 192, 336, 720}
```

Hyperparameter search space note: the XGBoost/MetaGB candidate generation should
remain the Random300 search space. `seq_len`, `pred_len`, split protocol, and
seed are experiment-design dimensions around the existing candidate search; they
should not silently change the Random300 hyperparameter distributions.

### Appendix Protocol: Tabularized Direct Horizon

当前 direct scalar 协议可保留为 appendix 或 ablation。每个任务预测未来第
H 个点：

```text
X_t = flatten(values[t : t + seq_len, all_channels])
y_t = target[t + seq_len + pred_len - 1]
pred_len in {1, 24, 48, 96}
```

该协议必须明确称为 direct H-step-ahead scalar forecasting，不应与严格
LTSF sequence-output 主表混写。它的价值是展示 tree-native forecasting
setting 下 MetaGB/XGBoost 的行为和计算效率。

## Seq-Len Grid

当前建议把 `seq_len` 作为 validation-selection 超参，按数据宽度设置候选组。

### Low/Medium Channel Datasets

适用于 ETTh1/ETTh2/ETTm1/ETTm2/exchange_rate/Weather/ILI 等低到中等通道
数据：

```text
seq_len in {12, 24, 48, 96}
```

对特别低维且样本充足的数据，如 ETT、exchange_rate，可在附录加入：

```text
seq_len in {24, 48, 96, 168, 336}
```

### Wide Datasets

适用于 electricity、traffic、solar、PEMS 等高通道数据：

```text
seq_len in {6, 12, 24, 48}
```

其中 `48` 是主表中最接近文献长输入方向的宽数据上限；若运行时间或内存
压力过大，可在宽数据消融中报告 `{6,12,24}`，但主协议仍记录
`{6,12,24,48}`。

### Frequency-Aware Notes

- 小时级数据：`24` 表示一天，`48` 表示两天，`168` 表示一周。
- 15 分钟数据：`96` 表示一天，`336` 表示 3.5 天。
- 周频 ILI：可额外考虑 `{12,24,52,104}`，其中 `52` 对应一年。

## Reporting Requirements

论文主表至少应包含：

- dataset
- channels
- selected `seq_len`
- fixed or task-specific `pred_len`
- rows after windowing
- feature count after windowing
- model
- RMSE/MAE/MSE
- train and predict runtime

补充材料建议包含：

- `seq_len x pred_len` sensitivity heatmap；
- selected `seq_len` distribution；
- 宽数据集使用 channel cap 或 shorter seq_len 的原因；
- strict sequence-output LTSF 与 appendix direct-scalar forecasting 的严格定义差异。

## How GBDT / XGBoost Is Used In Time-Series Work

GBDT/XGBoost 在时序论文和竞赛中通常不是 seq2seq 生成模型。更常见的做法是：

1. **Lag-window tabularization.**
   把过去若干 lag/window 展成表格特征，训练一个 single-output regressor。
   多 horizon 时通常用 direct strategy：每个 horizon 单独建模，或把 horizon
   作为任务条件，而不是一次生成完整 future sequence。
2. **Feature extraction before tabular regression.**
   Time Series Extrinsic Regression (TSER) 明确定义为从一段 univariate 或
   multivariate time series 预测一个 continuous scalar target；它放宽了
   forecasting 中“目标必须是未来值”的要求。TSER benchmark 将 XGBoost、
   Random Forest、SVR 等作为标准 ML baseline，并显示 ROCKET-style
   time-series feature transforms/adaptations 可以强于直接套普通 ML regressor。
   Sources:
   https://arxiv.org/abs/2006.12672,
   https://arxiv.org/abs/2006.10996
3. **Random/summary interval features.**
   后续 TSER 工作把重点放在特征构造上，例如 DrCIF 从随机区间上提取统计
   summary features，FreshPRINCE 将时间序列转换为大量 summary features 后
   接回归器。这说明树/表格模型在 TSER 中的关键不是 seq2seq decoder，而是
   合理的 flatten/propositionalization。
   Source: https://arxiv.org/abs/2305.01429
4. **Competition-style GBDT forecasting.**
   M5 等 forecasting competition 中，LightGBM/XGBoost 类方法通常把任务转为
   one-day/single-step regression，并依赖丰富的 lag、rolling、calendar、
   price/promotion/static features 以及 carefully constructed validation sets。
   一个 M5 gold-medal range 方法明确描述了“regression on sales for a single
   day + information-rich feature engineering + validation set tuning”的流程。
   Source: https://arxiv.org/abs/2310.13029
5. **Challenge-specific GBDT feature engineering.**
   NeurIPS Traffic4cast 2022 技术报告中的 GBDT 方案使用 similarity/static/node
   flow/segment combined features 后训练 GBDT，且能超过一些 graph-neural-network
   solutions。它同样不是 seq2seq，而是领域特征抽取后表格化建模。
   Source: https://arxiv.org/abs/2211.07031

Implication for this project:

- 当前 MetaGB/XGBoost TS adapter 的 lag-window flattening 是合理的
  tree-compatible input representation。
- 若目标是严格对齐 LTSF 表格，必须在该输入表示之上实现 sequence-output
  target 和整段未来序列指标；不能把当前 direct scalar target 当作 LTSF 主表。
- 如果要更贴近 TSER/GBDT 文献，可在后续 appendix 增加 feature-extraction
  ablation，例如 lag-window flattening vs summary statistics over windows
  (mean/std/min/max/slope) vs random-interval summaries；但主实验不应写成
  seq2seq。

## Split Protocol Decision

### Evidence From LTSF Codebases

Time-Series-Library 的 `data_provider/data_loader.py` 使用固定
`flag in ['train','val','test']` 边界，而不是 repeated outer folds：

- ETT hourly 数据固定为 12 个月 train、4 个月 validation、4 个月 test，
  并在 validation/test 边界前回看 `seq_len`。
- ETT minute 数据同样使用固定年月边界，只是乘以 4 对齐 15 分钟频率。
- custom/LTSF common 数据通常使用 chronological train/validation/test
  固定边界；本项目主实验采用计算更轻且验证集更充足的 60% train、
  20% validation、20% test。
- scaler 只在 train segment 上 fit，再 transform 全序列。

Source:
https://raw.githubusercontent.com/thuml/Time-Series-Library/main/data_provider/data_loader.py

这与 Autoformer、TimesNet、PatchTST 等 LTSF benchmark 的写法一致：
固定 train/val/test split，validation 用于 early stopping 或模型选择，
test 只用于最终报告；多 seed 可用于报告均值和标准差。

### Recommended Choice For This Paper

由于本论文其他 tabular chapter 使用 nested CV，而时间序列顶会 benchmark
通常使用固定 chronological train/val/test，建议采用“双口径但主次清晰”的
设计：

1. **Main TS protocol: fixed chronological train/val/test.**
   这是时间序列章节主表口径，优先对齐 LTSF 顶会实验习惯。
   Train 用于拟合模型，validation 用于选择 `seq_len`、XGBoost/MetaGB
   超参和 early stopping，test 只报告最终结果。每个配置跑多个 seed
   报告 mean/std。
2. **Nested CV: robustness appendix, not main TS table.**
   当前已有 expanding-window nested CV runner 可保留为稳健性分析：
   说明 MetaGB 在严格滚动验证下是否保持趋势。但不要把它作为唯一 TS
   主结果，否则与 LTSF 文献的固定 split benchmark 不完全可比。
3. **Unify wording across chapters.**
   Tabular chapter 可以继续说 nested CV 是表格数据主评估协议；TS chapter
   则说明遵循 forecasting literature 的 chronological split。两者的共同
   原则是：所有模型在同一数据划分、同一候选网格、同一验证集规则下比较，
   test 不参与调参。

### Accepted Main Protocol

若目标是严格对齐顶会 LTSF 表格，当前确定的 TS 主实验协议如下：

```text
split_protocol = fixed_chronological_train_val_test
split_ratio = 60/20/20
seeds = at least two seeds, preferably {0, 1, 2} if budget permits
hyperparameter_search = keep Random300 unchanged

small/medium seq_len = {12, 24, 48, 96}
wide seq_len         = {6, 12, 24, 48}
pred_len            = {96, 192, 336, 720}
target              = sequence output over future pred_len steps
preferred output    = multivariate output (features=M)
fallback output     = single-target sequence output, clearly labeled as MS/single-target variant
```

Reporting rules:

- Report each `pred_len` separately; do not merge horizons into a single headline
  score unless an additional aggregate is clearly labeled as secondary.
- Use validation only for model selection, including `seq_len` selection and
  Random300 candidate selection.
- The main table is not a full `seq_len x pred_len` grid. For fixed split,
  `pred_len` defines the reported task; all allowed `seq_len` values are
  evaluated on validation inside that task, and the selected `seq_len` is
  recorded in the summary/appendix.
- Report test once per seed after selection; summarize with mean/std over seeds.
- Keep the nested expanding-window runner as robustness evidence or appendix
  analysis, not the main TS benchmark table.
- State explicitly whether the output setting is full multivariate sequence
  output or single-target sequence output. This is essential because several
  public TS datasets can be used in both protocols.
- Keep the existing direct scalar H-step protocol as appendix/ablation unless
  the paper deliberately studies tree-native direct forecasting rather than
  strict LTSF tables.

### Practical Implementation Implication

Runner now supports `--split-protocol fixed`：

```text
fixed split:
  train -> fit estimator
  val   -> choose seq_len, pred_len-specific hyperparameters, early stopping
  test  -> final report only
  ratio -> 60/20/20 chronological boundaries
  outer task unit -> seed x dataset x model x pred_len
  inner selection -> seq_len candidates x Random300 candidates on validation

existing nested split:
  retained for appendix robustness checks
```

Implementation status: the runner now supports the fixed split, seed batching,
Random300 candidate selection, validation-based `seq_len` selection, and
strict sequence-output target construction via:

```text
Y_t = values[t + seq_len : t + seq_len + pred_len, output_channels]
```

Use `--target-mode sequence --output-scope all` for `features=M` style
multivariate sequence output. Use `--target-mode sequence --output-scope target`
for a single-target sequence-output fallback. Multi-output regression is
implemented with a column-wise wrapper so XGBoost and MetaGB share the same
protocol; this is compatible but multiplies training cost by the number of
output columns.

Thus the fixed LTSF runner now expands `seq_len` explicitly. For the first
deliverable table, use `seed=42` and one `seq_len` value; after that, rerun with
additional `SEQ_LENS` values. The runner submits tasks in
`seed -> seq_len -> pred_len -> dataset -> model` order.

推荐主实验矩阵：

```text
split_protocol = fixed
split_ratio = 60/20/20
seeds = {42} for first table; add more seeds later
seq_len = first table {48}; later expand to {24, 48, 96}
pred_len = {96, 192, 336, 720}; lighter early run may use {24, 48}
hyperparameter_search = Random300 unchanged
```

如果算力紧张，不建议退回 `{1,24,48,96}` 作为严格 LTSF 主表；更稳妥的
做法是减少 seeds、先固定一个 `seq_len`、或临时使用 `PRED_LENS="24 48"` 出
早期表；也可将 full multivariate output 暂时降为 single-target
sequence-output variant，并清楚标注限制。

Example smoke command:

```bash
python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode smoke \
  --split-protocol fixed \
  --datasets ETTh1 \
  --models xgb_cpu \
  --seeds 0 1 \
  --seq-lens 6 12 \
  --pred-lens 1 24 \
  --max-raw-rows 400 \
  --max-candidates 1 \
  --n-jobs 1 \
  --task-workers 1 \
  --run-id smoke_fixed_protocol_etth1_xgb_20260628
```

Example 14-dataset smoke command with XGB and MetaGB:

```bash
python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode smoke \
  --split-protocol fixed \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate electricity traffic solar Weather ILI PEMS03 PEMS04 PEMS07 PEMS08 \
  --models xgb_cpu current_metagb \
  --seeds 0 \
  --seq-lens 6 \
  --pred-lens 1 \
  --max-raw-rows 120 \
  --max-features 16 \
  --max-candidates 10 \
  --metagb-fixed-score-agg 0 \
  --metagb-score-agg 0 \
  --n-jobs 1 \
  --task-workers 2 \
  --progress-every 10 \
  --run-id smoke_ts14_fixed_622_xgb_mgb10agg0_20260628
```

For this smoke only, `--metagb-fixed-score-agg 0` keeps the 10
`pred_feature_fraction` values `{0.1,...,1.0}` with `score_agg=0`. Full
Random300 runs should omit this smoke filter unless the experiment explicitly
studies that ablation.

Example main-protocol command shape:

```bash
python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode full \
  --split-protocol fixed \
  --target-mode sequence \
  --output-scope all \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate Weather ILI \
  --models xgb_cpu current_metagb \
  --seeds 42 \
  --seq-lens 48 \
  --pred-lens 96 192 336 720 \
  --max-candidates 300 \
  --n-jobs 1 \
  --task-workers 2 \
  --run-id fixed_ts_main_small_medium
```

Wide datasets should use the wide-grid command shape:

```bash
python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode full \
  --split-protocol fixed \
  --target-mode sequence \
  --output-scope all \
  --datasets electricity traffic solar PEMS03 PEMS04 PEMS07 PEMS08 \
  --models xgb_cpu current_metagb \
  --seeds 42 \
  --seq-lens 48 \
  --pred-lens 96 192 336 720 \
  --max-candidates 300 \
  --n-jobs 1 \
  --task-workers 2 \
  --run-id fixed_ts_main_wide
```

Linux helper scripts expose the main grid through environment variables:

```bash
# First fast table: one seed, one seq_len, two lighter horizons.
SEEDS="42" SEQ_LENS="48" PRED_LENS="24 48" \
  bash timeseries_extra_random300/run_linux_comprehensive.sh

# Strict LTSF first table: one seed, one seq_len, four standard horizons.
SEEDS="42" SEQ_LENS="48" PRED_LENS="96 192 336 720" \
  bash timeseries_extra_random300/run_linux_comprehensive.sh

# Later seq_len expansion after the first table is available.
SEEDS="42" SEQ_LENS="24 48 96" PRED_LENS="96 192 336 720" \
  bash timeseries_extra_random300/run_linux_comprehensive.sh
```

The default helper setting is intentionally conservative:

```text
SEEDS=42
SEQ_LENS=48
PRED_LENS=96 192 336 720
TARGET_MODE=sequence
OUTPUT_SCOPE=all
```

## Initial Smoke Runtime Notes

本节记录初步 smoke runtime profile。该结果只用于估计运行压力，不作为论文
质量结论。正式主结果需要 fixed chronological train/validation/test、
至少两个 seed、test-only final reporting 和完整 Random300 候选搜索。

运行环境：macOS 本地环境，Python 3.13.9，单进程 `--n-jobs 1
--task-workers 1`。所有 smoke 均只运行 outer fold 0，使用较小
`max_raw_rows`，因此 wall time 主要用于比较相对压力，不代表正式实验时间。

### Completed Smoke Runs

| run_id | dataset | model | max_raw_rows | seq_len | pred_len | raw_channels | window_features | candidates | inner_folds | wall_time_sec | final_train_sec | final_predict_sec | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke_speed_etth1_s24_p1_xgb_20260628` | ETTh1 | xgb_cpu | 2000 | 24 | 1 | 7 | 168 | 1 | 2 | 2.31 | 0.077 | 0.001 | low-dimensional one-step baseline |
| `smoke_speed_etth1_s96_p24_xgb_20260628` | ETTh1 | xgb_cpu | 2000 | 96 | 24 | 7 | 672 | 1 | 2 | 1.86 | 0.313 | 0.001 | longer history plus direct 24-step horizon |
| `smoke_speed_electricity_s6_p1_xgb_20260628` | electricity | xgb_cpu | 2000 | 6 | 1 | 321 | 1926 | 1 | 2 | 4.85 | 1.193 | 0.001 | wide-data short-window smoke |
| `smoke_speed_electricity_s24_p1_xgb_20260628` | electricity | xgb_cpu | 2000 | 24 | 1 | 321 | 7704 | 1 | 2 | 14.70 | 4.848 | 0.004 | wide-data upper short-window smoke |
| `smoke_speed_etth1_s12_p1_metagb30_tiny_20260628` | ETTh1 | current_metagb | 300 | 12 | 1 | 7 | 84 | 30 | 1 | 142.20 | 1.862 | 0.000 | tiny MetaGB minimum-candidate smoke |

### 14-Dataset Fixed 60/20/20 Smoke

Run:
`smoke_ts14_fixed_622_xgb_mgb10agg0_20260628`.

Settings:

```text
split_protocol = fixed
split_ratio = 60/20/20
seed = 0
seq_len = 6
pred_len = 1
max_raw_rows = 120
max_features = 16
models = xgb_cpu, current_metagb
xgb candidates = 10
current_metagb candidates = 10 pred fractions with score_agg=0
```

All 28 tasks completed with status `ok`. Because this smoke caps both rows and
features aggressively, it is only a pipeline and relative-speed check.

| dataset | xgb test RMSE | mgb test RMSE | lower RMSE | xgb task sec | mgb task sec |
| --- | ---: | ---: | --- | ---: | ---: |
| ETTh1 | 0.633613 | 0.677425 | xgb | 0.48 | 10.97 |
| ETTh2 | 0.370351 | 0.417555 | xgb | 0.39 | 10.68 |
| ETTm1 | 0.288076 | 0.307894 | xgb | 0.18 | 10.99 |
| ETTm2 | 0.172995 | 0.263334 | xgb | 0.19 | 10.30 |
| exchange_rate | 0.764019 | 1.070540 | xgb | 0.21 | 11.83 |
| electricity | 0.522076 | 0.514191 | mgb | 0.39 | 50.80 |
| traffic | 0.505491 | 1.727490 | xgb | 0.41 | 44.80 |
| solar | 0.033520 | 0.288172 | xgb | 0.22 | 33.82 |
| Weather | 0.960425 | 0.842487 | mgb | 0.38 | 49.40 |
| ILI | 0.898690 | 0.699969 | mgb | 0.24 | 9.04 |
| PEMS03 | 1.369550 | 0.660559 | mgb | 0.30 | 45.11 |
| PEMS04 | 1.249330 | 1.596000 | xgb | 0.35 | 49.10 |
| PEMS07 | 0.920019 | 0.616869 | mgb | 0.36 | 42.41 |
| PEMS08 | 2.480950 | 2.514610 | xgb | 0.36 | 51.16 |

Smoke summary: MetaGB has lower RMSE on 5/14 datasets; XGB has lower RMSE on
9/14 datasets. The result should not be used for quality claims because of
the row/feature caps and single seed.

Commands used:

```bash
/usr/bin/time -p python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py --mode smoke --datasets ETTh1 --models xgb_cpu --seq-len 24 --pred-len 1 --max-raw-rows 2000 --outer-folds 2 --inner-folds 2 --outer-fold-index 0 --max-candidates 1 --n-jobs 1 --task-workers 1 --run-id smoke_speed_etth1_s24_p1_xgb_20260628

/usr/bin/time -p python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py --mode smoke --datasets ETTh1 --models xgb_cpu --seq-len 96 --pred-len 24 --max-raw-rows 2000 --outer-folds 2 --inner-folds 2 --outer-fold-index 0 --max-candidates 1 --n-jobs 1 --task-workers 1 --run-id smoke_speed_etth1_s96_p24_xgb_20260628

/usr/bin/time -p python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py --mode smoke --datasets electricity --models xgb_cpu --seq-len 6 --pred-len 1 --max-raw-rows 2000 --outer-folds 2 --inner-folds 2 --outer-fold-index 0 --max-candidates 1 --n-jobs 1 --task-workers 1 --run-id smoke_speed_electricity_s6_p1_xgb_20260628

/usr/bin/time -p python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py --mode smoke --datasets electricity --models xgb_cpu --seq-len 24 --pred-len 1 --max-raw-rows 2000 --outer-folds 2 --inner-folds 2 --outer-fold-index 0 --max-candidates 1 --n-jobs 1 --task-workers 1 --run-id smoke_speed_electricity_s24_p1_xgb_20260628

/usr/bin/time -p python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py --mode smoke --datasets ETTh1 --models current_metagb --seq-len 12 --pred-len 1 --max-raw-rows 300 --outer-folds 2 --inner-folds 1 --outer-fold-index 0 --max-candidates 30 --n-jobs 1 --task-workers 1 --run-id smoke_speed_etth1_s12_p1_metagb30_tiny_20260628
```

### Observations

- XGB 单候选在低维 ETTh1 上非常快；`seq_len=96,pred_len=24`
  的最终训练时间约为 `seq_len=24,pred_len=1` 的 4 倍，但 wall time 受
  Python 启动、I/O 和结果写入影响较大。
- electricity 从 `seq_len=6` 到 `seq_len=24`，窗口特征数从 1,926
  增至 7,704，wall time 从 4.85 秒增至 14.70 秒，说明宽数据上的窗口
  长度必须谨慎设置。
- `current_metagb` 当前最小合法候选数为 30。即使在
  `ETTh1,max_raw_rows=300,seq_len=12,inner_folds=1` 的 tiny smoke 下，
  wall time 仍为 142.20 秒。正式多窗口、多 horizon 协议需要并行化或
  分批排队运行。
- 一次更接近常规 smoke 的 MetaGB 尝试
  `ETTh1,max_raw_rows=1000,seq_len=24,inner_folds=2,max_candidates=30`
  超过交互式 smoke 时间后被中断，未纳入完成表。

### Immediate Implication

建议把后续实现分成两阶段：

1. XGB-only profiling：用 `max_candidates=1` 或极小候选数快速扫描 strict
   LTSF `seq_len x pred_len` 的可运行范围。
2. MetaGB paired runs：只对通过 profiling 的组合运行 30/300 candidates，
   并优先使用并行 worker 或后台批处理；full multivariate output 应在
   single-target sequence output 稳定后再评估是否可承受。
