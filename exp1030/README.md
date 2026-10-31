# CrossGB: archived pool experiment results (exp1030)

整理日期：2026-10-31。实验结果截止：2026-10-30 07:41（北京时间）。

本目录收集 `rebuttal-expnew-26102914` 的 **pool 原实验**，用于 GitHub 结果归档和研究讨论。原始文件从六台执行实例只读取回，结果与参数文件保持原字节；本次没有重新训练。**不混入 210 接续实验，也不包含其他日期 N1/N2/N5 的原始结果。**

## 从哪里开始

- [完整研究汇报](reports/researcher_report.md)：方法、实验设置、数据描述、精度与成本。
- [按新旧数据来源组织写作](reports/writing_index.md)：论文已有任务与本轮新增任务分开。
- [逐次拟合原始指标](tables/all_trials.csv)：全部11,580条记录，包含超时/失败。
- [逐seed确认指标](tables/confirmation_scores.csv)：从保存预测独立重算的1,019条最终指标。
- [全部确认面板](tables/all_confirmation_panels.csv)：包含完整与不完整面板，不只留前三。
- [公开任务详情](tables/public_datasets.csv)、[48个合成参数场景](tables/synthetic_48_cases.csv)、[来源分组](tables/dataset_origin_groups.csv)。

数据集使用投稿中的英文展示名，并保留精确的 `case` 标识以对应文件。任务说明和报告正文为中文。

## 收录规模

| 项目 | 数量 |
|---|---:|
| 逐次拟合记录 | 11,580 |
| 成功 / 超时 / 失败 | 10,562 / 1,015 / 3 |
| 实际执行任务 | 71：23个公开任务 + 48个合成参数场景 |
| 有确认结果的任务 | 68：67个完整面板 + Food delivery time的不完整面板 |
| 成功最终确认拟合 | 1,019 |
| 最终确认输入seed集合 | 204 |
| 原始记录归档 | 6个 `.tar.gz` |

每个完整面板包含5种方法×3次确认，共15次成功拟合。Food delivery time仅14/15，缺失项为TAO超时。71个执行任务中，Moneyball、Abalone和CPU activity未形成确认面板。

**数据来源两部分：**

- 论文正文/附录已有：本轮18个公开任务，14个完整确认（3第一/7第二/4第三）。这些是已有数据上的补充近邻结果。
- 本轮新增：5个公开任务完成（QSAR-TID-11、Superconductivity、Real Estate Valuation、Combined Cycle Power Plant、Auto MPG；2第一/2第二/1第三）；Yacht Hydrodynamics已准备但未执行。另有新构造6组×8变体=48个合成场景（24第一/22第二/2第三）。

原论文19个合成生成器与本轮6个生成器组有继承关系；本轮同时改变生成式、维度、分布与评价协议，不能当作原场景只扩大样本后的同条件复测。原论文PDF未随包收录，来源文件哈希和任务对应关系见 `provenance/paper_dataset_crosswalk.json`。

## 目录

```text
exp1030/
  README.md
  requirements.txt
  SHA256SUMS
  raw_archives/pool01.tar.gz ... pool06.tar.gz
  tables/
    all_trials.csv
    failed_and_timeout_trials.csv
    confirmation_scores.csv
    all_confirmation_panels.csv
    public_datasets.csv
    synthetic_48_cases.csv
    dataset_origin_groups.csv
    audited_seed_scores.csv
  reports/
    researcher_report.md
    discussion_outline.md
    writing_index.md
    data_notes.md
  provenance/
    collection_manifest.json
    verification.json
    release_checks.json
    environments.json
    case_registry.json
    frozen_parameters_and_source_versions.json
    paper_dataset_crosswalk.json
    public_registry_original20.json
    public_registry_uci4.json
    historical_audit_status.json
    historical_failures.json
    historical_report_checks.json
  tools/verify_results.py
```

`pool01`–`pool06`是独立执行实例的归档标签，并非六次独立重复。具体模型环境、CPU型号和作者代码版本见 `provenance/environments.json`。所有文件SHA256见 `SHA256SUMS`；每个原始压缩包内部另有逐文件的 `EXPORT_MANIFEST.json`。

## 原始文件与复现范围

每个原始压缩包保留以下目录结构：

```text
results/<case>/<stage>/<seed>/<method>/<candidate>/
  result.json                  # 原始指标、状态、有效参数、时间、实现指纹等
  params.json                  # 原始请求参数；包括超时/失败候选
  predictions.npz              # confirm阶段保存的原始预测与行索引
inputs/<case>/<stage>_<seed>.json
summaries/<case>/<stage>_<seed>.json
frozen/<case>.json
candidate_pools_24.json
case_registry.json
scripts/search.py
scripts/worker.py
evaluation_targets/<case>/confirm_<seed>.npz
EXPORT_MANIFEST.json
```

还有实际存在的 `decisions/`、`errors/` 和数据来源registry。`result.json`、`params.json`、`predictions.npz`及元数据为原文件字节；`evaluation_targets`是**本次导出的派生文件**，从原确认输入提取标准化后的`y_test`和`test_indices`，用于离线重算误差。它不含特征矩阵；原输入文件SHA256保留在元数据中。

记录内历史绝对路径用于来源追溯，不能直接当作上传后的可执行路径。原始CSV特征表、完整输入NPZ、拟合模型pickle、原生动态库、SSH连接脚本和凭据均未随包收录。归档内的 `search.py`/`worker.py` 是当时执行的脚本，**不是无需其他依赖即可重训五方法的完整软件发行包**。本目录可独立核对预测指标、参数冻结、候选覆盖和结果完整性；重训需要另行取得相应数据及固定版本实现。

提取某个实例便于查看（可选）：

```bash
mkdir -p unpacked/pool01
tar -xzf raw_archives/pool01.tar.gz -C unpacked/pool01
```

## 独立核验

Python 3 + NumPy：

```bash
python -m pip install -r requirements.txt
python tools/verify_results.py
```

脚本可从任意工作目录执行。它不启动训练、不加载pickle；检查外层文件SHA256及归档内逐文件SHA256，核对确认参数与冻结参数，检查搜索阶段没有test指标，再从全部成功最终预测与导出目标重算RMSE。

本次核验：6个原始归档、11,580条拟合记录、1,019份成功确认预测全部通过；标准化RMSE重算最大绝对误差为0。`provenance/verification.json`保存本次核验结果。

## 指标与实验口径

- 比较方法：CrossGB、GBDT-PL（表中PL）、LinXGBoost、PILOT、TAO。PILOT/TAO为所运行实现的单树；TAO为常数回归叶版本。
- 6配置预筛，潜力案例24配置HPO，validation选参冻结后3次确认。原始candidate池与完整实际状态均保留。
- 公开任务三次确认共享固定test，训练子采样不同；合成三次用新生成seed。不能把共享test重复当成独立外测。
- 原始result中的RMSE为训练目标标准化尺度。`*_rmse_original`及汇总表先乘各输入元数据的`y_scale`再汇总，不直接平均不同seed的标准化RMSE。`Houses`、`Space GA`的归档目标已经取对数，未逆对数。
- `all_trials.csv`保留所有阶段、候选和状态；未完成指标为空，不能按0解释。
- `all_confirmation_panels.csv`包含Food的不完整面板；其`rank_by_available_method_mean`须结合`successful_final_seeds`和`five_method_three_seed_panel_complete`阅读。
- 48个合成场景是相关参数变体，属于6个生成器组，不能视为48种独立机制。
- CrossGB对PL的46/48更低RMSE对应双方每场景24个成功HPO；LinX各合成场景仅4–18/24，超时覆盖不能省略。
- 训练/推理CPU时间为原并发执行时保存的采样成本；未完成隔离计时。前三是留存标准，不保证与最优误差接近，也不构成统计显著性声明。
- `historical_*`文件保留旧审核口径。Kin8nm归入仿真，以及来源分组、排名汇总笔误的更正，以当前报告和`data_notes.md`为准；逐任务分数没有修改。

## 上传 GitHub

本目录已整理为可搬移的相对链接结构；原始归档单文件均小于25 MiB，可使用常规Git提交。上传后如修改任何文件，需重新生成 `SHA256SUMS`，否则核验会报告哈希变化。请上传整个 `exp1030` 目录，保留6个原始归档及清单；`unpacked/`仅为本地查看目录，不需要提交。

发布目标为 [evelyne36/ABC_Project 的 rebuttal-v1031-exp 分支](https://github.com/evelyne36/ABC_Project/tree/rebuttal-v1031-exp/exp1030)。`provenance/release_checks.json`中的提交/推送状态记录打包阶段的检查时点；GitHub提交记录反映后续发布状态。
