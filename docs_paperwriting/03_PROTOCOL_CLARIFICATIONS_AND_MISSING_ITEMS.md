# 实验协议澄清与缺口清单

## 1. 已澄清，可直接写入 setup

### Synthetic

- 采用 `datacollection0702/synthetic_main1020full_random300`。
- data seed 42；另有 search/CV/model seeds 42/314159/271828。
- 5 outer folds、4 inner folds，属于 nested CV。
- 300 candidates。
- 正式结果仅 dim10/dim20，共 19 × 2 × 5 × 2 = 380 个 model/fold JSON。
- dim50 不纳入。
- 不额外要求多 data seed，也不做统计显著性检验。

### Time series

- 仅 `pred_len=1`。
- seed42、fixed split、outer=1、inner=1、`seq_len=24`。
- 7 个数据集，三类常规树完整，MetaGB 4/7。
- 这是 targeted evidence，不是完整时序 benchmark。

### Baselines

- 删除 RealMLP 和 TabM。
- 保留 Linear、ExtraTrees、RandomForest、LightGBM、CatBoost、TorchMLP。
- 不使用 oracle-best baseline 作为算法。

### 方法能力边界

- CPU hist only。
- regression、binary logistic、scalar multiclass。
- 不写 vector multiclass、GPU、external memory、native categorical。
- bias 说明叶假设类包含常数叶；不说明训练算法严格等价 XGBoost。

## 2. P0：不补实验也必须补齐

这些信息不需要重新跑大规模实验，但迁移或投稿前必须补：

1. XGBoost upstream base commit/tag 与完整 patch。
2. 运行时实际代码 commit；当前可记录仓库快照 `af730a29827c79c22ad8836c802fa7cf8db91372`。
3. 编译器、CMake、OpenMP、构建 flags、`libxgboost` hash。
4. Python、OS、CPU、RAM、BLAS/OpenMP runtime 和线程设置。
5. TabArena/extra-tabular/time-series 数据 URL、版本、license、SHA256。
6. 每类 run 的完整命令、配置、环境变量和硬件。
7. 统一解释 runtime：tabular wall-clock、synthetic candidate+final time、TS `task_elapsed_sec` 分开报告。
8. 生成 0702 raw → canonical CSV 的脚本；当前 `analysis_aaai_0702` 仍是 Markdown-only snapshot。
9. 重算 coverage，不直接相信 `task_status.json` 顶层 completed counters。
10. AAAI evidence snapshot 已在迁移复制品中完成重命名；原始 `datacollection0702` 快照保持不变。

## 3. P0：协议文字必须保守澄清

### Tabular preprocessing

当前 runner 在 outer split 前执行 one-hot/factorize/dropna。论文不能声称 preprocessing 完全 fold-local。

推荐冻结：

- drop 作为主协议；
- one-hot 作为 secondary robustness protocol；
- setup 明确预处理发生位置；
- 不把 drop 与 one-hot 的不重合数据集直接配对。

若作者要求严格 leakage-free preprocessing，则需要修改新工程 runner 并重跑相应实验，不能只改论文措辞。

### MetaGB 内部随机性

C++ updater 内部使用固定 `srand(1)`；实验 `random_state` 不控制全部随机性。论文应分别描述 candidate seed、CV seed、model seed 与 updater 内部固定随机流。

### 训练与预测数值语义

训练系数使用分箱代表值，预测乘 raw feature value。写作中应称 histogram approximation，不写成对 raw-value leaf objective 的严格闭式全程优化。

## 4. P1：已冻结的实验分叉

### A. Ablation

现有数据足够写参数敏感性，但不足以写正式组件消融。

作者决定后续提供 ablation 记录，当前仅保留空位。在新记录到达前，章节和图名统一使用 “Hyperparameter Sensitivity”，不使用 “Ablation Study”，空位中不填写任何数字。

### B. CatBoost 13 个缺失 folds

作者决定不补跑。正式比较只使用共同完整 dataset cells，表头明确 N；CatBoost 缺失单元不得插值或按 loss 计数。

### C. One-hot 定位

已冻结：drop 是主结果，one-hot 同时保留为 secondary/robustness protocol；两者保留独立 coverage 和协议标签。

## 5. 非缺口：不再要求

- 不要求 RealMLP 或 TabM。
- 不要求 synthetic 多 seed。
- 不要求统计显著性检验、多重比较或 CD diagram。
- 不要求 time-series `pred_len=6/12` 或更长 horizon。
- 不要求完成 synthetic dim50。

## 6. Introduction 最后写前仍需补的文献材料

- 原始 MetaGB / model tree / linear leaf boosting 的 closest work。
- XGBoost、LightGBM、CatBoost、TabArena、TorchMLP 等准确 BibTeX。
- claim--citation ledger：每个背景性断言至少一条存在且支持上下文的引用。

这些文献缺口不阻塞 Method/Setup/Results 的证据整理，但会阻塞最终 Introduction 和 Related Work。
