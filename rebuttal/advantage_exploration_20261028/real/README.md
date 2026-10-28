# 真实数据开发包：2026-10-28

已准备 **4 个回归任务 × 2 个 DEV 种子 = 8 个 NPZ/JSON 包**，未运行任何模型拟合或测试评估。这里包含两个此前本地清单未出现的新来源，以及两个已测来源的新目标；不能称为四个新的独立 benchmark。目标、输入政策与划分均在拟合前固定，未依据任何目标相关性、模型误差或优势筛选数据。

## 固定任务与来源

| 任务 | 实际行数 / 输入维数 | train / val / test（112001；112002） | 出处与解释 |
|---|---:|---|---|
| residential_cost | 372 / 103 | 226 / 73 / 73；相同 | [UCI437](https://archive.ics.uci.edu/dataset/437/residential+building+data+set)，V-10 actual construction costs；本地新来源 |
| communities_crime | 1994 / 122 | 600 / 805 / 307；600 / 386 / 307 | [UCI183](https://archive.ics.uci.edu/dataset/183/communities+and+crime)，ViolentCrimesPerPop；本地新来源 |
| parkinson_motor | 5875 / 16 | 600 / 1266 / 1256；600 / 1219 / 1256 | [UCI189](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)，motor_UPDRS；此前 fastlane 已测同源 total_UPDRS |
| gas_turbine_nox | 36733 / 9 | 600 / 7152 / 14542；相同计数 | [UCI551](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)，NOX；此前 fastlane/formal 已测同源 CO |

四份所选来源的官方页面均标注 CC BY 4.0。原始 ZIP、官方元数据 JSON、来源 URL、DOI、许可信息和 SHA256 在 `sources/`、`source_manifest.json` 及每个数据 JSON 中保留。下载时也取得 UCI315 music，但未纳入：无法从该文件获得 artist/album 分组，且上游已整体标准化。该候选档案只用于保留获取记录。

选择理由是事前假设：Residential 的 103 维、小样本与事前成本估计可能适合局部稀疏修正；Crime 提供 122 维、缺失值和跨州迁移检验；Parkinson 是明确的被试外推；Gas 是具有不同工况的真实物理传感测量。它们是否有利于任何模型，仍需完整报告本轮开发结果才能判断。

## 特征与划分边界

- Residential Excel 实际为 4 个日期元数据、103 个输入与 2 个输出；官方“105”并非 105 个预测变量。本任务保留全部 103 个输入，排除 V-9 实际售价，日期仅用于完成季度划分。V-7 实际工期保留，因此任务是**已完工项目的回顾性成本估计**，不能称开工时预测。经济指标为开工前 1–5 个季度。按完成季度完整分块，封存最近 73 行（19.6%）作为 test；其前面的 pool 再按时间分 train/val。两 DEV 数据和划分相同，仅模型随机种子可以不同，不能把两次当两个独立数据重复。
- Crime 使用官方说明中的全部 122 个预测变量。state 只作分组；county、community、communityname、fold 及 target 不进 X。沿用固定 96900 外层随机规则，封存约 20% **州组**，实际是 307 行；剩余州内按组作 DEV train/val。不同州样本量差异导致行数并非精确 60/20/20。上游已作全数据分箱、归一化和极值截断，不能还原原始人口数据，本任务也不能声称 raw-data 完全无预处理信息泄漏。
- Parkinson 保留既有方案的全部 16 个语音输入；subject ID、age、sex、test_time、motor_UPDRS 和 total_UPDRS 均不作为输入。subject ID 只作分组，外层 test 复用旧 96900 被试划分；另一输出 total_UPDRS 被排除。来源是真实录音，但临床标签在访视间经过线性插值。同源相关目标不能视作独立新数据。
- Gas 保留全部 9 个气象/工况传感输入，排除 CO、NOX 和年份。**沿用原已确认 2011–12 train / 2013 val / 2014–15 test**，不因历史结果改成新 20% 划分。官方确认各年文件内按时间排列，未提供具体时间戳。这与 CCPP 来自同一电厂、另一个时段；也不能把它与 CCPP 作为完全独立采集场所。

训练集统一 cap600，仅在大于 600 时无标签均匀抽样；小样本用完整 train。X 中位数填补、X 均值/标准差与 y 均值/标准差均只在**最终 cap 后 train**拟合，再把完全相同变换交给所有模型。每个 JSON 保存完整参数和 train 索引摘要。没有为 CGB 删掉不利特征，没有把分组或已知 oracle 拼入 X。

## 接口与封存

`data/*.npz` 含全行预处理 `X: float32`、互斥 `train/val/test: int64`、`task_type='regression'`、分成员 `y_train/y_val/y_test`、不进入 X 的 `groups` 及 `train_uncapped`。没有 combined `y` 成员。兼容既有 `rebuttal/fastlane_20261027/worker.py`。

搜索必须使用 `load_data(..., evaluate_test=False)`，不读取 `y_test`。预处理及封存程序需从官方档案形成测试标签成员，但未输出测试标签值、分布或模型成绩。最小验证使用读保护：如 worker 访问 `y` 或 `y_test` 即失败；8 包全部通过。检查包括实际 worker 加载、索引/分组隔离、全同 predictor 行无跨集重复、train-only 标准化及源内共享 test。详情见 `verification.json`。

DEV：112001、112002。CONFIRM：113001–113005，仅预留，尚未生成包、未评估。已检索 preexperiments_20261027、fastlane_20261027 和 fastlane_20261028_bc 的 JSON/Python/Markdown/CSV，未发现这些整数种子的历史碰撞。五个未来 CONFIRM 将共享各来源已固定 test；这不是五个独立测试集，也不能增加独立数据集数量。

## 重建

使用具备 numpy、pandas、openpyxl 的 Python 执行 `prepare_real.py`。无需下载、SSH、模型依赖或拟合。程序只写当前 `real/` 子目录，依赖原项目的 `fastlane_20261027/prepare_data.py` 中既有划分和标准化函数；其精确 SHA256 已记录于 manifest。

当前机器已验证的运行时为 `/Users/ancd/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`。`manifest.json` 列出全部 8 包的相对位置和文件校验值。
