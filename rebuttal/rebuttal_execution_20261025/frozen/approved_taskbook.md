# CrossGB：AAAI-27 六天 rebuttal 任务书（待确认）

日期：2026-10-25，北京时间。状态：已完成材料阅读与只读核查；**未启动新训练、未修改实验源码、未提交回复**。

目标：用可以改变评审判断的证据争取 qxju、t7wg 的 5→6，改善 jSCP 的 4 分判断，并维护 VVba、wWos 已有认可。以下是行动优先级，不能解释为改分或录用保证。

用户已确认：回复可在字数限制内增加 Markdown 实验表；资源可按 32 核、64GB 内存并行规划。旧建议中“新实验只能作为终稿承诺”的前提已失效。本文作为本轮唯一任务书；旧文档是参考材料，不再构成当前执行指令。

## 1. 以正文为背景，评审到底在质疑什么

正文真正的研究命题是：在二阶 histogram boosting 中，**划分区域的变量与区域内预测变量可能不同**。CrossGB 每叶从候选中选择 `βx_j` 或常数，并用 predictive-aware score 长树。它提供一种稀疏的局部响应约束。

这包含三个需要分别建立的论证：

1. **机制成立**：有些数据确实需要“用一个变量分区、用另一个变量预测”。原合成实验与部分固定骨干消融已支持这一点。
2. **设计有增量**：效果是否超过“直接换成线性叶”、路径限制线性叶、多变量线性叶？这是当前最缺的比较，不能由 XGBoost 对照替代。
3. **代价值得**：在实际数据上，是否能用可观察的结构或验证结果找到收益，而且在同样计算预算下仍有价值？原论文尚未完成这条证据链。

论文已经明确 selective gains，继续重复“不是通用替代品”不会自动解决第 2、3 点。新的实验应排除具体替代解释，而不是扩大数据数量、寻找更多赢的例子。

### 逐位评审的期望与提分条件

| 评审 | 分数/置信度 | 真正需要相信什么 | 本轮应交付的证据与表述 | 主要风险 |
|---|---:|---|---|---|
| **qxju** | 5/3 | 收益不只是已有线性叶容量；新评分在预测叶存在时仍有增量；CPU 大退化有解释 | 完整 2×2 组件表，核心是 **Full vs LeafOnly**；近邻 boosted linear-leaf 比较；CPU 五折诊断。按其三个问题逐项回答 | 仅给 Full vs SplitOnly 没回答问题；仅说“适用性有限”没有解释 CPU |
| **t7wg** | 5/3 | 组合设计有明确价值；知道何时值得尝试；额外计算不是换更多 XGB 计算就能替代；k=1 有合理取舍 | 近邻比较；训练内可计算的机制描述＋真实案例；预算匹配小表；小型 k 容量诊断 | 附录 G 不能当已验证选模器；等候选数不是等算力 |
| **jSCP** | 4/3 | 方法在成熟线性树文献中有明确位置；单变量约束有实证取舍；时序大收益可信 | 至少一个原始近邻实现＋LightGBM linear；k=1/2/3 小表；相同预测目标的 persistence/Ridge；说明 general tabular 与时间序列边界 | 只补 LightGBM linear 后宣称 PL Trees 已完成；用“不是特例”争论代替实际结果 |
| **wWos** | 6/3 | 保留对方法与合成设计的认可，同时确认近邻和成本问题得到实质处理 | 与 qxju 共用近邻证据；最终模型拟合成本、搜索成本分开；低预测候选比例的质量—时间工作点 | 回避真实成本，或把推理多一次乘法当训练成本答案 |
| **VVba** | 7/4 | 叶模型、分类边界、采样层级与符号均得到准确解释 | 解析表达能力例子；`βx OR bias` 的精确定义；树级 Bernoulli 抽样；p=ρp；SplitOnly 定义；承认 CatBoost 排名事实 | 按纪要错误回答“节点级”；声称 bias 候选就是同时拟合局部截距 |
| AI review | 无分数 | 数学解释、实现、图 1、anchor 归因没有硬冲突 | 修正 surrogate 与 exact-gain 的关系、bin/raw 近似；核查图 1；撤除过强 anchor 等价推论 | 因不计分而忽略有效技术问题；把新算法称为原算法的文字修正 |

**资源顺序**：证据可信性先守住；组件与近邻是 qxju 的直接提分抓手；成本与诊断是 t7wg 的关键；容量与时序控制给 jSCP 可重新评价的实质材料。高分评审不需要另开大型实验线。

## 2. 已读材料与本轮核查的新结论

已通读投稿正文 8 页、技术附录 18 页、审稿导出 11 页、会议纪要、metaDesign、Xia 的策略 DOCX、执行与回复建议、A2 草稿，以及两份 ZIP 中的核查报告、面板协议和工具说明；查阅了与任务相关的完整工程文档、原始结果、C++ updater/predictor、消融 runner 和时序数据/划分代码。没有逐行审阅整个 XGBoost 上游工程；没有把历史文档的“已完成”当成本轮验证。

### 2.1 可直接利用的原证据

| 证据 | 位置与内容 | 如何利用 | 不能据此声称什么 |
|---|---|---|---|
| 原正式消融 | `datacollection0702/0716dataAddition/metagb_ablation_syn/syn_formal_formal16_v1/`：14 格骨干、210 任务、3,360 条成功 fold 结果；分析表 672 个模型级 seed-OOF 单元 | 复用骨干、种子、划分、预测 shard；不重跑 300 候选骨干选择；原 Table 2 均值已对齐 | 缺失 LeafOnly 已得到回答；整个 native 训练已原样重放 |
| 正文与附录 D | Regime slope：Bias/Split/Full=0.5380/0.3224/0.3020；Feature switch=0.2257/0.2313/0.2049 | 现有部分归因可保留；说明两个组件作用依赖机制 | 两个组件普遍协同；Full 一定优于 LeafOnly |
| 真实回归原日志 | 本轮解析 400 份 endpoint 日志：40 个计划 dataset×preprocessing 格，39 格完整；CrossGB 16 胜、XGB 23 胜；one-hot 主口径 19 格 | 复用五折、选中配置、失败样本与候选轨迹；解释 39 与 19 的统计单位 | 39 个独立数据集；全部格统一 300 候选 |
| CPU 原五折 | CrossGB/XGB RMSE：3.34586/2.21988，3.56495/2.14182，3.60840/2.09186，4.09179/2.33591，5.06121/2.14461 | 退化五折同向，不是仅单折异常；选中候选的 final fit 可用于低成本重训 | 原因已知为过拟合、噪声或 k=1 |
| 附录 D.4 与候选记录 | 已有 p、aggregation、验证指标与计时信息 | 先做无需训练的验证集 sensitivity 和成本梳理 | 只凭验证曲线得到 held-out test 等算力曲线 |
| 附录 G 与 preflight 工程 | 单变量残差结构、解耦、异质性等训练内分析已有实现 | 复用特征计算，作为解释性 probe | 已验证可预测真实数据的胜负；独立 pilot p=.4 不能支持此结论 |
| 时序数据与 runner | ETT、ILI、Exchange 原文件在 full project 的若干镜像存在；window_summary、sequence target、scaler 与 gap 有实现/manifest | 不先下载新版本；恢复原 10 任务输入并加入参考模型 | 同名文件就证明训练当时 raw hash 完全一致 |

### 2.2 重要更新：Exchange h=12 的后期完成记录找到了

新发现位置：

`aaai_FullCodeProject-check_codereview0701/datacollection0702/0716dataAddition/ts0716/未命名/`

这是一份 **2026-07-24** 的后期运行，旧 `task_status.json` 对应早期快照，仍写 pending。新目录包含 summary、300 行 candidates、candidate configs、metadata、参数审计和 split manifest。本轮已核对：

- 300 候选均为 ok；唯一 selected 为第 267 项，也是 validation RMSE 的最小项。
- 最终 test RMSE=**0.0736811422**，匹配投稿表的 0.0737。
- seq24、h12、270 输入维、12 输出维、监督行数、scaler 边界与三个传统树 h12 对照一致。
- 新运行 `max_features=0`，旧对照为 64；实际八个原始通道、270 输入维相同。应按每次运行记录描述，而非声称 CLI 每字段相同。
- 合并原九项后，重新计算得到 **8/10 胜，平均相对 RMSE 降低 31.0283%，中位 35.4313%**，与正文一致。

**决策：不执行旧 A2 的“撤回 h12/8/10/35.4%”分支。** 现阶段记录层面已有支持；仍须在执行阶段重放 selected 配置、保存逐样本预测，并核对输入。不能把本轮只读核查写成“所有时序实验已独立复现”。

### 2.3 必须修正的解释与工程前提

1. **300/600 搜索口径**：21 个完整 TabArena 回归格实际是每模型 600 候选，额外集合 18 格是 300。投稿附录统一称 300 不准确；配对格内双方预算仍匹配。缺失格是 one-hot Food Delivery Time，只有启动日志，没有完成指标；没有证据判断未完成原因。原 Wilcoxon p 值的最终 canonical 生成链尚未独立复算。
2. **anchor 的 0.05%**：逐 seed 相对差先在格内平均，再跨 14 格取有符号中位数，确为 0.04893%；先逐格取绝对值后的中位数为 **3.108%**，格间范围约 −11.03% 至 +10.54%。原“因此 anchor 接近”推论过强。Categorical dim10 的约 95% 描述性分解成立，不能归为已证实的浮点误差。
3. **源码版本**：原消融记录中的 updater hash 精确对应 `project_paperwriting/vendor/xgboost-metagb/`，而非当前 `beforereview` 副本；两份 Python 执行文件也与历史 hash 不同。原二进制未恢复，必须重建并先做 replay。MetaGB→CrossGB 名称相似不代表运行等价。
4. **抽样确为树级**：投稿 updater 在树生长前生成候选池，再传给后续节点；每叶在池内独立择优。纪要“节点级抽样”不符源码。Bernoulli 纳入也不等于固定 `ceil(ρd)` 个特征；池独立抽样不等于不相交。
5. **训练 bin / 推理 raw**：`TrainValue()` 使用 bin representative，CPU predictor 乘原值。这是明确近似；不能说原值输出的精确最优系数已经由该公式证明。
6. **分裂 surrogate**：现有 `max_j[S_L(j)+S_R(j)−S_P(j)]` 不等于 `max_j S_L(j)+max_j S_R(j)−max_j S_P(j)`。后者是独立选叶的 binned 二阶分数变化；更换准则是新方法变体。
7. **默认分裂阈值**：当前参数代码实际默认 −1e10，文字说明却提零。历史实验未显式覆盖时不能按标准正 gain 解释。新实验若统一正阈值，全部四格必须重跑并标注为新受控协议。
8. **vector 不等于 k>1**：`leaf_features_vector` 为每个输出/类别存一个预测变量，并不是一个标量叶子存 k 个变量。旧建议不能直接拿它做 k=2/3。
9. **图 1 尚未闭合**：已目视核对冻结 PDF 的图及图注，确有 η=.1 与 one-tree noise-floor 并列；在自定义 Python/TeX 检索中未找到原生成器。不能未经依据称其只是 caption typo。停止用该精确树数说服评审，执行阶段恢复或更正。

上述只读核查的数值、路径和 hash 保存在 [证据核查.json](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/rebuttal_plan_20261025/证据核查.json)。

## 3. 核心实验任务与验收

### E0｜冻结证据、历史端点重放与技术澄清（最高优先）

**目的**：保护可信性，同时为后续消融和参考模型提供可靠起点。

- 冻结三份 PDF、找到的 h12 完成记录、原骨干和 source manifest；保留原目录不覆盖。
- 在独立实验副本中，以历史 vendor updater 为起点；先重放 Feature switch/dim10/seed1001/outer1 的 XGB、Bias、SplitOnly、Full 四个端点。核对数据与 split hash、参数、RMSE 和逐样本预测，不先加入 child-max、改正则或改采样。
- 同平台可比较时预设逐样本浮点容差；跨平台先报告预测差与 RMSE 差的量级，不以“方向相同”代替兼容性。重放不符先排查载入库、Python 历史差异和随机状态；最多用首日半天定位。
- 若历史兼容无法闭合，后续全四格作为**新受控组件面板**，绝不拼旧 Full 与新 LeafOnly。
- 重放 h12 selected 配置，无需重跑 300 搜索。查图 1 原脚本；未恢复时准备撤除精确单树结论的更正文字。

**交付**：证据索引、replay 差异表、实际 native 库路径/hash、确定哪些结论可引用。完成条件不是“所有旧代码原样恢复”，而是每项拟用于回复的证据边界清楚。

### E1｜补齐标准评分＋预测叶：完整 2×2（qxju 的首要任务）

| | 常数叶 | 预测叶 |
|---|---|---|
| bias-based/常数分裂评分 | Bias | **LeafOnly：新增** |
| predictive-aware surrogate | SplitOnly | Full |

**最低范围**：Regime slope、Feature switch、Curved control，dim10、原 n=3000、seeds1001/1002/1003、五折；每场景 60 fits，三场景 **180 fits**。三种机制在看新结果前冻结，包含正例与负控。

**推荐范围**：扩到原七场景 dim10，**420 fits**；如同时展示 vanilla XGB，另加 105 fits，共 525。扩展只由时间/资源决定，不由某场景结果是否有利决定。dim20 不列为本轮默认任务。

实现要求：

- LeafOnly 在**每一轮**用常数评分建树、在当前梯度上重新拟合预测叶并更新 margin，不能最后一次性替换整个 XGB 集成的叶子。
- 同一 updater 内分离“评分候选池”与“叶响应候选池”；保留关闭开关时原行为。四组的行、条件池与 predictive 原始抽样序列须一致，再按组件启用情况限制候选。固定骨干，不做 HPO。
- 历史兼容线先只加必要开关。若采用 p=1 Bias 和正 gain 的新控制，必须单独标明阈值、gain 单位、有效参数，四格整体重跑；不要声称与原 Table 2 相同。
- 以 seed-OOF loss 为主单位。报告 `Full−LeafOnly`、`Full−SplitOnly`、`LeafOnly−Bias`、`SplitOnly−Bias`；交互项用同一单位的绝对 loss，不能混加相对百分比。

**验收**：四格完整、切分和骨干匹配、逐样本预测齐全、每格有效参数可追踪。**不要求 LeafOnly 必须介于 Bias 与 Full 之间，也不要求 Full 获胜。** 三个种子的区间仅作探索性描述。

**结果决定表述**：Full 优于 LeafOnly 才能在对应机制上声称评分有额外收益；相当则将该机制收益归于叶响应；更差则报告冲突，并限定新评分的作用范围。

### E2｜近邻 boosted linear-leaf 比较（四位评审共享的证据缺口）

**顺序**：LightGBM constant/linear 先形成可靠对照；同时给**原作者 GBDT-PL 或 LinXGBoost** 一个限时集成任务。PILOT 是补充模型树参照，不取代 boosted 近邻；TAO/CLARITree 首先做文献定位，六天内不为凑名称强行加入五种大型实现。

LightGBM `linear_tree` 按常规方式划分，用分支数值变量拟合线性叶；作者仓库也称它是 GBDT-PL 的近似版本。因此**不能把它标作原始 PL Trees 已完成**。[官方参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html#linear_tree)、[GBDT-PL 作者仓库](https://github.com/GBDT-PL/GBDT-PL)、[原实现说明](https://github.com/lightgbm-org/LightGBM/pull/3299)。

**数据选择，提前冻结**：

- 合成：Regime slope、Feature switch、Sparse multiplicative、Curved control，dim10；与 E1 共享可用输入/骨干，但 HPO 比较和固定骨干消融分开。
- 真实回归主面板：**Kin8nm、Moneyball、CPU activity、Houses、Energy efficiency**，分别覆盖两种原胜例、两种明显负例和一项弱差异案例。它们是从原结果选出的诊断面板，不估计总体胜率。
- 分类扩展：ILPD、Phoneme、Credit approval，优先使用明确支持 logistic 输出的近邻。不要强行把回归实现改成分类器后仍用原方法名。

**协议**：保留原五个 outer folds；在每个 outer-train 内三折 inner CV、每模型 30 候选、独立合理搜索空间、验证选优后 refit。编码/scaler 在 inner-train 内拟合。CrossGB/XGB 也按同一新小预算重跑，原 300/600 结果只作背景。LightGBM constant 与 linear 应独立建 Dataset，避免参数缓存混用；linear 的尺度处理与 `linear_lambda` 明确记录。

五个真实回归数据：每模型 `5×5×(30×3+1)=2,275 fits`；先跑 XGB/CrossGB/LGB constant/LGB linear 四模型，为 9,100 fits；原始近邻通过集成后再加，不能把不同预算默认为公平。若 pilot 表明超时，预先统一缩为 12 候选并**全部模型同改**，保留缩预算说明。

合成近邻面板单独采用开发集选配置：每个机制用独立 seed9001 的 3-fold、30 候选/模型，随后冻结配置，在 seeds1001/1002/1003 的五折上评价。四机制每模型 `4×(30×3+3×5)=420 fits`；默认四模型共 1,680 fits。这个协议与 E1 的 XGB-selected frozen-backbone 归因不同，表格分开，不拼为原 Table 1 的复现。

**原始近邻限时门**：源码/API/目标/预测步长先核查，半天内完成 smoke，否则记录具体障碍；保留 LightGBM linear 结果并明确未覆盖的原方法。LinXGBoost 原仓库有独特的最终树 shrinkage 处理，不能无声“修正”为自己的训练语义。[作者仓库](https://github.com/ldv1/LinXGBoost)。

**交付**：预测误差＋fit/search time＋参数/叶模型说明的紧凑比较表。若近邻更强，就报告 CrossGB 的具体稀疏工作点或承认目前优势局限，不用额外挑选数据洗掉负结果。

### E3｜真实收益、失败归因与可观察机制（qxju/t7wg）

与 E2 的数据、模型输出共享，尽量不重复拟合。

1. 从原结果提取各 fold 的 selected 参数、inner validation 与 test、候选失败/异常值比例、p、aggregation、实际维度及时间。CPU 的第一折候选轨迹已出现大量极大验证 RMSE，值得检查稳定性；这仍不是已证实的根因。
2. 保存 Full/LeafOnly/SplitOnly/Bias 的模型与预测。记录非 bias 叶覆盖率、按训练样本量加权的**预测变量不在该叶路径中**的比例、系数/贡献幅度、训练支持外的预测误差、训练—验证—测试差距。比例的分母必须说明；浅树路径短会自然抬高“路径外”比例。
3. 用 outer-training 的便宜浅分区 probe，测单变量残差投影强度、前 1/前 3 变量的解释集中度、不同区域最佳变量的差异；与已观察收益作**回顾性解释**。不再调一套 suitability 权重来拟合同一批胜负。
4. CPU 的最小受控诊断：保持原选中骨干，比较偏置/预测叶；固定原路径及选中变量时做 raw-coefficient refit；在 inner validation 上比较增加叶正则或最小叶样本的有限设置；必要时分开测试尺度标准化。一次改变一个因素，负例与胜例使用相同诊断流程。

**结论边界**：只有干预改善且相应异常减少，才支持具体解释；相关性、叶选择频率和事后对齐只能说明机制相容。即使找不到单一根因，也交付“哪些解释被排除/仍未知”的诊断表，不硬编故事。

**额外成本**：原记录分析零训练；模型导出通常只需每个 selected 模型一次重训。先 CPU/Kin8nm 两数据完整五折，再加 Energy/Houses；不重启整个 benchmark。

### E4｜算力匹配与预测候选数的成本（t7wg/wWos）

**原文不能回答的核心**：同 300/600 候选与最终模型同树数，均不是等计算预算。

- 先整理现有 trace 的累计实际 fit/predict 时间与 validation incumbent；缺少中间候选 test 预测时，只叫**验证预算轨迹**。
- 推荐在 E2 的 Kin8nm、CPU、Energy 三数据做新预算面板。单模型单线程、同机器；每 outer fold 预先设 60/300/900 CPU 秒三个检查点，检查点取预算内完成的候选，用 inner validation 选优；对去重后的 incumbent 补 outer-train refit 和一次 test。
- 如果首轮 CrossGB 候选完成不了最小检查点，统一向上调整并冻结所有模型的检查点，再开正式比较；不依据 test 曲线改预算。900 秒/outer fold 是搜索预算，验证所选最终 refit 另记。
- 每个候选的完成时间计入完整 inner folds；超预算未完成候选不作为已取得结果。多 worker 的忙机 wall time 不等于 CPU 预算；以进程 CPU 计时为主要匹配尺度，并做少量串行计时校准。
- XGB 的候选空间覆盖更多树/深度与更广正则，至少给它实际使用相同预算的机会。不固定“3000 次就是等算力”。
- 另测最终 selected 模型的 fit/predict 时间、峰值 RSS、树数/叶数；p=.1/.2/.5/1 的少量固定骨干诊断说明实际候选 P 与成本关系，不做大维度 sweep。

**预算量级**：若两模型三数据五折均跑满 900 CPU 秒，搜索共 27,000 CPU 秒，约 7.5 CPU 小时；不是 7.5 小时 wall time，数据准备/refit/校准另计。

**交付**：每个数据一行的预算点结果、质量—CPU 秒表、最终 fit 与 search overhead 两种口径。若 XGB 在等预算下抹平收益，保留该结果并停止声称对应数据的优势来自更划算的计算。

### E5｜时序可信性：persistence、Ridge 与少量时间起点（jSCP）

**固定原面板**：ETTh1/2、ETTm1/2 的 h1；Exchange、ILI 的 h1/6/12；seq24，共 10 任务。h>1 的原目标是未来整段序列，不只是 h 时刻终点。

- Persistence：各步重复最后已观测的 target；Ridge：同原 24 步窗口、summary 和日历输入，多输出分别预测各 horizon，alpha 只由 train→validation 选择；然后 fresh refit(train+val)，test 一次。
- 复用原 loader、target index、60/20/20、sequence 的 h−1 gap、scaler 边界与单位。展示 standardized RMSE 并保存 raw-unit 指标，避免错用多通道平均。
- 优先改造 Xia 独立参考 runner 的路径适配；它不需要 CrossGB native。原包装布局和当前 full project 不同，尚不能直接执行其示例中的旧绝对路径。
- 原 h12 运行的 max_features=0，其余旧任务多为 64；按各任务真实配置重建，不一刀切覆盖。新参考模型与 CrossGB selected 配置同输入重放，保存可复算逐样本预测。
- 对 **Exchange h1、ILI h1、ETTh2 h1** 加三个 expanding/rolling origins：在原 test 之前的可用历史段内预先定相同长度评估块，避免因已知旧 test 而再调配置。每个 origin 仅用其过去数据做验证选优，不能把原 full-data best 参数直接称为新 origin 的独立 HPO。采用统一小候选预算；若用 frozen-config robustness，配置只能来自最早 origin 的过去验证段或独立开发资料，不能来自较晚时期的原 selected 配置。保留三任务所有结果。
- 重叠预测窗口有相关性；按 origin、系列/任务报告，不把窗口条数当独立 repeats。若做区间，需时间块方法；样本不足时优先逐 origin 表。

十任务便宜参考面板：10 个 alpha＋persistence，约 **130 次 panel 级 fit**；Ridge 多输出的底层拟合次数另记。再加原 selected 树重放，不重跑数天的 Random300。rolling 三起点是第二阶段，不阻止固定面板先出结果。

**判读**：若 CrossGB 只胜传统树、未胜 Ridge/persistence，应准确限定原 35.4% 的参照集合；若跨 origin 仍胜这些参考，才增加 forecasting 实用价值的力度。分析 lag 选择与外推误差是机制解释，不能只靠“大收益”证明原因。

### E6｜k=1/2/3 与截距：限时容量诊断（jSCP/t7wg，VVba 的补充）

不能简单放弃该问题，也不在六天内重构生产模型格式。建议先做 **最多一天开发、半天运行** 的小型 stage-wise 诊断；如前四条关键证据尚未闭合，限时八小时开发后停止。

- 数据：Regime slope、Sparse multiplicative、Curved control，加 CPU、Kin8nm。包含单变量机制、多变量需求和负控。
- 首选固定训练分区的逐轮 leaf-refit probe：在训练数据得到并冻结每轮路由后，对当前 residual/gradient 重新拟合 k=1/2/3，使用 `−(ZᵀHZ+Λ)⁻¹Zᵀg`；变量选择只看训练数据，正则仅 inner validation 选。
- k 比较保持截距策略相同；`βx OR bias` 与 `α+βx` 另列一项，避免把截距改善误当多变量改善。需要原生 CrossGB 等价性时，先加 binned k=1 控制；raw-refit 家族整体单列。
- 这种 probe 只回答**固定路由下的叶容量**，不等价于完整 k>1 CrossGB 的重选分裂。若能在已有接口内做正确的 stage-wise 全模型变体，再升级结论；不把 probe 包装成完整算法消融。
- 小面板先用三折/三 seed（真实数据用固定三个 split seeds），保存每格全部结果；报告 loss、系数数/有效叶容量、fit time/RSS。不给 k=1 预设胜出要求。

**若 k>1 更好**：一叶一个连续系数仍是原架构事实，不能说“主张崩溃”；应承认表达限制，并说明 k=1 相对于误差/复杂度的实际工作点。若没有有意义的取舍，就不主张 k=1 的优越性。LightGBM 多变量叶的胜负不能替代这个控制实验。

### E7｜surrogate/bin/raw/图 1 的针对性核查（技术风险控制）

- 解析上验证当前 score 与独立选叶 score 的区别，修订 exact-optimum 的适用对象和 Hessian/分母条件。γ、1/2 与接受阈值统一单位。
- 在 E1/E3 模型中记录已接受 split 上两种 score 符号不一致的情况；只做诊断。若要测试 child-max，单独加一列并注明新算法，不捆绑进 LeafOnly 最小补丁。
- 固定路由/预测变量比较 binned 拟合与 raw refit 的 held-out 误差，不能声称这使原分裂搜索变成 raw 精确优化。
- 图 1 若未找到原生成器，可做一个**明确标为新协议**的 folded-line replay，用相同 η=.1 比较实际收敛树数；新图不能倒推证明旧图 caption 的真实参数。

**停损**：不让 child-max 的额外方法开发挤占 E1/E2/E4/E5。技术解释中有效的限制必须说明；实验是否改善决定如何呈现，不决定是否承认数学事实。

## 4. 实验队列：按依赖和资源分，不按“跑不跑 CrossGB”误分类

| 队列 | 任务 | 是否需要新 CrossGB 训练 | 优先级 | 主要依赖 | 六天内最小交付 |
|---|---|---|---|---|---|
| 原记录分析 | E0 证据冻结、原表/候选/预算统计 | 否；replay 另计 | P0 | 本地记录 | 已有证据表＋源版本界定 |
| 原兼容性 | E0 四端点与时序 selected 重放 | 是，但无搜索 | P0 | Linux build、明确库路径 | 可追踪 replay 差异 |
| 组件 | E1 LeafOnly 与四格完整面板 | 是，固定骨干 | P0 | replay 或明确新协议 | 三机制 180 fits；推荐七机制 420 |
| 近邻 | E2 LGB linear、原始近邻 | 基线本身不需要；公平新预算需配对重跑 | P0 | 原 split、baseline smoke | 五真实回归案例＋四合成机制 |
| 失败/机制 | E3 | 主要复用 E1/E2；少量重训 | P1 | 模型与原 trace | CPU 对照、正负案例机制表 |
| 计算预算 | E4 | 是，小规模有上限 | P1 | 正确 CPU 计时、验证选优 | 三数据三预算点，test refit |
| 时序参考 | E5 | Persistence/Ridge 不需要；配对树 replay 需要 | P0 | 原数据/target/scaler 对齐 | 全十任务参考表；三任务 origins 后续 |
| 叶容量 | E6 | 诊断会拟合叶模型；完整算法才依赖原生扩展 | P1，限时 | 可复用模型路由 | k=1/2/3 的诚实小表 |
| 方法核查 | E7 | 解析部分不需要；受控 refit 少量需要 | P0 文字/P2 新方法 | E0、导出的结构与统计 | 明确近似与图 1 处置 |

## 5. 六天排期和并行预算

所有日期为北京时间；以 10-30 交付可提交稿、10-31 缓冲为目标。官网当前 author feedback window 为 10-24 至 10-31 AoE；按其 AoE 口径推算最后一分钟为北京时间 **11-01 19:59**，最终以本论文表单时间为准。[官网时间表](https://aaai.org/conference/aaai/aaai-27/submission-instructions/)

| 日期 | 首要产出 | 可并行工作 | 当日停止条件 |
|---|---|---|---|
| 10-25：阅读/确认 | 本任务书与 evidence audit；用户确认范围 | 已完成 h12 与原结果只读核对 | 等用户确认，不启动新 fit |
| 10-26 | Linux 构建、四端点 replay、E1 三机制；十任务参考先出 | 近邻实现 smoke；原候选与 CPU 日志分析 | 下午锁定唯一 source/library；不同时修多份 updater |
| 10-27 | 推荐七机制 E1、E2 主面板运行；E5 fixed panel 收齐 | E4 预算面板；真实模型导出 | 不再扩大数据集合；原始近邻集成失败即说明范围 |
| 10-28 | E2/E3/E4 完成；E6 限时容量诊断；rolling origins | 技术澄清与各评审证据卡 | 晚间冻结新任务；不再启动需超过一天的大搜索 |
| 10-29 | 表格、全部不利结果、逐位回复初稿 | 字符与数字核验、作者审阅 | 每个表格数字绑定源记录；清除未完成承诺 |
| 10-30 | 作者可直接审阅的最终回复文本 | 平台字符/Markdown 预检 | 用户最终审阅后再提交；本任务不自动外发 |
| 10-31 | 缓冲 | 仅修复事实/格式或已排队缺格 | 不用最后一天赌大型新实验 |

**32 核/64GB 调度**：默认每模型 `n_jobs=1`，OMP/BLAS 线程均为 1；先 4 workers 观察 RSS 和吞吐，再升至 16，内存允许时至 24。总常驻内存目标不超过 48GB，剩余供系统/数据/编译。不要直接按 32 个最大模型同时跑。任务间并行，避免多层 CV 内部再开线程；CPU 定时队列独立校准。

原 dim10 消融的单个 Full fit 中位时间约：Feature switch 5.3 秒、Regime slope 44.4 秒、Sparse multiplicative 254.3 秒。这些是历史机器观测，只用于排序，不直接作为新服务器 ETA。真正瓶颈更可能是实现接入、协议对齐与近邻适配，而不是再申请 GPU。原时序 Random300 的部分任务耗时数天，因此本轮只复用 selected 配置和小预算控制。

## 6. 回复应准备成什么样

暂按 Xia 邮件摘记的每条 1,500 **字符**规划，预留 Markdown/换行余量，正文目标 ≤1,350 字符；最终以表单实际限制为准。用户已经确认可放 Markdown 实验表。先准备完整内部证据表，再给每位评审抽取 2–4 行，不把大表塞进每条回复。

### 6.1 五位评审的回复骨架

**qxju：三个问题三段，核心表是四格。**

> We completed the missing stage-wise constant-scoring/predictive-leaf comparison under a shared frozen backbone. [Compact Bias/Split/Leaf/Full table.] Full versus LeafOnly tests the incremental value of the split surrogate; Full versus SplitOnly tests the leaf response. [Nearest-neighbor result.] On CPU, all five original folds deteriorate; [verified intervention and its result], which supports [bounded explanation].

方括号是任务书中的待填项，不是可提交文字。若没有根因证据，把最后一句换成可排除与尚未知的原因，不能直接填“overfitting”。

**t7wg：按其四个提问顺序。**

> Observable structure: [training-only probe and positive/negative cases], an exploratory cue rather than a calibrated selector. Cost: [matched CPU budget results], separating search from final fitting. Novelty: the specific combination within second-order histogram boosting, evaluated against [actual near methods]. Capacity: [k comparison]; k=1 is a sparse operating point, not a claimed optimum.

避免说“只要结构存在成本就合理”；必须有实测工作点。把附录 G 的失败排序直接说明，新增机制分析不是已经成功的 deployment rule。

**jSCP：先给近邻与容量实证，再给时序控制。**

> We position CrossGB within the established model-tree literature. [Direct baseline and k table.] The one-coefficient statement specifies the leaf family; its practical value is an empirical accuracy–complexity question. [Persistence/Ridge/origin results.] The forecasting improvement is relative to [explicit reference set], with [scope and limitations].

不要以“CrossGB 不是某方法特例”作为段首主论点；模型类包含关系与训练算法不同是两件事。承认先例不会否定 histogram boosting 组合的研究价值，直接比较才支撑其增量。

**wWos：维护认可，给两项新增事实。**

> The additional comparison with [actual closest method] and the matched-budget results [numbers] sharpen the evidence for the proposed structural bias. The reported task-elapsed overhead includes search; final-fit/predict costs are [numbers]. [Concrete regimes with and without a useful trade-off.]

不以“推理仅一个乘法”淡化训练开销；也不要重复用原 300 候选数证明公平计算。

**VVba：主要用准确解释，避免不必要的大改。**

> Routing remains axis-aligned, but the summed margin can have an oblique boundary: two root-only slope trees plus base margin represent `F(x)=x1+x2−t`. This is expressiveness, not a recovery guarantee. Each leaf currently fits `βx_j OR bias`; other trees/base margin can supply offsets. Candidate pools are sampled once per tree by Bernoulli inclusion; each leaf selects independently within that pool. Shared pools reuse conditional candidates; independent pools may overlap. In the ablation, `p=ρp`, and SplitOnly retains constant terminal responses.

分类图可在已有输出上辅助理解，但表单只允许文字/表格时，解析例子足够。若没有完成 affine 实验，就不给“会加入仿射叶”无条件承诺。CatBoost 更优是全训练算法比较，不能凭树形简单推断它应该更弱。

### 6.2 技术问题与 AC 的统一边界

- 固定候选的一维二阶系数可闭式求解，不意味着当前 split surrogate 就是最终独立叶的 exact objective gain。
- `βx OR bias` 包含常数响应形式，不意味着完整训练器数值等价于 vanilla XGB，更不保证泛化误差不劣。会议纪要中的“包含所以不应更差”不能成为回复论据。
- bin 拟合/raw 输出与图 1 的树数问题分别处理，不以一个新实验掩盖另一处解释缺口。
- h12 已有后期完成记录，弃用旧撤回草稿；没有必要因早期 pending 做不存在依据的自我否定。
- 不复用旧 Confidential 中“所有分歧都与正确性无关”的绝对说法。AC 看到的应是：共同认可、关键新证据、准确处理过的技术限制及可追踪更正。
- 新增实验作为 **rebuttal follow-up，缩减预算/新受控协议** 单列；不改写原 benchmark 的 9/10、8/2/13 总体计数。未来论文编辑清单与当前回复分开。官网禁止审稿期间替换原投稿材料；这与用户确认的回复内新增结果并不冲突。[官方修改规则](https://aaai.org/conference/aaai/aaai-27/paper-modification-guidelines/)

### 6.3 创新定位需要保留与放弃的表述

可以保留：在二阶 histogram boosting 内，将路径外单变量响应选择、预测感知分裂 surrogate、终叶重新择优与 bias 候选结合；单叶一个连续系数的受限响应与其可测取舍。

需要放弃：首次线性叶、首次每叶独立选一个变量、保证最终 raw 目标精确下降、等同 XGB 的 bias trainer、k=1 已证明最优、可靠预判真实数据胜负、普遍更快或更强。

单变量叶已有先例，相关工作应承认 [Van Den Bos et al. 2024](https://proceedings.mlr.press/v235/van-den-bos24a.html)；PILOT/TAO/CLARITree 的单树优化与 boosted linear-leaf 应区分。这里只决定必要定位，不在六天内新增整篇相关工作综述。[PILOT 作者实现](https://github.com/STAN-UAntwerp/PILOT)、[CLARITree 作者原文](https://arxiv.org/html/2606.12840v1)。

## 7. 验收与停止条件

每项新结果保留：数据来源/hash、源码/native hash、实际参数、split/时间边界、候选验证记录、最终预测、fit/search/predict 时间与失败记录。只用验证集选配置；原图脚本找不到不是更换参数后称已恢复的理由。

本轮核心验收：

- E1 至少三机制四格完整，能直接回答 Full vs LeafOnly。
- E2 有可靠 LightGBM linear 对照，并限时争取至少一个原始 boosted 近邻；未完成项准确披露。
- E3 有 CPU 与胜例的相同口径诊断，原因强度与证据对应。
- E4 有真实 CPU 预算下验证选优后的测试结果，或清楚界定未闭合的公平性问题。
- E5 十任务参考表与 selected 重放形成可核查链，时序大收益的参照集合明确。
- E6 完成可诚实标注的 k 容量小实验，或明确停止原因和问题仍未完全解决；不把 LightGBM 结果冒充 k 消融。
- 五位人工评审均有按问题回答的短文本；所有数字可追踪、无 TBD/未经执行的完成时态，字符和表格检查通过；作者完成最后审阅。

满足这些条件后停止扩大实验。实验方向不利也进入内部结论；不得以筛选结果或模糊协议来“满足”验收。

## 8. 待用户确认的范围

**推荐：同意核心方案。** 执行 E0–E5，E6 按本任务书限时开展，E7 先完成解析核查、只做必要小诊断；不默认增加 child-max 大面板、全维度 sweep、3000 候选搜索或完整 k>1 原生格式重构。先 32 核/64GB，最多 24 个单线程模型任务并行，按内存与实际吞吐调度。

更精简选项：E0/E1/E2/E5 先闭合，E3/E4 做最小案例、E6 后置；代价是 t7wg 的成本问题与 jSCP 的容量问题证据较弱。

确认后才开始源码修改、构建和模型拟合。远程执行所需的服务器入口在执行阶段接入；本任务书的确认不授权替换投稿 PDF、发邮件、发 OpenReview 回复或付费申请新机器。

## 9. 本地来源导航

- [投稿正文](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/aaai2027提交原件/24092_CrossGB_Modeling_Conditi.pdf)：§3–4，Table 1–5；图 1 和 Table 2 已目视核对。
- [投稿技术附录](</Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/aaai2027提交原件/24092_CrossGB_Modeling_Conditi_Technical Supplement.pdf>)：B 协议/计时，D 消融，E 真实数据，F 时序，G suitability。
- [审稿原文](</Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/Reviewers_CrossGB_ Modeling Conditional–Predictive Feature Interactions in Gradient-Boosted Trees _ OpenReview.pdf>)：人工评审在第 2–8 页，AI 在第 9–10 页。
- [会议纪要](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/Rebuttal讨论会1025纪要.md)：近邻、LeafOnly、胜负案例分析方向可采纳；自动听记的“STBOSS”“subset”及节点采样等技术措辞不当作事实。
- [Xia 建议目录](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/Xia'sSuggestions)：综合核查比早期执行稿更新；本轮找到 h12 后再次更新其状态。
- [完整工程实验参考手册](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/aaai_FullCodeProject-check_codereview0701/datacollection0702/experiment_data_reference_0719.md)：作为路径与协议索引，旧统计不覆盖新原始记录。
- [原正式消融目录](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/aaai_FullCodeProject-check_codereview0701/datacollection0702/0716dataAddition/metagb_ablation_syn/syn_formal_formal16_v1)。
- [h12 后期完成记录](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/aaai_FullCodeProject-check_codereview0701/datacollection0702/0716dataAddition/ts0716/未命名/summary.csv)。
- [历史 hash 匹配 updater](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/aaai_FullCodeProject-check_codereview0701/project_paperwriting/vendor/xgboost-metagb/src/tree/updater_metagb_hist.cc)。
- [投稿 updater](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/aaai2027提交原件/SupplementaryCode_CrossGBv1_github_openreview/CrossGB/xgboost-master/src/tree/updater_crossgb_hist.cc)：抽样约 200 行，bin training 约 379 行，score 约 628 行，树级池约 1589 行。
- [问题台账](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/rebuttal_plan_20261025/审稿问题台账.csv)、[实验队列](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/rebuttal_plan_20261025/实验队列.csv)、[本轮只读核查](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/rebuttal_plan_20261025/证据核查.json)。
