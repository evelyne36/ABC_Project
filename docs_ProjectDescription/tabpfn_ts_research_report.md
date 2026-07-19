# TabPFN-TS 如何把表格基础模型适配到时序预测：方法拆解与对本实验的启示

> 用途：汇报材料 / 实验设计参考　｜　调研日期：2026-06-30
> 主题：表格数据基础模型（TabPFN）跑时间序列预测的范式转化、时间依赖与跨变量依赖的取舍
> 关联本项目：`current_metagb` 在 `timeseries_extra_random300` 中用 lag-window flattening 跑 LTSF
> 见 [timeseries_window_horizon_protocol.md](docs/timeseries_window_horizon_protocol.md)

---

## 0. 一句话结论（先给汇报用的 takeaway）

TabPFN-TS 的核心做法是**不把"历史窗口"当特征，而是把"时间戳"当特征**：它把每个时间步变成表格里的一行，行的特征是从时间戳派生的日历特征 + 傅里叶季节性特征（外加可选协变量），然后让一个**冻结、零样本**的表格回归模型 TabPFN 在"过去行=上下文、未来行=待预测查询"的设定下做 in-context 回归。

它由此做出了一个**很激进、但很关键的简化**：

- **时间步之间的依赖**：不靠行与行之间的自回归/递归显式建模，而是**完全压进单行的时间戳特征里**（calendar + Fourier）。模型架构对行顺序是**置换不变（permutation-invariant）的**——这正是它从纯表格 i.i.d. 假设继承来的、对时序而言"不太对"的归纳偏置。
- **同一时间步内不同变量的依赖**：基本**不建模**。TabPFN-TS 本质是**单变量**的，多变量问题被拆成 N 个互相独立的单变量预测；变量间关联只能通过把"已知未来协变量"作为额外列**弱注入**，而不是联合建模。

这对你的实验是直接有用的对照：**你的 MetaGB 走的是 lag-window flattening（行间依赖塞进特征列），TabPFN-TS 走的是 timestamp-feature（行间依赖塞进时间戳列）**——两者是"时序→表格"范式里的两条不同技术路线，恰好能撑起"表格模型有泛化性、且路线选择本身是科学问题"的论述。

---

## 1. 时序 → 表格范式的具体转化（问题 1）

### 1.1 官方四步流水线（已验证，3-0）

TabPFN-TS 把预测拆成四步（PriorLabs 官方 quickstart 原文）：

1. **把一条时间序列转成一张表**（Transform a time series into a table）
2. **抽取特征并加进表里**（Extract features and add them to the table）
3. **用 TabPFN 在这张表上做回归**（Perform regression on the table using TabPFN）
4. **把回归输出当作预测结果**（Use regression results as forecasting outputs）

> 来源：https://github.com/PriorLabs/tabpfn-time-series/blob/main/examples/quickstart.ipynb （已 3 票验证）

关键点：第 3 步用的是**原版 TabPFN（v2）回归器本体，不做任何针对时序的再训练/微调**。所谓"适配"几乎全部发生在第 1、2 步的**特征工程**里。

### 1.2 表的结构：一行 = 一个时间步

- 每个时间戳 `t` 对应表中一行。
- **训练集（上下文）** = 过去已观测的行，`(特征行, 观测值)` 成对；
- **测试集（查询）** = 未来时间戳的行，只有特征、没有标签，由 TabPFN 一次性回归出来。

形式化（记 C=上下文长度/历史步数，H=预测步数，D=特征维度）：

```text
X_train ∈ R^{C × D},  y_train ∈ R^{C}     # 过去
X_test  ∈ R^{H × D}                        # 未来 H 步的时间戳特征
ŷ_test = TabPFN.predict(X_train, y_train, X_test)   # 一次 in-context 前向
```

### 1.3 特征构造（这是适配的真正发生地）

TabPFN-TS 注入"时间结构"的方式是**自动生成的时间戳派生特征**，而**不是**显式序列建模（已验证，3-0：calendar + sin/cos 工程特征）。具体包含三类：

| 特征类别 | 内容 | 作用 |
|---|---|---|
| **运行索引（running index）** | 单调递增整数计步 | 承载趋势 / 线性漂移 |
| **日历特征（calendar）** | 8 个周期分量：second-of-minute、minute-of-hour、hour-of-day、day-of-week、day-of-month、day-of-year、week-of-year、month-of-year，每个用 sin/cos 编码 | 承载多尺度周期性（小时/天/周/年） |
| **自动傅里叶季节性（FFT seasonal）** | 对历史做 FFT，取 top-k 主周期（默认 k≈5），每个生成一对 sin/cos | 承载数据驱动的季节性 |
| **协变量（可选）** | 已知未来的外生变量作为额外列 | 弱注入外部信息 |

> 来源：TabPFN-TS 原论文 arXiv:2501.02945（Hoo et al., *The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models Based on Simple Features*）。
> ⚠️ 证据强度说明：calendar 的精确 8 分量、FFT top-k≈5、sin/cos 维度展开等数字来自原论文，本次对抗核查因会话限额未跑完（票数 0-0 = 未被反驳但也未独立复核），属论文明文事实，汇报时可注明"据原论文"。

### 1.4 关键反差：**它故意不用 lag 特征、非自回归**

这是和你 MetaGB 路线最尖锐的对比点：

- TabPFN-TS **不放滑窗 lag 值、不放移动平均**作为特征；时间结构**只**通过"时间戳派生特征"注入。
- 因此整个预测 horizon 是**一次非自回归前向**直接吐出 H 步（未来 H 个时间戳就是 H 个查询行），**不是**逐步递推。

> 证据强度：来自原论文（arXiv:2501.02945），本次未独立复核。这是 TabPFN-TS 最具辨识度的设计主张，汇报建议明确标注来源。

> 对照你的协议：你的 `X_t = flatten(values[t:t+seq_len, all_channels])` **恰恰相反**——你把 lag 窗口展平成特征列。两条路线的本质差异见 §5。

---

## 2. 时间步间依赖 vs 同一时刻跨变量依赖：它怎么取舍（问题 2，核心科学问题）

这一节是汇报的"思想性"重点。

### 2.1 时间步之间的关联性（行间依赖）

**做法**：不显式建模行间依赖，而是**把时间结构编码进每一行的时间戳特征**。

**代价 / 简化（已验证，3-0，这是 ApolloPFN 论文点名的核心问题）**：
TabPFN 的注意力机制对样本（行）顺序是**置换不变的**——它把观测当 i.i.d.。

> 原文：*"a central problem is that the architecture of TabPFN-TS is invariant to the order of the data. Order invariance is a reasonable inductive bias in the tabular i.i.d. case, but it is not a reasonable inductive bias for the time series context."*
> 来源：ApolloPFN，arXiv:2603.15802（已 3 票验证）

含义：把训练上下文的行打乱，TabPFN-TS 的输出**不变**。它对"昨天比上个月更相关"这种**时间邻近性 (recency)**、对自相关结构，**没有内建的归纳偏置**——所有时间信息都得靠 calendar / Fourier 特征"显式喂进去"。能不能抓到趋势/周期，取决于特征工程是否覆盖了真实的周期成分。

**学界的修正方向（佐证这是真问题）**：后续工作 ApolloPFN 给注意力加 **RoPE（旋转位置编码）**，主动打破这种置换不变性，让模型"知道"行的时间顺序（已验证，3-0）。这反过来证明：**vanilla TabPFN-TS 在"时间步关联"上是被简化掉的**。

### 2.2 同一时间步内不同变量的关联性（跨变量依赖）

**做法**：**基本不建模**。TabPFN-TS 本质是**单变量（univariate）**的（已验证，3-0）：

> 原文：*"univariate, target-history + known-future covariate regression … Multivariate targets are decomposed into N independent univariate forecasts."*

- 多变量预测 = 跑 **N 个互相独立**的单变量预测，**没有跨序列信息共享**（已验证，3-0：*"each series is processed and forecast independently (no cross-series information sharing)"*，按 `item_id` 独立处理）。
- 唯一的"跨变量"通道：把**已知未来的协变量**作为额外列加进同一行（covariate-informed regression）。但这是**单向、弱**的——目标列能"看到"协变量列，**变量之间不做联合/对称的相关性建模**（不像 multivariate Transformer 的 cross-channel attention）。

> 协变量基准上 TabPFN-TS 表现强（据原论文，甚至超过 Toto 等多变量模型）——但这条本次未独立复核，汇报时谨慎引用。

### 2.3 取舍总结（一张表讲清"它放弃了什么、换来了什么"）

| 维度 | TabPFN-TS 的选择 | 放弃了什么 | 换来了什么 |
|---|---|---|---|
| 时间步间依赖 | 压进 timestamp 特征（calendar+Fourier）；架构置换不变 | recency / 自相关的内建归纳偏置 | 复用纯表格模型、零样本、非自回归一次出 H 步 |
| 跨变量依赖 | 单变量分解 + 协变量列弱注入 | 联合多变量 / cross-channel 相关 | 任意变量数都能跑、实现极简、无需多变量预训练 |
| 训练方式 | 冻结 TabPFN，in-context | 针对时序的表征学习 | 无需 per-series 训练，开箱即用 |

**一句汇报话术**：TabPFN-TS 用"特征工程承担时间结构 + 单变量分解回避跨变量耦合"换取了"直接复用一个强表格基础模型"。它赌的是——**很多真实序列的可预测性，主要来自周期 + 趋势这类能被时间戳特征显式表达的成分**，而非复杂的跨变量动力学。

---

## 3. 与原版 TabPFN 机制（ICL / PFN / 贝叶斯先验）的关系（问题 3）

### 3.1 继承

- **PFN（Prior-data Fitted Network）**：TabPFN 是在大量**合成表格任务**（由结构化先验生成）上预训练的，学会了"给定上下文 (X,y)，对查询 X' 输出后验预测分布"——这本身就是一种**近似贝叶斯推断 / in-context learning**。
- TabPFN-TS **原样复用**这套机制：过去观测当上下文，未来时间戳当查询，**一次前向 = 一次 in-context 回归**，无梯度更新、无微调（**零样本**）。回归输出是**离散化的分布**（Riemann 分布形式），天然给出**概率预测**——这对 LTSF 的分位数/区间指标是加分项。

> 证据强度：PFN/ICL/零样本/Riemann 分布表述来自原论文（arXiv:2501.02945），本次未独立复核但为公认事实。

### 3.2 错配（适配的"裂缝"）

原版 TabPFN 的先验假设是**表格行 i.i.d.**。时序数据**违反 i.i.d.**（行有时间顺序、强自相关）。于是：

- 置换不变性从"合理"（表格）变成"不合理"（时序）——见 §2.1。
- TabPFN-TS 的应对是**绕过而非修复**：用特征工程把时间塞进行内，让违反 i.i.d. 的部分"看起来"像普通表格特征。
- 学界的"修复"路线（ApolloPFN）则是**改架构**：仍是 PFN/ICL 范式，但**time-aware**（加 RoPE 位置编码）且**原生支持外生协变量**（已验证，3-0）。这条线说明：TabPFN→时序的适配是一个**活跃的、尚未定论**的研究方向，留给"表格模型泛化性"很大的论述空间。

---

## 4. 评测基准与表现（问题 4）

- **核心论点（标题即结论）**：在简单特征下，表格基础模型 TabPFN **超过了专门的时序预测模型**——这是该工作的卖点，也是"表格模型泛化到时序"最有力的外部证据。
- **GIFT-Eval**：TabPFN-TS 在 GIFT-Eval 等基准上有公开排名条目；社区/榜单实现"build upon"了 TabPFN-TS。
- **协变量基准**：据原论文在含协变量的数据上表现突出。

> ⚠️ 本次研究中，GIFT-Eval 具体名次、协变量基准胜负、与 Toto 等模型的对比数字**未通过独立对抗核查**（核查 agent 因会话限额失败，票数 0-0）。汇报时建议：要么标注"据原论文/榜单，待复核"，要么在正式投稿前回到 arXiv:2501.02945 与 GIFT-Eval leaderboard 核对具体数值。**不要把未复核的具体名次当作硬结论写进结果表。**

---

## 5. 对你"表格模型跑时序、证明泛化性"实验的直接启示

你的 `current_metagb` 走的是 **lag-window flattening** 路线，与 TabPFN-TS 的 **timestamp-feature** 路线形成天然对照。两者都属"时序→表格"，但在 §2 的两个科学问题上做了**不同取舍**：

| 对比项 | 你的 MetaGB（lag-window） | TabPFN-TS（timestamp-feature） |
|---|---|---|
| 行的含义 | 一行 = 一个监督样本（一个预测起点） | 一行 = 一个时间步 |
| 时间步间依赖 | **显式**塞进特征列（flatten 过去 seq_len 步） | 塞进 timestamp 特征（calendar+Fourier），架构置换不变 |
| 跨变量依赖 | 多变量输入 flatten 进同一行 → 树可在列间交互**隐式**捕捉 | 单变量分解，跨变量基本不建模 |
| 自回归 | direct H-step / sequence-output，均可 | 非自回归，一次出 H 步 |
| 训练 | per-dataset 训练 + Random300 搜索 | 冻结零样本 ICL |
| 特征维度压力 | `seq_len × channels`，宽数据爆炸（见你的 protocol） | 与 seq_len 解耦，约 O(日历+2k) 维 |

### 5.1 可写进汇报的论述点

1. **路线定位**：明确说明你的方法是"lag-window propositionalization"这一经典 GBDT 时序路线（M5、TSER 等文献支撑，见你的 protocol §"How GBDT/XGBoost Is Used"），而 TabPFN-TS 是"timestamp-feature + 基础模型 ICL"路线。**两者都证明了表格范式能泛化到时序**，但你的方法在**跨变量隐式交互**上反而比 TabPFN-TS 更强（树天然能在 flatten 后的列间做交互），这是你的差异化卖点。

2. **科学问题对齐**：TabPFN-TS 暴露的两个核心取舍——(a) 时间顺序的归纳偏置、(b) 跨变量联合建模——正好是你实验可以**主动讨论甚至消融**的维度：
   - 你的 lag-window **天然保留了时间顺序信息**（哪一列是 t-1、哪一列是 t-k 是固定的），不像 TabPFN-TS 那样置换不变——这可作为你方法的一个**理论优势点**。
   - 你支持 multivariate-input flatten，**跨变量交互是隐式但存在的**，比 TabPFN-TS 的单变量分解更强。

3. **建议补一个 baseline**：若算力允许，把 **TabPFN-TS 本体**或其特征工程（calendar+Fourier）作为一个对照 baseline 跑进你的 14 数据集 fixed-split 协议。这样能直接对比"lag-window vs timestamp-feature"两种表格化路线，极大增强"表格模型泛化性"论证的说服力。退一步，至少可做一个**特征消融**：lag-window vs (lag-window + calendar/Fourier) vs 纯 calendar/Fourier。

4. **诚实标注**：沿用你 protocol 里已有的严谨口径——区分 direct scalar H-step（appendix）与 sequence-output LTSF（主表），并说明你的跨变量是"隐式 flatten 交互"而非 cross-channel attention。这与 TabPFN-TS 论文如实标注"单变量+协变量"的做法一致，审稿友好。

### 5.2 一句汇报金句

> "TabPFN-TS 证明了表格基础模型能赢过专用时序模型，但它靠的是把时间塞进时间戳特征、并把多变量拆成独立单变量——这恰恰说明：表格范式做时序，胜负手在**特征如何承载时间与变量结构**。我们的 MetaGB 走 lag-window 路线，在保留时间顺序和隐式跨变量交互上做了不同取舍，正好补上 TabPFN-TS 简化掉的两块，从另一条路径印证表格模型的泛化性。"

---

## 6. 证据强度与待办（汇报前请注意）

| 结论 | 强度 | 来源 |
|---|---|---|
| 四步流水线（表→特征→TabPFN 回归→预测） | ✅ 已验证 3-0 | PriorLabs quickstart |
| 时间结构靠 calendar + sin/cos 工程特征注入 | ✅ 已验证 3-0 | PriorLabs quickstart |
| 本质单变量、多变量拆 N 个独立单变量 | ✅ 已验证 3-0 | PriorLabs quickstart |
| 各序列独立预测、无跨序列共享 | ✅ 已验证 3-0 | PriorLabs quickstart |
| 架构对行顺序置换不变（时序的错配核心） | ✅ 已验证 3-0 | ApolloPFN arXiv:2603.15802 |
| ApolloPFN 用 RoPE 注入时间顺序 | ✅ 已验证 3-0 | ApolloPFN |
| ApolloPFN 是 time-aware PFN、原生支持协变量 | ✅ 已验证 3-0 | ApolloPFN |
| "TabPFN-TS 不是真正时序模型，只是给表格 FM 加少量时序特征" | ⚠️ 2-1 | ApolloPFN |
| 8 日历分量 / FFT top-k≈5 / 维度展开 / 非自回归 / Riemann 分布 / 零样本 | ⚠️ 据原论文，未独立复核 | arXiv:2501.02945 |
| GIFT-Eval 具体名次、协变量基准胜 Toto 等数字 | ⚠️ 未复核，勿当硬结论 | 原论文 / 榜单 |

**待办（限额恢复后可补）**：回到 arXiv:2501.02945 与 GIFT-Eval leaderboard，核对 §1.3 的精确特征维度与 §4 的具体名次/数值，再写进正式结果表。

---

### 主要来源
- TabPFN-TS 官方实现：https://github.com/PriorLabs/tabpfn-time-series
- 官方 quickstart：https://github.com/PriorLabs/tabpfn-time-series/blob/main/examples/quickstart.ipynb
- TabPFN-TS 论文：arXiv:2501.02945（Hoo et al.）
- ApolloPFN（时序适配的修正与批判）：arXiv:2603.15802
- 本项目时序协议：[docs/timeseries_window_horizon_protocol.md](docs/timeseries_window_horizon_protocol.md)
