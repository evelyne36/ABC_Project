# CrossGB rebuttal：给另一台电脑 AI 的执行交接书

## 任务与已获授权

请接续已经批准的实验方案，不从头重新设计。作者硬截止是**北京时间2026年10月29日完成可提交回复**；目标为10月28日18:00主要实验收口、29日上午交作者审阅，给评审留讨论时间。原评分7/6/5/5/4，confidence4/3/3/3/3。目标是用有效证据争取提分，不能预设结果或保证评分。

作者已批准核心E0–E5与限时E6、四个近邻方法的兼容适配，并授权使用下列五台服务器；正式训练放在Linux。本机只负责准备和核查。TAO按作者官方**常数回归叶**版本运行，不自行补2025线性叶，不把常数叶TAO冒充该版本。CLARTree暂不纳入。

原论文、附录、评审、会议纪要、Xia建议、myself V1/V2及关键旧结果均随包提供。V1依据正文和裸评审提出需求，V2扣除附录已有证据；两者已经合并为当前方案。请先阅读下面四份文件，再按队列执行，不重复做已完成的文献搜索或大规模CGB调参。

1. `rebuttal_formal_20261026/正式执行规划_metaDesign.md`：当前权威计划。
2. `rebuttal_formal_20261026/正式实验矩阵.csv`：优先级、原E编号、拟合数量、状态。
3. `rebuttal_formal_20261026/Linux接续说明.md`：执行命令。
4. `rebuttal_formal_20261026/逐审稿人回复骨架_待填新结果.md`：回应重点及结果槽。

## 接手时的真实状态

| 内容 | 已完成 | 尚未完成 |
|---|---|---|
| V1/V2、论文与附录证据审计 | 全文阅读、合并、去重 | 真实新结果到齐后的终稿 |
| 历史消融 | 3360指标及原预测已重算；本轮105任务split/backbone、63,000个test target核对通过 | 新LeafOnly |
| 近邻 | GBDT-PL、LinXGBoost、PILOT、官方常数叶TAO，本机回归适配与集成测试通过 | Linux构建/检查、16-fit测速、正式HPO |
| CGB源码 | 历史副本保留；controlled副本新增默认关闭的LeafOnly开关；R0/N2驱动已准备 | **Linux编译、tiny检查、历史端点重放均未完成** |
| N3/N4 | 等搜索预算执行器、selected refit/叶结构导出、冻结配置已准备 | Linux执行验证、正式结果 |
| 时序 | 十任务数据与原split预检查通过；persistence/Ridge执行器已准备 | 正式参考结果 |
| 软件检查 | 5项边界测试通过；Python/Bash静态检查通过 | 不构成native算法正确性证明 |
| N5容量/多起点 | 设计已冻结、数量明确 | **执行器未实现**；不得称已可直接运行 |

**新增正式训练结果仍为0。** 旧结果、当前重分析、集成测试、新正式实验必须分开标注。

五台机器最初都成功免密登录并完成环境核查；随后网关重置传输，进一步在SSH握手前关闭连接，作者自己的终端也不能登录。**不要把这个故障归咎于密码/私钥，不要反复高并发重试。** 换电脑后先单台测试；若仍失败，向作者报告基础设施阻塞，继续离线核查与写作，不虚报训练已启动。本机以前建立的工作目录/部分归档可能存在，不能视作部署成功。

## 单文件交接包与本地准备

只需传递最新的 **`formal_linux_bundle.tar.gz`及同名`.sha256`文件**。外包已包含`rebuttal_baselines_20261025/baseline_linux_bundle.tar.gz`，不必另找完整FullCodeProject或重复传baseline包。

外包包含：本轮计划/输入/队列/执行器；完整controlled CGB源码与Python包、历史`param.h/updater`对照；四baseline的固定作者源码、9个已冻结数据输入；十个时序任务所需原始序列；论文/附录/评审/V1/V2/会议纪要/Xia建议；主合成4格×3方法×5折的60个旧任务JSON、对应协议与split manifest；R0七机制原始消融shard及消融协议/骨干/split；真实数据旧任务日志、已有证据复用审计。**没有随包提供可直接复用的已训练CGB树，也没有打包全部210个原始消融shard。** 没有打包密码或私钥。

在另一台电脑选择一个新的任务目录，按以下顺序操作。路径中的`/实际收到的路径/`由接手AI据实替换，不能照抄为真实目录。

```bash
mkdir crossgb_rebuttal_20261026
tar -xzf /实际收到的路径/formal_linux_bundle.tar.gz -C crossgb_rebuttal_20261026
cd crossgb_rebuttal_20261026
tar -xzf rebuttal_baselines_20261025/baseline_linux_bundle.tar.gz
python3 rebuttal_formal_20261026/scripts/verify_handoff_inputs.py
```

先按同名`.sha256`核对外包，再解包。验证器必须输出`status=verified`；它只验证传输/解包完整性。不要在接收电脑重新生成冻结输入、重抽候选或运行`prepare_protocol.py`来替换这些输入；原完整工程没有全部打包，也没有这种必要。

若使用`remote_dispatch.py`部署，先把外包复制到本地解包目录的`rebuttal_formal_20261026/formal_linux_bundle.tar.gz`，因为该脚本在这里查找归档。首次选择一台主机传输并核对；不要同时向五台上传。SSH别名在新电脑上缺失时，向作者获取当前连接配置，不猜端口，不复制任何凭据进交接包。

## 五台分工与环境

每台已核实为16核、约32GB，不能给每台按旧32核/64GB配置开并发。工作目录建议为已用的`/home/admin/crossgb_rebuttal_20261026`；若里面存在未知结果，使用新的后缀目录并同步调整部署器固定路径，保留旧目录。

| SSH别名 | 首要任务 | 开始并发 |
|---|---|---:|
| `cpu4_pool.zjuici.com` | CGB构建→tiny→R0/E0→N2 LeafOnly→N4 selected诊断 | build 8；fit 4，核实内存后可8 |
| `cpu2_pool.zjuici.com` | GBDT-PL构建、四方法集成检查/16-fit测速→GBDT-PL主面板 | pilot 4；正式按内存可8–12 |
| `cpu1_pool.zjuici.com` | LinXGBoost主面板 | 4，核实内存后可8 |
| `cpu_Calculator` | persistence/Ridge→PILOT/TAO补充对比 | 时序单进程单线程；后续按实测4–8 |
| `TabularData_Processor` | N3同机等搜索预算；预留专用CPU | 4专用核，N3期间不混跑其他训练 |

五台的`/home/admin/anaconda3/bin/python`均已核实为3.13.9，科学依赖与本机适配一致。用它在任务目录创建独立`--system-site-packages`环境：

```bash
/home/admin/anaconda3/bin/python -m venv --system-site-packages .venv
source .venv/bin/activate
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
```

不要使用系统Python3.10；已有`mgb`环境在几台服务器上版本不一致，不默认沿用。先复查版本，再决定是否需要任务内安装；不修改全局Anaconda环境。`cpu_Calculator`未找到系统CMake，时序任务不需要它。其余构建依赖以实际检查为准，GBDT-PL需BLAS/LAPACK链接。

所有方法单线程、并行发生在独立fit进程之间。每台各自监控内存；优先保留约30%内存余量。测速仅抽样，不证明最重候选也同样省内存。不同worker不能并发写同一个方法/数据输出目录。

## 执行顺序与验收条件

### A. 三条独立首批线路同时推进

**线路A1，cpu4：R0/E0先验证兼容。** 按《Linux接续说明》构建controlled源码，先4次tiny结构检查，再七机制dim10/seed1001/outer1的XGB/Bias/SplitOnly/Full，共21 CGB＋7 XGB。收集`R0_gate.json`。必须同一native库、输入/骨干一致、28个端点均通过预定容差，才允许拼旧三组与新LeafOnly。不要仅因编译成功就宣布复现。

若R0失败，先检查动态库实际加载路径、编译源、RNG、默认参数、bin/raw、gain阈值。原`metagb_min_gain_to_split`默认是−1e10，不能擅改为0。不能为了通过而放宽容差。兼容性确实无法关闭时，用已经列明的独立小面板备选，单独报告新四格，停止与旧结果拼表；不自动回到大规模HPO。

**线路A2，cpu2：baseline Linux构建/检查与测速。** 四方法都需Linux实际验证。`baseline_runtime_pilot.py`固定16次：CPU/Regime的outer1-inner1，GBDT-PL/Lin候选1/101/201，PILOT/TAO候选1；每fit120秒、4并发，无outer test访问。依据时间/内存/超时冻结可完成覆盖，写`configs/N1_coverage_freeze.json`，再看任何新test结果。保留失败和实际覆盖。

**线路A3，Calculator：N6a时序参考。** 原9任务与后期Exchange h12分两套参数运行。Linux上先`--plan-only`，再用新输出目录执行正式参考；正式命令去掉`--plan-only`并加`--acknowledge-unverified-config`。后者承认重建配置语义，不代表缺少数据/split核查。十任务共130次wrapper fit，20条终测；其中Ridge约462标量fit，**CGB=0**。

### B. 主攻缺失的对照

**N2，cpu4：LeafOnly。** R0通过后，先Regime/Feature/Curved三机制45次，再用`--resume`补齐七机制105次。七机制×3数据seed×5外折、dim10、开发seed9001选定的原骨干冻结，无新HPO。

这是每轮当前梯度下的bias-score split＋predictive-leaf，不是训练后换一次叶。原历史Bias与XGB不保证逐格等价；未关闭实现差异前不要称二者完全相同的“标准分裂”。每seed合并五折OOF，再报3seed均值/离散程度；15个相关外折不能当15次独立数据重复。

**N1，cpu2/cpu1：主近邻。** 核心固定CPU、Kin8nm、Regime slope dim10、Feature switch dim10、Curved dim10。保留正例和负对照。每方法五数据×五外折×(300候选×四内折＋一次final fit)=**30,025 fits**，两主近邻共60,050。四方法全部完成核心为120,100；不是100个便宜任务。

先保证至少一个主近邻完成完整五数据，再完成另一个；服务器可独立并行。PILOT/TAO按已冻结可行性完成补充覆盖。每个方法/数据必须五折齐全才报对应五折均值。候选减少或缺折时必须披露，不能声称与旧CGB 300候选等预算。不要默认扩到原九数据或dim20。

方法身份：GBDT-PL与LinXGBoost为主近邻；PILOT是单树；TAO官方包是斜分裂＋常数叶。Lin公开代码末棵树不乘shrinkage的行为已保留，不悄悄修算法。GBDT-PL用LAPACK兼容构建，不能宣称作者原MKL耗时；本机ARM耗时不能作为Linux基准。

### C. 核心补强与共用模型

**N4，cpu4：selected诊断。** CPU/Kin各五折，仅原selected配置refit：10 CGB＋10 XGB。使用`run_selected_real.py`，导出模型、预测、叶结构、路径外变量使用率与系数等。和旧终测误差比对；这是冻结配置的诊断重拟合，不是找回原训练模型。日志与结构相关性不能当已确定因果。

**N3，Tabular：等搜索预算。** CPU/Kin/Energy五折，XGB/CGB在同一CPU比较60/300/900秒前缀。用`run_equal_search_budget.py`；候选池每方法3000个、顺序冻结；只有截止前完成四内折的候选有资格选中。搜索包括启动/I/O/训练/验证/调度，final refit与test在搜索后单列成本。准确称**等wall-clock搜索预算**，不能说总训练成本严格相同。

必须将cpu4已验证的实际动态库、对应源/构造助手和gate一起迁移，核对库hash及运行依赖；对另一个编译产物不能套用原gate。另一主机的tiny检查另计。无法迁移时在同一已验证主机划出专用CPU完成N3，先暂停争用这些CPU的其他训练；不再悄悄多跑第二套E0。候选时间不足/终拟合超时就如实缺失，不用验证误差代替test。N3搜索总上限约7.5单核小时，final refit另计，四个专用CPU并行预计小时量级。

### D. 有剩余时间才做

**N5容量（旧E6）限时。** 55条固定路由：合成Regime/Sparse/Curved×3seed×5折=45，真实CPU/Kin=10。R0已有3条合成Full，N4提供10条真实路由；另补42 CGB即可。做k1/2/3共165次stagewise系数模型，raw-k1作为bin/raw控制；affine另55。原正则冻结、训练内选变量，不用已见inner-val的outer路由调参。**尚需实现**，开发最多8小时、运行最多4小时；P0未完成就止损，不阻塞回复。

原42次时序selected CGB重放、dim20新增105 LeafOnly、多时间起点、独立suitability、二维示意均不默认启动。跨起点不是换随机seed；做早期起点不得使用未来选出的原selected参数。

## 数量与取舍

| 队列 | 新正式CGB fits | 其他拟合 |
|---|---:|---|
| R0/E0 | 21 | 7 XGB |
| N2七机制dim10 | 105 | 0 |
| N4 selected | 10 | 10 XGB |
| N1近邻 | 0 | 30,025/方法；先两主近邻 |
| N6a参考 | 0 | 130 wrapper fit |
| N3等搜索预算 | 按实测，单独限时计数 | XGB同协议，数量动态 |
| N5容量可选 | 另42 | 165系数模型；affine可选55 |

**固定核心136 CGB；加容量178；再加可选42次时序重放才是220。** 不含N3、额外起点/干预、dim20及每套runtime4次tiny检查。明确记录实际尝试/完成/失败/超时fit数，不把固定小计当总开销。

## 汇总、写作与返回作者

不要只启动长作业就交差。维护`执行日志.md`：每个作业的host、PID/会话、输入/配置/源码/库hash、输出目录、已完成折数、失败、实际耗时、预计剩余量。长作业使用可靠后台会话与退出码，断SSH不应杀掉训练；不要用会话存在代替成功判定。

读取主合成任务JSON中的`fold_row`与候选/协议证据，再与新baseline逐折对齐。旧real来源在`已有结果复用审计_数据.json`及随包日志。近邻主HPO面板与固定骨干消融使用不同seed/选参协议，**不能混为同协议结果**。N2各任务JSON中的`historical_metrics`可用于重建旧OOF端点：五折同为600样本，回归取`sqrt(mean(fold_RMSE^2))`，log loss取加权均值；不把平均fold RMSE误作OOF RMSE。

至少返回作者：

1. **近邻对比表**：旧XGB/CGB/LGB与新GBDT-PL/Lin；PILOT/TAO单列身份、完整覆盖及失败。
2. **四格组件表**：Bias/LeafOnly/SplitOnly/Full，七机制、3seed OOF；保留负例。
3. **成本表**：N3明确搜索预算、final refit成本、test RMSE、缺失；旧elapsed另列。
4. **正负例/时序表**：CPU/Kin结构事实；十任务persistence/Ridge；有容量结果再加。
5. **逐评审回复审阅稿**：按现有骨架填真实数字，删占位符，注明旧证据Main/Supp表号及新增证据位置；没有完成的实验不写成已完成。
6. **可复核结果包**：CSV/JSON、日志、模型、预测、环境与hash、代码差异；镜像回接手机器并另存一份，不只留服务器。

qxju优先N2/N1/CPU；t7wg优先组合贡献/N2/预算/适用性；jSCP优先预算/近邻/时序参考；wWos强调选择性收益与实际成本；VVba优先准确解释slope/bias、p与k、独立采样可重叠、ensemble斜向margin。显著性、16变体、旧耗时图已有，不重跑充数。G仅探索性screen，不能称通用选模器。结果不利也保留，按《回复骨架》的结果分支收缩结论。

平台精确长度上限仍未核实；暂以每位约1350字符组织主干，表格可用但计入上限，不把1500视作已确认规则。10月29日交作者审阅并由作者发布，**不自动提交OpenReview，不联系评审/作者团队以外的人，不在公共仓库发布匿名稿件或评审**。
