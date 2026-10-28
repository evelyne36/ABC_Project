# 新公开表格数据与原生近邻筛选（2026-10-27，第2批）

**结论：未找到同时优于XGB、PL、LinXGBoost和LGB linear的新主正例。** 不把适合线性叶的物理数据直接当作适合CrossGB的证据。Servo仅补充对XGB/LGB的正向案例；Computer Hardware仅保留精度—成本观察项。建议优先补齐既有正向候选的近邻与预算比较，新增机制证据见[配对合成实验](../synthetic_nearest_02/README.md)。

## 数据与协议

从UCI官方获取7个数据源、9个回归任务。混凝土的三项输出是依赖任务，不能算三份独立数据；Servo源自仿真，不列为真实观测证据。所有下载URL、来源、原始ZIP和标准化CSV哈希均记录在冻结计划。数据留在项目外层local_sync/new_tabular_data，脚本可重新获取，无私有数据外发。

六种方法各4个预声明配置，2个开发划分选参数；按每方法开发RMSE均值选配置，锁定后拟合3次并评估同一小留出，共594次，全部成功。训练上限1200；seed42外层测试折一直未评分。预处理均训练内拟合。小数据的3次主要复测行顺序/计时，共享同一留出，不能当3个独立外折；CCPP还含训练子采样变化。本轮是有目的地寻找案例，不能用于总体胜率或显著性主张。

PL使用固定原作者代码、现有适配器和ARM兼容构建；Lin使用固定作者Python实现，保留其最后一棵树不缩减的预测语义。PL小网格同时含half_additive和linear叶，max_var为3/5；Lin含5/15棵树、两档正则。相同候选数量不等于相同搜索时间，结果不代表充分调参的性能上限。

## 全部留出结果

RMSE越低越好，各方法独立冻结参数。下表是单一小留出上的3次拟合均值，非三折CV。

| 数据 | CrossGB | XGB | LGB_constant | LGB_linear | PL | LinXGBoost |
|---|---:|---:|---:|---:|---:|---:|
| auto_mpg | 3.08396 | 2.95595 | 2.96912 | 3.02009 | 2.88089 | 2.66802 |
| ccpp | 4.17153 | 4.21043 | 4.15238 | 4.11092 | 4.09665 | 4.10784 |
| computer_hardware | 38.03182 | 37.38185 | 51.66301 | 41.26728 | 37.10261 | 44.45552 |
| concrete_flow | 16.15080 | 14.97090 | 16.39171 | 17.77153 | 17.69202 | 14.96030 |
| concrete_slump | 9.37630 | 7.96352 | 9.30320 | 9.15196 | 8.58302 | 7.25930 |
| concrete_strength | 5.86531 | 4.37657 | 3.97334 | 3.25763 | 4.32760 | 1.86746 |
| real_estate | 6.73954 | 6.55002 | 6.23771 | 6.68293 | 6.91485 | 7.88670 |
| servo | 0.46274 | 0.54962 | 0.61726 | 0.49427 | 0.30254 | 0.45773 |
| yacht | 1.04079 | 0.94198 | 0.81813 | 0.58746 | 0.81137 | 0.70842 |

## 候选取舍

- **Servo**：CGB相对XGB约低15.8% RMSE，相对LGB linear约低6.4%；但PL约0.30254，明显优于CGB 0.46274，Lin约0.45773也略好。CGB约39.8ms，LGB linear约5.34ms、PL约16.5ms。只适合仿真域精度补充，不能说明近邻或资源领先。
- **Computer Hardware**：CGB 38.03182，PL 37.10261，CGB误差高约2.50%，API fit约8.18ms vs20.10ms；XGB 37.38185、约8.07ms。它有相对PL的成本取舍线索，却没有胜过XGB。相对LGB linear/Lin则更准，但LGB linear更快。原数据209行，19个重复X；若正式纳入，须按硬件配置分组复查，不能直接把单留出数字当可靠优势。
- **CCPP**：CGB比XGB约低0.92%，但PL、Lin、LGB linear均更好，且收益小；不优先。
- **Yacht、Auto MPG、Real Estate、三个混凝土任务**：本轮未支持对XGB的优势。混凝土FLOW虽胜PL和LGB，但XGB和Lin更好。保留全部负项。

## 数据准备注意点及官方来源

| 来源 | 规模 | 本轮输入/目标 | 官方说明 |
|---|---:|---|---|
| Yacht | 308×6 | 六项几何/速度输入→阻力 | [UCI](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics) |
| CCPP | 9568×4 | 环境/排气变量→发电功率，仅第一张sheet | [UCI](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)；五张是同一数据的乱序副本 |
| Real Estate | 414×6 | 删除No ID→单位面积房价 | [UCI](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set) |
| Concrete Slump | 103×7 | 七原料→三项输出分别建任务 | [UCI](https://archive.ics.uci.edu/dataset/182/concrete+slump+test)；不把另外两输出当输入 |
| Servo | 167×4 | motor/screw onehot、两gain数值→rise time | [UCI](https://archive.ics.uci.edu/dataset/87/servo)；仿真数据 |
| Auto MPG | 398×7 | 去car_name，hp训练内填补→mpg | [UCI](https://archive.ics.uci.edu/dataset/9/auto+mpg) |
| Computer Hardware | 209×6 | 六硬件指标→PRP | [UCI](https://archive.ics.uci.edu/dataset/29/computer+hardware)；去ERP估算性能、model、vendor |

CCPP有41条完全重复行，Real Estate有20条重复X；这里报告IID随机行小留出，不宣称新配置/新地点泛化。当前这些数据未晋升为主正例，未继续使用更多划分寻找有利结果。若正式研究分组泛化，应提前统一定义组和评价目标。

## 产物与局限

- frozen_plan.json / frozen_choices.json：源数据、划分协议、参数空间与冻结选择。
- fits.jsonl / fits.csv：全部594次拟合，包括失败字段（本轮失败0）。
- splits.csv：划分数量与哈希；summary.csv：完整汇总。discovery汇总混合配置，选参应读逐fit和冻结选择，不能当最佳模型成绩。
- runtime_provenance.json：本机native库、适配器、作者代码SHA和PL构建详情。
- verification.json：完成计数，正式测试评估0。

计时仅fit，不含导入、预处理、参数搜索。PL API fit含CSV与数据读取；Lin是Python作者实现；PL为ARM标量兼容、非上游x86/MKL性能。本轮无统一内存峰值测量，不作内存/能耗/FLOPs优势结论。算法核心、正式配置、Git提交/推送均未改动。
