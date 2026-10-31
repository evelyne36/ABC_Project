# CrossGB 实验：15分钟讨论提纲（按论文已有/本轮新增分组）

整理2026-10-31；pool结果截止2026-10-30 07:41。本轮新实验不等于新数据集。

1. **方法与设置（3分钟）：** 条件变量决定路由，叶选择一个预测坐标或常数；多树加性组合。五方法6配置预筛→24配置HPO→validation冻结→三次确认；公开任务共享test，合成用新生成seed。
2. **第一部分：已有任务补充实验（4分钟）：** 本轮18个任务在正文/附录已有，14个完整确认，3第一/7第二/4第三。Used Fiat 500、Concrete compressive strength、Diamonds属于已有数据上的新近邻结果；保留小差距和HPO覆盖。原论文19个合成生成器作背景，N1旧比较与本轮分开，不改写历史结果。
3. **第二部分：新增数据与机制（5分钟）：** 五个新增公开任务：Auto MPG第一、Superconductivity第一、Combined Cycle Power Plant第二、Real Estate Valuation第三、QSAR-TID-11第二；Yacht Hydrodynamics未执行。新构造6组48场景，相对PL46/48更准，双方24成功HPO。解释与原门控/斜率机制的继承关系，以及规模、分布、协议同时变化。
4. **成本与写作决策（3分钟）：** 采样训练CPU支持部分工作点，隔离计时尚未做。第一部分写补齐近邻对照，第二部分写适配场景扩展；共同决定主表、缺失覆盖补强和配对机制实验。

[完整汇报](researcher_report.md) · [来源写作索引](writing_index.md) · [逐任务分组](../tables/dataset_origin_groups.csv)
