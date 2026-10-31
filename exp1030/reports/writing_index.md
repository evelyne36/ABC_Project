# 按数据来源组织写作

核对日期2026-10-31；pool结果截止2026-10-30 07:41。

## 第一部分：论文已有任务的补充实验

- 本轮18个任务已在论文出现，其中14个完整确认（3第一/7第二/4第三）。
- 正文Table 3及附录Table 14出现：Moneyball, QSAR_fish_toxicity, abalone, cpu_activity, energy_efficiency, kin8nm, naval_propulsion_plant, space_ga。
- 附录Table 14出现：Another-Dataset-on-used-Fiat-500, Food_Delivery_Time, airfoil_self_noise, concrete_compressive_strength, diamonds, healthcare_insurance_expenses, houses, miami_housing, physiochemical_protein, wine_quality。
- 已有19个合成生成器列于正文Table 1/附录Table 6–9；本轮没有在原生成式上重跑完整近邻面板。
- 可写：在先前报告的数据集上补充PL/LinX/PILOT/TAO直接比较。不可写：上述14个都是新发现的数据集。

## 第二部分：新增任务与新构造机制面板

- 新增公开任务5个完成：QSAR-TID-11、Superconductivity、Real Estate Valuation、Combined Cycle Power Plant、Auto MPG（2第一/2第二/1第三）；Yacht已准备未执行。
- 新合成6组×8变体=48：24第一/22第二/2第三；46/48平均RMSE低于PL，双方24成功HPO。LinX覆盖4–18/24须保留。
- 与旧机制有继承关系，同时改变生成式、维度、分布和协议；可写适配场景扩展，不可写同条件下旧失败实验被逆转。

方法与设置、分组后的全部数值见[完整汇报](researcher_report.md)；逐任务出处见[来源分组](../tables/dataset_origin_groups.csv)。
