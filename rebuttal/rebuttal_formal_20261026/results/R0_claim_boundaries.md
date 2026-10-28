# R0 数值与实现表述边界

来源为V2完整审计与当前历史updater。下列是需要收紧的表述，不是新增训练结果。

1. **BiasOnly≠已证明逐格等价XGB。** 原dim10 categorical格的OOF RMSE：XGB约0.377772，Bias约0.325801，Full约0.323129。总体signed median小，不能抹平这一格差异。E0先检查能否重放原数值，再查gain阈值、叶刷新、候选抽样等实现差异。没有实证前不指定唯一原因。

2. **Figure 1单树/学习率语义未闭环。** 若标准更新为F₁=F₀+ηh，且h恰好拟合无噪声残差、η=0.1，则剩余残差为0.9倍、平方误差为0.81倍；不能仅据该图声称一棵树在此设定达到noise floor。原作图脚本来源未找到，不能猜成“笔误”或擅自把η改成1来解释。回复中先避免依赖这个未闭环点。

3. **分裂分数是代理。** 原`FindBestSplit`以共同候选变量的子节点score增量进行best/mean/median聚合；`BestLeaf`/`RefreshLeaves`又独立选最终叶变量。因此聚合分数一般不等于最终叶模型精确优化目标的下降。只有明确限定同候选、同二阶近似、同正则等条件时才能谈对应的局部代数解释。

4. **训练bin与预测raw有近似差异。** 训练统计经`TrainValue`取得bin代表值，预测经`SparsePageValue`读取raw特征。关于“精确最优”的措辞必须限定训练代理；N5 raw-k1单列是容量对照的必要锚点，不能把bin/raw变化归功于k提升。

5. **旧CPU成本口径已核对。** 18.7168×为五折search elapsed比的中位数；19.2215×为search elapsed均值之比；28.1360×为selected final-fit elapsed逐折比中位数。三者均非CPU秒，也不构成同机等预算比较。可复算数据见`N4_archived_summary.csv`与`N4_selected_folds.csv`。

这些检查使新证据与主张保持一致；不要求为每一点新增大规模CGB搜索。
