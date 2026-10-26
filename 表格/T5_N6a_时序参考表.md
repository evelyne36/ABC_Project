# T5 · N6a 时序参考基线表

> 对应：Rev5-3「时序高收益与统计支撑薄弱」。
> 状态：**130 wrapper fits / 20 条终测（117 + 13）** 已在远端完成，待聚合。

## 1. 设计（已冻结，务必守住）

- 面板：附录 F 的原十任务（seq24 输入，fixed cutoff），另加 Exchange h12。
- 参照集合：**persistence（last value）+ seasonal naive（季节周期）+ 简单 Ridge**，与原 CrossGB selected 配置在同一 target、同一划分上比较。
- 只能报 **fixed-cutoff** 结果；**不得**出现 rolling-origin 措辞（原实验不是 rolling origin）。
- 结论写法上限："在原树基线 + 原切分下，CrossGB 相对 XGBoost 提升 X"，**不能**把单次 69.6% 说成泛化收益。

## 2. 中文内部表（待填）

行 = 任务 × horizon；列 = XGBoost / CrossGB / persistence / seasonal naive / Ridge / 相对 XGB 增益。

| 任务 | h | XGBoost | CrossGB | CrossGB vs XGB | persistence | seasonal naive | Ridge |
|---|---|---|---|---|---|---|---|
| [待填 × 10–11 行] | | | | | | | |

## 3. English appendix draft（骨架）

**Table S5.** Fixed-cutoff forecasting reference baselines on the ten tasks of Appendix F (plus Exchange at h12): last-value persistence, seasonal naive, and a ridge autoregression, against the selected XGBoost and CrossGB configurations. Errors are mean over the same fixed splits used in the main text.

*Notes.* All entries use a single fixed cut-off (no rolling origin); the reference baselines are included to bound how much of the reported gain is attributable to the temporal structure of the data rather than to the leaf model.

## 4. 填数要点

- 若某任务的 persistence 就打败 CrossGB（很可能出现在强趋势任务），必须保留并在正文承认"该任务的增益主要来自可外推趋势"。
- 十任务的指标方向要统一（部分任务原表是 RMSE、部分是 MAE/归一化指标），本表只放**同一定义下的相对比较**，绝对值单列脚注。
- h12 那一条（Exchange）是"找回的记录"，要标来源，不能与新跑的混为一谈。
