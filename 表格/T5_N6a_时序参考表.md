# T5 · N6a 时序参考基线表

> 对应：Rev5-3「时序高收益与统计支撑薄弱」。
> 状态：**✅ 参照基线已出数**（10-27 12:24 回传，22 行）；**seasonal naive 未跑**（明确不做，见 §4）。
> 结论方向：**不利**。CrossGB 在十任务上相对最强简单基线 **0/10 胜**。

## 1. 设计（已冻结，务必守住）

- 面板：附录 F 的原十任务（seq24 输入，fixed cutoff），另加 Exchange h12。
- 参照集合：**persistence（last value）+ 简单 Ridge**（seasonal naive 本轮未做）。
- 只能报 **fixed-cutoff** 结果；**不得**出现 rolling-origin 措辞（原实验不是 rolling origin）。
- 结论写法上限："在原树基线 + 原切分下，CrossGB 相对 XGBoost 提升 X"，**不能**把单次 69.6% 说成泛化收益。

## 2. 中文内部表

测试集 RMSE（越低越好）。XGBoost / CrossGB 两列即附录 F Table 18 Panel A；persistence 与 Ridge 为本轮新跑参照面板。

| 任务 | h | XGBoost | CrossGB | CrossGB vs XGB | persistence | Ridge | 最强简单基线 | CrossGB−最强 |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 1 | 0.0713 | 0.0735 | **+2.9%（更差）** | 0.0737 | 0.0729 | Ridge | +0.0006 |
| ETTh2 | 1 | 0.0668 | 0.0779 | +16.6%（更差） | 0.1094 | 0.0584 | Ridge | +0.0195 |
| ETTm1 | 1 | 0.0379 | 0.0362 | −4.5%（更好） | 0.0363 | 0.0361 | Ridge | +0.0001 |
| ETTm2 | 1 | 0.0389 | 0.0219 | −43.7%（更好） | 0.0304 | 0.0186 | Ridge | +0.0033 |
| exchange_rate | 1 | 0.0690 | 0.0507 | −26.5%（更好） | 0.0290 | 0.0318 | persistence | +0.0217 |
| exchange_rate | 6 | 0.1178 | 0.0548 | −53.5%（更好） | 0.0526 | 0.0568 | persistence | +0.0022 |
| exchange_rate | 12 | 0.1758 | 0.0737 | −58.1%（更好） | 0.0702 | 0.0796 | persistence | +0.0035 |
| ILI | 1 | 1.7782 | 0.5401 | −69.6%（更好） | 0.3488 | 0.4709 | persistence | **+0.1913** |
| ILI | 6 | 1.8807 | 0.8044 | −57.2%（更好） | 0.6702 | 0.6855 | persistence | **+0.1342** |
| ILI | 12 | 1.9653 | 1.0917 | −44.5%（更好） | 0.9163 | 0.8622 | Ridge | **+0.2295** |

**中性事实句**：
- 中：CrossGB 在 10 个任务上相对 XGBoost 有 7 个更低、3 个更高；相对"最强简单基线"（persistence 或 Ridge 取优）**0/10 更低**。
- 中：差距最大的是 ILI h12（+0.2295）与 ILI h1（+0.1913）；最小的是 ETTm1（+0.0001）与 exchange_rate h6（+0.0022）。
- 中：persistence 在 ILI h1/h6 与 exchange_rate 三个 horizon 上是十行中的最强基线；Ridge 在 ETT 四行与 ILI h12 上最强。
- En: CrossGB outperforms XGBoost on 7 of 10 tasks but no trivial baseline on any task; the largest margins against the best simple baseline appear on the ILI tasks.

> 表下按拍板只放中性事实句。"时序增益主要是相对弱调的树基线"属主张层，不写。

## 3. English appendix draft

**Table S5.** Fixed-cutoff forecasting reference baselines on the ten tasks of Appendix F plus Exchange at h12. Last-value persistence and a ridge autoregression are evaluated on the same target, split geometry and horizon as the reported XGBoost/CrossGB entries. RMSE, lower is better.

| Task | h | XGBoost | CrossGB | Persistence | Ridge | Best simple |
|---|---|---|---|---|---|---|
| ETTh1 | 1 | 0.0713 | 0.0735 | 0.0737 | **0.0729** | 0.0729 |
| ETTh2 | 1 | 0.0668 | 0.0779 | 0.1094 | **0.0584** | 0.0584 |
| ETTm1 | 1 | 0.0379 | 0.0362 | 0.0363 | **0.0361** | 0.0361 |
| ETTm2 | 1 | 0.0389 | 0.0219 | 0.0304 | **0.0186** | 0.0186 |
| exchange_rate | 1 | 0.0690 | 0.0507 | **0.0290** | 0.0318 | 0.0290 |
| exchange_rate | 6 | 0.1178 | 0.0548 | **0.0526** | 0.0568 | 0.0526 |
| exchange_rate | 12 | 0.1758 | 0.0737 | **0.0702** | 0.0796 | 0.0702 |
| ILI | 1 | 1.7782 | 0.5401 | **0.3488** | 0.4709 | 0.3488 |
| ILI | 6 | 1.8807 | 0.8044 | **0.6702** | 0.6855 | 0.6702 |
| ILI | 12 | 1.9653 | 1.0917 | 0.9163 | **0.8622** | 0.8622 |

*Notes.* All entries use a single fixed cut-off; there is no rolling origin and no future information in either the baselines or the learned models. The reference panel was run under a reconstruction of the archived split semantics, not as a bit-level replay of the original run; the XGBoost/CrossGB columns are the archived values reported in Appendix F, which we verified against the archived per-task records (e.g. ILI h1 = 0.5401007). Seasonal naive was not evaluated in this round and is therefore not reported.

## 4. 填数要点（已执行 ✅ / 需注意 ⚠️）

- ⚠️ **seasonal naive 没跑**：原骨架写了三项基线，实际只做了 persistence 与 Ridge。删除该列，并在脚注写明"本轮未评估"，不得留空列让人以为是 0。
- ⚠️ **口径性质**：`TS_nine_linux_20261026/manifest.json` 自述 `config_provenance = archived_semantics_reconstructed_new_reference_not_historical_reproduction`、`historical_reproduction_claim = false`。所以这是"同切分几何、不同运行"的对照，**不是同一进程内复跑**。要完全 airtight 需要 N6-replay（42 fits，本轮移出必做）。
- ⚠️ 十任务指标方向统一为 test RMSE，绝对值来自原 Panel A，可用归档记录逐位复核（`rebuttal_plan_20261025/已有结果复用审计_数据.json` 的 `timeseries/cgb_records`）。
- ✅ 若某任务的 persistence 打败 CrossGB，必须保留并在正文承认——现在这不是"很可能"，而是 10/10 全部发生。
