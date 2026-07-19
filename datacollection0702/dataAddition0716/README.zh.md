# 0716 数据补全：索引与最新结论

本目录保存对 `../datacollection0702/analysis_aaai_0702/` 所用实验数据的增量补全。
为了避免后续论文写作时混淆旧快照与新增结果，最新统一分析固定保存在：

- [`LATEST_CONCLUSIONS_20260718.zh.md`](LATEST_CONCLUSIONS_20260718.zh.md)：截至
  2026-07-18 的完整中文结论、关键数值、论文可用表述、统计边界和证据路径。

## 子目录含义

| 子目录 | 作用 | 当前论文口径 |
| --- | --- | --- |
| `tab_otherBaselinemodel0716/` | 补齐 random300 tabular 的 CatBoost、TorchMLP、RealMLP 等强基线 | 与 `datacollection0702/datacollection0707_baselinemodel_random300/` 按 task union 使用；TabM 已弃用并完全排除 |
| `synbaseline_lgbm0716/` | 为原本只有 MetaGB/XGBoost 的 synthetic 主实验补充 LightGBM | 仅纳入完整的 10/20 维 LightGBM；残留的部分 CatBoost 任务不进入统计 |
| `syn102050part0716/` | synthetic XGBoost/MetaGB 主实验的续跑快照 | 10/20 维用于确认旧结果未变化；dim50 仅作为不完整增量证据 |
| `ts0716/` | 非 PEMS 时间序列树模型实验继续运行后的快照 | 使用最新 `summary.csv`；严格区分 9 个 MetaGB 配对 cells 与完整的 21-cell 常规树 baseline |

## 一页结论

- **Synthetic**：在完整的 10/20 维三模型比较中，MetaGB、LightGBM、XGBoost
  分别取得 28、7、3 个逐 cell 最优；MetaGB 的优势仍集中在机制匹配族，
  并在负对照上消失。dim50 当前只有 11 个完整 XGBoost–MetaGB cells，
  不能写成完整维度结论。
- **真实 tabular 强基线**：排除已弃用的 TabM 后，7 个保留 baseline 完成
  1885/1890 个 folds。逐 cell strongest available baseline 相对 XGBoost
  为 42/2/7，相对 MetaGB 为 42/2/2。在所有 9 个模型严格共同完成的 45 个
  cells 上，MetaGB 综合平均 rank 为第 3/9，但仅有 2 个逐 cell 第一；
  它的优势主要是跨任务稳定的中上游表现。CatBoost 总体最稳健，RealMLP
  明显强化了回归基线。
- **时间序列**：MetaGB 配对覆盖由 5 增至 9 个 cells，对每个 cell 的最强常规
  树为 7 胜 2 负；但仍只覆盖 `seq_len=24` 的 9/21 个 cells，且端到端耗时
  中位数约为最强常规树的 100 倍。
- **统一论文叙事**：最新数据进一步支持
  **mechanism-aligned / selective advantage**，不支持 MetaGB 在真实 tabular
  或完整时间序列 benchmark 上的普遍 dominance。
