# MetaGB Synthetic 消融结论

> 使用方法：将本文件复制到
> `results/syn_formal_<RUN_TAG>/SYNTHETIC_CONCLUSION.md`，用真实运行结果替换所有
> 占位内容。只有删除全部 `TBD/TODO/待填写/<填写>`，并在最后写入两个完成标记后，
> `approve-synthetic` 才会放行 Real 阶段。

## 1. 运行身份与完整性

- Run ID：TBD
- Protocol version：TBD
- Config hash：TBD
- Execution hash：TBD
- Variant count：16
- Synthetic cells：14（7 scenarios × 2 dimensions）
- Evaluation seeds：3（1001、1002、1003）
- Outer tasks：210
- Expected/actual fold rows：3360 / TBD
- Completeness audit：TBD

## 2. 四类主消融结论

报告并解释 `XGB`、`Bias p=0`、`SplitOnly p=1`、`Full p=1`，重点回答：

1. `SplitOnly − Bias` 是否显示 predictive-aware split 的独立贡献；
2. `Full − SplitOnly` 是否显示 predictive leaf 的独立贡献；
3. `Full − Bias` 是否显示完整机制总贡献；
4. `Full − XGB` 在 fixed-backbone 条件下是否稳定。

结论：TBD

## 3. pred_fraction 剂量响应

分别总结 SplitOnly 与 Full 在 `p={0.2,0.4,0.6,0.8,1.0}` 下的：

- 效应方向与幅度；
- 单调性、平台区或反转；
- dimension 10 与 20 的差异；
- 三个 seed 的方向一致性；
- 是否存在只由 `p=1` 单点驱动的现象。

结论：TBD

## 4. score_agg 六格消融

在 `p=1` 下同时比较：

```text
SplitOnly × {greedy, mean, median}
Full      × {greedy, mean, median}
```

说明 greedy 的结论是否稳健，以及 aggregation 与 leaf mode 是否存在明显交互。

结论：TBD

## 5. 机制匹配场景与负对照

分别总结五个 mechanism-aligned scenarios 与两个 negative controls。若负对照也
出现同量级提升，必须降低机制归因强度并给出替代解释。

结论：TBD

## 6. 不确定性与结论边界

- 展示三个 seed-level paired points，而非把 outer folds 当作独立重复；
- bootstrap CI 仅作探索性不确定性描述；
- 明确跨 seed 方向不一致或效应接近零的单元；
- 不把 fixed-backbone diagnostic evidence 外推为“普遍优于 XGBoost”。

结论：TBD

## 7. Synthetic 总结与 Real 阶段决策

综合说明哪些机制主张获得支持、哪些被否定或需要收缩，以及真实数据阶段必须重点
验证的预注册问题。

完整结论：TBD

SYNTHETIC_ANALYSIS_STATUS: 待填写

REAL_STAGE_DECISION: 待填写
