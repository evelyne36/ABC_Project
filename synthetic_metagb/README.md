# MetaGB Synthetic Validation

This benchmark tests the structural hypothesis behind MetaGB:

```text
conditional features define regions
predictive features provide one-feature linear responses inside leaves
```

It follows the important parts of `random300`: paired hyperparameter backbones,
nested cross-validation, validation-based candidate selection, outer-fold test
metrics, and CSV/JSON artifacts.

Current main-experiment models:

- `xgb_cpu`: ordinary constant-leaf XGBoost
- `current_metagb`: MetaGB with all features available as predictive features

`current_metagb_pred0` is reserved for a separate ablation branch.

Current protocols:

```bash
python3.13 synthetic_metagb/run_benchmark.py --protocol synthetic_main_framework_smoke
python3.13 synthetic_metagb/run_benchmark.py --protocol synthetic_main_random300_single_seed
```

The rigorous runner uses the actual Random300/Random600 candidate generator,
resumable task shards, audited shared nested-CV splits, and independent data
seeds as the statistical pairing unit.

## Framework Smoke 初步观察

已完成 `synthetic_main_framework_smoke`：

```text
19 dim10 representative cells
76 / 76 tasks
2,280 / 2,280 candidate rows
models = xgb_cpu, current_metagb
```

该结果只用于验证框架和观察方向，不用于显著性声明。

初步表现：

- Type A XOR/乘法交互：4/4 个 case 中 **MetaGB 优于 XGB**。
- Type B Linear-in-Leaves：4/5 个 case 中 **MetaGB 优于 XGB**；`global_linear`
  在 smoke 中 XGB 更好，正式 Random300 需复查。
- Type C Conditional-Predictive Separation：3/3 个 case 中 **MetaGB 优于 XGB**。
- Type D Sparse Interaction：3/3 个 case 中 **MetaGB 优于 XGB**。
- Type E 负对照：`curved_control` 和 `additive_nonlinear_control` 中 XGB 更好；
  `pure_step_control` 中 **MetaGB 小幅领先**，需要后续复查。
- 独立高斯噪声压力：`noise_regime_slope__dim10` 中 **MetaGB 优于 XGB**。

更详细的每个 case 内涵和 smoke 结果见
`EXPERIMENT_DESIGN_ZH.md` 的 “Framework Smoke 初步观察” 部分。

See `EXPERIMENT_DESIGN_ZH.md` for the frozen A-E design,
`EXECUTION_PLAN_ZH.md` for completion tracking, and `LINUX_RUN_GUIDE.md` for
server commands.
