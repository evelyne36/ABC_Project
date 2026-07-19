# MetaGB Existing-Run Reanalysis Evidence

> 本报告从已保存的 `tasks/*.json` 离线重建；没有加载 XGBoost，也没有执行训练。
> “CI正向/负向”仅表示本次 paired bootstrap CI 是否完全位于 0 的一侧；
> paired unit 少于 2 时标记为“单位不足”。这些标签不是自动生成的普适论文结论。

## 1. 身份与完整性

- Protocol config hash: `aebd1e22a159119cf489ed70329fff84982b847e3ff8c68b0003e53b6586d2e3`
- Execution hash: `8ba754ac7c388e916f40e4e53bd7f26fea414ce75dc25521e14187aa3d209b69`
- Variant hash: `f3c608994792bec9cba601d6fcbba02de2900fafbf1f6db53e6e16f52fc5475d`
- Confirmatory: `True`
- Audit passed: `True`
- Tasks: `210/210`
- Fold rows: `3360/3360`
- Predictions: `2016000/2016000`

## 2. 主消融证据计数

正的 signed improvement 表示 target 优于 baseline。不要跨回归 RMSE 与分类
logloss 直接比较绝对 effect 大小；跨场景汇总使用相对改善的中位数。

| effect | scenario role | CI正向 cells | CI负向 cells | CI跨零 cells | 单位不足 cells | median relative improvement |
| --- | --- | --- | --- | --- | --- | --- |
| full_vs_xgb | mechanism_aligned | 2 | 2 | 4 | 0 | 0.0510561 |
| full_vs_xgb | mechanism_aligned_classification | 0 | 2 | 0 | 0 | -0.0287364 |
| full_vs_xgb | negative_control | 0 | 2 | 2 | 0 | 0.00653387 |
| leaf_vs_split | mechanism_aligned | 4 | 2 | 2 | 0 | 0.0755195 |
| leaf_vs_split | mechanism_aligned_classification | 0 | 2 | 0 | 0 | -0.0845775 |
| leaf_vs_split | negative_control | 2 | 0 | 2 | 0 | 0.0217415 |
| split_vs_bias | mechanism_aligned | 3 | 3 | 2 | 0 | -0.0558636 |
| split_vs_bias | mechanism_aligned_classification | 2 | 0 | 0 | 0 | 0.0503834 |
| split_vs_bias | negative_control | 0 | 3 | 1 | 0 | -0.0484791 |
| total_vs_bias | mechanism_aligned | 2 | 2 | 4 | 0 | 0.0795921 |
| total_vs_bias | mechanism_aligned_classification | 0 | 1 | 1 | 0 | -0.0299069 |
| total_vs_bias | negative_control | 0 | 3 | 1 | 0 | -0.028316 |

## 3. pred_fraction 剂量证据

| effect | pred_fraction | scenario role | CI正向 cells | CI负向 cells | CI跨零 cells | 单位不足 cells | median relative improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0.2 | mechanism_aligned | 3 | 4 | 1 | 0 | -0.0453724 |
| leaf | 0.2 | mechanism_aligned_classification | 0 | 2 | 0 | 0 | -0.0144668 |
| leaf | 0.2 | negative_control | 1 | 2 | 1 | 0 | -0.00150448 |
| leaf | 0.4 | mechanism_aligned | 3 | 4 | 1 | 0 | -0.0355233 |
| leaf | 0.4 | mechanism_aligned_classification | 0 | 2 | 0 | 0 | -0.0257156 |
| leaf | 0.4 | negative_control | 1 | 0 | 3 | 0 | 0.00156749 |
| leaf | 0.6 | mechanism_aligned | 4 | 3 | 1 | 0 | 0.0109309 |
| leaf | 0.6 | mechanism_aligned_classification | 0 | 2 | 0 | 0 | -0.0350291 |
| leaf | 0.6 | negative_control | 2 | 0 | 2 | 0 | 0.00743706 |
| leaf | 0.8 | mechanism_aligned | 4 | 4 | 0 | 0 | -0.000754999 |
| leaf | 0.8 | mechanism_aligned_classification | 0 | 2 | 0 | 0 | -0.0500433 |
| leaf | 0.8 | negative_control | 2 | 0 | 2 | 0 | 0.017379 |
| leaf | 1.0 | mechanism_aligned | 4 | 2 | 2 | 0 | 0.0755195 |
| leaf | 1.0 | mechanism_aligned_classification | 0 | 2 | 0 | 0 | -0.0845775 |
| leaf | 1.0 | negative_control | 2 | 0 | 2 | 0 | 0.0217415 |
| split | 0.2 | mechanism_aligned | 5 | 1 | 2 | 0 | 0.217922 |
| split | 0.2 | mechanism_aligned_classification | 1 | 0 | 1 | 0 | 0.0521299 |
| split | 0.2 | negative_control | 1 | 2 | 1 | 0 | -0.0727434 |
| split | 0.4 | mechanism_aligned | 3 | 2 | 3 | 0 | -0.0128906 |
| split | 0.4 | mechanism_aligned_classification | 2 | 0 | 0 | 0 | 0.0571672 |
| split | 0.4 | negative_control | 1 | 2 | 1 | 0 | -0.0517531 |
| split | 0.6 | mechanism_aligned | 3 | 1 | 4 | 0 | -0.0385347 |
| split | 0.6 | mechanism_aligned_classification | 2 | 0 | 0 | 0 | 0.0574527 |
| split | 0.6 | negative_control | 0 | 3 | 1 | 0 | -0.0450286 |
| split | 0.8 | mechanism_aligned | 3 | 2 | 3 | 0 | -0.0418784 |
| split | 0.8 | mechanism_aligned_classification | 2 | 0 | 0 | 0 | 0.0577851 |
| split | 0.8 | negative_control | 0 | 3 | 1 | 0 | -0.0490405 |
| split | 1.0 | mechanism_aligned | 3 | 3 | 2 | 0 | -0.0558636 |
| split | 1.0 | mechanism_aligned_classification | 2 | 0 | 0 | 0 | 0.0503834 |
| split | 1.0 | negative_control | 0 | 3 | 1 | 0 | -0.0484791 |
| total | 0.2 | mechanism_aligned | 3 | 2 | 3 | 0 | 0.0952564 |
| total | 0.2 | mechanism_aligned_classification | 1 | 0 | 1 | 0 | 0.0381943 |
| total | 0.2 | negative_control | 0 | 2 | 2 | 0 | -0.0739232 |
| total | 0.4 | mechanism_aligned | 2 | 2 | 4 | 0 | -0.0145731 |
| total | 0.4 | mechanism_aligned_classification | 1 | 0 | 1 | 0 | 0.0324645 |
| total | 0.4 | negative_control | 0 | 2 | 2 | 0 | -0.0492771 |
| total | 0.6 | mechanism_aligned | 2 | 2 | 4 | 0 | 0.0364487 |
| total | 0.6 | mechanism_aligned_classification | 1 | 1 | 0 | 0 | 0.023801 |
| total | 0.6 | negative_control | 0 | 3 | 1 | 0 | -0.0306902 |
| total | 0.8 | mechanism_aligned | 3 | 3 | 2 | 0 | 0.0510414 |
| total | 0.8 | mechanism_aligned_classification | 1 | 1 | 0 | 0 | 0.0100516 |
| total | 0.8 | negative_control | 0 | 4 | 0 | 0 | -0.0302466 |
| total | 1.0 | mechanism_aligned | 2 | 2 | 4 | 0 | 0.0795921 |
| total | 1.0 | mechanism_aligned_classification | 0 | 1 | 1 | 0 | -0.0299069 |
| total | 1.0 | negative_control | 0 | 3 | 1 | 0 | -0.028316 |

## 4. score_agg 证据

| effect | scenario role | CI正向 cells | CI负向 cells | CI跨零 cells | 单位不足 cells | median relative improvement |
| --- | --- | --- | --- | --- | --- | --- |
| mean_vs_greedy | mechanism_aligned | 3 | 1 | 4 | 0 | 0.0176178 |
| mean_vs_greedy | mechanism_aligned_classification | 0 | 1 | 1 | 0 | -0.0215313 |
| mean_vs_greedy | negative_control | 0 | 2 | 2 | 0 | -0.0172739 |
| median_vs_greedy | mechanism_aligned | 2 | 6 | 0 | 0 | -2.01676 |
| median_vs_greedy | mechanism_aligned_classification | 0 | 1 | 1 | 0 | -0.0146114 |
| median_vs_greedy | negative_control | 0 | 4 | 0 | 0 | -2.08321 |
| split_mean_vs_greedy | mechanism_aligned | 4 | 3 | 1 | 0 | 0.0581873 |
| split_mean_vs_greedy | mechanism_aligned_classification | 0 | 2 | 0 | 0 | -0.0620469 |
| split_mean_vs_greedy | negative_control | 0 | 1 | 3 | 0 | 0.00571293 |
| split_median_vs_greedy | mechanism_aligned | 1 | 7 | 0 | 0 | -2.92878 |
| split_median_vs_greedy | mechanism_aligned_classification | 0 | 2 | 0 | 0 | -0.134879 |
| split_median_vs_greedy | negative_control | 0 | 4 | 0 | 0 | -2.01871 |

## 5. 主消融逐 cell 明细

| scenario/dataset | dimension | role | effect | paired units | signed improvement | 95% CI | relative improvement | win rate | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| categorical_regime_slope | 10 | mechanism_aligned | full_vs_xgb | 3 | 0.0546435 | [-0.111818, 0.155248] | 0.0014772 | 0.666667 | CI跨零 |
| categorical_regime_slope | 10 | mechanism_aligned | leaf_vs_split | 3 | 0.0572104 | [0.0440692, 0.0679378] | 0.150787 | 1 | CI正向 |
| categorical_regime_slope | 10 | mechanism_aligned | split_vs_bias | 3 | -0.0545383 | [-0.177474, 0.0212736] | -0.292094 | 0.333333 | CI跨零 |
| categorical_regime_slope | 10 | mechanism_aligned | total_vs_bias | 3 | 0.00267214 | [-0.109536, 0.0808979] | -0.0880328 | 0.666667 | CI跨零 |
| categorical_regime_slope | 20 | mechanism_aligned | full_vs_xgb | 3 | 0.0709239 | [-0.0187026, 0.190692] | 0.155435 | 0.666667 | CI跨零 |
| categorical_regime_slope | 20 | mechanism_aligned | leaf_vs_split | 3 | 0.00732486 | [-0.0427555, 0.0332929] | 0.0241053 | 0.666667 | CI跨零 |
| categorical_regime_slope | 20 | mechanism_aligned | split_vs_bias | 3 | 0.105667 | [0.0274294, 0.207888] | 0.222471 | 1 | CI正向 |
| categorical_regime_slope | 20 | mechanism_aligned | total_vs_bias | 3 | 0.112992 | [-0.015326, 0.241181] | 0.2299 | 0.666667 | CI跨零 |
| curved_control | 10 | negative_control | full_vs_xgb | 3 | -0.00175421 | [-0.00286677, -0.000904738] | -0.0110205 | 0 | CI负向 |
| curved_control | 10 | negative_control | leaf_vs_split | 3 | 0.00799161 | [0.00759847, 0.00843707] | 0.0472449 | 1 | CI正向 |
| curved_control | 10 | negative_control | split_vs_bias | 3 | -0.0101135 | [-0.0104609, -0.00966236] | -0.0635945 | 0 | CI负向 |
| curved_control | 10 | negative_control | total_vs_bias | 3 | -0.0021219 | [-0.00227798, -0.00202384] | -0.0133397 | 0 | CI负向 |
| curved_control | 20 | negative_control | full_vs_xgb | 3 | -0.00301407 | [-0.00406035, -0.0022397] | -0.0189301 | 0 | CI负向 |
| curved_control | 20 | negative_control | leaf_vs_split | 3 | 0.00808616 | [0.00733794, 0.00875323] | 0.0473864 | 1 | CI正向 |
| curved_control | 20 | negative_control | split_vs_bias | 3 | -0.0113257 | [-0.0114239, -0.0112385] | -0.0711256 | 0 | CI负向 |
| curved_control | 20 | negative_control | total_vs_bias | 3 | -0.00323958 | [-0.00397685, -0.00248527] | -0.0203704 | 0 | CI负向 |
| feature_switch | 10 | mechanism_aligned | full_vs_xgb | 3 | 0.0138741 | [-0.0199349, 0.040998] | 0.0556662 | 0.666667 | CI跨零 |
| feature_switch | 10 | mechanism_aligned | leaf_vs_split | 3 | 0.0263697 | [0.00380324, 0.0470102] | 0.112723 | 1 | CI正向 |
| feature_switch | 10 | mechanism_aligned | split_vs_bias | 3 | -0.00559112 | [-0.0341956, 0.0365528] | -0.035036 | 0.333333 | CI跨零 |
| feature_switch | 10 | mechanism_aligned | total_vs_bias | 3 | 0.0207786 | [-0.00589986, 0.040356] | 0.0891496 | 0.666667 | CI跨零 |
| feature_switch | 20 | mechanism_aligned | full_vs_xgb | 3 | 0.00987833 | [-0.0181757, 0.0248729] | 0.046446 | 0.666667 | CI跨零 |
| feature_switch | 20 | mechanism_aligned | leaf_vs_split | 3 | 0.0311337 | [0.0243716, 0.0380993] | 0.135543 | 1 | CI正向 |
| feature_switch | 20 | mechanism_aligned | split_vs_bias | 3 | -0.0165404 | [-0.043374, -0.00106856] | -0.0766913 | 0 | CI负向 |
| feature_switch | 20 | mechanism_aligned | total_vs_bias | 3 | 0.0145933 | [-0.00527471, 0.0257515] | 0.0700347 | 0.666667 | CI跨零 |
| pure_step_control | 10 | negative_control | full_vs_xgb | 3 | 0.0067584 | [-0.000279949, 0.0190071] | 0.0240883 | 0.666667 | CI跨零 |
| pure_step_control | 10 | negative_control | leaf_vs_split | 3 | -0.000876312 | [-0.0083106, 0.00881127] | -0.00376191 | 0.333333 | CI跨零 |
| pure_step_control | 10 | negative_control | split_vs_bias | 3 | -0.00792545 | [-0.0150149, -0.00277402] | -0.0329782 | 0 | CI负向 |
| pure_step_control | 10 | negative_control | total_vs_bias | 3 | -0.00880177 | [-0.0110846, -0.0062036] | -0.0362616 | 0 | CI负向 |
| pure_step_control | 20 | negative_control | full_vs_xgb | 3 | 0.00601289 | [-0.00920945, 0.0224634] | 0.0248681 | 0.666667 | CI跨零 |
| pure_step_control | 20 | negative_control | leaf_vs_split | 3 | -0.00286955 | [-0.0228931, 0.0223769] | -0.00987882 | 0.333333 | CI跨零 |
| pure_step_control | 20 | negative_control | split_vs_bias | 3 | -0.00760528 | [-0.0181393, 0.00698444] | -0.0333637 | 0.333333 | CI跨零 |
| pure_step_control | 20 | negative_control | total_vs_bias | 3 | -0.0104748 | [-0.0197535, 0.00423761] | -0.0404102 | 0.333333 | CI跨零 |
| regime_logit | 10 | mechanism_aligned_classification | full_vs_xgb | 3 | -0.0192239 | [-0.0323838, -0.0111769] | -0.0473441 | 0 | CI负向 |
| regime_logit | 10 | mechanism_aligned_classification | leaf_vs_split | 3 | -0.0407729 | [-0.0481445, -0.0370723] | -0.106672 | 0 | CI负向 |
| regime_logit | 10 | mechanism_aligned_classification | split_vs_bias | 3 | 0.020618 | [0.0138468, 0.0253809] | 0.0515296 | 1 | CI正向 |
| regime_logit | 10 | mechanism_aligned_classification | total_vs_bias | 3 | -0.0201549 | [-0.0342977, -0.011721] | -0.0497753 | 0 | CI负向 |
| regime_logit | 20 | mechanism_aligned_classification | full_vs_xgb | 3 | -0.00434742 | [-0.00575306, -0.00261418] | -0.0101286 | 0 | CI负向 |
| regime_logit | 20 | mechanism_aligned_classification | leaf_vs_split | 3 | -0.0256473 | [-0.0303702, -0.0193543] | -0.0624825 | 0 | CI负向 |
| regime_logit | 20 | mechanism_aligned_classification | split_vs_bias | 3 | 0.0214552 | [0.0108654, 0.0351974] | 0.0492373 | 1 | CI正向 |
| regime_logit | 20 | mechanism_aligned_classification | total_vs_bias | 3 | -0.00419204 | [-0.0120673, 0.00798003] | -0.0100385 | 0.333333 | CI跨零 |
| regime_slope | 10 | mechanism_aligned | full_vs_xgb | 3 | 0.242231 | [0.136399, 0.314778] | 0.436484 | 1 | CI正向 |
| regime_slope | 10 | mechanism_aligned | leaf_vs_split | 3 | 0.0203596 | [-0.000651985, 0.0398266] | 0.0654677 | 0.666667 | CI跨零 |
| regime_slope | 10 | mechanism_aligned | split_vs_bias | 3 | 0.215626 | [0.129909, 0.332263] | 0.388691 | 1 | CI正向 |
| regime_slope | 10 | mechanism_aligned | total_vs_bias | 3 | 0.235986 | [0.151813, 0.331611] | 0.431256 | 1 | CI正向 |
| regime_slope | 20 | mechanism_aligned | full_vs_xgb | 3 | 0.275311 | [0.186524, 0.415271] | 0.479976 | 1 | CI正向 |
| regime_slope | 20 | mechanism_aligned | leaf_vs_split | 3 | 0.0268173 | [0.00311729, 0.0477713] | 0.0855713 | 1 | CI正向 |
| regime_slope | 20 | mechanism_aligned | split_vs_bias | 3 | 0.294654 | [0.246097, 0.36282] | 0.485788 | 1 | CI正向 |
| regime_slope | 20 | mechanism_aligned | total_vs_bias | 3 | 0.321471 | [0.293868, 0.365938] | 0.532209 | 1 | CI正向 |
| sparse_multiplicative | 10 | mechanism_aligned | full_vs_xgb | 3 | -1.22653 | [-1.39895, -1.11828] | -1.94074 | 0 | CI负向 |
| sparse_multiplicative | 10 | mechanism_aligned | leaf_vs_split | 3 | -0.71411 | [-0.811913, -0.639614] | -0.619289 | 0 | CI负向 |
| sparse_multiplicative | 10 | mechanism_aligned | split_vs_bias | 3 | -0.487711 | [-0.524238, -0.458576] | -0.735028 | 0 | CI负向 |
| sparse_multiplicative | 10 | mechanism_aligned | total_vs_bias | 3 | -1.20182 | [-1.33615, -1.11993] | -1.81267 | 0 | CI负向 |
| sparse_multiplicative | 20 | mechanism_aligned | full_vs_xgb | 3 | -0.975575 | [-1.07028, -0.847893] | -1.28893 | 0 | CI负向 |
| sparse_multiplicative | 20 | mechanism_aligned | leaf_vs_split | 3 | -0.841166 | [-0.881427, -0.817943] | -0.931581 | 0 | CI负向 |
| sparse_multiplicative | 20 | mechanism_aligned | split_vs_bias | 3 | -0.116542 | [-0.141818, -0.0684503] | -0.149916 | 0 | CI负向 |
| sparse_multiplicative | 20 | mechanism_aligned | total_vs_bias | 3 | -0.957708 | [-1.02324, -0.886393] | -1.22177 | 0 | CI负向 |

## 6. 写结论时的边界

- `split_vs_bias` 对应 predictive-aware split 的独立贡献；
- `leaf_vs_split` 对应 predictive leaf 的增量贡献；
- `total_vs_bias` 对应完整机制相对 Bias 的总贡献；
- `full_vs_xgb` 是 fixed-backbone 诊断对比，不等同于各自调优后的最终性能；
- 必须单独检查 negative controls；若其改善与机制匹配场景同量级，应收缩机制归因；
- paired unit 是 seed/repeat，不能把 outer folds 当作 iid 重复。
