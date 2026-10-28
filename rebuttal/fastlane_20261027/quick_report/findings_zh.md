# Fastlane 结果汇总

验证集与测试集分别汇总；RMSE 已按各训练重复的 y_scale 换回原始目标单位。缺少缩放元数据时不输出伪装为原单位的 RMSE。
真实数据的重复共享固定测试集，不能作为独立新数据集；合成确认种子生成独立数据。容量探针固定树路由，仿射叶子近似不代表完整 k>1 算法。

| 套件 / 数据 | 方法 | 指标 | 成功覆盖 | 均值 ± 样本标准差 | 状态 |
|---|---|---|---|---|---|
| quick_v1 / S1_dense | CrossGB | rmse | 1/1 | 2.12202 ± — | 完整 |
| quick_v1 / S1_dense | LGB_constant | rmse | 1/1 | 2.174 ± — | 完整 |
| quick_v1 / S1_dense | LGB_linear | rmse | 1/1 | 2.17512 ± — | 完整 |
| quick_v1 / S1_dense | LinXGBoost | rmse | 1/1 | 0.439099 ± — | 完整 |
| quick_v1 / S1_dense | PL | rmse | 1/1 | 2.18985 ± — | 完整 |
| quick_v1 / S1_dense | XGB | rmse | 1/1 | 1.86321 ± — | 完整 |
| quick_v1 / S1_offset | CrossGB | rmse | 1/1 | 0.350697 ± — | 完整 |
| quick_v1 / S1_offset | LGB_constant | rmse | 1/1 | 0.327185 ± — | 完整 |
| quick_v1 / S1_offset | LGB_linear | rmse | 1/1 | 0.325484 ± — | 完整 |
| quick_v1 / S1_offset | LinXGBoost | rmse | 1/1 | 0.418552 ± — | 完整 |
| quick_v1 / S1_offset | PL | rmse | 1/1 | 0.390883 ± — | 完整 |
| quick_v1 / S1_offset | XGB | rmse | 1/1 | 0.32736 ± — | 完整 |
| quick_v1 / S1_sparse | CrossGB | rmse | 1/1 | 0.339737 ± — | 完整 |
| quick_v1 / S1_sparse | LGB_constant | rmse | 1/1 | 2.0002 ± — | 完整 |
| quick_v1 / S1_sparse | LGB_linear | rmse | 1/1 | 1.83068 ± — | 完整 |
| quick_v1 / S1_sparse | LeafOnly | rmse | 1/1 | 1.63358 ± — | 完整 |
| quick_v1 / S1_sparse | LinXGBoost | rmse | 1/1 | 0.429155 ± — | 完整 |
| quick_v1 / S1_sparse | PL | rmse | 1/1 | 0.534932 ± — | 完整 |
| quick_v1 / S1_sparse | XGB | rmse | 1/1 | 1.54484 ± — | 完整 |
| quick_v1 / S2_sparse | CrossGB | rmse | 1/1 | 0.952771 ± — | 完整 |
| quick_v1 / S2_sparse | LGB_constant | rmse | 1/1 | 1.60256 ± — | 完整 |
| quick_v1 / S2_sparse | LGB_linear | rmse | 1/1 | 1.47737 ± — | 完整 |
| quick_v1 / S2_sparse | LeafOnly | rmse | 1/1 | 1.5082 ± — | 完整 |
| quick_v1 / S2_sparse | LinXGBoost | rmse | 1/1 | 1.74115 ± — | 完整 |
| quick_v1 / S2_sparse | PL | rmse | 1/1 | 1.21943 ± — | 完整 |
| quick_v1 / S2_sparse | XGB | rmse | 1/1 | 1.57135 ± — | 完整 |
| quick_v1 / S3_sparse | CrossGB | rmse | 1/1 | 0.467461 ± — | 完整 |
| quick_v1 / S3_sparse | LGB_constant | rmse | 1/1 | 1.49681 ± — | 完整 |
| quick_v1 / S3_sparse | LGB_linear | rmse | 1/1 | 1.40863 ± — | 完整 |
| quick_v1 / S3_sparse | LeafOnly | rmse | 1/1 | 1.41269 ± — | 完整 |
| quick_v1 / S3_sparse | LinXGBoost | rmse | 1/1 | 0.464052 ± — | 完整 |
| quick_v1 / S3_sparse | PL | rmse | 1/1 | 0.941982 ± — | 完整 |
| quick_v1 / S3_sparse | XGB | rmse | 1/1 | 1.51778 ± — | 完整 |
| quick_v1 / cpu_activity | CrossGB | rmse | 1/1 | 3.03695 ± — | 完整 |
| quick_v1 / cpu_activity | LGB_constant | rmse | 1/1 | 2.51978 ± — | 完整 |
| quick_v1 / cpu_activity | LGB_linear | rmse | 1/1 | 2.42826 ± — | 完整 |
| quick_v1 / cpu_activity | PL | rmse | 1/1 | 2.528 ± — | 完整 |
| quick_v1 / cpu_activity | XGB | rmse | 1/1 | 2.99385 ± — | 完整 |
| quick_v1 / gas_turbine_co | CrossGB | rmse | 1/1 | 1.45597 ± — | 完整 |
| quick_v1 / gas_turbine_co | LGB_constant | rmse | 1/1 | 1.42827 ± — | 完整 |
| quick_v1 / gas_turbine_co | LGB_linear | rmse | 1/1 | 1.40187 ± — | 完整 |
| quick_v1 / gas_turbine_co | LinXGBoost | rmse | 1/1 | 1.86405 ± — | 完整 |
| quick_v1 / gas_turbine_co | PL | rmse | 1/1 | 1.42551 ± — | 完整 |
| quick_v1 / gas_turbine_co | XGB | rmse | 1/1 | 1.51581 ± — | 完整 |
| quick_v1 / kin8nm | CrossGB | rmse | 1/1 | 0.126013 ± — | 完整 |
| quick_v1 / kin8nm | LGB_constant | rmse | 1/1 | 0.156042 ± — | 完整 |
| quick_v1 / kin8nm | LGB_linear | rmse | 1/1 | 0.146051 ± — | 完整 |
| quick_v1 / kin8nm | LinXGBoost | rmse | 1/1 | 0.185374 ± — | 完整 |
| quick_v1 / kin8nm | PL | rmse | 1/1 | 0.122452 ± — | 完整 |
| quick_v1 / kin8nm | XGB | rmse | 1/1 | 0.154705 ± — | 完整 |
| quick_v1 / parkinson_total | CrossGB | rmse | 1/1 | 11 ± — | 完整 |
| quick_v1 / parkinson_total | LGB_constant | rmse | 1/1 | 10.0297 ± — | 完整 |
| quick_v1 / parkinson_total | LGB_linear | rmse | 1/1 | 10.4168 ± — | 完整 |
| quick_v1 / parkinson_total | PL | rmse | 1/1 | 10.9055 ± — | 完整 |
| quick_v1 / parkinson_total | XGB | rmse | 1/1 | 10.0526 ± — | 完整 |
| quick_v1 / protein | CrossGB | rmse | 1/1 | 4.51077 ± — | 完整 |
| quick_v1 / protein | LGB_constant | rmse | 1/1 | 4.58513 ± — | 完整 |
| quick_v1 / protein | LGB_linear | rmse | 1/1 | 4.53684 ± — | 完整 |
| quick_v1 / protein | LinXGBoost | rmse | 1/1 | 5.09768 ± — | 完整 |
| quick_v1 / protein | PL | rmse | 1/1 | 4.50769 ± — | 完整 |
| quick_v1 / protein | XGB | rmse | 1/1 | 4.56702 ± — | 完整 |
| quick_v1 / superconductivity | CrossGB | rmse | 1/1 | 13.0887 ± — | 完整 |
| quick_v1 / superconductivity | LGB_constant | rmse | 1/1 | 12.6774 ± — | 完整 |
| quick_v1 / superconductivity | LGB_linear | rmse | 1/1 | 12.7285 ± — | 完整 |
| quick_v1 / superconductivity | PL | rmse | 1/1 | 13.2791 ± — | 完整 |
| quick_v1 / superconductivity | XGB | rmse | 1/1 | 12.8497 ± — | 完整 |

上述不完整行仅描述已成功的重复，不冒充完整计划结果；不得据不完整覆盖给出稳定性或总体胜出结论。
审计：读取结果文件 858，容量文件 3；镜像重复 0，冲突组 0，缺失元数据 0，开发阶段测试暴露 0。
训练耗时为原始单次拟合计时；容量探针拟合耗时不含既有树结构训练或共享路由计算，不能与完整模型训练直接比较。

未获得完整开发候选、未进入最终确认的方法：parkinson_total / LinXGBoost；superconductivity / LinXGBoost；cpu_activity / LinXGBoost。
