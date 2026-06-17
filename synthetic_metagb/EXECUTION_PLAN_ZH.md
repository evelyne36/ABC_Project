# Synthetic MetaGB 执行计划

## 冻结范围

- 主实验：仅 `xgb_cpu` vs `current_metagb`
- 协议：单 seed Random300
- `data_seed=42`，`search_seed=42`
- 57 个 cells
- 5 outer × 4 inner nested CV
- 300 candidates/model
- Linux：`task_workers=15, n_jobs=1`
- 本地只运行框架 smoke，不运行完整 Random300

## 执行清单

| 步骤 | 工作项 | 验收标准 | 状态 |
| --- | --- | --- | --- |
| 1 | 冻结 A-E 与独立噪声 case 设计 | `EXPERIMENT_DESIGN_ZH.md` 完整描述 57 cells | 已完成 |
| 2 | 重构 scenario registry | 每个 case 有 family、任务类型、公式和生成器 | 已完成 |
| 3 | 实现维度展开 | 每个 case 可稳定生成 10/50/100 个特征 | 已完成 |
| 4 | 实现主协议 | 单 seed Random300 精确生成 57 cells、570 tasks | 已完成 |
| 5 | 主实验移除 pred0 | 候选、公平性审计、任务和统计仅包含两个模型 | 已完成 |
| 6 | 扩展趋势统计 | 输出 cell/case/type 三级结果和维度趋势 | 已完成 |
| 7 | 补充测试 | 生成器复现、维度、Gaussian noise、协议数量、nested CV、公平性 | 已完成 |
| 8 | 运行框架 smoke | 每个 A-E/噪声基础 case 的 dim10 代表 cell 跑通，无缺失 | 已完成 |
| 9 | 更新 Linux 指南 | 给出单 seed Random300 命令、恢复和完整性检查 | 已完成 |
| 10 | Linux 正式执行 | 用户在服务器运行完整单 seed Random300 | 服务器阶段 |

## 质量门槛

正式服务器运行前必须满足：

1. 18 个 A-E 基础生成器和独立噪声生成器均可固定 seed 完全复现。
2. 每个生成器的 `dim10/dim50/dim100` 输出维度准确。
3. 无关特征均来自独立 `Normal(0,1)`。
4. 目标噪声均为 Gaussian noise。
5. 主实验候选中不出现 pred0。
6. 两模型使用相同候选预算和 paired backbone。
7. split manifest 证明 outer test 不参与候选选择。
8. 57 cells、570 tasks、171,000 candidate rows、684,570 fits 的协议算术通过。
9. 恢复运行不会重复已完成任务。
10. 本地框架 smoke 使用 19 个 dim10 代表 cells；三种维度的正确性由生成器契约测试覆盖。
11. 本地框架 smoke 全部成功后才提供 Linux 正式命令。

## 本轮完成证据

- 契约测试：21 passed。
- Framework smoke：19 cells，76/76 tasks，2,280/2,280 candidate rows。
- Smoke 模型集合：仅 `xgb_cpu`、`current_metagb`。
- Resume 验证：`pending_tasks=0`。
- 新输出：`case_summary.csv`、`type_summary.csv`、
  `selected_complexity.csv`。
- Framework smoke 结果：
  `results/synthetic_main_framework_smoke_dim10_20260615/`。
