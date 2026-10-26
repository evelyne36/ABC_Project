# 表格生产清单（10/27）

用途：把已有逐折结果转成 rebuttal 可引用的表。**每张表两份**：中文内部表（拍板用）+ 英文附录草稿（可直接承诺 "we will add Table Sx"）。

| 表 | 内容 | 对应审稿诉求 | 数据状态 | 本研究进度 |
|---|---|---|---|---|
| [T1_N1_近邻主表](T1_N1_近邻主表.md) | 六方法 × 五数据集 test RMSE | Rev1-1 / Rev3-3 / Rev5-1 / Rev2-3 | ✅ 已回传 131 条逐折 | ✅ **已完成（中性版 + 名次版 + 英文草稿）** |
| [T2_N2_四格反事实表](T2_N2_四格反事实表.md) | XGB anchor / SplitOnly / LeafOnly / Full | Rev1-2（Table 2 缺格） | 🟡 105/105 fit 完成，待聚合 | 结构已定，待填数 |
| [T3_N3_等预算表](T3_N3_等预算表.md) | 60/300/900 s × 3 数据集 budget–performance | Rev2-2 / Rev3-2 / Rev5-2 | 🟡 EXIT=0，3721 候选文件 | 结构已定，待填数 |
| [T4_N4b_结构诊断表](T4_N4b_结构诊断表.md) | 正/负例机制对照（cpu vs kin） | Rev1-3 / Rev3-2 | 🟡 20/20 selected refit 完成 | 结构已定，待填数 |
| [T5_N6a_时序参考表](T5_N6a_时序参考表.md) | 十任务 × persistence/naive/Ridge/CrossGB | Rev5-3 | 🟡 130 wrapper fits | 结构已定，待填数 |
| T6（可选）N5 容量表 | k=1/2/3 + affine k=1 | Rev3-4 / Rev5-4 | 🔵 用户已自行启动 | 接到通知后聚合 |

## 硬口径（每张表都要守）

1. **不混口径**：18.72×（搜索逐折比中位数）≠ 28.14×（final-fit 比）；censored 折不进精度均值。
2. **不偷偷剔除**：GBDT-PL `cpu_activity/outer1 = 79.68` 必须披露（报中位数 + 脚注），核实工作排在 N5 之后。
3. **不冒称版本**：TAO = 2026-07-22 官方包（斜分裂 + 常数回归叶），不是 NeurIPS2025 线性叶版。
4. **不越权解释**：表下只写中性事实句，不写因果与主张（主命题待合作者拍板）。
5. **英文草稿统一脚注**：shared protocol / candidate budget / censored / single-tree 四条必须出现。

## 待合作者拍板（我会在表里预留两版或脚注）

- N1 主表是否进 appendix、LinXGBoost 用哪个口径（600 s censored 还是 2400 s 补充）作为主报数。
- TAO / PILOT 单树是否进表。
- 主命题是否收缩（不影响出数，影响表下文字）。
