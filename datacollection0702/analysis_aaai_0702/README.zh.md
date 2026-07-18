# datacollection0702 NeurIPS 风格分析说明

本目录保存针对 `datacollection0702` 实验快照的增量 Markdown 分析。目录
被刻意放在一个新的子文件夹下，避免将 0702 的新解释与此前
`mgb_datacollection_by0629` 中的 0629 报告混淆。

## 文件

- `mgb_vs_xgb_neurips_analysis_0702.zh.md`：主分析报告，按照面向论文实验
  章节的口径组织。
- `appendix_tables_0702.zh.md`：主报告使用的支持性表格、coverage 说明以及
  正负代表案例。

对应英文版保留在同一目录：

- `mgb_vs_xgb_neurips_analysis_0702.md`
- `appendix_tables_0702.md`

## 范围约定

本分析仅为文档工作，不修改 benchmark 代码、原始日志、JSON 产物、CSV 汇总
或任何实验输出。

主要统计单位是 paired outer fold：即在相同数据集、任务、预处理方式、实验
协议和 outer-fold index 下，`xgb_cpu` 与 `current_metagb` 都成功完成的外层
fold。
