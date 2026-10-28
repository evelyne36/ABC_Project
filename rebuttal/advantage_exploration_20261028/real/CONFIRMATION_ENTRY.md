# NOX confirmation 生成入口

入口为 `prepare_confirmation.py`；当前仅准备入口，未生成 confirmation 数据。

默认读取根任务 `../confirmation_freeze.json` 并只校验。必须具有 UTC `frozen_utc`、两个完整 cases（`A06_two_sparse_modules`、`gas_turbine_nox`）、六个完整 methods（CrossGB、XGB、LGB_constant、LGB_linear、PL、LinXGBoost）。`choices[case][method]` 必须同时含非空 `params`、匹配的 `params_sha256`、非负整数 `candidate` 和有限数值 `mean_development_primary`；缺少任一 case 的任一 method 都不放行。

参数摘要固定为 `sha256(json.dumps(params, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())`。

校验调用：

```bash
python prepare_confirmation.py
```

冻结单完整后，显式生成：

```bash
python prepare_confirmation.py --generate
```

结果写入全新 `confirmation/data/gas_turbine_nox__s113001.npz` 至 `__s113005.npz`、各自 JSON 和 `confirmation/manifest.json`；拒绝覆盖已有 confirmation 或未完成的 pending 目录。生成前逐项核对 DEV 生成器、既有标准化实现、官方源档案的冻结 SHA256，以及两个 DEV 的全部时间边界。严格复用 2011–12 / 2013 / 2014–15 划分、NOX 目标、全部 9 项输入、cap600 与原训练内预处理。只用新种子决定训练 subsample，不修改目标/特征/划分。

五包共享既定 2013 validation 和 2014–15 test。既有 DEV 生成器、manifest 和包不被改写。脚本记录完整冻结单摘要和六方法选定配置；只生成数据，不拟合、不评分。每个 NPZ 的 `y_test` 独立封存，仍应由搜索 worker 的 `evaluate_test=False` 保持未读。

当前入口校验记录在 `confirmation_entry_verification.json`：完整 schema 被接受；另一 case 缺方法、错误参数摘要、缺冻结时间、空参数及非有限选择指标均被拒绝；confirmation 包数量仍为 0。
