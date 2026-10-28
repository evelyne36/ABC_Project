# 四种近邻baseline：本机适配与实验入口

2026-10-25。用户已授权本机适配。这里是独立实验目录；原论文、历史结果及原工程未改动。**本机回归训练/预测核验已通过，正式300/600候选面板尚未启动，CGB训练次数为0。**

## 1. 这次实际接入的是什么

| 命令中的方法名 | 实现 | 必须保留的实验标记 |
|---|---|---|
| `linxgboost` | 作者固定commit的Python代码 | 保留上游“最后一棵树不乘学习率”的公开predict语义；未静默修成标准shrinkage |
| `pilot` | 作者固定commit的Python/Numba实现 | 直接加载作者Python模块，绕过包初始化中不必要的C++导入；没有换成其他模型 |
| `gbdt_pl` | 原始GBDT-PL C++＋平台兼容层 | 本机为ARM标量指令兼容＋Accelerate LAPACK；记录全部补丁，不能把本机耗时当原x86/MKL速度 |
| `tao` | 官方2026-07-22 Python包 | **斜分裂＋常数回归叶**；按用户确认尊重源码，不增加自实现线性叶，不冒称NeurIPS2025线性叶版本 |

本轮公共入口面向**标量回归**；没有把PILOT或TAO自行包装成boosting，也没有宣称分类接口已完成。仓库版本、archive SHA256和代码指纹见 `reports/sources.json`。上游代码修改会被运行入口拒绝；修改自有适配代码、候选或输入后，也不会静默沿用旧调参结果。

## 2. 本机已完成的核验

- 四方法的独立进程真实训练、有限预测，保留参数、来源、环境、预测NPZ和时间。
- 小合成例与Moneyball原outer1各一组集成检查；不是最终benchmark数值。
- 输入不被修改，NumPy随机状态恢复；预测重复调用和行顺序一致。
- LinXGBoost公开predict与原树组合规则一致；TAO单叶退化为原训练标签均值。
- GBDT-PL本机库编译、必要SIMD/BMI2兼容语义及DGESV求解检查；不加载未识别的native库。
- 原四个合成机制的外/内折与归档manifest逐项一致。真实数据用原loader和split函数重建，未找回独立的历史real索引hash，证据范围在输入manifest中明确记录。
- 嵌套验证按全部四内折RMSE选优，再做一次outer refit/test；断点续跑校验配置与输入身份。

测试：

```bash
cd rebuttal_baselines_20261025
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/smoke_all.py
.venv/bin/python scripts/smoke_all.py --data inputs/moneyball
```

当前完整测试为16项。TAO原始代码有NumPy数组转标量的弃用提示；锁定的环境中训练已通过，原算法代码没有为消除提示而改写。

## 3. 已准备的数据与原协议

`inputs/`包含五真实回归与四合成机制的数据NPZ、全部五外折/四内折索引、来源和hash，服务器无需重新读历史工程就能运行。

| 数据 | 行数 | 实际训练输入维数 | 原CGB候选预算 |
|---|---:|---:|---:|
| Kin8nm | 8192 | 8 | 300 |
| Moneyball | 114 | **42，原14列经one-hot** | 300 |
| CPU activity | 8192 | 21 | 300 |
| Houses | 20640 | 8 | 600 |
| Energy efficiency | 768 | 8 | 300 |
| Regime slope / Feature switch / Sparse multiplicative / Curved control | 各3000 | 各10 | 各300 |

真实输入保留历史的全输入one-hot schema；不要将其描述为“编码器在每折内拟合”。四种方法看到同一冻结输入。可选`--standardize`会另在每个fit的训练分区拟合scaler，默认关闭并明确记录。以后若改用严格fold-local编码，应作为新协议，不能静默替换这套归档匹配结果。

真实外折seed42，内折seed42＋从0开始的outer index。主合成data seed42，outer seed314159，inner seed314159＋从1开始的outer index；不要与E1独立开发seed冻结骨干消融混用。

必要时从原工程重建输入，输出目录必须尚不存在：

```bash
.venv/bin/python scripts/prepare_protocol.py --dataset cpu_activity --output-root new_inputs
```

## 4. 直接跑一个方法的原五折/四内折比较

已生成 `configs/方法名_300.json` 和 `configs/方法名_600.json`，每方法固定search seed42，600配置的前300项与300文件一致。包含一个作者起始配置，以及覆盖小树数/多树数、容量和正则的后续空间；这是本轮预先声明的新baseline搜索空间，**不是声称完整复现作者论文的原调参**。应先在Linux看运行成本，再冻结正式范围；不按新的test胜负筛选数据。

下面以GBDT-PL在CPU outer1为例，**默认仅预览数量，不训练**：

```bash
.venv/bin/python scripts/run_nested.py --method gbdt_pl --data inputs/cpu_activity --candidate-file configs/gbdt_pl_300.json --outer-fold 1 --output runs/gbdt_pl_cpu_outer1
```

正式启动时在同一命令后加`--execute --workers 4`。不写`--outer-fold`则评估全部五折。Houses使用对应600文件。方法名替换为`linxgboost`、`pilot`或`tao`即可。

- 一个300候选outer job：300×4＋1＝**1201个完整拟合**；五折6005次。Houses每outer2401次。
- 五真实＋四合成完整覆盖：每方法60,045 fits，四方法共240,180 fits，均不含CGB。先完成原定核心子面板，不能把“已适配”当作这些任务已经跑完。
- 并行发生在独立fit进程，单fit单线程。32核机器可先总计4个并行fit看内存，再扩；若同时开多个outer job，要控制其worker总数。
- 默认每fit最多600秒，可用`--fit-timeout`调整并在正式评测前冻结。候选任一内折失败/超时，整个候选不参加选优，失败记录保留；不是忽略失败折后取剩余平均。
- 加`--resume`只复用身份一致的成功任务，失败尝试另存，不覆盖原结果。改参数、源版本、编码/缩放或适配代码时使用新输出目录。
- 本机GBDT-PL使用标量兼容指令；Linux x86构建使用原AVX2/BMI2指令＋平台LAPACK。两边数值/耗时须分别核验，不能把Mac上的集成耗时用于与历史Linux CGB宣称等CPU优势。

这一步需要用户实际启动，本轮没有运行正式面板。

## 5. 单fit接口：供已有实验工程调用

```python
from baselines import make_regressor

model = make_regressor("pilot", {"max_depth": 6}, seed=42, n_jobs=1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

已有runner可以返回这个对象作为estimator。但GBDT-PL上游没有完整释放模型的C API，**长时间搜索应使用本包的独立fit进程入口**，避免在同一常驻进程堆积native模型。

独立入口读取NPZ中的 `X_train, y_train, X_predict`，可选 `y_evaluate`。后者仅计算指标，不传入fit：

```bash
.venv/bin/python scripts/run_task.py --method linxgboost --input one_fit.npz --params-file one_params.json --output one_result
```

每个结果包含 `result.json`、`predictions.npz`；TAO另存官方训练日志。输出目录已存在时拒绝覆盖。GBDT-PL使用`verbose=1`：上游`verbose=0`会访问未初始化metric；原预测/分裂/叶拟合公式保留。

计时字段分别记录wall和process CPU；**单进程冷启动的fit时间包含首次导入/JIT**，不能冒称预热后的训练内核速度。完整预算/成本比较仍按E4另行设计。

## 6. Linux接续

迁移本目录的最小包 `baseline_linux_bundle.tar.gz`；不迁移Mac `.venv`、`.dylib`和对象文件。它包含四方法必要源码、9份冻结输入、候选配置、测试和来源记录。

在Linux解包后，进入 `rebuttal_baselines_20261025/`：

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/build_gbdt_pl.py
.venv/bin/python scripts/smoke_all.py
.venv/bin/python -m pytest -q tests/test_adapters.py tests/test_task_cli.py tests/test_nested_cli.py
```

GBDT-PL需要g++/OpenMP、LAPACK和BLAS开发库。Ubuntu/Debian缺失时安装`build-essential liblapack-dev libblas-dev`；无需为本包安装MKL或PILOT的C++依赖。x86服务器需支持AVX2/BMI2/POPCNT；本轮尚未在目标Linux运行，不称其部署已通过。

上述Linux测试为14项运行/续跑检查。另两项“从历史原工程重新生成输入”的测试需要原工程，最小迁移包不含原工程；本机完整16项已验证，服务器使用已冻结的inputs即可。

源码有意保持可追溯：未修改下载的vendor；仅在生成的build副本中补C接口成功返回和单线程设置，其他平台差异集中在compat头文件。补丁及完整编译命令见 `reports/gbdt_pl_compatibility.patch` 与 `reports/gbdt_pl_build.json`。
