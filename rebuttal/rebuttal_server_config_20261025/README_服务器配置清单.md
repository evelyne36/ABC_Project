# CrossGB rebuttal：Linux 服务器配置清单

> **复用审计后更新：** 当前默认方案见[复用优先实验规划](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/rebuttal_plan_20261025/实验侧规划书.md)和[证据审计](/Users/ancd/VSCodeProject/crossgbRebuttalV1-261024/rebuttal_plan_20261025/已有证据复用审计_E0-E6.md)。本包原E1/E2/E6 JSON与420/100行矩阵保留为旧全量协议备份，**不能直接启动**；须先生成复用队列。依赖、源码、数据映射与TS参考入口仍可沿用。新推荐核心CGB为178次，含容量路由补齐为220次；严格预算搜索等另计。输入校验通过不表示复用队列已实现。

> **后续本机适配更新：** 用户已另行授权四种baseline本机适配，成果位于 `../rebuttal_baselines_20261025/`。已验证回归入口；TAO为官方常数叶。该目录有独立Linux迁移包与冻结输入，本包原全量矩阵仍不能直接启动。

日期：2026-10-25。核心方案已经批准；按你的最新要求，本轮只交付服务器配置，不继续在本地安装、编译或训练。**本包没有新增实验结果。**

最先看本页，其次看 `configs/`。配置中所有数据与源码路径都相对于整个项目根目录，迁移时不依赖 Mac 的绝对路径。

## 1. 服务器与环境

| 项目 | 配置 |
|---|---|
| 系统 | Linux，优先 x86_64，GCC/G++、CMake、OpenMP/libgomp、Git |
| 计算 | 32 CPU 核、64GB RAM；GPU 不需要 |
| 磁盘 | 建议至少 50GB 空闲，留给两份 native build、预测和候选记录；原日志全集无需上传 |
| Python | 优先现有 3.13.x；历史记录为 3.13.14。使用 3.12.x 时记录环境差异，不称逐位复现 |
| 模型线程 | 每模型 1，CV 内不再开并行 |
| 并发 | 先 4 workers 看内存，升到 16；峰值允许时最多 24 |
| 内存限制 | 所有 worker 的峰值常驻内存合计目标 ≤48GB |
| 原生库 | 自行构建本包指定的历史 MetaGB/XGBoost 源码；不安装 pip 的 XGBoost 来替代 |

`configs/server.json` 给出机器、线程环境、源版本和两个近邻仓库。`requirements-core.txt` 固定与原记录相近的科学库；构建时应记录实际 Python、编译器、glibc、OpenMP 和 native library SHA256。

在远程项目根目录中，设置三个变量即可：

```bash
export CGB_PROJECT="$PWD"
export CGB_CFG="$CGB_PROJECT/rebuttal_server_config_20261025"
export CGB_RUNS="$CGB_PROJECT/rebuttal_server_runs"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

`CGB_PROJECT` 必须指向保留当前目录结构的远程项目根，不是原工程的某个子目录。不需要改 HOME 或其他系统变量。

## 2. 输入与传输

已映射 5 个主回归、3 个可选分类、6 份时序原始文件，并保存当前归档 SHA256：

| 主回归 | 原文件 | 原行数/输入列数 |
|---|---|---:|
| Kin8nm | `tabular_extra_random300/data/regression__kin8nm.csv` | 8192 / 8 |
| Moneyball | `tabular_extra_random300/data/regression__Moneyball.csv` | 114 / 14 |
| CPU activity | `tabular_extra_random300/data/regression__cpu_activity.csv` | 8192 / 21 |
| Houses | `dataset/TabArena_regression/houses.csv` | 20640 / 8 |
| Energy efficiency | `tabular_extra_random300/data/regression__energy_efficiency.csv` | 768 / 8 |

上表路径均在 `aaai_FullCodeProject-check_codereview0701/` 下。时序统一从其 `paper_writing/timeseries_extra_random300/data/raw/` 取文件，那里包含完整六份输入，包括 `exchange_rate.txt`。配置中的 hash 是**当前归档快照**的校验，不冒称训练当时的原始 hash。

`transfer_files.txt` 列出最小传输文件，包含 native 源码、必要数据、配置和证据。避免把几十万个历史 task 日志、Mac 虚拟环境、构建目录一并迁移。脚本和目录均保留项目相对路径。

在本地项目根目录执行传输时，替换为你自己的 SSH 别名与远程目录：

```bash
export CGB_HOST='你的SSH别名'
export CGB_REMOTE_ROOT='/data/crossgb_rebuttal'
rsync -a --files-from=rebuttal_server_config_20261025/transfer_files.txt ./ "$CGB_HOST:$CGB_REMOTE_ROOT/"
```

远程目录须先在你的 SSH 会话中创建并确认可写。这只是待你使用的命令，本轮没有建立 SSH 连接。材料不会自动发给评审或任何外部平台。

传输后在远程检查输入，不导入模型、不拟合：

```bash
python3 "$CGB_CFG/scripts/verify_inputs.py" --project-root "$CGB_PROJECT"
```

检查成功的含义是文件与配置完整；不是实验已经运行，也不是整包部署已经通过。

## 3. 原分组配置备份（复用队列转换前不直接运行）

| 顺序 | 文件 | 固定设置 | 规模与当前可执行性 |
|---|---|---|---|
| 1 | `configs/e0_replay.json` | Feature switch/dim10/seed1001/outer1；原骨干、原四端点 | 4 fits，无 HPO；**独立 replay driver 待实现** |
| 2 | `configs/e1_components.json` | dim10、3000 样本、三 seed、五折；Bias/SplitOnly/LeafOnly/Full | 最小三机制180 fits；推荐七机制420，XGB参照另105；**LeafOnly native flag/runner 待实现** |
| 并行 A | `configs/e5_timeseries.json` | seq24；原10任务；persistence＋Ridge 10 alpha | 130次面板fit、20行终测；**有现成参考入口，未在本轮运行** |
| 并行 B | `configs/e2_neighbors.json` | 五主回归、五外折、三内折、30候选/模型 | 四模型9100 fits；合成另1680；**新近邻面板 runner 待实现** |
| 随模型产出 | `configs/e3_failure_analysis.json` | CPU/Kin8nm先，再扩Energy/Houses；一次改变一个因素 | 主要复用前述输出；**模型导出/诊断适配待实现** |
| 3 | `configs/e4_compute_budget.json` | Kin8nm/CPU/Energy；60/300/900 CPU秒；两模型五折 | 搜索合计上限约7.5 CPU小时，最终refit另记；**CPU预算 runner 待实现** |
| 4，限时 | `configs/e6_capacity.json` | 三合成＋CPU/Kin8nm；k=1/2/3，截距对照另列 | 最多8小时开发、4小时运行；**容量probe待实现** |

下表与原配置文件记录的是旧全量协议，复用修订以页首新版规划为准。配置文件本身**不是已经接到旧 CLI 的开关**。不要把 `e1_components.json` 直接传给原 `run_ablation.py`；旧 runner 会另选骨干或运行16变体，不能因此得到所需四格。

可先推进 E0/E1 和 E5 两条线；E2 的近邻部署与数据核对可以同时做。E3 复用模型结果，不另开大搜索。E4 不要等全部 HPO 完成才开始。E6 不阻止前四项核心证据收口。

## 4. 原生源码和最小改动边界

历史 hash 匹配的起点是：

```text
aaai_FullCodeProject-check_codereview0701/project_paperwriting/vendor/xgboost-metagb/
```

其 updater SHA256：

```text
30081ac7e9e61e741add7a32f207f712ad16168b46b5dc3d91abb0b9d2589442
```

不要把 `beforereview` 当前副本或投稿 CrossGB 改名版直接当作历史运行时源码。

在服务器复制一份构建，保留原档案：

```bash
mkdir -p "$CGB_RUNS/native"
cp -a "$CGB_PROJECT/aaai_FullCodeProject-check_codereview0701/project_paperwriting/vendor/xgboost-metagb" "$CGB_RUNS/native/historical"
cmake -S "$CGB_RUNS/native/historical" -B "$CGB_RUNS/native/historical/build" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_OPENMP=ON -DGOOGLE_TEST=OFF
cmake --build "$CGB_RUNS/native/historical/build" --parallel 8
```

上述复制命令仅用于**目标目录尚不存在的首次部署**。已有目录先检查，不重复复制成嵌套目录，不覆盖旧结果。

指定并核对实际 Python 包和库路径：

```bash
export PYTHONPATH="$CGB_RUNS/native/historical/python-package"
python -c 'import xgboost; print(xgboost.__file__); print(xgboost.build_info())'
```

下一步先跑 E0 原四端点。只有通过后，才在另一份 `controlled` 源码中加入最小 LeafOnly 开关。内部使用 MetaGB 参数名即可，报告方法名仍为 CrossGB。

LeafOnly 需要新增 `metagb_split_only_use_bias`。下表是 p=1 新受控路径的真值表；历史兼容路径的 Bias 保持原 p=0，新增评分开关关闭，以免改变原路径。每组实际参数已分别写入配置：

| 组 | split评分仅bias | 最终叶仅bias |
|---|---:|---:|
| Bias | true | true |
| SplitOnly | false | true |
| LeafOnly | true | false |
| Full | false | false |

两个开关同时 true 是合法的 Bias 控制。必须在**每轮**更新梯度/预测叶；不是全部树训练完后换一次叶。必须处理 split pool 与 leaf pool 不同时的训练缓存。关闭新增开关时保留原行为；本轮不默认捆绑 child-max。

E1 配置同时描述两条互斥路径：

- **历史兼容路径**：保留原 p=0 Bias、−1e10 阈值，只增加缺失变体。
- **新受控后备路径**：若历史 replay 无法认证，四格全部重跑，用 p=1 原始候选池及正阈值；标注新协议，不能拼接原表结果。

原 C++ 的采样使用 C `srand(1)`，并非只受 sklearn `random_state` 控制。记录 Linux libc；不要为追求同分而先改随机数实现。

## 5. 依赖安装与现成时序入口

仅在服务器执行。先选已有 Python 3.13 或 3.12，再创建独立环境：

```bash
python3.13 -m venv "$CGB_RUNS/.venv"
source "$CGB_RUNS/.venv/bin/activate"
python -m pip install -r "$CGB_CFG/requirements-core.txt"
python "$CGB_CFG/scripts/verify_inputs.py" --project-root "$CGB_PROJECT" --packages
```

若没有 3.13，改为现有 3.12 并记录差异即可，不需要为本轮先重装整个系统。`requirements-dev.txt` 只用于后续测试/原 Parquet 入口；TS 参考实验不依赖 native XGBoost。

为了准确保留历史设置，时序拆成**原九任务 max_features=64** 与 **后期 Exchange h12 max_features=0** 两份配置。两份运行时仍会保存全部同口径结果。

先仅检查原九任务的输入和切分，不拟合：

```bash
python "$CGB_CFG/scripts/run_ts_reference_panel.py" --source-root "$CGB_CFG/source" --series-root "$CGB_PROJECT/aaai_FullCodeProject-check_codereview0701/paper_writing/timeseries_extra_random300" --registry "$CGB_CFG/inputs/ts_dataset_registry.json" --run-args-json "$CGB_CFG/inputs/ts_args_nine.json" --tasks-json "$CGB_CFG/inputs/ts_tasks_nine.json" --expected-data-hashes "$CGB_CFG/inputs/ts_expected_data_hashes.json" --original-split-manifest "$CGB_CFG/historical/ts_submitted_split_manifest.json" --output-dir "$CGB_RUNS/E5/nine_plan" --plan-only
```

再检查 h12：

```bash
python "$CGB_CFG/scripts/run_ts_reference_panel.py" --source-root "$CGB_CFG/source" --series-root "$CGB_PROJECT/aaai_FullCodeProject-check_codereview0701/paper_writing/timeseries_extra_random300" --registry "$CGB_CFG/inputs/ts_dataset_registry.json" --run-args-json "$CGB_CFG/inputs/ts_args_h12.json" --tasks-json "$CGB_CFG/inputs/ts_tasks_h12.json" --expected-data-hashes "$CGB_CFG/inputs/ts_expected_data_hashes.json" --original-split-manifest "$CGB_CFG/historical/ts_submitted_split_manifest.json" --output-dir "$CGB_RUNS/E5/h12_plan" --plan-only
```

两个 plan 都 ready 后，训练时：

1. 将各自输出目录换成未存在的 `nine_run` / `h12_run`。
2. 删除 `--plan-only`。
3. 增加 `--acknowledge-unverified-config`，准确标注为“根据归档语义重建的新参考实验”，不是原树结果已复现。

这个入口来自 Xia 的现有工具，原样保留。本轮只做静态检查，**没有在新服务器执行其 preflight 或训练**。它只支持 fixed split；rolling origins 和 selected-tree replay 另需实现。130 次面板 fit 包含117次九任务和13次h12；底层多输出 Ridge 约462次拟合，不能把这两种计数混用。

原 h12 已找到完成记录，无需采用旧草稿的撤回分支。十任务 selected 树参数保存在 `historical/ts_selected_models.json`，可用于后续无搜索重放。

## 6. 近邻、预算与容量的实现要求

**近邻面板**：先 XGB/CrossGB/LightGBM constant/LightGBM linear；限时接入原作者 GBDT-PL 或 LinXGBoost。每模型30候选，五外折三内折；超时改为12时须在看新test前对所有模型统一更改。线性叶的缩放器仅在当前训练集拟合，`linear_lambda` 单列搜索。新面板冻结 greedy aggregation，依据投稿附录 D.4 的已有观察，不能称原300/600搜索复现。

**预算面板**：只用预算内完成的候选，按 inner validation 选优，随后对去重 incumbent 做 outer test。完成不了检查点时记 NA；未完成候选消耗的 CPU 仍计成本。不能把已有验证曲线直接变成 test 曲线。

**容量面板**：固定路由逐轮refit k=1/2/3，截距条件一致；单变量 affine 另列。`leaf_features_vector` 是多输出字段，不能直接充当标量叶子的 k>1 实现。这个 probe 不等同于完整重选分裂的 k>1 CrossGB。

**停止规则**：近邻接入限半天；容量实现限八小时。不得因为结果不利而删除场景，不得因时间不够把未实现项称为已完成。

## 7. 结果保存与回传

每个运行使用独立 run_id，保存：

- 配置原文、源代码与实际 native 库 hash、版本/线程/硬件记录。
- 数据 hash、真实 split 索引或时间边界、种子、候选选择轨迹。
- 每 fold / origin 的目标与预测，最终指标、fit/search/predict CPU秒和 wall秒。
- 失败、超时、不完整单元；不只保存获胜的汇总行。

CSV、JSON、NPZ 即可形成最小可复核包。服务器产出后先汇总成紧凑表，再改写逐位 rebuttal；不自动上传 OpenReview。

## 8. 本包当前状态

已准备：8份分组/公共配置、14格原骨干、420行组件任务矩阵、100个近邻外折模型任务、十任务时序配置、40条原树selected记录、数据与源码校验清单、现成时序参考入口。

已验证范围：JSON/CSV可读取、关键计数与路径/hash、Python静态语法。未声称依赖安装成功、native构建成功、服务器环境可用或训练完成。

待实现项已写入配置中的 `status`：E0独立replay、E1开关与runner、E2面板、E3导出、E4预算、E6容量probe。这些状态是下一次在Linux上接续工作的清单。
