# 复原材料清单（换另一台机器复原用）

> **一句话**：`git clone` 得到全部代码与配置（35 MB），再按下面 §2 拷 **6 条主路径 + 2 处小件（72 文件 / 约 43.5 MB）** 就能跑 E / N 系列实验。数据集本机都有，本文只说"用哪些、放哪、谁在读它"。

| 层 | 内容 | 体积 |
|---|---|---|
| 仓库 | 代码 / 脚本 / 配置 / 文档 / 汇报 / 送审 PDF / SupplementaryCode | **35.14 MB**（4942 文件） |
| §2 必拷本地材料（A） | 原始数据集 + 冻结协议 + 合成 ablation 的 npz + baseline npz | **≈ 43.5 MB**（72 文件） |
| §3 按需（B） | 重造输入的 json shards、改版 xgboost 源码、baseline 三方源码 | ≈ 1.1 GB |
| §4 不要拷（C） | venv / 编译产物 / 历史采集 / 打包件 / 嵌套 .git | ≈ 6 GB |

## 0. 复原三步

```bash
# 1) 取仓库
git clone https://github.com/AncD8972/aaai27-rebuttal.git crossgbRebuttalV1-261024
cd crossgbRebuttalV1-261024 && git checkout v26102714      # 默认分支也是它

# 2) 从旧机拷本地材料（见 §2，保持相同的项目内相对路径）
OLD=/Volumes/old/crossgbRebuttalV1-261024
for p in \
  "aaai_FullCodeProject-check_codereview0701/tabular_extra_random300/data" \
  "aaai_FullCodeProject-check_codereview0701/paper_writing/timeseries_extra_random300/data" \
  "aaai_FullCodeProject-check_codereview0701/dataset/TabArena_regression/houses.csv" \
  "rebuttal_server_config_20261025/historical" \
  "rebuttal_formal_20261026/inputs" \
  "rebuttal_baselines_20261025/inputs" ; do
  mkdir -p "$(dirname "$p")" && rsync -a "$OLD/$p" "$(dirname "$p")/"
done

# 3) 校验 + 建环境
python3 rebuttal_server_config_20261025/scripts/verify_inputs.py --project-root "$PWD"
python3 -m venv .venv && .venv/bin/pip install -r rebuttal_server_config_20261025/requirements-core.txt
```

## 1. 仓库自带，无需拷（`git ls-files` 可见）

关键几个：

- `rebuttal_server_config_20261025/inputs/dataset_registry.json` —— **数据集登记表**：每个数据集的相对路径 + sha256 + 目标列 + 是否主面板。想知道"用哪些"，先看这个。
- `rebuttal_server_config_20261025/reports/input_transfer_manifest.json` —— **传输清单**：1721 个输入文件的全量 sha256（51.38 MB）。仓库已含其中 **110** 个，剩余 **1611** 个（48.22 MB）就是 §2/§3 要补的。
- `rebuttal_execution_20261025/frozen/input_manifest.json` —— 14 个冻结输入的 sha256（含两份送审 PDF）。
- `rebuttal_formal_20261026/configs/` —— E/N 系列全部队列：`cgb_tasks/` 107 个任务、`N1_outer_jobs.csv`、`real_candidate_pools.json`(2.4 MB)、`N4_params/`、`N5_cgb_queue.json`。
- `rebuttal_formal_20261026/scripts/` —— 含 `prepare_cgb_ablation.py`（从 §3 的 shards 重造 §2 的 npz）、`run_cgb_task.py`、`verify_handoff_inputs.py`。
- `aaai2027提交原件/SupplementaryCode_CrossGBv1_github_openreview/`（16 MB）—— 自洽的提交版实现，含 `CrossGB/xgboost-master` fork 源码，可脱离本机直接编译。

## 2. 必须拷的本地材料（A 类，≈ 43.5 MB）

| # | 路径（相对项目根） | 体积 | 文件 | 用途 | 谁在读 |
|---|---|---|---|---|---|
| A1 | `aaai_FullCodeProject-check_codereview0701/tabular_extra_random300/data/` | 1.91 MB | 7 | 主面板 8 个表格数据集：`regression__kin8nm.csv`、`regression__Moneyball.csv`、`regression__cpu_activity.csv`、`regression__energy_efficiency.csv`、`binary__ilpd.csv`、`binary__phoneme.csv`、`binary__credit-approval.csv` | `inputs/dataset_registry.json`（tabular，`main_panel` 标了 5 个） |
| A2 | `aaai_FullCodeProject-check_codereview0701/dataset/TabArena_regression/houses.csv` | 1.36 MB | 1 | TabArena 回归主面板数据集 `houses` | 同上（registry 里显式指向此路径） |
| A3 | `aaai_FullCodeProject-check_codereview0701/paper_writing/timeseries_extra_random300/data/raw/` | 24.47 MB | 6 | E5 时序 6 数据集：`ETTh1/ETTh2/ETTm1/ETTm2.csv`、`exchange_rate.txt`、`national_illness.csv`(ILI) | `inputs/ts_dataset_registry.json`、`scripts/run_ts_reference_panel.py` |
| A4 | `rebuttal_server_config_20261025/historical/` | 8.96 MB | 8 | **冻结协议**：`ablation_protocol.json`、`ablation_synthetic_backbones.json`、`ablation_split_manifest.json`、`ablation_execution_manifest.json`、`ablation_variant_manifest.csv`、`ts_selected_models.json`、`ts_submitted_split_manifest.json`、`synthetic__feature_switch__dim10__seed1001__out1.json`(8.5 MB) | `scripts/prepare_cgb_ablation.py` 直接读；缺了就无法重造 N 系列输入 |
| A5 | `rebuttal_formal_20261026/inputs/cgb_ablation/` | 2.9 MB | 21 | N 系列 21 个 `data.npz`（7 场景 × 3 seed，dim10 合成数据） | 每个 `configs/cgb_tasks/*.json` 的 `data_npz` 字段（相对 `rebuttal_formal_20261026/`） |
| A6 | `rebuttal_baselines_20261025/inputs/` | 3.2 MB | 18 | 9 个 baseline 数据集目录（各一 `data.npz` + `manifest.json`）：真实 5 个 `cpu_activity`/`energy_efficiency`/`houses`/`kin8nm`/`moneyball`，合成 4 个 `regime_slope`/`feature_switch`/`curved_control`/`sparse_multiplicative` | `scripts/prepare_protocol.py`、`run_nested*.py` |
| A7 | 见下「小件」 | 0.69 MB | 11 | `datacollection0702/0716dataAddition/metagb_ablation_syn/syn_formal_formal16_v1/{synthetic_backbones,protocol,split_manifest}.json`（328 KB）与 `…/0716dataAddition/ts0716/未命名/`（8 文件 376 KB） | `frozen/input_manifest.json` 逐个 pin 了 sha256 |

> A5 已被 `.gitignore` 的 `*.npz` 规则挡住、A4 被 `rebuttal_server_config_20261025/historical/` 规则挡住——**它们其实是输入不是产物**，所以务必照拷。若希望以后新机器零手工，可对这几条 `git add -f` 后单独补一次提交（A4+A5+A6 合计仅 15 MB）。

## 3. 按需才拷（B 类）

| # | 路径 | 体积 | 什么时候需要 |
|---|---|---|---|
| B1 | `…/datacollection0702/0716dataAddition/metagb_ablation_syn/syn_formal_formal16_v1/` 下 105 个 `synthetic__*__dim10__seed*__out*.json` | ≈ 0.88 GB | 只有要**重造** A5（`prepare_cgb_ablation.py` 从这些 shard 反推 split 与 backbone）或对 A5 做全链 sha256 溯源时才需要。已有 A5 的 npz 可直接跳过 |
| B2 | `aaai_FullCodeProject-check_codereview0701/project_paperwriting/vendor/xgboost-metagb/` | 11.49 MB / 1586 文件 | 要**编译**改版 xgboost（native 后端）时。与仓库内 `aaai2027提交原件/SupplementaryCode_CrossGBv1_github_openreview/CrossGB/xgboost-master` 同源，二者取一即可 |
| B3 | `rebuttal_baselines_20261025/{vendor,downloads}/` | 149 MB + 57 MB | 要重跑 baseline（gbdt_pl / linxgboost / pilot / tao）且不想联网拉源码时 |

## 4. 不要拷（C 类，可重建或纯历史）

| 路径 | 体积 | 说明 |
|---|---|---|
| `rebuttal_baselines_20261025/.venv/`、`rebuttal_execution_20261025/.venv/` | 167 MB | 重建即可（见 §5） |
| `rebuttal_execution_20261025/native/` | 91 MB | `controlled/`、`historical/` 编译产物，从 B2 源码重编 |
| `rebuttal_baselines_20261025/build/` | 2.5 MB | CMake 中间物 |
| `*.tar.gz`（`formal_linux_bundle` 18 MB、`baseline_linux_bundle` 2.7 MB） | 20.7 MB | 只是给远端机打的包，内容 = 仓库 + A 类 |
| `aaai_FullCodeProject-check_codereview0701/.git_nested_backup/` | 306 MB | `ABC_Project` 克隆的历史，见 README 恢复命令 |
| `aaai_FullCodeProject-check_codereview0701/beforereview/` | 176 MB | 旧评审版工程（含自带 `xgboost-master` 171 MB） |
| `aaai_FullCodeProject-check_codereview0701/project_paperwriting/{datasets,artifacts,experiments}/` | 599 MB | 写作期快照，不参与 E/N 实验 |
| `aaai_FullCodeProject-check_codereview0701/datacollection*/`（除 §3 B1 的 shards） | ≈ 4.3 GB | 历史采集结果库，正式实验只引用其中 A5/A7 与 B1 |
| `**/__pycache__/`、`.pytest_cache/`、`.DS_Store` | — | 垃圾 |

## 5. 环境口径

| 项 | 值 | 来源 |
|---|---|---|
| Python（baselines） | **3.13.9** | `rebuttal_baselines_20261025/.venv/bin/python -V` |
| Python（execution） | **3.12.14** | `rebuttal_execution_20261025/.venv/bin/python -V` |
| 核心依赖 | `numpy==2.3.5`、`pandas==2.3.3`、`scipy==1.16.3`、`scikit-learn==1.7.2`、`lightgbm==4.6.0` | `rebuttal_server_config_20261025/requirements-core.txt` |
| xgboost / MetaGB | **从冻结 native 源码编译，不走 PyPI** | 同上文件首行注释 |

## 6. 校验口径（拷完必须过）

本机（Darwin arm64 / Python 3.9.6）已跑通第一条，输出要点：

```json
{ "status": "inputs_and_configs_valid", "manifest_files_verified": 1721,
  "planned_e1_rows": 420, "planned_e2_outer_model_jobs": 100, "planned_e5_tasks": 10,
  "errors": [], "warnings": ["Static audit only: this host is not the target Linux server"] }
```

```bash
# 全量输入 sha256 对账：1721 条，缺/改一个都会报出来
python3 rebuttal_server_config_20261025/scripts/verify_inputs.py --project-root "$PWD"

# 冻结输入（14 条，含两份送审 PDF 与 A7 小件）
python3 - <<'PY'
import json, hashlib, pathlib
m = json.load(open("rebuttal_execution_20261025/frozen/input_manifest.json"))["source_files"]
bad = [p for p, h in m.items() if not pathlib.Path(p).exists()
       or hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() != h]
print("输入校验：", "全部通过" if not bad else f"{len(bad)} 个不符 -> {bad}")
PY

# 交接自检（formal 层）
python3 rebuttal_formal_20261026/scripts/verify_handoff_inputs.py --help
```

## 7. 一句话备忘

- **想知道"用哪些数据集"**：读 `rebuttal_server_config_20261025/inputs/dataset_registry.json`（表格 8 个 + 时序 6 个，含 sha256 与主面板标记），再读 `reports/input_transfer_manifest.json`（1721 个输入文件）。
- **想重造而不是拷**：`prepare_cgb_ablation.py`（N 系列 npz ← B1 shards + A4 协议）、`rebuttal_baselines_20261025/scripts/prepare_protocol.py`（A6 npz ← A1/A2 原始 CSV）。
- **路径约定**：所有配置内的路径都是 `path_base = project_root` 的**相对路径**，所以新机器上项目目录叫什么都可以，但 §2 的目录层级不能变。
