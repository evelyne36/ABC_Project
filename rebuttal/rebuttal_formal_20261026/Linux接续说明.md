# Linux接续说明

本文件使用项目根目录作为当前目录。用户已授权五个SSH别名：`cpu4_pool.zjuici.com`、`cpu2_pool.zjuici.com`、`cpu1_pool.zjuici.com`、`cpu_Calculator`、`TabularData_Processor`。每台已核实16核、约32GB；独立工作目录为`/home/admin/crossgb_rebuttal_20261026`。新增正式训练在Linux运行，本机仅准备与核查。目前首次环境核查成功后，部署连接被网关重置，cpu4/cpu2进一步诊断为SSH握手前关闭；正式训练未启动。

只需传递本轮`rebuttal_formal_20261026/formal_linux_bundle.tar.gz`及同名SHA256校验文件；外包已经包含`rebuttal_baselines_20261025/baseline_linux_bundle.tar.gz`。先在新的工作目录解外包，再解内层baseline包，保持目录名，不覆盖未知实验目录。前者提供本轮队列、CGB源码、时序输入和原始研究材料，内层提供四baseline作者源码/冻结输入。运行`python3 rebuttal_formal_20261026/scripts/verify_handoff_inputs.py`核对解包内容；这不代表Linux构建及运行通过。跨电脑交接优先读《交给另一台电脑AI_执行交接书.md》。

五台的`/home/admin/anaconda3/bin/python`均为3.13.9，且numpy/scipy/pandas/sklearn/numba等版本与本机适配环境一致；优先由它创建任务内`--system-site-packages`环境，避免修改已有环境。不要误用系统Python3.10，也不要默认使用版本不一致的旧`mgb`环境。CGB需要CMake≥3.18、GCC≥8.1和OpenMP；GBDT-PL还需BLAS/LAPACK开发库。以下`python`指任务实验环境。

## 第一批：硬件/依赖与可行性

先记录CPU分配、内存、Python/编译器版本及可写路径。所有模型单线程；先4并发，按实测峰值内存调节。原单台32核/64GB规划已由五台16核/32GB配置替代，每台分别限制并发。

```bash
python rebuttal_baselines_20261025/scripts/build_gbdt_pl.py
python rebuttal_baselines_20261025/scripts/smoke_all.py
python rebuttal_formal_20261026/scripts/baseline_runtime_pilot.py --output rebuttal_formal_20261026/results/N1_runtime_pilot --workers 4 --timeout 120 --execute
```

此16-fit测速使用CPU/Regime、固定候选索引及outer1-inner1输入，不访问outer test。查看`report.json`的失败、超时和峰值内存后，在正式测试前写定可完成的方法/面板。不要直接启动四方法全部120,100次拟合。

## 第二批：R0与LeafOnly

```bash
bash rebuttal_formal_20261026/scripts/build_controlled_linux.sh
python rebuttal_formal_20261026/scripts/check_native_leafonly.py --xgboost-python-package rebuttal_execution_20261025/native/controlled/python-package --output rebuttal_formal_20261026/results/native_check
python rebuttal_formal_20261026/scripts/run_cgb_queue.py --queue rebuttal_formal_20261026/configs/R0_E0_queue.json --output rebuttal_formal_20261026/results/R0_E0 --xgboost-python-package rebuttal_execution_20261025/native/controlled/python-package --workers 4 --execute
python rebuttal_formal_20261026/scripts/collect_cgb_gate.py --results rebuttal_formal_20261026/results/R0_E0 --native-check rebuttal_formal_20261026/results/native_check/report.json --output rebuttal_formal_20261026/results/R0_gate.json
```

只有`R0_gate.json`的`passed=true`才可接着运行：

```bash
python rebuttal_formal_20261026/scripts/run_cgb_queue.py --queue rebuttal_formal_20261026/configs/N2_leafonly_queue.json --output rebuttal_formal_20261026/results/N2_leafonly --xgboost-python-package rebuttal_execution_20261025/native/controlled/python-package --gate rebuttal_formal_20261026/results/R0_gate.json --workers 4 --scenarios regime_slope feature_switch curved_control --execute
python rebuttal_formal_20261026/scripts/run_cgb_queue.py --queue rebuttal_formal_20261026/configs/N2_leafonly_queue.json --output rebuttal_formal_20261026/results/N2_leafonly --xgboost-python-package rebuttal_execution_20261025/native/controlled/python-package --gate rebuttal_formal_20261026/results/R0_gate.json --workers 4 --execute --resume
```

首条45次，第二条补齐七机制105次，已成功且身份匹配的45次不重跑。每fit独立进程，保存模型/预测/参数/库hash。失败目录保留供定位；不自动删结果。源码、库、输入或构造助手改变时不得沿用旧gate或缓存。

门控要求28个端点全部在预定数值容差内匹配历史预测；失败时先定位。不能为了获得通过任意放宽容差，也不能直接拼历史Full与不兼容LeafOnly。

## 第三批：时序参考（可与前述小并发独立推进）

TS执行器已补齐原始`random600`辅助依赖。本机对九任务＋后期Exchange h12共十任务做过plan-only并通过；Linux仍需重新执行预检查。正式执行比下列命令多加`--acknowledge-unverified-config`，去掉`--plan-only`，并换一个全新输出目录。这一标志承认配置是从档案语义重建，不把它冒称原训练时刻认证快照。

```bash
python rebuttal_server_config_20261025/scripts/run_ts_reference_panel.py --source-root rebuttal_server_config_20261025/source --series-root aaai_FullCodeProject-check_codereview0701/paper_writing/timeseries_extra_random300 --run-args-json rebuttal_server_config_20261025/inputs/ts_args_nine.json --tasks-json rebuttal_server_config_20261025/inputs/ts_tasks_nine.json --original-split-manifest rebuttal_server_config_20261025/historical/ts_submitted_split_manifest.json --expected-data-hashes rebuttal_server_config_20261025/inputs/ts_expected_data_hashes.json --output-dir rebuttal_formal_20261026/results/TS_nine_linux_preflight --plan-only
python rebuttal_server_config_20261025/scripts/run_ts_reference_panel.py --source-root rebuttal_server_config_20261025/source --series-root aaai_FullCodeProject-check_codereview0701/paper_writing/timeseries_extra_random300 --run-args-json rebuttal_server_config_20261025/inputs/ts_args_h12.json --tasks-json rebuttal_server_config_20261025/inputs/ts_tasks_h12.json --original-split-manifest rebuttal_server_config_20261025/historical/ts_submitted_split_manifest.json --expected-data-hashes rebuttal_server_config_20261025/inputs/ts_expected_data_hashes.json --output-dir rebuttal_formal_20261026/results/TS_h12_linux_preflight --plan-only
```

运行前将OMP/BLAS等线程设为1；这是避免意外超订阅的资源配置，与原输入/任务语义无关。结果不自动合成“跨时间起点稳定”的结论。

## 正式近邻面板入口

在测速后冻结覆盖的前提下，以下示例完成**一个方法×一个数据**的300候选、五外折完整面板，约6,005次fit。`--workers`是这个进程内部的全局拟合并发，不能给多个面板各开32并发。

```bash
python rebuttal_baselines_20261025/scripts/run_nested.py --method gbdt_pl --data rebuttal_baselines_20261025/inputs/regime_slope --candidate-file rebuttal_baselines_20261025/configs/gbdt_pl_300.json --output rebuttal_formal_20261026/results/N1/gbdt_pl/regime_slope --workers 8 --fit-timeout 600 --execute --label formal_rebuttal_20261026
```

其余预定项在`configs/N1_outer_jobs.csv`。它是覆盖矩阵，不是让100个outer任务各自覆盖同一输出目录的并行脚本；若逐外折运行，沿用`run_nested.py --outer-fold`/`--resume`的既有约定，同一方法/数据的目录不要被两个进程同时写。

## N3/N4入口与仍待验证部分

以下代码已准备，但尚未在Linux执行验证。N3明确是**相同搜索时间上限**，最终refit另计并报告，不宣称总训练成本完全相同。该机预留专用CPU，不与其他训练争用。

```bash
python rebuttal_formal_20261026/scripts/run_selected_real.py --xgboost-python-package rebuttal_execution_20261025/native/controlled/python-package --output rebuttal_formal_20261026/results/N4_selected --gate rebuttal_formal_20261026/results/R0_gate.json --workers 4 --execute
python rebuttal_formal_20261026/scripts/run_equal_search_budget.py --xgboost-python-package rebuttal_execution_20261025/native/controlled/python-package --output rebuttal_formal_20261026/results/N3_equal_search --gate rebuttal_formal_20261026/results/R0_gate.json --workers 4 --execute
```

跨主机接续须复用完全相同的native动态库并检查源/库hash及运行兼容性；不能把cpu4的gate套在另一份不同编译产物上。

N5固定路由容量probe、多起点适配仍未实现。所有已提供执行器都须Linux实际运行验证；不因存在命令就视为通过验证。默认不启动dim20、全九数据×四近邻等扩展队列。
