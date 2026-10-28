# 四种近邻baseline适配记录

授权：用户2026-10-25明确要求本机适配LinXGBoost、PILOT、GBDT-PL、TAO。此前不在本地调试的限制已由本次授权更新。目标是可迁移的训练/预测入口和真实小样例核验，不启动正式大面板或CGB训练。

边界：原投稿、历史数据、原工程不修改；上游代码固定版本；兼容补丁、方法语义、实际测试和限制可追溯。TAO须先确认官方包支持线性叶。PILOT/TAO按实际单树或集成身份标记，不自动改造成新算法。

步骤与验收：
1. 下载三个固定commit和TAO官方包，安全解包并记录SHA256。
2. 检查官方训练/预测API；编写适配契约与测试，再实现最小接口。
3. 必要的依赖和native构建仅写入本目录，不修改系统环境或全局库。
4. 每方法真实训练、有限预测、重复调用/隔离性与上游一致性核验。记录真实运行与尚未核验的平台。
5. 提供小样例、单任务入口、原协议接入方式和Linux命令。正式大面板仍另行启动。

当前：Conda base已有Python3.13、numpy2.3.5、scipy1.16.3、sklearn1.7.2、numba0.62.1、pytest8.4.2；本机arm64，Homebrew有CMake/libomp。curl默认网络DNS失败，需申请下载公开源码的网络权限。


## 完成记录

- 三仓库固定commit＋TAO官方包已下载并记录archive/code hash。源码保留，未更改原工程。
- 用户确认TAO只跑官方常数叶；没有增加自实现线性叶。
- 四个回归adapter已通过独立进程小合成和Moneyball原outer1训练/预测。Moneyball按原onehot为42维。
- GBDT-PL运行故障定位：两个OpenMP runtime混用导致读取崩溃；改为同一Conda libomp后解决。另上游verbose=0不构造metric却在结束时访问，采用有效verbose=1。接口缺return补齐；native线程硬编码统一为单worker，原split/leaf公式不改。兼容层手算fixture通过。
- 真实数据使用原loader/split函数；四合成机制全部20份outer及其inner hash与归档匹配。已冻结9份输入，不需要大规模CGB重跑。
- 单fit与nested CV入口可运行；默认nested仅规划，明确--execute才训练，支持身份校验续跑、逐fit进程隔离、超时失败记录。候选任一内折失败即不参与选优。
- 300/600候选配置已生成并验证前缀；作者起始配置和小ensemble均覆盖。未执行正式搜索。
- 完整16项测试通过；TAO有7个上游NumPy弃用提示。Linux尚未实测，需重建native并做14项运行检查；其余两项输入再生检查依赖原工程。
- 最终复核为单代理自审：核对上游语义、数据/选参隔离、cache身份、native来源、进程资源与版本记录。未宣称独立评审或原论文数值复现完成。

详见reports/setup_status.json及README。CGB训练0，正式面板0。
