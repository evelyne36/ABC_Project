# 新增真实数据预实验（2026-10-27，扩充第1批）

**新增推荐：website_phishing（精度型补充）、physiochemical_protein（稳定XGB正向）、wine_quality数值版（弱观察项）。** 原先主攻的kin8nm、ILPD与vowel保留。新候选的强度和成本不同，不将它们统一称为近邻优势数据。

## 本次范围与协议

- 8个本地数据：wine、protein、Miami、QSAR fish、insurance、breast-w、maternal health risk、website phishing。
- 10个预处理设置（wine和insurance各含onehot/drop），5次探索性重划分×4方法，**200次成功拟合**。
- 沿用上一轮发现阶段冻结的每方法配置，不在本轮验证集重选参数；100 boosting rounds，训练最多4000行。
- seed42的outer1测试折继续保留、未评分。每次只重划分其余训练池；预处理在当前训练部分拟合。
- 这些划分的样本彼此重叠，而且候选由历史与此前预实验选择，不能当成5个独立数据集或无偏正式benchmark。
- 针对两个重复特征较多的分类数据，另做5次按特征组合分组的复核（40次成功拟合）；组间没有相同特征组合。该测试考察新组合泛化，和IID随机行划分是不同目标。

## 新增名单

表内RMSE/logloss越低越好；降幅按5次平均指标之比计算。逐划分平均改善、SD和配对胜负另存CSV，二者不能混用。

| 数据 | 类型/模式 | CrossGB | XGB | LGB linear | 对XGB降幅 | 对XGB较好次数 | 定位 |
|---|---|---:|---:|---:|---:|---:|---|
| **website_phishing** | 3类分类，onehot，logloss | **0.22078** | 0.22588 | 0.23669 | **2.26%** | **4/5** | 新增精度型补充 |
| **physiochemical_protein** | 回归，数值，RMSE | **4.52903** | 4.58836 | 4.52843 | **1.29%** | **5/5** | XGB稳定正向；与linear近似持平 |
| **wine_quality** | 回归，drop，RMSE | **0.68491** | 0.69145 | 0.68298 | **0.95%** | **5/5** | 仅观察；linear与constant都更好 |

### website_phishing：建议加入精度补充面板

随机行划分中，相对LGB linear的平均指标降幅约6.72%。原数据1353行，仅676个不同特征组合；因此补做分组测试：

| 按特征组合分组，logloss | CrossGB | XGB | LGB constant | LGB linear |
|---|---:|---:|---:|---:|
| 5次均值 | **0.29075** | 0.31019 | 0.32495 | 0.31891 |

相对XGB降低**6.27%**、相对LGB linear降低**8.83%**；分别4/5次更好。收益在这项更严格测试中保持，值得正式排跑。

**成本限制：** 随机行划分终拟合中位数约CrossGB 514ms、LGB linear 24.5ms（约21倍）；分组测试也有相近成本差距。它适合补精度/机制覆盖，不能作速度优势主例。未经PL及合理搜索预算比较，不宣称优于全部近邻。

### protein：适合补一个较大样本回归案例

原数据45730行，本轮每次训练4000行、验证9146行。CrossGB对XGB和LGB constant均5/5次更好，RMSE改善约1.3%/1.1%；与LGB linear差距仅约0.01%，没有明确近邻收益。

CrossGB终拟合中位数约106ms，约为XGB的7.6倍、LGB linear的4.7倍。因此是XGB选择性收益案例，仍需time-to-quality曲线，不能把同轮数结果写成效费优势。

### wine：只保留数值版观察项

drop版对XGB5/5次正向，但收益不到1%，且两个LGB版本都更好。保留全部类别信息的onehot版，CrossGB对XGB平均反而略差约0.74%；XGB onehot的平均RMSE也优于CrossGB drop。

不能只拿drop版宣称整个wine数据集上CrossGB更强。若时间紧，wine排在website和protein之后，不优先大搜索。

## 已降级的项目

- **maternal_health_risk**：随机行划分原本正向，但1014行只有416个不同特征组合。分组后CrossGB logloss0.81137、XGB0.72348、LGB linear0.78678；CrossGB对XGB **0/5胜、平均指标高约12.15%**。降级为划分敏感案例，不新增为主正例。
- **QSAR fish**：本轮CrossGB比XGB差约5.1%，历史微弱正向没有保持。
- **Miami housing**：相对XGB没有稳定收益，且更慢。
- **insurance**：onehot和drop均不支持对XGB的优势；drop丢掉重要类别信息后所有方法误差明显增大，不把这种低信息设置包装成优势。
- **breast-w**：AUC接近天花板，对XGB没有稳定增量。

## 当前建议的真实面板

1. **核心：kin8nm、ILPD。** 继续确认精度—时间工作点。
2. **精度扩展：vowel、website_phishing。** 保留强基线调参与重复结构复核，成本单报。
3. **XGB对照补充：physiochemical_protein。** 不宣称胜线性叶。
4. **观察：wine_quality/drop。** 当前不足以作为近邻优势主张。

## 文件与验证

- `added_candidates.csv`：新增名单及适用范围。
- `summary.csv`、`fits.jsonl`：全部10设置的结果，没有删除负项。
- `splits.csv`、`frozen_plan.json`：划分哈希、源文件哈希与冻结参数。
- `feature_group_check/`：两项重复特征诊断、全部40个拟合和分组结果。
- `verification.json`：200次拟合全部成功；分组复核另40次全部成功；正式outer-test评估0次。

运行环境沿用旧本地Darwin CrossGB与LGB4.6；计时为fit、不含预处理/HPO，不能直接推广到正式Linux环境。本轮没有新增合成数据或正式训练，也未改核心算法、提交或推送。
