# TabArena 51 数据集完整目录

> 本文档按任务类型（Binary / Multiclass / Regression）分类，每类内按数据集名称首字母（A-Z）排序，共 **30 Binary + 8 Multiclass + 13 Regression = 51 个数据集**。
>
> 各数据集附有 **样本量 (Instances)**、**特征数 (Features)**、**领域 (Domain)**、**来源 (Source)**、**发布年份 (Year)** 等信息，方便筛选大/小数据集用于模型训练。

---

## 目录

- [一、Binary Classification（二分类）— 30 个](#一binary-classification二分类--30-个)
- [二、Multiclass Classification（多分类）— 8 个](#二multiclass-classification多分类--8-个)
- [三、Regression（回归）— 13 个](#三regression回归--13-个)

---

## 一、Binary Classification（二分类）— 30 个

| # | 数据集名称 | 特征数 | 样本量 | 领域 | 来源 | 年份 | 许可证 | 大小分级 |
|---|-----------|:-----:|:-----:|------|:----:|:----:|--------|:-------:|
| 1 | Amazon_employee_access | 10 | 32,769 | 商业与营销 | Kaggle | 2010 | Public Domain | **大** |
| 2 | APSFailure | 171 | 76,000 | 工业与制造业 | UCI | 2016 | CC BY 4.0 | **大** |
| 3 | Bank_Customer_Churn | 11 | 10,000 | 金融 | Kaggle | 2020 | Public | **中** |
| 4 | bank-marketing | 14 | 45,211 | 金融 | UCI | 2012 | CC BY 4.0 | **大** |
| 5 | Bioresponse | 1,777 | 3,751 | 生物学与生命科学 | Kaggle | 2012 | Public Domain | **小** |
| 6 | blood-transfusion-service-center | 5 | 748 | 医学与医疗 | UCI | 2008 | CC BY 4.0 | **小** |
| 7 | churn | 20 | 5,000 | 商业与营销 | OpenML | 2005 | MIT License | **中** |
| 8 | coil2000_insurance_policies | 86 | 9,822 | 商业与营销 | UCI | 2000 | CC BY 4.0 | **中** |
| 9 | credit_card_clients_default | 24 | 30,000 | 金融 | UCI | 2009 | CC BY 4.0 | **大** |
| 10 | credit-g | 21 | 1,000 | 金融 | UCI | 1994 | CC BY 4.0 | **小** |
| 11 | customer_satisfaction_in_airline | 22 | 129,880 | 商业与营销 | Kaggle | 2023 | CC0: Public Domain | **大** |
| 12 | diabetes | 9 | 768 | 医学与医疗 | OpenML | 1988 | CC0: Public Domain | **小** |
| 13 | Diabetes130US | 48 | 71,518 | 医学与医疗 | UCI | 2014 | CC BY 4.0 | **大** |
| 14 | E-CommereShippingData | 11 | 10,999 | 商业与营销 | Kaggle | 2021 | Public Domain | **中** |
| 15 | Fitness_Club | 7 | 1,500 | 商业与营销 | Kaggle | 2023 | Public Domain | **小** |
| 16 | GiveMeSomeCredit | 11 | 150,000 | 金融 | Kaggle | 2011 | Public | **大** |
| 17 | hazelnut-spread-contaminant-detection | 31 | 2,400 | 生物学与生命科学 | OpenML | 2020 | CC BY-SA | **小** |
| 18 | heloc | 24 | 10,459 | 金融 | Kaggle | 2021 | Public | **中** |
| 19 | HR_Analytics_Job_Change_of_Data_Scientists | 13 | 19,158 | 商业与营销 | Kaggle | 2021 | CC0: Public Domain | **中** |
| 20 | in_vehicle_coupon_recommendation | 25 | 12,684 | 商业与营销 | UCI | 2017 | CC BY 4.0 | **中** |
| 21 | Is-this-a-good-customer | 14 | 1,723 | 商业与营销 | Kaggle | 2020 | CC0: Public Domain | **小** |
| 22 | jm1 | 22 | 10,885 | 技术与互联网 | OpenML | 2004 | Public Domain | **中** |
| 23 | kddcup09_appetency | 213 | 50,000 | 商业与营销 | Other (KDD Cup) | 2008 | Public | **大** |
| 24 | Marketing_Campaign | 26 | 2,240 | 商业与营销 | Kaggle | 2020 | Public | **小** |
| 25 | NATICUSdroid | 87 | 7,491 | 技术与互联网 | UCI | 2021 | CC BY 4.0 | **中** |
| 26 | online_shoppers_intention | 18 | 12,330 | 商业与营销 | UCI | 2017 | CC BY 4.0 | **中** |
| 27 | polish_companies_bankruptcy | 65 | 5,910 | 金融 | UCI | 2010 | CC BY 4.0 | **中** |
| 28 | qsar-biodeg | 42 | 1,054 | 生物学与生命科学 | UCI | 2013 | CC BY 4.0 | **小** |
| 29 | seismic-bumps | 16 | 2,584 | 环境科学与气候 | UCI | 2013 | CC BY 4.0 | **小** |
| 30 | taiwanese_bankruptcy_prediction | 95 | 6,819 | 金融 | UCI | 2009 | CC BY 4.0 | **中** |

### Binary 数据集统计摘要

| 指标 | 数值 |
|------|:----:|
| 样本量范围 | 748 ~ 150,000 |
| 样本量中位数 | ~9,060 |
| 特征数范围 | 5 ~ 1,777 |
| 特征数中位数 | ~21 |
| **小数据集 (≤5,000 样本)** | 9 个 |
| **中数据集 (5,001~20,000 样本)** | 11 个 |
| **大数据集 (>20,000 样本)** | 10 个 |

---

## 二、Multiclass Classification（多分类）— 8 个

| # | 数据集名称 | 特征数 | 样本量 | 类别数 | 领域 | 来源 | 年份 | 许可证 | 大小分级 |
|---|-----------|:-----:|:-----:|:-----:|------|:----:|:----:|--------|:-------:|
| 1 | anneal | 39 | 898 | 5 | 化学与材料科学 | UCI | 1990 | CC BY 4.0 | **小** |
| 2 | hiva_agnostic | 1,618 | 3,845 | 3 | 化学与材料科学 | Other | 2007 | Public | **小** |
| 3 | maternal_health_risk | 7 | 1,014 | 3 | 医学与医疗 | UCI | 2020 | CC BY 4.0 | **小** |
| 4 | MIC | 112 | 1,699 | 8 | 医学与医疗 | UCI | 2020 | CC BY 4.0 | **小** |
| 5 | SDSS17 | 12 | 78,053 | 3 | 物理学与天文学 | Kaggle | 2022 | Public Domain | **大** |
| 6 | splice | 61 | 3,190 | 3 | 生物学与生命科学 | UCI | 1991 | CC BY 4.0 | **中** |
| 7 | students_dropout_and_academic_success | 37 | 4,424 | 3 | 教育 | UCI | 2021 | CC BY 4.0 | **小** |
| 8 | website_phishing | 10 | 1,353 | 3 | 技术与互联网 | UCI | 2014 | CC BY 4.0 | **小** |

### Multiclass 数据集统计摘要

| 指标 | 数值 |
|------|:----:|
| 样本量范围 | 898 ~ 78,053 |
| 样本量中位数 | ~2,445 |
| 特征数范围 | 7 ~ 1,618 |
| 特征数中位数 | ~38 |
| 类别数 | 3 ~ 8 |
| **小数据集 (≤5,000 样本)** | 6 个 |
| **中数据集 (5,001~20,000 样本)** | 1 个 |
| **大数据集 (>20,000 样本)** | 1 个 |

---

## 三、Regression（回归）— 13 个

| # | 数据集名称 | 特征数 | 样本量 | 领域 | 来源 | 年份 | 许可证 | 大小分级 |
|---|-----------|:-----:|:-----:|------|:----:|:----:|--------|:-------:|
| 1 | airfoil_self_noise | 6 | 1,503 | 物理学与天文学 | UCI | 2014 | CC BY 4.0 | **小** |
| 2 | Another-Dataset-on-used-Fiat-500 | 8 | 1,538 | 技术与互联网 | Kaggle | 2020 | CC0: Public Domain | **小** |
| 3 | concrete_compressive_strength | 9 | 1,030 | 化学与材料科学 | UCI | 1998 | CC BY 4.0 | **小** |
| 4 | diamonds | 10 | 53,940 | 商业与营销 | Other (ggplot2) | 2015 | MIT License | **大** |
| 5 | Food_Delivery_Time | 10 | 45,451 | 商业与营销 | Kaggle | 2023 | DbCL v1.0 | **大** |
| 6 | healthcare_insurance_expenses | 7 | 1,338 | 医学与医疗 | Kaggle | 2023 | DbCL v1.0 | **小** |
| 7 | houses | 9 | 20,640 | 商业与营销 | Other (CMU StatLib) | 1990 | Public | **大** |
| 8 | miami_housing | 16 | 13,776 | 商业与营销 | OpenML | 2016 | CC BY-NC-SA | **中** |
| 9 | physiochemical_protein | 10 | 45,730 | 化学与材料科学 | UCI | 2013 | CC BY 4.0 | **大** |
| 10 | QSAR_fish_toxicity | 7 | 907 | 生物学与生命科学 | UCI | 2015 | CC BY 4.0 | **小** |
| 11 | QSAR-TID-11 | 1,025 | 5,742 | 化学与材料科学 | OpenML | 2015 | Public | **中** |
| 12 | superconductivity | 82 | 21,263 | 物理学与天文学 | UCI | 2018 | CC BY 4.0 | **大** |
| 13 | wine_quality | 13 | 6,497 | 化学与材料科学 | UCI | 2009 | CC BY 4.0 | **中** |

### Regression 数据集统计摘要

| 指标 | 数值 |
|------|:----:|
| 样本量范围 | 907 ~ 53,940 |
| 样本量中位数 | ~6,497 |
| 特征数范围 | 6 ~ 1,025 |
| 特征数中位数 | ~10 |
| **小数据集 (≤5,000 样本)** | 6 个 |
| **中数据集 (5,001~20,000 样本)** | 3 个 |
| **大数据集 (>20,000 样本)** | 4 个 |

---

## 附录：全数据集横向对比

### 各任务类型样本量分布

```
Binary (30 个):
  小 (≤5K):    9 个  ████████████░░░░░░░░░░  30%
  中 (5K~20K): 11 个 ████████████████░░░░░░  37%
  大 (>20K):   10 个 ██████████████░░░░░░░░  33%

Multiclass (8 个):
  小 (≤5K):    6 个  ████████████████████░░  75%
  中 (5K~20K): 1 个  ████░░░░░░░░░░░░░░░░░░  12%
  大 (>20K):   1 个  ████░░░░░░░░░░░░░░░░░░  12%

Regression (13 个):
  小 (≤5K):    6 个  ██████████████░░░░░░░░  46%
  中 (5K~20K): 3 个  ███████░░░░░░░░░░░░░░░  23%
  大 (>20K):   4 个  ██████████░░░░░░░░░░░░  31%
```

### 全体 51 个样本量分段排序（小→大）

| 大小分级 | 样本量范围 | 数据集数 | 数据集列表 |
|:--------:|:----------:|:--------:|-----------|
| **小** | ≤ 1,000 | 6 | blood-transfusion-service-center, concrete_compressive_strength, credit-g, diabetes, QSAR_fish_toxicity, anneal |
| **小** | 1,001 ~ 2,000 | 7 | airfoil_self_noise, Another-Dataset-on-used-Fiat-500, Fitness_Club, Is-this-a-good-customer, maternal_health_risk, MIC, website_phishing |
| **小** | 2,001 ~ 5,000 | 8 | Bioresponse, hazelnut-spread-contaminant-detection, hiva_agnostic, Marketing_Campaign, qsar-biodeg, seismic-bumps, splice, students_dropout_and_academic_success |
| **中** | 5,001 ~ 10,000 | 7 | churn, coil2000_insurance_policies, NATICUSdroid, polish_companies_bankruptcy, QSAR-TID-11, taiwanese_bankruptcy_prediction, wine_quality |
| **中** | 10,001 ~ 20,000 | 7 | Bank_Customer_Churn, E-CommereShippingData, heloc, HR_Analytics_Job_Change_of_Data_Scientists, in_vehicle_coupon_recommendation, jm1, online_shoppers_intention, miami_housing |
| **大** | 20,001 ~ 50,000 | 8 | bank-marketing, credit_card_clients_default, houses, kddcup09_appetency, physiochemical_protein, superconductivity, Food_Delivery_Time, diamonds |
| **大** | 50,001 ~ 100,000 | 4 | Amazon_employee_access, APSFailure, Diabetes130US, SDSS17 |
| **大** | > 100,000 | 3 | customer_satisfaction_in_airline, GiveMeSomeCredit |

> **说明**：
> - **小数据集**（≤5,000 样本）：共 **21 个**，适合快速实验、调试与初步验证。
> - **中数据集**（5,001 ~ 20,000 样本）：共 **14 个**，平衡训练速度与代表性。
> - **大数据集**（>20,000 样本）：共 **16 个**，需要更多算力与训练时间，作为最终性能评估。
>
> 建议训练时从小数据集开始快速调参，逐步过渡到中/大数据集以验证泛化能力，避免大数据集未收敛而小数据集迟迟未开始计算的情况。

---

*生成日期：2025-07*
*数据来源：TabArena OpenML Suite (ID 457) 及 dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv*
