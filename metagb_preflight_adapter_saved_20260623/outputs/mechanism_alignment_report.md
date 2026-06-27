# MetaGB Mechanism-Alignment Preflight Report

## Scoring Contract

- The score is intrinsic: it is computed from dataset/mechanism proxy features.
- Full MetaGB-vs-XGBoost outcomes are not used to compute the score.
- Outcome columns, when present, are used only for retrospective audit.
- Runtime penalty is intentionally not part of the score.

## Configuration

- input: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/metagb_preflight_adapter/outputs/preflight_dataset_table.csv`
- rows: 11
- weights: linear=0.35, decoupling=0.25, heterogeneity=0.2, concentration=0.1, support=0.1
- strong_threshold: 0.65
- moderate_threshold: 0.4

## Band Counts

- MODERATE_MECHANISM_MATCH: 3
- WEAK_MECHANISM_MATCH: 8

## Top Mechanism Matches

| dataset                              | mechanism_band           |   mechanism_alignment_score |   component_linear_leaf_advantage |   component_cond_pred_decoupling |   component_local_heterogeneity |   component_predictive_concentration |   component_signal_support |
|:-------------------------------------|:-------------------------|----------------------------:|----------------------------------:|---------------------------------:|--------------------------------:|-------------------------------------:|---------------------------:|
| airfoil_self_noise.csv               | MODERATE_MECHANISM_MATCH |                   0.63959   |                         1         |                        0.0833333 |                        0.628279 |                             0.500516 |                   0.930495 |
| miami_housing.csv                    | MODERATE_MECHANISM_MATCH |                   0.555864  |                         1         |                        0.5       |                        0        |                             0        |                   0.808639 |
| concrete_compressive_strength.csv    | MODERATE_MECHANISM_MATCH |                   0.415223  |                         0         |                        0.916667  |                        0.229378 |                             0.462432 |                   0.939377 |
| physiochemical_protein.csv           | WEAK_MECHANISM_MATCH     |                   0.397543  |                         1         |                        0         |                        0        |                             0.313288 |                   0.162137 |
| QSAR_fish_toxicity.csv               | WEAK_MECHANISM_MATCH     |                   0.323612  |                         0         |                        0.916667  |                        0        |                             0.448561 |                   0.495888 |
| wine_quality.csv                     | WEAK_MECHANISM_MATCH     |                   0.276658  |                         0         |                        0.916667  |                        0        |                             0.2619   |                   0.213018 |
| healthcare_insurance_expenses.csv    | WEAK_MECHANISM_MATCH     |                   0.229623  |                         0.158361  |                        0.291667  |                        0        |                             0.969635 |                   0.043163 |
| houses.csv                           | WEAK_MECHANISM_MATCH     |                   0.179107  |                         0         |                        0.291667  |                        0        |                             0.548958 |                   0.512942 |
| diamonds.csv                         | WEAK_MECHANISM_MATCH     |                   0.13145   |                         0.0510362 |                        0.708333  |                        0        |                             0.259972 |                   0.384638 |
| Food_Delivery_Time.csv               | WEAK_MECHANISM_MATCH     |                   0.092982  |                         0         |                        0.142857  |                        0        |                             0.572677 |                   0        |
| Another-Dataset-on-used-Fiat-500.csv | WEAK_MECHANISM_MATCH     |                   0.0227298 |                         0         |                        0         |                        0        |                             0.796446 |                   0.505808 |

## Retrospective Audit Only

| mechanism_band           |   count |       mean |      median |
|:-------------------------|--------:|-----------:|------------:|
| MODERATE_MECHANISM_MATCH |       3 | -0.0264028 | -0.00261404 |
| WEAK_MECHANISM_MATCH     |       8 | -0.0286382 | -0.00868313 |

This audit is not training evidence. It only checks whether the intrinsic mechanism score is aligned with existing completed experiments.

## Academic Interpretation

A high score means the dataset appears structurally compatible with MetaGB's feature-selected local linear leaves and conditional/predictive feature decoupling. It does not guarantee a benchmark win; it says the dataset deserves a MetaGB trial on mechanism grounds.
