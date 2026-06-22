# TabPFN/OpenML Tree-Model Candidate Pool

This file lists low- and mid-dimensional tabular datasets that are plausible follow-up candidates for XGBoost vs MetaGB. Prefer datasets with <=80 raw features; avoid high-dimensional sparse chemistry/text/image-derived tables for first-pass MetaGB runs.

## Regression

| Dataset | Why consider it |
|---|---|
| `kin8nm` | 8 features, 8192 rows; smooth nonlinear regression; strongest first-run candidate. |
| `energy_efficiency` | 8 features, small engineering regression; likely local linear structure. |
| `naval_propulsion_plant` | low/mid dimensional engineering regression; very low noise, useful but may be easy. |
| `cpu_activity` | about 21 features; common OpenML regression benchmark. |
| `space_ga` | low/mid dimensional regression; useful CTR23-style complement. |
| `Moneyball` | small semantic tabular regression; good lightweight robustness check. |
| `abalone` | 8 features; regression/classification variants exist; not a priority because target is ordinal-like. |
| `boston` | 13 features; legacy benchmark, use only as appendix because of dataset concerns. |
| `friedman1` | synthetic low-dimensional nonlinear regression; useful diagnostic, less strong as real-data evidence. |

## Binary

| Dataset | Why consider it |
|---|---|
| `banknote-authentication` | 4 features; very fast sanity check, but possibly too easy. |
| `credit-approval` | mixed tabular credit data, about 15 features; good binary supplement. |
| `breast-w` | 9 features; classic low-dimensional medical binary benchmark. |
| `ilpd` | 10 features; small clinical binary dataset. |
| `haberman` | 3 features; tiny and noisy, useful for small-data behavior. |
| `kc1` | 21 features; software defect prediction, tree-friendly. |
| `electricity` | 8 features, many rows; useful scalability check with few columns. |
| `phoneme` | 5 features; binary acoustic benchmark, often good for nonlinear classifiers. |
| `mozilla4` | 5 features; binary performance dataset, if available. |
| `pc1` | 21 features; software defect benchmark similar to kc1. |

## Multiclass

| Dataset | Why consider it |
|---|---|
| `vehicle` | 18 features, 4 classes; compact multiclass benchmark. |
| `segment` | 19 features, 7 classes; strong low-dimensional multiclass candidate. |
| `satimage` | 36 features, 6 classes; larger but still feasible. |
| `vowel` | 10 features, 11 classes; small multiclass benchmark. |
| `optdigits` | 64 features, 10 classes; feasible but slower than the top choices. |
| `balance-scale` | 4 features, 3 classes; fast but very simple. |
| `mfeat-fourier` | 76 features, 10 classes; feasible medium-width check. |
| `mfeat-karhunen` | 64 features, 10 classes; feasible medium-width check. |
