# Win/Tie/Loss by Group

| family | protocol | encoding | task_type | datasets | paired_folds | dataset_mgb_win | dataset_tie | dataset_xgb_win | fold_mgb_win | fold_tie | fold_xgb_win | median_runtime_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | binary | 6 | 30.000 | 6 | 0 | 0 | 26.000 | 1 | 3 | 46.403 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | 29.000 | 145.000 | 24.000 | 1 | 4 | 124.000 | 0 | 21.000 | 15.231 |
| tabarena30_0629 | xgb_best_mgb30 | drop | binary | 9 | 38.000 | 3 | 0 | 6 | 16.000 | 1 | 21.000 | 2.625 |
| tabarena30_0629 | xgb_best_mgb30 | onehot | binary | 8 | 40.000 | 3 | 1 | 4 | 20.000 | 1 | 19.000 | 2.01982 |
| tabarena30_0629 | xgb_best_mgb30 | onehot | multiclass | 3 | 11.000 | 3 | 0 | 0 | 10.000 | 0 | 1 | 4.26528 |
| tabarena_main_0624 | random300 | drop | binary | 17.000 | 85.000 | 8 | 0 | 9 | 40.000 | 1 | 44.000 | 22.015 |
| tabarena_main_0624 | random300 | drop | multiclass | 4 | 20.000 | 2 | 1 | 1 | 13.000 | 1 | 6 | 26.228 |
| tabarena_main_0624 | random300 | onehot | binary | 13.000 | 65.000 | 5 | 0 | 8 | 35.000 | 1 | 29.000 | 58.984 |
| tabarena_main_0624 | random300 | onehot | multiclass | 3 | 15.000 | 2 | 0 | 1 | 12.000 | 0 | 3 | 21.480 |
| tabarena_main_0624 | random600 | drop | regression | 11.000 | 55.000 | 3 | 0 | 8 | 18.000 | 4 | 33.000 | 10.301 |
| tabarena_main_0624 | random600 | onehot | regression | 10.000 | 50.000 | 5 | 0 | 5 | 27.000 | 2 | 21.000 | 13.810 |
| tabular_otherdata_0629 | random300 | drop | binary | 8 | 40.000 | 2 | 2 | 4 | 15.000 | 4 | 21.000 | 14.018 |
| tabular_otherdata_0629 | random300 | drop | multiclass | 2 | 10.000 | 1 | 0 | 1 | 5 | 0 | 5 | 25.495 |
| tabular_otherdata_0629 | random300 | drop | regression | 7 | 35.000 | 3 | 0 | 4 | 18.000 | 1 | 16.000 | 10.090 |
| tabular_otherdata_0629 | random300 | onehot | binary | 5 | 25.000 | 2 | 2 | 1 | 13.000 | 4 | 8 | 10.368 |
| tabular_otherdata_0629 | random300 | onehot | multiclass | 1 | 5 | 0 | 0 | 1 | 0 | 0 | 5 | 13.141 |
| tabular_otherdata_0629 | random300 | onehot | regression | 6 | 30.000 | 2 | 0 | 4 | 14.000 | 1 | 15.000 | 22.371 |
