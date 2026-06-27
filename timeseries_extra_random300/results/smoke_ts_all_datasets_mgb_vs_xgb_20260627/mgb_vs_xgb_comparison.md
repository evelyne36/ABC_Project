# TS all-dataset smoke MGB vs XGB

| dataset       | metric   | direction    |   xgb_cpu |   current_metagb | winner         |   relative_gain_vs_xgb |   xgb_elapsed_sec |   metagb_elapsed_sec |   rows_used |   features_used |   seq_len |   max_raw_rows |   max_features |
|:--------------|:---------|:-------------|----------:|-----------------:|:---------------|-----------------------:|------------------:|---------------------:|------------:|----------------:|----------:|---------------:|---------------:|
| exchange_rate | rmse     | lower_better |  4.22759  |         2.15707  | current_metagb |              0.489765  |          0.103332 |             0.413808 |         240 |               8 |         4 |            240 |             32 |
| ETTh2         | rmse     | lower_better |  1.85733  |         1.44838  | current_metagb |              0.220184  |          0.262732 |             0.508112 |         240 |               7 |         4 |            240 |             32 |
| Weather       | rmse     | lower_better |  1.38965  |         1.10053  | current_metagb |              0.208052  |          0.317054 |             2.08446  |         240 |              21 |         4 |            240 |             32 |
| PEMS03        | rmse     | lower_better |  1.08871  |         0.995462 | current_metagb |              0.0856489 |          0.174161 |             3.97499  |         240 |              32 |         4 |            240 |             32 |
| ILI           | rmse     | lower_better |  0.669335 |         0.643296 | current_metagb |              0.0389022 |          0.279648 |             0.54742  |         240 |               7 |         4 |            240 |             32 |
| ETTm2         | rmse     | lower_better |  0.230865 |         0.226249 | current_metagb |              0.019993  |          0.266923 |             0.491551 |         240 |               7 |         4 |            240 |             32 |
| solar         | rmse     | lower_better |  0.235246 |         0.23203  | current_metagb |              0.0136736 |          0.154432 |             3.84646  |         240 |              32 |         4 |            240 |             32 |
| PEMS08        | rmse     | lower_better |  2.75737  |         2.85354  | xgb_cpu        |             -0.0348793 |          0.178185 |             4.41095  |         240 |              32 |         4 |            240 |             32 |
| electricity   | rmse     | lower_better |  0.404263 |         0.422354 | xgb_cpu        |             -0.0447501 |          0.190717 |             4.60597  |         240 |              32 |         4 |            240 |             32 |
| ETTm1         | rmse     | lower_better |  0.838664 |         0.908471 | xgb_cpu        |             -0.0832351 |          0.275752 |             0.497628 |         240 |               7 |         4 |            240 |             32 |
| PEMS07        | rmse     | lower_better |  0.750988 |         0.816517 | xgb_cpu        |             -0.0872571 |          0.174576 |             3.73931  |         240 |              32 |         4 |            240 |             32 |
| PEMS04        | rmse     | lower_better |  3.97649  |         4.44158  | xgb_cpu        |             -0.116957  |          0.192586 |             4.40848  |         240 |              32 |         4 |            240 |             32 |
| traffic       | rmse     | lower_better |  0.546004 |         0.645502 | xgb_cpu        |             -0.182229  |          0.202897 |             4.49289  |         240 |              32 |         4 |            240 |             32 |
| ETTh1         | rmse     | lower_better |  0.749937 |         0.98856  | xgb_cpu        |             -0.31819   |          0.270696 |             0.499851 |         240 |               7 |         4 |            240 |             32 |

Lightweight smoke only: seq_len/max_raw_rows/max_features are capped for speed; not a final full-protocol result.
