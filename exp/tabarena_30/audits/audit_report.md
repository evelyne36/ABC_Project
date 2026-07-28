# TabArena Missing-Row and Queue Audit

## Missing Row Drop Rate by Task Type

| task_type   |   datasets |   raw_rows |   feature_missing_rows_after_target_drop |   target_missing_rows |   feature_missing_drop_rate_after_target_drop |
|:------------|-----------:|-----------:|-----------------------------------------:|----------------------:|----------------------------------------------:|
| binary      |         30 |     724703 |                                   252593 |                     0 |                                     0.348547  |
| multiclass  |          8 |      94476 |                                     1699 |                     0 |                                     0.0179834 |
| regression  |         13 |     219355 |                                        0 |                     0 |                                     0         |

## Queue Notes

- `in_vehicle_coupon_recommendation.csv` is excluded from the queue because one-hot strict missing-row removal leaves only 108 rows.
- Drop preprocessing in `tabarena_30` uses strict row filtering first, then drops discrete columns; this aligns row counts with one-hot for datasets such as HR.
- User-marked slow datasets are forced to the tail of `queue_priority.csv`.

## First 20 Queue Items

| task_type   | handling   | dataset                                        |   processed_rows |   processed_features |   pressure_proxy |   xgb_pending_folds |   metagb_pending_folds | note   |   slow_rank | preprocess_error   |
|:------------|:-----------|:-----------------------------------------------|-----------------:|---------------------:|-----------------:|--------------------:|-----------------------:|:-------|------------:|:-------------------|
| binary      | drop       | bank-marketing.csv                             |            45211 |                    5 |          1130275 |                   5 |                      5 |        |           0 |                    |
| binary      | drop       | online_shoppers_intention.csv                  |            12330 |                   14 |          2416680 |                   5 |                      5 |        |           0 |                    |
| binary      | drop       | jm1.csv                                        |            10880 |                   21 |          4798080 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | jm1.csv                                        |            10880 |                   21 |          4798080 |                   5 |                      5 |        |           0 |                    |
| binary      | drop       | heloc.csv                                      |            10459 |                   23 |          5532811 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | heloc.csv                                      |            10459 |                   23 |          5532811 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | online_shoppers_intention.csv                  |            12330 |                   29 |         10369530 |                   5 |                      5 |        |           0 |                    |
| binary      | drop       | credit_card_clients_default.csv                |            30000 |                   23 |         15870000 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | credit_card_clients_default.csv                |            30000 |                   23 |         15870000 |                   5 |                      5 |        |           0 |                    |
| binary      | drop       | APSFailure.csv                                 |              756 |                  170 |         21848400 |                   0 |                      5 |        |           0 |                    |
| binary      | onehot     | APSFailure.csv                                 |              756 |                  170 |         21848400 |                   0 |                      5 |        |           0 |                    |
| binary      | drop       | customer_satisfaction_in_airline.csv           |           129487 |                   18 |         41953788 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | NATICUSdroid.csv                               |             7491 |                   86 |         55403436 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | taiwanese_bankruptcy_prediction.csv            |             6819 |                   94 |         60252684 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | bank-marketing.csv                             |            45211 |                   37 |         61893859 |                   5 |                      5 |        |           0 |                    |
| binary      | drop       | coil2000_insurance_policies.csv                |             9822 |                   80 |         62860800 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | customer_satisfaction_in_airline.csv           |           129487 |                   25 |         80929375 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | coil2000_insurance_policies.csv                |             9822 |                  149 |        218058222 |                   5 |                      5 |        |           0 |                    |
| binary      | onehot     | HR_Analytics_Job_Change_of_Data_Scientists.csv |             8955 |                  177 |        280551195 |                   0 |                      5 |        |           0 |                    |
| binary      | onehot     | Bioresponse.csv                                |             3751 |                 1776 |      11831314176 |                   0 |                      5 |        |           0 |                    |
