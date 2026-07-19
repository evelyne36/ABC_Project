# Time-Series Extra Random300 Dataset Description

Raw data directory: `timeseries_extra_random300/data/raw`

Total registered datasets: 14

Current raw files present: 14

Total raw size: 529.32 MiB

For `npz_array` datasets, `shape` is the stored raw array shape. The benchmark
loader flattens 3D arrays from `(timesteps, nodes, features)` into
`(timesteps, nodes * features)`.

| Dataset | Family | Frequency | Raw file | Timesteps | Shape | Size MiB | Description |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| ETTh1 | ETDataset | h | ETTh1.csv | 17420 | 17420x7 | 2.47 | Informer/Autoformer/PatchTST standard ETT hourly benchmark. |
| ETTh2 | ETDataset | h | ETTh2.csv | 17420 | 17420x7 | 2.31 | Informer/Autoformer/PatchTST standard ETT hourly benchmark. |
| ETTm1 | ETDataset | 15min | ETTm1.csv | 69680 | 69680x7 | 9.88 | Informer/Autoformer/PatchTST standard ETT minute benchmark. |
| ETTm2 | ETDataset | 15min | ETTm2.csv | 69680 | 69680x7 | 9.23 | Informer/Autoformer/PatchTST standard ETT minute benchmark. |
| exchange_rate | LSTNet multivariate time-series | d | exchange_rate.txt | 7588 | 7588x8 | 0.52 | Daily exchange rates of eight countries from 1990 to 2016. |
| electricity | LSTNet multivariate time-series | h | electricity.txt | 26304 | 26304x321 | 90.63 | Hourly electricity consumption benchmark, retained for later expansion. |
| traffic | LSTNet multivariate time-series | h | traffic.txt | 17544 | 17544x862 | 129.80 | Traffic occupancy benchmark, retained for later expansion. |
| solar | LSTNet multivariate time-series | 10min | solar_AL.txt | 52560 | 52560x137 | 171.68 | Solar power production benchmark; registered as `solar`, raw file is `solar_AL.txt`. |
| Weather | Time-Series-Library long-term forecasting | 10min | weather.csv | 52696 | 52696x21 | 6.90 | Official TSLib Hugging Face mirror for Weather long-term forecasting. |
| ILI | Time-Series-Library long-term forecasting | w | national_illness.csv | 966 | 966x7 | 0.06 | Official TSLib Hugging Face mirror for national illness / ILI long-term forecasting. |
| PEMS03 | PEMS traffic forecasting | 5min | PEMS03.npz | 26208 | 26208x358x1 | 15.07 | Traffic benchmark; very wide after windowing. |
| PEMS04 | PEMS traffic forecasting | 5min | PEMS04.npz | 16992 | 16992x307x3 | 31.43 | Traffic benchmark; loader flattens to 921 model channels. |
| PEMS07 | PEMS traffic forecasting | 5min | PEMS07.npz | 28224 | 28224x883x1 | 41.68 | Traffic benchmark; extremely wide after windowing. |
| PEMS08 | PEMS traffic forecasting | 5min | PEMS08.npz | 17856 | 17856x170x3 | 17.67 | Traffic benchmark; loader flattens to 510 model channels. |
