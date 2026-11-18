# Public-data experiment card

Source: Fanaee-T, H. (2013), Bike Sharing, UCI Machine Learning Repository, https://doi.org/10.24432/C5W894, CC BY 4.0. The original data are historical bike rental counts, not gas-pipeline or market data. Data license: https://creativecommons.org/licenses/by/4.0/.

Preparation: `scripts/prepare_uci_bikes.py` reads `hour.csv` from the original ZIP. It chooses the longest sequence of consecutive nominal calendar hours, based on timestamps rather than the target. This makes lag lengths meaningful without fabricating missing values. It does use timestamp availability over the full dataset to choose a window; the evaluation is conditional on this selected window and excludes other periods. The file's date/hour has no offset; `Z` in the converted file is an artificial clock encoding to satisfy the loader, not a statement about real UTC time. Do not compare it to actual UTC events.

Original ZIP SHA-256: `b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401`. Prepared CSV SHA-256: `d40692621fc0000899e57ec9cb39aa0e607c23637a732cc88f35c52df7f24680`. The prepared data are derived from the original by selecting the longest contiguous interval and retaining only nominal time and rental count.

Evaluation: indices 0–3373 are training, 3374–4096 validation, and 4097–4819 test. The training targets begin at index 168 because the longest lag is 168 hours. Ridge uses standardized features with statistics fitted on training only. Alpha is fixed at 10; validation was not used to select it. For each test target, history contains all actual past values; there is no future target in the feature vector. The test contains 723 one-hour-ahead predictions. MAE and RMSE are in rentals/hour. This is a single historical split; estimates are correlated in time and no confidence intervals or external validation are claimed.

The report was generated on 19 November 2026 using the commands in the README. Re-running may show small floating-point differences across machines. The report has validation metrics as well as test metrics. The model's limited features do not use known future weather, holidays, or special-event data.

A local smoke benchmark on the synthetic default model used 10 warmup and 100 sequential HTTP requests on an Intel Core i7-4770HQ, macOS 14.8.9, Python 3.11.15. It measured p50 6.33 ms and p95 14.96 ms, including client and response parsing. This is one noisy local run, not a throughput guarantee or a comparison with another model. Run the included benchmark on the target host before making any performance claim.
