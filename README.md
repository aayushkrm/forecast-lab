# Forecast Lab

A small, reproducible forecasting service and evaluation lab. Train on past hourly observations, compare against simple baselines, serve one-step forecasts, and inspect a basic input-shift signal. It runs locally on a CPU without API keys.

```text
hourly data ──> chronological split ──> ridge model ──> JSON artifact ──> HTTP API
                      │                    │                              │
                      └── baseline comparison + holdout report           └── shift check
```

## Quick start

Requires Python 3.10 or newer. The default dataset is generated locally and is not real demand.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m forecasting.train
uvicorn forecasting.api:app --host 127.0.0.1 --port 8000
```

Open [the local interactive API](http://127.0.0.1:8000/docs), or use the request example in [CORE.md](CORE.md). In another terminal, run `python scripts/benchmark.py` for sequential local HTTP timings. Run `python -m pytest -q` for the test suite.

The API accepts the last 168 or more consecutive hourly observations, oldest first, and the zero-based index of the next hour relative to the start of the training series. Forecasts use the observations supplied with the request; they are not stored by the service. `POST /monitor` compares a supplied window's mean against the training mean. This is an on-demand heuristic, not an automated drift detector or evidence that accuracy has changed. See [CORE.md](CORE.md) for the full contract, CSV format, and limitations.

## Experiment on public data

The example below uses [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (Fanaee-T, 2013, [DOI](https://doi.org/10.24432/C5W894), CC BY 4.0). It contains historical hourly bike rentals in Washington, DC, from 2011–2012. The preparation script selects the longest uninterrupted sequence using timestamps alone; it excludes gaps and all other periods. The result describes this selected historical window, not all conditions or present-day performance. Source date/hour fields lack UTC offsets: the generated `Z` timestamps are **nominal clock encodings**, not verified UTC instants. Do not use them as real UTC times.

```bash
curl -fsSL 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip' -o /tmp/bike-sharing.zip
python scripts/prepare_uci_bikes.py /tmp/bike-sharing.zip data/uci-bike-hourly.csv
python -m forecasting.train --csv data/uci-bike-hourly.csv --output artifacts/uci-bike.json
```

The recorded run selected 4,820 hours, nominally 11 April–29 October 2012. The first 70% trains the model; the next 15% is validation and the final 15% is the held-out test. Each prediction is one hour ahead, using actual earlier observations, including observations in the test period as they become available. This is rolling evaluation, not a multi-step forecast. Ridge is fitted on training data only; there was no hyperparameter search. Counts are rentals per hour.

| Held-out model | MAE ↓ | RMSE ↓ |
|---|---:|---:|
| Previous hour | 97.41 | 144.38 |
| Same hour previous day | 98.21 | 155.31 |
| Same hour previous week | 60.22 | 103.01 |
| Ridge, lagged features | 57.10 | 84.24 |

For offline uncertainty evaluation, the fixed ridge model uses the 723 validation residuals to calibrate a symmetric interval with a 90% target. The 723 test forecasts achieved **89.35%** coverage, with mean width **280.70 rentals/hour**. The interval can extend below zero; it is not clipped. The service still returns point forecasts only. These are empirical results from one dependent time series, not a coverage guarantee.

The model improves on these baselines in this one selected historical period. It is not a deployment, a current demand forecast, or a claim of generalization across cities. See [experiment details](docs/experiment.md) and the machine-readable [report](artifacts/uci-bike_report.json).

## Design and limits

- Features use lags, a past 24-hour mean and periodic terms. Numeric JSON artifacts avoid executable pickle loading.
- Input must be finite and hourly, without gaps. The index origin must remain consistent across training and inference.
- There is no authentication, automatic monitoring, persistence, uncertainty interval, missing-data imputation, or load-balanced deployment. Bind locally and treat this as a research demo.
- The mean-shift heuristic is sensitive to seasonality and is not calibrated for operational alerts.
- The benchmark measures sequential local HTTP requests, including response parsing; it is not a concurrency or production capacity test.

The separate [agent trajectory evaluation project](https://github.com/aayushkrm/agent-trace-eval) demonstrates deterministic tests for tool-use workflows. [Career research](docs/career-research.md) explains why these projects fit the resume and current hiring examples.
