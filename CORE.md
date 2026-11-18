# Forecasting service core

This project predicts the next hour of a synthetic demand series. The data is generated locally; it is not a real dataset. The model's accuracy on it does not establish performance on real demand.

The generator uses a fixed random seed, daily and weekly cycles, a small trend, and Gaussian noise. Training uses the first 70% of 8,760 hourly values. Validation uses the next 15%, and test uses the final 15%. The ridge model is fitted only on the training segment. Both evaluations are rolling one-step forecasts: each forecast uses the actual observations available before that hour. The naive baselines predict the previous hour, the same hour one day ago, and the same hour one week ago. Validation and test report MAE and RMSE. The ridge model is not tuned or retrained on validation or test values. A separate offline interval radius is calibrated on validation residuals, then measured on the later test window; the service does not return intervals.

Features are the values at lags 1, 2, 24, 48, and 168 hours; the mean of the previous 24 hours; and sine/cosine terms for hour and week. The index starts at zero for the first generated hour. The fitted model is stored as numeric JSON, which avoids executable pickle deserialization. Artifact fields and dimensions are validated on load.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m forecasting.train
uvicorn forecasting.api:app --host 127.0.0.1 --port 8000
```

Training writes `artifacts/model.json` and `artifacts/model_report.json`. Use `--seed` to change the generated data or `--output` to change the model path. Set `FORECAST_MODEL_PATH` before starting the server if the model is elsewhere. The server loads the model on startup; restart it after retraining.

`GET /health` checks that the model loaded. `POST /forecast` accepts JSON containing `history`, at least 168 consecutive hourly observations in chronological order, and `next_index`, the zero-based index of the hour to forecast. The final observation is from `next_index - 1`. The index must share the generator's time origin so calendar features align. The response gives the predicted value and model version. See `/docs` for the interactive API schema.

Example using generated observations:

```sh
python - <<'PY'
import json
from urllib.request import Request, urlopen
from forecasting.data import generate_series
history = generate_series()[:168].tolist()
request = Request('http://127.0.0.1:8000/forecast', data=json.dumps({'history': history, 'next_index': 168}).encode(), headers={'Content-Type': 'application/json'})
print(urlopen(request).read().decode())
PY
```

This service predicts one step only and has no intervals, automatic monitoring, authentication, persistent storage, or missing-data handling. It assumes contiguous hourly inputs and the same time origin as training. It is a demonstration for local use; production deployment would need domain data and a fresh evaluation.

To train on your own data, prepare a CSV with a header `timestamp,value`. Timestamps must be ISO 8601 UTC (for example `2024-01-01T00:00:00Z`), ordered, on the hour, and consecutive. Values must be finite numbers. At least 300 rows are required. Provide the data source, license, units, and any preprocessing separately when sharing your results.

```sh
python -m forecasting.train --csv path/to/hourly.csv --output artifacts/custom.json
FORECAST_MODEL_PATH=artifacts/custom.json uvicorn forecasting.api:app --host 127.0.0.1 --port 8000
```

For a CSV, index zero corresponds to its first row; send `next_index` relative to that row. The model uses relative positions for its periodic features, and the same origin must be used for every request. The report records the CSV path but does not make claims about its source or license.

`POST /monitor` accepts `{"recent_values": [...]}` with 24 or more observations. It compares their mean with the training mean in units of training standard deviation and flags an absolute shift greater than 2. This is a simple heuristic, not a statistical drift test or a measure of model accuracy. Seasonality, trend and window size can trigger or hide changes; calibrate the threshold and windows for the real domain. It does not store or monitor requests automatically.
