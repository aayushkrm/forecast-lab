"""Serve one-step forecasts from a locally trained JSON artifact."""
from contextlib import asynccontextmanager
import math
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .model import MIN_HISTORY, load_model, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("FORECAST_MODEL_PATH", "artifacts/model.json")
    app.state.model = load_model(path)
    yield


app = FastAPI(title="Hourly Forecast API", version="1.0.0", lifespan=lifespan)


class ForecastRequest(BaseModel):
    history: list[float] = Field(min_length=MIN_HISTORY, max_length=10000, description="Consecutive hourly observations, oldest first")
    next_index: int = Field(ge=MIN_HISTORY, description="Zero-based index of the hour to predict relative to the start of the training series")


@app.get("/health")
def health():
    return {"status": "ok", "model": "ridge", "model_version": app.state.model["version"]}


@app.post("/forecast")
def forecast(request: ForecastRequest):
    if not all(math.isfinite(v) for v in request.history):
        raise HTTPException(status_code=422, detail="history values must be finite")
    if len(request.history) > request.next_index:
        raise HTTPException(status_code=422, detail="history cannot start before index zero")
    value = predict(app.state.model, request.history, request.next_index)
    if not math.isfinite(value):
        raise HTTPException(status_code=422, detail="prediction is not finite")
    return {"prediction": value, "next_index": request.next_index, "model_version": app.state.model["version"]}


class MonitorRequest(BaseModel):
    recent_values: list[float] = Field(min_length=24, max_length=10000)


@app.post("/monitor")
def monitor(request: MonitorRequest):
    if not all(math.isfinite(v) for v in request.recent_values):
        raise HTTPException(status_code=422, detail="recent_values must be finite")
    mean = sum(request.recent_values) / len(request.recent_values)
    training_mean = app.state.model["training_mean"]
    training_std = app.state.model["training_std"]
    shift = (mean - training_mean) / training_std
    if not math.isfinite(shift):
        raise HTTPException(status_code=422, detail="mean shift is not finite")
    return {"recent_mean": mean, "training_mean": training_mean,
            "training_std": training_std, "mean_shift_in_training_std": shift,
            "threshold": 2.0, "alert": abs(shift) > 2.0}
