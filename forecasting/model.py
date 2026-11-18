"""Transparent ridge-regression forecasting with JSON-only model artifacts."""
import json
import math
from pathlib import Path
import numpy as np

LAGS = (1, 2, 24, 48, 168)
MIN_HISTORY = max(LAGS)
FEATURE_COUNT = len(LAGS) + 5
VERSION = 1


def build_features(history, next_index: int) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or len(values) < MIN_HISTORY or not np.isfinite(values).all():
        raise ValueError(f"history must contain at least {MIN_HISTORY} finite values")
    if next_index < MIN_HISTORY:
        raise ValueError(f"next_index must be at least {MIN_HISTORY}")
    hour = 2 * np.pi * (next_index % 24) / 24
    week = 2 * np.pi * (next_index % 168) / 168
    return np.array([*(values[-lag] for lag in LAGS), values[-24:].mean(),
                     np.sin(hour), np.cos(hour), np.sin(week), np.cos(week)])


def feature_matrix(series, start: int, end: int):
    return np.stack([build_features(series[:i], i) for i in range(start, end)])


def fit_model(series, train_end: int, alpha: float = 10.0) -> dict:
    """Fit standardized ridge features using only observations before train_end."""
    if not MIN_HISTORY < train_end <= len(series) or alpha <= 0:
        raise ValueError("invalid train_end or alpha")
    training_std = float(np.std(series[:train_end]))
    if not math.isfinite(training_std) or training_std <= 0:
        raise ValueError("training series must have finite, nonzero standard deviation for monitoring")
    x = feature_matrix(series, MIN_HISTORY, train_end)
    y = np.asarray(series[MIN_HISTORY:train_end], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    intercept = float(y.mean())
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - intercept))
    return {"version": VERSION, "model": "ridge", "lags": list(LAGS),
            "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(),
            "intercept": intercept, "alpha": alpha,
            "training_mean": float(np.mean(series[:train_end])),
            "training_std": training_std}


def validate_model(model: dict) -> dict:
    if not isinstance(model, dict) or model.get("version") != VERSION or model.get("model") != "ridge" or model.get("lags") != list(LAGS):
        raise ValueError("unsupported model artifact")
    for key in ("mean", "scale", "coef"):
        value = model.get(key)
        if not isinstance(value, list) or len(value) != FEATURE_COUNT or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
            raise ValueError(f"invalid model field: {key}")
    if any(v <= 0 for v in model["scale"]):
        raise ValueError("model scales must be positive")
    value = model.get("intercept")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid intercept")
    for key in ("training_mean", "training_std"):
        value = model.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid model field: {key}")
    if model["training_std"] <= 0:
        raise ValueError("training_std must be positive")
    return model


def predict(model: dict, history, next_index: int) -> float:
    validate_model(model)
    x = build_features(history, next_index)
    return float(model["intercept"] + np.dot((x - model["mean"]) / model["scale"], model["coef"]))


def load_model(path) -> dict:
    path = Path(path)
    if path.stat().st_size > 100_000:
        raise ValueError("model artifact is too large")
    with path.open(encoding="utf-8") as f:
        return validate_model(json.load(f))
