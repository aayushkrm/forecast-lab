"""Synthetic demand data; not a real-world dataset or performance claim."""
import numpy as np


def generate_series(n: int = 24 * 365, seed: int = 42) -> np.ndarray:
    """Generate hourly demand with daily/weekly cycles, trend and random noise."""
    if n < 200:
        raise ValueError("n must be at least 200")
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return (100 + 0.002 * t + 15 * np.sin(2 * np.pi * (t - 7) / 24)
            + 6 * np.sin(2 * np.pi * t / 168)
            + 3 * np.cos(2 * np.pi * t / 12) + rng.normal(0, 3, n))


def load_csv(path) -> np.ndarray:
    """Load timestamp,value CSV with contiguous hourly ISO 8601 timestamps."""
    import csv
    from datetime import datetime, timedelta

    values = []
    previous = None
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"timestamp", "value"}.issubset(reader.fieldnames):
            raise ValueError("CSV must have timestamp and value columns")
        for line, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                value = float(row["value"])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid timestamp or value on CSV line {line}") from exc
            if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
                raise ValueError(f"timestamp on CSV line {line} must use UTC")
            if timestamp.minute or timestamp.second or timestamp.microsecond:
                raise ValueError(f"timestamp on CSV line {line} must be on the hour")
            if previous is not None and timestamp - previous != timedelta(hours=1):
                raise ValueError(f"timestamps must be consecutive hourly UTC values (line {line})")
            if not np.isfinite(value):
                raise ValueError(f"value on CSV line {line} must be finite")
            previous = timestamp
            values.append(value)
    if len(values) < 300:
        raise ValueError("CSV must contain at least 300 hourly rows")
    return np.asarray(values, dtype=float)
