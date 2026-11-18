"""Train on synthetic data and evaluate rolling one-step forecasts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import generate_series, load_csv
from .model import MIN_HISTORY, fit_model, predict
from .intervals import calibrate, evaluate_intervals


def metrics(actual, predicted):
    errors = np.asarray(predicted) - np.asarray(actual)
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2)))}


def evaluate(series, start, end, model):
    actual = series[start:end]
    naive = series[start - 1:end - 1]
    predictions = [predict(model, series[:i], i) for i in range(start, end)]
    return {"naive_last_value": metrics(actual, naive),
            "seasonal_naive_24h": metrics(actual, series[start - 24:end - 24]),
            "seasonal_naive_168h": metrics(actual, series[start - 168:end - 168]),
            "ridge": metrics(actual, predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/model.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage", type=float, default=0.9, help="Target interval coverage, strictly between 0 and 1")
    parser.add_argument("--csv", help="CSV with consecutive UTC hourly timestamp,value columns")
    args = parser.parse_args()
    series = load_csv(args.csv) if args.csv else generate_series(seed=args.seed)
    train_end, validation_end = int(len(series) * .7), int(len(series) * .85)
    model = fit_model(series, train_end)
    report = {"data": {"source": "csv" if args.csv else "synthetic", "csv_path": args.csv, "seed": None if args.csv else args.seed, "points": len(series),
                       "train_end_exclusive": train_end, "validation_end_exclusive": validation_end},
              "evaluation": {"validation": evaluate(series, train_end, validation_end, model),
                             "test": evaluate(series, validation_end, len(series), model)}}
    # The model stays fixed. Each forecast uses only values before its target.
    validation_predictions = [predict(model, series[:i], i) for i in range(train_end, validation_end)]
    calibration = calibrate(series[train_end:validation_end], validation_predictions, args.coverage)
    test_predictions = [predict(model, series[:i], i) for i in range(validation_end, len(series))]
    report["intervals"] = {"method": "symmetric held-out absolute residual order statistic",
                           "calibration_window": "validation", "calibration": calibration,
                           "test": evaluate_intervals(series[validation_end:], test_predictions, calibration)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path = output.with_name(output.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Model: {output}; report: {report_path}")


if __name__ == "__main__":
    main()
