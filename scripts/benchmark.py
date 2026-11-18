#!/usr/bin/env python3
"""Measure sequential HTTP forecast latency against a running local service."""
import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request


def percentile(values, fraction):
    """Linearly interpolated percentile, expressed in the same units as values."""
    if not values:
        raise ValueError("at least one measurement is required")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def request_forecast(url, body, timeout):
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if (not isinstance(payload, dict) or isinstance(payload.get("prediction"), bool)
            or not isinstance(payload.get("prediction"), (int, float))
            or not math.isfinite(payload["prediction"]) or payload.get("next_index") != 168
            or payload.get("model_version") != 1):
        raise ValueError("invalid forecast response")
    return elapsed_ms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/forecast")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.requests < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("requests and timeout must be positive; warmup must be nonnegative")
    history = [100 + 10 * math.sin(2 * math.pi * i / 24) for i in range(168)]
    body = json.dumps({"history": history, "next_index": len(history)}, allow_nan=False).encode()
    try:
        for _ in range(args.warmup):
            request_forecast(args.url, body, args.timeout)
        start = time.perf_counter()
        samples = [request_forecast(args.url, body, args.timeout) for _ in range(args.requests)]
        wall_seconds = time.perf_counter() - start
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "url": args.url,
        "requests": args.requests,
        "warmup": args.warmup,
        "concurrency": 1,
        "latency_ms": {"mean": statistics.mean(samples), "p50": percentile(samples, .5),
                       "p95": percentile(samples, .95), "p99": percentile(samples, .99)},
        "wall_seconds": wall_seconds,
        "requests_per_second": args.requests / wall_seconds,
        "scope": "End-to-end sequential HTTP latency including response body parsing; excludes warmup.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
