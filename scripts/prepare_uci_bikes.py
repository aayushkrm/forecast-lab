"""Convert the UCI bike-sharing ZIP to one contiguous hourly CSV.

Select the longest uninterrupted run using timestamps only. The dataset has
missing hours; do not silently fill them with invented counts.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zipfile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    runs = []
    current = []
    with ZipFile(args.zipfile) as archive:
        with archive.open("hour.csv") as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline="")):
                # The source contains calendar date/hour, without UTC offsets.
                # Treat these as a nominal hourly clock, not real UTC instants.
                timestamp = datetime.fromisoformat(row["dteday"]).replace(
                    hour=int(row["hr"]), tzinfo=timezone.utc
                )
                if current and timestamp != current[-1][0] + timedelta(hours=1):
                    runs.append(current)
                    current = []
                current.append((timestamp, int(row["cnt"])))
    if current:
        runs.append(current)
    if not runs:
        raise ValueError("no rows found")
    selected = max(runs, key=len)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "value"])
        for timestamp, value in selected:
            writer.writerow([timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), value])
    print(f"Selected {len(selected)} contiguous rows from {len(runs)} runs; "
          f"nominal first={selected[0][0].isoformat()}, last={selected[-1][0].isoformat()}")


if __name__ == "__main__":
    main()
