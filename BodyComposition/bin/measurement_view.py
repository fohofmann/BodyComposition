"""CLI convenience views over canonical measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from BodyComposition.measurement.api import l3_measurements, range_measurements


def _json_value(value):
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        return None if bool(pd.isna(value)) else value
    except (TypeError, ValueError):
        return value


def _records(table: pd.DataFrame) -> list[dict]:
    return [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in table.to_dict(orient="records")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read L3 or named-range views from canonical measurement Parquet tables."
    )
    parser.add_argument("--tables", type=Path, required=True, help="Directory containing the three Parquet tables.")
    parser.add_argument("--view", choices=("l3", "range"), required=True)
    parser.add_argument(
        "--aggregation",
        choices=("territory_mean", "slice"),
        default="territory_mean",
    )
    parser.add_argument("--start-level")
    parser.add_argument("--end-level")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--height-m",
        type=float,
        help="Measured height in metres; enables explicit SMI/volume indices.",
    )
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    args = parser.parse_args()

    if args.view == "l3":
        table = l3_measurements(
            args.tables,
            aggregation=args.aggregation,
            height_m=args.height_m,
        )
    else:
        if not args.start_level or not args.end_level:
            parser.error("--start-level and --end-level are required for --view range.")
        table = range_measurements(
            args.tables,
            start_level=args.start_level,
            end_level=args.end_level,
            allow_partial=args.allow_partial,
            height_m=args.height_m,
        )
    if args.format == "csv":
        print(table.to_csv(index=False), end="")
    else:
        print(json.dumps(_records(table), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
