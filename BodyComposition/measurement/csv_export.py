"""Deterministic convenience CSV mirrors of canonical Parquet tables."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from BodyComposition.measurement.api import TABLE_NAMES


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _csv_ready_table(table: pd.DataFrame) -> pd.DataFrame:
    """Serialize list-valued cells as compact JSON while retaining all columns."""

    output = table.copy()
    for column in output.columns:
        values = output[column].tolist()
        if not any(isinstance(value, (list, tuple, np.ndarray)) for value in values):
            continue
        output[column] = [
            (
                json.dumps(
                    [_json_scalar(item) for item in value],
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                if isinstance(value, (list, tuple, np.ndarray))
                else value
            )
            for value in values
        ]
    return output


def write_csv_table(
    table: pd.DataFrame,
    destination: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Atomically write one UTF-8 CSV mirror with deterministic row ordering."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial.csv")
    try:
        _csv_ready_table(table).to_csv(
            temporary,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            na_rep="",
        )
        if path.is_file() and not overwrite:
            if (
                path.stat().st_size == temporary.stat().st_size
                and _file_sha256(path) == _file_sha256(temporary)
            ):
                return path
            raise FileExistsError(f"CSV destination already exists with other content: {path}.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_csv_mirrors(
    tables: Mapping[str, pd.DataFrame],
    destination: str | Path,
    *,
    overwrite: bool,
) -> dict[str, Path]:
    """Write all five canonical measurement tables as convenience CSV files."""

    missing = [name for name in TABLE_NAMES if name not in tables]
    if missing:
        raise ValueError(f"CSV export is missing canonical tables: {missing}.")
    root = Path(destination)
    return {
        name: write_csv_table(
            tables[name],
            root / f"{name}.csv",
            overwrite=overwrite,
        )
        for name in TABLE_NAMES
    }
