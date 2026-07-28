"""Read-only convenience views over canonical measurement Parquet tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from BodyComposition.measurement.aggregation import (
    aggregate_named_range,
    select_l3_view,
)
from BodyComposition.measurement.contracts import (
    BINS_PER_VERTEBRAL_TERRITORY,
    HU_DISTRIBUTION_REQUIRED_COLUMNS,
    MEASUREMENT_SCHEMA_VERSION,
    SIGNATURE_REQUIRED_COLUMNS,
    SLICE_REQUIRED_COLUMNS,
    SUMMARY_REQUIRED_COLUMNS,
    VERTEBRA_REQUIRED_COLUMNS,
    VertebralTerritory,
    canonical_arrow_schema,
    validate_hu_distribution_contract,
    validate_signature_contract,
)

TABLE_NAMES = (
    "slices",
    "vertebrae",
    "summaries",
    "signature",
    "hu_distributions",
)
IDENTITY_COLUMNS = ("schema_version", "run_id", "analysis_id", "case_id")


@dataclass(frozen=True)
class SignatureComponents:
    """The identity-linked longitudinal and global tissue-quality signature."""

    longitudinal: pd.DataFrame
    tissue_hu_distributions: pd.DataFrame


def _validated_height_m(height_m: float | None) -> float | None:
    if height_m is None:
        return None
    if isinstance(height_m, bool) or not isinstance(height_m, (int, float)):
        raise TypeError("height_m must be a measured numeric height in metres.")
    value = float(height_m)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("height_m must be finite and positive.")
    return value


def _add_l3_height_index(table: pd.DataFrame, height_m: float | None) -> pd.DataFrame:
    height = _validated_height_m(height_m)
    if height is None:
        return table
    output = table.copy()
    source = next(
        (
            column
            for column in (
                "skeletal_muscle_tissue_hu_m29_150_area_cm2",
                "skeletal_muscle_tissue_hu_m29_150_mean_csa_cm2",
            )
            if column in output
        ),
        None,
    )
    output["height_m"] = height
    output["height_source"] = "caller_supplied_measured_height"
    name = "smi_skeletal_muscle_tissue_hu_m29_150_cm2_m2"
    if source is None:
        output[name] = np.nan
        output[f"{name}_valid"] = False
        output[f"{name}_reason"] = "invalid_measurement"
        return output
    values = pd.to_numeric(output[source], errors="coerce")
    source_valid_column = source.removesuffix("_area_cm2") + "_area_valid"
    if source.endswith("_mean_csa_cm2"):
        source_valid_column = source.removesuffix("_mean_csa_cm2") + "_mean_csa_cm2_valid"
    source_valid = (
        output[source_valid_column].fillna(False)
        if source_valid_column in output
        else values.notna()
    )
    valid = source_valid & values.notna()
    output[name] = values / (height**2)
    output[f"{name}_valid"] = valid
    output[f"{name}_reason"] = np.where(valid, None, "invalid_measurement")
    return output


def _add_range_volume_indices(
    table: pd.DataFrame,
    height_m: float | None,
) -> pd.DataFrame:
    height = _validated_height_m(height_m)
    if height is None:
        return table
    output = table.copy()
    output["height_m"] = height
    output["height_source"] = "caller_supplied_measured_height"
    for prefix in (
        "skeletal_muscle_tissue_hu_m29_150",
        "sat_total_hu_m190_m30",
        "vat_total_hu_m190_m30",
    ):
        source = f"{prefix}_volume_cm3"
        if source not in output:
            continue
        name = f"{prefix}_volume_index_cm3_m2"
        values = pd.to_numeric(output[source], errors="coerce")
        valid = (
            output.get(f"{source}_valid", values.notna()).fillna(False)
            & values.notna()
        )
        output[name] = values / (height**2)
        output[f"{name}_valid"] = valid
        output[f"{name}_reason"] = np.where(valid, None, "invalid_measurement")
    return output


def _full_coverage_tolerance(slices: pd.DataFrame) -> float:
    values = pd.to_numeric(
        slices["full_coverage_tolerance"],
        errors="raise",
    ).dropna().unique()
    if len(values) != 1:
        raise ValueError("slices.parquet must contain one full_coverage_tolerance.")
    tolerance = float(values[0])
    if not 0 < tolerance <= 1:
        raise ValueError("full_coverage_tolerance must be in (0, 1].")
    return tolerance


def _identity(slices: pd.DataFrame) -> dict[str, str]:
    identity: dict[str, str] = {}
    for column in IDENTITY_COLUMNS:
        if column not in slices:
            raise ValueError(f"slices.parquet is missing {column}.")
        values = slices[column].dropna().astype(str).unique()
        if len(values) != 1:
            raise ValueError(f"slices.parquet must contain one {column}.")
        identity[column] = str(values[0])
    return identity


def load_measurement_tables(directory: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(directory)
    tables: dict[str, pd.DataFrame] = {}
    schemas = {}
    metadata_by_table: dict[str, dict[bytes, bytes]] = {}
    for name in TABLE_NAMES:
        path = root / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Canonical measurement table not found: {path}.")
        schemas[name] = pq.read_schema(path)
        metadata_by_table[name] = dict(schemas[name].metadata or {})
        tables[name] = pd.read_parquet(path, engine="pyarrow")
    required = {
        "slices": SLICE_REQUIRED_COLUMNS,
        "vertebrae": VERTEBRA_REQUIRED_COLUMNS,
        "summaries": SUMMARY_REQUIRED_COLUMNS,
        "signature": SIGNATURE_REQUIRED_COLUMNS,
        "hu_distributions": HU_DISTRIBUTION_REQUIRED_COLUMNS,
    }
    for name, table in tables.items():
        missing = sorted(required[name] - set(table.columns))
        if missing:
            raise ValueError(f"Canonical {name}.parquet is missing columns: {missing}.")
        expected_schema = canonical_arrow_schema(table)
        observed_schema = schemas[name]
        if observed_schema.names != expected_schema.names or any(
            observed_schema.field(index).type != expected_schema.field(index).type
            for index in range(len(expected_schema))
        ):
            raise ValueError(
                f"Canonical {name}.parquet has inconsistent Arrow field types or order."
            )
    if tables["slices"].empty:
        raise ValueError("Canonical slices.parquet cannot be empty.")
    if len(tables["summaries"]) != 1:
        raise ValueError("Canonical summaries.parquet must contain one row.")

    expected_identity = _identity(tables["slices"])
    if expected_identity["schema_version"] != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported canonical measurement schema_version: "
            f"{expected_identity['schema_version']!r}."
        )
    for name, table in tables.items():
        metadata = metadata_by_table[name]
        if metadata.get(b"bodycomposition.measurement_schema_version") != (
            MEASUREMENT_SCHEMA_VERSION.encode("ascii")
        ):
            raise ValueError(
                f"Canonical {name}.parquet has inconsistent measurement schema metadata."
            )
        if metadata.get(b"bodycomposition.table") != name.encode("ascii"):
            raise ValueError(
                f"Canonical {name}.parquet has missing or inconsistent table metadata."
            )
        for column, expected in expected_identity.items():
            key = f"bodycomposition.{column}".encode("ascii")
            if metadata.get(key) != expected.encode("utf-8"):
                raise ValueError(
                    f"Canonical {name}.parquet has inconsistent {column} metadata."
                )
            if not table.empty:
                values = table[column].dropna().astype(str).unique()
                if len(values) != 1 or str(values[0]) != expected:
                    raise ValueError(
                        f"Canonical {name}.parquet has inconsistent {column}."
                    )

    slices = tables["slices"]
    if slices["slice_id"].duplicated().any():
        raise ValueError("Canonical slices.parquet contains duplicate slice_id values.")
    positions = pd.to_numeric(
        slices["position_superior_mm"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(positions)) or (
        len(positions) > 1 and not np.all(np.diff(positions) > 0)
    ):
        raise ValueError("slices.parquet must be strictly ordered inferior to superior.")
    if not np.array_equal(
        slices["longitudinal_order"].to_numpy(dtype=np.int64),
        np.arange(len(slices), dtype=np.int64),
    ):
        raise ValueError("slices.parquet has inconsistent longitudinal_order values.")

    vertebrae = tables["vertebrae"]
    if vertebrae[["vertebral_level", "territory_bin"]].duplicated().any():
        raise ValueError("vertebrae.parquet contains duplicate level/bin identities.")
    for level, group in vertebrae.groupby("vertebral_level", sort=False):
        observed = group["territory_bin"].to_numpy(dtype=int).tolist()
        expected = list(range(1, BINS_PER_VERTEBRAL_TERRITORY + 1))
        if observed != expected:
            raise ValueError(
                f"vertebrae.parquet has unstable bins for {level!r}: {observed}."
            )
    signature = tables["signature"]
    validate_signature_contract(signature, table_name="signature.parquet")
    validate_hu_distribution_contract(
        tables["hu_distributions"],
        table_name="hu_distributions.parquet",
    )
    return tables


def signature_measurements(directory: str | Path) -> pd.DataFrame:
    """Load the validated fixed-mm longitudinal signature for one case."""

    return load_measurement_tables(directory)["signature"]


def load_signature(directory: str | Path) -> SignatureComponents:
    """Load both canonical components of one body-composition signature."""

    tables = load_measurement_tables(directory)
    return SignatureComponents(
        longitudinal=tables["signature"],
        tissue_hu_distributions=tables["hu_distributions"],
    )


def l3_measurements(
    directory: str | Path,
    *,
    aggregation: str = "territory_mean",
    height_m: float | None = None,
) -> pd.DataFrame:
    tables = load_measurement_tables(directory)
    selected = select_l3_view(
        tables["slices"],
        tables["vertebrae"],
        aggregation=aggregation,
        full_coverage_tolerance=_full_coverage_tolerance(tables["slices"]),
    )
    return _add_l3_height_index(selected, height_m)


def _territories_from_table(table: pd.DataFrame) -> dict[str, VertebralTerritory]:
    territories: dict[str, VertebralTerritory] = {}
    for level, group in table.groupby("vertebral_level", sort=False):
        row = group.iloc[0]
        centroid_columns = (
            "centroid_lps_x_mm",
            "centroid_lps_y_mm",
            "centroid_lps_z_mm",
        )
        centroid = (
            None
            if any(pd.isna(row[column]) for column in centroid_columns)
            else (
                float(row["centroid_lps_x_mm"]),
                float(row["centroid_lps_y_mm"]),
                float(row["centroid_lps_z_mm"]),
            )
        )
        territories[str(level)] = VertebralTerritory(
            native_label=int(row["native_label"]),
            anatomical_label=str(level),
            inferior_mm=(
                None
                if pd.isna(row["territory_inferior_mm"])
                else float(row["territory_inferior_mm"])
            ),
            superior_mm=(
                None
                if pd.isna(row["territory_superior_mm"])
                else float(row["territory_superior_mm"])
            ),
            centroid_lps_xyz=centroid,
            centroid_superior_mm=(
                None
                if pd.isna(row["centroid_superior_mm"])
                else float(row["centroid_superior_mm"])
            ),
            extent_valid=bool(row["vertebral_extent_valid"]),
            complete=bool(row["territory_complete"]),
            sequence_gap_cranial=bool(row["sequence_gap_cranial"]),
            sequence_gap_caudal=bool(row["sequence_gap_caudal"]),
            missing_reason=(
                None
                if pd.isna(row["territory_reason"])
                else str(row["territory_reason"])
            ),
        )
    return territories


def range_measurements(
    directory: str | Path,
    *,
    start_level: str,
    end_level: str,
    allow_partial: bool = False,
    height_m: float | None = None,
) -> pd.DataFrame:
    tables = load_measurement_tables(directory)
    slices = tables["slices"]
    tolerance = _full_coverage_tolerance(slices)
    identity = _identity(slices)
    territories = _territories_from_table(tables["vertebrae"])
    result = aggregate_named_range(
        slices,
        territories,
        start_level,
        end_level,
        allow_partial=allow_partial,
        full_coverage_tolerance=tolerance,
    )
    result.update(
        {
            **identity,
            "aggregation": "physical_range",
        }
    )
    return _add_range_volume_indices(pd.DataFrame([result]), height_m)
