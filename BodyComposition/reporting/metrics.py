"""Single reporting aggregation over the authoritative measurement tables."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from BodyComposition.measurement.aggregation import anatomical_rank
from BodyComposition.measurement.contracts import (
    BINS_PER_VERTEBRAL_TERRITORY,
    MeasurementBundle,
)
from BodyComposition.reporting.contracts import (
    MEASUREMENT_DEFINITIONS,
    ReviewEntry,
)
from BodyComposition.vertebral.contracts import QCFlag, VertebralResult

_PATH_FRAGMENT = re.compile(
    r"(?:^|[\s:=])(?:/[^\s;]+|~[/\\][^\s;]+|[A-Za-z]:\\[^\s;]+|file://[^\s;]+|\\\\[^\s;]+)",
    flags=re.IGNORECASE,
)


def public_text(value: Any, *, maximum: int = 120) -> str:
    """Return compact printable text without accepting local path fragments."""

    text = " ".join(str(value).split())
    if _PATH_FRAGMENT.search(text):
        return "Details withheld; inspect the canonical QC artifact."
    text = "".join(character if 32 <= ord(character) < 127 else "?" for character in text)
    return text[:maximum].rstrip() or "Unspecified canonical review finding"


def _first_reason(group: pd.DataFrame, column: str, fallback: str) -> str:
    reason_column = f"{column}_reason"
    if reason_column in group:
        reasons = group[reason_column].dropna().astype(str)
        if not reasons.empty:
            return public_text(reasons.iloc[0], maximum=60)
    missing = group.get("bin_missing_reason")
    if missing is not None:
        reasons = missing.dropna().astype(str)
        if not reasons.empty:
            return public_text(reasons.iloc[0], maximum=60)
    return fallback


def _metric_value(
    group: pd.DataFrame,
    name: str,
) -> tuple[float | None, bool, str | None, dict[str, Any]]:
    definition = MEASUREMENT_DEFINITIONS[name]
    source = definition["source"]
    audit: dict[str, Any] = {
        "source_column": source,
        "bin_values": [],
        "bin_integration_length_mm": [],
        "aggregation": "integration_length_weighted_mean",
    }
    if source not in group:
        return None, False, "invalid_measurement", audit
    weights = pd.to_numeric(
        group["bin_integration_length_mm"], errors="coerce"
    ).to_numpy(dtype=float)
    values = pd.to_numeric(group[source], errors="coerce").to_numpy(dtype=float)
    audit["bin_values"] = [float(value) if math.isfinite(value) else None for value in values]
    audit["bin_integration_length_mm"] = [
        float(value) if math.isfinite(value) else None for value in weights
    ]
    valid_column = f"{source}_valid"
    source_valid = (
        group[valid_column].fillna(False).astype(bool).to_numpy()
        if valid_column in group
        else np.isfinite(values)
    )
    bin_valid = group["bin_valid"].fillna(False).astype(bool).to_numpy()
    base_valid = (
        len(group) == BINS_PER_VERTEBRAL_TERRITORY
        and np.all(bin_valid)
        and np.all(np.isfinite(weights))
        and np.all(weights > 0)
    )

    if definition["kind"] == "hu":
        area_source = definition["weight_source"]
        if area_source not in group:
            return None, False, "invalid_measurement", audit
        areas = pd.to_numeric(group[area_source], errors="coerce").to_numpy(dtype=float)
        area_valid_column = f"{area_source}_valid"
        area_valid = (
            group[area_valid_column].fillna(False).astype(bool).to_numpy()
            if area_valid_column in group
            else np.isfinite(areas)
        )
        volume_weights = areas * weights
        audit["aggregation"] = "tissue_volume_weighted_mean"
        audit["weight_source_column"] = area_source
        audit["weight_source_bin_values"] = [
            float(value) if math.isfinite(value) else None for value in areas
        ]
        if (
            not base_valid
            or not np.all(area_valid)
            or not np.all(np.isfinite(areas))
            or np.any(areas < 0)
        ):
            return None, False, _first_reason(group, source, "invalid_measurement"), audit
        nonempty = areas > 0
        if not np.any(nonempty):
            return None, False, "empty_tissue", audit
        if not np.all(source_valid[nonempty]) or not np.all(np.isfinite(values[nonempty])):
            return None, False, _first_reason(group, source, "invalid_measurement"), audit
        value = float(np.average(values[nonempty], weights=volume_weights[nonempty]))
    else:
        valid = base_valid and np.all(source_valid) and np.all(np.isfinite(values))
        if not valid:
            return None, False, _first_reason(group, source, "invalid_measurement"), audit
        value = float(np.average(values, weights=weights))
    audit["unrounded_value"] = value
    return value, True, None, audit


def aggregate_vertebral_measurements(
    bundle: MeasurementBundle,
    measurement_columns: Iterable[str],
) -> list[dict[str, Any]]:
    """Reconstruct one auditable whole-territory row from each group of three."""

    columns = tuple(measurement_columns)
    unsupported = sorted(set(columns) - set(MEASUREMENT_DEFINITIONS))
    if unsupported:
        raise ValueError(f"Unsupported reporting measurements: {unsupported}.")
    rows: list[dict[str, Any]] = []
    for level, group in bundle.vertebrae.groupby("vertebral_level", sort=False):
        group = group.sort_values("territory_bin", kind="stable")
        bins = group["territory_bin"].to_numpy(dtype=int).tolist()
        if bins != list(range(1, BINS_PER_VERTEBRAL_TERRITORY + 1)):
            raise ValueError(f"Vertebral territory {level!r} does not contain canonical bins 1..3.")
        first = group.iloc[0]
        metrics: dict[str, Any] = {}
        audit: dict[str, Any] = {}
        for column in columns:
            value, valid, reason, source_audit = _metric_value(group, column)
            metrics[column] = {"value": value, "valid": valid, "reason": reason}
            audit[column] = source_audit
        rows.append(
            {
                "vertebral_level": str(level),
                "native_label": int(first["native_label"]),
                "centroid_superior_mm": (
                    float(first["centroid_superior_mm"])
                    if pd.notna(first["centroid_superior_mm"])
                    else None
                ),
                "centroid_lps_xyz": [
                    float(first[name]) if pd.notna(first[name]) else None
                    for name in ("centroid_lps_x_mm", "centroid_lps_y_mm", "centroid_lps_z_mm")
                ],
                "territory_inferior_mm": (
                    float(first["territory_inferior_mm"])
                    if pd.notna(first["territory_inferior_mm"])
                    else None
                ),
                "territory_superior_mm": (
                    float(first["territory_superior_mm"])
                    if pd.notna(first["territory_superior_mm"])
                    else None
                ),
                "territory_complete": bool(first["territory_complete"]),
                "territory_qc_status": str(first["territory_qc_status"]),
                "territory_reason": (
                    public_text(first["territory_reason"], maximum=60)
                    if pd.notna(first["territory_reason"])
                    else None
                ),
                "anatomical_variant": bool(first["anatomical_variant"]),
                "metrics": metrics,
                "source_audit": audit,
            }
        )
    return sorted(rows, key=lambda row: anatomical_rank(row["vertebral_level"]))


def _orientation_mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    raise TypeError("Orientation report must be an OrientationResult or mapping.")


def _qc_entry(case_id: str, domain: str, flag: QCFlag) -> ReviewEntry:
    return ReviewEntry(
        case_id=case_id,
        domain=domain,
        code=public_text(flag.code, maximum=50),
        reason=public_text(flag.reason),
        automatic_action="Processing completed; inspect the canonical QC artifact.",
        review_status=public_text(flag.adjudication or "not_adjudicated", maximum=40),
    )


def collect_review_entries(
    case_id: str,
    orientation: Any,
    vertebral_result: VertebralResult,
    bundle: MeasurementBundle,
    extra: Iterable[ReviewEntry] = (),
) -> tuple[ReviewEntry, ...]:
    """Copy canonical review state into concise report index entries."""

    entries: list[ReviewEntry] = []
    orientation_data = _orientation_mapping(orientation)
    if bool(orientation_data.get("manual_review_required", False)):
        state = public_text(orientation_data.get("state", "orientation_review"), maximum=50)
        changed = bool(orientation_data.get("orientation_changed", False))
        action = (
            "Lossless orientation transform applied."
            if changed
            else "Continued without orientation repair."
        )
        flags = orientation_data.get("review_flags", [])
        if not isinstance(flags, list):
            flags = []
        if not flags:
            flags = [{"code": state, "reason": state}]
        for flag in flags:
            if not isinstance(flag, Mapping):
                continue
            entries.append(
                ReviewEntry(
                    case_id=case_id,
                    domain="orientation",
                    code=public_text(flag.get("code", state), maximum=50),
                    reason=public_text(flag.get("reason", state)),
                    automatic_action=action,
                    review_status=public_text(flag.get("adjudication") or "not_adjudicated", maximum=40),
                )
            )
    entries.extend(
        _qc_entry(case_id, "vertebral", flag)
        for flag in vertebral_result.qc_flags
        if flag.stage != "orientation"
    )
    entries.extend(
        _qc_entry(case_id, "measurement", flag)
        for flag in bundle.qc_flags
        if flag.stage not in {"orientation", "vertebral"}
        and flag.severity.value in {"warning", "error"}
    )
    entries.extend(extra)
    deduplicated: list[ReviewEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        identity = (entry.domain, entry.code, entry.reason)
        if identity not in seen:
            seen.add(identity)
            deduplicated.append(entry)
    return tuple(deduplicated)
