"""Exact-overlap vertebral-territory, range, and anthropometric summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from BodyComposition.measurement.contracts import (
    BINS_PER_VERTEBRAL_TERRITORY,
    VERTEBRAL_TERRITORY_SCHEMA_VERSION,
    LandmarkSet,
    MeasurementIdentity,
    VertebralExtent,
    VertebralTerritory,
)
from BodyComposition.measurement.physical import (
    clean_largest_component,
    covered_interval_length_mm,
    interval_overlap_mm,
    mask_physical_extent,
)
from BodyComposition.measurement.tissues import DERIVED_RATIO_DEFINITIONS
from BodyComposition.vertebral.contracts import VertebralResult

DEFAULT_FULL_COVERAGE_TOLERANCE = 0.999
AREA_SUFFIX = "_area_cm2"
HU_SUFFIX = "_mean_hu"


def anatomical_rank(label: str) -> int:
    normalized = label.upper().replace(" ", "")
    if normalized.startswith("C") and normalized[1:].isdigit():
        return int(normalized[1:])
    if normalized.startswith("T") and normalized[1:].isdigit():
        return 100 + int(normalized[1:])
    if normalized.startswith("L") and normalized[1:].isdigit():
        return 200 + int(normalized[1:])
    if normalized == "SACRUM":
        return 300
    if normalized == "COCCYX":
        return 400
    return 1000


def is_thoracic_or_lumbar(label: str) -> bool:
    normalized = label.upper().replace(" ", "")
    return (
        len(normalized) >= 2
        and normalized[0] in {"T", "L"}
        and normalized[1:].isdigit()
    )


def is_vertebral_territory_label(label: str) -> bool:
    normalized = label.upper().replace(" ", "")
    return normalized == "SACRUM" or (
        len(normalized) >= 2
        and normalized[0] in {"C", "T", "L"}
        and normalized[1:].isdigit()
    )


def is_variant_label(label: str) -> bool:
    return label.upper().replace(" ", "") in {"T13", "L6"}


def inferred_anatomical_variants(
    extents: Mapping[str, VertebralExtent],
) -> tuple[str, ...]:
    names = {str(name).upper().replace(" ", "") for name in extents}
    variants = sorted(name for name in names if is_variant_label(name))
    if {"L4", "SACRUM"}.issubset(names) and not names.intersection({"L5", "L6"}):
        variants.append("four_lumbar_or_missing_l5")
    if {"T11", "L1"}.issubset(names) and not names.intersection({"T12", "T13"}):
        variants.append("eleven_thoracic_or_missing_t12")
    return tuple(variants)


def derive_vertebral_extents(
    result: VertebralResult,
    *,
    minimum_component_voxels: int = 20,
    maximum_removed_fraction: float = 0.05,
) -> dict[str, VertebralExtent]:
    """Derive robust physical extents from canonical vertebral-body instances."""

    if result.geometry is None or result.vertebral_body_labels is None:
        raise ValueError("A successful VertebralResult with body labels is required.")
    if minimum_component_voxels <= 0:
        raise ValueError("minimum_component_voxels must be positive.")
    if not 0 <= maximum_removed_fraction < 1:
        raise ValueError("maximum_removed_fraction must be in [0, 1).")

    extents: dict[str, VertebralExtent] = {}
    for native_label in sorted(
        int(value) for value in np.unique(result.vertebral_body_labels) if value != 0
    ):
        anatomical_label = result.label_schema.get(
            native_label,
            f"UNKNOWN_{native_label}",
        )
        if anatomical_label in extents:
            raise ValueError(
                f"Vertebral schema contains duplicate anatomical label {anatomical_label!r}."
            )
        component = clean_largest_component(
            result.vertebral_body_labels == native_label
        )
        retained_enough = component.retained_voxel_count >= minimum_component_voxels
        fragmentation_acceptable = component.removed_fraction <= maximum_removed_fraction
        if retained_enough:
            inferior, superior, centroid_lps, centroid_superior = mask_physical_extent(
                component.mask_zyx,
                result.geometry,
            )
        else:
            inferior = superior = centroid_superior = None
            centroid_lps = None
        if not retained_enough or not fragmentation_acceptable:
            valid = False
            complete = False
            reason = "invalid_extent"
        else:
            valid = True
            complete = not component.touches_fov
            reason = None if complete else "truncated_vertebra"
        extents[anatomical_label] = VertebralExtent(
            native_label=native_label,
            anatomical_label=anatomical_label,
            inferior_mm=inferior,
            superior_mm=superior,
            centroid_lps_xyz=centroid_lps,
            centroid_superior_mm=centroid_superior,
            voxel_count=component.voxel_count,
            retained_voxel_count=component.retained_voxel_count,
            removed_voxel_fraction=component.removed_fraction,
            component_count=component.component_count,
            touches_fov=component.touches_fov,
            complete=complete,
            valid=valid,
            missing_reason=reason,
        )
    return extents


def _sequence_gap(cranial_label: str, caudal_label: str) -> bool:
    cranial = cranial_label.upper().replace(" ", "")
    caudal = caudal_label.upper().replace(" ", "")
    if (
        cranial[0:1] == caudal[0:1]
        and cranial[0:1] in {"C", "T", "L"}
        and cranial[1:].isdigit()
        and caudal[1:].isdigit()
    ):
        return int(caudal[1:]) - int(cranial[1:]) != 1
    if (cranial, caudal) == ("C7", "T1"):
        return False
    if cranial in {"T12", "T13"} and caudal == "L1":
        return False
    return not (cranial in {"L5", "L6"} and caudal == "SACRUM")


def derive_vertebral_territories(
    extents: Mapping[str, VertebralExtent],
) -> dict[str, VertebralTerritory]:
    """Bound native territories at physical midpoints of detected centroids."""

    supported = [
        extent
        for extent in extents.values()
        if is_vertebral_territory_label(extent.anatomical_label)
    ]
    ordered = sorted(
        (
            extent
            for extent in supported
            if extent.valid and extent.centroid_superior_mm is not None
        ),
        key=lambda extent: -float(extent.centroid_superior_mm),
    )
    index_by_level = {
        extent.anatomical_label: index for index, extent in enumerate(ordered)
    }
    territories: dict[str, VertebralTerritory] = {}
    for extent in supported:
        index = index_by_level.get(extent.anatomical_label)
        if index is None:
            territories[extent.anatomical_label] = VertebralTerritory(
                native_label=extent.native_label,
                anatomical_label=extent.anatomical_label,
                inferior_mm=None,
                superior_mm=None,
                centroid_lps_xyz=extent.centroid_lps_xyz,
                centroid_superior_mm=extent.centroid_superior_mm,
                extent_valid=False,
                complete=False,
                missing_reason=extent.missing_reason or "invalid_extent",
            )
            continue
        cranial = ordered[index - 1] if index > 0 else None
        caudal = ordered[index + 1] if index + 1 < len(ordered) else None
        superior_mm = (
            (float(cranial.centroid_superior_mm) + float(extent.centroid_superior_mm))
            / 2.0
            if cranial is not None
            else extent.superior_mm
        )
        inferior_mm = (
            (float(extent.centroid_superior_mm) + float(caudal.centroid_superior_mm))
            / 2.0
            if caudal is not None
            else extent.inferior_mm
        )
        cranial_complete = cranial is not None or extent.anatomical_label == "C1"
        caudal_complete = caudal is not None or extent.anatomical_label == "SACRUM"
        complete = bool(
            extent.complete
            and cranial_complete
            and caudal_complete
            and inferior_mm is not None
            and superior_mm is not None
        )
        if not extent.complete:
            reason = extent.missing_reason or "truncated_vertebra"
        elif not cranial_complete or not caudal_complete:
            reason = "missing_neighbor"
        else:
            reason = None
        territories[extent.anatomical_label] = VertebralTerritory(
            native_label=extent.native_label,
            anatomical_label=extent.anatomical_label,
            inferior_mm=float(inferior_mm) if inferior_mm is not None else None,
            superior_mm=float(superior_mm) if superior_mm is not None else None,
            centroid_lps_xyz=extent.centroid_lps_xyz,
            centroid_superior_mm=extent.centroid_superior_mm,
            extent_valid=extent.valid,
            complete=complete,
            sequence_gap_cranial=bool(
                cranial is not None
                and _sequence_gap(cranial.anatomical_label, extent.anatomical_label)
            ),
            sequence_gap_caudal=bool(
                caudal is not None
                and _sequence_gap(extent.anatomical_label, caudal.anatomical_label)
            ),
            missing_reason=reason,
        )
    return territories


def annotate_slice_anatomy(
    slices: pd.DataFrame,
    territories: Mapping[str, VertebralTerritory],
) -> pd.DataFrame:
    """Assign slice centres for display while retaining exact-overlap aggregation."""

    output = slices.copy()
    assignable = [
        territory
        for territory in territories.values()
        if territory.inferior_mm is not None and territory.superior_mm is not None
    ]
    levels: list[str | None] = []
    statuses: list[str] = []
    bins: list[int | None] = []
    for position in output["position_superior_mm"].to_numpy(dtype=float):
        candidates = [
            territory
            for territory in assignable
            if float(territory.inferior_mm) <= position <= float(territory.superior_mm)
        ]
        if not candidates:
            levels.append(None)
            statuses.append("unassigned_edge" if assignable else "no_valid_territory")
            bins.append(None)
            continue
        territory = min(
            candidates,
            key=lambda item: (
                abs(position - float(item.centroid_superior_mm)),
                -float(item.centroid_superior_mm),
            ),
        )
        levels.append(territory.anatomical_label)
        if not territory.complete:
            statuses.append("assigned_partial_edge")
        elif territory.sequence_gap_cranial or territory.sequence_gap_caudal:
            statuses.append("assigned_sequence_gap_review")
        else:
            statuses.append("assigned")
        relative_from_superior = (
            float(territory.superior_mm) - position
        ) / float(territory.height_mm)
        bins.append(
            int(
                np.clip(
                    np.floor(
                        relative_from_superior * BINS_PER_VERTEBRAL_TERRITORY
                    )
                    + 1,
                    1,
                    BINS_PER_VERTEBRAL_TERRITORY,
                )
            )
        )
    output["assigned_vertebral_level"] = levels
    output["vertebral_assignment_status"] = statuses
    output["vertebral_territory_bin"] = pd.array(bins, dtype="Int64")
    return output


def _metric_coverage(
    table: pd.DataFrame,
    valid: np.ndarray,
    target_lower_mm: float,
    target_upper_mm: float,
) -> float:
    return covered_interval_length_mm(
        table.loc[valid, "slice_slab_inferior_mm"].to_numpy(dtype=float),
        table.loc[valid, "slice_slab_superior_mm"].to_numpy(dtype=float),
        target_lower_mm,
        target_upper_mm,
    ) / (target_upper_mm - target_lower_mm)


def _set_metric(
    output: dict[str, Any],
    name: str,
    value: float | None,
    valid: bool,
    reason: str | None,
    coverage: float,
) -> None:
    output[name] = value if valid else np.nan
    output[f"{name}_valid"] = bool(valid)
    output[f"{name}_reason"] = None if valid else reason
    output[f"{name}_coverage_fraction"] = float(coverage)


def _add_aggregate_composition_ratios(output: dict[str, Any]) -> None:
    """Derive ratios from integrated volumes, never from averaged slice ratios."""

    for name, numerator, denominator in DERIVED_RATIO_DEFINITIONS:
        required = tuple(dict.fromkeys((numerator, *denominator)))
        value_keys = [f"{prefix}_volume_cm3" for prefix in required]
        if any(key not in output for key in value_keys):
            continue
        components_valid = all(
            bool(output.get(f"{prefix}_volume_cm3_valid", False))
            for prefix in required
        )
        numerator_value = output[f"{numerator}_volume_cm3"]
        denominator_values = [
            output[f"{prefix}_volume_cm3"] for prefix in denominator
        ]
        finite = bool(
            np.isfinite(numerator_value)
            and all(np.isfinite(value) for value in denominator_values)
        )
        denominator_value = (
            float(np.sum(denominator_values)) if finite else np.nan
        )
        valid = bool(components_valid and finite and denominator_value > 0)
        value = float(numerator_value / denominator_value) if valid else None
        coverage = min(
            float(output.get(f"{prefix}_volume_cm3_coverage_fraction", 0.0))
            for prefix in required
        )
        reason = None if valid else (
            "zero_denominator"
            if components_valid and finite and denominator_value <= 0
            else "invalid_measurement"
        )
        _set_metric(output, name, value, valid, reason, coverage)


def _validity_column(table: pd.DataFrame, column: str) -> np.ndarray:
    if column not in table:
        return np.ones(len(table), dtype=bool)
    return table[column].fillna(False).to_numpy(dtype=bool)


def aggregate_physical_range(
    slices: pd.DataFrame,
    inferior_mm: float,
    superior_mm: float,
    *,
    allow_partial: bool = False,
    full_coverage_tolerance: float = DEFAULT_FULL_COVERAGE_TOLERANCE,
) -> dict[str, Any]:
    """Aggregate one physical superior-axis interval with exact slab overlap."""

    if not np.isfinite(inferior_mm) or not np.isfinite(superior_mm):
        raise ValueError("Physical range bounds must be finite.")
    if inferior_mm >= superior_mm:
        raise ValueError("Physical range requires inferior_mm < superior_mm.")
    if not 0 < full_coverage_tolerance <= 1:
        raise ValueError("full_coverage_tolerance must be in (0, 1].")
    required = {
        "slice_id",
        "slice_slab_inferior_mm",
        "slice_slab_superior_mm",
        "slice_thickness_normal_mm",
        "normal_mm_per_superior_mm",
    }
    missing = sorted(required - set(slices.columns))
    if missing:
        raise ValueError(f"Slice table is missing physical overlap columns: {missing}.")

    overlap_superior_mm = interval_overlap_mm(
        slices["slice_slab_inferior_mm"].to_numpy(dtype=float),
        slices["slice_slab_superior_mm"].to_numpy(dtype=float),
        inferior_mm,
        superior_mm,
    )
    contributing = overlap_superior_mm > 0
    normal_overlap_mm = overlap_superior_mm * slices[
        "normal_mm_per_superior_mm"
    ].to_numpy(dtype=float)
    target_length_mm = superior_mm - inferior_mm
    normal_scale = slices["normal_mm_per_superior_mm"].to_numpy(dtype=float)
    if not np.allclose(normal_scale, normal_scale[0], rtol=0.0, atol=1e-9):
        raise ValueError(
            "All slices in one prepared CT must use the same normal-to-superior scale."
        )
    observed_length_mm = covered_interval_length_mm(
        slices.loc[contributing, "slice_slab_inferior_mm"].to_numpy(dtype=float),
        slices.loc[contributing, "slice_slab_superior_mm"].to_numpy(dtype=float),
        inferior_mm,
        superior_mm,
    )
    coverage = min(observed_length_mm / target_length_mm, 1.0)
    physically_complete = coverage >= full_coverage_tolerance
    output: dict[str, Any] = {
        "range_inferior_mm": float(inferior_mm),
        "range_superior_mm": float(superior_mm),
        "target_length_mm": float(target_length_mm),
        "observed_length_mm": float(observed_length_mm),
        "target_integration_length_mm": float(target_length_mm * normal_scale[0]),
        "observed_integration_length_mm": float(
            observed_length_mm * normal_scale[0]
        ),
        "coverage_fraction": float(coverage),
        "full_coverage_tolerance": float(full_coverage_tolerance),
        "allow_partial": bool(allow_partial),
        "range_valid": bool(
            physically_complete or (allow_partial and observed_length_mm > 0)
        ),
        "range_missing_reason": (
            None
            if physically_complete
            else ("partial_fov" if observed_length_mm > 0 else "outside_fov")
        ),
        "contributing_slice_count": int(np.count_nonzero(contributing)),
        "contributing_slice_ids": [
            int(value)
            for value in slices.loc[contributing, "slice_id"].to_numpy(dtype=int)
        ],
    }
    strict_range_valid = physically_complete or allow_partial

    def metric_reason(*, invalid: bool = False, empty: bool = False) -> str:
        if not physically_complete and not allow_partial:
            return str(output["range_missing_reason"] or "invalid_measurement")
        if empty:
            return "empty_tissue"
        if invalid:
            return "invalid_measurement"
        return "invalid_measurement"

    area_columns = [column for column in slices if column.endswith(AREA_SUFFIX)]
    for column in area_columns:
        prefix = column.removesuffix(AREA_SUFFIX)
        values = pd.to_numeric(slices[column], errors="coerce").to_numpy(dtype=float)
        value_valid = (
            contributing
            & np.isfinite(values)
            & _validity_column(slices, f"{prefix}_area_valid")
        )
        metric_coverage = _metric_coverage(
            slices,
            value_valid,
            inferior_mm,
            superior_mm,
        )
        metric_valid = bool(
            strict_range_valid
            and np.any(value_valid)
            and (allow_partial or metric_coverage >= full_coverage_tolerance)
        )
        if metric_valid:
            weights = normal_overlap_mm[value_valid]
            mean_csa = float(np.average(values[value_valid], weights=weights))
            volume_cm3 = float(np.sum(values[value_valid] * weights) / 10.0)
        else:
            mean_csa = volume_cm3 = None
        reason = metric_reason(invalid=observed_length_mm > 0)
        _set_metric(
            output,
            f"{prefix}_mean_csa_cm2",
            mean_csa,
            metric_valid,
            reason,
            metric_coverage,
        )
        _set_metric(
            output,
            f"{prefix}_volume_cm3",
            volume_cm3,
            metric_valid,
            reason,
            metric_coverage,
        )

    if "trunk_circumference_cm" in slices:
        values = pd.to_numeric(
            slices["trunk_circumference_cm"],
            errors="coerce",
        ).to_numpy(dtype=float)
        value_valid = (
            contributing
            & np.isfinite(values)
            & _validity_column(slices, "trunk_contour_valid")
        )
        metric_coverage = _metric_coverage(
            slices,
            value_valid,
            inferior_mm,
            superior_mm,
        )
        metric_valid = bool(
            strict_range_valid
            and np.any(value_valid)
            and (allow_partial or metric_coverage >= full_coverage_tolerance)
        )
        value = (
            float(np.average(values[value_valid], weights=normal_overlap_mm[value_valid]))
            if metric_valid
            else None
        )
        _set_metric(
            output,
            "trunk_mean_circumference_cm",
            value,
            metric_valid,
            metric_reason(invalid=observed_length_mm > 0),
            metric_coverage,
        )

    slice_thickness_normal_mm = slices["slice_thickness_normal_mm"].to_numpy(
        dtype=float
    )
    overlap_fraction = np.divide(
        normal_overlap_mm,
        slice_thickness_normal_mm,
        out=np.zeros_like(normal_overlap_mm),
        where=slice_thickness_normal_mm > 0,
    )
    for column in [column for column in slices if column.endswith(HU_SUFFIX)]:
        prefix = column.removesuffix(HU_SUFFIX)
        voxel_column = f"{prefix}_voxel_count"
        if voxel_column not in slices:
            continue
        values = pd.to_numeric(slices[column], errors="coerce").to_numpy(dtype=float)
        counts = pd.to_numeric(slices[voxel_column], errors="coerce").to_numpy(
            dtype=float
        )
        value_valid = (
            contributing
            & np.isfinite(values)
            & np.isfinite(counts)
            & (counts > 0)
            & _validity_column(slices, f"{prefix}_hu_valid")
        )
        invalid_nonempty = contributing & (
            ~np.isfinite(counts) | ((counts > 0) & ~value_valid)
        )
        weights = counts * overlap_fraction
        denominator = float(np.sum(weights[value_valid]))
        metric_coverage = _metric_coverage(
            slices,
            value_valid,
            inferior_mm,
            superior_mm,
        )
        metric_valid = bool(
            strict_range_valid
            and denominator > 0
            and (allow_partial or not np.any(invalid_nonempty))
        )
        value = (
            float(np.sum(values[value_valid] * weights[value_valid]) / denominator)
            if metric_valid
            else None
        )
        _set_metric(
            output,
            column,
            value,
            metric_valid,
            metric_reason(
                invalid=bool(np.any(invalid_nonempty)),
                empty=denominator == 0,
            ),
            metric_coverage,
        )
    _add_aggregate_composition_ratios(output)
    return output


def _null_aggregate_values(output: dict[str, Any], reason: str) -> dict[str, Any]:
    protected = {
        "range_inferior_mm",
        "range_superior_mm",
        "target_length_mm",
        "observed_length_mm",
        "target_integration_length_mm",
        "observed_integration_length_mm",
        "coverage_fraction",
        "full_coverage_tolerance",
        "allow_partial",
        "contributing_slice_count",
        "contributing_slice_ids",
    }
    for key in list(output):
        if key in protected or key.endswith("_coverage_fraction"):
            continue
        if key.endswith("_valid"):
            output[key] = False
        elif key.endswith("_reason"):
            output[key] = reason
        elif isinstance(output[key], (int, float, np.integer, np.floating)):
            output[key] = np.nan
    output["range_valid"] = False
    output["range_missing_reason"] = reason
    return output


def empty_physical_range_aggregate(
    slices: pd.DataFrame,
    *,
    reason: str,
    inferior_mm: float | None = None,
    superior_mm: float | None = None,
    allow_partial: bool = False,
    full_coverage_tolerance: float = DEFAULT_FULL_COVERAGE_TOLERANCE,
) -> dict[str, Any]:
    """Return a stable nullable range schema when physical bounds are absent."""

    if slices.empty:
        raise ValueError("A canonical slice table must contain at least one slice.")
    probe_lower = float(slices["slice_slab_superior_mm"].max()) + 1.0
    template = aggregate_physical_range(
        slices,
        probe_lower,
        probe_lower + 1.0,
        allow_partial=allow_partial,
        full_coverage_tolerance=full_coverage_tolerance,
    )
    template = _null_aggregate_values(template, reason)
    has_bounds = (
        inferior_mm is not None
        and superior_mm is not None
        and np.isfinite(inferior_mm)
        and np.isfinite(superior_mm)
        and inferior_mm < superior_mm
    )
    template.update(
        {
            "range_inferior_mm": float(inferior_mm) if has_bounds else np.nan,
            "range_superior_mm": float(superior_mm) if has_bounds else np.nan,
            "target_length_mm": (
                float(superior_mm - inferior_mm) if has_bounds else np.nan
            ),
            "observed_length_mm": 0.0 if has_bounds else np.nan,
            "target_integration_length_mm": (
                float(superior_mm - inferior_mm)
                * float(slices["normal_mm_per_superior_mm"].iloc[0])
                if has_bounds
                else np.nan
            ),
            "observed_integration_length_mm": 0.0 if has_bounds else np.nan,
            "coverage_fraction": 0.0 if has_bounds else np.nan,
            "allow_partial": bool(allow_partial),
            "range_valid": False,
            "range_missing_reason": reason,
            "contributing_slice_count": 0,
            "contributing_slice_ids": [],
        }
    )
    if not has_bounds:
        for key in tuple(template):
            if key.endswith("_coverage_fraction"):
                template[key] = np.nan
    return template


def aggregate_named_range(
    slices: pd.DataFrame,
    territories: Mapping[str, VertebralTerritory],
    start_level: str,
    end_level: str,
    *,
    allow_partial: bool = False,
    full_coverage_tolerance: float = DEFAULT_FULL_COVERAGE_TOLERANCE,
) -> dict[str, Any]:
    anchors = [territories.get(start_level), territories.get(end_level)]
    valid_anchors = all(
        anchor is not None
        and anchor.inferior_mm is not None
        and anchor.superior_mm is not None
        and (anchor.complete or allow_partial)
        for anchor in anchors
    )
    if not valid_anchors:
        output = empty_physical_range_aggregate(
            slices,
            reason="missing_anchor",
            allow_partial=allow_partial,
            full_coverage_tolerance=full_coverage_tolerance,
        )
    else:
        inferior_mm = min(float(anchor.inferior_mm) for anchor in anchors)
        superior_mm = max(float(anchor.superior_mm) for anchor in anchors)
        output = aggregate_physical_range(
            slices,
            inferior_mm,
            superior_mm,
            allow_partial=allow_partial,
            full_coverage_tolerance=full_coverage_tolerance,
        )
    output.update(
        {
            "range_name": f"{start_level}_{end_level}",
            "start_level": start_level,
            "end_level": end_level,
        }
    )
    return output


def build_vertebra_table(
    slices: pd.DataFrame,
    extents: Mapping[str, VertebralExtent],
    territories: Mapping[str, VertebralTerritory],
    identity: MeasurementIdentity,
    *,
    full_coverage_tolerance: float = DEFAULT_FULL_COVERAGE_TOLERANCE,
) -> pd.DataFrame:
    """Return three physical mean-CSA rows per detected native territory."""

    rows: list[dict[str, Any]] = []
    ordered = sorted(
        territories.values(),
        key=lambda item: (
            -(
                float(item.centroid_superior_mm)
                if item.centroid_superior_mm is not None
                else -np.inf
            ),
            anatomical_rank(item.anatomical_label),
        ),
    )
    for territory in ordered:
        extent = extents[territory.anatomical_label]
        territory_review = (
            territory.sequence_gap_cranial or territory.sequence_gap_caudal
        )
        territory_reason = (
            territory.missing_reason
            if not territory.complete
            else ("sequence_gap" if territory_review else None)
        )
        territory_height = territory.height_mm
        bin_height = (
            territory_height / BINS_PER_VERTEBRAL_TERRITORY
            if territory_height is not None
            else None
        )
        for bin_index in range(1, BINS_PER_VERTEBRAL_TERRITORY + 1):
            base: dict[str, Any] = {
                **identity.as_columns(),
                "vertebral_territory_schema_version": (
                    VERTEBRAL_TERRITORY_SCHEMA_VERSION
                ),
                "native_label": territory.native_label,
                "vertebral_level": territory.anatomical_label,
                "vertebral_extent_valid": territory.extent_valid,
                "vertebral_extent_complete": extent.complete,
                "vertebral_extent_reason": extent.missing_reason,
                "vertebral_body_height_mm": extent.height_mm,
                "vertebral_body_voxel_count": extent.voxel_count,
                "centroid_lps_x_mm": (
                    territory.centroid_lps_xyz[0]
                    if territory.centroid_lps_xyz is not None
                    else np.nan
                ),
                "centroid_lps_y_mm": (
                    territory.centroid_lps_xyz[1]
                    if territory.centroid_lps_xyz is not None
                    else np.nan
                ),
                "centroid_lps_z_mm": (
                    territory.centroid_lps_xyz[2]
                    if territory.centroid_lps_xyz is not None
                    else np.nan
                ),
                "centroid_superior_mm": territory.centroid_superior_mm,
                "anatomical_variant": is_variant_label(territory.anatomical_label),
                "territory_inferior_mm": territory.inferior_mm,
                "territory_superior_mm": territory.superior_mm,
                "territory_height_mm": territory_height,
                "territory_complete": territory.complete,
                "territory_reason": territory_reason,
                "territory_qc_status": (
                    "pass"
                    if territory.complete and not territory_review
                    else ("review" if territory.extent_valid else "fail")
                ),
                "sequence_gap_cranial": territory.sequence_gap_cranial,
                "sequence_gap_caudal": territory.sequence_gap_caudal,
                "territory_bin": bin_index,
                "bins_per_territory": BINS_PER_VERTEBRAL_TERRITORY,
                "aggregation": "vertebral_territory_third_mean",
            }
            if territory_height is not None and bin_height is not None:
                bin_superior = float(
                    territory.superior_mm - (bin_index - 1) * bin_height
                )
                bin_inferior = float(
                    territory.superior_mm - bin_index * bin_height
                )
                aggregate = aggregate_physical_range(
                    slices,
                    bin_inferior,
                    bin_superior,
                    full_coverage_tolerance=full_coverage_tolerance,
                )
                if not territory.complete:
                    aggregate = _null_aggregate_values(
                        aggregate,
                        territory.missing_reason or "invalid_measurement",
                    )
                base.update(
                    {
                        "bin_inferior_mm": bin_inferior,
                        "bin_superior_mm": bin_superior,
                        "bin_height_mm": float(bin_height),
                    }
                )
            else:
                aggregate = empty_physical_range_aggregate(
                    slices,
                    reason=territory.missing_reason or "invalid_extent",
                    full_coverage_tolerance=full_coverage_tolerance,
                )
                base.update(
                    {
                        "bin_inferior_mm": np.nan,
                        "bin_superior_mm": np.nan,
                        "bin_height_mm": np.nan,
                    }
                )
            base["bin_integration_length_mm"] = aggregate.get(
                "target_integration_length_mm",
                np.nan,
            )
            base["bin_valid"] = bool(
                territory.complete and aggregate.get("range_valid", False)
            )
            base["bin_missing_reason"] = (
                None
                if base["bin_valid"]
                else (
                    territory.missing_reason
                    or aggregate.get("range_missing_reason")
                    or "invalid_measurement"
                )
            )
            aggregate = {
                key: value
                for key, value in aggregate.items()
                if not key.endswith("_volume_cm3")
                and key
                not in {
                    "contributing_slice_ids",
                    "allow_partial",
                    "target_integration_length_mm",
                    "observed_integration_length_mm",
                }
            }
            rows.append({**base, **aggregate})
    if rows:
        return pd.DataFrame(rows)

    empty = empty_physical_range_aggregate(
        slices,
        reason="segmentation_failure",
        full_coverage_tolerance=full_coverage_tolerance,
    )
    columns = [
        *identity.as_columns(),
        "vertebral_territory_schema_version",
        "native_label",
        "vertebral_level",
        "vertebral_extent_valid",
        "vertebral_extent_complete",
        "vertebral_extent_reason",
        "vertebral_body_height_mm",
        "vertebral_body_voxel_count",
        "centroid_lps_x_mm",
        "centroid_lps_y_mm",
        "centroid_lps_z_mm",
        "centroid_superior_mm",
        "anatomical_variant",
        "territory_inferior_mm",
        "territory_superior_mm",
        "territory_height_mm",
        "territory_complete",
        "territory_reason",
        "territory_qc_status",
        "sequence_gap_cranial",
        "sequence_gap_caudal",
        "territory_bin",
        "bins_per_territory",
        "aggregation",
        "bin_inferior_mm",
        "bin_superior_mm",
        "bin_height_mm",
        "bin_integration_length_mm",
        "bin_valid",
        "bin_missing_reason",
        *(
            key
            for key in empty
            if not key.endswith("_volume_cm3")
            and key
            not in {
                "contributing_slice_ids",
                "allow_partial",
                "target_integration_length_mm",
                "observed_integration_length_mm",
            }
        ),
    ]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def _nearest_level(
    position_superior_mm: float,
    territories: Mapping[str, VertebralTerritory],
) -> str | None:
    candidates = [
        territory
        for territory in territories.values()
        if territory.centroid_superior_mm is not None
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda territory: abs(
            position_superior_mm - float(territory.centroid_superior_mm)
        ),
    ).anatomical_label


def _extremum(
    slices: pd.DataFrame,
    inferior_mm: float,
    superior_mm: float,
    *,
    kind: str,
    full_coverage_tolerance: float,
) -> dict[str, Any]:
    if kind not in {"minimum", "maximum"}:
        raise ValueError("Extremum kind must be minimum or maximum.")
    overlap = interval_overlap_mm(
        slices["slice_slab_inferior_mm"].to_numpy(dtype=float),
        slices["slice_slab_superior_mm"].to_numpy(dtype=float),
        inferior_mm,
        superior_mm,
    )
    contributing = overlap > 0
    contour_valid = (
        slices["trunk_contour_valid"].fillna(False).to_numpy(dtype=bool)
        & pd.to_numeric(
            slices["trunk_circumference_cm"],
            errors="coerce",
        ).notna().to_numpy(dtype=bool)
    )
    valid_contributing = contributing & contour_valid
    acquisition_coverage = _metric_coverage(
        slices,
        contributing,
        inferior_mm,
        superior_mm,
    )
    valid_coverage = _metric_coverage(
        slices,
        valid_contributing,
        inferior_mm,
        superior_mm,
    )
    common = {
        "search_inferior_mm": float(inferior_mm),
        "search_superior_mm": float(superior_mm),
        "search_target_length_mm": float(superior_mm - inferior_mm),
        "search_acquisition_coverage_fraction": float(acquisition_coverage),
        "search_valid_contour_coverage_fraction": float(valid_coverage),
        "search_slice_count": int(np.count_nonzero(contributing)),
        "search_valid_slice_count": int(np.count_nonzero(valid_contributing)),
    }
    if acquisition_coverage < full_coverage_tolerance:
        return {
            **common,
            "valid": False,
            "eligible": False,
            "reason": "partial_fov" if acquisition_coverage > 0 else "outside_fov",
        }
    if valid_coverage < full_coverage_tolerance:
        return {
            **common,
            "valid": False,
            "eligible": False,
            "reason": "invalid_measurement",
        }
    candidates = slices.loc[valid_contributing]
    index = (
        candidates["trunk_circumference_cm"].idxmin()
        if kind == "minimum"
        else candidates["trunk_circumference_cm"].idxmax()
    )
    selected = slices.loc[index]
    positions = candidates["position_superior_mm"].to_numpy(dtype=float)
    boundary = bool(
        np.isclose(selected["position_superior_mm"], positions.min())
        or np.isclose(selected["position_superior_mm"], positions.max())
    )
    return {
        **common,
        "valid": True,
        "eligible": not boundary,
        "reason": "boundary_extremum" if boundary else None,
        "boundary": boundary,
        "value_cm": float(selected["trunk_circumference_cm"]),
        "position_superior_mm": float(selected["position_superior_mm"]),
        "slice_id": int(selected["slice_id"]),
    }


def build_case_summaries(
    slices: pd.DataFrame,
    extents: Mapping[str, VertebralExtent],
    territories: Mapping[str, VertebralTerritory],
    identity: MeasurementIdentity,
    *,
    landmarks: LandmarkSet | None = None,
    slab_length_mm: float = 200.0,
    full_coverage_tolerance: float = DEFAULT_FULL_COVERAGE_TOLERANCE,
) -> pd.DataFrame:
    row: dict[str, Any] = {**identity.as_columns()}

    l3_extent = extents.get("L3")
    if (
        l3_extent is not None
        and l3_extent.valid
        and l3_extent.complete
        and l3_extent.centroid_superior_mm is not None
    ):
        half_slab = slab_length_mm / 2.0
        slab = aggregate_physical_range(
            slices,
            l3_extent.centroid_superior_mm - half_slab,
            l3_extent.centroid_superior_mm + half_slab,
            full_coverage_tolerance=full_coverage_tolerance,
        )
        slab_reason = slab["range_missing_reason"]
    else:
        slab = empty_physical_range_aggregate(
            slices,
            reason="missing_anchor",
            full_coverage_tolerance=full_coverage_tolerance,
        )
        slab_reason = "missing_anchor"
    row["l3_200mm_slab_valid"] = bool(slab.get("range_valid", False))
    row["l3_200mm_slab_reason"] = slab_reason
    for key, value in slab.items():
        row[f"l3_200mm_{key}"] = value

    l3_territory = territories.get("L3")
    if l3_territory is not None and l3_territory.complete:
        l3_summary = aggregate_physical_range(
            slices,
            float(l3_territory.inferior_mm),
            float(l3_territory.superior_mm),
            full_coverage_tolerance=full_coverage_tolerance,
        )
    else:
        l3_summary = empty_physical_range_aggregate(
            slices,
            reason=(
                l3_territory.missing_reason
                if l3_territory is not None
                else "missing_anchor"
            ),
            full_coverage_tolerance=full_coverage_tolerance,
        )
    row["l3_territory_valid"] = bool(l3_summary["range_valid"])
    row["l3_territory_reason"] = l3_summary["range_missing_reason"]
    for key, value in l3_summary.items():
        row[f"l3_territory_{key}"] = value

    t10 = territories.get("T10")
    l5 = territories.get("L5")
    if t10 is not None and l5 is not None and t10.complete and l5.complete:
        waist = _extremum(
            slices,
            float(l5.inferior_mm),
            float(t10.superior_mm),
            kind="minimum",
            full_coverage_tolerance=full_coverage_tolerance,
        )
    else:
        waist = {"valid": False, "reason": "missing_anchor", "eligible": False}
    prefix = "ct_min_trunk_circumference_t10_l5"
    row[f"{prefix}_cm"] = waist.get("value_cm", np.nan)
    row[f"{prefix}_valid"] = bool(waist.get("valid", False))
    row[f"{prefix}_eligible"] = bool(waist.get("eligible", False))
    row[f"{prefix}_reason"] = waist.get("reason")
    row[f"{prefix}_position_superior_mm"] = waist.get("position_superior_mm", np.nan)
    row[f"{prefix}_slice_id"] = waist.get("slice_id", pd.NA)
    row[f"{prefix}_at_search_boundary"] = bool(waist.get("boundary", False))
    for key in (
        "search_inferior_mm",
        "search_superior_mm",
        "search_target_length_mm",
        "search_acquisition_coverage_fraction",
        "search_valid_contour_coverage_fraction",
        "search_slice_count",
        "search_valid_slice_count",
    ):
        row[f"{prefix}_{key}"] = waist.get(
            key,
            pd.NA if key.endswith("count") else np.nan,
        )
    row[f"{prefix}_native_level"] = (
        _nearest_level(waist["position_superior_mm"], territories)
        if waist.get("valid", False)
        else None
    )

    if (
        landmarks is not None
        and landmarks.lowest_rib_inferior.valid
        and landmarks.iliac_crest_superior.valid
    ):
        target = float(
            (
                landmarks.lowest_rib_inferior.position_superior_mm
                + landmarks.iliac_crest_superior.position_superior_mm
            )
            / 2.0
        )
        valid_slices = slices.loc[
            slices["trunk_contour_valid"]
            & slices["trunk_circumference_cm"].notna()
            & slices["slice_slab_inferior_mm"].le(target)
            & slices["slice_slab_superior_mm"].ge(target)
        ]
        if valid_slices.empty:
            midwaist_valid = False
            acquisition_lower = float(slices["slice_slab_inferior_mm"].min())
            acquisition_upper = float(slices["slice_slab_superior_mm"].max())
            midwaist_reason = (
                "outside_fov"
                if target < acquisition_lower or target > acquisition_upper
                else "invalid_measurement"
            )
        else:
            selected_index = (
                valid_slices["position_superior_mm"] - target
            ).abs().idxmin()
            selected = slices.loc[selected_index]
            midwaist_valid = True
            midwaist_reason = None
            row["ct_midwaist_circumference_cm"] = float(
                selected["trunk_circumference_cm"]
            )
            row["ct_midwaist_position_superior_mm"] = target
            row["ct_midwaist_slice_position_superior_mm"] = float(
                selected["position_superior_mm"]
            )
            row["ct_midwaist_slice_plane_distance_mm"] = abs(
                float(selected["position_superior_mm"]) - target
            )
            row["ct_midwaist_slice_id"] = int(selected["slice_id"])
    else:
        midwaist_valid = False
        if landmarks is None:
            midwaist_reason = "missing_landmark"
        else:
            reasons = {
                landmarks.lowest_rib_inferior.reason,
                landmarks.iliac_crest_superior.reason,
            }
            midwaist_reason = (
                "uncertain_landmark"
                if "uncertain_landmark" in reasons
                else "missing_landmark"
            )
    if not midwaist_valid:
        row["ct_midwaist_circumference_cm"] = np.nan
        row["ct_midwaist_position_superior_mm"] = np.nan
        row["ct_midwaist_slice_position_superior_mm"] = np.nan
        row["ct_midwaist_slice_plane_distance_mm"] = np.nan
        row["ct_midwaist_slice_id"] = pd.NA
    row["ct_midwaist_circumference_valid"] = midwaist_valid
    row["ct_midwaist_circumference_reason"] = midwaist_reason
    row["ct_midwaist_landmark_backend"] = (
        landmarks.lowest_rib_inferior.backend_id if landmarks is not None else None
    )
    for output_name, landmark in (
        (
            "lowest_rib_inferior",
            landmarks.lowest_rib_inferior if landmarks is not None else None,
        ),
        (
            "iliac_crest_superior",
            landmarks.iliac_crest_superior if landmarks is not None else None,
        ),
    ):
        row[f"ct_midwaist_{output_name}_position_superior_mm"] = (
            landmark.position_superior_mm if landmark is not None else np.nan
        )
        row[f"ct_midwaist_{output_name}_valid"] = bool(
            landmark.valid if landmark is not None else False
        )
        row[f"ct_midwaist_{output_name}_reason"] = (
            landmark.reason if landmark is not None else "missing_landmark"
        )
    row["ct_midwaist_lowest_rib_uncertainty_mm"] = (
        landmarks.lowest_rib_inferior.uncertainty_mm
        if landmarks is not None
        else np.nan
    )
    row["ct_midwaist_iliac_crest_uncertainty_mm"] = (
        landmarks.iliac_crest_superior.uncertainty_mm
        if landmarks is not None
        else np.nan
    )

    sacrum = territories.get("SACRUM")
    if sacrum is not None and sacrum.complete:
        pelvic = _extremum(
            slices,
            float(sacrum.inferior_mm),
            float(sacrum.superior_mm),
            kind="maximum",
            full_coverage_tolerance=full_coverage_tolerance,
        )
    else:
        pelvic = {"valid": False, "reason": "missing_anchor", "eligible": False}
    prefix = "ct_max_pelvic_circumference"
    row[f"{prefix}_cm"] = pelvic.get("value_cm", np.nan)
    row[f"{prefix}_valid"] = bool(pelvic.get("valid", False))
    row[f"{prefix}_eligible"] = bool(pelvic.get("eligible", False))
    row[f"{prefix}_reason"] = pelvic.get("reason")
    row[f"{prefix}_position_superior_mm"] = pelvic.get("position_superior_mm", np.nan)
    row[f"{prefix}_slice_id"] = pelvic.get("slice_id", pd.NA)
    row[f"{prefix}_at_search_boundary"] = bool(pelvic.get("boundary", False))
    for key in (
        "search_inferior_mm",
        "search_superior_mm",
        "search_target_length_mm",
        "search_acquisition_coverage_fraction",
        "search_valid_contour_coverage_fraction",
        "search_slice_count",
        "search_valid_slice_count",
    ):
        row[f"{prefix}_{key}"] = pelvic.get(
            key,
            pd.NA if key.endswith("count") else np.nan,
        )
    row["ct_max_pelvic_search_definition"] = "sacral_territory_v1"

    min_ratio_valid = bool(waist.get("eligible", False) and pelvic.get("eligible", False))
    mid_ratio_valid = bool(midwaist_valid and pelvic.get("eligible", False))
    if min_ratio_valid:
        min_ratio_reason = None
    elif not waist.get("valid", False):
        min_ratio_reason = waist.get("reason") or "invalid_measurement"
    elif not waist.get("eligible", False):
        min_ratio_reason = waist.get("reason") or "boundary_extremum"
    elif not pelvic.get("valid", False):
        min_ratio_reason = pelvic.get("reason") or "invalid_measurement"
    else:
        min_ratio_reason = pelvic.get("reason") or "boundary_extremum"
    if mid_ratio_valid:
        mid_ratio_reason = None
    elif not midwaist_valid:
        mid_ratio_reason = midwaist_reason or "invalid_measurement"
    elif not pelvic.get("valid", False):
        mid_ratio_reason = pelvic.get("reason") or "invalid_measurement"
    else:
        mid_ratio_reason = pelvic.get("reason") or "boundary_extremum"
    row["ct_min_waist_to_pelvic_ratio"] = (
        waist["value_cm"] / pelvic["value_cm"] if min_ratio_valid else np.nan
    )
    row["ct_min_waist_to_pelvic_ratio_valid"] = min_ratio_valid
    row["ct_min_waist_to_pelvic_ratio_reason"] = min_ratio_reason
    row["ct_midwaist_to_pelvic_ratio"] = (
        row["ct_midwaist_circumference_cm"] / pelvic["value_cm"]
        if mid_ratio_valid
        else np.nan
    )
    row["ct_midwaist_to_pelvic_ratio_valid"] = mid_ratio_valid
    row["ct_midwaist_to_pelvic_ratio_reason"] = mid_ratio_reason
    return pd.DataFrame([row])


def _centroid_slice_plane_distances_mm(
    slices: pd.DataFrame,
    centroid_lps_xyz: tuple[float, float, float],
) -> np.ndarray:
    centers = slices[
        [
            "slice_center_lps_x_mm",
            "slice_center_lps_y_mm",
            "slice_center_lps_z_mm",
        ]
    ].to_numpy(dtype=float)
    normals = slices[
        ["slice_normal_lps_x", "slice_normal_lps_y", "slice_normal_lps_z"]
    ].to_numpy(dtype=float)
    centroid = np.asarray(centroid_lps_xyz, dtype=float)
    return np.abs(np.einsum("ij,ij->i", centroid - centers, normals))


def select_l3_view(
    slices: pd.DataFrame,
    vertebrae: pd.DataFrame,
    *,
    aggregation: str,
    full_coverage_tolerance: float = DEFAULT_FULL_COVERAGE_TOLERANCE,
) -> pd.DataFrame:
    """Return an in-memory L3 territory mean or conventional centroid slice."""

    if aggregation not in {"territory_mean", "slice"}:
        raise ValueError("aggregation must be 'territory_mean' or 'slice'.")
    identity = {
        column: slices.iloc[0][column]
        for column in ("schema_version", "run_id", "analysis_id", "case_id")
        if column in slices and not slices.empty
    }
    l3 = vertebrae.loc[vertebrae["vertebral_level"].eq("L3")]
    if l3.empty:
        return pd.DataFrame(
            [
                {
                    **identity,
                    "aggregation": aggregation,
                    "valid": False,
                    "reason": "missing_vertebra",
                }
            ]
        )
    first = l3.iloc[0]
    if not bool(first["territory_complete"]):
        return pd.DataFrame(
            [
                {
                    **identity,
                    "aggregation": aggregation,
                    "vertebral_level": "L3",
                    "valid": False,
                    "reason": first["territory_reason"] or "invalid_extent",
                }
            ]
        )
    if aggregation == "territory_mean":
        aggregate = aggregate_physical_range(
            slices,
            float(first["territory_inferior_mm"]),
            float(first["territory_superior_mm"]),
            full_coverage_tolerance=full_coverage_tolerance,
        )
        return pd.DataFrame(
            [
                {
                    **identity,
                    "vertebral_level": "L3",
                    "aggregation": "territory_mean",
                    "valid": bool(aggregate["range_valid"]),
                    "reason": aggregate["range_missing_reason"],
                    **aggregate,
                }
            ]
        )
    centroid = (
        float(first["centroid_lps_x_mm"]),
        float(first["centroid_lps_y_mm"]),
        float(first["centroid_lps_z_mm"]),
    )
    distances = _centroid_slice_plane_distances_mm(slices, centroid)
    output = slices.iloc[[int(np.argmin(distances))]].copy()
    output["aggregation"] = "slice"
    output["vertebral_level"] = "L3"
    output["valid"] = True
    output["reason"] = None
    output["centroid_slice_plane_distance_mm"] = float(np.min(distances))
    return output.reset_index(drop=True)
