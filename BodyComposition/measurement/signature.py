"""Comparable fixed-millimetre longitudinal body-composition signatures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from BodyComposition.measurement.aggregation import aggregate_physical_range
from BodyComposition.measurement.contracts import (
    REFERENCE_ALIGNMENT_VERSION,
    SIGNATURE_BIN_COUNT,
    SIGNATURE_BIN_WIDTH_MM,
    SIGNATURE_CORE_CHANNELS,
    SIGNATURE_REFERENCE_LEVEL,
    SIGNATURE_SCHEMA_VERSION,
    MeasurementIdentity,
    VertebralExtent,
    VertebralTerritory,
)
from BodyComposition.measurement.physical import interval_overlap_mm


@dataclass(frozen=True)
class ReferenceAlignment:
    """Translation that maps physical superior coordinates to L3-relative mm."""

    valid: bool
    origin_position_superior_mm: float | None
    method: str
    confidence: str
    anchor_levels: tuple[str, ...]
    slope_mm_per_level: float | None
    residual_mm: float | None
    review_required: bool
    variant_sequence: str
    reason: str | None = None


def _normalized_level(level: str) -> str:
    return str(level).upper().replace(" ", "")


def _reference_sequence(levels: set[str]) -> tuple[str, ...]:
    sequence = [
        *(f"C{index}" for index in range(1, 8)),
        *(f"T{index}" for index in range(1, 13)),
    ]
    if "T13" in levels:
        sequence.append("T13")
    sequence.extend(f"L{index}" for index in range(1, 6))
    if "L6" in levels:
        sequence.append("L6")
    return tuple(sequence)


def estimate_reference_alignment(
    extents: Mapping[str, VertebralExtent],
    territories: Mapping[str, VertebralTerritory],
) -> ReferenceAlignment:
    """Estimate the L3 origin without stretching the acquired CT.

    A directly observed valid L3 centroid is authoritative. Otherwise, a
    robust linear relation between native vertebral order and physical
    superior position estimates the latent L3 position. The fitted spacing is
    used only for this translation; tissue profiles retain their measured
    physical length and are never rescaled.
    """

    normalized = {
        _normalized_level(name): extent for name, extent in extents.items()
    }
    observed_levels = set(normalized)
    variant_levels = tuple(
        level for level in ("T13", "L6") if level in observed_levels
    )
    variant_sequence = (
        "+".join(variant_levels) if variant_levels else "standard_visible_sequence"
    )
    gap_review = any(
        territory.sequence_gap_cranial or territory.sequence_gap_caudal
        for territory in territories.values()
    )
    l3 = normalized.get(SIGNATURE_REFERENCE_LEVEL)
    if (
        l3 is not None
        and l3.valid
        and l3.complete
        and l3.centroid_superior_mm is not None
        and np.isfinite(l3.centroid_superior_mm)
    ):
        return ReferenceAlignment(
            valid=True,
            origin_position_superior_mm=float(l3.centroid_superior_mm),
            method="direct_l3",
            confidence="direct",
            anchor_levels=(SIGNATURE_REFERENCE_LEVEL,),
            slope_mm_per_level=None,
            residual_mm=0.0,
            review_required=bool(gap_review or variant_levels),
            variant_sequence=variant_sequence,
        )

    sequence = _reference_sequence(observed_levels)
    index_by_level = {level: index for index, level in enumerate(sequence)}
    reference_index = index_by_level[SIGNATURE_REFERENCE_LEVEL]
    anchors: list[tuple[str, float, float]] = []
    for level, extent in normalized.items():
        if (
            level not in index_by_level
            or not extent.valid
            or not extent.complete
            or extent.centroid_superior_mm is None
            or not np.isfinite(extent.centroid_superior_mm)
        ):
            continue
        anchors.append(
            (
                level,
                float(index_by_level[level] - reference_index),
                float(extent.centroid_superior_mm),
            )
        )
    anchors.sort(key=lambda item: item[1])
    if not anchors:
        return ReferenceAlignment(
            valid=False,
            origin_position_superior_mm=None,
            method="unresolved",
            confidence="unresolved",
            anchor_levels=(),
            slope_mm_per_level=None,
            residual_mm=None,
            review_required=True,
            variant_sequence=variant_sequence,
            reason="missing_anchor",
        )

    if len(anchors) == 1:
        # A single level cannot estimate patient-specific spacing. Retain a
        # low-confidence but deterministic coordinate using the versioned
        # 30-mm nominal pitch; every such row remains explicitly reviewable.
        level, relative_index, position = anchors[0]
        nominal_slope = -30.0
        return ReferenceAlignment(
            valid=True,
            origin_position_superior_mm=float(
                position - nominal_slope * relative_index
            ),
            method="single_anchor_inferred",
            confidence="low",
            anchor_levels=(level,),
            slope_mm_per_level=nominal_slope,
            residual_mm=None,
            review_required=True,
            variant_sequence=variant_sequence,
        )

    pairwise_slopes: list[float] = []
    for left_index, (_, left_x, left_y) in enumerate(anchors[:-1]):
        for _, right_x, right_y in anchors[left_index + 1 :]:
            if right_x != left_x:
                pairwise_slopes.append((right_y - left_y) / (right_x - left_x))
    slope = float(np.median(pairwise_slopes))
    if not np.isfinite(slope) or not -60.0 <= slope <= -10.0:
        return ReferenceAlignment(
            valid=False,
            origin_position_superior_mm=None,
            method="unresolved",
            confidence="unresolved",
            anchor_levels=tuple(level for level, _, _ in anchors),
            slope_mm_per_level=slope if np.isfinite(slope) else None,
            residual_mm=None,
            review_required=True,
            variant_sequence=variant_sequence,
            reason="invalid_extent",
        )

    origins = np.asarray(
        [position - slope * relative_index for _, relative_index, position in anchors],
        dtype=float,
    )
    origin = float(np.median(origins))
    residuals = np.asarray(
        [
            abs(position - (origin + slope * relative_index))
            for _, relative_index, position in anchors
        ],
        dtype=float,
    )
    residual = float(np.median(residuals))
    span = float(anchors[-1][1] - anchors[0][1])
    nearest_anchor_steps = min(abs(relative_index) for _, relative_index, _ in anchors)
    if (
        len(anchors) >= 4
        and span >= 3
        and nearest_anchor_steps <= 3
        and residual <= 5.0
        and not gap_review
    ):
        confidence = "high"
    elif (
        len(anchors) >= 3
        and span >= 2
        and nearest_anchor_steps <= 6
        and residual <= 10.0
    ):
        confidence = "moderate"
    else:
        confidence = "low"
    return ReferenceAlignment(
        valid=True,
        origin_position_superior_mm=origin,
        method="multi_anchor_inferred",
        confidence=confidence,
        anchor_levels=tuple(level for level, _, _ in anchors),
        slope_mm_per_level=slope,
        residual_mm=residual,
        review_required=bool(
            confidence == "low" or gap_review or variant_levels
        ),
        variant_sequence=variant_sequence,
    )


def _dominant_territory(
    inferior_mm: float,
    superior_mm: float,
    territories: Mapping[str, VertebralTerritory],
) -> tuple[str | None, float | None, str]:
    candidates: list[tuple[float, VertebralTerritory]] = []
    for territory in territories.values():
        if territory.inferior_mm is None or territory.superior_mm is None:
            continue
        overlap = float(
            interval_overlap_mm(
                np.asarray([float(territory.inferior_mm)]),
                np.asarray([float(territory.superior_mm)]),
                inferior_mm,
                superior_mm,
            )[0]
        )
        if overlap > 0:
            candidates.append((overlap, territory))
    if not candidates:
        return None, None, "unassigned"
    overlap, territory = max(
        candidates,
        key=lambda item: (
            item[0],
            (
                float(item[1].centroid_superior_mm)
                if item[1].centroid_superior_mm is not None
                else -np.inf
            ),
        ),
    )
    if territory.sequence_gap_cranial or territory.sequence_gap_caudal:
        status = "assigned_sequence_gap_review"
    elif not territory.complete:
        status = "assigned_partial_edge"
    else:
        status = "assigned"
    return (
        territory.anatomical_label,
        float(overlap / (superior_mm - inferior_mm)),
        status,
    )


def _copy_metric(
    row: dict[str, Any],
    aggregate: Mapping[str, Any] | None,
    *,
    source: str,
    destination: str,
    metric: str,
    unresolved_reason: str,
    unresolved_coverage: float = np.nan,
) -> None:
    source_name = f"{source}_{metric}"
    destination_name = f"{destination}_{metric}"
    if aggregate is None:
        row[destination_name] = np.nan
        row[f"{destination_name}_valid"] = False
        row[f"{destination_name}_reason"] = unresolved_reason
        row[f"{destination_name}_coverage_fraction"] = unresolved_coverage
        return
    row[destination_name] = aggregate.get(source_name, np.nan)
    row[f"{destination_name}_valid"] = bool(
        aggregate.get(f"{source_name}_valid", False)
    )
    row[f"{destination_name}_reason"] = aggregate.get(
        f"{source_name}_reason"
    )
    row[f"{destination_name}_coverage_fraction"] = aggregate.get(
        f"{source_name}_coverage_fraction",
        np.nan,
    )


def _add_trunk_fraction(
    row: dict[str, Any],
    *,
    channel: str,
    unresolved_reason: str,
) -> None:
    area_name = f"{channel}_mean_csa_cm2"
    trunk_name = "trunk_mean_csa_cm2"
    output_name = f"{channel}_mean_csa_fraction_of_trunk"
    area_valid = bool(row.get(f"{area_name}_valid", False))
    trunk_valid = bool(row.get(f"{trunk_name}_valid", False))
    area = row.get(area_name)
    trunk = row.get(trunk_name)
    area_coverage = row.get(f"{area_name}_coverage_fraction")
    trunk_coverage = row.get(f"{trunk_name}_coverage_fraction")
    matched_coverage = bool(
        area_coverage is not None
        and trunk_coverage is not None
        and np.isfinite(area_coverage)
        and np.isfinite(trunk_coverage)
        and np.isclose(
            float(area_coverage),
            float(trunk_coverage),
            rtol=0.0,
            atol=1e-9,
        )
    )
    valid = bool(
        area_valid
        and trunk_valid
        and matched_coverage
        and area is not None
        and trunk is not None
        and np.isfinite(area)
        and np.isfinite(trunk)
        and float(trunk) > 0
    )
    row[output_name] = float(area) / float(trunk) if valid else np.nan
    row[f"{output_name}_valid"] = valid
    if valid:
        reason = None
    elif area_valid and trunk_valid and not matched_coverage:
        reason = "invalid_measurement"
    elif trunk_valid and trunk is not None and np.isfinite(trunk) and float(trunk) <= 0:
        reason = "zero_denominator"
    else:
        reason = (
            row.get(f"{area_name}_reason")
            or row.get(f"{trunk_name}_reason")
            or unresolved_reason
        )
    row[f"{output_name}_reason"] = reason


def build_longitudinal_signature(
    slices: pd.DataFrame,
    extents: Mapping[str, VertebralExtent],
    territories: Mapping[str, VertebralTerritory],
    identity: MeasurementIdentity,
    *,
    tissue_profile_id: str,
    extra_tissue_definitions: Sequence[str] = (),
    full_coverage_tolerance: float,
) -> tuple[pd.DataFrame, ReferenceAlignment]:
    """Return the fixed 100 x 20-mm signature and its reference alignment."""

    if slices.empty:
        raise ValueError("The longitudinal signature requires acquired CT slices.")
    if not tissue_profile_id:
        raise ValueError("The longitudinal signature requires a tissue profile ID.")
    alignment = estimate_reference_alignment(extents, territories)
    channels = list(SIGNATURE_CORE_CHANNELS)
    known_destinations = {destination for destination, _ in channels}
    for name in extra_tissue_definitions:
        if name in known_destinations:
            continue
        channels.append((str(name), str(name)))
        known_destinations.add(str(name))

    required_slice_columns = {
        "trunk_area_cm2",
        "trunk_area_valid",
        "trunk_circumference_cm",
        "trunk_contour_valid",
    }
    for _, source in channels:
        required_slice_columns.update(
            {
                f"{source}_area_cm2",
                f"{source}_area_valid",
                f"{source}_mean_hu",
                f"{source}_hu_valid",
                f"{source}_voxel_count",
            }
        )
    missing = sorted(required_slice_columns - set(slices.columns))
    if missing:
        raise ValueError(
            "The slice table is missing signature source columns: "
            f"{missing}."
        )

    rows: list[dict[str, Any]] = []
    unresolved_reason = alignment.reason or "missing_anchor"
    for signature_bin in range(SIGNATURE_BIN_COUNT):
        relative_bin_index = signature_bin - (SIGNATURE_BIN_COUNT // 2)
        center_mm = relative_bin_index * SIGNATURE_BIN_WIDTH_MM
        inferior_relative_mm = center_mm - SIGNATURE_BIN_WIDTH_MM / 2.0
        superior_relative_mm = center_mm + SIGNATURE_BIN_WIDTH_MM / 2.0
        row: dict[str, Any] = {
            **identity.as_columns(),
            "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
            "signature_profile_id": tissue_profile_id,
            "reference_alignment_version": REFERENCE_ALIGNMENT_VERSION,
            "signature_bin": signature_bin,
            "relative_bin_index": relative_bin_index,
            "reference_inferior_mm": inferior_relative_mm,
            "reference_center_mm": center_mm,
            "reference_superior_mm": superior_relative_mm,
            "bin_width_mm": SIGNATURE_BIN_WIDTH_MM,
            "reference_level": SIGNATURE_REFERENCE_LEVEL,
            "reference_origin_position_superior_mm": (
                alignment.origin_position_superior_mm
                if alignment.valid
                else np.nan
            ),
            "reference_alignment_valid": alignment.valid,
            "reference_alignment_method": alignment.method,
            "reference_alignment_confidence": alignment.confidence,
            "reference_alignment_anchor_count": len(alignment.anchor_levels),
            "reference_alignment_anchor_levels": ",".join(alignment.anchor_levels),
            "reference_alignment_slope_mm_per_level": (
                alignment.slope_mm_per_level
                if alignment.slope_mm_per_level is not None
                else np.nan
            ),
            "reference_alignment_residual_mm": (
                alignment.residual_mm
                if alignment.residual_mm is not None
                else np.nan
            ),
            "reference_alignment_review_required": alignment.review_required,
            "reference_variant_sequence": alignment.variant_sequence,
        }
        if alignment.valid:
            origin = float(alignment.origin_position_superior_mm)
            physical_inferior = origin + inferior_relative_mm
            physical_superior = origin + superior_relative_mm
            aggregate = aggregate_physical_range(
                slices,
                physical_inferior,
                physical_superior,
                allow_partial=True,
                full_coverage_tolerance=full_coverage_tolerance,
            )
            coverage = float(aggregate["coverage_fraction"])
            level, level_fraction, assignment_status = _dominant_territory(
                physical_inferior,
                physical_superior,
                territories,
            )
            row.update(
                {
                    "physical_inferior_position_superior_mm": physical_inferior,
                    "physical_superior_position_superior_mm": physical_superior,
                    "coverage_fraction": coverage,
                    "contributing_slice_count": int(
                        aggregate["contributing_slice_count"]
                    ),
                    "bin_valid": coverage > 0,
                    "bin_reason": (
                        None
                        if coverage >= full_coverage_tolerance
                        else ("partial_fov" if coverage > 0 else "outside_fov")
                    ),
                    "dominant_vertebral_level": level,
                    "dominant_vertebral_fraction": (
                        level_fraction if level_fraction is not None else np.nan
                    ),
                    "vertebral_assignment_status": assignment_status,
                }
            )
            metric_aggregate = aggregate if coverage > 0 else None
            metric_unresolved_reason = (
                unresolved_reason if coverage > 0 else "outside_fov"
            )
            metric_unresolved_coverage = 0.0 if coverage <= 0 else np.nan
        else:
            aggregate = None
            metric_aggregate = None
            metric_unresolved_reason = unresolved_reason
            metric_unresolved_coverage = np.nan
            row.update(
                {
                    "physical_inferior_position_superior_mm": np.nan,
                    "physical_superior_position_superior_mm": np.nan,
                    "coverage_fraction": np.nan,
                    "contributing_slice_count": 0,
                    "bin_valid": False,
                    "bin_reason": unresolved_reason,
                    "dominant_vertebral_level": None,
                    "dominant_vertebral_fraction": np.nan,
                    "vertebral_assignment_status": "unresolved_alignment",
                }
            )

        _copy_metric(
            row,
            metric_aggregate,
            source="trunk",
            destination="trunk",
            metric="mean_csa_cm2",
            unresolved_reason=metric_unresolved_reason,
            unresolved_coverage=metric_unresolved_coverage,
        )
        _copy_metric(
            row,
            metric_aggregate,
            source="trunk",
            destination="trunk",
            metric="mean_circumference_cm",
            unresolved_reason=metric_unresolved_reason,
            unresolved_coverage=metric_unresolved_coverage,
        )
        for destination, source in channels:
            _copy_metric(
                row,
                metric_aggregate,
                source=source,
                destination=destination,
                metric="mean_csa_cm2",
                unresolved_reason=metric_unresolved_reason,
                unresolved_coverage=metric_unresolved_coverage,
            )
            _copy_metric(
                row,
                metric_aggregate,
                source=source,
                destination=destination,
                metric="mean_hu",
                unresolved_reason=metric_unresolved_reason,
                unresolved_coverage=metric_unresolved_coverage,
            )
            _add_trunk_fraction(
                row,
                channel=destination,
                unresolved_reason=metric_unresolved_reason,
            )
        rows.append(row)
    return pd.DataFrame(rows), alignment
