"""Canonical one-row-per-prepared-slice measurements."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from BodyComposition.measurement.contours import external_contour_measurement
from BodyComposition.measurement.contracts import BodySurfaceResult, MeasurementIdentity
from BodyComposition.measurement.physical import slice_geometry_table, validate_array_zyx
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain


TISSUE_NAME_ALIASES = {
    "sm": "sm",
    "skeletalmuscle": "sm",
    "bone": "bone",
    "sat": "sat",
    "avat": "avat",
    "tvat": "tvat",
    "vat": "vat",
    "heart": "heart",
    "lung": "lung",
    "imat": "imat",
    "psoas": "psoas",
}


def canonical_tissue_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return TISSUE_NAME_ALIASES.get(normalized, normalized)


def _validated_label_schema(labels: Mapping[int, str]) -> dict[int, str]:
    output: dict[int, str] = {}
    for label, name in labels.items():
        if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
            raise ValueError("Tissue label identifiers must be positive integers.")
        canonical = canonical_tissue_name(name)
        if not canonical:
            raise ValueError(f"Tissue label {label} has an empty name.")
        if canonical in output.values():
            raise ValueError(f"Tissue label name {canonical!r} is duplicated.")
        output[label] = canonical
    if not output:
        raise ValueError("At least one tissue label is required.")
    return output


def _internal_support_gaps(voxel_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(voxel_counts, dtype=np.int64)
    gaps = np.zeros(counts.shape, dtype=bool)
    nonempty = np.flatnonzero(counts > 0)
    if nonempty.size > 1:
        first = int(nonempty[0])
        last = int(nonempty[-1])
        gaps[first : last + 1] = counts[first : last + 1] == 0
    return gaps


def _add_tissue_measurements(
    table: pd.DataFrame,
    *,
    name: str,
    tissue_mask_zyx: np.ndarray,
    image_zyx: np.ndarray,
    body_mask_zyx: np.ndarray,
    storage_indices_z: np.ndarray,
    pixel_area_cm2: float,
) -> None:
    counts = np.count_nonzero(tissue_mask_zyx, axis=(1, 2)).astype(np.int64)
    inside_body = np.count_nonzero(
        tissue_mask_zyx & body_mask_zyx,
        axis=(1, 2),
    ).astype(np.int64)
    outside_body = counts - inside_body
    area_valid = outside_body == 0
    hu_sum = np.where(tissue_mask_zyx, image_zyx, 0.0).sum(axis=(1, 2), dtype=float)
    mean_hu = np.full(counts.size, np.nan, dtype=float)
    nonempty = counts > 0
    mean_hu[nonempty] = hu_sum[nonempty] / counts[nonempty]

    selected_counts = counts[storage_indices_z]
    selected_nonempty = nonempty[storage_indices_z]
    selected_area_valid = area_valid[storage_indices_z]
    table[f"{name}_voxel_count"] = selected_counts
    table[f"{name}_area_cm2"] = selected_counts.astype(float) * pixel_area_cm2
    table[f"{name}_area_valid"] = selected_area_valid
    table[f"{name}_area_reason"] = np.where(
        selected_area_valid,
        None,
        "invalid_measurement",
    )
    table[f"{name}_mean_hu"] = mean_hu[storage_indices_z]
    hu_valid = selected_nonempty & selected_area_valid
    table[f"{name}_hu_valid"] = hu_valid
    table[f"{name}_hu_reason"] = np.where(
        hu_valid,
        None,
        np.where(selected_nonempty, "invalid_measurement", "empty_tissue"),
    )


def calculate_canonical_slice_measurements(
    image_zyx: np.ndarray,
    tissue_labels_zyx: np.ndarray,
    geometry: ImageGeometry,
    tissue_label_schema: Mapping[int, str],
    body_surface: BodySurfaceResult,
    identity: MeasurementIdentity,
    *,
    tissue_backend_id: str,
    tissue_preprocessing: Mapping[str, Any] | None = None,
    orientation_changed: bool = False,
) -> pd.DataFrame:
    """Calculate physical CSA, pooled-HU primitives, and trunk contour per slice.

    Backend and preprocessing provenance are accepted for analysis identity and
    QC, but are intentionally not repeated in every Parquet row.
    """

    del tissue_backend_id, tissue_preprocessing
    image = validate_array_zyx(image_zyx, geometry, "image_zyx")
    labels = validate_array_zyx(tissue_labels_zyx, geometry, "tissue_labels_zyx")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Tissue labels must be integer-valued.")
    assert_same_physical_domain(
        geometry,
        body_surface.geometry,
        reference_name="prepared CT",
        candidate_name="body-surface masks",
    )
    schema = _validated_label_schema(tissue_label_schema)
    unknown = sorted(
        int(value)
        for value in np.unique(labels)
        if value != 0 and int(value) not in schema
    )
    if unknown:
        raise ValueError(f"Tissue mask contains labels outside its schema: {unknown}.")

    table = slice_geometry_table(geometry).drop(columns=["original_storage_index"])
    del orientation_changed
    for column, value in identity.as_columns().items():
        table[column] = value

    storage = table["slice_index_zyx_z"].to_numpy(dtype=int)
    pixel_area_cm2 = float(geometry.in_plane_area_mm2 / 100.0)
    body_counts = np.count_nonzero(body_surface.body_mask_zyx, axis=(1, 2)).astype(np.int64)
    trunk_counts = np.count_nonzero(body_surface.trunk_mask_zyx, axis=(1, 2)).astype(np.int64)
    body_internal_gaps = _internal_support_gaps(body_counts)
    trunk_internal_gaps = _internal_support_gaps(trunk_counts)
    table["body_voxel_count"] = body_counts[storage]
    table["trunk_voxel_count"] = trunk_counts[storage]
    table["body_mask_internal_gap"] = body_internal_gaps[storage]
    table["trunk_mask_internal_gap"] = trunk_internal_gaps[storage]
    table["trunk_area_cm2"] = trunk_counts[storage].astype(float) * pixel_area_cm2

    trunk_perimeters: list[float | None] = []
    trunk_components: list[int] = []
    trunk_closed: list[bool] = []
    trunk_touching: list[bool] = []
    trunk_valid: list[bool] = []
    trunk_reasons: list[str | None] = []
    for slice_index_z in storage:
        contour = external_contour_measurement(
            body_surface.trunk_mask_zyx[slice_index_z],
            geometry,
            int(slice_index_z),
            component_mode="primary",
        )
        trunk_perimeters.append(contour.perimeter_cm)
        trunk_components.append(contour.component_count)
        trunk_closed.append(contour.contour_closed)
        trunk_touching.append(contour.touching_image_boundary)
        trunk_valid.append(contour.valid)
        trunk_reasons.append(contour.reason)

    fragmented = np.asarray(trunk_components, dtype=int) > 1
    touching = np.asarray(trunk_touching, dtype=bool)
    nonempty_trunk = trunk_counts[storage] > 0
    area_valid = (
        nonempty_trunk
        & ~touching
        & ~fragmented
        & ~trunk_internal_gaps[storage]
    )
    contour_valid = np.asarray(trunk_valid, dtype=bool) & ~fragmented
    table["trunk_area_valid"] = area_valid
    table["trunk_area_reason"] = np.where(
        area_valid,
        None,
        np.where(
            ~nonempty_trunk,
            "empty_mask",
            np.where(
                touching,
                "touching_image_boundary",
                np.where(fragmented, "fragmented_mask", "invalid_measurement"),
            ),
        ),
    )
    table["trunk_circumference_cm"] = pd.array(trunk_perimeters, dtype="Float64")
    table["trunk_contour_valid"] = contour_valid
    table["trunk_contour_reason"] = [
        reason if reason is not None else ("fragmented_mask" if is_fragmented else None)
        for reason, is_fragmented in zip(trunk_reasons, fragmented, strict=True)
    ]
    table["trunk_contour_component_count"] = trunk_components
    table["trunk_contour_closed"] = trunk_closed
    table["trunk_touching_fov"] = touching
    table["trunk_mask_fragmented"] = fragmented

    tissue_masks: dict[str, np.ndarray] = {}
    for label, name in schema.items():
        tissue_mask = labels == label
        tissue_masks[name] = tissue_mask
        _add_tissue_measurements(
            table,
            name=name,
            tissue_mask_zyx=tissue_mask,
            image_zyx=image,
            body_mask_zyx=body_surface.body_mask_zyx,
            storage_indices_z=storage,
            pixel_area_cm2=pixel_area_cm2,
        )

    if "avat" in tissue_masks and "tvat" in tissue_masks:
        total_vat_mask = tissue_masks["avat"] | tissue_masks["tvat"]
        table["total_vat_source"] = "avat_plus_tvat"
    elif "vat" in tissue_masks:
        total_vat_mask = tissue_masks["vat"]
        table["total_vat_source"] = "native_vat"
    else:
        total_vat_mask = None
        table["total_vat_source"] = "unavailable"
    if total_vat_mask is not None:
        _add_tissue_measurements(
            table,
            name="total_vat",
            tissue_mask_zyx=total_vat_mask,
            image_zyx=image,
            body_mask_zyx=body_surface.body_mask_zyx,
            storage_indices_z=storage,
            pixel_area_cm2=pixel_area_cm2,
        )
    else:
        table["total_vat_voxel_count"] = pd.array([pd.NA] * len(table), dtype="Int64")
        table["total_vat_area_cm2"] = np.nan
        table["total_vat_area_valid"] = False
        table["total_vat_area_reason"] = "invalid_measurement"
        table["total_vat_mean_hu"] = np.nan
        table["total_vat_hu_valid"] = False
        table["total_vat_hu_reason"] = "invalid_measurement"

    segmented_counts = np.count_nonzero(labels, axis=(1, 2)).astype(np.int64)
    tissue_outside_body = (labels != 0) & ~body_surface.body_mask_zyx
    outside_counts = np.count_nonzero(tissue_outside_body, axis=(1, 2)).astype(np.int64)
    total_valid = outside_counts[storage] == 0
    table["total_segmented_tissue_voxel_count"] = segmented_counts[storage]
    table["total_segmented_tissue_area_cm2"] = (
        segmented_counts[storage].astype(float) * pixel_area_cm2
    )
    table["total_segmented_tissue_area_valid"] = total_valid
    table["total_segmented_tissue_area_reason"] = np.where(
        total_valid,
        None,
        "invalid_measurement",
    )
    table["tissue_outside_body_voxel_count"] = outside_counts[storage]
    table["tissue_exceeds_body_area"] = ~total_valid
    table["body_surface_invalid"] = body_internal_gaps[storage]
    table["trunk_surface_invalid"] = ~area_valid | ~contour_valid
    table["slice_measurement_valid"] = ~(~total_valid | table["trunk_surface_invalid"])
    table["slice_qc_status"] = np.where(table["slice_measurement_valid"], "pass", "review")
    return table.copy()


def annotate_longitudinal_circumference_qc(
    slices: pd.DataFrame,
    *,
    maximum_gap_mm: float,
    minimum_absolute_jump_cm: float,
    minimum_relative_jump: float,
) -> pd.DataFrame:
    """Flag abrupt adjacent-slice circumference changes using frozen thresholds."""

    if maximum_gap_mm <= 0 or minimum_absolute_jump_cm <= 0 or minimum_relative_jump <= 0:
        raise ValueError("Longitudinal circumference-QC thresholds must be positive.")
    output = slices.copy()
    circumference = pd.to_numeric(output["trunk_circumference_cm"], errors="coerce")
    position = pd.to_numeric(output["position_superior_mm"], errors="coerce")
    gap = position.diff().abs()
    signed_delta = circumference.diff()
    absolute_jump = signed_delta.abs()
    previous = circumference.shift(1).abs()
    denominator = pd.concat((circumference.abs(), previous), axis=1).min(axis=1)
    relative_jump = absolute_jump / denominator.where(denominator > 0)
    adjacent_valid = output["trunk_contour_valid"] & output["trunk_contour_valid"].shift(
        1,
        fill_value=False,
    )
    flagged = (
        adjacent_valid
        & gap.le(maximum_gap_mm)
        & absolute_jump.ge(minimum_absolute_jump_cm)
        & relative_jump.ge(minimum_relative_jump)
    )
    output["trunk_circumference_adjacent_gap_mm"] = gap
    output["trunk_circumference_adjacent_delta_cm"] = signed_delta
    output["trunk_circumference_adjacent_jump_cm"] = absolute_jump
    output["trunk_circumference_adjacent_jump_fraction"] = relative_jump
    output["trunk_circumference_jump_flag"] = flagged.fillna(False)
    output.loc[output["trunk_circumference_jump_flag"], "slice_qc_status"] = "review"
    return output
