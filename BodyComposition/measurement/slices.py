"""Canonical one-row-per-prepared-slice measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from BodyComposition.measurement.contours import external_contour_measurement
from BodyComposition.measurement.contracts import BodySurfaceResult, MeasurementIdentity
from BodyComposition.measurement.physical import slice_geometry_table, validate_array_zyx
from BodyComposition.measurement.tissues import (
    CORE_COMPARTMENT_NAMES,
    DERIVED_RATIO_DEFINITIONS,
    canonical_compartment_name,
    iter_configured_tissue_masks,
)
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain


def _validated_label_schema(labels: Mapping[int, str]) -> dict[int, str]:
    output: dict[int, str] = {}
    for label, name in labels.items():
        if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
            raise ValueError("Label identifiers must be positive integers.")
        canonical = canonical_compartment_name(name)
        if not canonical:
            raise ValueError(f"Label {label} has an empty name.")
        output[label] = canonical
    if not output:
        raise ValueError("At least one label is required.")
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


def _mask_measurement_columns(
    *,
    name: str,
    mask_zyx: np.ndarray,
    image_zyx: np.ndarray,
    body_mask_zyx: np.ndarray,
    storage_indices_z: np.ndarray,
    pixel_area_cm2: float,
    empty_reason: str,
) -> dict[str, Any]:
    counts = np.count_nonzero(mask_zyx, axis=(1, 2)).astype(np.int64)
    inside_body = np.sum(
        body_mask_zyx,
        axis=(1, 2),
        dtype=np.int64,
        where=mask_zyx,
    )
    outside_body = counts - inside_body
    area_valid = outside_body == 0
    hu_sum = np.sum(
        image_zyx,
        axis=(1, 2),
        dtype=float,
        where=mask_zyx,
    )
    mean_hu = np.full(counts.size, np.nan, dtype=float)
    nonempty = counts > 0
    mean_hu[nonempty] = hu_sum[nonempty] / counts[nonempty]

    selected_counts = counts[storage_indices_z]
    selected_nonempty = nonempty[storage_indices_z]
    selected_area_valid = area_valid[storage_indices_z]
    hu_valid = selected_nonempty & selected_area_valid
    return {
        f"{name}_voxel_count": selected_counts,
        f"{name}_area_cm2": selected_counts.astype(float) * pixel_area_cm2,
        f"{name}_area_valid": selected_area_valid,
        f"{name}_area_reason": np.where(
            selected_area_valid,
            None,
            "invalid_measurement",
        ),
        f"{name}_mean_hu": mean_hu[storage_indices_z],
        f"{name}_hu_valid": hu_valid,
        f"{name}_hu_reason": np.where(
            hu_valid,
            None,
            np.where(selected_nonempty, "invalid_measurement", empty_reason),
        ),
    }


def _slice_composition_ratio_columns(
    table: pd.DataFrame,
    *,
    name: str,
    numerator: str,
    denominator: tuple[str, ...],
) -> dict[str, Any]:
    required = (numerator, *denominator)
    if any(f"{prefix}_voxel_count" not in table for prefix in required):
        return {}
    numerator_count = pd.to_numeric(
        table[f"{numerator}_voxel_count"],
        errors="coerce",
    ).to_numpy(dtype=float)
    denominator_counts = [
        pd.to_numeric(table[f"{prefix}_voxel_count"], errors="coerce").to_numpy(dtype=float)
        for prefix in denominator
    ]
    denominator_count = np.sum(denominator_counts, axis=0)
    components_valid = np.ones(len(table), dtype=bool)
    for prefix in dict.fromkeys(required):
        components_valid &= table[f"{prefix}_area_valid"].fillna(False).to_numpy(dtype=bool)
    valid = (
        components_valid
        & np.isfinite(numerator_count)
        & np.isfinite(denominator_count)
        & (denominator_count > 0)
    )
    value = np.full(len(table), np.nan, dtype=float)
    value[valid] = numerator_count[valid] / denominator_count[valid]
    return {
        name: value,
        f"{name}_valid": valid,
        f"{name}_reason": np.where(
            valid,
            None,
            np.where(components_valid, "zero_denominator", "invalid_measurement"),
        ),
    }


def calculate_canonical_slice_measurements(
    image_zyx: np.ndarray,
    tissue_labels_zyx: np.ndarray,
    geometry: ImageGeometry,
    tissue_label_schema: Mapping[int, str],
    body_surface: BodySurfaceResult,
    identity: MeasurementIdentity,
    *,
    compartment_labels_zyx: np.ndarray,
    compartment_label_schema: Mapping[int, str],
    tissue_definitions: Mapping[str, Mapping[str, Any]] | None = None,
    analysis_scope: str = "full_ct",
    analyzed_slices_z: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate physical CSA, pooled-HU primitives, and trunk contour per slice."""

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
        int(value) for value in np.unique(labels) if value != 0 and int(value) not in schema
    )
    if unknown:
        raise ValueError(f"Tissue mask contains labels outside its schema: {unknown}.")
    compartment_labels = validate_array_zyx(
        compartment_labels_zyx,
        geometry,
        "compartment_labels_zyx",
    )
    if not np.issubdtype(compartment_labels.dtype, np.integer):
        raise TypeError("Compartment labels must be integer-valued.")
    compartment_schema = _validated_label_schema(compartment_label_schema)
    unknown_compartments = sorted(
        int(value)
        for value in np.unique(compartment_labels)
        if value != 0 and int(value) not in compartment_schema
    )
    if unknown_compartments:
        raise ValueError(
            f"Compartment mask contains labels outside its schema: {unknown_compartments}."
        )
    if analysis_scope not in {"full_ct", "l3_vertebral_level"}:
        raise ValueError(f"Unknown analysis scope {analysis_scope!r}.")
    analysis_available_z: np.ndarray
    if analyzed_slices_z is None:
        analysis_available_z = np.ones(geometry.size_xyz[2], dtype=bool)
    else:
        analysis_available_z = np.asarray(analyzed_slices_z)
        if analysis_available_z.ndim != 1 or analysis_available_z.shape != (geometry.size_xyz[2],):
            raise ValueError("analyzed_slices_z must contain one boolean per array-z slice.")
        if analysis_available_z.dtype != np.bool_:
            raise TypeError("analyzed_slices_z must be a boolean array.")
        analysis_available_z = np.array(analysis_available_z, copy=True)
    if analysis_scope == "full_ct" and not bool(analysis_available_z.all()):
        raise ValueError("full_ct analysis requires every acquired slice.")
    if analysis_scope == "l3_vertebral_level" and not bool(analysis_available_z.any()):
        raise ValueError("l3_vertebral_level analysis requires at least one slice.")

    table = slice_geometry_table(geometry)
    for column, value in identity.as_columns().items():
        table[column] = value

    storage = table["slice_index_zyx_z"].to_numpy(dtype=int)
    table["analysis_scope"] = analysis_scope
    table["body_composition_analysis_available"] = analysis_available_z[storage]
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
    body = body_surface.body_mask_zyx
    body_touching = (
        body[:, 0, :].any(axis=1)
        | body[:, -1, :].any(axis=1)
        | body[:, :, 0].any(axis=1)
        | body[:, :, -1].any(axis=1)
    )
    table["body_touching_fov"] = body_touching[storage]

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
    area_valid = nonempty_trunk & ~touching & ~fragmented & ~trunk_internal_gaps[storage]
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

    canonical_compartment_names = CORE_COMPARTMENT_NAMES
    native_output_names = {f"{name}_compartment" for name in canonical_compartment_names}
    vat_masks: dict[str, np.ndarray] = {}
    measurement_columns: dict[str, Any] = {}
    for name in canonical_compartment_names:
        source_labels = tuple(
            label for label, canonical in compartment_schema.items() if canonical == name
        )
        output_name = f"{name}_compartment"
        if not source_labels:
            measurement_columns.update(
                {
                    f"{output_name}_voxel_count": pd.array(
                        [pd.NA] * len(table),
                        dtype="Int64",
                    ),
                    f"{output_name}_area_cm2": np.full(len(table), np.nan),
                    f"{output_name}_area_valid": np.zeros(len(table), dtype=bool),
                    f"{output_name}_area_reason": ["missing_compartment"] * len(table),
                    f"{output_name}_mean_hu": np.full(len(table), np.nan),
                    f"{output_name}_hu_valid": np.zeros(len(table), dtype=bool),
                    f"{output_name}_hu_reason": ["missing_compartment"] * len(table),
                }
            )
            continue
        compartment_mask = np.isin(compartment_labels, source_labels)
        if name in {"avat", "tvat", "vat"}:
            vat_masks[name] = compartment_mask
        measurement_columns.update(
            _mask_measurement_columns(
                name=output_name,
                mask_zyx=compartment_mask,
                image_zyx=image,
                body_mask_zyx=body_surface.body_mask_zyx,
                storage_indices_z=storage,
                pixel_area_cm2=pixel_area_cm2,
                empty_reason="empty_compartment",
            )
        )

    if "avat" in vat_masks and "tvat" in vat_masks:
        vat_compartment_union_mask = vat_masks["avat"] | vat_masks["tvat"]
        measurement_columns["vat_compartment_union_source"] = ["avat_plus_tvat"] * len(table)
    elif "vat" in vat_masks:
        vat_compartment_union_mask = vat_masks["vat"]
        measurement_columns["vat_compartment_union_source"] = ["native_vat"] * len(table)
    else:
        vat_compartment_union_mask = None
        measurement_columns["vat_compartment_union_source"] = ["unavailable"] * len(table)
    if vat_compartment_union_mask is not None:
        measurement_columns.update(
            _mask_measurement_columns(
                name="vat_compartment_union",
                mask_zyx=vat_compartment_union_mask,
                image_zyx=image,
                body_mask_zyx=body_surface.body_mask_zyx,
                storage_indices_z=storage,
                pixel_area_cm2=pixel_area_cm2,
                empty_reason="empty_compartment",
            )
        )
    else:
        measurement_columns.update(
            {
                "vat_compartment_union_voxel_count": pd.array(
                    [pd.NA] * len(table),
                    dtype="Int64",
                ),
                "vat_compartment_union_area_cm2": np.full(len(table), np.nan),
                "vat_compartment_union_area_valid": np.zeros(len(table), dtype=bool),
                "vat_compartment_union_area_reason": ["invalid_measurement"] * len(table),
                "vat_compartment_union_mean_hu": np.full(len(table), np.nan),
                "vat_compartment_union_hu_valid": np.zeros(len(table), dtype=bool),
                "vat_compartment_union_hu_reason": ["invalid_measurement"] * len(table),
            }
        )

    if tissue_definitions:
        derived_names = {
            str(name)
            for name, definition in tissue_definitions.items()
            if bool(definition["enabled"])
        }
        collisions = sorted(
            {*native_output_names, "vat_compartment_union"}.intersection(derived_names)
        )
        if collisions:
            raise ValueError(
                f"Derived tissue definitions collide with native outputs: {collisions}."
            )
        for name, tissue_mask in iter_configured_tissue_masks(
            image,
            compartment_labels,
            compartment_schema,
            tissue_definitions,
            spacing_xyz=geometry.spacing_xyz,
        ):
            measurement_columns.update(
                _mask_measurement_columns(
                    name=name,
                    mask_zyx=tissue_mask,
                    image_zyx=image,
                    body_mask_zyx=body_surface.body_mask_zyx,
                    storage_indices_z=storage,
                    pixel_area_cm2=pixel_area_cm2,
                    empty_reason="empty_tissue",
                )
            )

    table = pd.concat(
        [table, pd.DataFrame(measurement_columns, index=table.index)],
        axis=1,
    )
    if tissue_definitions:
        ratio_columns: dict[str, Any] = {}
        for ratio_name, numerator, denominator in DERIVED_RATIO_DEFINITIONS:
            ratio_columns.update(
                _slice_composition_ratio_columns(
                    table,
                    name=ratio_name,
                    numerator=numerator,
                    denominator=denominator,
                )
            )
        table = pd.concat(
            [table, pd.DataFrame(ratio_columns, index=table.index)],
            axis=1,
        )

    segmented_counts = np.count_nonzero(
        compartment_labels,
        axis=(1, 2),
    ).astype(np.int64)
    inside_body_counts = np.sum(
        body_surface.body_mask_zyx,
        axis=(1, 2),
        dtype=np.int64,
        where=compartment_labels != 0,
    )
    outside_counts = segmented_counts - inside_body_counts
    total_valid = outside_counts[storage] == 0
    table["total_segmented_compartment_voxel_count"] = segmented_counts[storage]
    table["total_segmented_compartment_area_cm2"] = (
        segmented_counts[storage].astype(float) * pixel_area_cm2
    )
    table["total_segmented_compartment_area_valid"] = total_valid
    table["total_segmented_compartment_area_reason"] = np.where(
        total_valid,
        None,
        "invalid_measurement",
    )
    table["compartment_outside_body_voxel_count"] = outside_counts[storage]
    table["compartment_exceeds_body_area"] = ~total_valid
    table["body_surface_invalid"] = body_internal_gaps[storage]
    table["trunk_surface_invalid"] = ~area_valid | ~contour_valid
    table["slice_measurement_valid"] = ~(~total_valid | table["trunk_surface_invalid"])
    table["slice_qc_status"] = np.where(table["slice_measurement_valid"], "pass", "review")
    unavailable = ~table["body_composition_analysis_available"].to_numpy(dtype=bool)
    if np.any(unavailable):
        for column in table.columns:
            if column.endswith(("_area_valid", "_hu_valid")):
                table.loc[unavailable, column] = False
            elif column.endswith(("_area_reason", "_hu_reason")):
                table.loc[unavailable, column] = "outside_analysis_region"
        table.loc[unavailable, "trunk_area_valid"] = False
        table.loc[unavailable, "trunk_area_reason"] = "outside_analysis_region"
        table.loc[unavailable, "trunk_contour_valid"] = False
        table.loc[unavailable, "trunk_contour_reason"] = "outside_analysis_region"
        table.loc[unavailable, "total_segmented_compartment_area_valid"] = False
        table.loc[
            unavailable,
            "total_segmented_compartment_area_reason",
        ] = "outside_analysis_region"
        table.loc[unavailable, "body_surface_invalid"] = True
        table.loc[unavailable, "trunk_surface_invalid"] = True
        table.loc[unavailable, "slice_measurement_valid"] = False
        table.loc[unavailable, "slice_qc_status"] = "not_assessed"
    return table


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
