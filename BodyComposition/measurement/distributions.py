"""Whole-volume HU distributions from model-native tissue compartments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from BodyComposition.measurement.contracts import (
    HU_DISTRIBUTION_BIN_WIDTH_HU,
    HU_DISTRIBUTION_MAX_HU,
    HU_DISTRIBUTION_MIN_HU,
    HU_DISTRIBUTION_SCOPE,
    HU_DISTRIBUTION_TISSUES,
    TISSUE_HU_DISTRIBUTION_SCHEMA_VERSION,
    MeasurementIdentity,
)
from BodyComposition.measurement.physical import validate_array_zyx
from BodyComposition.measurement.tissues import canonical_tissue_name
from BodyComposition.utils.geometry import ImageGeometry


def _scope_bounds(geometry: ImageGeometry) -> tuple[float, float]:
    """Return the complete voxel-cell extent on the patient superior axis."""

    corner_indices_xyz = np.asarray(
        [
            (x, y, z)
            for x in (-0.5, geometry.size_xyz[0] - 0.5)
            for y in (-0.5, geometry.size_xyz[1] - 0.5)
            for z in (-0.5, geometry.size_xyz[2] - 0.5)
        ],
        dtype=float,
    )
    superior_positions = geometry.physical_point(corner_indices_xyz)[:, 2]
    scope_inferior = float(np.min(superior_positions))
    scope_superior = float(np.max(superior_positions))
    if not scope_inferior < scope_superior:
        raise ValueError("HU distribution scope must have positive physical extent.")
    return scope_inferior, scope_superior


def _native_labels_by_tissue(
    label_schema: Mapping[int, str],
) -> dict[str, tuple[int, ...]]:
    labels_by_tissue: dict[str, list[int]] = {}
    for native_label, name in label_schema.items():
        if isinstance(native_label, bool) or not isinstance(native_label, int) or native_label <= 0:
            raise ValueError("Compartment label identifiers must be positive integers.")
        labels_by_tissue.setdefault(canonical_tissue_name(name), []).append(native_label)
    return {
        tissue: tuple(sorted(native_labels)) for tissue, native_labels in labels_by_tissue.items()
    }


def build_tissue_hu_distributions(
    *,
    image_zyx: np.ndarray,
    compartment_labels_zyx: np.ndarray,
    compartment_label_schema: Mapping[int, str],
    geometry: ImageGeometry,
    identity: MeasurementIdentity,
) -> pd.DataFrame:
    """Return fixed-bin HU distributions for native SM, SAT, aVAT, and tVAT.

    The scope is the complete analyzed CT volume. Tissue membership comes only
    from the unfiltered model-native compartment mask. Histogram values and
    exact summary statistics use the unchanged orientation-prepared CT voxels.
    """

    image = np.asarray(image_zyx)
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("HU distributions require one numeric image_zyx volume.")
    image = validate_array_zyx(image, geometry, "image_zyx")
    compartment_labels = validate_array_zyx(
        compartment_labels_zyx,
        geometry,
        "compartment_labels_zyx",
        finite=False,
    )
    if not np.issubdtype(compartment_labels.dtype, np.integer):
        raise ValueError("HU distributions require aligned integer compartment_labels_zyx.")
    voxel_volume_mm3 = geometry.voxel_volume_mm3
    labels_by_tissue = _native_labels_by_tissue(compartment_label_schema)
    unknown_labels = sorted(
        int(label)
        for label in np.unique(compartment_labels)
        if int(label) != 0 and int(label) not in compartment_label_schema
    )
    if unknown_labels:
        raise ValueError(
            f"Compartment labels contain values outside the model-native schema: {unknown_labels}."
        )
    edges = np.arange(
        HU_DISTRIBUTION_MIN_HU,
        HU_DISTRIBUTION_MAX_HU + HU_DISTRIBUTION_BIN_WIDTH_HU,
        HU_DISTRIBUTION_BIN_WIDTH_HU,
        dtype=float,
    )
    scope_inferior, scope_superior = _scope_bounds(geometry)
    rows: list[dict[str, Any]] = []
    identity_columns = identity.as_columns()

    for tissue_key, display_label, source_compartment in HU_DISTRIBUTION_TISSUES:
        source_label_ids = labels_by_tissue.get(source_compartment, ())
        mask = (
            np.isin(compartment_labels, source_label_ids)
            if source_label_ids
            else np.zeros(image.shape, dtype=bool)
        )

        selected = np.asarray(image[mask], dtype=float)
        total_voxel_count = int(selected.size)
        finite = selected[np.isfinite(selected)]
        finite_voxel_count = int(finite.size)
        nonfinite_voxel_count = total_voxel_count - finite_voxel_count
        below_count = int(np.count_nonzero(finite < HU_DISTRIBUTION_MIN_HU))
        above_count = int(np.count_nonzero(finite > HU_DISTRIBUTION_MAX_HU))
        histogram_values = finite[
            (finite >= HU_DISTRIBUTION_MIN_HU) & (finite <= HU_DISTRIBUTION_MAX_HU)
        ]
        counts, _ = np.histogram(histogram_values, bins=edges)
        in_histogram_voxel_count = int(counts.sum())
        contributing_slice_count = int(np.count_nonzero(np.any(mask, axis=(1, 2))))

        if not source_label_ids:
            valid = False
            reason = "missing_compartment"
        elif total_voxel_count == 0:
            valid = False
            reason = "empty_tissue"
        elif nonfinite_voxel_count:
            valid = False
            reason = "invalid_measurement"
        else:
            valid = True
            reason = None

        if finite_voxel_count:
            mean_hu = float(np.mean(finite))
            standard_deviation_hu = float(np.std(finite, ddof=0))
            q1_hu, median_hu, q3_hu = np.quantile(
                finite,
                [0.25, 0.5, 0.75],
                method="linear",
            )
        else:
            mean_hu = standard_deviation_hu = np.nan
            q1_hu = median_hu = q3_hu = np.nan

        common = {
            **identity_columns,
            "distribution_schema_version": TISSUE_HU_DISTRIBUTION_SCHEMA_VERSION,
            "distribution_scope": HU_DISTRIBUTION_SCOPE,
            "tissue_key": tissue_key,
            "display_label": display_label,
            "source_compartment": source_compartment,
            "source_label_ids": ",".join(str(value) for value in source_label_ids),
            "source_semantics": "model_native_compartment",
            "histogram_min_hu": HU_DISTRIBUTION_MIN_HU,
            "histogram_max_hu": HU_DISTRIBUTION_MAX_HU,
            "bin_width_hu": HU_DISTRIBUTION_BIN_WIDTH_HU,
            "bin_semantics": "left_closed_right_open_final_closed",
            "total_voxel_count": total_voxel_count,
            "finite_voxel_count": finite_voxel_count,
            "nonfinite_voxel_count": nonfinite_voxel_count,
            "in_histogram_voxel_count": in_histogram_voxel_count,
            "below_histogram_voxel_count": below_count,
            "above_histogram_voxel_count": above_count,
            "below_histogram_voxel_fraction": (
                float(below_count / total_voxel_count) if total_voxel_count else 0.0
            ),
            "above_histogram_voxel_fraction": (
                float(above_count / total_voxel_count) if total_voxel_count else 0.0
            ),
            "contributing_slice_count": contributing_slice_count,
            "voxel_volume_mm3": float(voxel_volume_mm3),
            "total_volume_cm3": float(total_voxel_count * voxel_volume_mm3 / 1000.0),
            "distribution_valid": valid,
            "distribution_reason": reason,
            "mean_hu": mean_hu,
            "standard_deviation_hu": standard_deviation_hu,
            "q1_hu": float(q1_hu),
            "median_hu": float(median_hu),
            "q3_hu": float(q3_hu),
            "quantile_method": "linear",
            "scope_inferior_position_superior_mm": scope_inferior,
            "scope_superior_position_superior_mm": scope_superior,
        }
        for bin_index, count in enumerate(counts):
            rows.append(
                {
                    **common,
                    "bin_index": bin_index,
                    "bin_lower_hu": float(edges[bin_index]),
                    "bin_upper_hu": float(edges[bin_index + 1]),
                    "bin_upper_inclusive": bin_index == len(counts) - 1,
                    "voxel_count": int(count),
                    "voxel_fraction": (
                        float(count / total_voxel_count) if total_voxel_count else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)
