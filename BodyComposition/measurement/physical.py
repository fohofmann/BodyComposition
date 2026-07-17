"""Affine-aware physical helpers for per-slice and interval measurements."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from BodyComposition.utils.geometry import ImageGeometry


SUPERIOR_LPS = np.asarray((0.0, 0.0, 1.0), dtype=float)
MIN_SUPERIOR_COSINE = 1e-3


@dataclass(frozen=True)
class CleanedComponent:
    mask_zyx: np.ndarray
    voxel_count: int
    retained_voxel_count: int
    component_count: int
    removed_fraction: float
    touches_fov: bool


def validate_array_zyx(
    array_zyx: np.ndarray,
    geometry: ImageGeometry,
    name: str,
    *,
    finite: bool = True,
) -> np.ndarray:
    array = np.asarray(array_zyx)
    expected_shape = tuple(reversed(geometry.size_xyz))
    if array.ndim != 3 or array.shape != expected_shape:
        raise ValueError(f"{name} must have array_zyx shape {expected_shape}, got {array.shape}.")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def geometry_digest(geometry: ImageGeometry) -> str:
    payload = {
        "size_xyz": geometry.size_xyz,
        "spacing_xyz": geometry.spacing_xyz,
        "origin_lps_xyz": geometry.origin_lps_xyz,
        "direction_lps": geometry.direction_lps,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def slice_geometry_table(geometry: ImageGeometry) -> pd.DataFrame:
    """Return one row per prepared array-z slice, ordered inferior to superior."""

    geometry.validate()
    basis_lps = geometry.basis_lps
    step_lps = basis_lps[:, 2]
    slice_normal_lps = step_lps / np.linalg.norm(step_lps)
    superior_step_signed_mm = float(np.dot(step_lps, SUPERIOR_LPS))
    superior_thickness_mm = abs(superior_step_signed_mm)
    if superior_thickness_mm < MIN_SUPERIOR_COSINE * geometry.spacing_xyz[2]:
        raise ValueError(
            "Prepared CT slice axis is nearly orthogonal to the patient superior axis; "
            "longitudinal measurements are undefined."
        )

    center_x = (geometry.size_xyz[0] - 1) / 2.0
    center_y = (geometry.size_xyz[1] - 1) / 2.0
    storage_indices = np.arange(geometry.size_xyz[2], dtype=int)
    centers_xyz = np.column_stack(
        (
            np.full(storage_indices.size, center_x),
            np.full(storage_indices.size, center_y),
            storage_indices.astype(float),
        )
    )
    centers_lps = geometry.physical_point(centers_xyz)
    position_superior_mm = centers_lps @ SUPERIOR_LPS
    order = np.argsort(position_superior_mm, kind="stable")

    table = pd.DataFrame(
        {
            "slice_index_zyx_z": storage_indices,
            "original_storage_index": storage_indices,
            "position_superior_mm": position_superior_mm,
            "slice_center_lps_x_mm": centers_lps[:, 0],
            "slice_center_lps_y_mm": centers_lps[:, 1],
            "slice_center_lps_z_mm": centers_lps[:, 2],
            "slice_normal_lps_x": float(slice_normal_lps[0]),
            "slice_normal_lps_y": float(slice_normal_lps[1]),
            "slice_normal_lps_z": float(slice_normal_lps[2]),
            "slice_thickness_normal_mm": float(geometry.spacing_xyz[2]),
            "slice_thickness_superior_mm": superior_thickness_mm,
            "slice_slab_inferior_mm": position_superior_mm - superior_thickness_mm / 2.0,
            "slice_slab_superior_mm": position_superior_mm + superior_thickness_mm / 2.0,
            "normal_mm_per_superior_mm": float(geometry.spacing_xyz[2])
            / superior_thickness_mm,
            "in_plane_pixel_area_mm2": geometry.in_plane_area_mm2,
        }
    )
    table = table.iloc[order].reset_index(drop=True)
    table.insert(0, "longitudinal_order", np.arange(len(table), dtype=int))
    table.insert(1, "slice_id", table["slice_index_zyx_z"].to_numpy(dtype=int))
    return table


def interval_overlap_mm(
    lower_a: np.ndarray | float,
    upper_a: np.ndarray | float,
    lower_b: float,
    upper_b: float,
) -> np.ndarray:
    if not lower_b < upper_b:
        raise ValueError("A physical interval requires lower < upper.")
    lower = np.maximum(np.asarray(lower_a, dtype=float), lower_b)
    upper = np.minimum(np.asarray(upper_a, dtype=float), upper_b)
    return np.maximum(upper - lower, 0.0)


def covered_interval_length_mm(
    lowers: np.ndarray,
    uppers: np.ndarray,
    target_lower: float,
    target_upper: float,
) -> float:
    """Return union coverage of slice intervals clipped to one target interval."""

    intervals = sorted(
        (
            max(float(lower), target_lower),
            min(float(upper), target_upper),
        )
        for lower, upper in zip(lowers, uppers, strict=True)
        if min(float(upper), target_upper) > max(float(lower), target_lower)
    )
    if not intervals:
        return 0.0
    total = 0.0
    current_lower, current_upper = intervals[0]
    for lower, upper in intervals[1:]:
        if lower <= current_upper:
            current_upper = max(current_upper, upper)
        else:
            total += current_upper - current_lower
            current_lower, current_upper = lower, upper
    return total + current_upper - current_lower


def clean_largest_component(mask_zyx: np.ndarray) -> CleanedComponent:
    mask = np.asarray(mask_zyx, dtype=bool)
    voxel_count = int(np.count_nonzero(mask))
    if voxel_count == 0:
        return CleanedComponent(mask, 0, 0, 0, 0.0, False)
    structure = ndimage.generate_binary_structure(3, 2)
    labels, component_count = ndimage.label(mask, structure=structure)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    cleaned = labels == largest_label
    retained = int(counts[largest_label])
    touches = bool(
        cleaned[0].any()
        or cleaned[-1].any()
        or cleaned[:, 0].any()
        or cleaned[:, -1].any()
        or cleaned[:, :, 0].any()
        or cleaned[:, :, -1].any()
    )
    return CleanedComponent(
        mask_zyx=cleaned,
        voxel_count=voxel_count,
        retained_voxel_count=retained,
        component_count=int(component_count),
        removed_fraction=float((voxel_count - retained) / voxel_count),
        touches_fov=touches,
    )


def mask_physical_extent(
    mask_zyx: np.ndarray,
    geometry: ImageGeometry,
) -> tuple[float, float, tuple[float, float, float], float]:
    """Project the complete voxel-cell extent and centroid onto patient superior."""

    mask = validate_array_zyx(mask_zyx, geometry, "mask_zyx", finite=False).astype(bool)
    indices_zyx = np.argwhere(mask)
    if indices_zyx.size == 0:
        raise ValueError("A physical mask extent requires at least one voxel.")
    indices_xyz = indices_zyx[:, ::-1].astype(float)
    centers_lps = geometry.physical_point(indices_xyz)
    center_superior = centers_lps @ SUPERIOR_LPS
    half_cell_superior_mm = 0.5 * float(
        np.abs(geometry.basis_lps.T @ SUPERIOR_LPS).sum()
    )
    inferior_mm = float(center_superior.min() - half_cell_superior_mm)
    superior_mm = float(center_superior.max() + half_cell_superior_mm)
    centroid_lps = tuple(float(value) for value in centers_lps.mean(axis=0))
    centroid_superior_mm = float(np.dot(np.asarray(centroid_lps), SUPERIOR_LPS))
    return inferior_mm, superior_mm, centroid_lps, centroid_superior_mm
