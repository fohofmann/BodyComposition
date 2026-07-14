from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from skimage import measure

from BodyComposition.utils.geometry import ImageGeometry


class MeasurementError(ValueError):
    """Raised when measurement inputs do not satisfy the measurement contract."""


def validate_label_mapping(labels: Mapping[int, str]) -> dict[int, str]:
    if not isinstance(labels, Mapping) or not labels:
        raise MeasurementError("Label mapping must be a non-empty integer-to-name mapping.")

    validated: dict[int, str] = {}
    for label, name in labels.items():
        if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
            raise MeasurementError(f"Label identifiers must be positive integers, got {label!r}.")
        if not isinstance(name, str) or not name.strip():
            raise MeasurementError(f"Label {label} must have a non-empty string name.")
        validated[label] = name.strip()

    if len(set(validated.values())) != len(validated):
        raise MeasurementError("Label names must be unique.")
    return validated


def _validate_array(array_zyx: np.ndarray, geometry: ImageGeometry, name: str) -> np.ndarray:
    array = np.asarray(array_zyx)
    expected_shape = tuple(reversed(geometry.size_xyz))
    if array.ndim != 3 or array.shape != expected_shape:
        raise MeasurementError(
            f"{name} must have array_zyx shape {expected_shape}, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise MeasurementError(f"{name} contains non-finite values.")
    return array


def contour_metrics(
    binary_mask_yx: np.ndarray,
    geometry: ImageGeometry,
    slice_index_z: int,
) -> tuple[float, float, str]:
    """Return largest external contour perimeter (cm), area (cm2), and status."""
    binary_mask = np.asarray(binary_mask_yx, dtype=bool)
    if binary_mask.ndim != 2:
        raise MeasurementError("A contour mask must be a two-dimensional y-x array.")
    if not binary_mask.any():
        return np.nan, np.nan, "empty_mask"

    padded = np.pad(binary_mask, 1, mode="constant", constant_values=False)
    contours_yx = measure.find_contours(
        padded.astype(np.uint8),
        level=0.5,
        fully_connected="high",
        positive_orientation="low",
    )
    if not contours_yx:
        return np.nan, np.nan, "contour_not_found"

    metrics: list[tuple[float, float]] = []
    for contour_yx in contours_yx:
        area_contour_yx = contour_yx - 1.0
        perimeter_contour_yx = measure.approximate_polygon(area_contour_yx, tolerance=1.0)
        if area_contour_yx.shape[0] < 3 or perimeter_contour_yx.shape[0] < 3:
            continue

        perimeter_index_xyz = np.column_stack(
            (
                perimeter_contour_yx[:, 1],
                perimeter_contour_yx[:, 0],
                np.full(perimeter_contour_yx.shape[0], float(slice_index_z)),
            )
        )
        perimeter_physical_lps = geometry.physical_point(perimeter_index_xyz)
        if not np.allclose(perimeter_physical_lps[0], perimeter_physical_lps[-1]):
            perimeter_physical_lps = np.vstack(
                (perimeter_physical_lps, perimeter_physical_lps[0])
            )

        area_index_xyz = np.column_stack(
            (
                area_contour_yx[:, 1],
                area_contour_yx[:, 0],
                np.full(area_contour_yx.shape[0], float(slice_index_z)),
            )
        )
        area_physical_lps = geometry.physical_point(area_index_xyz)
        if not np.allclose(area_physical_lps[0], area_physical_lps[-1]):
            area_physical_lps = np.vstack((area_physical_lps, area_physical_lps[0]))

        perimeter_mm = float(
            np.linalg.norm(np.diff(perimeter_physical_lps, axis=0), axis=1).sum()
        )
        area_vector = np.cross(area_physical_lps[:-1], area_physical_lps[1:]).sum(axis=0)
        area_mm2 = float(0.5 * np.linalg.norm(area_vector))
        metrics.append((perimeter_mm, area_mm2))

    if not metrics:
        return np.nan, np.nan, "contour_not_found"

    perimeter_mm, area_mm2 = max(metrics, key=lambda item: item[1])
    status = "ok" if len(metrics) == 1 else "multiple_components_largest_used"
    return perimeter_mm / 10.0, area_mm2 / 100.0, status


def calculate_slice_measurements(
    mask_zyx: np.ndarray,
    geometry: ImageGeometry,
    labels: Mapping[int, str],
    *,
    image_zyx: np.ndarray | None = None,
    contour_mask_zyx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate model-free per-slice voxel, area, HU, and contour measurements."""
    geometry.validate()
    mask = _validate_array(mask_zyx, geometry, "mask_zyx")
    image = None if image_zyx is None else _validate_array(image_zyx, geometry, "image_zyx")
    contour_mask = (
        None
        if contour_mask_zyx is None
        else _validate_array(contour_mask_zyx, geometry, "contour_mask_zyx")
    )
    label_mapping = validate_label_mapping(labels)

    z_size = mask.shape[0]
    result = pd.DataFrame(index=pd.RangeIndex(z_size, name="slice_index_zyx_z"))
    pixel_area_cm2 = geometry.in_plane_area_mm2 / 100.0

    for label, name in label_mapping.items():
        label_mask = mask == label
        voxel_count = np.count_nonzero(label_mask, axis=(1, 2)).astype(np.int64)
        result[f"Vx_{name}"] = voxel_count
        result[f"CSA_{name}"] = voxel_count.astype(float) * pixel_area_cm2

        if image is not None:
            hu_sum = np.where(label_mask, image, 0.0).sum(axis=(1, 2), dtype=float)
            mean_hu = np.full(z_size, np.nan, dtype=float)
            non_empty = voxel_count > 0
            mean_hu[non_empty] = hu_sum[non_empty] / voxel_count[non_empty]
            result[f"HU_{name}"] = mean_hu
            result[f"HU_status_{name}"] = np.where(non_empty, "ok", "empty_tissue")

    if contour_mask is not None:
        perimeter_cm = np.full(z_size, np.nan, dtype=float)
        contour_area_cm2 = np.full(z_size, np.nan, dtype=float)
        contour_status: list[str] = []
        for slice_index_z in range(z_size):
            perimeter_cm[slice_index_z], contour_area_cm2[slice_index_z], status = contour_metrics(
                contour_mask[slice_index_z] != 0,
                geometry,
                slice_index_z,
            )
            contour_status.append(status)
        result["CIR_contour"] = perimeter_cm
        result["CSA_contour"] = contour_area_cm2
        result["Contour_status"] = contour_status

    return result


def weighted_mean_hu(voxel_count: pd.Series, mean_hu: pd.Series) -> tuple[float, str]:
    voxel_count = pd.to_numeric(voxel_count, errors="coerce")
    mean_hu = pd.to_numeric(mean_hu, errors="coerce")
    valid = voxel_count.gt(0) & mean_hu.notna()
    denominator = float(voxel_count[valid].sum())
    if denominator == 0:
        return np.nan, "empty_tissue"
    numerator = float((voxel_count[valid] * mean_hu[valid]).sum())
    return numerator / denominator, "ok"
