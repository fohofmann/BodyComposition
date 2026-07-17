"""Physical-space sagittal thick-slab projection for reports."""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
from itertools import product
import math
import re
from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from BodyComposition.reporting.contracts import ReportingSettings
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.vertebral.contracts import VertebralResult


class ProjectionError(RuntimeError):
    """Raised when the frozen sagittal view cannot be constructed safely."""


@dataclass(frozen=True)
class SagittalProjection:
    rgb: np.ndarray
    label_projection: np.ndarray
    anterior_mm: float
    posterior_mm: float
    inferior_mm: float
    superior_mm: float
    slab_left_mm: float
    slab_right_mm: float
    native_labels: tuple[int, ...]
    method: str = "sagittal_thick_slab_v1"

    def y_for_superior(self, value_mm: float, bottom: float, top: float) -> float:
        fraction = (float(value_mm) - self.inferior_mm) / (
            self.superior_mm - self.inferior_mm
        )
        return bottom + fraction * (top - bottom)


def _region_and_number(label: str) -> tuple[str, int]:
    normalized = label.upper().replace(" ", "")
    match = re.fullmatch(r"([CTL])(\d+)", normalized)
    if match:
        return match.group(1), int(match.group(2))
    if normalized == "SACRUM":
        return "S", 1
    return "U", sum(ord(character) for character in normalized) % 17


def vertebral_color(anatomical_label: str) -> tuple[int, int, int]:
    """Stable region-family palette with label/boundary redundancy."""

    region, number = _region_and_number(anatomical_label)
    base = {
        "C": (0.57, 0.68, 0.50),
        "T": (0.47, 0.60, 0.43),
        "L": (0.08, 0.78, 0.52),
        "S": (0.79, 0.48, 0.52),
        "U": (0.0, 0.0, 0.55),
    }[region]
    hue, saturation, lightness = base
    lightness = float(np.clip(lightness + ((number - 1) % 5 - 2) * 0.045, 0.34, 0.74))
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return tuple(int(round(channel * 255)) for channel in (red, green, blue))


def tissue_color(name: str) -> tuple[int, int, int]:
    return {
        "sm": (0, 114, 178),
        "imat": (204, 121, 167),
        "sat": (240, 228, 66),
        "avat": (213, 94, 0),
        "tvat": (0, 158, 115),
        "vat": (213, 94, 0),
    }[name]


def _physical_cell_bounds(geometry: ImageGeometry) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(
        [
            geometry.physical_point(index)
            for index in product(
                (-0.5, geometry.size_xyz[0] - 0.5),
                (-0.5, geometry.size_xyz[1] - 0.5),
                (-0.5, geometry.size_xyz[2] - 0.5),
            )
        ],
        dtype=float,
    )
    return corners.min(axis=0), corners.max(axis=0)


def _mask_bounds_lps(labels_zyx: np.ndarray, geometry: ImageGeometry) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.argwhere(labels_zyx != 0)
    if occupied.size == 0:
        raise ProjectionError("The vertebral-body mask is empty.")
    lower_zyx = occupied.min(axis=0).astype(float) - 0.5
    upper_zyx = occupied.max(axis=0).astype(float) + 0.5
    corners = np.asarray(
        [
            geometry.physical_point((x, y, z))
            for z, y, x in product(
                (lower_zyx[0], upper_zyx[0]),
                (lower_zyx[1], upper_zyx[1]),
                (lower_zyx[2], upper_zyx[2]),
            )
        ],
        dtype=float,
    )
    return corners.min(axis=0), corners.max(axis=0)


def _canonical_resample(
    image: sitk.Image,
    *,
    lower_lps: np.ndarray,
    upper_lps: np.ndarray,
    spacing_mm: float,
    interpolator: int,
    default_value: float,
    output_pixel_type: int,
) -> sitk.Image:
    extent = upper_lps - lower_lps
    size = tuple(int(math.ceil(float(value) / spacing_mm)) + 1 for value in extent)
    if any(value < 2 or value > 2048 for value in size) or int(np.prod(size)) > 50_000_000:
        raise ProjectionError(f"Validated sagittal projection grid is unsafe: {size}.")
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing((spacing_mm, spacing_mm, spacing_mm))
    resampler.SetOutputOrigin(tuple(float(value) for value in lower_lps))
    resampler.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetOutputPixelType(output_pixel_type)
    return resampler.Execute(image)


def build_sagittal_projection(
    prepared_image: sitk.Image,
    vertebral_result: VertebralResult,
    settings: ReportingSettings,
    *,
    centroids_lps_xyz: Sequence[Sequence[float]] | None = None,
) -> SagittalProjection:
    """Build the frozen canonical LPS slab without ad-hoc array-axis assumptions."""

    if prepared_image.GetDimension() != 3:
        raise ProjectionError("The prepared CT is not three-dimensional.")
    if vertebral_result.geometry is None or vertebral_result.vertebral_body_labels is None:
        raise ProjectionError("Vertebral-body labels are unavailable.")
    reference_geometry = ImageGeometry.from_sitk(prepared_image)
    assert_same_physical_domain(
        reference_geometry,
        vertebral_result.geometry,
        reference_name="prepared CT",
        candidate_name="vertebral-body labels",
    )
    labels = np.asarray(vertebral_result.vertebral_body_labels)
    expected_shape = tuple(reversed(reference_geometry.size_xyz))
    if labels.shape != expected_shape:
        raise ProjectionError(
            f"Vertebral-body array_zyx shape {labels.shape} differs from {expected_shape}."
        )
    present = tuple(sorted(int(value) for value in np.unique(labels) if value != 0))
    if centroids_lps_xyz is None:
        centroid_points = [
            tuple(float(value) for value in centroid.physical_lps_xyz)
            for centroid in vertebral_result.centroids
            if centroid.native_label in present
        ]
    else:
        centroid_points = [
            tuple(float(value) for value in point)
            for point in centroids_lps_xyz
            if len(point) == 3 and all(np.isfinite(point))
        ]
    if len(centroid_points) < 2:
        raise ProjectionError("At least two valid vertebral-body centroids are required.")

    ct_lower, ct_upper = _physical_cell_bounds(reference_geometry)
    mask_lower, mask_upper = _mask_bounds_lps(labels, reference_geometry)
    centroid_x = np.asarray([item[0] for item in centroid_points], dtype=float)
    slab_lower = max(float(ct_lower[0]), float(centroid_x.min() - settings.sagittal_slab_margin_mm))
    slab_upper = min(float(ct_upper[0]), float(centroid_x.max() + settings.sagittal_slab_margin_mm))
    if slab_upper - slab_lower < settings.projection_spacing_mm:
        raise ProjectionError("The spine-centered lateral slab has no physical width.")
    lower = np.asarray(
        [slab_lower, ct_lower[1], max(ct_lower[2], mask_lower[2] - 12.0)],
        dtype=float,
    )
    upper = np.asarray(
        [slab_upper, ct_upper[1], min(ct_upper[2], mask_upper[2] + 12.0)],
        dtype=float,
    )
    if np.any(upper <= lower):
        raise ProjectionError("The vertebral-body projection bounds are invalid.")

    label_image = sitk.GetImageFromArray(labels.astype(np.uint16, copy=False))
    label_image.CopyInformation(prepared_image)
    resampled_ct = _canonical_resample(
        prepared_image,
        lower_lps=lower,
        upper_lps=upper,
        spacing_mm=settings.projection_spacing_mm,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
        output_pixel_type=sitk.sitkFloat32,
    )
    resampled_labels = _canonical_resample(
        label_image,
        lower_lps=lower,
        upper_lps=upper,
        spacing_mm=settings.projection_spacing_mm,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0.0,
        output_pixel_type=sitk.sitkUInt16,
    )
    ct_zyx = sitk.GetArrayFromImage(resampled_ct)
    label_zyx = sitk.GetArrayFromImage(resampled_labels)
    ct_mip_zy = np.max(ct_zyx, axis=2)
    x_coordinates = lower[0] + np.arange(label_zyx.shape[2]) * settings.projection_spacing_mm
    center_x = float(np.median(centroid_x))
    nearest_order = np.argsort(np.abs(x_coordinates - center_x), kind="stable")
    ordered = label_zyx[:, :, nearest_order]
    occupied = ordered != 0
    any_label = occupied.any(axis=2)
    nearest = occupied.argmax(axis=2)
    label_projection = np.zeros(any_label.shape, dtype=np.uint16)
    selected = np.take_along_axis(ordered, nearest[..., None], axis=2)[..., 0]
    label_projection[any_label] = selected[any_label]

    window_lower, window_upper = settings.ct_window
    grey = np.clip((ct_mip_zy - window_lower) / (window_upper - window_lower), 0, 1)
    grey = np.rint(grey * 255).astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=2).astype(np.float32)
    for native_label in present:
        anatomical = vertebral_result.label_schema.get(native_label, f"UNKNOWN_{native_label}")
        mask = label_projection == native_label
        if not np.any(mask):
            continue
        color = np.asarray(vertebral_color(anatomical), dtype=float)
        alpha = float(settings.overlay_opacity)
        rgb[mask] = rgb[mask] * (1.0 - alpha) + color * alpha
        boundary = mask & ~ndimage.binary_erosion(mask)
        rgb[boundary] = color * 0.35
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Canonical LPS z increases from inferior to superior. Raster page rows run
    # downwards, so flip only the display plane after all physical operations.
    rgb = np.flipud(rgb)
    label_projection = np.flipud(label_projection)
    return SagittalProjection(
        rgb=rgb,
        label_projection=label_projection,
        anterior_mm=float(lower[1]),
        posterior_mm=float(upper[1]),
        inferior_mm=float(lower[2]),
        superior_mm=float(upper[2]),
        slab_left_mm=float(lower[0]),
        slab_right_mm=float(upper[0]),
        native_labels=present,
    )
