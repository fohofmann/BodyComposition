"""External body-contour measurements in full physical coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure

from BodyComposition.utils.geometry import ImageGeometry


@dataclass(frozen=True)
class ExternalContourMeasurement:
    perimeter_cm: float | None
    component_count: int
    contour_closed: bool
    touching_image_boundary: bool
    valid: bool
    reason: str | None = None


def _polygon_area_index_yx(contour_yx: np.ndarray) -> float:
    y = contour_yx[:, 0]
    x = contour_yx[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _component_perimeter_mm(
    component_yx: np.ndarray,
    geometry: ImageGeometry,
    slice_index_z: int,
) -> tuple[float, bool]:
    filled = ndimage.binary_fill_holes(component_yx)
    padded = np.pad(filled, 1, mode="constant", constant_values=False)
    contours = measure.find_contours(
        padded.astype(np.uint8),
        level=0.5,
        fully_connected="high",
        positive_orientation="low",
    )
    if not contours:
        raise ValueError("No external contour could be traced.")
    contour_yx = max((contour - 1.0 for contour in contours), key=_polygon_area_index_yx)
    if contour_yx.shape[0] < 3:
        raise ValueError("The external contour has fewer than three vertices.")
    closed = bool(np.allclose(contour_yx[0], contour_yx[-1], atol=1e-8, rtol=0))
    if not closed:
        contour_yx = np.vstack((contour_yx, contour_yx[0]))
    index_xyz = np.column_stack(
        (
            contour_yx[:, 1],
            contour_yx[:, 0],
            np.full(contour_yx.shape[0], float(slice_index_z)),
        )
    )
    physical_lps_xyz = geometry.physical_point(index_xyz)
    perimeter_mm = float(
        np.linalg.norm(np.diff(physical_lps_xyz, axis=0), axis=1).sum()
    )
    return perimeter_mm, closed


def external_contour_measurement(
    mask_yx: np.ndarray,
    geometry: ImageGeometry,
    slice_index_z: int,
    *,
    component_mode: str,
) -> ExternalContourMeasurement:
    """Measure all external components or only the primary component.

    Internal holes are filled before tracing and therefore never add to the
    reported perimeter. Every contour vertex is transformed through the full
    LPS affine before segment lengths are summed.
    """

    if component_mode not in {"all", "primary"}:
        raise ValueError("component_mode must be 'all' or 'primary'.")
    mask = np.asarray(mask_yx, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("A body contour input must be a two-dimensional y-x mask.")
    if not mask.any():
        return ExternalContourMeasurement(None, 0, False, False, False, "empty_mask")

    touching = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    labels, component_count = ndimage.label(
        mask,
        structure=ndimage.generate_binary_structure(2, 2),
    )
    component_ids = np.arange(1, component_count + 1, dtype=int)
    counts = ndimage.sum(mask, labels, index=component_ids)
    if component_mode == "primary":
        component_ids = np.asarray((component_ids[int(np.argmax(counts))],), dtype=int)

    perimeters_mm: list[float] = []
    closed_flags: list[bool] = []
    for component_id in component_ids:
        try:
            perimeter_mm, closed = _component_perimeter_mm(
                labels == component_id,
                geometry,
                slice_index_z,
            )
        except ValueError:
            continue
        perimeters_mm.append(perimeter_mm)
        closed_flags.append(closed)
    if not perimeters_mm:
        return ExternalContourMeasurement(
            None,
            int(component_count),
            False,
            touching,
            False,
            "contour_not_found",
        )
    return ExternalContourMeasurement(
        perimeter_cm=float(sum(perimeters_mm) / 10.0),
        component_count=int(component_count),
        contour_closed=all(closed_flags),
        touching_image_boundary=touching,
        valid=not touching and all(closed_flags),
        reason=(
            "touching_image_boundary"
            if touching
            else ("open_contour" if not all(closed_flags) else None)
        ),
    )
