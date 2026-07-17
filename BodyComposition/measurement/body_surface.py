"""Explicit body-surface backends used by canonical measurements."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image

from BodyComposition.measurement.contracts import BodySurfaceResult
from BodyComposition.measurement.physical import validate_array_zyx
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import QCFlag, QCSeverity


TOTALSEGMENTATOR_BODY_BACKEND = "totalsegmentator_body_task299_v1"
DETERMINISTIC_BODY_BACKEND = "deterministic_body_mask_v1"
TISSUE_ENVELOPE_BACKEND = "tissue_segmentation_envelope_v1"


def _component_count(mask_zyx: np.ndarray) -> int:
    _, count = ndimage.label(mask_zyx, structure=ndimage.generate_binary_structure(3, 2))
    return int(count)


def _elliptical_footprint(
    radius_mm: float,
    spacing_yx_mm: tuple[float, float],
) -> np.ndarray:
    if radius_mm <= 0:
        return np.ones((1, 1), dtype=bool)
    radius_y = max(1, int(np.ceil(radius_mm / spacing_yx_mm[0])))
    radius_x = max(1, int(np.ceil(radius_mm / spacing_yx_mm[1])))
    coordinates_y, coordinates_x = np.ogrid[
        -radius_y : radius_y + 1,
        -radius_x : radius_x + 1,
    ]
    return (
        (coordinates_y * spacing_yx_mm[0] / radius_mm) ** 2
        + (coordinates_x * spacing_yx_mm[1] / radius_mm) ** 2
    ) <= 1.0


def _smoothed_component_envelope(
    component_yx: np.ndarray,
    *,
    spacing_yx_mm: tuple[float, float],
    closing_radius_mm: float,
    smoothing_sigma_mm: float,
) -> np.ndarray:
    closed = ndimage.binary_closing(
        component_yx,
        structure=_elliptical_footprint(closing_radius_mm, spacing_yx_mm),
    )
    envelope = convex_hull_image(closed | component_yx)
    if smoothing_sigma_mm > 0:
        sigma_yx = tuple(
            smoothing_sigma_mm / spacing for spacing in spacing_yx_mm
        )
        smoothed = ndimage.gaussian_filter(
            envelope.astype(np.float32),
            sigma=sigma_yx,
            mode="constant",
        ) >= 0.5
        envelope = ndimage.binary_fill_holes(smoothed | component_yx)
    return np.asarray(envelope, dtype=bool)


def tissue_segmentation_envelope(
    tissue_labels_zyx: np.ndarray,
    geometry: ImageGeometry,
    *,
    minimum_component_area_mm2: float = 100.0,
    closing_radius_mm: float = 6.0,
    smoothing_sigma_mm: float = 2.0,
    provenance: Mapping[str, Any] | None = None,
) -> BodySurfaceResult:
    """Derive a smoothed axial body/trunk envelope from existing tissue labels.

    The body mask contains every non-zero tissue voxel. On each slice, retained
    tissue components are closed and wrapped by their physical convex envelope.
    The largest retained component defines the trunk candidate, which avoids
    joining clearly separated arms or devices to the primary torso contour.
    """

    labels = validate_array_zyx(
        tissue_labels_zyx,
        geometry,
        "tissue_labels_zyx",
    )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Tissue labels used for an envelope must be integer-valued.")
    for name, value, allow_zero in (
        ("minimum_component_area_mm2", minimum_component_area_mm2, False),
        ("closing_radius_mm", closing_radius_mm, True),
        ("smoothing_sigma_mm", smoothing_sigma_mm, True),
    ):
        if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}.")

    foreground = labels != 0
    body = foreground.copy()
    trunk = np.zeros_like(foreground)
    pixel_area_mm2 = float(geometry.in_plane_area_mm2)
    minimum_pixels = max(1, int(np.ceil(minimum_component_area_mm2 / pixel_area_mm2)))
    spacing_yx_mm = (float(geometry.spacing_xyz[1]), float(geometry.spacing_xyz[0]))
    retained_components = 0
    discarded_components = 0
    multi_component_slices = 0

    for slice_index_z, foreground_yx in enumerate(foreground):
        component_labels, component_count = ndimage.label(
            foreground_yx,
            structure=ndimage.generate_binary_structure(2, 2),
        )
        if component_count == 0:
            continue
        sizes = np.bincount(component_labels.ravel())
        sizes[0] = 0
        retained = [
            int(label)
            for label in range(1, component_count + 1)
            if int(sizes[label]) >= minimum_pixels
        ]
        discarded_components += component_count - len(retained)
        if not retained:
            continue
        retained_components += len(retained)
        multi_component_slices += int(len(retained) > 1)
        largest = max(retained, key=lambda label: int(sizes[label]))
        for component_label in retained:
            envelope = _smoothed_component_envelope(
                component_labels == component_label,
                spacing_yx_mm=spacing_yx_mm,
                closing_radius_mm=float(closing_radius_mm),
                smoothing_sigma_mm=float(smoothing_sigma_mm),
            )
            body[slice_index_z] |= envelope
            if component_label == largest:
                trunk[slice_index_z] = envelope

    body |= trunk
    flags: list[QCFlag] = []
    if not foreground.any():
        flags.append(
            QCFlag(
                code="tissue_envelope_empty_input",
                stage="body_surface",
                severity=QCSeverity.ERROR,
                reason="The tissue-derived body envelope received an empty segmentation.",
            )
        )
    elif not trunk.any():
        flags.append(
            QCFlag(
                code="tissue_envelope_no_trunk_component",
                stage="body_surface",
                severity=QCSeverity.ERROR,
                reason="No tissue component met the minimum area for a trunk envelope.",
                thresholds={"minimum_component_area_mm2": float(minimum_component_area_mm2)},
            )
        )
    return BodySurfaceResult(
        backend_id=TISSUE_ENVELOPE_BACKEND,
        geometry=geometry,
        body_mask_zyx=body,
        trunk_mask_zyx=trunk,
        provenance={
            "algorithm": "per_slice_component_physical_closing_convex_envelope_gaussian_smoothing",
            "source": "existing_tissue_segmentation",
            "minimum_component_area_mm2": float(minimum_component_area_mm2),
            "closing_radius_mm": float(closing_radius_mm),
            "smoothing_sigma_mm": float(smoothing_sigma_mm),
            "retained_component_count": retained_components,
            "discarded_component_count": discarded_components,
            "multi_component_slice_count": multi_component_slices,
            "validation_status": "technical_validation_pending",
            **dict(provenance or {}),
        },
        qc_flags=tuple(flags),
    )


def body_surface_from_totalsegmentator(
    labels_zyx: np.ndarray,
    geometry: ImageGeometry,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> BodySurfaceResult:
    """Adapt upstream Task 299 labels: 1=trunk and 2=extremities."""

    labels = validate_array_zyx(labels_zyx, geometry, "totalsegmentator_body_labels_zyx")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("TotalSegmentator body labels must be integer-valued.")
    unknown = sorted(int(value) for value in np.unique(labels) if int(value) not in {0, 1, 2})
    flags: list[QCFlag] = []
    if unknown:
        flags.append(
            QCFlag(
                code="body_surface_unknown_labels",
                stage="body_surface",
                severity=QCSeverity.ERROR,
                reason="The TotalSegmentator body task returned labels outside 0, 1, and 2.",
                observed={"labels": unknown},
            )
        )
    body = np.isin(labels, (1, 2))
    trunk = labels == 1
    if not body.any():
        flags.append(
            QCFlag(
                code="body_surface_empty",
                stage="body_surface",
                severity=QCSeverity.ERROR,
                reason="The body-surface backend returned an empty body mask.",
            )
        )
    if not trunk.any():
        flags.append(
            QCFlag(
                code="trunk_surface_empty",
                stage="body_surface",
                severity=QCSeverity.ERROR,
                reason="The body-surface backend returned an empty trunk mask.",
            )
        )
    trunk_components = _component_count(trunk)
    if trunk_components > 1:
        flags.append(
            QCFlag(
                code="trunk_surface_fragmented",
                stage="body_surface",
                reason="The three-dimensional trunk mask contains multiple components.",
                observed={"component_count": trunk_components},
            )
        )
    return BodySurfaceResult(
        backend_id=TOTALSEGMENTATOR_BODY_BACKEND,
        geometry=geometry,
        body_mask_zyx=body,
        trunk_mask_zyx=trunk,
        provenance={
            "upstream_project": "TotalSegmentator",
            "task": "body",
            "task_id": 299,
            "native_labels": {"1": "body_trunc", "2": "body_extremities"},
            **dict(provenance or {}),
        },
        qc_flags=tuple(flags),
    )


def deterministic_body_surface(
    image_zyx: np.ndarray,
    geometry: ImageGeometry,
    *,
    threshold_hu: float = -500.0,
    min_component_volume_mm3: float = 10_000.0,
) -> BodySurfaceResult:
    """Create the named non-model QC alternative from calibrated CT values."""

    if not np.isfinite(threshold_hu):
        raise ValueError("threshold_hu must be finite.")
    if not np.isfinite(min_component_volume_mm3) or min_component_volume_mm3 <= 0:
        raise ValueError("min_component_volume_mm3 must be finite and positive.")
    image = validate_array_zyx(image_zyx, geometry, "image_zyx")
    candidate = image > float(threshold_hu)
    candidate = ndimage.binary_closing(candidate, iterations=1)
    for slice_index_z in range(candidate.shape[0]):
        candidate[slice_index_z] = ndimage.binary_fill_holes(candidate[slice_index_z])

    labels, count = ndimage.label(
        candidate,
        structure=ndimage.generate_binary_structure(3, 2),
    )
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    minimum_voxels = max(1, int(np.ceil(min_component_volume_mm3 / geometry.voxel_volume_mm3)))
    retained_labels = np.flatnonzero(component_sizes >= minimum_voxels)
    if retained_labels.size:
        largest = int(retained_labels[np.argmax(component_sizes[retained_labels])])
        body = labels == largest
    else:
        body = np.zeros(candidate.shape, dtype=bool)

    trunk = np.zeros_like(body)
    for slice_index_z, body_slice in enumerate(body):
        slice_labels, slice_count = ndimage.label(
            body_slice,
            structure=ndimage.generate_binary_structure(2, 2),
        )
        if slice_count == 0:
            continue
        sizes = np.bincount(slice_labels.ravel())
        sizes[0] = 0
        trunk[slice_index_z] = ndimage.binary_fill_holes(slice_labels == int(np.argmax(sizes)))

    flags = [
        QCFlag(
            code="deterministic_body_surface_selected",
            stage="body_surface",
            reason=(
                "The deterministic threshold/connected-component backend was explicitly selected; "
                "its trunk semantics require separate validation."
            ),
            observed={
                "threshold_hu": float(threshold_hu),
                "initial_component_count": int(count),
            },
        )
    ]
    if not body.any() or not trunk.any():
        flags.append(
            QCFlag(
                code="deterministic_body_surface_empty",
                stage="body_surface",
                severity=QCSeverity.ERROR,
                reason="The deterministic body-surface backend produced an empty mask.",
            )
        )
    return BodySurfaceResult(
        backend_id=DETERMINISTIC_BODY_BACKEND,
        geometry=geometry,
        body_mask_zyx=body,
        trunk_mask_zyx=trunk,
        provenance={
            "algorithm": "hu_threshold_largest_3d_component_primary_2d_trunk",
            "threshold_hu": float(threshold_hu),
            "min_component_volume_mm3": float(min_component_volume_mm3),
        },
        qc_flags=tuple(flags),
    )
