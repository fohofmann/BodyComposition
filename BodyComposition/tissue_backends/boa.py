"""Narrow adapter for BOA's learned Task 542 body-region compartments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.utils.nifti import LPS_TO_RAS

BOA_BACKEND_ID = "boa_body_regions_task542_v1"
BOA_INPUT_PROFILE_ID = "boa_v1.0.2_ras_5mm_thickness_v1"
BOA_TARGET_THICKNESS_MM = 5.0

# BOA v1.0.2 ``BodyRegion`` values and the pinned Task 542 dataset metadata
# agree on this schema. Keep the upstream names in the persisted native
# compartment contract; the measurement layer owns any downstream aliases.
BOA_NATIVE_COMPARTMENT_LABELS: Mapping[int, str] = {
    1: "SUBCUTANEOUS_TISSUE",
    2: "MUSCLE",
    3: "ABDOMINAL_CAVITY",
    4: "THORACIC_CAVITY",
    5: "BONE",
    6: "GLANDS",
    7: "PERICARDIUM",
    8: "BREAST_IMPLANT",
    9: "MEDIASTINUM",
    10: "BRAIN",
    11: "NERVOUS_SYSTEM",
}

_THORACIC_STRUCTURE_LABELS = (4, 7, 9)
_SINGLE_COMPONENT_LABELS = (3, 7)


@dataclass(frozen=True)
class BoaInferenceContext:
    """Geometry needed to return a model-space prediction to the prepared CT."""

    prepared_geometry: ImageGeometry
    ras_geometry: ImageGeometry
    prepared_orientation: str
    model_shape_zyx: tuple[int, int, int]
    model_spacing_zyx: tuple[float, float, float]
    thickness_resampled: bool

    def provenance(self) -> dict[str, object]:
        return {
            "input_profile_id": BOA_INPUT_PROFILE_ID,
            "orientation": "RAS_model_space_copy",
            "target_slice_thickness_mm": BOA_TARGET_THICKNESS_MM,
            "interpolation_to_model_space": "cubic_order_3_nearest_boundary",
            "interpolation_to_prepared_domain": "nearest_neighbor",
            "model_shape_zyx": list(self.model_shape_zyx),
            "model_spacing_zyx": list(self.model_spacing_zyx),
            "thickness_resampled": self.thickness_resampled,
            "prepared_domain_preserved": True,
        }


def prepare_boa_inference(
    prepared_image: sitk.Image,
) -> tuple[np.ndarray, BoaInferenceContext]:
    """Create BOA v1.0.2 model input without changing the prepared CT.

    BOA first makes an RAS-oriented copy and then resamples only the slice
    direction to 5 mm with cubic interpolation.  nnU-Net's own anisotropic
    preprocessing uses a different through-plane interpolation, so this step
    must remain explicit for compatibility with the released Task 542 model.
    """

    if prepared_image.GetDimension() != 3:
        raise ValueError("BOA Task 542 requires one three-dimensional CT image.")
    prepared_geometry = ImageGeometry.from_sitk(prepared_image)
    prepared_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        prepared_image.GetDirection()
    )
    ras_image = sitk.DICOMOrient(prepared_image, "RAS")
    ras_geometry = ImageGeometry.from_sitk(ras_image)
    prepared_xyz = sitk.GetArrayFromImage(prepared_image).transpose(2, 1, 0)
    prepared_nifti = nib.Nifti1Image(
        prepared_xyz,
        LPS_TO_RAS @ prepared_geometry.affine_lps,
    )
    canonical = nib.as_closest_canonical(prepared_nifti)
    canonical_spacing_xyz = tuple(float(value) for value in canonical.header.get_zooms()[:3])
    source_thickness = canonical_spacing_xyz[2]
    thickness_resampled = not np.isclose(
        source_thickness,
        BOA_TARGET_THICKNESS_MM,
    )
    if thickness_resampled:
        model_xyz = ndimage.zoom(
            canonical.get_fdata(),
            zoom=(1.0, 1.0, source_thickness / BOA_TARGET_THICKNESS_MM),
            order=3,
            mode="nearest",
        ).astype(np.int32)
    else:
        model_xyz = canonical.get_fdata().astype(np.int32)
    model_zyx = model_xyz.transpose(2, 1, 0)
    if model_zyx.ndim != 3 or any(size <= 0 for size in model_zyx.shape):
        raise ValueError("BOA thickness staging produced an empty model input.")

    model_spacing_zyx = (
        BOA_TARGET_THICKNESS_MM,
        canonical_spacing_xyz[1],
        canonical_spacing_xyz[0],
    )
    model_shape_zyx = (
        int(model_zyx.shape[0]),
        int(model_zyx.shape[1]),
        int(model_zyx.shape[2]),
    )
    context = BoaInferenceContext(
        prepared_geometry=prepared_geometry,
        ras_geometry=ras_geometry,
        prepared_orientation=prepared_orientation,
        model_shape_zyx=model_shape_zyx,
        model_spacing_zyx=model_spacing_zyx,
        thickness_resampled=thickness_resampled,
    )
    return model_zyx, context


def restore_boa_prediction(
    prediction_zyx: np.ndarray,
    context: BoaInferenceContext,
) -> np.ndarray:
    """Restore Task 542 labels to the exact orientation-prepared CT domain."""

    prediction = np.asarray(prediction_zyx)
    if prediction.ndim != 3 or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError("BOA Task 542 prediction must be one integer z-y-x volume.")
    if prediction.shape != context.model_shape_zyx:
        raise ValueError(
            "BOA Task 542 prediction shape differs from its model input: "
            f"{prediction.shape} != {context.model_shape_zyx}."
        )

    target_shape_zyx = tuple(reversed(context.ras_geometry.size_xyz))
    if prediction.shape == target_shape_zyx:
        restored_ras_zyx = np.asarray(prediction, dtype=np.uint8)
    else:
        restored_ras_zyx = ndimage.zoom(
            prediction,
            zoom=tuple(
                target / source
                for target, source in zip(
                    target_shape_zyx,
                    prediction.shape,
                    strict=True,
                )
            ),
            order=0,
            mode="nearest",
        ).astype(np.uint8)
    if restored_ras_zyx.shape != target_shape_zyx:
        raise ValueError(
            "BOA label restoration produced the wrong RAS shape: "
            f"{restored_ras_zyx.shape} != {target_shape_zyx}."
        )

    restored_ras = sitk.GetImageFromArray(restored_ras_zyx)
    restored_ras.SetOrigin(context.ras_geometry.origin_lps_xyz)
    restored_ras.SetSpacing(context.ras_geometry.spacing_xyz)
    restored_ras.SetDirection(context.ras_geometry.direction_lps)
    restored = sitk.DICOMOrient(restored_ras, context.prepared_orientation)
    assert_same_physical_domain(
        context.prepared_geometry,
        ImageGeometry.from_sitk(restored),
        reference_name="orientation-prepared CT",
        candidate_name="restored BOA Task 542 labels",
    )
    return sitk.GetArrayFromImage(restored).astype(np.uint8, copy=False)


def _retain_largest_component(
    segmentation_zyx: np.ndarray,
    support_zyx: np.ndarray,
) -> None:
    from skimage.measure import label

    components = label(support_zyx)
    component_ids, counts = np.unique(components[components != 0], return_counts=True)
    if component_ids.size <= 1:
        return
    largest = int(component_ids[int(np.argmax(counts))])
    segmentation_zyx[(components != 0) & (components != largest)] = 0


def postprocess_boa_body_regions(prediction_zyx: np.ndarray) -> np.ndarray:
    """Apply BOA's component policy while retaining its native label values.

    BOA's public postprocessor marks discarded components with label 255 before
    its HU subclassification.  This adapter stores only anatomical
    compartments, so discarded voxels are background instead.  The retained
    support is otherwise equivalent: largest body, thoracic structure union,
    abdominal cavity, and pericardium components.
    """

    prediction = np.asarray(prediction_zyx)
    if prediction.ndim != 3 or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError("BOA Task 542 prediction must be one integer z-y-x volume.")
    unknown = sorted(
        int(value)
        for value in np.unique(prediction)
        if int(value) != 0 and int(value) not in BOA_NATIVE_COMPARTMENT_LABELS
    )
    if unknown:
        raise ValueError(f"BOA Task 542 returned unknown body-region labels: {unknown}.")

    result = np.asarray(prediction, dtype=np.uint8).copy()
    _retain_largest_component(result, result != 0)
    _retain_largest_component(result, np.isin(result, _THORACIC_STRUCTURE_LABELS))
    for native_label in _SINGLE_COMPONENT_LABELS:
        _retain_largest_component(result, result == native_label)
    return result
