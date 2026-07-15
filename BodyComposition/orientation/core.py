"""Orientation state machine, lossless repair, and review artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from itertools import permutations
from numbers import Real
from typing import Any, Mapping, Protocol

import cv2
import numpy as np
import SimpleITK as sitk

from BodyComposition.orientation.ctdeeprot import (
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    CODE_LICENSE,
    CTDeepRotPrediction,
    CTDeepRotPredictor,
    MODEL_CITATION_DOI,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)
from BodyComposition.orientation.rotations import (
    apply_volume_rotation,
    array_transform_spec,
    identity_class_index,
    inverse_class_index,
    is_axial_half_turn,
    rotation_class,
    signed_permutation_matrix,
    sitk_transform_spec,
)
from BodyComposition.utils.geometry import ImageGeometry


ORIENTATION_SCHEMA_VERSION = "1.0.0"
PHYSICAL_ROUNDTRIP_TOLERANCE_MM = 1e-4


class OrientationIntegrityError(RuntimeError):
    """Raised when orientation processing cannot produce a physical image."""


class OrientationState(str, Enum):
    PASS_METADATA_MATCH = "PASS_METADATA_MATCH"
    PASS_METADATA_UNCERTAIN = "PASS_METADATA_UNCERTAIN"
    MISMATCH_REPAIRED = "MISMATCH_REPAIRED"
    MISMATCH_UNCERTAIN = "MISMATCH_UNCERTAIN"
    HEADER_UNCERTAIN = "HEADER_UNCERTAIN"
    ORIENTATION_FAILED = "ORIENTATION_FAILED"


@dataclass(frozen=True)
class OrientationSettings:
    enabled: bool = True
    min_equivariant_votes: int = 23
    min_body_extent_mm: float = 250.0
    max_obliquity_deg: float = 15.0
    body_threshold_hu: float = -700.0
    min_body_pixels_per_slice: int = 100
    allow_axial_180_repair: bool = False
    metal_threshold_hu: float = 2500.0
    max_metal_fraction: float = 0.0005
    review_on_possible_truncation: bool = False
    report_enabled: bool = True
    checkpoint_path: Path = Path("./models/CTDeepRot/net2d.pt")
    checkpoint_sha256: str = CHECKPOINT_SHA256
    checkpoint_url: str = CHECKPOINT_URL
    auto_download: bool = True
    device: str = "cpu"
    batch_size: int = 24
    header_uncertain_reasons: tuple[str, ...] = ()
    force_review_reasons: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "OrientationSettings":
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise TypeError("Orientation configuration must be a mapping.")
        section = config.get("orientation", config)
        if not isinstance(section, Mapping):
            raise ValueError("orientation must be a mapping.")
        model = section.get("model", {})
        confidence = section.get("confidence", {})
        artifact = section.get("artifact_rules", {})
        report = section.get("report", {})
        for name, value in (
            ("orientation.model", model),
            ("orientation.confidence", confidence),
            ("orientation.artifact_rules", artifact),
            ("orientation.report", report),
        ):
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping.")
        header_reasons = section.get("header_uncertain_reasons", ())
        force_reasons = section.get("force_review_reasons", ())
        for name, reasons in (
            ("orientation.header_uncertain_reasons", header_reasons),
            ("orientation.force_review_reasons", force_reasons),
        ):
            if not isinstance(reasons, (list, tuple)) or any(
                not isinstance(reason, str) for reason in reasons
            ):
                raise ValueError(f"{name} must be a list or tuple of strings.")
        checkpoint_path = model.get(
            "checkpoint_path",
            "./models/CTDeepRot/net2d.pt",
        )
        if not isinstance(checkpoint_path, (str, Path)):
            raise ValueError("orientation.model.checkpoint_path must be a path.")
        return cls(
            enabled=section.get("enabled", True),
            min_equivariant_votes=confidence.get("min_equivariant_votes", 23),
            min_body_extent_mm=confidence.get("min_body_extent_mm", 250.0),
            max_obliquity_deg=confidence.get("max_obliquity_deg", 15.0),
            body_threshold_hu=confidence.get("body_threshold_hu", -700.0),
            min_body_pixels_per_slice=confidence.get("min_body_pixels_per_slice", 100),
            allow_axial_180_repair=confidence.get(
                "allow_axial_180_repair",
                False,
            ),
            metal_threshold_hu=artifact.get("metal_threshold_hu", 2500.0),
            max_metal_fraction=artifact.get("max_metal_fraction", 0.0005),
            review_on_possible_truncation=artifact.get(
                "review_on_possible_truncation",
                False,
            ),
            report_enabled=report.get("enabled", True),
            checkpoint_path=Path(checkpoint_path),
            checkpoint_sha256=model.get("sha256", CHECKPOINT_SHA256),
            checkpoint_url=model.get("download_url", CHECKPOINT_URL),
            auto_download=model.get("auto_download", True),
            device=model.get("device", "cpu"),
            batch_size=model.get("batch_size", 24),
            header_uncertain_reasons=tuple(header_reasons),
            force_review_reasons=tuple(force_reasons),
        )

    def validate(self) -> None:
        for name, value in (
            ("orientation.enabled", self.enabled),
            (
                "orientation.artifact_rules.review_on_possible_truncation",
                self.review_on_possible_truncation,
            ),
            ("orientation.report.enabled", self.report_enabled),
            ("orientation.model.auto_download", self.auto_download),
            (
                "orientation.confidence.allow_axial_180_repair",
                self.allow_axial_180_repair,
            ),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean.")
        if (
            isinstance(self.min_equivariant_votes, bool)
            or not isinstance(self.min_equivariant_votes, int)
            or not 1 <= self.min_equivariant_votes <= 24
        ):
            raise ValueError("orientation.confidence.min_equivariant_votes must be 1..24.")
        for name, value in (
            ("orientation.confidence.min_body_extent_mm", self.min_body_extent_mm),
            ("orientation.confidence.max_obliquity_deg", self.max_obliquity_deg),
            ("orientation.confidence.body_threshold_hu", self.body_threshold_hu),
            ("orientation.artifact_rules.metal_threshold_hu", self.metal_threshold_hu),
            ("orientation.artifact_rules.max_metal_fraction", self.max_metal_fraction),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.min_body_extent_mm < 0 or self.max_obliquity_deg < 0:
            raise ValueError("Orientation extent and obliquity thresholds must be non-negative.")
        if (
            isinstance(self.min_body_pixels_per_slice, bool)
            or not isinstance(self.min_body_pixels_per_slice, int)
            or self.min_body_pixels_per_slice < 1
        ):
            raise ValueError("min_body_pixels_per_slice must be positive.")
        if not 0 <= self.max_metal_fraction <= 1:
            raise ValueError("max_metal_fraction must be between zero and one.")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("orientation.model.device must be cpu, cuda, or auto.")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= 24
        ):
            raise ValueError("orientation.model.batch_size must be 1..24.")
        if (
            not isinstance(self.checkpoint_sha256, str)
            or len(self.checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.checkpoint_sha256)
        ):
            raise ValueError("orientation.model.sha256 must contain a SHA-256 digest.")
        if (
            not isinstance(self.checkpoint_url, str)
            or not self.checkpoint_url.startswith(("http://", "https://"))
        ):
            raise ValueError("orientation.model.download_url must be an HTTP(S) URL.")


@dataclass(frozen=True)
class ReviewFlag:
    code: str
    severity: str
    reason: str
    observed: Mapping[str, Any] = field(default_factory=dict)
    threshold: Mapping[str, Any] = field(default_factory=dict)
    suggested_action: str = "Review the orientation scout and source metadata."


@dataclass(frozen=True)
class OrientationResult:
    state: OrientationState
    execution_status: str
    qc_status: str
    manual_review_required: bool
    orientation_changed: bool
    severe_misorientation: bool
    original_orientation_code: str
    prepared_orientation_code: str
    direction_determinant: float
    residual_obliquity_deg: float
    original_geometry: Mapping[str, Any]
    prepared_geometry: Mapping[str, Any]
    original_pixel_sha256: str
    prepared_pixel_sha256: str
    original_affine_sha256: str
    prepared_affine_sha256: str
    prediction: Mapping[str, Any]
    body_extent_mm: float
    metal_fraction: float
    possible_fov_truncation: bool
    prepared_input: Mapping[str, Any]
    applied_transform: Mapping[str, Any] | None
    review_flags: tuple[ReviewFlag, ...]
    model: Mapping[str, Any]
    schema_version: str = ORIENTATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "execution_status": self.execution_status,
            "qc_status": self.qc_status,
            "manual_review_required": self.manual_review_required,
            "orientation_changed": self.orientation_changed,
            "severe_misorientation": self.severe_misorientation,
            "original_orientation_code": self.original_orientation_code,
            "prepared_orientation_code": self.prepared_orientation_code,
            "direction_determinant": self.direction_determinant,
            "residual_obliquity_deg": self.residual_obliquity_deg,
            "original_geometry": dict(self.original_geometry),
            "prepared_geometry": dict(self.prepared_geometry),
            "original_pixel_sha256": self.original_pixel_sha256,
            "prepared_pixel_sha256": self.prepared_pixel_sha256,
            "original_affine_sha256": self.original_affine_sha256,
            "prepared_affine_sha256": self.prepared_affine_sha256,
            "prediction": dict(self.prediction),
            "body_extent_mm": self.body_extent_mm,
            "metal_fraction": self.metal_fraction,
            "possible_fov_truncation": self.possible_fov_truncation,
            "prepared_input": dict(self.prepared_input),
            "applied_transform": (
                dict(self.applied_transform) if self.applied_transform is not None else None
            ),
            "review_flags": [asdict(flag) for flag in self.review_flags],
            "model": dict(self.model),
        }


@dataclass(frozen=True)
class OrientationOutcome:
    result: OrientationResult
    prepared_image: sitk.Image
    report_json: Path | None = None
    review_png: Path | None = None
    corrected_input: Path | None = None


class OrientationPredictor(Protocol):
    def predict(self, image: sitk.Image) -> CTDeepRotPrediction: ...


def _orientation_code(image: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
        image.GetDirection()
    )


def _geometry(image: sitk.Image) -> ImageGeometry:
    return ImageGeometry(
        size_xyz=image.GetSize(),
        spacing_xyz=image.GetSpacing(),
        origin_lps_xyz=image.GetOrigin(),
        direction_lps=image.GetDirection(),
    )


def _geometry_dict(image: sitk.Image) -> dict[str, Any]:
    geometry = _geometry(image)
    return {
        "size_xyz": list(geometry.size_xyz),
        "spacing_xyz": list(geometry.spacing_xyz),
        "origin_lps_xyz": list(geometry.origin_lps_xyz),
        "direction_lps": list(geometry.direction_lps),
    }


def _pixel_digest(image: sitk.Image) -> str:
    array_zyx = np.ascontiguousarray(sitk.GetArrayViewFromImage(image))
    digest = hashlib.sha256()
    digest.update(str(array_zyx.dtype).encode("ascii"))
    digest.update(np.asarray(array_zyx.shape, dtype=np.int64).tobytes())
    digest.update(array_zyx.tobytes())
    return digest.hexdigest()


def _affine_digest(image: sitk.Image) -> str:
    affine = _geometry(image).affine_lps.astype("<f8", copy=False)
    return hashlib.sha256(affine.tobytes()).hexdigest()


def _closest_direction(direction_lps: tuple[float, ...]) -> tuple[np.ndarray, float]:
    direction = np.asarray(direction_lps, dtype=float).reshape(3, 3)
    best_permutation = max(
        permutations(range(3)),
        key=lambda candidate: sum(abs(direction[candidate[column], column]) for column in range(3)),
    )
    discrete = np.zeros((3, 3), dtype=float)
    angles = []
    for column, row in enumerate(best_permutation):
        sign = 1.0 if direction[row, column] >= 0 else -1.0
        discrete[row, column] = sign
        dot = float(np.clip(np.dot(direction[:, column], discrete[:, column]), -1.0, 1.0))
        angles.append(math.degrees(math.acos(dot)))
    return discrete, float(max(angles))


def _validate_pixels(image: sitk.Image) -> None:
    if not isinstance(image, sitk.Image) or image.GetDimension() != 3:
        raise OrientationIntegrityError("A three-dimensional SimpleITK image is required.")
    try:
        _geometry(image).validate()
    except Exception as exc:
        raise OrientationIntegrityError(f"Invalid CT geometry: {exc}") from exc
    pixels = sitk.GetArrayViewFromImage(image)
    if not np.all(np.isfinite(pixels)):
        raise OrientationIntegrityError("CT pixels contain non-finite values.")


def _image_center_lps(image: sitk.Image) -> np.ndarray:
    center_index = tuple((length - 1.0) / 2.0 for length in image.GetSize())
    return np.asarray(image.TransformContinuousIndexToPhysicalPoint(center_index))


def _set_centered_geometry(
    image: sitk.Image,
    *,
    direction_lps: tuple[float, ...],
    center_lps: np.ndarray,
) -> sitk.Image:
    output = sitk.Image(image)
    output.SetDirection(direction_lps)
    direction = np.asarray(direction_lps, dtype=float).reshape(3, 3)
    center_index = (np.asarray(output.GetSize(), dtype=float) - 1.0) / 2.0
    offset = direction @ (np.asarray(output.GetSpacing()) * center_index)
    output.SetOrigin(tuple(float(value) for value in center_lps - offset))
    return output


def apply_model_rotation_to_sitk(
    image: sitk.Image,
    class_index: int,
    *,
    inverse: bool,
    target_direction_lps: tuple[float, ...] | None = None,
    target_center_lps: np.ndarray | None = None,
) -> sitk.Image:
    """Apply a model-array rotation with SimpleITK permutation and flip filters."""

    order_xyz, flips_xyz = sitk_transform_spec(class_index, inverse=inverse)
    output = sitk.PermuteAxes(image, order_xyz)
    output = sitk.Flip(output, flips_xyz, flipAboutOrigin=False)
    return _set_centered_geometry(
        output,
        direction_lps=target_direction_lps or image.GetDirection(),
        center_lps=(
            np.asarray(target_center_lps, dtype=float)
            if target_center_lps is not None
            else _image_center_lps(image)
        ),
    )


def _physical_roundtrip_error(left: sitk.Image, right: sitk.Image) -> float:
    if left.GetSize() != right.GetSize():
        return math.inf
    size = np.asarray(left.GetSize(), dtype=float)
    points = (
        np.zeros(3),
        size - 1.0,
        (size - 1.0) / 2.0,
        np.asarray((size[0] - 1.0, 0.0, size[2] - 1.0)),
    )
    errors = []
    for point in points:
        left_point = left.TransformContinuousIndexToPhysicalPoint(tuple(point))
        right_point = right.TransformContinuousIndexToPhysicalPoint(tuple(point))
        errors.append(np.linalg.norm(np.asarray(left_point) - np.asarray(right_point)))
    return float(max(errors))


def _repair(
    original: sitk.Image,
    class_index: int,
) -> tuple[sitk.Image, dict[str, Any]]:
    original_code = _orientation_code(original)
    metadata_canonical = sitk.DICOMOrient(original, "LPS")
    center_lps = _image_center_lps(metadata_canonical)
    corrected_working = apply_model_rotation_to_sitk(
        metadata_canonical,
        class_index,
        inverse=True,
        target_direction_lps=metadata_canonical.GetDirection(),
        target_center_lps=center_lps,
    )
    corrected = sitk.DICOMOrient(corrected_working, original_code)

    restored_working = sitk.DICOMOrient(corrected, "LPS")
    restored_canonical = apply_model_rotation_to_sitk(
        restored_working,
        class_index,
        inverse=False,
        target_direction_lps=metadata_canonical.GetDirection(),
        target_center_lps=center_lps,
    )
    restored_original = sitk.DICOMOrient(restored_canonical, original_code)
    exact_array_roundtrip = np.array_equal(
        sitk.GetArrayViewFromImage(original),
        sitk.GetArrayViewFromImage(restored_original),
    )
    physical_error = _physical_roundtrip_error(original, restored_original)
    if not exact_array_roundtrip or physical_error >= PHYSICAL_ROUNDTRIP_TOLERANCE_MM:
        raise OrientationIntegrityError(
            "Lossless orientation repair failed its inverse round-trip: "
            f"array_equal={exact_array_roundtrip}, physical_error_mm={physical_error:.6g}."
        )

    model_permutation, model_flips = array_transform_spec(class_index, inverse=True)
    sitk_order, sitk_flips = sitk_transform_spec(class_index, inverse=True)
    transform = {
        "predicted_class_index": class_index,
        "predicted_angles_deg": list(rotation_class(class_index).angles_deg),
        "applied_inverse_class_index": inverse_class_index(class_index),
        "model_array_order": "yxz",
        "output_axis_to_input_axis_yxz": list(model_permutation),
        "flip_output_axes_yxz": list(model_flips),
        "signed_permutation_output_to_input_yxz": signed_permutation_matrix(
            class_index,
            inverse=True,
        ).tolist(),
        "sitk_permute_order_xyz": list(sitk_order),
        "sitk_flip_axes_xyz": list(sitk_flips),
        "transform_steps": [
            {
                "operation": "DICOMOrient",
                "target_orientation_code": "LPS",
                "interpolation": "none",
            },
            {
                "operation": "PermuteAxes+Flip",
                "sitk_permute_order_xyz": list(sitk_order),
                "sitk_flip_axes_xyz": list(sitk_flips),
                "interpolation": "none",
            },
            {
                "operation": "DICOMOrient",
                "target_orientation_code": original_code,
                "interpolation": "none",
            },
        ],
        "final_orientation_code": original_code,
        "interpolation": "none",
        "center_preserved_lps": list(float(value) for value in center_lps),
        "inverse_array_exact": exact_array_roundtrip,
        "inverse_physical_point_max_error_mm": physical_error,
    }
    return corrected, transform


def _body_metrics(
    metadata_canonical: sitk.Image,
    prediction: CTDeepRotPrediction,
    settings: OrientationSettings,
) -> tuple[float, float, bool, bool]:
    array_zyx = sitk.GetArrayViewFromImage(metadata_canonical)
    array_yxz = np.transpose(array_zyx, (1, 2, 0))
    body = array_yxz > settings.body_threshold_hu
    metal_fraction = float(np.mean(array_yxz > settings.metal_threshold_hu))

    correction_permutation, _ = array_transform_spec(
        prediction.winning_class_index,
        inverse=True,
    )
    source_si_axis = correction_permutation[2]
    reduction_axes = tuple(axis for axis in range(3) if axis != source_si_axis)
    counts = np.count_nonzero(body, axis=reduction_axes)
    plane_size = int(np.prod([body.shape[axis] for axis in reduction_axes]))
    minimum_count = max(settings.min_body_pixels_per_slice, int(math.ceil(0.002 * plane_size)))
    occupied = np.flatnonzero(counts >= minimum_count)
    spacing_yxz = (
        metadata_canonical.GetSpacing()[1],
        metadata_canonical.GetSpacing()[0],
        metadata_canonical.GetSpacing()[2],
    )
    extent_mm = (
        float((occupied[-1] - occupied[0] + 1) * spacing_yxz[source_si_axis])
        if occupied.size
        else 0.0
    )

    corrected_body = apply_volume_rotation(
        body,
        inverse_class_index(prediction.winning_class_index),
    )
    boundary_fractions = []
    for axis in (0, 1):
        for position in (0, corrected_body.shape[axis] - 1):
            boundary = np.take(corrected_body, position, axis=axis)
            boundary_fractions.append(float(np.mean(boundary)))
    possible_truncation = any(fraction >= 0.01 for fraction in boundary_fractions)
    return extent_mm, metal_fraction, possible_truncation, occupied.size == 0


def _review_flags(
    *,
    prediction: CTDeepRotPrediction,
    extent_mm: float,
    metal_fraction: float,
    possible_truncation: bool,
    empty_body: bool,
    obliquity_deg: float,
    mismatch: bool,
    changed: bool,
    ambiguous_axial_half_turn: bool,
    settings: OrientationSettings,
) -> tuple[ReviewFlag, ...]:
    flags = []
    if prediction.agreement_count < settings.min_equivariant_votes:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_LOW_EQUIVARIANCE",
                severity="high" if mismatch else "medium",
                reason="CTDeepRot rotations did not produce the required consensus.",
                observed={"agreement_count": prediction.agreement_count},
                threshold={"minimum": settings.min_equivariant_votes, "total": 24},
            )
        )
    if empty_body:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_EMPTY_BODY_EXTENT",
                severity="high",
                reason="No adequate body cross-section was detected for coverage assessment.",
            )
        )
    elif extent_mm < settings.min_body_extent_mm:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_SHORT_FOV",
                severity="medium",
                reason="Detected cranio-caudal body coverage is below the repair threshold.",
                observed={"body_extent_mm": extent_mm},
                threshold={"minimum_mm": settings.min_body_extent_mm},
            )
        )
    if obliquity_deg > settings.max_obliquity_deg:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_SEVERE_OBLIQUITY",
                severity="high",
                reason="Header direction is too oblique for a discrete automatic repair.",
                observed={"residual_obliquity_deg": obliquity_deg},
                threshold={"maximum_deg": settings.max_obliquity_deg},
            )
        )
    if metal_fraction > settings.max_metal_fraction:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_METAL_ARTEFACT",
                severity="medium",
                reason="High-density voxel burden exceeds the configured orientation gate.",
                observed={"metal_fraction": metal_fraction},
                threshold={"maximum": settings.max_metal_fraction},
            )
        )
    if possible_truncation and settings.review_on_possible_truncation:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_POSSIBLE_TRUNCATION",
                severity="medium",
                reason="The detected body touches a transverse field-of-view boundary.",
            )
        )
    for reason in settings.force_review_reasons:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_FORCED_REVIEW",
                severity="high",
                reason=str(reason),
            )
        )
    for reason in settings.header_uncertain_reasons:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_HEADER_UNCERTAIN",
                severity="high",
                reason=str(reason),
                suggested_action="Verify the original DICOM geometry before adjudication.",
            )
        )
    if mismatch and not changed:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_MISMATCH_NOT_REPAIRED",
                severity="high",
                reason="Anatomy and metadata differ, but one or more repair gates failed.",
            )
        )
    if ambiguous_axial_half_turn and not settings.allow_axial_180_repair:
        flags.append(
            ReviewFlag(
                code="ORIENTATION_AMBIGUOUS_AXIAL_180",
                severity="high",
                reason=(
                    "The model proposed an SI-preserving 180-degree in-plane "
                    "rotation, which is advisory-only under the default safety policy."
                ),
                observed={
                    "class_index": prediction.winning_class_index,
                    "angles_deg": list(prediction.winning_angles_deg),
                },
                threshold={"automatic_repair_enabled": False},
                suggested_action=(
                    "Use the metadata and candidate scouts to adjudicate anterior, "
                    "posterior, left, and right before any manual correction."
                ),
            )
        )
    if changed:
        flags.extend(
            (
                ReviewFlag(
                    code="ORIENTATION_CHANGED",
                    severity="high",
                    reason="The prepared CT was losslessly reoriented before segmentation.",
                    suggested_action=(
                        "Confirm the corrected scout before accepting downstream measurements."
                    ),
                ),
                ReviewFlag(
                    code="SEVERE_METADATA_ANATOMY_MISMATCH",
                    severity="high",
                    reason="CTDeepRot indicated a discrete proper rotation inconsistent with metadata.",
                    observed={
                        "class_index": prediction.winning_class_index,
                        "angles_deg": list(prediction.winning_angles_deg),
                    },
                    suggested_action=(
                        "Compare original and corrected scouts and adjudicate the case."
                    ),
                ),
            )
        )
    return tuple(flags)


def _load_image(image: Any) -> sitk.Image:
    if isinstance(image, sitk.Image):
        return sitk.Image(image)
    if isinstance(image, (str, Path)):
        return sitk.ReadImage(str(image))
    try:
        from nibabel import Nifti1Image
    except ImportError:  # pragma: no cover
        Nifti1Image = ()
    if Nifti1Image and isinstance(image, Nifti1Image):
        from BodyComposition.utils.nifti import NiftiDataContainer

        container = NiftiDataContainer(Path("in_memory_orientation.nii.gz"))
        container.img = image
        return container.img
    raise TypeError("Orientation input must be a path, SimpleITK image, or Nifti1Image.")


def assess_orientation(
    image: Any,
    *,
    config: Mapping[str, Any] | None = None,
    predictor: OrientationPredictor | None = None,
    output_directory: str | Path | None = None,
    case_id: str = "case",
) -> OrientationOutcome:
    """Assess and, when safe, prepare a CT for downstream segmentation."""

    settings = OrientationSettings.from_mapping(config)
    settings.validate()
    original = _load_image(image)
    _validate_pixels(original)

    original_geometry = _geometry(original)
    discrete_direction, obliquity_deg = _closest_direction(original.GetDirection())
    direction_determinant = float(np.linalg.det(original_geometry.direction_matrix_lps))
    metadata_canonical = sitk.DICOMOrient(original, "LPS")

    if predictor is None:
        predictor = CTDeepRotPredictor(
            settings.checkpoint_path,
            auto_download=settings.auto_download,
            download_url=settings.checkpoint_url,
            expected_sha256=settings.checkpoint_sha256,
            device=settings.device,
            batch_size=settings.batch_size,
        )
    prediction = predictor.predict(metadata_canonical)
    mismatch = prediction.winning_class_index != identity_class_index()
    ambiguous_axial_half_turn = bool(
        mismatch and is_axial_half_turn(prediction.winning_class_index)
    )
    extent_mm, metal_fraction, possible_truncation, empty_body = _body_metrics(
        metadata_canonical,
        prediction,
        settings,
    )

    confidence_ok = prediction.agreement_count >= settings.min_equivariant_votes
    coverage_ok = extent_mm >= settings.min_body_extent_mm and not empty_body
    header_ok = obliquity_deg <= settings.max_obliquity_deg
    artifact_ok = metal_fraction <= settings.max_metal_fraction
    if settings.review_on_possible_truncation:
        artifact_ok = artifact_ok and not possible_truncation
    artifact_ok = (
        artifact_ok
        and not settings.force_review_reasons
        and not settings.header_uncertain_reasons
    )
    rotation_ok = (
        settings.allow_axial_180_repair or not ambiguous_axial_half_turn
    )
    repair_allowed = (
        confidence_ok
        and coverage_ok
        and header_ok
        and artifact_ok
        and rotation_ok
    )

    changed = bool(mismatch and repair_allowed)
    if not header_ok or settings.header_uncertain_reasons:
        state = OrientationState.HEADER_UNCERTAIN
    elif mismatch and changed:
        state = OrientationState.MISMATCH_REPAIRED
    elif mismatch:
        state = OrientationState.MISMATCH_UNCERTAIN
    elif repair_allowed:
        state = OrientationState.PASS_METADATA_MATCH
    else:
        state = OrientationState.PASS_METADATA_UNCERTAIN

    if changed:
        prepared, transform = _repair(original, prediction.winning_class_index)
    else:
        prepared = sitk.Image(original)
        transform = None

    flags = _review_flags(
        prediction=prediction,
        extent_mm=extent_mm,
        metal_fraction=metal_fraction,
        possible_truncation=possible_truncation,
        empty_body=empty_body,
        obliquity_deg=obliquity_deg,
        mismatch=mismatch,
        changed=changed,
        ambiguous_axial_half_turn=ambiguous_axial_half_turn,
        settings=settings,
    )
    manual_review = state is not OrientationState.PASS_METADATA_MATCH
    if changed:
        manual_review = True
    qc_status = "review" if manual_review else "pass"

    original_pixel_digest = _pixel_digest(original)
    prepared_pixel_digest = _pixel_digest(prepared)
    result = OrientationResult(
        state=state,
        execution_status="succeeded",
        qc_status=qc_status,
        manual_review_required=manual_review,
        orientation_changed=changed,
        severe_misorientation=changed,
        original_orientation_code=_orientation_code(original),
        prepared_orientation_code=_orientation_code(prepared),
        direction_determinant=direction_determinant,
        residual_obliquity_deg=obliquity_deg,
        original_geometry=_geometry_dict(original),
        prepared_geometry=_geometry_dict(prepared),
        original_pixel_sha256=original_pixel_digest,
        prepared_pixel_sha256=prepared_pixel_digest,
        original_affine_sha256=_affine_digest(original),
        prepared_affine_sha256=_affine_digest(prepared),
        prediction=prediction.to_dict(),
        body_extent_mm=extent_mm,
        metal_fraction=metal_fraction,
        possible_fov_truncation=possible_truncation,
        prepared_input={
            "kind": "corrected_derivative" if changed else "original",
            "relative_path": "corrected_input.nii.gz" if changed else None,
        },
        applied_transform=transform,
        review_flags=flags,
        model={
            "name": "CTDeepRot-2D",
            "required_for": "default_orientation_integrity_stage",
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "code_license": CODE_LICENSE,
            "citation_doi": MODEL_CITATION_DOI,
            "checkpoint_sha256": prediction.checkpoint_sha256,
            "metadata_discrete_direction_lps": discrete_direction.astype(int).tolist(),
        },
    )
    outcome = OrientationOutcome(result=result, prepared_image=prepared)
    if output_directory is not None and settings.report_enabled:
        report_json, review_png, corrected_input = write_orientation_artifacts(
            outcome,
            output_directory=output_directory,
            case_id=case_id,
            original_image=original,
        )
        outcome = OrientationOutcome(
            result=result,
            prepared_image=prepared,
            report_json=report_json,
            review_png=review_png,
            corrected_input=corrected_input,
        )
    return outcome


def _window_image(values: np.ndarray, low: float = -1000.0, high: float = 1000.0) -> np.ndarray:
    scaled = np.clip(values.astype(np.float32), low, high)
    scaled = ((scaled - low) * (255.0 / (high - low))).astype(np.uint8)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _central_slab_mean(
    array_zyx: np.ndarray,
    *,
    axis: int,
    spacing_mm: float,
    thickness_mm: float,
) -> np.ndarray:
    slab_voxels = max(1, min(array_zyx.shape[axis], round(thickness_mm / spacing_mm)))
    start = max(0, (array_zyx.shape[axis] - slab_voxels) // 2)
    stop = min(array_zyx.shape[axis], start + slab_voxels)
    selection = [slice(None)] * 3
    selection[axis] = slice(start, stop)
    return np.mean(array_zyx[tuple(selection)], axis=axis)


def _scale_view_to_physical_aspect(
    view: np.ndarray,
    *,
    row_spacing_mm: float,
    column_spacing_mm: float,
    max_pixels: int = 1000,
) -> np.ndarray:
    height_mm = view.shape[0] * row_spacing_mm
    width_mm = view.shape[1] * column_spacing_mm
    if height_mm <= 0 or width_mm <= 0:
        raise OrientationIntegrityError("Scout view has invalid physical dimensions.")
    if width_mm >= height_mm:
        width = max_pixels
        height = max(1, round(max_pixels * height_mm / width_mm))
    else:
        height = max_pixels
        width = max(1, round(max_pixels * width_mm / height_mm))
    return cv2.resize(view.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def _scouts(image: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    display_image = sitk.DICOMOrient(image, "LPS")
    array_zyx = sitk.GetArrayFromImage(display_image)
    spacing_xyz = display_image.GetSpacing()
    sagittal = _scale_view_to_physical_aspect(
        np.flipud(
            _central_slab_mean(
                array_zyx,
                axis=2,
                spacing_mm=spacing_xyz[0],
                thickness_mm=40.0,
            )
        ),
        row_spacing_mm=spacing_xyz[2],
        column_spacing_mm=spacing_xyz[1],
    )
    coronal = _scale_view_to_physical_aspect(
        np.flipud(
            _central_slab_mean(
                array_zyx,
                axis=1,
                spacing_mm=spacing_xyz[1],
                thickness_mm=40.0,
            )
        ),
        row_spacing_mm=spacing_xyz[2],
        column_spacing_mm=spacing_xyz[0],
    )
    axial = _scale_view_to_physical_aspect(
        _central_slab_mean(
            array_zyx,
            axis=0,
            spacing_mm=spacing_xyz[2],
            thickness_mm=5.0,
        ),
        row_spacing_mm=spacing_xyz[1],
        column_spacing_mm=spacing_xyz[0],
    )
    return tuple(_window_image(view) for view in (sagittal, coronal, axial))


def _annotate_panel(
    panel: np.ndarray,
    title: str,
    horizontal_left: str,
    horizontal_right: str,
    vertical_top: str,
    vertical_bottom: str,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, title, (12, 28), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, horizontal_left, (8, panel.shape[0] // 2), font, 0.7, (0, 255, 255), 2)
    cv2.putText(
        panel,
        horizontal_right,
        (panel.shape[1] - 28, panel.shape[0] // 2),
        font,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        vertical_top,
        (panel.shape[1] // 2, 58),
        font,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        vertical_bottom,
        (panel.shape[1] // 2, panel.shape[0] - 10),
        font,
        0.7,
        (0, 255, 255),
        2,
    )


def render_orientation_review(
    original_image: sitk.Image,
    outcome: OrientationOutcome,
) -> np.ndarray:
    """Render a deterministic, identifier-free orientation review sheet."""

    result = outcome.result
    mismatch = result.prediction["winning_class_index"] != identity_class_index()
    if mismatch and not result.orientation_changed:
        comparison_image, _ = _repair(
            original_image,
            result.prediction["winning_class_index"],
        )
        comparison_title = "Candidate (not applied)"
    else:
        comparison_image = outcome.prepared_image
        comparison_title = "Prepared"

    original_sagittal, original_coronal, original_axial = _scouts(original_image)
    prepared_sagittal, prepared_coronal, prepared_axial = _scouts(comparison_image)
    views = (
        (original_sagittal, "Metadata sagittal", "A", "P", "S", "I"),
        (original_coronal, "Metadata coronal", "R", "L", "S", "I"),
        (original_axial, "Metadata axial", "R", "L", "A", "P"),
        (
            prepared_sagittal,
            f"{comparison_title} sagittal",
            "A",
            "P",
            "S",
            "I",
        ),
        (
            prepared_coronal,
            f"{comparison_title} coronal",
            "R",
            "L",
            "S",
            "I",
        ),
        (
            prepared_axial,
            f"{comparison_title} axial",
            "R",
            "L",
            "A",
            "P",
        ),
    )
    canvas = np.full((1080, 1400, 3), 24, dtype=np.uint8)
    for index, (view, title, left, right, top, bottom) in enumerate(views):
        panel = _fit_panel(view, 450, 300)
        _annotate_panel(panel, title, left, right, top, bottom)
        row, column = divmod(index, 3)
        x = 10 + column * 465
        y = 10 + row * 310
        canvas[y : y + 300, x : x + 450] = panel

    if result.applied_transform is None:
        if mismatch:
            order, flips = sitk_transform_spec(
                result.prediction["winning_class_index"],
                inverse=True,
            )
            transform_line = (
                "Proposed, NOT APPLIED: SITK order "
                f"{order} | flips {flips} | interpolation none"
            )
        else:
            transform_line = "Applied transform: none"
    else:
        transform_line = (
            "Applied SITK order "
            f"{tuple(result.applied_transform['sitk_permute_order_xyz'])} | flips "
            f"{tuple(result.applied_transform['sitk_flip_axes_xyz'])} | interpolation none"
        )
    raw_class = result.prediction.get("raw_winning_class_index")
    raw_class_label = "n/a" if raw_class is None else str(raw_class)
    reference_class = result.prediction.get("model_reference_class_index")
    reference_class_label = "n/a" if reference_class is None else str(reference_class)
    lines = (
        f"Decision: {result.state.value}",
        f"QC: {result.qc_status} | manual review: {str(result.manual_review_required).lower()}",
        f"Metadata: {result.original_orientation_code} | prepared: {result.prepared_orientation_code}",
        (
            "CTDeepRot relative: class "
            f"{result.prediction['winning_class_index']} "
            f"{tuple(result.prediction['winning_angles_deg'])}"
        ),
        (
            "CTDeepRot raw/reference: "
            f"{raw_class_label}/{reference_class_label}"
        ),
        (
            f"Equivariance: {result.prediction['agreement_count']}/24 | "
            f"body extent: {result.body_extent_mm:.1f} mm"
        ),
        f"Orientation changed: {str(result.orientation_changed).lower()}",
        transform_line,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, line in enumerate(lines):
        color = (80, 190, 255) if result.orientation_changed and index in {0, 5} else (235, 235, 235)
        cv2.putText(canvas, line, (30, 675 + index * 43), font, 0.76, color, 2, cv2.LINE_AA)

    arrow_x = 1290
    cv2.arrowedLine(canvas, (arrow_x, 970), (arrow_x, 700), (0, 255, 255), 5, tipLength=0.1)
    cv2.putText(canvas, "HEAD / S", (1180, 685), font, 0.65, (0, 255, 255), 2)
    cv2.putText(canvas, "FEET / I", (1185, 1010), font, 0.65, (0, 255, 255), 2)

    flag_codes = ", ".join(flag.code for flag in result.review_flags) or "none"
    cv2.putText(canvas, f"Review flags: {flag_codes[:100]}", (30, 1040), font, 0.62, (210, 210, 210), 2)
    return canvas


def write_orientation_artifacts(
    outcome: OrientationOutcome,
    *,
    output_directory: str | Path,
    case_id: str,
    original_image: sitk.Image,
) -> tuple[Path, Path, Path | None]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "orientation_report.json"
    review_path = directory / "orientation_review.png"
    corrected_path = None
    if outcome.result.orientation_changed:
        corrected_path = directory / "corrected_input.nii.gz"
        sitk.WriteImage(outcome.prepared_image, str(corrected_path))
        persisted = sitk.ReadImage(str(corrected_path))
        if _pixel_digest(persisted) != outcome.result.prepared_pixel_sha256:
            raise OrientationIntegrityError(
                f"Persisted corrected input failed pixel verification: {corrected_path}."
            )
        persisted_physical_error = _physical_roundtrip_error(
            outcome.prepared_image,
            persisted,
        )
        if persisted_physical_error >= PHYSICAL_ROUNDTRIP_TOLERANCE_MM:
            raise OrientationIntegrityError(
                "Persisted corrected input failed physical-geometry verification: "
                f"maximum error {persisted_physical_error:.6g} mm."
            )
    payload = outcome.result.to_dict()
    payload["case_id"] = str(case_id)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    review = render_orientation_review(original_image, outcome)
    if not cv2.imwrite(str(review_path), review, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise OrientationIntegrityError(f"Could not write orientation review image: {review_path}.")
    return report_path, review_path, corrected_path


def write_orientation_failure_artifacts(
    *,
    output_directory: str | Path,
    case_id: str,
    error: Exception,
) -> tuple[Path, Path]:
    """Write a minimal identifier-free failure report before aborting a case."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "orientation_report.json"
    review_path = directory / "orientation_review.png"
    payload = {
        "schema_version": ORIENTATION_SCHEMA_VERSION,
        "case_id": str(case_id),
        "state": OrientationState.ORIENTATION_FAILED.value,
        "execution_status": "failed",
        "qc_status": "fail",
        "manual_review_required": True,
        "orientation_changed": False,
        "severe_misorientation": False,
        "error_type": type(error).__name__,
        "error_summary": (
            "Orientation processing failed before a physically valid prepared CT "
            "could be produced. Inspect the runtime log for technical details."
        ),
        "review_flags": [
            {
                "code": "ORIENTATION_PROCESSING_FAILED",
                "severity": "high",
                "reason": "No physically valid prepared CT could be produced.",
                "suggested_action": "Inspect the source pixels and geometry before retrying.",
            }
        ],
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    canvas = np.full((500, 1000, 3), 24, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "ORIENTATION FAILED",
        (40, 90),
        font,
        1.2,
        (80, 80, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, type(error).__name__, (40, 170), font, 0.8, (235, 235, 235), 2)
    summary = (
        "No physically valid prepared CT could be produced. "
        "Inspect the runtime log for technical details."
    )
    for line_index in range(0, min(len(summary), 240), 80):
        cv2.putText(
            canvas,
            summary[line_index : line_index + 80],
            (40, 230 + (line_index // 80) * 45),
            font,
            0.62,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(review_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise OrientationIntegrityError(f"Could not write orientation failure image: {review_path}.")
    return report_path, review_path
