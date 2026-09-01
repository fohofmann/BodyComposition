"""Strict, geometry-preserving DICOM CT series input and conversion."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import SimpleITK as sitk

from BodyComposition.provenance import file_sha256, image_pixel_sha256
from BodyComposition.schema_validation import validate_payload
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain

CONVERSION_METADATA_SCHEMA_VERSION = "1.5.0"
CONVERSION_METADATA_TYPE = "bodycomposition-dicom-conversion"
CONVERSION_BATCH_MANIFEST_NAME = "bodycomposition-batch.json"
CONVERSION_BATCH_REPORT_NAME = "bodycomposition-conversion.json"
# NIfTI-1 stores affine components as float32. Large, valid LPS origins can
# therefore shift by a few 1e-5 mm after a write/read round trip even though
# the voxel grid is unchanged. Keep this tolerance local to that file-format
# boundary; all other physical-domain comparisons retain GEOMETRY_ATOL.
NIFTI_ROUNDTRIP_GEOMETRY_ATOL = 1e-4
INSTANCE_ORIENTATION_ATOL = 1e-4
INSTANCE_ORIENTATION_EXACT_ATOL = 1e-6
AXIAL_NORMAL_MIN_ABS_Z = 0.98
SLICE_POSITION_DUPLICATE_ATOL_MM = 1e-5
# DICOM Image Position Patient commonly rounds sub-millimetre slice coordinates.
# Accept small representation noise, but still reject duplicated planes and a
# missing slice (whose delta is approximately twice the median spacing).
SLICE_SPACING_ABS_ATOL_MM = 1e-2
SLICE_SPACING_REL_ATOL = 1e-2
SLICE_IN_PLANE_DRIFT_ATOL_MM = 1e-1
PIXEL_SPACING_ABS_ATOL_MM = 1e-3
PIXEL_SPACING_REL_ATOL = 1e-3
SLICE_THICKNESS_ABS_ATOL_MM = 5e-2
SLICE_THICKNESS_REL_ATOL = 1e-2
CT_IMAGE_STORAGE_SOP_CLASS_UIDS = frozenset(
    {
        "1.2.840.10008.5.1.4.1.1.2",
        "1.2.840.10008.5.1.4.1.1.2.1",
        "1.2.840.10008.5.1.4.1.1.2.2",
    }
)
ENHANCED_CT_IMAGE_STORAGE_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.2.1"
SECONDARY_CAPTURE_SOP_CLASS_PREFIX = "1.2.840.10008.5.1.4.1.1.7"
ACCESSORY_IMAGE_TYPE_TOKENS = frozenset(
    {"LOCALIZER", "SCOUT", "TOPOGRAM", "SURVIEW", "SCREEN SAVE", "SCREENSHOT"}
)


class DicomInputError(ValueError):
    """Raised when a DICOM input cannot define one safe CT volume."""

    public_summary = "The DICOM input could not be read as one valid three-dimensional CT series."


class DicomSeriesSelectionError(DicomInputError):
    """Raised when DICOM series selection is absent, invalid, or ambiguous."""


class DicomConversionMetadataError(ValueError):
    """Raised when a conversion sidecar does not match its NIfTI CT."""

    public_summary = "The NIfTI conversion metadata is invalid or does not match the CT."


@dataclass(frozen=True)
class DicomSeriesInfo:
    """Minimal DICOM series facts, including the raw UID for local selection."""

    series_instance_uid: str
    modality: str
    instance_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "series_instance_uid": self.series_instance_uid,
            "modality": self.modality,
            "instance_count": self.instance_count,
        }


@dataclass(frozen=True)
class DicomSeriesSource:
    """One locally discovered DICOM series and its containing directory."""

    input_path: Path
    series_instance_uid: str
    modality: str
    instance_count: int


@dataclass(frozen=True)
class DicomStudySource:
    """One locally discovered study or one series without a Study UID."""

    input_path: Path
    study_instance_uid: str | None
    series_instance_uids: tuple[str, ...]
    series_count: int


def _conversion_case_id(input_summary: Mapping[str, Any]) -> str:
    dicom = input_summary.get("dicom")
    if isinstance(dicom, Mapping):
        study_digest = dicom.get("study_instance_uid_sha256")
        if isinstance(study_digest, str) and len(study_digest) == 64:
            return f"case-{study_digest[:16]}"
    return f"case-{str(input_summary['input_pixel_sha256'])[:16]}"


@dataclass(frozen=True)
class DicomConversionResult:
    """Verified result of converting one selected DICOM CT series."""

    output_path: Path
    output_content_sha256: str
    output_byte_size: int
    metadata_path: Path
    metadata_content_sha256: str
    metadata_byte_size: int
    conversion_created_at: str
    series_instance_uid: str
    input_summary: Mapping[str, Any]

    @property
    def case_id(self) -> str:
        return _conversion_case_id(self.input_summary)

    def as_dict(self) -> dict[str, Any]:
        dicom = dict(self.input_summary["dicom"])
        excluded_hashes = dicom.pop("excluded_instance_content_sha256", None)
        if isinstance(excluded_hashes, list):
            dicom["excluded_instance_content_sha256_count"] = len(excluded_hashes)
        sop_hashes = dicom.pop("sop_instance_uid_sha256", None)
        if isinstance(sop_hashes, list):
            dicom["sop_instance_uid_sha256_count"] = len(sop_hashes)
        return {
            "case_id": self.case_id,
            "output_path": self.output_path,
            "output_content_sha256": self.output_content_sha256,
            "output_byte_size": self.output_byte_size,
            "metadata_path": self.metadata_path,
            "metadata_content_sha256": self.metadata_content_sha256,
            "metadata_byte_size": self.metadata_byte_size,
            "conversion_created_at": self.conversion_created_at,
            "series_instance_uid": self.series_instance_uid,
            "input_format": "dicom",
            "source_content_sha256": self.input_summary["source_content_sha256"],
            "source_byte_size": self.input_summary["source_byte_size"],
            "input_pixel_sha256": self.input_summary["input_pixel_sha256"],
            "geometry": dict(self.input_summary["geometry"]),
            "dicom": dicom,
        }


@dataclass(frozen=True)
class DicomConversionFailure:
    """Privacy-safe failure for one discovered DICOM study group."""

    source_group_sha256: str
    candidate_series_instance_uid_sha256: tuple[str, ...]
    code: str
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_group_sha256": self.source_group_sha256,
            "candidate_series_instance_uid_sha256": list(self.candidate_series_instance_uid_sha256),
            "code": self.code,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class DicomConversionBatchResult:
    """Result of selecting and converting one CT stack per discovered study."""

    output_path: Path
    manifest_path: Path | None
    report_path: Path
    conversions: tuple[DicomConversionResult, ...]
    failures: tuple[DicomConversionFailure, ...]
    discovered_series_count: int
    discovered_study_count: int

    @property
    def succeeded(self) -> bool:
        return not self.failures and bool(self.conversions)

    @property
    def execution_status(self) -> str:
        return "succeeded" if self.succeeded else "failed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_status": self.execution_status,
            "output_path": self.output_path,
            "manifest_path": self.manifest_path,
            "report_path": self.report_path,
            "discovered_series_count": self.discovered_series_count,
            "discovered_study_count": self.discovered_study_count,
            "converted_count": len(self.conversions),
            "failed_count": len(self.failures),
            "conversions": [value.as_dict() for value in self.conversions],
            "failures": [value.as_dict() for value in self.failures],
        }


@dataclass(frozen=True)
class _DicomCandidate:
    info: DicomSeriesInfo
    files: tuple[Path, ...]


@dataclass(frozen=True)
class _DicomInstanceHeader:
    path: Path
    orientation_lps: tuple[float, ...] | None
    position_lps_xyz: tuple[float, ...] | None
    image_type_tokens: frozenset[str]
    acquisition_number: str | None
    patient_id: str | None
    study_instance_uid: str | None
    frame_of_reference_uid: str | None
    series_instance_uid: str | None
    sop_instance_uid: str | None
    sop_class_uid: str | None
    modality: str | None
    samples_per_pixel: int | None
    photometric_interpretation: str | None
    rows: int | None
    columns: int | None
    pixel_spacing_rc_mm: tuple[float, float] | None
    slice_thickness_mm: float | None
    rescale_intercept: float | None
    rescale_slope: float | None


@dataclass(frozen=True)
class _DicomStackMetrics:
    instance_count: int
    coverage_mm: float
    slice_spacing_mm: float
    axial_alignment: float
    effective_orientation_lps: tuple[float, ...]
    effective_pixel_spacing_rc_mm: tuple[float, float]
    effective_slice_thickness_mm: float | None
    rescale_intercept: float
    rescale_slope: float
    header_repairs: tuple[Mapping[str, Any], ...]

    @property
    def rank_key(self) -> tuple[int, int, int]:
        """Prefer coverage, then finer sampling, then more retained planes."""

        return (
            round(self.coverage_mm * 1000.0),
            -round(self.slice_spacing_mm * 1000.0),
            self.instance_count,
        )


@dataclass(frozen=True)
class _DicomInstanceSelection:
    files: tuple[Path, ...]
    discovered_instance_count: int
    strategy: str
    excluded_instance_content_sha256: tuple[str, ...]
    acquisition_number: str | None
    acquisition_strategy: str
    metrics: _DicomStackMetrics


@dataclass(frozen=True)
class _DicomSelectionDecision:
    candidate: _DicomCandidate
    instances: _DicomInstanceSelection
    series_strategy: str
    evaluated_ct_series_count: int
    eligible_stack_count: int
    eligible_candidates: tuple[Mapping[str, Any], ...]
    rejected_candidates: tuple[Mapping[str, Any], ...]

    @property
    def manual_review_recommended(self) -> bool:
        return (
            self.evaluated_ct_series_count > 1
            or self.eligible_stack_count > 1
            or bool(self.instances.excluded_instance_content_sha256)
            or bool(self.instances.metrics.header_repairs)
        )


def _metadata(reader: Any, key: str, *, index: int | None = None) -> str | None:
    try:
        if index is None:
            if not reader.HasMetaDataKey(key):
                return None
            value = reader.GetMetaData(key)
        else:
            if not reader.HasMetaDataKey(index, key):
                return None
            value = reader.GetMetaData(index, key)
    except RuntimeError:
        return None
    stripped = str(value).strip().strip("\x00")
    return stripped or None


def _numeric_metadata_tuple(
    reader: Any,
    key: str,
    *,
    count: int,
    index: int | None = None,
) -> tuple[float, ...] | None:
    raw = _metadata(reader, key, index=index)
    if raw is None:
        return None
    try:
        values = tuple(float(item.strip()) for item in raw.split("\\"))
    except ValueError:
        return None
    if len(values) != count or not all(np.isfinite(value) for value in values):
        return None
    return values


def _float_metadata(reader: Any, key: str) -> float | None:
    raw = _metadata(reader, key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _integer_metadata(reader: Any, key: str) -> int | None:
    raw = _metadata(reader, key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _instance_header(path: Path) -> _DicomInstanceHeader:
    reader = sitk.ImageFileReader()
    reader.SetImageIO("GDCMImageIO")
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOff()
    try:
        reader.ReadImageInformation()
    except RuntimeError as error:
        raise DicomInputError("A selected DICOM instance header could not be read.") from error
    image_type = _metadata(reader, "0008|0008") or ""
    rows = _integer_metadata(reader, "0028|0010")
    columns = _integer_metadata(reader, "0028|0011")
    pixel_spacing = _numeric_metadata_tuple(reader, "0028|0030", count=2)
    slice_thickness = _float_metadata(reader, "0018|0050")
    return _DicomInstanceHeader(
        path=path,
        orientation_lps=_numeric_metadata_tuple(reader, "0020|0037", count=6),
        position_lps_xyz=_numeric_metadata_tuple(reader, "0020|0032", count=3),
        image_type_tokens=frozenset(
            token.strip().upper() for token in image_type.split("\\") if token.strip()
        ),
        acquisition_number=_metadata(reader, "0020|0012"),
        patient_id=_metadata(reader, "0010|0020"),
        study_instance_uid=_metadata(reader, "0020|000d"),
        frame_of_reference_uid=_metadata(reader, "0020|0052"),
        series_instance_uid=_metadata(reader, "0020|000e"),
        sop_instance_uid=_metadata(reader, "0008|0018"),
        sop_class_uid=_metadata(reader, "0008|0016"),
        modality=(_metadata(reader, "0008|0060") or "").upper() or None,
        samples_per_pixel=_integer_metadata(reader, "0028|0002"),
        photometric_interpretation=((_metadata(reader, "0028|0004") or "").upper() or None),
        rows=rows,
        columns=columns,
        pixel_spacing_rc_mm=(
            (float(pixel_spacing[0]), float(pixel_spacing[1]))
            if pixel_spacing is not None
            else None
        ),
        slice_thickness_mm=slice_thickness,
        rescale_intercept=_float_metadata(reader, "0028|1052"),
        rescale_slope=_float_metadata(reader, "0028|1053"),
    )


def _orientation_normal(orientation: tuple[float, ...]) -> np.ndarray:
    row = np.asarray(orientation[:3], dtype=float)
    column = np.asarray(orientation[3:], dtype=float)
    normal = np.cross(row, column)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise DicomInputError("A selected DICOM instance has an invalid orientation.")
    return normal / norm


def _orientation_groups(
    headers: tuple[_DicomInstanceHeader, ...],
) -> tuple[tuple[_DicomInstanceHeader, ...], ...]:
    groups: list[list[_DicomInstanceHeader]] = []
    reference_orientations: list[tuple[float, ...]] = []
    for header in headers:
        if header.orientation_lps is None:
            raise DicomInputError(
                "The orthogonal-localizer exclusion requires Image Orientation Patient "
                "on every selected DICOM instance."
            )
        for index, reference in enumerate(reference_orientations):
            if np.allclose(
                header.orientation_lps,
                reference,
                rtol=0.0,
                atol=INSTANCE_ORIENTATION_ATOL,
            ):
                groups[index].append(header)
                break
        else:
            reference_orientations.append(header.orientation_lps)
            groups.append([header])
    return tuple(tuple(group) for group in groups)


def _is_secondary_capture(header: _DicomInstanceHeader) -> bool:
    return bool(
        header.sop_class_uid and header.sop_class_uid.startswith(SECONDARY_CAPTURE_SOP_CLASS_PREFIX)
    )


def _is_explicit_accessory_image(header: _DicomInstanceHeader) -> bool:
    return (
        bool(header.image_type_tokens & ACCESSORY_IMAGE_TYPE_TOKENS)
        or _is_secondary_capture(header)
        or (header.samples_per_pixel is not None and header.samples_per_pixel != 1)
        or (
            header.photometric_interpretation is not None
            and header.photometric_interpretation not in {"MONOCHROME1", "MONOCHROME2"}
        )
    )


def _is_supported_accessory_image(header: _DicomInstanceHeader) -> bool:
    tokens = header.image_type_tokens
    return _is_explicit_accessory_image(header) or {
        "DERIVED",
        "SECONDARY",
        "REFORMATTED",
    }.issubset(tokens)


def _repair_record(
    code: str,
    summary: str,
    *,
    observed_range: tuple[float, float] | None,
    applied_value: float | list[float] | None,
) -> Mapping[str, Any]:
    return {
        "code": code,
        "summary": summary,
        "observed_range": (
            [float(observed_range[0]), float(observed_range[1])]
            if observed_range is not None
            else None
        ),
        "applied_value": applied_value,
    }


def _assert_uniform_axial_slice_stack(
    headers: tuple[_DicomInstanceHeader, ...],
) -> _DicomStackMetrics:
    if len(headers) < 2:
        raise DicomInputError(
            "The retained axial DICOM orientation group has fewer than two instances."
        )
    for label, values in (
        ("patient identifier", {header.patient_id for header in headers}),
        ("Study Instance UID", {header.study_instance_uid for header in headers}),
        ("Series Instance UID", {header.series_instance_uid for header in headers}),
        ("Frame of Reference UID", {header.frame_of_reference_uid for header in headers}),
    ):
        if len(values) != 1:
            raise DicomInputError(
                f"The retained axial DICOM stack has inconsistent {label} values."
            )
    sop_instance_uids = [header.sop_instance_uid for header in headers]
    if any(value is None for value in sop_instance_uids) or len(set(sop_instance_uids)) != len(
        sop_instance_uids
    ):
        raise DicomInputError("The retained axial DICOM stack requires unique SOP Instance UIDs.")
    if any(header.modality != "CT" for header in headers):
        raise DicomInputError(
            "The retained DICOM stack does not contain consistently declared CT instances."
        )
    if any(header.sop_class_uid not in CT_IMAGE_STORAGE_SOP_CLASS_UIDS for header in headers):
        raise DicomInputError(
            "The retained DICOM stack contains a non-CT-image SOP class or lacks its SOP Class UID."
        )
    if any(header.samples_per_pixel != 1 for header in headers) or any(
        header.photometric_interpretation not in {"MONOCHROME1", "MONOCHROME2"}
        for header in headers
    ):
        raise DicomInputError(
            "The retained DICOM stack is not consistently single-channel monochrome CT."
        )
    if any(header.rescale_intercept is None for header in headers):
        raise DicomInputError(
            "The retained DICOM stack requires a finite Rescale Intercept on every instance."
        )
    if any(header.rescale_slope is None for header in headers):
        raise DicomInputError(
            "The retained DICOM stack requires a finite Rescale Slope on every instance."
        )
    rescale_intercepts = np.asarray(
        [header.rescale_intercept for header in headers],
        dtype=float,
    )
    rescale_slopes = np.asarray(
        [header.rescale_slope for header in headers],
        dtype=float,
    )
    if np.any(rescale_slopes <= 0.0):
        raise DicomInputError(
            "The retained DICOM stack requires a positive Rescale Slope on every instance."
        )
    if not np.allclose(rescale_intercepts, rescale_intercepts[0], rtol=0.0, atol=1e-6):
        raise DicomInputError("The retained DICOM stack has inconsistent Rescale Intercept values.")
    if not np.allclose(rescale_slopes, rescale_slopes[0], rtol=0.0, atol=1e-6):
        raise DicomInputError("The retained DICOM stack has inconsistent Rescale Slope values.")
    if any(header.orientation_lps is None for header in headers) or any(
        header.position_lps_xyz is None for header in headers
    ):
        raise DicomInputError(
            "The retained axial DICOM stack requires complete position and orientation tags."
        )
    orientations = np.asarray([header.orientation_lps for header in headers], dtype=float)
    reference_orientation = np.median(orientations, axis=0)
    if not np.allclose(
        orientations,
        reference_orientation,
        rtol=0.0,
        atol=INSTANCE_ORIENTATION_ATOL,
    ):
        raise DicomInputError("The retained axial DICOM stack has inconsistent image orientation.")
    row = reference_orientation[:3]
    row_norm = float(np.linalg.norm(row))
    column = reference_orientation[3:]
    if row_norm <= np.finfo(float).eps:
        raise DicomInputError("The retained DICOM orientation is invalid.")
    row = row / row_norm
    column = column - float(np.dot(column, row)) * row
    column_norm = float(np.linalg.norm(column))
    if column_norm <= np.finfo(float).eps:
        raise DicomInputError("The retained DICOM orientation is invalid.")
    column = column / column_norm
    effective_orientation = tuple(float(value) for value in np.concatenate((row, column)))
    normal = _orientation_normal(effective_orientation)
    axial_alignment = abs(float(normal[2]))
    if axial_alignment < AXIAL_NORMAL_MIN_ABS_Z:
        raise DicomInputError("The retained DICOM orientation group is not axial.")
    if any(header.rows is None or header.columns is None for header in headers):
        raise DicomInputError(
            "The retained axial DICOM stack requires Rows and Columns on every instance."
        )
    if len({(header.rows, header.columns) for header in headers}) != 1:
        raise DicomInputError(
            "The retained axial DICOM stack has inconsistent in-plane dimensions."
        )
    if any(header.pixel_spacing_rc_mm is None for header in headers):
        raise DicomInputError(
            "The retained axial DICOM stack requires Pixel Spacing on every instance."
        )
    pixel_spacings = np.asarray(
        [header.pixel_spacing_rc_mm for header in headers],
        dtype=float,
    )
    effective_pixel_spacing = np.median(pixel_spacings, axis=0)
    pixel_tolerance = np.maximum(
        PIXEL_SPACING_ABS_ATOL_MM,
        effective_pixel_spacing * PIXEL_SPACING_REL_ATOL,
    )
    if np.any(np.max(np.abs(pixel_spacings - effective_pixel_spacing), axis=0) > pixel_tolerance):
        raise DicomInputError(
            "The retained axial DICOM stack has inconsistent in-plane pixel spacing."
        )
    repairs: list[Mapping[str, Any]] = []
    orientation_delta = float(np.max(np.abs(orientations - reference_orientation)))
    if orientation_delta > INSTANCE_ORIENTATION_EXACT_ATOL:
        repairs.append(
            _repair_record(
                "normalize_orientation_rounding",
                "Normalized small per-instance orientation-vector rounding differences.",
                observed_range=(0.0, orientation_delta),
                applied_value=list(effective_orientation),
            )
        )
    pixel_delta = float(np.max(np.abs(pixel_spacings - effective_pixel_spacing)))
    if pixel_delta > 1e-6:
        repairs.append(
            _repair_record(
                "normalize_pixel_spacing_rounding",
                "Normalized small per-instance Pixel Spacing rounding differences.",
                observed_range=(
                    float(np.min(pixel_spacings)),
                    float(np.max(pixel_spacings)),
                ),
                applied_value=[float(value) for value in effective_pixel_spacing],
            )
        )
    declared_thicknesses = [header.slice_thickness_mm for header in headers]
    present_thicknesses = [value for value in declared_thicknesses if value is not None]
    effective_slice_thickness: float | None = None
    if present_thicknesses:
        if any(value <= 0 for value in present_thicknesses):
            raise DicomInputError(
                "The retained axial DICOM stack has an invalid declared slice thickness."
            )
        effective_slice_thickness = float(np.median(present_thicknesses))
        thickness_tolerance = max(
            SLICE_THICKNESS_ABS_ATOL_MM,
            effective_slice_thickness * SLICE_THICKNESS_REL_ATOL,
        )
        if (
            max(abs(value - effective_slice_thickness) for value in present_thicknesses)
            > thickness_tolerance
        ):
            raise DicomInputError(
                "The retained axial DICOM stack has materially inconsistent declared slice thickness."
            )
        if len(present_thicknesses) != len(headers) or not np.allclose(
            present_thicknesses,
            effective_slice_thickness,
            rtol=0.0,
            atol=1e-6,
        ):
            repairs.append(
                _repair_record(
                    "normalize_declared_slice_thickness",
                    "Used the median positive Slice Thickness after minor missing or rounded values.",
                    observed_range=(
                        float(min(present_thicknesses)),
                        float(max(present_thicknesses)),
                    ),
                    applied_value=effective_slice_thickness,
                )
            )
    positions = np.asarray([header.position_lps_xyz for header in headers], dtype=float)
    projected_unsorted = positions @ normal
    order = np.argsort(projected_unsorted)
    projected = projected_unsorted[order]
    ordered_positions = positions[order]
    deltas = np.diff(projected)
    if np.any(deltas <= SLICE_POSITION_DUPLICATE_ATOL_MM):
        raise DicomInputError("The retained axial DICOM stack contains duplicate slice positions.")
    median = float(np.median(deltas))
    tolerance = max(
        SLICE_SPACING_ABS_ATOL_MM,
        median * SLICE_SPACING_REL_ATOL,
    )
    if float(np.max(np.abs(deltas - median))) > tolerance:
        raise DicomInputError("The retained axial DICOM stack is not uniformly spaced.")
    displacement = np.diff(ordered_positions, axis=0)
    in_plane = displacement - deltas[:, None] * normal[None, :]
    max_in_plane_drift = float(np.max(np.linalg.norm(in_plane, axis=1)))
    if max_in_plane_drift > SLICE_IN_PLANE_DRIFT_ATOL_MM:
        raise DicomInputError(
            "The retained axial DICOM stack has inconsistent in-plane slice origins."
        )
    spacing_delta = float(np.max(np.abs(deltas - median)))
    if spacing_delta > SLICE_POSITION_DUPLICATE_ATOL_MM:
        repairs.append(
            _repair_record(
                "normalize_slice_position_rounding",
                "Used the median slice-centre spacing after minor Image Position rounding.",
                observed_range=(float(np.min(deltas)), float(np.max(deltas))),
                applied_value=median,
            )
        )
    if max_in_plane_drift > 1e-3:
        repairs.append(
            _repair_record(
                "normalize_slice_origin_rounding",
                "Normalized small in-plane Image Position rounding differences.",
                observed_range=(0.0, max_in_plane_drift),
                applied_value=0.0,
            )
        )
    return _DicomStackMetrics(
        instance_count=len(headers),
        coverage_mm=float(projected[-1] - projected[0]),
        slice_spacing_mm=median,
        axial_alignment=axial_alignment,
        effective_orientation_lps=effective_orientation,
        effective_pixel_spacing_rc_mm=(
            float(effective_pixel_spacing[0]),
            float(effective_pixel_spacing[1]),
        ),
        effective_slice_thickness_mm=effective_slice_thickness,
        rescale_intercept=float(rescale_intercepts[0]),
        rescale_slope=float(rescale_slopes[0]),
        header_repairs=tuple(repairs),
    )


def _select_header_subset(
    candidate: _DicomCandidate,
    all_headers: tuple[_DicomInstanceHeader, ...],
    headers: tuple[_DicomInstanceHeader, ...],
    *,
    acquisition_number: str | None,
    acquisition_strategy: str,
) -> _DicomInstanceSelection:
    if len(headers) == 1 and headers[0].sop_class_uid == ENHANCED_CT_IMAGE_STORAGE_SOP_CLASS_UID:
        raise DicomInputError(
            "Single-file Enhanced CT multi-frame objects are not supported. "
            "Provide a classic single-frame CT series or a validated NIfTI conversion."
        )
    headers = tuple(header for header in headers if not _is_explicit_accessory_image(header))
    if not headers:
        raise DicomInputError(
            "The selected DICOM series contains only localizer, secondary-capture, "
            "or non-monochrome accessory images."
        )
    missing_orientation = tuple(header for header in headers if header.orientation_lps is None)
    if any(not _is_supported_accessory_image(header) for header in missing_orientation):
        raise DicomInputError(
            "Safe axial-only DICOM conversion requires Image Orientation Patient on "
            "every non-accessory instance."
        )
    oriented = tuple(header for header in headers if header.orientation_lps is not None)
    if not oriented:
        raise DicomInputError("The selected DICOM series does not contain an oriented axial stack.")
    groups = _orientation_groups(oriented)
    if len(groups) == 1:
        retained = groups[0]
        metrics = _assert_uniform_axial_slice_stack(retained)
    else:
        axial_groups: list[tuple[tuple[_DicomInstanceHeader, ...], _DicomStackMetrics]] = []
        for group in groups:
            try:
                metrics = _assert_uniform_axial_slice_stack(group)
            except DicomInputError:
                continue
            axial_groups.append((group, metrics))
        if len(axial_groups) != 1:
            raise DicomInputError(
                "The selected DICOM Series Instance UID does not contain exactly one "
                "uniformly spaced axial orientation group."
            )
        retained, metrics = axial_groups[0]
        retained_paths = {header.path for header in retained}
        other_subset_headers = tuple(
            header for header in oriented if header.path not in retained_paths
        )
        if not all(_is_supported_accessory_image(header) for header in other_subset_headers):
            raise DicomInputError(
                "The selected DICOM Series Instance UID contains non-axial orientation "
                "groups that are not tagged LOCALIZER or "
                "DERIVED/SECONDARY/REFORMATTED."
            )
    if not retained:
        raise DicomInputError("The selected DICOM series does not contain a retained axial stack.")
    retained_paths = {header.path for header in retained}
    excluded = tuple(header for header in all_headers if header.path not in retained_paths)
    selected_files = tuple(path for path in candidate.files if path in retained_paths)
    if len(selected_files) != len(retained):
        raise DicomInputError("The retained DICOM instance selection is inconsistent.")
    return _DicomInstanceSelection(
        files=selected_files,
        discovered_instance_count=len(candidate.files),
        strategy=(
            "explicit-acquisition-number"
            if acquisition_strategy == "explicit-acquisition-number"
            else "automatic-best-axial-stack"
            if acquisition_strategy == "automatic-best-acquisition"
            else "exclude-tagged-accessory-instances"
            if excluded
            else "all-series-instances"
        ),
        excluded_instance_content_sha256=tuple(file_sha256(header.path) for header in excluded),
        acquisition_number=acquisition_number,
        acquisition_strategy=acquisition_strategy,
        metrics=metrics,
    )


def _candidate_instance_options(
    candidate: _DicomCandidate,
    *,
    acquisition_number: str | None,
    rejected_candidates: list[Mapping[str, Any]] | None = None,
) -> tuple[_DicomInstanceSelection, ...]:
    headers = tuple(_instance_header(path) for path in candidate.files)
    requested = acquisition_number.strip() if acquisition_number is not None else None
    if requested == "":
        raise DicomSeriesSelectionError("The requested Acquisition Number is empty.")
    if requested is not None:
        selected = tuple(header for header in headers if header.acquisition_number == requested)
        if not selected:
            available = sorted(
                {
                    header.acquisition_number
                    for header in headers
                    if header.acquisition_number is not None
                }
            )
            raise DicomSeriesSelectionError(
                f"Acquisition Number {requested!r} was not found. Available acquisition "
                f"numbers: {', '.join(available) or 'none'}."
            )
        return (
            _select_header_subset(
                candidate,
                headers,
                selected,
                acquisition_number=requested,
                acquisition_strategy="explicit-acquisition-number",
            ),
        )

    acquisition_values = sorted(
        {
            header.acquisition_number
            for header in headers
            if header.acquisition_number is not None and not _is_explicit_accessory_image(header)
        }
    )
    whole_series_error: DicomInputError | None = None
    try:
        return (
            _select_header_subset(
                candidate,
                headers,
                headers,
                acquisition_number=(
                    acquisition_values[0] if len(acquisition_values) == 1 else None
                ),
                acquisition_strategy="whole-series",
            ),
        )
    except DicomInputError as error:
        whole_series_error = error

    partition_headers = tuple(
        header for header in headers if not _is_explicit_accessory_image(header)
    )
    if (
        len(acquisition_values) > 1
        and partition_headers
        and all(header.acquisition_number is not None for header in partition_headers)
    ):
        options: list[_DicomInstanceSelection] = []
        for value in acquisition_values:
            selected = tuple(
                header for header in partition_headers if header.acquisition_number == value
            )
            try:
                options.append(
                    _select_header_subset(
                        candidate,
                        headers,
                        selected,
                        acquisition_number=value,
                        acquisition_strategy="automatic-best-acquisition",
                    )
                )
            except DicomInputError as error:
                if rejected_candidates is not None:
                    rejected_candidates.append(
                        {
                            "series_instance_uid_sha256": hashlib.sha256(
                                candidate.info.series_instance_uid.encode("ascii", errors="strict")
                            ).hexdigest(),
                            "acquisition_number": value,
                            "code": type(error).__name__,
                            "summary": str(error)[:240],
                        }
                    )
        if options:
            return tuple(options)

    assert whole_series_error is not None
    raise whole_series_error


def _select_candidate_instances(
    candidate: _DicomCandidate,
    *,
    acquisition_number: str | None = None,
) -> _DicomInstanceSelection:
    options = _candidate_instance_options(
        candidate,
        acquisition_number=acquisition_number,
        rejected_candidates=None,
    )
    if len(options) == 1:
        return options[0]
    best_key = max(option.metrics.rank_key for option in options)
    best = tuple(option for option in options if option.metrics.rank_key == best_key)
    if len(best) != 1:
        raise DicomSeriesSelectionError(
            "Multiple eligible axial acquisitions have identical quality metrics; "
            "select one explicitly with its Acquisition Number."
        )
    return best[0]


def _consistent_text_metadata(
    reader: Any,
    key: str,
    instance_count: int,
    *,
    maximum: int = 80,
) -> str | None:
    """Return one consistent printable value from the selected instances."""

    values = [
        _ascii_header_text(
            _metadata(reader, key, index=index),
            maximum=maximum,
        )
        for index in range(instance_count)
    ]
    if any(value is None for value in values) or len(set(values)) != 1:
        return None
    return values[0]


def _consistent_positive_float_metadata(
    reader: Any,
    key: str,
    instance_count: int,
) -> float | None:
    """Return one finite positive numeric DICOM value when all instances agree."""

    values: list[float] = []
    for index in range(instance_count):
        raw = _metadata(reader, key, index=index)
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if not np.isfinite(value) or value <= 0:
            return None
        values.append(value)
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
        return None
    return float(values[0])


def _read_modality(path: Path) -> str:
    reader = sitk.ImageFileReader()
    reader.SetImageIO("GDCMImageIO")
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOff()
    try:
        reader.ReadImageInformation()
    except RuntimeError as error:
        raise DicomInputError("A selected DICOM instance header could not be read.") from error
    return (_metadata(reader, "0008|0060") or "UNKNOWN").upper()


def _candidate_directories(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        return (source.parent,)
    directories: list[Path] = []
    for root, _, files in os.walk(source):
        if files:
            directories.append(Path(root))
    return tuple(sorted(directories, key=lambda item: item.as_posix()))


def _scan_candidates(source: str | Path) -> tuple[_DicomCandidate, ...]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input CT not found: {path}.")
    if not (path.is_file() or path.is_dir()):
        raise DicomInputError("The DICOM input must be a file or directory.")

    selected_file = path.resolve() if path.is_file() else None
    by_uid: dict[str, _DicomCandidate] = {}
    for directory in _candidate_directories(path):
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or ()
        except RuntimeError:
            continue
        for uid in sorted(str(value) for value in series_ids):
            try:
                names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), uid)
            except RuntimeError as error:
                raise DicomInputError(
                    "GDCM could not enumerate a discovered DICOM series."
                ) from error
            files = tuple(Path(name) for name in names)
            if not files:
                continue
            if selected_file is not None and selected_file not in {
                item.resolve() for item in files
            }:
                continue
            candidate = _DicomCandidate(
                info=DicomSeriesInfo(
                    series_instance_uid=uid,
                    modality=_read_modality(files[0]),
                    instance_count=len(files),
                ),
                files=files,
            )
            previous = by_uid.get(uid)
            if previous is not None and previous.files != candidate.files:
                raise DicomInputError(
                    "The same Series Instance UID occurs in multiple directories; "
                    "place the complete series in one directory before conversion."
                )
            by_uid[uid] = candidate

    return tuple(by_uid[key] for key in sorted(by_uid))


def _discover_candidates(source: str | Path) -> tuple[_DicomCandidate, ...]:
    candidates = _scan_candidates(source)
    if not candidates:
        raise DicomInputError("No readable DICOM series was found at the input path.")
    return candidates


def discover_dicom_series(source: str | Path) -> tuple[DicomSeriesInfo, ...]:
    """List DICOM series without returning patient, study, or path metadata."""

    return tuple(candidate.info for candidate in _discover_candidates(source))


def discover_dicom_series_sources(source: str | Path) -> tuple[DicomSeriesSource, ...]:
    """Return deterministic local sources for recursively discovered series."""

    return tuple(
        DicomSeriesSource(
            input_path=candidate.files[0].parent,
            series_instance_uid=candidate.info.series_instance_uid,
            modality=candidate.info.modality,
            instance_count=candidate.info.instance_count,
        )
        for candidate in _scan_candidates(source)
    )


def discover_dicom_study_count(source: str | Path) -> int:
    """Count locally distinct CT patient-study groups without exposing identifiers."""

    return len(discover_dicom_study_sources(source))


def _candidate_identity(
    candidate: _DicomCandidate,
) -> tuple[str | None, str | None]:
    all_headers = tuple(_instance_header(path) for path in candidate.files)
    headers = (
        tuple(header for header in all_headers if not _is_explicit_accessory_image(header))
        or all_headers
    )
    studies = {header.study_instance_uid for header in headers}
    patients = {header.patient_id for header in headers}
    series = {header.series_instance_uid for header in headers}
    if len(studies) != 1 or len(patients) != 1 or series != {candidate.info.series_instance_uid}:
        raise DicomInputError(
            "A DICOM series has inconsistent patient, study, or series identifiers."
        )
    return next(iter(patients)), next(iter(studies))


def _study_group_digest(series_instance_uids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(series_instance_uids)).encode("ascii", errors="strict")
    ).hexdigest()


def _study_candidate_groups(
    candidates: tuple[_DicomCandidate, ...],
) -> tuple[tuple[_DicomCandidate, ...], ...]:
    grouped: dict[tuple[str, str], list[_DicomCandidate]] = {}
    for candidate in candidates:
        if candidate.info.modality != "CT":
            continue
        first = _instance_header(candidate.files[0])
        key = (
            ("study", first.study_instance_uid)
            if first.study_instance_uid is not None
            else ("series", candidate.info.series_instance_uid)
        )
        grouped.setdefault(key, []).append(candidate)
    return tuple(
        tuple(sorted(values, key=lambda item: item.info.series_instance_uid))
        for _, values in sorted(grouped.items())
    )


def discover_dicom_study_sources(source: str | Path) -> tuple[DicomStudySource, ...]:
    """Return one deterministic local source for each discovered DICOM study."""

    groups = _study_candidate_groups(_scan_candidates(source))
    result: list[DicomStudySource] = []
    for group in groups:
        series_uids = tuple(candidate.info.series_instance_uid for candidate in group)
        first_header = _instance_header(group[0].files[0])
        parents = [str(candidate.files[0].parent.resolve()) for candidate in group]
        result.append(
            DicomStudySource(
                input_path=Path(os.path.commonpath(parents)),
                study_instance_uid=first_header.study_instance_uid,
                series_instance_uids=series_uids,
                series_count=len(series_uids),
            )
        )
    return tuple(result)


def _selection_candidate_summary(
    candidate: _DicomCandidate,
    selection: _DicomInstanceSelection,
) -> dict[str, Any]:
    return {
        "series_instance_uid_sha256": hashlib.sha256(
            candidate.info.series_instance_uid.encode("ascii", errors="strict")
        ).hexdigest(),
        "acquisition_number": selection.acquisition_number,
        "instance_count": selection.metrics.instance_count,
        "coverage_mm": selection.metrics.coverage_mm,
        "slice_spacing_mm": selection.metrics.slice_spacing_mm,
        "axial_alignment": selection.metrics.axial_alignment,
        "quality_rank": list(selection.metrics.rank_key),
    }


def _select_decision(
    source: str | Path,
    *,
    series_uid: str | None,
    study_uid: str | None = None,
    acquisition_number: str | None = None,
) -> _DicomSelectionDecision:
    candidates = _discover_candidates(source)
    if series_uid is not None and study_uid is not None:
        raise DicomSeriesSelectionError(
            "Select either a Study Instance UID group or an exact Series Instance UID, not both."
        )
    if study_uid is not None:
        requested_study = study_uid.strip()
        if not requested_study:
            raise DicomSeriesSelectionError("The requested Study Instance UID is empty.")
        matched: list[_DicomCandidate] = []
        for candidate in candidates:
            if candidate.info.modality != "CT":
                continue
            first_header = _instance_header(candidate.files[0])
            if first_header.study_instance_uid != requested_study:
                continue
            _, candidate_study = _candidate_identity(candidate)
            if candidate_study == requested_study:
                matched.append(candidate)
        if not matched:
            raise DicomSeriesSelectionError(
                "The requested DICOM study does not contain a readable CT series."
            )
        candidates = tuple(matched)
    series_strategy = "single-ct-series"
    if series_uid is not None:
        requested = series_uid.strip()
        if not requested:
            raise DicomSeriesSelectionError("The requested Series Instance UID is empty.")
        matches = [
            candidate for candidate in candidates if candidate.info.series_instance_uid == requested
        ]
        if not matches:
            available = ", ".join(candidate.info.series_instance_uid for candidate in candidates)
            raise DicomSeriesSelectionError(
                f"Series Instance UID {requested!r} was not found. Available series: {available}."
            )
        selected_candidate = matches[0]
        if selected_candidate.info.modality != "CT":
            raise DicomSeriesSelectionError(
                f"Series Instance UID {requested!r} has modality "
                f"{selected_candidate.info.modality!r}, not CT."
            )
        ct_candidates = [selected_candidate]
        series_strategy = "explicit-series-instance-uid"
    else:
        ct_candidates = [candidate for candidate in candidates if candidate.info.modality == "CT"]
    if not ct_candidates:
        modalities = ", ".join(
            f"{candidate.info.series_instance_uid} ({candidate.info.modality})"
            for candidate in candidates
        )
        raise DicomSeriesSelectionError(f"No CT series was found. Available series: {modalities}.")
    identities = tuple(_candidate_identity(candidate) for candidate in ct_candidates)
    if series_uid is None and len(ct_candidates) > 1:
        patient_ids = {patient for patient, _ in identities}
        study_uids = {study for _, study in identities}
        if None in study_uids or len(study_uids) != 1 or len(patient_ids) != 1:
            error = DicomSeriesSelectionError(
                "Multiple patients or DICOM studies were found. Point the command at one "
                "study, use a directory output for a cohort, or select an exact series."
            )
            error.public_summary = (
                "Multiple patients or DICOM studies were found; automatic best-series "
                "selection is limited to one study at a time."
            )
            raise error
        series_strategy = "automatic-best-series"

    options: list[tuple[_DicomCandidate, _DicomInstanceSelection]] = []
    rejected: list[Mapping[str, Any]] = []
    first_error: Exception | None = None
    for candidate in ct_candidates:
        try:
            candidate_rejections: list[Mapping[str, Any]] = []
            candidate_options = _candidate_instance_options(
                candidate,
                acquisition_number=acquisition_number,
                rejected_candidates=candidate_rejections,
            )
            options.extend((candidate, option) for option in candidate_options)
            rejected.extend(candidate_rejections)
        except (DicomInputError, DicomSeriesSelectionError) as error:
            if first_error is None:
                first_error = error
            rejected.append(
                {
                    "series_instance_uid_sha256": hashlib.sha256(
                        candidate.info.series_instance_uid.encode("ascii", errors="strict")
                    ).hexdigest(),
                    "code": type(error).__name__,
                    "summary": str(error)[:240],
                }
            )
    if not options:
        if len(ct_candidates) == 1 and first_error is not None:
            raise first_error
        raise DicomSeriesSelectionError(
            "None of the discovered CT series contains a complete, uniformly spaced axial stack."
        )

    best_key = max(selection.metrics.rank_key for _, selection in options)
    best = tuple(
        (candidate, selection)
        for candidate, selection in options
        if selection.metrics.rank_key == best_key
    )
    if len(best) != 1:
        raise DicomSeriesSelectionError(
            "Multiple CT series or acquisitions contain eligible axial stacks with "
            "identical quality metrics; select a Series Instance UID or Acquisition "
            "Number explicitly."
        )
    selected_candidate, selected_instances = best[0]
    eligible = tuple(
        _selection_candidate_summary(candidate, selection)
        for candidate, selection in sorted(
            options,
            key=lambda item: (
                item[0].info.series_instance_uid,
                item[1].acquisition_number or "",
            ),
        )
    )
    return _DicomSelectionDecision(
        candidate=selected_candidate,
        instances=selected_instances,
        series_strategy=series_strategy,
        evaluated_ct_series_count=len(ct_candidates),
        eligible_stack_count=len(options),
        eligible_candidates=eligible,
        rejected_candidates=tuple(rejected),
    )


def _select_candidate(
    source: str | Path,
    *,
    series_uid: str | None,
    study_uid: str | None = None,
    acquisition_number: str | None = None,
) -> _DicomCandidate:
    return _select_decision(
        source,
        series_uid=series_uid,
        study_uid=study_uid,
        acquisition_number=acquisition_number,
    ).candidate


def _ascii_header_text(value: str | None, *, maximum: int) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKD", value)
    text = normalized.encode("ascii", errors="ignore").decode("ascii")
    text = " ".join(text.replace("\x00", "").split())
    text = "".join(character for character in text if 32 <= ord(character) < 127)
    return text[:maximum].strip() or None


def _dicom_date(value: str | None) -> str | None:
    text = _ascii_header_text(value, maximum=32)
    if text is None:
        return None
    compact = text.replace("-", "").replace(".", "")
    if len(compact) < 8 or not compact[:8].isdigit():
        return None
    try:
        parsed = datetime.strptime(compact[:8], "%Y%m%d")
    except ValueError:
        return None
    return parsed.strftime("%Y-%m-%d")


def _consistent_float_metadata(
    reader: Any,
    key: str,
    instance_count: int,
) -> float | None:
    values: list[float] = []
    for index in range(instance_count):
        raw = _metadata(reader, key, index=index)
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if not np.isfinite(value):
            return None
        values.append(value)
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
        return None
    return float(values[0])


def _consistent_date_metadata(
    reader: Any,
    key: str,
    instance_count: int,
) -> str | None:
    values = {_dicom_date(_metadata(reader, key, index=index)) for index in range(instance_count)}
    return next(iter(values)) if len(values) == 1 and None not in values else None


def _consistent_hashed_identifier_metadata(
    reader: Any,
    key: str,
    instance_count: int,
) -> str | None:
    values = {_metadata(reader, key, index=index) for index in range(instance_count)}
    if len(values) != 1 or None in values:
        return None
    value = next(iter(values))
    assert value is not None
    return hashlib.sha256(value.encode("ascii", errors="strict")).hexdigest()


def _sop_instance_uid_hashes(reader: Any, instance_count: int) -> list[str] | None:
    values = [_metadata(reader, "0008|0018", index=index) for index in range(instance_count)]
    if any(value is None for value in values) or len(set(values)) != len(values):
        return None
    return [
        hashlib.sha256(str(value).encode("ascii", errors="strict")).hexdigest() for value in values
    ]


def _image_type_values(reader: Any, instance_count: int) -> list[str]:
    values = {
        "\\".join(
            token.strip().upper()
            for token in (_metadata(reader, "0008|0008", index=index) or "").split("\\")
            if token.strip()
        )
        for index in range(instance_count)
    }
    values.discard("")
    return sorted(values)


def _dicom_person_name(value: str | None) -> str | None:
    text = _ascii_header_text(value, maximum=160)
    if text is None:
        return None
    primary = text.split("=", maxsplit=1)[0]
    components = [component.strip() for component in primary.split("^")]
    components.extend([""] * (5 - len(components)))
    family, given, middle, prefix, suffix = components[:5]
    display = " ".join(
        component for component in (prefix, given, middle, family, suffix) if component
    )
    return _ascii_header_text(display or primary, maximum=96)


def _consistent_dicom_header_value(
    files: tuple[Path, ...],
    key: str,
) -> str | None:
    """Return a standard tag only when the first and last instances agree."""

    indices = (0,) if len(files) == 1 else (0, len(files) - 1)
    values: list[str | None] = []
    for index in indices:
        reader = sitk.ImageFileReader()
        reader.SetImageIO("GDCMImageIO")
        reader.SetFileName(str(files[index]))
        reader.LoadPrivateTagsOff()
        try:
            reader.ReadImageInformation()
        except RuntimeError:
            return None
        values.append(_metadata(reader, key))
    return values[0] if len(set(values)) == 1 else None


def _consistent_dicom_date(
    files: tuple[Path, ...],
    key: str,
) -> str | None:
    """Return one calendar date even when acquisition times differ by slice."""

    indices = (0,) if len(files) == 1 else (0, len(files) - 1)
    values: list[str | None] = []
    for index in indices:
        reader = sitk.ImageFileReader()
        reader.SetImageIO("GDCMImageIO")
        reader.SetFileName(str(files[index]))
        reader.LoadPrivateTagsOff()
        try:
            reader.ReadImageInformation()
        except RuntimeError:
            return None
        values.append(_dicom_date(_metadata(reader, key)))
    return values[0] if values[0] is not None and len(set(values)) == 1 else None


def dicom_report_patient_metadata(
    source: str | Path,
    *,
    series_uid: str | None = None,
    study_uid: str | None = None,
    acquisition_number: str | None = None,
) -> dict[str, str]:
    """Read the minimal local DICOM demographics requested for a patient PDF.

    These values are intentionally separate from the path-free technical input
    provenance and must not be copied into canonical case manifests or logs.
    """

    decision = _select_decision(
        source,
        series_uid=series_uid,
        study_uid=study_uid,
        acquisition_number=acquisition_number,
    )
    files = decision.instances.files
    patient_name = _dicom_person_name(_consistent_dicom_header_value(files, "0010|0010"))
    date_of_birth = _dicom_date(_consistent_dicom_header_value(files, "0010|0030"))
    sex_value = _ascii_header_text(
        _consistent_dicom_header_value(files, "0010|0040"),
        maximum=8,
    )
    sex = sex_value.upper() if sex_value and sex_value.upper() in {"F", "M", "O", "U"} else None
    scan_date = next(
        (
            parsed
            for key in ("0008|002a", "0008|0022", "0008|0021", "0008|0020", "0008|0023")
            if (parsed := _consistent_dicom_date(files, key)) is not None
        ),
        None,
    )
    return {
        key: value
        for key, value in (
            ("patient_name", patient_name),
            ("date_of_birth", date_of_birth),
            ("scan_date", scan_date),
            ("sex", sex),
        )
        if value is not None
    }


def _series_content_identity(files: tuple[Path, ...]) -> tuple[str, int]:
    digest = hashlib.sha256(b"BodyComposition-DICOM-series-v1\x00")
    byte_size = 0
    for path in files:
        size = path.stat().st_size
        byte_size += size
        digest.update(size.to_bytes(8, byteorder="big", signed=False))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), byte_size


def _geometry_summary(image: sitk.Image) -> dict[str, Any]:
    geometry = ImageGeometry.from_sitk(image)
    geometry.validate()
    return {
        "size_xyz": list(geometry.size_xyz),
        "spacing_xyz": list(geometry.spacing_xyz),
        "origin_lps_xyz": list(geometry.origin_lps_xyz),
        "direction_lps": list(geometry.direction_lps),
    }


def _apply_minor_header_repairs(
    image: sitk.Image,
    metrics: _DicomStackMetrics,
) -> sitk.Image:
    if not metrics.header_repairs:
        return image
    row = np.asarray(metrics.effective_orientation_lps[:3], dtype=float)
    column = np.asarray(metrics.effective_orientation_lps[3:], dtype=float)
    normal = np.cross(row, column)
    existing_direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    if float(np.dot(existing_direction[:, 2], normal)) < 0:
        normal = -normal
    direction = np.column_stack((row, column, normal))
    repaired = sitk.Image(image)
    repaired.SetDirection(tuple(float(value) for value in direction.ravel()))
    repaired.SetSpacing(
        (
            metrics.effective_pixel_spacing_rc_mm[1],
            metrics.effective_pixel_spacing_rc_mm[0],
            metrics.slice_spacing_mm,
        )
    )
    return repaired


def _execute_selected_dicom_series(
    files: tuple[Path, ...],
) -> tuple[sitk.ImageSeriesReader, sitk.Image]:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in files])
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    try:
        image = reader.Execute()
    except RuntimeError as error:
        raise DicomInputError("GDCM could not assemble the selected DICOM CT series.") from error
    if image.GetDimension() != 3:
        raise DicomInputError("The selected DICOM CT series is not three-dimensional.")
    return reader, image


def _read_selected_dicom_series(
    selection: _DicomInstanceSelection,
) -> tuple[sitk.ImageSeriesReader, sitk.Image, _DicomInstanceSelection]:
    """Read selected instances in the reader's physical through-plane order.

    GDCM preserves the file-name order supplied to ``ImageSeriesReader``, but
    its output slice direction can legitimately oppose the orientation normal
    when a scanner declares negative Spacing Between Slices. Discovery order
    is therefore insufficient after a localizer or acquisition subset has been
    removed. Read once to obtain GDCM's physical slice axis, then reorder the
    selected instances by Image Position Patient along that axis when needed.
    """

    reader, image = _execute_selected_dicom_series(selection.files)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    slice_axis = direction[:, 2]
    positions = []
    for index in range(len(selection.files)):
        position = _numeric_metadata_tuple(
            reader,
            "0020|0032",
            count=3,
            index=index,
        )
        if position is None:
            raise DicomInputError(
                "The selected axial DICOM stack requires complete Image Position Patient metadata."
            )
        positions.append(position)
    projections = np.asarray(positions, dtype=float) @ slice_axis
    order = tuple(int(value) for value in np.argsort(projections, kind="stable"))
    identity_order = tuple(range(len(selection.files)))
    if order != identity_order:
        ordered_files = tuple(selection.files[index] for index in order)
        reader, image = _execute_selected_dicom_series(ordered_files)
        selection = replace(selection, files=ordered_files)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        slice_axis = direction[:, 2]
        positions = []
        for index in range(len(selection.files)):
            position = _numeric_metadata_tuple(
                reader,
                "0020|0032",
                count=3,
                index=index,
            )
            if position is None:
                raise DicomInputError(
                    "The selected axial DICOM stack requires complete Image Position Patient metadata."
                )
            positions.append(position)
        projections = np.asarray(positions, dtype=float) @ slice_axis
    if np.any(np.diff(projections) <= SLICE_POSITION_DUPLICATE_ATOL_MM):
        raise DicomInputError(
            "GDCM could not assemble the selected DICOM CT series in physical slice order."
        )
    return reader, image, selection


def dicom_image_summary(
    source: str | Path,
    *,
    series_uid: str | None = None,
    study_uid: str | None = None,
    acquisition_number: str | None = None,
) -> tuple[sitk.Image, dict[str, Any], str]:
    """Read the best verified axial DICOM CT stack plus allowlisted provenance."""

    decision = _select_decision(
        source,
        series_uid=series_uid,
        study_uid=study_uid,
        acquisition_number=acquisition_number,
    )
    candidate = decision.candidate
    selection = decision.instances
    reader, image, selection = _read_selected_dicom_series(selection)
    image = _apply_minor_header_repairs(image, selection.metrics)

    pixels = sitk.GetArrayViewFromImage(image)
    if not np.all(np.isfinite(pixels)):
        raise DicomInputError("The selected DICOM CT contains non-finite pixels.")
    geometry_summary = _geometry_summary(image)

    modalities = {
        (_metadata(reader, "0008|0060", index=index) or "UNKNOWN").upper()
        for index in range(len(selection.files))
    }
    if modalities != {"CT"}:
        raise DicomInputError(
            "The selected DICOM series does not contain consistently declared CT instances."
        )
    orientation_complete = all(
        _metadata(reader, "0020|0037", index=index) is not None
        for index in range(len(selection.files))
    )
    position_complete = all(
        _metadata(reader, "0020|0032", index=index) is not None
        for index in range(len(selection.files))
    )
    source_digest, source_byte_size = _series_content_identity(selection.files)
    uid_digest = hashlib.sha256(
        candidate.info.series_instance_uid.encode("ascii", errors="strict")
    ).hexdigest()
    instance_count = len(selection.files)
    summary = {
        "source_reference": "content-addressed-local-input",
        "source_content_sha256": source_digest,
        "source_byte_size": source_byte_size,
        "input_pixel_sha256": image_pixel_sha256(image),
        "input_format": "dicom",
        "geometry": geometry_summary,
        "dicom": {
            "series_instance_uid_sha256": uid_digest,
            "instance_count": len(selection.files),
            "discovered_instance_count": selection.discovered_instance_count,
            "excluded_instance_count": (selection.discovered_instance_count - len(selection.files)),
            "instance_selection": selection.strategy,
            "excluded_instance_content_sha256": list(selection.excluded_instance_content_sha256),
            "selection": {
                "series_strategy": decision.series_strategy,
                "acquisition_strategy": selection.acquisition_strategy,
                "evaluated_ct_series_count": decision.evaluated_ct_series_count,
                "eligible_stack_count": decision.eligible_stack_count,
                "manual_review_recommended": decision.manual_review_recommended,
                "selected_metrics": {
                    "instance_count": selection.metrics.instance_count,
                    "coverage_mm": selection.metrics.coverage_mm,
                    "slice_spacing_mm": selection.metrics.slice_spacing_mm,
                    "axial_alignment": selection.metrics.axial_alignment,
                    "quality_rank": list(selection.metrics.rank_key),
                },
                "header_repairs": [dict(value) for value in selection.metrics.header_repairs],
                "eligible_candidates": [dict(value) for value in decision.eligible_candidates],
                "rejected_candidates": [dict(value) for value in decision.rejected_candidates],
            },
            "modality": "CT",
            "scanner_manufacturer": _consistent_text_metadata(reader, "0008|0070", instance_count),
            "scanner_model": _consistent_text_metadata(reader, "0008|1090", instance_count),
            "slice_thickness_mm": selection.metrics.effective_slice_thickness_mm,
            "acquisition_date": (
                _consistent_date_metadata(reader, "0008|0022", instance_count)
                or _consistent_date_metadata(reader, "0008|002a", instance_count)
            ),
            "acquisition_datetime": _consistent_text_metadata(reader, "0008|002a", instance_count),
            "study_date": _consistent_date_metadata(reader, "0008|0020", instance_count),
            "series_date": _consistent_date_metadata(reader, "0008|0021", instance_count),
            "content_date": _consistent_date_metadata(reader, "0008|0023", instance_count),
            "instance_creation_date": _consistent_date_metadata(
                reader, "0008|0012", instance_count
            ),
            "instance_creation_time": _consistent_text_metadata(
                reader, "0008|0013", instance_count
            ),
            "series_number": _consistent_text_metadata(reader, "0020|0011", instance_count),
            "acquisition_number": (
                selection.acquisition_number
                or _consistent_text_metadata(reader, "0020|0012", instance_count)
            ),
            "series_description": _consistent_text_metadata(
                reader, "0008|103e", instance_count, maximum=240
            ),
            "protocol_name": _consistent_text_metadata(
                reader, "0018|1030", instance_count, maximum=240
            ),
            "study_description": _consistent_text_metadata(
                reader, "0008|1030", instance_count, maximum=240
            ),
            "body_part_examined": _consistent_text_metadata(
                reader, "0018|0015", instance_count, maximum=240
            ),
            "patient_position": _consistent_text_metadata(reader, "0018|5100", instance_count),
            "contrast_bolus_agent": _consistent_text_metadata(
                reader, "0018|0010", instance_count, maximum=240
            ),
            "contrast_bolus_route": _consistent_text_metadata(
                reader, "0018|1040", instance_count, maximum=240
            ),
            "contrast_bolus_volume_ml": _consistent_positive_float_metadata(
                reader, "0018|1041", instance_count
            ),
            "study_instance_uid_sha256": _consistent_hashed_identifier_metadata(
                reader, "0020|000d", instance_count
            ),
            "frame_of_reference_uid_sha256": _consistent_hashed_identifier_metadata(
                reader, "0020|0052", instance_count
            ),
            "sop_instance_uid_sha256": _sop_instance_uid_hashes(reader, instance_count),
            "image_type_values": _image_type_values(reader, instance_count),
            "convolution_kernel": _consistent_text_metadata(reader, "0018|1210", instance_count),
            "kvp": _consistent_positive_float_metadata(reader, "0018|0060", instance_count),
            "pixel_spacing_row_column_mm": list(selection.metrics.effective_pixel_spacing_rc_mm),
            "spacing_between_slices_mm": _consistent_float_metadata(
                reader, "0018|0088", instance_count
            ),
            "reconstruction_diameter_mm": _consistent_positive_float_metadata(
                reader, "0018|1100", instance_count
            ),
            "rescale_intercept": selection.metrics.rescale_intercept,
            "rescale_slope": selection.metrics.rescale_slope,
            "gantry_detector_tilt_degrees": _consistent_float_metadata(
                reader, "0018|1120", instance_count
            ),
            "exposure_time_ms": _consistent_positive_float_metadata(
                reader, "0018|1150", instance_count
            ),
            "xray_tube_current_ma": _consistent_positive_float_metadata(
                reader, "0018|1151", instance_count
            ),
            "exposure_mas": _consistent_positive_float_metadata(
                reader, "0018|1152", instance_count
            ),
            "image_orientation_patient_complete": orientation_complete,
            "image_position_patient_complete": position_complete,
            "conversion_backend": "SimpleITK ImageSeriesReader (GDCM)",
            "conversion_backend_version": sitk.Version_VersionString(),
        },
    }
    return image, summary, candidate.info.series_instance_uid


def _nifti_suffix(path: Path) -> str:
    lowered = path.name.lower()
    if lowered.endswith(".nii.gz"):
        return ".nii.gz"
    if lowered.endswith(".nii"):
        return ".nii"
    raise ValueError("The conversion output must end in .nii or .nii.gz.")


def _conversion_metadata_path(path: Path) -> Path:
    suffix = _nifti_suffix(path)
    return path.with_name(f"{path.name[: -len(suffix)]}.bodycomposition.json")


def conversion_metadata_path(path: str | Path) -> Path:
    """Return the adjacent BodyComposition sidecar path for one NIfTI file."""

    return _conversion_metadata_path(Path(path))


def _conversion_metadata_payload(
    *,
    input_summary: Mapping[str, Any],
    output_content_sha256: str,
    output_byte_size: int,
    output_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONVERSION_METADATA_SCHEMA_VERSION,
        "sidecar_type": CONVERSION_METADATA_TYPE,
        "conversion_created_at": datetime.now(UTC).isoformat(),
        "nifti": {
            "content_sha256": output_content_sha256,
            "byte_size": output_byte_size,
            "pixel_sha256": input_summary["input_pixel_sha256"],
            "geometry": dict(output_geometry),
        },
        "source_dicom": {
            "content_sha256": input_summary["source_content_sha256"],
            "byte_size": input_summary["source_byte_size"],
            "dicom": dict(input_summary["dicom"]),
        },
    }


def enrich_nifti_summary_from_conversion_metadata(
    path: str | Path,
    nifti_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a matching allowlisted DICOM conversion sidecar when present."""

    source = Path(path)
    metadata_path = _conversion_metadata_path(source)
    summary = dict(nifti_summary)
    if not metadata_path.is_file():
        return summary
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("conversion metadata must contain a JSON object")
        validate_payload(payload, "dicom_conversion.schema.json")
    except (OSError, TypeError, ValueError) as error:
        raise DicomConversionMetadataError(
            "The BodyComposition conversion metadata sidecar could not be validated."
        ) from error

    nifti = payload["nifti"]
    mismatches = []
    if nifti["content_sha256"] != summary["source_content_sha256"]:
        mismatches.append("file hash")
    if nifti["byte_size"] != summary["source_byte_size"]:
        mismatches.append("file size")
    if nifti["pixel_sha256"] != summary["input_pixel_sha256"]:
        mismatches.append("pixel hash")
    if nifti["geometry"] != summary["geometry"]:
        mismatches.append("physical geometry")
    if mismatches:
        raise DicomConversionMetadataError(
            "The BodyComposition conversion metadata sidecar does not match the NIfTI CT "
            f"({', '.join(mismatches)})."
        )

    source_dicom = payload["source_dicom"]
    summary["dicom"] = dict(source_dicom["dicom"])
    prestage = {
        "schema_version": payload["schema_version"],
        "sidecar_type": payload["sidecar_type"],
        "source_format": "dicom",
        "source_content_sha256": source_dicom["content_sha256"],
        "source_byte_size": source_dicom["byte_size"],
        "sidecar_content_sha256": file_sha256(metadata_path),
    }
    if isinstance(payload.get("conversion_created_at"), str):
        prestage["conversion_created_at"] = payload["conversion_created_at"]
    summary["prestage"] = prestage
    return summary


def _write_conversion(
    image: sitk.Image,
    input_summary: Mapping[str, Any],
    selected_uid: str,
    destination: Path,
    *,
    overwrite: bool,
) -> DicomConversionResult:
    suffix = _nifti_suffix(destination)
    metadata_destination = _conversion_metadata_path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Conversion output already exists: {destination}. Use overwrite=True to replace it."
        )
    if metadata_destination.exists() and not overwrite:
        raise FileExistsError(
            "Conversion metadata already exists: "
            f"{metadata_destination}. Use overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    for key in image.GetMetaDataKeys():
        image.EraseMetaData(key)

    transaction_id = uuid4().hex
    temporary = destination.parent / f".{destination.name}.{transaction_id}.partial{suffix}"
    metadata_temporary = destination.parent / (
        f".{metadata_destination.name}.{transaction_id}.partial"
    )
    try:
        sitk.WriteImage(image, str(temporary), useCompression=suffix == ".nii.gz")
        verified = sitk.ReadImage(str(temporary))
        assert_same_physical_domain(
            ImageGeometry.from_sitk(image),
            ImageGeometry.from_sitk(verified),
            reference_name="DICOM CT",
            candidate_name="converted NIfTI CT",
            atol=NIFTI_ROUNDTRIP_GEOMETRY_ATOL,
        )
        if image_pixel_sha256(verified) != input_summary["input_pixel_sha256"]:
            raise DicomInputError("The converted NIfTI pixels differ from the selected DICOM CT.")
        output_digest = file_sha256(temporary)
        output_size = temporary.stat().st_size
        metadata_payload = _conversion_metadata_payload(
            input_summary=input_summary,
            output_content_sha256=output_digest,
            output_byte_size=output_size,
            output_geometry=_geometry_summary(verified),
        )
        validate_payload(metadata_payload, "dicom_conversion.schema.json")
        metadata_temporary.write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        metadata_digest = file_sha256(metadata_temporary)
        metadata_size = metadata_temporary.stat().st_size
        os.replace(temporary, destination)
        os.replace(metadata_temporary, metadata_destination)
    finally:
        temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)

    return DicomConversionResult(
        output_path=destination,
        output_content_sha256=output_digest,
        output_byte_size=output_size,
        metadata_path=metadata_destination,
        metadata_content_sha256=metadata_digest,
        metadata_byte_size=metadata_size,
        conversion_created_at=str(metadata_payload["conversion_created_at"]),
        series_instance_uid=selected_uid,
        input_summary=input_summary,
    )


def _existing_conversion(
    destination: Path,
    input_summary: Mapping[str, Any],
    selected_uid: str,
) -> DicomConversionResult:
    metadata_path = _conversion_metadata_path(destination)
    if not destination.is_file() or not metadata_path.is_file():
        raise FileExistsError(
            f"Conversion output is incomplete at {destination}; use overwrite=True to replace it."
        )
    image = sitk.ReadImage(str(destination))
    summary = {
        "source_reference": "content-addressed-local-input",
        "source_content_sha256": file_sha256(destination),
        "source_byte_size": destination.stat().st_size,
        "input_pixel_sha256": image_pixel_sha256(image),
        "input_format": "nifti",
        "geometry": _geometry_summary(image),
    }
    enriched = enrich_nifti_summary_from_conversion_metadata(destination, summary)
    if (
        enriched.get("prestage", {}).get("source_content_sha256")
        != input_summary["source_content_sha256"]
        or enriched["input_pixel_sha256"] != input_summary["input_pixel_sha256"]
    ):
        raise FileExistsError(
            f"Conversion output belongs to another CT at {destination}; "
            "use overwrite=True to replace it."
        )
    return DicomConversionResult(
        output_path=destination,
        output_content_sha256=str(summary["source_content_sha256"]),
        output_byte_size=destination.stat().st_size,
        metadata_path=metadata_path,
        metadata_content_sha256=file_sha256(metadata_path),
        metadata_byte_size=metadata_path.stat().st_size,
        conversion_created_at=str(enriched["prestage"]["conversion_created_at"]),
        series_instance_uid=selected_uid,
        input_summary=input_summary,
    )


def _write_generated_json(
    payload: Mapping[str, Any],
    destination: Path,
    *,
    overwrite: bool,
) -> Path:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.is_file():
        if destination.read_text(encoding="utf-8") == text:
            return destination
        if not overwrite:
            raise FileExistsError(
                f"Generated conversion manifest already exists at {destination}; "
                "use overwrite=True to replace it."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _series_uid_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii", errors="strict")).hexdigest()


def _conversion_failure(source: DicomStudySource, error: Exception) -> DicomConversionFailure:
    series_hashes = tuple(_series_uid_sha256(value) for value in source.series_instance_uids)
    return DicomConversionFailure(
        source_group_sha256=_study_group_digest(source.series_instance_uids),
        candidate_series_instance_uid_sha256=series_hashes,
        code=type(error).__name__,
        summary=str(
            getattr(
                error,
                "public_summary",
                "The selected DICOM CT series could not be converted.",
            )
        ),
    )


def _select_study_sources(
    source: str | Path,
    *,
    series_uid: str | None,
    study_uid: str | None,
) -> tuple[DicomStudySource, ...]:
    if series_uid is not None and study_uid is not None:
        raise DicomSeriesSelectionError(
            "Select either a Study Instance UID group or an exact Series Instance UID, not both."
        )
    sources = tuple(item for item in discover_dicom_series_sources(source) if item.modality == "CT")
    if series_uid is not None:
        requested = series_uid.strip()
        if not requested:
            raise DicomSeriesSelectionError("The requested Series Instance UID is empty.")
        selected = tuple(item for item in sources if item.series_instance_uid == requested)
        if not selected:
            available = ", ".join(item.series_instance_uid for item in sources)
            raise DicomSeriesSelectionError(
                f"Series Instance UID {requested!r} was not found. Available CT series: "
                f"{available or 'none'}."
            )
        item = selected[0]
        return (
            DicomStudySource(
                input_path=item.input_path,
                study_instance_uid=None,
                series_instance_uids=(item.series_instance_uid,),
                series_count=1,
            ),
        )
    studies = discover_dicom_study_sources(source)
    if not studies:
        raise DicomSeriesSelectionError("No CT series was found at the input path.")
    if study_uid is not None:
        requested = study_uid.strip()
        if not requested:
            raise DicomSeriesSelectionError("The requested Study Instance UID is empty.")
        studies = tuple(item for item in studies if item.study_instance_uid == requested)
        if not studies:
            raise DicomSeriesSelectionError(
                "The requested DICOM study does not contain a readable CT series."
            )
    return studies


def _convert_dicom_directory(
    source: str | Path,
    destination: Path,
    *,
    series_uid: str | None,
    study_uid: str | None,
    acquisition_number: str | None,
    overwrite: bool,
) -> DicomConversionBatchResult:
    if destination.exists() and not destination.is_dir():
        raise ValueError("A multi-series conversion output must be a directory.")
    destination.mkdir(parents=True, exist_ok=True)
    sources = _select_study_sources(
        source,
        series_uid=series_uid,
        study_uid=study_uid,
    )
    discovered_series_count = sum(item.series_count for item in sources)
    conversions: list[DicomConversionResult] = []
    failures: list[DicomConversionFailure] = []
    seen_case_ids: set[str] = set()

    for discovered in sources:
        try:
            selected_series_uid = (
                series_uid
                if series_uid is not None
                else discovered.series_instance_uids[0]
                if discovered.study_instance_uid is None
                else None
            )
            image, input_summary, selected_uid = dicom_image_summary(
                discovered.input_path,
                series_uid=selected_series_uid,
                study_uid=(discovered.study_instance_uid if selected_series_uid is None else None),
                acquisition_number=acquisition_number,
            )
            case_id = _conversion_case_id(input_summary)
            if case_id in seen_case_ids:
                raise DicomInputError(
                    "Two discovered CT series resolve to the same content-based case ID."
                )
            seen_case_ids.add(case_id)
            output = destination / f"{case_id}.nii.gz"
            conversion = (
                _write_conversion(
                    image,
                    input_summary,
                    selected_uid,
                    output,
                    overwrite=overwrite,
                )
                if overwrite or not output.exists()
                else _existing_conversion(output, input_summary, selected_uid)
            )
            conversions.append(conversion)
        except Exception as error:
            failures.append(_conversion_failure(discovered, error))

    manifest_path: Path | None = None
    if conversions:
        manifest_payload = {
            "schema_version": "1.0.0",
            "cases": [
                {
                    "case_id": value.case_id,
                    "input_path": value.output_path.relative_to(destination).as_posix(),
                }
                for value in conversions
            ],
        }
        validate_payload(manifest_payload, "batch_input.schema.json")
        manifest_path = _write_generated_json(
            manifest_payload,
            destination / CONVERSION_BATCH_MANIFEST_NAME,
            overwrite=overwrite,
        )

    report_payload = {
        "schema_version": "1.1.0",
        "report_type": "bodycomposition-dicom-conversion-batch",
        "execution_status": "succeeded" if not failures and conversions else "failed",
        "discovered_series_count": discovered_series_count,
        "discovered_study_count": len(sources),
        "converted_count": len(conversions),
        "failed_count": len(failures),
        "analysis_manifest": manifest_path.name if manifest_path is not None else None,
        "cases": [
            {
                "case_id": value.case_id,
                "status": "converted",
                "series_instance_uid_sha256": value.input_summary["dicom"][
                    "series_instance_uid_sha256"
                ],
                "nifti": value.output_path.relative_to(destination).as_posix(),
                "metadata": value.metadata_path.relative_to(destination).as_posix(),
                "failure": None,
            }
            for value in conversions
        ]
        + [
            {
                "case_id": None,
                "status": "failed",
                "source_group_sha256": value.source_group_sha256,
                "candidate_series_instance_uid_sha256": list(
                    value.candidate_series_instance_uid_sha256
                ),
                "nifti": None,
                "metadata": None,
                "failure": {"code": value.code, "summary": value.summary},
            }
            for value in failures
        ],
    }
    validate_payload(report_payload, "dicom_conversion_batch.schema.json")
    report_path = _write_generated_json(
        report_payload,
        destination / CONVERSION_BATCH_REPORT_NAME,
        overwrite=overwrite,
    )
    return DicomConversionBatchResult(
        output_path=destination,
        manifest_path=manifest_path,
        report_path=report_path,
        conversions=tuple(conversions),
        failures=tuple(failures),
        discovered_series_count=discovered_series_count,
        discovered_study_count=len(sources),
    )


def convert_dicom(
    source: str | Path,
    output_path: str | Path,
    *,
    series_uid: str | None = None,
    study_uid: str | None = None,
    acquisition_number: str | None = None,
    overwrite: bool = False,
) -> DicomConversionResult | DicomConversionBatchResult:
    """Convert one selected axial CT stack or one stack per study in a cohort.

    A NIfTI output path requests the best eligible axial stack from one study.
    A directory output selects one stack per discovered DICOM study and writes a
    ready-to-analyze batch manifest. Physical orientation is preserved,
    patient/study identifiers are omitted, and each NIfTI receives a hash-bound
    technical sidecar. Tagged accessory orientation groups may be excluded;
    incomplete geometry and exact quality ties fail closed before analysis.
    """

    destination = Path(output_path)
    if study_uid is not None and series_uid is not None:
        raise ValueError("Select either study_uid or series_uid, not both.")
    if acquisition_number is not None and series_uid is None:
        raise ValueError("An Acquisition Number override requires an exact Series Instance UID.")
    try:
        _nifti_suffix(destination)
    except ValueError:
        if destination.exists() and not destination.is_dir():
            raise
        if not destination.exists() and destination.suffix:
            raise
        return _convert_dicom_directory(
            source,
            destination,
            series_uid=series_uid,
            study_uid=study_uid,
            acquisition_number=acquisition_number,
            overwrite=overwrite,
        )

    image, input_summary, selected_uid = dicom_image_summary(
        source,
        series_uid=series_uid,
        study_uid=study_uid,
        acquisition_number=acquisition_number,
    )
    return _write_conversion(
        image,
        input_summary,
        selected_uid,
        destination,
        overwrite=overwrite,
    )


__all__ = [
    "CONVERSION_BATCH_MANIFEST_NAME",
    "CONVERSION_BATCH_REPORT_NAME",
    "DicomConversionBatchResult",
    "DicomConversionFailure",
    "DicomConversionResult",
    "DicomConversionMetadataError",
    "DicomInputError",
    "DicomSeriesInfo",
    "DicomSeriesSource",
    "DicomStudySource",
    "DicomSeriesSelectionError",
    "convert_dicom",
    "conversion_metadata_path",
    "dicom_image_summary",
    "discover_dicom_series",
    "discover_dicom_series_sources",
    "discover_dicom_study_count",
    "discover_dicom_study_sources",
    "enrich_nifti_summary_from_conversion_metadata",
]
