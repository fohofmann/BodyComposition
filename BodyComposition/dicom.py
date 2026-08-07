"""Strict, geometry-preserving DICOM CT series input and conversion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import SimpleITK as sitk

from BodyComposition.provenance import file_sha256, image_pixel_sha256
from BodyComposition.schema_validation import validate_payload
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain

CONVERSION_METADATA_SCHEMA_VERSION = "1.1.0"
CONVERSION_METADATA_TYPE = "bodycomposition-dicom-conversion"
CONVERSION_BATCH_MANIFEST_NAME = "bodycomposition-batch.json"
CONVERSION_BATCH_REPORT_NAME = "bodycomposition-conversion.json"


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
class DicomConversionResult:
    """Verified result of converting one selected DICOM CT series."""

    output_path: Path
    output_content_sha256: str
    output_byte_size: int
    metadata_path: Path
    metadata_content_sha256: str
    metadata_byte_size: int
    series_instance_uid: str
    input_summary: Mapping[str, Any]

    @property
    def case_id(self) -> str:
        return f"case-{self.input_summary['input_pixel_sha256'][:16]}"

    def as_dict(self) -> dict[str, Any]:
        dicom = dict(self.input_summary["dicom"])
        return {
            "case_id": self.case_id,
            "output_path": self.output_path,
            "output_content_sha256": self.output_content_sha256,
            "output_byte_size": self.output_byte_size,
            "metadata_path": self.metadata_path,
            "metadata_content_sha256": self.metadata_content_sha256,
            "metadata_byte_size": self.metadata_byte_size,
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
    """Privacy-safe failure for one discovered DICOM series."""

    series_instance_uid_sha256: str
    code: str
    summary: str

    def as_dict(self) -> dict[str, str]:
        return {
            "series_instance_uid_sha256": self.series_instance_uid_sha256,
            "code": self.code,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class DicomConversionBatchResult:
    """Result of converting every discovered CT series beneath one directory."""

    output_path: Path
    manifest_path: Path | None
    report_path: Path
    conversions: tuple[DicomConversionResult, ...]
    failures: tuple[DicomConversionFailure, ...]
    discovered_series_count: int

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
            "converted_count": len(self.conversions),
            "failed_count": len(self.failures),
            "conversions": [value.as_dict() for value in self.conversions],
            "failures": [value.as_dict() for value in self.failures],
        }


@dataclass(frozen=True)
class _DicomCandidate:
    info: DicomSeriesInfo
    files: tuple[Path, ...]


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


def _equipment_metadata(reader: Any, key: str, instance_count: int) -> str | None:
    """Return one consistent, printable equipment value without free-text tags."""

    values = [_metadata(reader, key, index=index) for index in range(instance_count)]
    if any(value is None for value in values) or len(set(values)) != 1:
        return None
    text = " ".join(str(values[0]).split())
    text = re.sub(r"[^A-Za-z0-9 ._+()-]", " ", text)
    text = " ".join(text.split())
    return text[:80] or None


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


def _select_candidate(
    source: str | Path,
    *,
    series_uid: str | None,
) -> _DicomCandidate:
    candidates = _discover_candidates(source)
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
        selected = matches[0]
        if selected.info.modality != "CT":
            raise DicomSeriesSelectionError(
                f"Series Instance UID {requested!r} has modality "
                f"{selected.info.modality!r}, not CT."
            )
        return selected

    ct_candidates = [candidate for candidate in candidates if candidate.info.modality == "CT"]
    if len(ct_candidates) == 1:
        return ct_candidates[0]
    if not ct_candidates:
        modalities = ", ".join(
            f"{candidate.info.series_instance_uid} ({candidate.info.modality})"
            for candidate in candidates
        )
        raise DicomSeriesSelectionError(f"No CT series was found. Available series: {modalities}.")
    available = ", ".join(candidate.info.series_instance_uid for candidate in ct_candidates)
    error = DicomSeriesSelectionError(
        f"Multiple CT series were found. Select one with its Series Instance UID: {available}."
    )
    error.public_summary = (
        "Multiple CT DICOM series were found; select one explicitly with its Series Instance UID."
    )
    raise error


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
) -> dict[str, str]:
    """Read the minimal local DICOM demographics requested for a patient PDF.

    These values are intentionally separate from the privacy-safe input
    provenance and must not be copied into canonical case manifests or logs.
    """

    candidate = _select_candidate(source, series_uid=series_uid)
    patient_name = _dicom_person_name(
        _consistent_dicom_header_value(candidate.files, "0010|0010")
    )
    date_of_birth = _dicom_date(
        _consistent_dicom_header_value(candidate.files, "0010|0030")
    )
    sex_value = _ascii_header_text(
        _consistent_dicom_header_value(candidate.files, "0010|0040"),
        maximum=8,
    )
    sex = sex_value.upper() if sex_value and sex_value.upper() in {"F", "M", "O", "U"} else None
    scan_date = next(
        (
            parsed
            for key in ("0008|002a", "0008|0022", "0008|0021", "0008|0020", "0008|0023")
            if (parsed := _consistent_dicom_date(candidate.files, key)) is not None
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


def dicom_image_summary(
    source: str | Path,
    *,
    series_uid: str | None = None,
) -> tuple[sitk.Image, dict[str, Any], str]:
    """Read one DICOM CT series and return pixels plus privacy-safe provenance."""

    candidate = _select_candidate(source, series_uid=series_uid)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in candidate.files])
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    try:
        image = reader.Execute()
    except RuntimeError as error:
        raise DicomInputError("GDCM could not assemble the selected DICOM CT series.") from error
    if image.GetDimension() != 3:
        raise DicomInputError("The selected DICOM CT series is not three-dimensional.")

    pixels = sitk.GetArrayViewFromImage(image)
    if not np.all(np.isfinite(pixels)):
        raise DicomInputError("The selected DICOM CT contains non-finite pixels.")
    geometry_summary = _geometry_summary(image)

    modalities = {
        (_metadata(reader, "0008|0060", index=index) or "UNKNOWN").upper()
        for index in range(len(candidate.files))
    }
    if modalities != {"CT"}:
        raise DicomInputError(
            "The selected DICOM series does not contain consistently declared CT instances."
        )
    orientation_complete = all(
        _metadata(reader, "0020|0037", index=index) is not None
        for index in range(len(candidate.files))
    )
    position_complete = all(
        _metadata(reader, "0020|0032", index=index) is not None
        for index in range(len(candidate.files))
    )
    source_digest, source_byte_size = _series_content_identity(candidate.files)
    uid_digest = hashlib.sha256(
        candidate.info.series_instance_uid.encode("ascii", errors="strict")
    ).hexdigest()
    summary = {
        "source_reference": "content-addressed-local-input",
        "source_content_sha256": source_digest,
        "source_byte_size": source_byte_size,
        "input_pixel_sha256": image_pixel_sha256(image),
        "input_format": "dicom",
        "geometry": geometry_summary,
        "dicom": {
            "series_instance_uid_sha256": uid_digest,
            "instance_count": candidate.info.instance_count,
            "modality": "CT",
            "scanner_manufacturer": _equipment_metadata(
                reader, "0008|0070", candidate.info.instance_count
            ),
            "scanner_model": _equipment_metadata(
                reader, "0008|1090", candidate.info.instance_count
            ),
            "slice_thickness_mm": _consistent_positive_float_metadata(
                reader, "0018|0050", candidate.info.instance_count
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
    """Load a matching privacy-safe DICOM conversion sidecar when present."""

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
    summary["prestage"] = {
        "schema_version": payload["schema_version"],
        "sidecar_type": payload["sidecar_type"],
        "source_format": "dicom",
        "source_content_sha256": source_dicom["content_sha256"],
        "source_byte_size": source_dicom["byte_size"],
        "sidecar_content_sha256": file_sha256(metadata_path),
    }
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


def _conversion_failure(source: DicomSeriesSource, error: Exception) -> DicomConversionFailure:
    return DicomConversionFailure(
        series_instance_uid_sha256=_series_uid_sha256(source.series_instance_uid),
        code=type(error).__name__,
        summary=str(
            getattr(
                error,
                "public_summary",
                "The selected DICOM CT series could not be converted.",
            )
        ),
    )


def _select_sources(
    source: str | Path,
    *,
    series_uid: str | None,
) -> tuple[DicomSeriesSource, ...]:
    sources = tuple(
        item for item in discover_dicom_series_sources(source) if item.modality == "CT"
    )
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
        return selected
    if not sources:
        raise DicomSeriesSelectionError("No CT series was found at the input path.")
    return sources


def _convert_dicom_directory(
    source: str | Path,
    destination: Path,
    *,
    series_uid: str | None,
    overwrite: bool,
) -> DicomConversionBatchResult:
    if destination.exists() and not destination.is_dir():
        raise ValueError("A multi-series conversion output must be a directory.")
    destination.mkdir(parents=True, exist_ok=True)
    sources = _select_sources(source, series_uid=series_uid)
    conversions: list[DicomConversionResult] = []
    failures: list[DicomConversionFailure] = []
    seen_case_ids: set[str] = set()

    for discovered in sources:
        try:
            image, input_summary, selected_uid = dicom_image_summary(
                discovered.input_path,
                series_uid=discovered.series_instance_uid,
            )
            case_id = f"case-{input_summary['input_pixel_sha256'][:16]}"
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
        "schema_version": "1.0.0",
        "report_type": "bodycomposition-dicom-conversion-batch",
        "execution_status": "succeeded" if not failures and conversions else "failed",
        "discovered_series_count": len(sources),
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
                "series_instance_uid_sha256": value.series_instance_uid_sha256,
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
        discovered_series_count=len(sources),
    )


def convert_dicom(
    source: str | Path,
    output_path: str | Path,
    *,
    series_uid: str | None = None,
    overwrite: bool = False,
) -> DicomConversionResult | DicomConversionBatchResult:
    """Convert one selected series or every CT series beneath a directory.

    A NIfTI output path requests one series. A directory output converts every
    discovered CT series and writes a ready-to-analyze batch manifest. Physical
    orientation is preserved, patient/study metadata are omitted, and each
    NIfTI receives a hash-bound technical sidecar.
    """

    destination = Path(output_path)
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
            overwrite=overwrite,
        )

    image, input_summary, selected_uid = dicom_image_summary(
        source,
        series_uid=series_uid,
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
    "DicomSeriesSelectionError",
    "convert_dicom",
    "conversion_metadata_path",
    "dicom_image_summary",
    "discover_dicom_series",
    "discover_dicom_series_sources",
    "enrich_nifti_summary_from_conversion_metadata",
]
