"""Strict, geometry-preserving DICOM CT series input and conversion."""

from __future__ import annotations

import hashlib
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
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain


class DicomInputError(ValueError):
    """Raised when a DICOM input cannot define one safe CT volume."""

    public_summary = "The DICOM input could not be read as one valid three-dimensional CT series."


class DicomSeriesSelectionError(DicomInputError):
    """Raised when DICOM series selection is absent, invalid, or ambiguous."""


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
class DicomConversionResult:
    """Verified result of converting one selected DICOM CT series."""

    output_path: Path
    output_content_sha256: str
    output_byte_size: int
    series_instance_uid: str
    input_summary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        dicom = dict(self.input_summary["dicom"])
        return {
            "output_path": self.output_path,
            "output_content_sha256": self.output_content_sha256,
            "output_byte_size": self.output_byte_size,
            "series_instance_uid": self.series_instance_uid,
            "input_format": "dicom",
            "source_content_sha256": self.input_summary["source_content_sha256"],
            "source_byte_size": self.input_summary["source_byte_size"],
            "input_pixel_sha256": self.input_summary["input_pixel_sha256"],
            "geometry": dict(self.input_summary["geometry"]),
            "dicom": dicom,
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


def _discover_candidates(source: str | Path) -> tuple[_DicomCandidate, ...]:
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

    if not by_uid:
        raise DicomInputError("No readable DICOM series was found at the input path.")
    return tuple(by_uid[key] for key in sorted(by_uid))


def discover_dicom_series(source: str | Path) -> tuple[DicomSeriesInfo, ...]:
    """List DICOM series without returning patient, study, or path metadata."""

    return tuple(candidate.info for candidate in _discover_candidates(source))


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


def convert_dicom(
    source: str | Path,
    output_path: str | Path,
    *,
    series_uid: str | None = None,
    overwrite: bool = False,
) -> DicomConversionResult:
    """Convert one selected DICOM CT series to a verified NIfTI file.

    Conversion preserves the SimpleITK/GDCM physical domain and does not
    reorient the image. DICOM patient and study metadata are not copied into
    the NIfTI output.
    """

    destination = Path(output_path)
    suffix = _nifti_suffix(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Conversion output already exists: {destination}. Use overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    image, input_summary, selected_uid = dicom_image_summary(
        source,
        series_uid=series_uid,
    )
    for key in image.GetMetaDataKeys():
        image.EraseMetaData(key)

    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.partial{suffix}"
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
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return DicomConversionResult(
        output_path=destination,
        output_content_sha256=output_digest,
        output_byte_size=output_size,
        series_instance_uid=selected_uid,
        input_summary=input_summary,
    )


__all__ = [
    "DicomConversionResult",
    "DicomInputError",
    "DicomSeriesInfo",
    "DicomSeriesSelectionError",
    "convert_dicom",
    "dicom_image_summary",
    "discover_dicom_series",
]
