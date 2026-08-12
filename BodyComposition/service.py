"""One orchestration service shared by the Python API and command line."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import monotonic, sleep
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from BodyComposition.config import (
    ConfigError,
    PipelineConfig,
    low_resource_config,
    resolve_tissue_backend,
)
from BodyComposition.dicom import (
    CONVERSION_BATCH_MANIFEST_NAME,
    DicomConversionResult,
    conversion_metadata_path,
    convert_dicom,
    dicom_report_patient_metadata,
    discover_dicom_series_sources,
    discover_dicom_study_sources,
)
from BodyComposition.execution import (
    CaseLease,
    ClaimLostError,
    ExecutionPolicy,
    RestartAfterOutOfMemory,
    RestartAfterWorkerTimeout,
    SharedExecutionState,
    execution_hardware_profile,
    hardware_profile_id,
    is_out_of_memory_error,
    release_accelerator_cache,
    shared_file_guard,
)
from BodyComposition.measurement.api import load_measurement_tables
from BodyComposition.measurement.csv_export import (
    write_csv_mirrors,
    write_csv_table,
)
from BodyComposition.model_manager import ModelStatus, require_models
from BodyComposition.pipeline import InternalPipeline
from BodyComposition.provenance import (
    analysis_identity,
    canonical_digest,
    file_sha256,
    image_summary,
    pixel_array_sha256,
    source_state,
)
from BodyComposition.results import (
    BatchResult,
    BatchWorkerResult,
    CaseResult,
    ExecutionStatus,
    QCStatus,
    validate_public_id,
)
from BodyComposition.schema_validation import resolve_bundle_path, validate_payload
from BodyComposition.utils.nifti import NiftiDataContainer

DEFAULT_OUTPUT_ROOT = Path(os.environ.get("BODYCOMPOSITION_OUTPUT_ROOT", "output")).expanduser()
IMPLICIT_UPDATE_RUN_ID = "current"


class DirtySourceError(RuntimeError):
    """Raised when a cohort/release run uses uncommitted source code."""


class ExistingRunError(RuntimeError):
    """Raised when an existing run cannot safely accept the requested plan."""


class InputChangedError(RuntimeError):
    """Raised when an input changes between preflight and execution."""


class InputDiscoveryError(ValueError):
    """Raised when a directory cannot be mapped safely to CT cases."""

    public_summary = "The input directory could not be mapped safely to CT cases."


def _validate_dicom_uid(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    if (
        not result
        or len(result) > 64
        or any(not component.isdigit() for component in result.split("."))
    ):
        raise ValueError(f"{name} must be a valid DICOM UID.")
    return result


def _validate_series_uid(value: str | None) -> str | None:
    return _validate_dicom_uid(value, name="series_uid")


def _validate_study_uid(value: str | None) -> str | None:
    return _validate_dicom_uid(value, name="study_uid")


@dataclass(frozen=True)
class CaseInput:
    input_path: Path
    case_id: str | None = None
    series_uid: str | None = None
    study_uid: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        if self.case_id is not None:
            object.__setattr__(
                self,
                "case_id",
                validate_public_id(str(self.case_id), name="case_id"),
            )
        object.__setattr__(self, "series_uid", _validate_series_uid(self.series_uid))
        object.__setattr__(self, "study_uid", _validate_study_uid(self.study_uid))
        if self.series_uid is not None and self.study_uid is not None:
            raise ValueError("A case input cannot select both a series_uid and study_uid.")

    @classmethod
    def model_validate(cls, value: CaseInput | str | Path | Mapping[str, Any]) -> CaseInput:
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            return cls(Path(value))
        if isinstance(value, Mapping):
            unknown = set(value) - {"input_path", "case_id", "series_uid", "study_uid"}
            if unknown:
                raise ValueError(f"Unknown case input values: {sorted(unknown)}.")
            if "input_path" not in value:
                raise ValueError("A case input requires input_path.")
            case_id = value.get("case_id")
            if case_id is not None:
                validate_public_id(str(case_id), name="case_id")
            return cls(
                Path(value["input_path"]),
                str(case_id) if case_id is not None else None,
                _validate_series_uid(
                    str(value["series_uid"]) if value.get("series_uid") is not None else None
                ),
                _validate_study_uid(
                    str(value["study_uid"]) if value.get("study_uid") is not None else None
                ),
            )
        raise TypeError("Case input must be a path, mapping, or CaseInput.")


@dataclass(frozen=True)
class _PreparedCase:
    case: CaseInput
    input_summary: Mapping[str, Any]
    case_id: str
    analysis_id: str
    provenance: Mapping[str, Any]
    preflight_failure: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _InputPreflight:
    input_summary: Mapping[str, Any]
    failure: Mapping[str, Any] | None = None


def _input_snapshot_digest(path: Path) -> str:
    """Build a cheap cache key without reading image payloads."""

    source = path.resolve(strict=False)
    lowered = source.name.lower()
    is_nifti = source.is_file() and (lowered.endswith(".nii") or lowered.endswith(".nii.gz"))
    entries: list[dict[str, Any]] = []

    def append(candidate: Path, relative: str) -> None:
        try:
            stat = candidate.stat()
        except OSError as error:
            entries.append(
                {
                    "relative": relative,
                    "available": False,
                    "error": type(error).__name__,
                }
            )
            return
        entries.append(
            {
                "relative": relative,
                "available": True,
                "byte_size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
        )

    if is_nifti:
        append(source, source.name)
        suffix_length = len(".nii.gz") if lowered.endswith(".nii.gz") else len(".nii")
        sidecar = source.with_name(f"{source.name[:-suffix_length]}.bodycomposition.json")
        append(sidecar, sidecar.name)
    else:
        directory = source.parent if source.is_file() else source
        append(directory, ".")
        if directory.is_dir():
            for root, directories, files in os.walk(directory):
                directories.sort()
                files.sort()
                base = Path(root)
                for name in files:
                    candidate = base / name
                    append(candidate, candidate.relative_to(directory).as_posix())

    return canonical_digest(
        {
            "source": str(source),
            "entries": entries,
        }
    )


def _preflight_request_digest(cases: Sequence[CaseInput]) -> str:
    return canonical_digest(
        {
            "schema_version": 1,
            "cases": [
                {
                    "ordinal": ordinal,
                    "case_id": case.case_id,
                    "series_uid": case.series_uid,
                    "study_uid": case.study_uid,
                    "input_snapshot_sha256": _input_snapshot_digest(case.input_path),
                }
                for ordinal, case in enumerate(cases)
            ],
        }
    )


def _build_input_preflight(cases: Sequence[CaseInput]) -> tuple[_InputPreflight, ...]:
    values: list[_InputPreflight] = []
    for ordinal, case in enumerate(cases):
        failure = None
        try:
            _, input_summary = image_summary(
                case.input_path,
                series_uid=case.series_uid,
                study_uid=case.study_uid,
            )
        except Exception as error:
            content_digest = file_sha256(case.input_path) if case.input_path.is_file() else None
            pseudo_pixel_digest = canonical_digest(
                {
                    "invalid_input": content_digest,
                    "ordinal": ordinal,
                    "error_code": type(error).__name__,
                }
            )
            input_summary = {
                "source_reference": "content-addressed-local-input",
                "source_content_sha256": content_digest,
                "source_byte_size": (
                    case.input_path.stat().st_size if case.input_path.is_file() else None
                ),
                "input_pixel_sha256": pseudo_pixel_digest,
                "input_format": "unreadable_or_missing",
                "geometry": {},
            }
            public_summary = getattr(
                error,
                "public_summary",
                "The input could not be read as a valid three-dimensional NIfTI or DICOM CT.",
            )
            failure = {
                "stage": "input_validation",
                "code": type(error).__name__,
                "summary": str(public_summary),
            }
        values.append(_InputPreflight(input_summary=input_summary, failure=failure))
    return tuple(values)


def _load_input_preflight(
    path: Path,
    *,
    request_digest: str,
    case_count: int,
) -> tuple[_InputPreflight, ...] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or payload.get("request_sha256") != request_digest
            or payload.get("case_count") != case_count
            or not isinstance(payload.get("cases"), list)
            or len(payload["cases"]) != case_count
        ):
            return None
        values = []
        for ordinal, item in enumerate(payload["cases"]):
            if (
                not isinstance(item, Mapping)
                or item.get("ordinal") != ordinal
                or not isinstance(item.get("input_summary"), Mapping)
                or (
                    item.get("failure") is not None and not isinstance(item.get("failure"), Mapping)
                )
            ):
                return None
            values.append(
                _InputPreflight(
                    input_summary=dict(item["input_summary"]),
                    failure=(
                        dict(item["failure"]) if isinstance(item.get("failure"), Mapping) else None
                    ),
                )
            )
        return tuple(values)
    except (OSError, TypeError, ValueError):
        return None


def _cached_input_preflight(
    cases: Sequence[CaseInput],
    output: Path,
) -> tuple[_InputPreflight, ...]:
    """Read each batch input once across compatible concurrent workers."""

    request_digest = _preflight_request_digest(cases)
    path = output / ".bodycomposition" / "preflight" / f"{request_digest}.json"
    lock_path = path.with_suffix(".lock")
    with shared_file_guard(lock_path):
        cached = _load_input_preflight(
            path,
            request_digest=request_digest,
            case_count=len(cases),
        )
        if cached is not None:
            return cached
        prepared = _build_input_preflight(cases)
        _atomic_json(
            {
                "schema_version": 1,
                "request_sha256": request_digest,
                "case_count": len(cases),
                "cases": [
                    {
                        "ordinal": ordinal,
                        "input_summary": dict(item.input_summary),
                        "failure": dict(item.failure) if item.failure else None,
                    }
                    for ordinal, item in enumerate(prepared)
                ],
            },
            path,
        )
        return prepared


def _run_input_directory_names(prepared: Sequence[_PreparedCase]) -> tuple[str, ...]:
    """Return bounded directory labels without exposing absolute input paths."""

    directories = []
    for item in prepared:
        path = item.case.input_path
        directory = path if path.is_dir() else path.parent
        directories.append(directory.resolve(strict=False))
    unique_directories = list(dict.fromkeys(directories))
    if not unique_directories:
        return ("Not recorded",)
    if len(unique_directories) == 1:
        return (unique_directories[0].name or "root",)
    common = Path(os.path.commonpath([str(path) for path in unique_directories]))
    if all(path.parent == common for path in unique_directories):
        return (common.name or "root",)
    return tuple(path.name or "root" for path in unique_directories)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_run_period(
    started_at: str,
    ended_values: Sequence[Any],
) -> tuple[str | None, float | None]:
    """Return the latest terminal timestamp and elapsed run wall time."""

    try:
        started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
    except ValueError:
        return None, None
    if started.utcoffset() is None:
        return None, None
    started = started.astimezone(UTC)
    ended_candidates: list[datetime] = []
    for value in ended_values:
        if value is None:
            continue
        try:
            ended = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        if ended.utcoffset() is None:
            continue
        ended_candidates.append(ended.astimezone(UTC))
    if not ended_candidates:
        return None, None
    ended = max(ended_candidates)
    duration = (ended - started).total_seconds()
    if duration < 0:
        return None, None
    return ended.isoformat(), duration


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:16]}"


def _json_mapping_key(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise TypeError("QC evidence mapping keys must be strings or integers.")


def _json_native(value: Any) -> Any:
    """Convert nested QC evidence to deterministic, safe JSON values."""

    if isinstance(value, Enum):
        return _json_native(value.value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("QC evidence contains a non-finite floating-point value.")
        return value
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            normalized_key = _json_mapping_key(key)
            if normalized_key in normalized:
                raise TypeError("QC evidence mapping keys collide after JSON normalization.")
            normalized[normalized_key] = _json_native(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_json_native(item) for item in value]
    raise TypeError(f"Unsupported QC evidence type: {type(value).__name__}.")


def _default_run_id(
    ordered_cases: Sequence[Mapping[str, str]],
    queue_configuration_sha256: str,
) -> str:
    """Derive the queue identity shared by compatible independent workers."""

    digest = canonical_digest(
        {
            "ordered_cases": [dict(value) for value in ordered_cases],
            "queue_configuration_sha256": queue_configuration_sha256,
        }
    )
    return f"run-{digest[:24]}"


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_text(value: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _run_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": value.get("run_id"),
        "ordered_cases": value.get("ordered_cases"),
        "configuration_sha256": value.get("configuration_sha256"),
    }


def _active_plan_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "plan_sha256": canonical_digest(plan),
        "plan": dict(plan),
    }


def _prepare_run_generation(
    output_root: Path,
    run_id: str,
    plan: Mapping[str, Any],
    *,
    update: bool,
) -> Path:
    """Create or join one run generation, optionally advancing a stable run ID."""

    run_root = output_root / "runs" / run_id
    active_plan_path = run_root / ".active-plan.json"
    requested_payload = _active_plan_payload(plan)
    requested_digest = str(requested_payload["plan_sha256"])
    guard = output_root / ".bodycomposition" / "guards" / f"run-generation-{run_id}.lock"

    with shared_file_guard(guard):
        manifest_path = run_root / "run_manifest.json"
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(previous, Mapping):
                raise ExistingRunError("The existing run manifest is unreadable.")
            validate_payload(previous, "run_manifest.schema.json")
            previous_plan = _run_plan(previous)
            if previous_plan == dict(plan):
                if not active_plan_path.is_file():
                    _atomic_json(requested_payload, active_plan_path)
                return run_root
            if not update:
                raise ExistingRunError(
                    "The requested run_id already belongs to a different immutable "
                    "input/configuration plan. Use --update only when this "
                    "completed run should become the prior generation of the same cohort."
                )
            if previous_plan.get("configuration_sha256") != plan.get("configuration_sha256"):
                raise ExistingRunError(
                    "--update may refresh cohort inputs, but it cannot change the "
                    "queue-compatible configuration. Omit --update for a new immutable "
                    "run, use another output root, or choose an advanced new run ID."
                )

            previous_digest = canonical_digest(previous_plan)
            generation = (
                f"{previous_digest[:24]}-"
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid4().hex[:8]}"
            )
            archive_root = output_root / "superseded" / run_id / generation
            archive_root.parent.mkdir(parents=True, exist_ok=True)
            run_root.rename(archive_root)

            state_root = output_root / ".bodycomposition" / "execution" / run_id
            state_archive = (
                output_root / ".bodycomposition" / "execution-history" / run_id / generation
            )
            try:
                if state_root.exists():
                    state_archive.parent.mkdir(parents=True, exist_ok=True)
                    state_root.rename(state_archive)
            except OSError:
                archive_root.rename(run_root)
                raise

            _atomic_json(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "plan_sha256": previous_digest,
                    "superseded_at": _utc_now(),
                    "superseded_by_plan_sha256": requested_digest,
                },
                archive_root / ".superseded.json",
            )
            run_root.mkdir(parents=True, exist_ok=False)
            _atomic_json(requested_payload, active_plan_path)
            return run_root

        if run_root.exists():
            if active_plan_path.is_file():
                active = json.loads(active_plan_path.read_text(encoding="utf-8"))
                if (
                    isinstance(active, Mapping)
                    and active.get("plan_sha256") == requested_digest
                    and active.get("plan") == dict(plan)
                ):
                    return run_root
                raise ExistingRunError(
                    "The requested run_id has an active incomplete run for another plan. "
                    "Finish or recover that run before replacing it."
                )
            if any(run_root.iterdir()):
                raise ExistingRunError(
                    "The requested run_id has incomplete legacy state without a verifiable "
                    "plan marker; it cannot be replaced automatically."
                )
        else:
            run_root.mkdir(parents=True)
        _atomic_json(requested_payload, active_plan_path)
        return run_root


def _check_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".write-check-{uuid4().hex}"
    try:
        probe.write_bytes(b"")
    except OSError as error:
        raise PermissionError(f"Output directory is not writable: {directory}.") from error
    finally:
        probe.unlink(missing_ok=True)


def _safe_case_id(
    requested: str | None,
    input_summary: Mapping[str, Any],
) -> str:
    if requested is not None:
        return validate_public_id(requested, name="case_id")
    dicom = input_summary.get("dicom")
    if isinstance(dicom, Mapping):
        study_digest = dicom.get("study_instance_uid_sha256")
        if isinstance(study_digest, str) and len(study_digest) == 64:
            return f"case-{study_digest[:16]}"
    return f"case-{str(input_summary['input_pixel_sha256'])[:16]}"


def _normalise_flag(value: Any, *, default_stage: str) -> dict[str, Any]:
    if is_dataclass(value):
        payload = asdict(cast(Any, value))
    elif hasattr(value, "as_dict"):
        payload = dict(value.as_dict())
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"Unsupported QC flag type: {type(value)!r}.")
    thresholds = payload.get("thresholds", payload.get("threshold", {})) or {}
    action = payload.get(
        "suggested_review_action",
        payload.get("suggested_action", "Review the canonical QC artifacts."),
    )
    severity = payload.get("severity", "warning")
    severity = getattr(severity, "value", severity)
    normalized = {
        "code": str(payload.get("code", "unspecified_review")),
        "stage": str(payload.get("stage", default_stage)),
        "severity": str(severity),
        "reason": str(payload.get("reason", "Manual review is required.")),
        "observed": dict(payload.get("observed", {})),
        "thresholds": dict(thresholds),
        "suggested_review_action": str(action),
        "reviewer": payload.get("reviewer"),
        "reviewed_at": payload.get("reviewed_at"),
        "adjudication": payload.get("adjudication"),
        "review_comment": payload.get("review_comment"),
    }
    return cast(dict[str, Any], _json_native(normalized))


def _case_qc(memory: Mapping[str, Any]) -> tuple[QCStatus, tuple[dict[str, Any], ...]]:
    statuses: list[str] = []
    flags: list[dict[str, Any]] = []
    orientation = memory.get("tmp/orientation_result")
    if orientation is not None:
        statuses.append(str(orientation.qc_status))
        flags.extend(
            _normalise_flag(value, default_stage="orientation")
            for value in orientation.review_flags
        )
    vertebral = memory.get("tmp/vertebral_result")
    if vertebral is not None:
        statuses.append(str(vertebral.qc_status.value))
        flags.extend(
            _normalise_flag(value, default_stage="vertebral") for value in vertebral.qc_flags
        )
    measurement = memory.get("tmp/measurement_bundle")
    if measurement is not None:
        statuses.append(str(measurement.qc_status.value))
        flags.extend(
            _normalise_flag(value, default_stage="measurement") for value in measurement.qc_flags
        )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for flag in flags:
        key = json.dumps(flag, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            unique.append(flag)
            seen.add(key)
    severities = {str(flag.get("severity", "warning")).lower() for flag in unique}
    if "fail" in statuses or "error" in severities:
        status = QCStatus.FAIL
    elif "review" in statuses or any(severity != "info" for severity in severities):
        status = QCStatus.REVIEW
    elif statuses:
        status = QCStatus.PASS
    else:
        status = QCStatus.NOT_ASSESSED
    return status, tuple(unique)


def _scientific_status(memory: Mapping[str, Any], config: PipelineConfig) -> dict[str, Any]:
    orientation = memory.get("tmp/orientation_result")
    vertebral = memory.get("tmp/vertebral_result")
    measurement = memory.get("tmp/measurement_bundle")
    return {
        "orientation": orientation.to_dict() if orientation is not None else None,
        "vertebral_backend": (
            vertebral.backend_id if vertebral is not None else config.vertebral_backend
        ),
        "tissue_backend": config.tissue_backend,
        "body_surface_backend": config.normalized()["body_surface"]["backend"],
        "measurement_schema_version": (
            str(measurement.slices.iloc[0]["schema_version"]) if measurement is not None else None
        ),
    }


def _artifact_inventory(bundle: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        relative = path.relative_to(bundle).as_posix()
        if relative == "case_manifest.json" or ".partial" in path.name:
            continue
        records.append(
            {
                "relative_path": relative,
                "byte_size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


def _write_stage_log(memory: Mapping[str, Any], bundle: Path) -> None:
    events = list(memory.get("tmp/stage_events", ()))
    if not events:
        return
    value = "".join(json.dumps(event, sort_keys=True, allow_nan=False) + "\n" for event in events)
    _atomic_text(value, bundle / "logs" / "stages.jsonl")


def _case_manifest(
    *,
    case_id: str,
    run_id: str,
    analysis_id: str,
    attempt_id: str,
    execution_status: ExecutionStatus,
    qc_status: QCStatus,
    flags: Sequence[Mapping[str, Any]],
    input_summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    scientific: Mapping[str, Any],
    started_at: str,
    duration_seconds: float,
    artifacts: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0.0",
        "manifest_type": "case",
        "case_id": case_id,
        "run_id": run_id,
        "analysis_id": analysis_id,
        "attempt_id": attempt_id,
        "execution_status": execution_status.value,
        "qc_status": qc_status.value,
        "manual_review_required": qc_status in {QCStatus.REVIEW, QCStatus.FAIL},
        "started_at": started_at,
        "ended_at": _utc_now(),
        "duration_seconds": duration_seconds,
        "input": dict(input_summary),
        "scientific": dict(scientific),
        "qc_flags": [dict(value) for value in flags],
        "provenance": dict(provenance),
        "artifacts": [dict(value) for value in artifacts],
        "failure": dict(failure) if failure is not None else None,
    }
    validate_payload(payload, "case_manifest.schema.json")
    return payload


def _report_technical_metadata(
    *,
    started_at: str,
    input_summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    hardware_profile: Mapping[str, Any],
) -> dict[str, str | None]:
    """Select the small non-identifying metadata surface permitted in case PDFs."""

    def clean(value: Any) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).split())
        text = "".join(character if 32 <= ord(character) < 127 else "?" for character in text)
        return text[:160] or None

    code = provenance.get("code")
    code = code if isinstance(code, Mapping) else {}
    runtime = provenance.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    dicom = input_summary.get("dicom")
    dicom = dicom if isinstance(dicom, Mapping) else {}
    backend = hardware_profile.get("backend") or runtime.get("device_type")
    hardware = hardware_profile.get("accelerator_name") or runtime.get("gpu_name")
    if hardware is None:
        processor = clean(hardware_profile.get("processor"))
        machine = clean(hardware_profile.get("machine"))
        components = [
            value
            for value in (processor, machine)
            if value is not None and value.lower() != "unknown"
        ]
        hardware = " / ".join(dict.fromkeys(components)) or runtime.get("host_class")
    return {
        "analysis_started_at": clean(started_at),
        "input_format": clean(input_summary.get("input_format")) or "unknown",
        "pipeline_version": clean(code.get("package_version")),
        "runtime_backend": clean(backend),
        "runtime_hardware": clean(hardware),
        "scanner_manufacturer": clean(dicom.get("scanner_manufacturer")),
        "scanner_model": clean(dicom.get("scanner_model")),
        "dicom_slice_thickness_mm": clean(dicom.get("slice_thickness_mm")),
    }


def load_batch_manifest(path: str | Path) -> tuple[CaseInput, ...]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Batch manifest must contain an object.")
    validate_payload(payload, "batch_input.schema.json")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Batch manifest requires a non-empty ordered cases list.")
    result = []
    for value in cases:
        case = CaseInput.model_validate(value)
        input_path = case.input_path
        if not input_path.is_absolute():
            input_path = source.parent / input_path
        result.append(
            CaseInput(
                input_path=input_path,
                case_id=case.case_id,
                series_uid=case.series_uid,
                study_uid=case.study_uid,
            )
        )
    return tuple(result)


def _is_nifti_path(path: Path) -> bool:
    lowered = path.name.lower()
    return path.is_file() and (lowered.endswith(".nii") or lowered.endswith(".nii.gz"))


def _directory_nifti_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if _is_nifti_path(path)
                and not any(part.startswith(".") for part in path.relative_to(root).parts)
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def _auto_manifest_inputs(root: Path, manifest: Path) -> tuple[CaseInput, ...]:
    cases = load_batch_manifest(manifest)
    resolved_root = root.resolve()
    for case in cases:
        try:
            case.input_path.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise InputDiscoveryError(
                f"Automatically discovered {CONVERSION_BATCH_MANIFEST_NAME} references "
                "an input outside its directory."
            ) from error
    return cases


def discover_case_inputs(
    input_path: str | Path,
    *,
    series_uid: str | None = None,
) -> tuple[CaseInput, ...]:
    """Map one file or directory to deterministic CT case inputs.

    Files remain single-case inputs. Directories prefer a conversion-generated
    batch manifest, otherwise accept verified conversion sidecars or one selected
    DICOM CT stack per recursively discovered study. Multiple arbitrary NIfTI files are rejected so
    segmentation masks cannot be mistaken for source CTs.
    """

    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input CT not found: {source}.")
    if source.is_file():
        if series_uid is not None and _is_nifti_path(source):
            raise InputDiscoveryError(
                "A DICOM Series Instance UID cannot be used with a NIfTI input."
            )
        return (CaseInput(source, series_uid=series_uid),)
    if not source.is_dir():
        raise InputDiscoveryError("The analysis input must be a file or directory.")

    generated_manifest = source / CONVERSION_BATCH_MANIFEST_NAME
    if generated_manifest.is_file():
        if series_uid is not None:
            raise InputDiscoveryError(
                "--series cannot be combined with a converted cohort manifest."
            )
        return _auto_manifest_inputs(source, generated_manifest)

    nifti_paths = _directory_nifti_paths(source)
    dicom_series_sources = tuple(
        item for item in discover_dicom_series_sources(source) if item.modality == "CT"
    )
    if nifti_paths and dicom_series_sources:
        raise InputDiscoveryError(
            "The input directory contains both NIfTI files and DICOM CT series. "
            "Point analyze at one collection or use an explicit batch manifest."
        )
    if nifti_paths:
        if series_uid is not None:
            raise InputDiscoveryError(
                "A DICOM Series Instance UID cannot be used with NIfTI inputs."
            )
        staged = tuple(path for path in nifti_paths if conversion_metadata_path(path).is_file())
        if len(nifti_paths) == 1:
            return (CaseInput(nifti_paths[0]),)
        if len(staged) != len(nifti_paths):
            raise InputDiscoveryError(
                "Multiple NIfTI files were found, but not every file has a matching "
                "BodyComposition conversion sidecar. Point analyze at one CT or use "
                "an explicit batch manifest so masks cannot be mistaken for inputs."
            )
        return tuple(CaseInput(path) for path in staged)

    selected_uid = _validate_series_uid(series_uid)
    if selected_uid is not None:
        selected = tuple(
            item for item in dicom_series_sources if item.series_instance_uid == selected_uid
        )
        if not selected:
            available = ", ".join(item.series_instance_uid for item in dicom_series_sources)
            raise InputDiscoveryError(
                f"Series Instance UID {selected_uid!r} was not found. Available CT "
                f"series: {available or 'none'}."
            )
        return (
            CaseInput(
                selected[0].input_path,
                series_uid=selected[0].series_instance_uid,
            ),
        )
    study_sources = discover_dicom_study_sources(source)
    if not study_sources:
        raise InputDiscoveryError(
            "No NIfTI CT, converted NIfTI/sidecar pair, or DICOM CT series was found."
        )
    return tuple(
        CaseInput(
            item.input_path,
            series_uid=(item.series_instance_uids[0] if item.study_instance_uid is None else None),
            study_uid=item.study_instance_uid,
        )
        for item in study_sources
    )


class PipelineService:
    """The only supported orchestration implementation for API and CLI calls."""

    def __init__(
        self,
        config: PipelineConfig | Mapping[str, Any] | str | Path | None = None,
        *,
        pipeline_factory: Callable[..., Any] = InternalPipeline,
        model_provider: Callable[[PipelineConfig], Sequence[ModelStatus]] = require_models,
        source_provider: Callable[[], Mapping[str, Any]] = source_state,
        hardware_provider: Callable[[object | None], Mapping[str, Any]] = (
            execution_hardware_profile
        ),
    ):
        self.config = (
            PipelineConfig.load(config)
            if isinstance(config, (str, Path))
            else PipelineConfig.model_validate(config)
        )
        self._pipeline_factory = pipeline_factory
        self._model_provider = model_provider
        self._hardware_provider = hardware_provider
        self._source = dict(source_provider())
        if self._source.get("source_dirty") and not self.config.allow_dirty:
            raise DirtySourceError(
                "The source tree has uncommitted changes. Cohort/release execution is "
                "blocked; commit the reviewed source or explicitly set "
                "runtime.allow_dirty=true for a recorded development run."
            )
        self._models: tuple[ModelStatus, ...] | None = None
        self._pipeline: Any | None = None

    @property
    def model_statuses(self) -> tuple[ModelStatus, ...]:
        if self._models is None:
            self._models = tuple(self._model_provider(self.config))
        return self._models

    def _executor(self) -> Any:
        if self._pipeline is None:
            # Model validation is deliberately completed before any inference
            # object can trigger upstream loading or network behavior.
            _ = self.model_statuses
            self._pipeline = self._pipeline_factory(
                self.config,
                timestamp=int(datetime.now(UTC).timestamp()),
            )
        return self._pipeline

    def analyze(
        self,
        input_path: str | Path,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        *,
        case_id: str | None = None,
        run_id: str | None = None,
        series_uid: str | None = None,
        update: bool = False,
        worker_mode: bool = False,
        drain_requested: Callable[[], bool] | None = None,
        drain_reason: str = "requested",
    ) -> CaseResult | BatchResult | BatchWorkerResult:
        """Analyze one CT or every safely discovered case in a directory."""

        cases = discover_case_inputs(input_path, series_uid=series_uid)
        if case_id is not None:
            if len(cases) != 1:
                raise InputDiscoveryError(
                    "--case-id can be used only when analyze resolves exactly one CT. "
                    "Use an explicit batch manifest to name multiple cases."
                )
            selected = cases[0]
            cases = (
                CaseInput(
                    selected.input_path,
                    case_id=case_id,
                    series_uid=selected.series_uid,
                    study_uid=selected.study_uid,
                ),
            )
        result = cast(
            BatchResult | BatchWorkerResult,
            self.analyze_batch(
                cases,
                output_root,
                run_id=run_id,
                update=update,
                worker_mode=worker_mode,
                drain_requested=drain_requested,
                drain_reason=drain_reason,
            ),
        )
        return result.cases[0] if len(cases) == 1 and not worker_mode else result

    def analyze_case(
        self,
        input_path: str | Path,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        *,
        case_id: str | None = None,
        run_id: str | None = None,
        series_uid: str | None = None,
        update: bool = False,
    ) -> CaseResult:
        return self.analyze_batch(
            [CaseInput(Path(input_path), case_id, series_uid)],
            output_root,
            run_id=run_id,
            update=update,
        ).cases[0]

    def analyze_batch(
        self,
        inputs: Iterable[CaseInput | str | Path | Mapping[str, Any]],
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        *,
        run_id: str | None = None,
        update: bool = False,
        worker_mode: bool = False,
        drain_requested: Callable[[], bool] | None = None,
        drain_reason: str = "requested",
    ) -> BatchResult | BatchWorkerResult:
        if drain_requested is not None and not worker_mode:
            raise ValueError("A worker drain callback requires worker_mode=True.")
        if worker_mode:
            validate_public_id(drain_reason, name="drain_reason")
        cases = tuple(CaseInput.model_validate(value) for value in inputs)
        if not cases:
            raise ValueError("analyze_batch requires at least one explicit input.")
        output = Path(output_root)
        _check_writable(output)

        input_preflight = _cached_input_preflight(cases, output)

        # Resolve identities locally because worker devices and model mounts may
        # differ even though the expensive input summaries are shared.
        prepared: list[_PreparedCase] = []
        for case, input_result in zip(cases, input_preflight, strict=True):
            input_info = input_result.input_summary
            resolved_case_id = _safe_case_id(
                case.case_id,
                input_info,
            )
            analysis_id, provenance = analysis_identity(
                input_summary=input_info,
                config=self.config,
                source=self._source,
                models=self.model_statuses,
            )
            prepared.append(
                _PreparedCase(
                    case=case,
                    input_summary=input_info,
                    case_id=resolved_case_id,
                    analysis_id=analysis_id,
                    provenance=provenance,
                    preflight_failure=input_result.failure,
                )
            )
        resolved_ids = [value.case_id for value in prepared]
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("A batch cannot contain duplicate resolved case IDs.")
        ordered_cases = [
            {"case_id": value.case_id, "analysis_id": value.analysis_id} for value in prepared
        ]
        queue_configuration_sha256 = self.config.queue_digest()
        selected_run_id = validate_public_id(
            run_id
            or (
                IMPLICIT_UPDATE_RUN_ID
                if update
                else _default_run_id(ordered_cases, queue_configuration_sha256)
            ),
            name="run_id",
        )
        plan = {
            "run_id": selected_run_id,
            "ordered_cases": ordered_cases,
            "configuration_sha256": queue_configuration_sha256,
        }
        run_root = _prepare_run_generation(
            output,
            selected_run_id,
            plan,
            update=update,
        )
        existing_manifest = run_root / "run_manifest.json"
        reuse_completed_run = existing_manifest.is_file()
        if reuse_completed_run:
            previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
            validate_payload(previous, "run_manifest.schema.json")
            previous_plan = {
                "run_id": previous.get("run_id"),
                "ordered_cases": previous.get("ordered_cases"),
                "configuration_sha256": previous.get("configuration_sha256"),
            }
            if previous_plan != plan:
                raise ExistingRunError(
                    "The requested run_id already belongs to a different immutable input/configuration plan."
                )
        config_text = yaml.safe_dump(self.config.queue_normalized(), sort_keys=False)
        config_path = run_root / "normalized_config.yaml"
        if config_path.is_file() and config_path.read_text(encoding="utf-8") != config_text:
            raise ExistingRunError(
                "The existing run has a different queue-compatible configuration."
            )
        _atomic_text(config_text, config_path)

        if reuse_completed_run:
            previous_results = tuple(
                CaseResult.from_manifest(
                    resolve_bundle_path(
                        run_root,
                        str(value["manifest"]),
                        field="cases.manifest",
                    )
                )
                for value in previous.get("cases", ())
            )
            if len(previous_results) != len(prepared):
                raise ExistingRunError("The existing run manifest has an incomplete case list.")
            aggregate_paths = {
                name: resolve_bundle_path(
                    run_root,
                    str(relative),
                    field=f"aggregate.{name}",
                )
                for name, relative in previous.get("aggregate", {}).items()
            }
            return BatchResult(
                run_id=selected_run_id,
                output_path=run_root,
                manifest_path=existing_manifest,
                cases=tuple(
                    self._as_skipped_identical(result) if result.succeeded else result
                    for result in previous_results
                ),
                aggregate_paths=aggregate_paths,
                reporting=previous.get("reporting"),
            )

        worker_started_at = _utc_now()
        state = SharedExecutionState(
            output,
            selected_run_id,
            policy=ExecutionPolicy.from_environment(
                default_max_claim_seconds=(
                    float(self.config.model_dump()["runtime"]["timeout_seconds"]) + 600.0
                )
            ),
        )
        started_at = state.initialize_run_clock(
            started_at=worker_started_at,
            plan_sha256=canonical_digest(plan),
        )
        results, _, queue_terminal, active_case_count, worker_drain_reason = (
            self._drain_shared_queue(
                prepared=prepared,
                run_id=selected_run_id,
                run_root=run_root,
                state=state,
                started_at=started_at,
                worker_mode=worker_mode,
                drain_requested=drain_requested,
                drain_reason=drain_reason,
            )
        )
        if not queue_terminal:
            self._release_executor()
            if worker_drain_reason is None:
                raise RuntimeError("A non-terminal worker exit requires a drain reason.")
            return BatchWorkerResult(
                run_id=selected_run_id,
                output_path=run_root,
                cases=tuple(results),
                planned_case_count=len(prepared),
                active_case_count=active_case_count,
                drain_reason=worker_drain_reason,
            )

        with state.coordination_guard("run-finalize"):
            # Another compatible worker may have completed finalization while
            # this process was waiting for the guard. The published manifest is
            # the completion marker and must never be rewritten.
            if existing_manifest.is_file():
                previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
                validate_payload(previous, "run_manifest.schema.json")
                previous_plan = {
                    "run_id": previous.get("run_id"),
                    "ordered_cases": previous.get("ordered_cases"),
                    "configuration_sha256": previous.get("configuration_sha256"),
                }
                if previous_plan != plan:
                    raise ExistingRunError(
                        "The requested run_id already belongs to a different immutable "
                        "input/configuration plan."
                    )
                previous_results = tuple(
                    CaseResult.from_manifest(
                        resolve_bundle_path(
                            run_root,
                            str(value["manifest"]),
                            field="cases.manifest",
                        )
                    )
                    for value in previous.get("cases", ())
                )
                if len(previous_results) != len(prepared):
                    raise ExistingRunError("The existing run manifest has an incomplete case list.")
                previous_aggregate_paths = {
                    name: resolve_bundle_path(
                        run_root,
                        str(relative),
                        field=f"aggregate.{name}",
                    )
                    for name, relative in previous.get("aggregate", {}).items()
                }
                return BatchResult(
                    run_id=selected_run_id,
                    output_path=run_root,
                    manifest_path=existing_manifest,
                    cases=previous_results,
                    aggregate_paths=previous_aggregate_paths,
                    reporting=previous.get("reporting"),
                )

            aggregate_paths = aggregate_results(
                run_root,
                results,
                include_csv=self.config.normalized()["output"]["save_csv_tables"],
            )
            # Always rebuild the terminal snapshot under the finalization guard.
            # Incremental snapshots are useful during processing, but only this
            # refresh is allowed to back the completed run manifest.
            reporting = self._refresh_incremental_run_reports(
                run_id=selected_run_id,
                run_root=run_root,
                prepared=prepared,
                state=state,
                local_results={result.case_id: result for result in results},
                started_at=started_at,
            )
            if reporting is None:
                reporting = self._collate_run_reports(
                    run_id=selected_run_id,
                    run_root=run_root,
                    results=results,
                    planned_case_ids=[item.case_id for item in prepared],
                    input_directory_names=_run_input_directory_names(prepared),
                    started_at=started_at,
                )
            case_payloads = [
                json.loads(result.manifest_path.read_text(encoding="utf-8")) for result in results
            ]
            ended_at, duration_seconds = _elapsed_run_period(
                started_at,
                [payload.get("ended_at") for payload in case_payloads],
            )
            if ended_at is None or duration_seconds is None:
                ended_at = _utc_now()
                _, duration_seconds = _elapsed_run_period(started_at, [ended_at])
            if duration_seconds is None:
                raise RuntimeError("The canonical run period could not be determined.")
            run_manifest = {
                "schema_version": "1.0.0",
                "manifest_type": "run",
                **plan,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": duration_seconds,
                "execution_status": (
                    "cancelled"
                    if any(item.execution_status == ExecutionStatus.CANCELLED for item in results)
                    else "failed"
                    if (
                        any(item.execution_status == ExecutionStatus.FAILED for item in results)
                        or reporting is not None
                        and reporting.get("status") == "failed"
                    )
                    else "succeeded"
                ),
                "case_counts": {
                    status.value: sum(item.execution_status == status for item in results)
                    for status in ExecutionStatus
                    if status not in {ExecutionStatus.PENDING, ExecutionStatus.RUNNING}
                },
                "cases": [
                    {
                        "case_id": result.case_id,
                        "analysis_id": result.analysis_id,
                        "attempt_id": result.attempt_id,
                        "execution_status": result.execution_status.value,
                        "qc_status": result.qc_status.value,
                        "manifest": result.manifest_path.relative_to(run_root).as_posix(),
                    }
                    for result in results
                ],
                "aggregate": {
                    name: path.relative_to(run_root).as_posix()
                    for name, path in aggregate_paths.items()
                },
                "reporting": reporting,
            }
            validate_payload(run_manifest, "run_manifest.schema.json")
            _atomic_json(run_manifest, existing_manifest)
            return BatchResult(
                run_id=selected_run_id,
                output_path=run_root,
                manifest_path=existing_manifest,
                cases=tuple(results),
                aggregate_paths=aggregate_paths,
                reporting=reporting,
            )

    def _collate_run_reports(
        self,
        *,
        run_id: str,
        run_root: Path,
        results: Sequence[CaseResult],
        planned_case_ids: Sequence[str],
        input_directory_names: Sequence[str],
        started_at: str,
    ) -> dict[str, Any] | None:
        """Collate one complete or in-progress immutable run snapshot."""

        if not self.config.reporting_enabled:
            return None
        settings_data = self.config.normalized()["reporting"]
        if not settings_data["combined_pdf"] or len(planned_case_ids) == 1:
            return {
                "enabled": True,
                "status": "individual_only",
                "layout": settings_data["layout"],
                "combined_pdf": None,
                "combined_manifest": None,
                "failure_code": None,
                "failure_summary": None,
            }
        output = run_root / "aggregate" / "reports" / run_id
        combined_pdf_path = output / "case_reports.pdf"
        combined_manifest_path = output / "report_manifest.json"
        try:
            from BodyComposition.reporting.contracts import ReportingSettings
            from BodyComposition.reporting.service import (
                ReportValidationError,
                collate_reports,
                load_case_report_result,
                render_failed_case_report,
            )

            settings = ReportingSettings.from_mapping(settings_data)
            reports = []
            for result in results:
                report_manifest = result.output_path / "reports/report_manifest.json"
                try:
                    report = load_case_report_result(report_manifest)
                except (FileNotFoundError, ReportValidationError):
                    failure = result.failure or {}
                    if not failure:
                        raise
                    render_failed_case_report(
                        case_id=result.case_id,
                        analysis_id=result.analysis_id,
                        output_directory=result.output_path / "reports",
                        settings=settings,
                        failure_stage=str(failure.get("stage") or "reporting"),
                        failure_code=str(failure.get("code") or "case_report_unavailable"),
                    )
                    report = load_case_report_result(report_manifest)
                reports.append(report)
            cover_summary = self._run_report_cover_summary(
                run_id=run_id,
                run_root=run_root,
                results=results,
                planned_case_ids=planned_case_ids,
                input_directory_names=input_directory_names,
                started_at=started_at,
                settings=settings,
            )
            collated = collate_reports(
                tuple(reports),
                export_id=run_id,
                output_directory=output,
                settings=settings,
                run_summary=cover_summary,
            )
            return {
                "enabled": True,
                "status": "succeeded",
                "layout": settings.layout,
                "combined_pdf": Path(collated["pdf_path"]).relative_to(run_root).as_posix(),
                "combined_manifest": Path(collated["manifest_path"])
                .relative_to(run_root)
                .as_posix(),
                "failure_code": None,
                "failure_summary": None,
            }
        except Exception as error:
            logging.error(
                "Run report collation failed (%s).",
                type(error).__name__,
            )
            retained_pdf = (
                combined_pdf_path.relative_to(run_root).as_posix()
                if combined_pdf_path.is_file()
                else None
            )
            retained_manifest = (
                combined_manifest_path.relative_to(run_root).as_posix()
                if combined_manifest_path.is_file()
                else None
            )
            return {
                "enabled": True,
                "status": "failed",
                "layout": settings_data["layout"],
                "combined_pdf": retained_pdf,
                "combined_manifest": retained_manifest,
                "failure_code": type(error).__name__,
                "failure_summary": (
                    "The latest ordered snapshot could not be published; any previously "
                    "validated combined PDF remains available, and individual scientific "
                    "results remain authoritative."
                ),
            }

    def _run_report_cover_summary(
        self,
        *,
        run_id: str,
        run_root: Path,
        results: Sequence[CaseResult],
        planned_case_ids: Sequence[str],
        input_directory_names: Sequence[str],
        started_at: str,
        settings: Any,
    ) -> dict[str, Any]:
        """Build the privacy-safe technical snapshot printed on the run cover."""

        from BodyComposition import __version__
        from BodyComposition.reporting.service import run_cover_configuration_rows

        payloads = [
            json.loads(result.manifest_path.read_text(encoding="utf-8")) for result in results
        ]

        def joined(values: Sequence[Any]) -> str | None:
            unique = [
                value
                for value in dict.fromkeys(
                    str(item).strip() for item in values if item is not None and str(item).strip()
                )
                if value
            ]
            return " | ".join(unique) if unique else None

        def nested(payload: Mapping[str, Any], *keys: str) -> Any:
            value: Any = payload
            for key in keys:
                if not isinstance(value, Mapping):
                    return None
                value = value.get(key)
            return value

        case_counts = {
            status.value: sum(result.execution_status == status for result in results)
            for status in (
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.SKIPPED_IDENTICAL,
                ExecutionStatus.CANCELLED,
            )
        }
        qc_counts = {
            status.value: sum(result.qc_status == status for result in results)
            for status in (
                QCStatus.PASS,
                QCStatus.REVIEW,
                QCStatus.FAIL,
                QCStatus.NOT_ASSESSED,
            )
        }
        durations = [
            float(payload["duration_seconds"])
            for result, payload in zip(results, payloads, strict=True)
            if result.execution_status != ExecutionStatus.SKIPPED_IDENTICAL
            and isinstance(payload.get("duration_seconds"), (int, float))
            and math.isfinite(float(payload["duration_seconds"]))
            and float(payload["duration_seconds"]) >= 0
        ]
        errors = Counter(
            (
                str((result.failure or {}).get("stage") or "pipeline"),
                str((result.failure or {}).get("code") or "pipeline_failed"),
            )
            for result in results
            if result.failure is not None
        )
        included = len(results)
        planned = len(planned_case_ids)
        if included < planned:
            document_state = "in_progress"
        elif case_counts["cancelled"]:
            document_state = "cancelled"
        elif case_counts["failed"]:
            document_state = "completed_with_errors"
        else:
            document_state = "complete"
        analysis_ended_at: str | None = None
        run_duration_seconds: float | None = None
        if document_state != "in_progress":
            analysis_ended_at, run_duration_seconds = _elapsed_run_period(
                started_at,
                [
                    payload.get("ended_at")
                    for result, payload in zip(results, payloads, strict=True)
                    if result.execution_status != ExecutionStatus.SKIPPED_IDENTICAL
                ],
            )
        config = self.config.normalized()
        landmark_config = config["measurements"]["landmarks"]
        models = [
            {"role": "Orientation", "identifier": config["orientation"]["backend"]},
            {"role": "Vertebral bodies", "identifier": config["vertebrae"]["backend"]},
            {"role": "Native compartments", "identifier": config["tissue"]["backend"]},
            {"role": "Body surface", "identifier": config["body_surface"]["backend"]},
        ]
        if landmark_config["enabled"]:
            models.append({"role": "Landmarks", "identifier": landmark_config["backend"]})
        package_versions = [
            nested(payload, "provenance", "code", "package_version") for payload in payloads
        ]
        source_revisions = [
            nested(payload, "provenance", "code", "git_commit") for payload in payloads
        ]
        output_root = run_root.parents[1]
        return {
            "run_id": run_id,
            "document_state": document_state,
            "planned_case_count": planned,
            "included_case_count": included,
            "remaining_case_count": planned - included,
            "included_case_ids": [result.case_id for result in results],
            "case_counts": case_counts,
            "qc_counts": qc_counts,
            "manual_review_case_count": sum(
                result.qc_status in {QCStatus.REVIEW, QCStatus.FAIL} for result in results
            ),
            "mean_runtime_seconds": (sum(durations) / len(durations) if durations else None),
            "runtime_case_count": len(durations),
            "started_at": started_at,
            "ended_at": analysis_ended_at,
            "run_duration_seconds": run_duration_seconds,
            "updated_at": _utc_now(),
            "layout": settings.layout,
            "pipeline": {
                "name": "BodyComposition",
                "version": joined(package_versions) or __version__,
                "configuration_sha256": self.config.queue_digest(),
                "source_revision": joined(source_revisions),
            },
            "configuration_rows": run_cover_configuration_rows(
                settings,
                normalized_config=config,
                orientation_policy=str(config["orientation"]["policy"]),
                tissue_definitions=config["measurements"]["tissue_definitions"],
            ),
            "models": models,
            "runtime": {
                "device": joined(
                    [
                        nested(payload, "provenance", "runtime", "device_type")
                        or nested(payload, "provenance", "execution", "device")
                        for payload in payloads
                    ]
                )
                or str(config["runtime"]["device"]),
                "hardware": joined(
                    [nested(payload, "provenance", "runtime", "gpu_name") for payload in payloads]
                ),
                "cuda": joined(
                    [
                        nested(payload, "provenance", "runtime", "cuda_version")
                        for payload in payloads
                    ]
                ),
                "torch": joined(
                    [
                        nested(payload, "provenance", "runtime", "torch_version")
                        for payload in payloads
                    ]
                ),
                "python": joined(
                    [
                        nested(payload, "provenance", "runtime", "python_version")
                        for payload in payloads
                    ]
                ),
                "strategy": joined(
                    [nested(payload, "provenance", "execution", "strategy") for payload in payloads]
                ),
            },
            "technical_errors": [
                {"stage": stage, "code": code, "count": count}
                for (stage, code), count in sorted(errors.items())
            ],
            "paths": {
                "output_root": output_root.name or ".",
                "run_folder": f"runs/{run_id}",
                "input_directories": list(input_directory_names),
                "output_directory": run_id,
                "case_folder_pattern": "cases/<case-id>/<analysis-id>",
                "failed_folder_pattern": "failed/<case-id>/<attempt-id>",
                "combined_pdf": (f"aggregate/reports/{run_id}/case_reports.pdf"),
            },
        }

    def _refresh_incremental_run_reports(
        self,
        *,
        run_id: str,
        run_root: Path,
        prepared: Sequence[_PreparedCase],
        state: SharedExecutionState,
        local_results: Mapping[str, CaseResult],
        started_at: str,
    ) -> dict[str, Any] | None:
        """Atomically refresh the ordered combined PDF after each terminal case."""

        settings_data = self.config.normalized()["reporting"]
        if (
            not self.config.reporting_enabled
            or not settings_data["combined_pdf"]
            or len(prepared) <= 1
        ):
            return None
        with state.coordination_guard("combined-report"):
            current: dict[str, CaseResult] = dict(local_results)
            for item in prepared:
                if item.case_id in current:
                    continue
                manifest = state.result_manifest(item.case_id)
                if manifest is not None and manifest.is_file():
                    current[item.case_id] = CaseResult.from_manifest(manifest)
            ordered = [current[item.case_id] for item in prepared if item.case_id in current]
            if not ordered:
                return None
            return self._collate_run_reports(
                run_id=run_id,
                run_root=run_root,
                results=ordered,
                planned_case_ids=[item.case_id for item in prepared],
                input_directory_names=_run_input_directory_names(prepared),
                started_at=started_at,
            )

    def _render_terminal_report(
        self,
        *,
        case_id: str,
        analysis_id: str,
        destination: Path,
        failure_stage: str,
        failure_code: str,
    ) -> None:
        """Create a one-page status report for a case that never entered inference."""

        if not self.config.reporting_enabled:
            return
        try:
            from BodyComposition.reporting.contracts import ReportingSettings
            from BodyComposition.reporting.service import render_failed_case_report

            settings = ReportingSettings.from_mapping(self.config.normalized()["reporting"])
            render_failed_case_report(
                case_id=case_id,
                analysis_id=analysis_id,
                output_directory=destination / "reports",
                settings=settings,
                failure_stage=failure_stage,
                failure_code=failure_code,
            )
        except Exception as error:
            logging.error(
                "Terminal status-page rendering failed for %s (%s).",
                case_id,
                type(error).__name__,
            )

    def _release_executor(self) -> None:
        device = None
        if self._pipeline is not None:
            device = getattr(self._pipeline, "device", None)
            release = getattr(self._pipeline, "release_models", None)
            if callable(release):
                release()
        self._pipeline = None
        release_accelerator_cache(device)

    def _ensure_runtime_strategy(
        self,
        state: SharedExecutionState,
    ) -> tuple[bool, Mapping[str, Any]]:
        executor = self._executor()
        hardware_profile = dict(
            self._hardware_provider(getattr(executor, "device", self.config.device))
        )
        requested = bool(self.config.normalized()["runtime"]["unload_models_between_stages"])
        if not requested and not state.low_memory_enabled(hardware_profile):
            return False, hardware_profile
        enable = getattr(executor, "enable_low_memory_mode", None)
        if callable(enable):
            enable()
        else:
            logging.warning(
                "Persisted low-memory execution is active, but the injected pipeline "
                "executor does not expose enable_low_memory_mode()."
            )
        return True, hardware_profile

    @staticmethod
    def _canonical_case_manifest(run_root: Path, item: _PreparedCase) -> Path:
        return run_root / "cases" / item.case_id / item.analysis_id / "case_manifest.json"

    @staticmethod
    def _superseded_case_manifests(
        run_root: Path,
        item: _PreparedCase,
    ) -> tuple[Path, ...]:
        output_root = run_root.parents[1]
        history = output_root / "superseded" / run_root.name
        if not history.is_dir():
            return ()
        candidates = tuple(
            history.glob(f"*/cases/{item.case_id}/{item.analysis_id}/case_manifest.json")
        )
        return tuple(
            sorted(
                candidates,
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        )

    def _reuse_superseded_case(
        self,
        *,
        item: _PreparedCase,
        run_root: Path,
        lease: CaseLease,
    ) -> CaseResult | None:
        """Promote an identical prior generation without rerunning inference."""

        source_manifest: Path | None = None
        source_result: CaseResult | None = None
        for candidate in self._superseded_case_manifests(run_root, item):
            try:
                result = CaseResult.from_manifest(candidate)
            except (FileNotFoundError, ValueError):
                logging.warning(
                    "Ignored an invalid superseded result for %s.",
                    item.case_id,
                )
                continue
            if (
                result.succeeded
                and result.run_id == run_root.name
                and result.case_id == item.case_id
                and result.analysis_id == item.analysis_id
            ):
                source_manifest = candidate
                source_result = result
                break
        if source_manifest is None or source_result is None:
            return None

        final = run_root / "cases" / item.case_id / item.analysis_id
        final_manifest = final / "case_manifest.json"
        if final_manifest.is_file():
            return self._as_skipped_identical(CaseResult.from_manifest(final_manifest))

        attempt_id = _new_id("attempt-reuse")
        attempt_root = run_root / ".attempts" / attempt_id
        bundle = attempt_root / "bundle"
        bundle.mkdir(parents=True)
        try:
            source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
            validate_payload(source_payload, "case_manifest.schema.json")
            for index, artifact in enumerate(source_payload["artifacts"]):
                relative = str(artifact["relative_path"])
                source_artifact = resolve_bundle_path(
                    source_manifest.parent,
                    relative,
                    field=f"artifacts[{index}].relative_path",
                )
                destination = bundle / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_artifact.resolve(strict=True), destination)
                except OSError:
                    shutil.copy2(source_artifact, destination, follow_symlinks=True)

            reused_at = _utc_now()
            source_execution = source_result.provenance.get("execution")
            source_execution = source_execution if isinstance(source_execution, Mapping) else {}
            source_payload.update(
                {
                    "attempt_id": attempt_id,
                    "execution_status": ExecutionStatus.SUCCEEDED.value,
                    "started_at": reused_at,
                    "ended_at": reused_at,
                    "duration_seconds": 0.0,
                    "input": dict(item.input_summary),
                    "provenance": {
                        **dict(item.provenance),
                        "execution": {
                            "strategy": "reused_identical_case_bundle",
                            "worker_id": lease.state.worker_id,
                            "claim_token": lease.token,
                            "reused_outputs": True,
                            "source_attempt_id": source_result.attempt_id,
                            "source_manifest_sha256": file_sha256(source_manifest),
                            "source_execution_strategy": source_execution.get("strategy"),
                        },
                    },
                }
            )
            validate_payload(source_payload, "case_manifest.schema.json")
            _atomic_json(source_payload, bundle / "case_manifest.json")
            lease.assert_owned()
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(bundle, final)
        finally:
            shutil.rmtree(attempt_root, ignore_errors=True)

        return self._as_skipped_identical(CaseResult.from_manifest(final_manifest))

    def _record_coordination_failure(
        self,
        *,
        item: _PreparedCase,
        run_root: Path,
        record: Mapping[str, Any],
    ) -> CaseResult:
        """Materialize a bounded stale-worker failure as a normal case result."""

        attempt_id = "attempt-worker-recovery"
        destination = run_root / "failed" / item.case_id / attempt_id
        manifest_path = destination / "case_manifest.json"
        if manifest_path.is_file():
            return CaseResult.from_manifest(manifest_path)
        destination.mkdir(parents=True, exist_ok=True)
        failure = {
            "stage": str(record.get("last_stage") or "worker_coordination"),
            "code": str(record.get("failure_code") or "worker_attempt_limit_exceeded"),
            "summary": str(
                record.get("failure_summary")
                or "The bounded worker retry limit was reached before a case result could be committed."
            ),
        }
        self._render_terminal_report(
            case_id=item.case_id,
            analysis_id=item.analysis_id,
            destination=destination,
            failure_stage=failure["stage"],
            failure_code=failure["code"],
        )
        flag = {
            "code": "pipeline_incomplete",
            "stage": failure["stage"],
            "severity": "error",
            "reason": failure["summary"],
            "observed": {"attempts": record.get("attempts")},
            "thresholds": {"max_attempts": record.get("attempts")},
            "suggested_review_action": (
                "Inspect the archived worker claims and explicitly start a new run after "
                "the resource or input problem has been corrected."
            ),
            "reviewer": None,
            "reviewed_at": None,
            "adjudication": None,
            "review_comment": None,
        }
        manifest = _case_manifest(
            case_id=item.case_id,
            run_id=run_root.name,
            analysis_id=item.analysis_id,
            attempt_id=attempt_id,
            execution_status=ExecutionStatus.FAILED,
            qc_status=QCStatus.FAIL,
            flags=(flag,),
            input_summary=item.input_summary,
            provenance={
                **dict(item.provenance),
                "execution": {
                    "strategy": "bounded_filesystem_queue",
                    "terminal_reason": failure["code"],
                },
            },
            scientific=_scientific_status({}, self.config),
            started_at=_utc_now(),
            duration_seconds=0.0,
            artifacts=_artifact_inventory(destination),
            failure=failure,
        )
        _atomic_json(manifest, manifest_path)
        return CaseResult.from_manifest(manifest_path)

    def _shared_result(
        self,
        *,
        item: _PreparedCase,
        run_root: Path,
        state: SharedExecutionState,
    ) -> CaseResult | None:
        canonical = self._canonical_case_manifest(run_root, item)
        if canonical.is_file():
            result = CaseResult.from_manifest(canonical)
            if not result.succeeded or result.analysis_id != item.analysis_id:
                raise ExistingRunError(
                    f"Canonical output is not a reusable success for {item.case_id}."
                )
            execution = result.provenance.get("execution", {})
            result_token = execution.get("claim_token") if isinstance(execution, Mapping) else None
            if result_token:
                completed = state.completed_record(item.case_id) or {}
                active = state.active_claim_record(item.case_id) or {}
                completed_matches = completed.get("claim_token") == result_token
                active_matches = active.get(
                    "token"
                ) == result_token and not state.active_claim_is_stale(item.case_id)
                if not completed_matches and not active_matches:
                    orphan = (
                        run_root
                        / ".orphaned"
                        / item.case_id
                        / f"{item.analysis_id}-{uuid4().hex[:12]}"
                    )
                    orphan.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        canonical.parent.rename(orphan)
                    except OSError:
                        logging.warning(
                            "Could not quarantine an unfenced late result for %s; retrying the queue scan.",
                            item.case_id,
                        )
                        return None
                    logging.warning(
                        "Quarantined an unfenced late result for %s at %s.",
                        item.case_id,
                        orphan,
                    )
                    return None
            return (
                self._as_skipped_identical(result)
                if isinstance(execution, Mapping) and execution.get("reused_outputs")
                else result
            )

        manifest = state.result_manifest(item.case_id)
        if manifest is not None and manifest.is_file():
            result = CaseResult.from_manifest(manifest)
            if (
                result.case_id != item.case_id
                or result.run_id != run_root.name
                or result.analysis_id != item.analysis_id
            ):
                raise ExistingRunError(
                    f"Filesystem queue state does not match the immutable plan for {item.case_id}."
                )
            execution = result.provenance.get("execution")
            return (
                self._as_skipped_identical(result)
                if isinstance(execution, Mapping) and execution.get("reused_outputs")
                else result
            )

        terminal = state.terminal_failure_record(item.case_id)
        if terminal is not None:
            result = self._record_coordination_failure(
                item=item,
                run_root=run_root,
                record=terminal,
            )
            state.attach_terminal_manifest(item.case_id, result.manifest_path)
            return result
        return None

    def _drain_shared_queue(
        self,
        *,
        prepared: Sequence[_PreparedCase],
        run_id: str,
        run_root: Path,
        state: SharedExecutionState,
        started_at: str,
        worker_mode: bool,
        drain_requested: Callable[[], bool] | None,
        drain_reason: str,
    ) -> tuple[list[CaseResult], dict[str, Any] | None, bool, int, str | None]:
        """Claim complete cases, optionally leaving when only live claims remain."""

        results: dict[str, CaseResult] = {}
        latest_reporting: dict[str, Any] | None = None

        def remember(result: CaseResult) -> None:
            nonlocal latest_reporting
            results[result.case_id] = result
            refreshed = self._refresh_incremental_run_reports(
                run_id=run_id,
                run_root=run_root,
                prepared=prepared,
                state=state,
                local_results=results,
                started_at=started_at,
            )
            if refreshed is not None:
                latest_reporting = refreshed

        def should_drain() -> bool:
            return bool(worker_mode and drain_requested is not None and drain_requested())

        def active_unfinished_count() -> int:
            return sum(
                state.active_claim_record(item.case_id) is not None
                and not state.active_claim_is_stale(item.case_id)
                for item in prepared
                if item.case_id not in results
            )

        def collect_shared_results() -> None:
            for item in prepared:
                if item.case_id in results:
                    continue
                shared = self._shared_result(item=item, run_root=run_root, state=state)
                if shared is not None:
                    remember(shared)

        while len(results) < len(prepared):
            made_progress = False
            for item in prepared:
                if item.case_id in results:
                    continue
                if should_drain():
                    break
                shared = self._shared_result(item=item, run_root=run_root, state=state)
                if shared is not None:
                    remember(shared)
                    made_progress = True
                    continue

                if state.stop_requested():
                    lease = state.try_claim(item.case_id)
                    if lease is None:
                        continue
                    made_progress = True
                    with lease:
                        shared = self._shared_result(
                            item=item,
                            run_root=run_root,
                            state=state,
                        )
                        if shared is not None:
                            remember(shared)
                            continue
                        result = self._record_unstarted_cancellation(
                            run_root=run_root,
                            case_id=item.case_id,
                            analysis_id=item.analysis_id,
                            input_summary=item.input_summary,
                            provenance=item.provenance,
                        )
                        state.mark_terminal_failure(
                            lease,
                            manifest_path=result.manifest_path,
                            stage="batch_scheduler",
                            exception_type="BatchStopRequested",
                        )
                        remember(result)
                    continue

                lease = state.try_claim(item.case_id)
                if lease is None:
                    continue
                made_progress = True
                with lease:
                    if should_drain():
                        continue
                    # Another worker may complete the case between the scan and claim.
                    shared = self._shared_result(item=item, run_root=run_root, state=state)
                    if shared is not None:
                        if shared.succeeded:
                            state.mark_completed(
                                lease,
                                manifest_path=shared.manifest_path,
                                reused_outputs=True,
                            )
                        remember(shared)
                        continue

                    if should_drain():
                        continue

                    lease.set_stage("input_validation")
                    if item.preflight_failure is not None:
                        result = self._record_preflight_failure(
                            run_root=run_root,
                            case_id=item.case_id,
                            analysis_id=item.analysis_id,
                            input_summary=item.input_summary,
                            provenance=item.provenance,
                            failure=item.preflight_failure,
                        )
                        state.mark_terminal_failure(
                            lease,
                            manifest_path=result.manifest_path,
                            stage="input_validation",
                            exception_type=str(item.preflight_failure["code"]),
                        )
                        remember(result)
                        if self.config.fail_fast:
                            state.request_stop(reason="fail_fast_input_failure")
                        continue

                    lease.set_stage("reuse_identical_result")
                    reused = self._reuse_superseded_case(
                        item=item,
                        run_root=run_root,
                        lease=lease,
                    )
                    if reused is not None:
                        state.mark_completed(
                            lease,
                            manifest_path=reused.manifest_path,
                            reused_outputs=True,
                        )
                        remember(reused)
                        continue

                    low_memory_mode, hardware_profile = self._ensure_runtime_strategy(state)
                    if should_drain():
                        continue
                    try:
                        result = self._execute_case(
                            case=item.case,
                            input_summary=item.input_summary,
                            case_id=item.case_id,
                            analysis_id=item.analysis_id,
                            provenance=item.provenance,
                            run_id=run_id,
                            run_root=run_root,
                            lease=lease,
                            low_memory_mode=low_memory_mode,
                            hardware_profile=hardware_profile,
                        )
                    except RestartAfterWorkerTimeout as restart:
                        result = restart.result
                        if isinstance(result, CaseResult):
                            stage = str(
                                (result.failure or {}).get("stage") or lease.stage or "pipeline"
                            )
                            with suppress(ClaimLostError):
                                state.record_failure(
                                    lease,
                                    restart.original_error,
                                    stage=stage,
                                    out_of_memory=False,
                                    manifest_path=result.manifest_path,
                                )
                        self._release_executor()
                        raise
                    except RestartAfterOutOfMemory as restart:
                        state.record_out_of_memory(
                            item.case_id,
                            restart.original_error,
                            hardware_profile=hardware_profile,
                        )
                        result = restart.result
                        if isinstance(result, CaseResult):
                            stage = str(
                                (result.failure or {}).get("stage") or lease.stage or "pipeline"
                            )
                            with suppress(ClaimLostError):
                                state.record_failure(
                                    lease,
                                    restart.original_error,
                                    stage=stage,
                                    out_of_memory=True,
                                    manifest_path=result.manifest_path,
                                )
                        self._release_executor()
                        raise
                    except ClaimLostError:
                        logging.warning(
                            "Discarded late result for %s because another worker took over its stale claim.",
                            item.case_id,
                        )
                        continue

                    if result.succeeded:
                        state.mark_completed(lease, manifest_path=result.manifest_path)
                        remember(result)
                    elif result.execution_status == ExecutionStatus.CANCELLED:
                        state.mark_terminal_failure(
                            lease,
                            manifest_path=result.manifest_path,
                            stage=str((result.failure or {}).get("stage") or lease.stage),
                            exception_type="KeyboardInterrupt",
                        )
                        remember(result)
                        state.request_stop(reason="cancelled_by_user")
                    else:
                        failure = result.failure or {}
                        # Repeating the same deterministic model/input failure in
                        # the same invocation adds cost without changing state.
                        # Only stale-worker recovery and the explicit OOM restart
                        # strategy consume the bounded automatic retry budget.
                        state.mark_terminal_failure(
                            lease,
                            manifest_path=result.manifest_path,
                            stage=str(failure.get("stage") or lease.stage),
                            exception_type=str(failure.get("code") or "pipeline_failed"),
                        )
                        remember(result)
                        if self.config.fail_fast:
                            state.request_stop(reason="fail_fast_case_failure")

            if len(results) == len(prepared):
                break
            if should_drain():
                collect_shared_results()
                if len(results) == len(prepared):
                    break
                return (
                    [results[item.case_id] for item in prepared if item.case_id in results],
                    latest_reporting,
                    False,
                    active_unfinished_count(),
                    drain_reason,
                )
            if not made_progress:
                active_case_count = active_unfinished_count()
                if worker_mode and active_case_count:
                    return (
                        [results[item.case_id] for item in prepared if item.case_id in results],
                        latest_reporting,
                        False,
                        active_case_count,
                        "other_workers_active",
                    )
                sleep(state.policy.poll_seconds)

        return (
            [results[item.case_id] for item in prepared],
            latest_reporting,
            True,
            0,
            None,
        )

    @staticmethod
    def _as_skipped_identical(result: CaseResult) -> CaseResult:
        if result.execution_status == ExecutionStatus.SKIPPED_IDENTICAL:
            return result
        return CaseResult(
            case_id=result.case_id,
            run_id=result.run_id,
            analysis_id=result.analysis_id,
            attempt_id=result.attempt_id,
            execution_status=ExecutionStatus.SKIPPED_IDENTICAL,
            qc_status=result.qc_status,
            output_path=result.output_path,
            manifest_path=result.manifest_path,
            flags=result.flags,
            provenance={**dict(result.provenance), "reused_identical": True},
        )

    def _execute_case(
        self,
        *,
        case: CaseInput,
        input_summary: Mapping[str, Any],
        case_id: str,
        analysis_id: str,
        provenance: Mapping[str, Any],
        run_id: str,
        run_root: Path,
        lease: CaseLease,
        low_memory_mode: bool,
        hardware_profile: Mapping[str, Any],
    ) -> CaseResult:
        final = run_root / "cases" / case_id / analysis_id
        final_manifest = final / "case_manifest.json"
        if final_manifest.is_file():
            existing = CaseResult.from_manifest(final_manifest)
            execution = existing.provenance.get("execution", {})
            existing_token = (
                execution.get("claim_token") if isinstance(execution, Mapping) else None
            )
            if existing_token and existing_token != lease.token:
                orphan = run_root / ".orphaned" / case_id / f"{analysis_id}-{uuid4().hex[:12]}"
                orphan.parent.mkdir(parents=True, exist_ok=True)
                final.rename(orphan)
                logging.warning(
                    "Quarantined a late result that appeared after %s was reclaimed.",
                    case_id,
                )
            elif existing.succeeded and existing.analysis_id == analysis_id:
                return CaseResult(
                    case_id=case_id,
                    run_id=run_id,
                    analysis_id=analysis_id,
                    attempt_id=existing.attempt_id,
                    execution_status=ExecutionStatus.SKIPPED_IDENTICAL,
                    qc_status=existing.qc_status,
                    output_path=final,
                    manifest_path=final_manifest,
                    flags=existing.flags,
                    provenance={**dict(existing.provenance), "reused_identical": True},
                )
            else:
                raise ExistingRunError(
                    f"Canonical output already exists but is not a reusable successful result for {case_id}."
                )

        attempt_id = _new_id("attempt")
        attempt_root = run_root / ".attempts" / attempt_id
        bundle = attempt_root / "bundle"
        bundle.mkdir(parents=True)
        started_at = _utc_now()
        timer = monotonic()
        patient_metadata: Mapping[str, str] = {}
        if input_summary.get("input_format") == "dicom":
            try:
                patient_metadata = dicom_report_patient_metadata(
                    case.input_path,
                    series_uid=case.series_uid,
                    study_uid=case.study_uid,
                )
            except Exception as error:
                logging.warning(
                    "DICOM demographics are unavailable for the local report header (%s).",
                    type(error).__name__,
                )
        memory: dict[str, Any] = {
            "id": case_id,
            "run_id": run_id,
            "analysis_id": analysis_id,
            "attempt_id": attempt_id,
            "workspace": bundle,
            "tmp/input_summary": dict(input_summary),
            "tmp/report_metadata": _report_technical_metadata(
                started_at=started_at,
                input_summary=input_summary,
                provenance=provenance,
                hardware_profile=hardware_profile,
            ),
            "tmp/report_patient_metadata": dict(patient_metadata),
            "tmp/case_lease": lease,
        }
        failure: dict[str, Any] | None = None
        caught_error: BaseException | None = None
        execution_status = ExecutionStatus.SUCCEEDED
        try:
            memory["tmp/current_action"] = "input_validation"
            analysis_input = case.input_path
            if input_summary.get("input_format") == "dicom":
                analysis_input = attempt_root / "input" / "converted_input.nii.gz"
                conversion = convert_dicom(
                    case.input_path,
                    analysis_input,
                    series_uid=case.series_uid,
                    study_uid=case.study_uid,
                )
                if not isinstance(conversion, DicomConversionResult):
                    raise RuntimeError(
                        "Internal DICOM staging unexpectedly returned a cohort result."
                    )
                if canonical_digest(conversion.input_summary) != canonical_digest(input_summary):
                    raise InputChangedError(
                        "The selected DICOM series changed after preflight; rerun the analysis."
                    )
            else:
                current_digest = file_sha256(case.input_path)
                current_size = case.input_path.stat().st_size
                if current_digest != input_summary.get(
                    "source_content_sha256"
                ) or current_size != input_summary.get("source_byte_size"):
                    raise InputChangedError(
                        "The NIfTI input changed after preflight; rerun the analysis."
                    )
            source = NiftiDataContainer(analysis_input)
            source.load_from_file()
            source.validate()
            if pixel_array_sha256(source.data) != input_summary.get("input_pixel_sha256"):
                raise InputChangedError(
                    "The loaded CT pixels differ from preflight; rerun the analysis."
                )
            memory["tmp/index"] = source
            self._executor()(memory)
        except KeyboardInterrupt:
            execution_status = ExecutionStatus.CANCELLED
            stage = str(memory.get("tmp/current_action", "pipeline"))
            failure = {
                "stage": stage,
                "code": "cancelled_by_user",
                "summary": "Execution was cancelled before the case completed.",
            }
        except Exception as error:
            caught_error = error
            execution_status = ExecutionStatus.FAILED
            stage = str(memory.get("tmp/current_action", "pipeline"))
            if is_out_of_memory_error(error):
                # Persist before report rendering, artifact hashing, or cleanup so
                # a subsequent process can select the safe strategy even if this
                # process is terminated during failure handling.
                lease.state.record_out_of_memory(
                    case_id,
                    error,
                    hardware_profile=hardware_profile,
                )
                failure = {
                    "stage": stage,
                    "code": "accelerator_out_of_memory",
                    "summary": (
                        "Accelerator memory was exhausted. The low-memory strategy was "
                        "persisted and will be used by the next invocation on equivalent "
                        "hardware."
                    ),
                }
            elif isinstance(error, TimeoutError):
                failure = {
                    "stage": stage,
                    "code": "worker_timeout",
                    "summary": (
                        "This worker's case time budget was exceeded. The attempt "
                        "remains eligible for bounded recovery by another compatible "
                        "worker or a restart with a larger timeout."
                    ),
                }
            else:
                failure = {
                    "stage": stage,
                    "code": type(error).__name__,
                    "summary": "The pipeline stage failed; no incomplete result was promoted as successful.",
                }
            logging.error(
                "Case %s failed in stage %s (%s).",
                case_id,
                stage,
                type(error).__name__,
            )

        qc_status, flags = _case_qc(memory)
        if execution_status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            qc_status = (
                QCStatus.FAIL
                if execution_status == ExecutionStatus.FAILED
                else QCStatus.NOT_ASSESSED
            )
        if execution_status != ExecutionStatus.SUCCEEDED:
            pipeline_flag = {
                "code": "pipeline_incomplete",
                "stage": failure["stage"] if failure else "pipeline",
                "severity": "error" if execution_status == ExecutionStatus.FAILED else "warning",
                "reason": failure["summary"] if failure else "Pipeline execution did not complete.",
                "observed": {"result_code": failure["code"] if failure else execution_status.value},
                "thresholds": {},
                "suggested_review_action": "Resolve the execution problem and rerun the same analysis plan.",
                "reviewer": None,
                "reviewed_at": None,
                "adjudication": None,
                "review_comment": None,
            }
            flags = (*flags, pipeline_flag)
        _write_stage_log(memory, bundle)
        scientific = _scientific_status(memory, self.config)

        if (
            execution_status != ExecutionStatus.SUCCEEDED
            and self.config.reporting_enabled
            and failure is not None
            and failure["stage"] != "RenderCaseReport"
        ):
            try:
                from BodyComposition.reporting.contracts import ReportingSettings
                from BodyComposition.reporting.service import render_failed_case_report

                prepared = memory.get("tmp/prepared_image")
                render_failed_case_report(
                    case_id=case_id,
                    analysis_id=analysis_id,
                    output_directory=bundle / "reports",
                    settings=ReportingSettings.from_mapping(self.config.normalized()),
                    failure_stage=failure["stage"],
                    failure_code=failure["code"],
                    prepared_image=(prepared.prepared_image if prepared is not None else None),
                    vertebral_result=memory.get("tmp/vertebral_result"),
                )
            except Exception as report_error:
                logging.warning(
                    "Failure-page rendering also failed for %s (%s).",
                    case_id,
                    type(report_error).__name__,
                )

        try:
            lease.assert_owned()
        except ClaimLostError:
            shutil.rmtree(attempt_root, ignore_errors=True)
            raise
        execution_provenance = {
            **dict(provenance),
            "execution": {
                "strategy": (
                    "single_segmentation_bundle"
                    if low_memory_mode
                    else getattr(
                        self._pipeline,
                        "execution_strategy",
                        "persistent_models",
                    )
                ),
                "device": str(getattr(self._pipeline, "device", "not_initialized")),
                "hardware_profile_id": hardware_profile_id(hardware_profile),
                "worker_id": lease.state.worker_id,
                "claim_token": lease.token,
            },
        }
        manifest = _case_manifest(
            case_id=case_id,
            run_id=run_id,
            analysis_id=analysis_id,
            attempt_id=attempt_id,
            execution_status=execution_status,
            qc_status=qc_status,
            flags=flags,
            input_summary=input_summary,
            provenance=execution_provenance,
            scientific=scientific,
            started_at=started_at,
            duration_seconds=monotonic() - timer,
            artifacts=_artifact_inventory(bundle),
            failure=failure,
        )
        _atomic_json(manifest, bundle / "case_manifest.json")
        lease.assert_owned()
        if execution_status == ExecutionStatus.SUCCEEDED:
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(bundle, final)
            except OSError:
                if not final_manifest.is_file():
                    raise
                late = CaseResult.from_manifest(final_manifest)
                late_execution = late.provenance.get("execution", {})
                late_token = (
                    late_execution.get("claim_token")
                    if isinstance(late_execution, Mapping)
                    else None
                )
                if not late_token or late_token == lease.token:
                    raise
                orphan = run_root / ".orphaned" / case_id / f"{analysis_id}-{uuid4().hex[:12]}"
                orphan.parent.mkdir(parents=True, exist_ok=True)
                final.rename(orphan)
                lease.assert_owned()
                os.replace(bundle, final)
            destination = final
        else:
            destination = run_root / "failed" / case_id / attempt_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(bundle, destination)
        shutil.rmtree(attempt_root, ignore_errors=True)
        result = CaseResult(
            case_id=case_id,
            run_id=run_id,
            analysis_id=analysis_id,
            attempt_id=attempt_id,
            execution_status=execution_status,
            qc_status=qc_status,
            output_path=destination,
            manifest_path=destination / "case_manifest.json",
            flags=tuple(flags),
            provenance=execution_provenance,
            failure=failure,
        )
        if caught_error is not None and is_out_of_memory_error(caught_error):
            raise RestartAfterOutOfMemory(
                case_id,
                caught_error,
                result=result,
            ) from caught_error
        if isinstance(caught_error, TimeoutError):
            raise RestartAfterWorkerTimeout(
                case_id,
                caught_error,
                result=result,
            ) from caught_error
        return result

    def _record_preflight_failure(
        self,
        *,
        run_root: Path,
        case_id: str,
        analysis_id: str,
        input_summary: Mapping[str, Any],
        provenance: Mapping[str, Any],
        failure: Mapping[str, Any],
    ) -> CaseResult:
        attempt_id = _new_id("attempt")
        destination = run_root / "failed" / case_id / attempt_id
        destination.mkdir(parents=True, exist_ok=False)
        self._render_terminal_report(
            case_id=case_id,
            analysis_id=analysis_id,
            destination=destination,
            failure_stage=str(failure["stage"]),
            failure_code=str(failure["code"]),
        )
        flag = {
            "code": "invalid_input",
            "stage": "input_validation",
            "severity": "error",
            "reason": failure["summary"],
            "observed": {"result_code": failure["code"]},
            "thresholds": {},
            "suggested_review_action": "Verify the input format, pixels, and physical metadata before rerunning.",
            "reviewer": None,
            "reviewed_at": None,
            "adjudication": None,
            "review_comment": None,
        }
        manifest = _case_manifest(
            case_id=case_id,
            run_id=run_root.name,
            analysis_id=analysis_id,
            attempt_id=attempt_id,
            execution_status=ExecutionStatus.FAILED,
            qc_status=QCStatus.FAIL,
            flags=(flag,),
            input_summary=input_summary,
            provenance=provenance,
            scientific={
                "orientation": None,
                "vertebral_backend": self.config.vertebral_backend,
                "tissue_backend": self.config.tissue_backend,
                "body_surface_backend": self.config.normalized()["body_surface"]["backend"],
                "measurement_schema_version": None,
            },
            started_at=_utc_now(),
            duration_seconds=0.0,
            artifacts=_artifact_inventory(destination),
            failure=failure,
        )
        path = _atomic_json(manifest, destination / "case_manifest.json")
        return CaseResult(
            case_id=case_id,
            run_id=run_root.name,
            analysis_id=analysis_id,
            attempt_id=attempt_id,
            execution_status=ExecutionStatus.FAILED,
            qc_status=QCStatus.FAIL,
            output_path=destination,
            manifest_path=path,
            flags=(flag,),
            provenance=dict(provenance),
            failure=failure,
        )

    def _record_unstarted_cancellation(
        self,
        *,
        run_root: Path,
        case_id: str,
        analysis_id: str,
        input_summary: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> CaseResult:
        attempt_id = _new_id("attempt")
        destination = run_root / "failed" / case_id / attempt_id
        destination.mkdir(parents=True, exist_ok=False)
        failure = {
            "stage": "batch_scheduler",
            "code": "not_started_after_stop",
            "summary": "The case was not started because the batch had already stopped.",
        }
        self._render_terminal_report(
            case_id=case_id,
            analysis_id=analysis_id,
            destination=destination,
            failure_stage=failure["stage"],
            failure_code=failure["code"],
        )
        manifest = _case_manifest(
            case_id=case_id,
            run_id=run_root.name,
            analysis_id=analysis_id,
            attempt_id=attempt_id,
            execution_status=ExecutionStatus.CANCELLED,
            qc_status=QCStatus.NOT_ASSESSED,
            flags=(),
            input_summary=input_summary,
            provenance=provenance,
            scientific={
                "orientation": None,
                "vertebral_backend": self.config.vertebral_backend,
                "tissue_backend": self.config.tissue_backend,
                "body_surface_backend": self.config.normalized()["body_surface"]["backend"],
                "measurement_schema_version": None,
            },
            started_at=_utc_now(),
            duration_seconds=0.0,
            artifacts=_artifact_inventory(destination),
            failure=failure,
        )
        path = _atomic_json(manifest, destination / "case_manifest.json")
        return CaseResult(
            case_id=case_id,
            run_id=run_root.name,
            analysis_id=analysis_id,
            attempt_id=attempt_id,
            execution_status=ExecutionStatus.CANCELLED,
            qc_status=QCStatus.NOT_ASSESSED,
            output_path=destination,
            manifest_path=path,
            provenance=dict(provenance),
            failure=failure,
        )


def aggregate_results(
    run_path: str | Path,
    results: Sequence[CaseResult] | None = None,
    *,
    include_csv: bool | None = None,
) -> dict[str, Path]:
    """Build deterministic run tables without silently dropping failures."""

    root = Path(run_path)
    if include_csv is None:
        config_path = root / "normalized_config.yaml"
        if config_path.is_file():
            stored_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            include_csv = bool(
                isinstance(stored_config, Mapping)
                and isinstance(stored_config.get("output"), Mapping)
                and stored_config["output"].get("save_csv_tables", True)
            )
        else:
            include_csv = True
    if results is None:
        run_manifest = root / "run_manifest.json"
        if run_manifest.is_file():
            payload = json.loads(run_manifest.read_text(encoding="utf-8"))
            validate_payload(payload, "run_manifest.schema.json")
            manifests = [
                resolve_bundle_path(root, str(item["manifest"]), field="cases.manifest")
                for item in payload["cases"]
            ]
        else:
            manifests = sorted(root.glob("cases/*/*/case_manifest.json"))
            manifests.extend(sorted(root.glob("failed/*/*/case_manifest.json")))
        results = tuple(CaseResult.from_manifest(path) for path in manifests)
    ordered = list(results)
    case_rows = []
    failure_rows = []
    review_rows = []
    for ordinal, result in enumerate(ordered):
        relative_manifest = result.manifest_path.relative_to(root).as_posix()
        case_rows.append(
            {
                "schema_version": "1.0.0",
                "run_id": result.run_id,
                "ordinal": ordinal,
                "case_id": result.case_id,
                "analysis_id": result.analysis_id,
                "attempt_id": result.attempt_id,
                "execution_status": result.execution_status.value,
                "qc_status": result.qc_status.value,
                "manual_review_required": result.manual_review_required,
                "case_manifest": relative_manifest,
            }
        )
        if result.failure is not None:
            failure_rows.append(
                {
                    "schema_version": "1.0.0",
                    "run_id": result.run_id,
                    "case_id": result.case_id,
                    "analysis_id": result.analysis_id,
                    "attempt_id": result.attempt_id,
                    "execution_status": result.execution_status.value,
                    "failure_stage": result.failure.get("stage"),
                    "failure_code": result.failure.get("code"),
                    "failure_summary": result.failure.get("summary"),
                    "case_manifest": relative_manifest,
                }
            )
        for flag in result.flags:
            if str(flag.get("severity", "warning")).lower() == "info":
                continue
            review_rows.append(
                {
                    "schema_version": "1.0.0",
                    "run_id": result.run_id,
                    "case_id": result.case_id,
                    "analysis_id": result.analysis_id,
                    "attempt_id": result.attempt_id,
                    "qc_status": result.qc_status.value,
                    "flag_code": flag.get("code"),
                    "stage": flag.get("stage"),
                    "severity": flag.get("severity"),
                    "reason": flag.get("reason"),
                    "observed_json": json.dumps(flag.get("observed", {}), sort_keys=True),
                    "thresholds_json": json.dumps(flag.get("thresholds", {}), sort_keys=True),
                    "suggested_review_action": flag.get("suggested_review_action"),
                    "reviewer": flag.get("reviewer"),
                    "reviewed_at": flag.get("reviewed_at"),
                    "adjudication": flag.get("adjudication"),
                    "review_comment": flag.get("review_comment"),
                    "case_manifest": relative_manifest,
                }
            )
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    definitions = {
        "cases": (
            case_rows,
            [
                "schema_version",
                "run_id",
                "ordinal",
                "case_id",
                "analysis_id",
                "attempt_id",
                "execution_status",
                "qc_status",
                "manual_review_required",
                "case_manifest",
            ],
        ),
        "failures": (
            failure_rows,
            [
                "schema_version",
                "run_id",
                "case_id",
                "analysis_id",
                "attempt_id",
                "execution_status",
                "failure_stage",
                "failure_code",
                "failure_summary",
                "case_manifest",
            ],
        ),
        "review_queue": (
            review_rows,
            [
                "schema_version",
                "run_id",
                "case_id",
                "analysis_id",
                "attempt_id",
                "qc_status",
                "flag_code",
                "stage",
                "severity",
                "reason",
                "observed_json",
                "thresholds_json",
                "suggested_review_action",
                "reviewer",
                "reviewed_at",
                "adjudication",
                "review_comment",
                "case_manifest",
            ],
        ),
    }
    paths: dict[str, Path] = {}
    for name, (rows, columns) in definitions.items():
        table = pd.DataFrame(rows, columns=columns)
        path = aggregate / f"{name}.parquet"
        temporary = aggregate / f".{name}.{uuid4().hex}.partial.parquet"
        try:
            table.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        paths[name] = path
        if include_csv:
            paths[f"{name}_csv"] = write_csv_table(
                table,
                aggregate / f"{name}.csv",
                overwrite=True,
            )
    return paths


def inspect_result(path: str | Path) -> CaseResult | BatchResult:
    target = Path(path)
    if target.is_dir():
        if (target / "case_manifest.json").is_file():
            target = target / "case_manifest.json"
        elif (target / "run_manifest.json").is_file():
            target = target / "run_manifest.json"
        else:
            raise FileNotFoundError("No case_manifest.json or run_manifest.json found.")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("manifest_type") == "case":
        return CaseResult.from_manifest(target)
    if payload.get("manifest_type") != "run":
        raise ValueError("Not a BodyComposition case or run manifest.")
    validate_payload(payload, "run_manifest.schema.json")
    root = target.parent
    cases = tuple(
        CaseResult.from_manifest(
            resolve_bundle_path(root, str(item["manifest"]), field="cases.manifest")
        )
        for item in payload.get("cases", ())
    )
    aggregate = {
        name: resolve_bundle_path(root, str(relative), field=f"aggregate.{name}")
        for name, relative in payload.get("aggregate", {}).items()
    }
    return BatchResult(
        run_id=str(payload["run_id"]),
        output_path=root,
        manifest_path=target,
        cases=cases,
        aggregate_paths=aggregate,
        reporting=payload.get("reporting"),
        schema_version=str(payload.get("schema_version", "1.0.0")),
    )


def export_result_csv(
    path: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export validated case Parquet tables without modifying the case bundle."""

    result = inspect_result(path)
    if not isinstance(result, CaseResult):
        raise ValueError("CSV export requires one case manifest, not a run manifest.")
    if not result.succeeded:
        raise ValueError("CSV export requires a successful case result.")
    source_root = result.output_path.resolve()
    output_root = Path(destination).resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("CSV export destination must be outside the immutable case bundle.")
    tables = load_measurement_tables(result.output_path / "tables")
    return write_csv_mirrors(
        tables,
        output_root,
        overwrite=overwrite,
    )


def _analysis_config(
    config: PipelineConfig | Mapping[str, Any] | str | Path | None,
    *,
    low_resource: bool,
    tissue_backend: str | None,
) -> PipelineConfig | Mapping[str, Any] | str | Path | None:
    if tissue_backend is not None:
        tissue_backend = resolve_tissue_backend(tissue_backend)
    selected_config = (
        low_resource_config(
            PipelineConfig.load(config) if isinstance(config, (str, Path)) else config
        )
        if low_resource
        else config
    )
    if low_resource and tissue_backend not in {None, "bodycomposition_resenc_m_v1"}:
        raise ConfigError(
            "low_resource=True has a fixed ResEncM tissue backend and cannot be "
            "combined with another tissue_backend."
        )
    if tissue_backend is not None and not low_resource:
        resolved = (
            PipelineConfig.load(selected_config)
            if isinstance(selected_config, (str, Path))
            else PipelineConfig.model_validate(selected_config)
        )
        value = resolved.normalized()
        value["tissue"]["backend"] = tissue_backend
        selected_config = PipelineConfig.model_validate(value)
    return selected_config


def analyze(
    input_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    config: PipelineConfig | Mapping[str, Any] | str | Path | None = None,
    case_id: str | None = None,
    run_id: str | None = None,
    series_uid: str | None = None,
    update: bool = False,
    low_resource: bool = False,
    tissue_backend: str | None = None,
) -> CaseResult | BatchResult:
    """Analyze one CT file or every safely discovered CT in a directory."""

    selected_config = _analysis_config(
        config,
        low_resource=low_resource,
        tissue_backend=tissue_backend,
    )
    return cast(
        CaseResult | BatchResult,
        PipelineService(selected_config).analyze(
            input_path,
            output_root,
            case_id=case_id,
            run_id=run_id,
            series_uid=series_uid,
            update=update,
        ),
    )


def analyze_case(
    input_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    config: PipelineConfig | Mapping[str, Any] | str | Path | None = None,
    case_id: str | None = None,
    run_id: str | None = None,
    series_uid: str | None = None,
    update: bool = False,
    low_resource: bool = False,
    tissue_backend: str | None = None,
) -> CaseResult:
    """Analyze exactly one NIfTI or selected DICOM CT using release defaults."""

    selected_config = _analysis_config(
        config,
        low_resource=low_resource,
        tissue_backend=tissue_backend,
    )
    return PipelineService(selected_config).analyze_case(
        input_path,
        output_root,
        case_id=case_id,
        run_id=run_id,
        series_uid=series_uid,
        update=update,
    )


def analyze_batch(
    inputs: Iterable[CaseInput | str | Path | Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    config: PipelineConfig | Mapping[str, Any] | str | Path | None = None,
    run_id: str | None = None,
    update: bool = False,
    low_resource: bool = False,
    tissue_backend: str | None = None,
) -> BatchResult:
    """Analyze an ordered collection using the same defaults as :func:`analyze_case`."""

    selected_config = _analysis_config(
        config,
        low_resource=low_resource,
        tissue_backend=tissue_backend,
    )
    return cast(
        BatchResult,
        PipelineService(selected_config).analyze_batch(
            inputs,
            output_root,
            run_id=run_id,
            update=update,
        ),
    )


def _reporting_settings(
    config: PipelineConfig | Mapping[str, Any] | str | Path | None,
    *,
    layout: str | None = None,
):
    from BodyComposition.reporting.contracts import ReportingSettings

    pipeline_config = (
        PipelineConfig.load(config)
        if isinstance(config, (str, Path))
        else PipelineConfig.model_validate(config)
    )
    section = dict(pipeline_config.normalized()["reporting"])
    section["enabled"] = True
    if layout is not None:
        section["layout"] = layout
    return ReportingSettings.from_mapping(section)


def render_report(
    case_input_manifest: str | Path,
    output_directory: str | Path,
    config: PipelineConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    layout: str | None = None,
):
    """Render one report from an explicit immutable report-input manifest."""

    from BodyComposition.reporting.io import load_case_report_input
    from BodyComposition.reporting.service import render_case_report

    settings = _reporting_settings(config, layout=layout)
    case = load_case_report_input(case_input_manifest)
    return render_case_report(case, output_directory, settings)


def collate_report_export(
    export_manifest: str | Path,
    output_directory: str | Path,
    config: PipelineConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    layout: str | None = None,
) -> Mapping[str, Any]:
    """Render and collate an ordered explicit report-export manifest."""

    from BodyComposition.reporting.io import (
        load_case_report_input,
        load_export_manifest,
    )
    from BodyComposition.reporting.service import (
        collate_reports,
        render_case_report,
        render_failed_case_report,
    )

    settings = _reporting_settings(config, layout=layout)
    export_id, case_manifests = load_export_manifest(export_manifest)
    output = Path(output_directory)
    rendered = []
    for manifest in case_manifests:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        case_id = validate_public_id(str(payload.get("case_id")), name="case_id")
        try:
            case = load_case_report_input(manifest)
            result = render_case_report(
                case,
                output / "cases" / case.case_id,
                settings,
            )
        except Exception as error:
            result = render_failed_case_report(
                case_id=case_id,
                analysis_id=str(payload.get("analysis_id", "analysis_unavailable")),
                output_directory=output / "cases" / case_id,
                settings=settings,
                failure_stage="report_input_validation",
                failure_code=type(error).__name__,
            )
        rendered.append(result)
    return collate_reports(
        rendered,
        export_id=export_id,
        output_directory=output,
        settings=settings,
    )


def inspect_report(
    manifest: str | Path,
    *,
    pdf_path: str | Path | None = None,
) -> Mapping[str, Any]:
    from BodyComposition.reporting.service import validate_report_manifest

    return validate_report_manifest(manifest, pdf_path=pdf_path)
