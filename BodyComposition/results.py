"""Immutable public execution results and manifest loading."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from BodyComposition.schema_validation import resolve_bundle_path, validate_payload

RESULT_SCHEMA_VERSION = "1.0.0"
RUN_SCHEMA_VERSION = "1.0.0"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED_IDENTICAL = "skipped_identical"
    CANCELLED = "cancelled"


class QCStatus(StrEnum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


def validate_public_id(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{name} must be a 1-64 character path-safe pseudonymous identifier."
        )
    return value


@dataclass(frozen=True)
class CaseResult:
    """Stable public result for one analysis attempt."""

    case_id: str
    run_id: str
    analysis_id: str
    attempt_id: str
    execution_status: ExecutionStatus
    qc_status: QCStatus
    output_path: Path
    manifest_path: Path
    flags: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    failure: Mapping[str, Any] | None = None
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_public_id(self.case_id, name="case_id")
        validate_public_id(self.run_id, name="run_id")
        validate_public_id(self.attempt_id, name="attempt_id")
        if not re.fullmatch(r"[a-f0-9]{64}", self.analysis_id):
            raise ValueError("analysis_id must be a lowercase SHA-256 digest.")
        if self.execution_status in {
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        } and not self.failure:
            raise ValueError("Failed or cancelled results require public failure details.")
        if self.execution_status in {
            ExecutionStatus.PENDING,
            ExecutionStatus.RUNNING,
        }:
            raise ValueError("A returned CaseResult must be terminal.")

    @property
    def manual_review_required(self) -> bool:
        return self.qc_status in {QCStatus.REVIEW, QCStatus.FAIL}

    @property
    def succeeded(self) -> bool:
        return self.execution_status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.SKIPPED_IDENTICAL,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "analysis_id": self.analysis_id,
            "attempt_id": self.attempt_id,
            "execution_status": self.execution_status.value,
            "qc_status": self.qc_status.value,
            "manual_review_required": self.manual_review_required,
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path),
            "flags": [dict(item) for item in self.flags],
            "provenance": dict(self.provenance),
            "failure": dict(self.failure) if self.failure is not None else None,
        }

    @classmethod
    def from_manifest(cls, path: str | Path) -> CaseResult:
        manifest = Path(path)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Not a BodyComposition case manifest.")
        validate_payload(payload, "case_manifest.schema.json")
        expected_artifacts: set[str] = set()
        for index, artifact in enumerate(payload["artifacts"]):
            relative_path = str(artifact["relative_path"])
            expected_artifacts.add(relative_path)
            artifact_path = resolve_bundle_path(
                manifest.parent,
                relative_path,
                field=f"artifacts[{index}].relative_path",
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Case artifact is missing: {artifact['relative_path']}."
                )
            if artifact_path.stat().st_size != int(artifact["byte_size"]):
                raise ValueError(
                    f"Case artifact size changed: {artifact['relative_path']}."
                )
            digest = hashlib.sha256()
            with artifact_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact["sha256"]:
                raise ValueError(
                    f"Case artifact digest changed: {artifact['relative_path']}."
                )
        actual_artifacts = {
            item.relative_to(manifest.parent).as_posix()
            for item in manifest.parent.rglob("*")
            if item.is_file()
            and item != manifest
            and ".partial" not in item.name
        }
        unexpected = sorted(actual_artifacts - expected_artifacts)
        if unexpected:
            raise ValueError(
                "Case artifact digest changed: unlisted artifact "
                f"{unexpected[0]}."
            )
        output_path = manifest.parent
        return cls(
            case_id=str(payload["case_id"]),
            run_id=str(payload["run_id"]),
            analysis_id=str(payload["analysis_id"]),
            attempt_id=str(payload["attempt_id"]),
            execution_status=ExecutionStatus(str(payload["execution_status"])),
            qc_status=QCStatus(str(payload["qc_status"])),
            output_path=output_path,
            manifest_path=manifest,
            flags=tuple(payload.get("qc_flags", ())),
            provenance=dict(payload.get("provenance", {})),
            failure=(dict(payload["failure"]) if payload.get("failure") else None),
            schema_version=str(payload.get("schema_version", RESULT_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class BatchResult:
    """Stable public result for one ordered batch execution."""

    run_id: str
    output_path: Path
    manifest_path: Path
    cases: tuple[CaseResult, ...]
    aggregate_paths: Mapping[str, Path] = field(default_factory=dict)
    reporting: Mapping[str, Any] | None = None
    schema_version: str = RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_public_id(self.run_id, name="run_id")
        if any(case.run_id != self.run_id for case in self.cases):
            raise ValueError("All case results must belong to the batch run_id.")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("A batch cannot contain duplicate case IDs.")

    @property
    def execution_status(self) -> ExecutionStatus:
        if self.reporting is not None and self.reporting.get("status") == "failed":
            return ExecutionStatus.FAILED
        if any(case.execution_status == ExecutionStatus.CANCELLED for case in self.cases):
            return ExecutionStatus.CANCELLED
        if any(case.execution_status == ExecutionStatus.FAILED for case in self.cases):
            return ExecutionStatus.FAILED
        return ExecutionStatus.SUCCEEDED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "execution_status": self.execution_status.value,
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path),
            "aggregate_paths": {
                key: str(value) for key, value in self.aggregate_paths.items()
            },
            "reporting": dict(self.reporting) if self.reporting is not None else None,
            "cases": [case.as_dict() for case in self.cases],
        }
