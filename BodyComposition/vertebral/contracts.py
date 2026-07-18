"""Backend-neutral result objects for vertebral segmentation and labeling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from BodyComposition.utils.geometry import ImageGeometry


class ExecutionStatus(StrEnum):
    """Execution status shared by vertebral backends."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED_IDENTICAL = "skipped_identical"
    CANCELLED = "cancelled"


class QCStatus(StrEnum):
    """Independent scientific-review status."""

    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


class QCSeverity(StrEnum):
    """Severity attached to a structured QC flag."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class QCFlag:
    """One stable, machine-readable vertebral QC finding."""

    code: str
    reason: str
    severity: QCSeverity = QCSeverity.WARNING
    stage: str = "vertebral"
    observed: Mapping[str, Any] = field(default_factory=dict)
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    suggested_review_action: str = "Review the vertebral overlay and native labels."
    reviewer: str | None = None
    reviewed_at: str | None = None
    adjudication: str | None = None
    review_comment: str | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.code.replace("_", "").isalnum():
            raise ValueError("QC flag code must be a non-empty alphanumeric snake-case value.")
        if not self.reason.strip():
            raise ValueError("QC flag reason must not be empty.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "severity": self.severity.value,
            "reason": self.reason,
            "observed": dict(self.observed),
            "thresholds": dict(self.thresholds),
            "suggested_review_action": self.suggested_review_action,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "adjudication": self.adjudication,
            "review_comment": self.review_comment,
        }


@dataclass(frozen=True)
class VertebralCentroid:
    """A native vertebral label centroid in array and physical coordinates."""

    native_label: int
    anatomical_label: str
    index_zyx: tuple[float, float, float]
    physical_lps_xyz: tuple[float, float, float]
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.native_label <= 0:
            raise ValueError("A vertebral centroid requires a positive native label.")
        if len(self.index_zyx) != 3 or len(self.physical_lps_xyz) != 3:
            raise ValueError("Centroid coordinates must contain three values.")
        if not np.all(np.isfinite(self.index_zyx)) or not np.all(np.isfinite(self.physical_lps_xyz)):
            raise ValueError("Centroid coordinates must be finite.")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Centroid confidence must be between zero and one.")


@dataclass(frozen=True)
class VertebralResult:
    """Canonical result returned by every selectable vertebral backend."""

    backend_id: str
    execution_status: ExecutionStatus
    geometry: ImageGeometry | None
    whole_vertebra_labels: np.ndarray | None
    vertebral_body_labels: np.ndarray | None
    centroids: tuple[VertebralCentroid, ...] = ()
    label_schema: Mapping[int, str] = field(default_factory=dict)
    native_outputs: Mapping[str, Path] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    qc_flags: tuple[QCFlag, ...] = ()
    error_summary: str | None = None

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("backend_id must not be empty.")
        if self.execution_status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.SKIPPED_IDENTICAL}:
            self._validate_successful_result()
        elif self.execution_status == ExecutionStatus.FAILED and not self.error_summary:
            raise ValueError("A failed vertebral result requires an error summary.")

    def _validate_successful_result(self) -> None:
        if self.geometry is None:
            raise ValueError("A successful vertebral result requires image geometry.")
        if self.vertebral_body_labels is None:
            raise ValueError("A successful vertebral result requires vertebral-body labels.")

        expected_shape_zyx = tuple(reversed(self.geometry.size_xyz))
        arrays = [("vertebral_body_labels", self.vertebral_body_labels)]
        if self.whole_vertebra_labels is not None:
            arrays.append(("whole_vertebra_labels", self.whole_vertebra_labels))
        for name, array in arrays:
            if array.ndim != 3 or array.shape != expected_shape_zyx:
                raise ValueError(
                    f"{name} shape {array.shape} does not match geometry {expected_shape_zyx}."
                )
            if not np.issubdtype(array.dtype, np.integer):
                raise TypeError(f"{name} must contain integer labels.")
            if np.any(array < 0):
                raise ValueError(f"{name} contains negative labels.")

        body = self.vertebral_body_labels
        whole = self.whole_vertebra_labels
        if whole is not None and np.any((body != 0) & (body != whole)):
            raise ValueError("Vertebral-body labels must be a labeled subset of whole vertebrae.")

        labels_present = set(int(value) for value in np.unique(body) if value != 0)
        centroid_labels = {centroid.native_label for centroid in self.centroids}
        if not centroid_labels.issubset(labels_present):
            raise ValueError("A centroid references a label absent from the vertebral-body mask.")

    @property
    def qc_status(self) -> QCStatus:
        if self.execution_status == ExecutionStatus.FAILED:
            return QCStatus.FAIL
        if self.execution_status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.SKIPPED_IDENTICAL}:
            return QCStatus.NOT_ASSESSED
        if any(flag.severity == QCSeverity.ERROR for flag in self.qc_flags):
            return QCStatus.FAIL
        if self.qc_flags:
            return QCStatus.REVIEW
        return QCStatus.PASS

    def qc_as_dicts(self) -> list[dict[str, Any]]:
        return [flag.as_dict() for flag in self.qc_flags]

    def summary(self) -> dict[str, Any]:
        """Return the JSON-safe result metadata without serializing image arrays."""

        return {
            "backend_id": self.backend_id,
            "execution_status": self.execution_status.value,
            "qc_status": self.qc_status.value,
            "label_schema": {str(key): value for key, value in self.label_schema.items()},
            "centroids": [
                {
                    "native_label": item.native_label,
                    "anatomical_label": item.anatomical_label,
                    "index_zyx": list(item.index_zyx),
                    "physical_lps_xyz": list(item.physical_lps_xyz),
                    "confidence": item.confidence,
                }
                for item in self.centroids
            ],
            "native_outputs": sorted(self.native_outputs),
            "provenance": dict(self.provenance),
            "qc_flags": self.qc_as_dicts(),
            "error_summary": self.error_summary,
        }
