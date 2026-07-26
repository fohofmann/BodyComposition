"""Stable contracts for the optional case-reporting stage."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from BodyComposition.measurement.contracts import MeasurementBundle, MeasurementIdentity
from BodyComposition.orientation.core import OrientationOutcome, OrientationResult
from BodyComposition.vertebral.contracts import QCFlag, VertebralResult

REPORT_SCHEMA_VERSION = "1.0.0"
CASE_REPORT_NAME = "case_report.pdf"
CASE_MANIFEST_NAME = "report_manifest.json"
COMBINED_REPORT_NAME = "case_reports.pdf"
SUPPORTED_LAYOUTS = ("spine_overview_v1", "spine_profile_v2")
PAGE_SIZE = "A4_landscape"
SPINE_VIEW = "sagittal_thick_slab_v1"
MAX_REVIEW_SUMMARY_ROWS = 24
TECHNICAL_METADATA_FIELDS = frozenset(
    {
        "analysis_started_at",
        "data_attribution",
        "input_format",
        "pipeline_version",
        "runtime_backend",
        "runtime_hardware",
        "scanner_manufacturer",
        "scanner_model",
    }
)

MEASUREMENT_DEFINITIONS: dict[str, dict[str, str]] = {
    "sm_mean_csa_cm2": {
        "label": "SM area",
        "short_label": "SM",
        "unit": "cm2",
        "kind": "area",
        "source": "sm_mean_csa_cm2",
    },
    "sm_mean_hu": {
        "label": "SM attenuation",
        "short_label": "SM HU",
        "unit": "HU",
        "kind": "hu",
        "source": "sm_mean_hu",
        "weight_source": "sm_mean_csa_cm2",
    },
    "skeletal_muscle_tissue_hu_m29_150_mean_csa_cm2": {
        "label": "Skeletal muscle tissue area",
        "short_label": "SM",
        "unit": "cm2",
        "kind": "area",
        "source": "skeletal_muscle_tissue_hu_m29_150_mean_csa_cm2",
    },
    "skeletal_muscle_tissue_hu_m29_150_mean_hu": {
        "label": "Skeletal muscle tissue attenuation",
        "short_label": "SM HU",
        "unit": "HU",
        "kind": "hu",
        "source": "skeletal_muscle_tissue_hu_m29_150_mean_hu",
        "weight_source": "skeletal_muscle_tissue_hu_m29_150_mean_csa_cm2",
    },
    "vat_total_hu_m190_m30_mean_csa_cm2": {
        "label": "VAT area (-190 to -30 HU)",
        "short_label": "VAT",
        "unit": "cm2",
        "kind": "area",
        "source": "vat_total_hu_m190_m30_mean_csa_cm2",
    },
    "sat_total_hu_m190_m30_mean_csa_cm2": {
        "label": "SAT area (-190 to -30 HU)",
        "short_label": "SAT",
        "unit": "cm2",
        "kind": "area",
        "source": "sat_total_hu_m190_m30_mean_csa_cm2",
    },
    "total_vat_mean_csa_cm2": {
        "label": "Total VAT area",
        "short_label": "VAT",
        "unit": "cm2",
        "kind": "area",
        "source": "total_vat_mean_csa_cm2",
    },
    "sat_mean_csa_cm2": {
        "label": "SAT area",
        "short_label": "SAT",
        "unit": "cm2",
        "kind": "area",
        "source": "sat_mean_csa_cm2",
    },
    "trunk_mean_circumference_cm": {
        "label": "Trunk circumference",
        "short_label": "Trunk",
        "unit": "cm",
        "kind": "length",
        "source": "trunk_mean_circumference_cm",
    },
}
DEFAULT_MEASUREMENT_COLUMNS = (
    "skeletal_muscle_tissue_hu_m29_150_mean_csa_cm2",
    "vat_total_hu_m190_m30_mean_csa_cm2",
    "sat_total_hu_m190_m30_mean_csa_cm2",
    "trunk_mean_circumference_cm",
)

_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_case_id(case_id: str) -> str:
    """Accept only an explicitly pseudonymous, path-safe display identifier."""

    if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
        raise ValueError(
            "Report case_id must be a 1-64 character pseudonymous identifier "
            "containing only letters, numbers, dot, underscore, or hyphen."
        )
    return case_id


@dataclass(frozen=True)
class ReportingSettings:
    """Validated frozen rendering policy for both released layouts."""

    enabled: bool = False
    layout: str = "spine_profile_v2"
    individual_pdf: bool = True
    combined_pdf: bool = True
    page_size: str = PAGE_SIZE
    spine_view: str = SPINE_VIEW
    measure_aggregation: str = "territory_mean"
    measurement_columns: tuple[str, ...] = DEFAULT_MEASUREMENT_COLUMNS
    vertebral_range: str = "detected"
    include_qc_flags: bool = True
    manual_review_summary: str = "auto"
    numeric_precision: int = 1
    locale: str = "en"
    missing_value_symbol: str = "NA"
    ct_window: tuple[float, float] = (-450.0, 1050.0)
    overlay_opacity: float = 0.42
    sagittal_slab_margin_mm: float = 25.0
    projection_spacing_mm: float = 1.5

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> ReportingSettings:
        if config is None:
            settings = cls()
        else:
            if not isinstance(config, Mapping):
                raise TypeError("Reporting configuration must be a mapping.")
            section = config.get("reporting", config)
            if not isinstance(section, Mapping):
                raise ValueError("reporting must be a mapping.")
            unknown = sorted(
                set(section)
                - {
                    "enabled",
                    "layout",
                    "individual_pdf",
                    "combined_pdf",
                    "page_size",
                    "spine_view",
                    "measure_aggregation",
                    "measurement_columns",
                    "vertebral_range",
                    "include_qc_flags",
                    "manual_review_summary",
                    "numeric_precision",
                    "locale",
                    "missing_value_symbol",
                    "ct_window",
                    "overlay_opacity",
                    "sagittal_slab_margin_mm",
                    "projection_spacing_mm",
                }
            )
            if unknown:
                raise ValueError(f"Unknown reporting configuration values: {unknown}.")
            values = dict(section)
            if "measurement_columns" in values:
                columns = values["measurement_columns"]
                if not isinstance(columns, (list, tuple)):
                    raise ValueError("reporting.measurement_columns must be an ordered list.")
                values["measurement_columns"] = tuple(columns)
            if "ct_window" in values:
                window = values["ct_window"]
                if not isinstance(window, (list, tuple)) or len(window) != 2:
                    raise ValueError("reporting.ct_window must contain lower and upper HU.")
                values["ct_window"] = tuple(window)
            settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        for name in ("enabled", "individual_pdf", "combined_pdf", "include_qc_flags"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"reporting.{name} must be boolean.")
        if self.enabled and not self.individual_pdf:
            raise ValueError("Released reporting always retains one individual PDF per case.")
        if self.layout not in SUPPORTED_LAYOUTS:
            raise ValueError(f"reporting.layout must be one of {SUPPORTED_LAYOUTS}.")
        if self.page_size != PAGE_SIZE:
            raise ValueError("Released reports use fixed ISO A4 landscape pages.")
        if self.spine_view != SPINE_VIEW:
            raise ValueError(f"reporting.spine_view must be {SPINE_VIEW!r}.")
        if self.measure_aggregation != "territory_mean":
            raise ValueError("Released reports require territory_mean aggregation.")
        if self.vertebral_range != "detected":
            raise ValueError("Released reports currently support vertebral_range=detected only.")
        if self.include_qc_flags is not True:
            raise ValueError("Released reports always include canonical QC flags.")
        if self.manual_review_summary != "auto":
            raise ValueError("Released reports require manual_review_summary=auto.")
        if self.locale != "en":
            raise ValueError("Released reports currently support locale=en only.")
        if (
            isinstance(self.numeric_precision, bool)
            or not isinstance(self.numeric_precision, int)
            or not 0 <= self.numeric_precision <= 3
        ):
            raise ValueError("reporting.numeric_precision must be an integer from 0 to 3.")
        if not isinstance(self.missing_value_symbol, str) or not self.missing_value_symbol:
            raise ValueError("reporting.missing_value_symbol must not be empty.")
        if any(ord(character) > 127 for character in self.missing_value_symbol):
            raise ValueError("reporting.missing_value_symbol must contain ASCII characters only.")
        columns = self.measurement_columns
        if not columns or len(columns) > 6 or len(set(columns)) != len(columns):
            raise ValueError("Released layouts require one to six unique measurement columns.")
        unsupported = sorted(set(columns) - set(MEASUREMENT_DEFINITIONS))
        if unsupported:
            raise ValueError(f"Unsupported report measurement columns: {unsupported}.")
        lower, upper = self.ct_window
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (lower, upper)
        ):
            raise ValueError("reporting.ct_window values must be numeric.")
        if not -2048 <= float(lower) < float(upper) <= 4096:
            raise ValueError("reporting.ct_window must be ordered within [-2048, 4096] HU.")
        if (
            not isinstance(self.overlay_opacity, (int, float))
            or isinstance(self.overlay_opacity, bool)
            or not 0 <= self.overlay_opacity <= 1
        ):
            raise ValueError("reporting.overlay_opacity must be between zero and one.")
        for name in ("sagittal_slab_margin_mm", "projection_spacing_mm"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"reporting.{name} must be positive.")
        if self.projection_spacing_mm > 3:
            raise ValueError("reporting.projection_spacing_mm must not exceed 3 mm.")

    def normalized(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "layout": self.layout,
            "individual_pdf": self.individual_pdf,
            "combined_pdf": self.combined_pdf,
            "page_size": self.page_size,
            "spine_view": self.spine_view,
            "measure_aggregation": self.measure_aggregation,
            "measurement_columns": list(self.measurement_columns),
            "vertebral_range": self.vertebral_range,
            "include_qc_flags": self.include_qc_flags,
            "manual_review_summary": self.manual_review_summary,
            "numeric_precision": self.numeric_precision,
            "locale": self.locale,
            "missing_value_symbol": self.missing_value_symbol,
            "ct_window": [float(value) for value in self.ct_window],
            "overlay_opacity": float(self.overlay_opacity),
            "sagittal_slab_margin_mm": float(self.sagittal_slab_margin_mm),
            "projection_spacing_mm": float(self.projection_spacing_mm),
        }


@dataclass(frozen=True)
class ReviewEntry:
    case_id: str
    domain: str
    code: str
    reason: str
    automatic_action: str
    review_status: str = "not_adjudicated"

    def __post_init__(self) -> None:
        validate_case_id(self.case_id)
        if self.domain not in {"orientation", "vertebral", "measurement", "pipeline", "reporting"}:
            raise ValueError(f"Unsupported review domain {self.domain!r}.")
        for name in ("code", "reason", "automatic_action", "review_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Review {name} must not be empty.")

    def as_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "code": self.code,
            "reason": self.reason,
            "automatic_action": self.automatic_action,
            "review_status": self.review_status,
        }


@dataclass(frozen=True)
class ReportMeasurementData:
    """Validated read-only measurement-table view used by post-hoc reporting."""

    identity: MeasurementIdentity
    slices: pd.DataFrame
    vertebrae: pd.DataFrame
    summaries: pd.DataFrame
    provenance: Mapping[str, Any] = field(default_factory=dict)
    qc_flags: tuple[QCFlag, ...] = ()

    def __post_init__(self) -> None:
        if self.slices.empty or len(self.summaries) != 1:
            raise ValueError("Post-hoc reporting requires nonempty slices and one summary row.")
        for name, table in (
            ("slices", self.slices),
            ("vertebrae", self.vertebrae),
            ("summaries", self.summaries),
        ):
            if not isinstance(table, pd.DataFrame):
                raise TypeError(f"Report {name} must be a pandas DataFrame.")
            for column, expected in self.identity.as_columns().items():
                if not table.empty and not table[column].eq(expected).all():
                    raise ValueError(f"Report {name} has inconsistent {column} values.")


@dataclass(frozen=True)
class CaseReportInput:
    """Complete in-memory inputs for one derived report page."""

    case_id: str
    prepared_image: sitk.Image
    vertebral_result: VertebralResult
    measurement_bundle: MeasurementBundle | ReportMeasurementData
    tissue_labels_zyx: np.ndarray
    orientation: OrientationResult | OrientationOutcome | Mapping[str, Any]
    technical_metadata: Mapping[str, str | None] = field(default_factory=dict)
    extra_review_entries: tuple[ReviewEntry, ...] = ()

    def __post_init__(self) -> None:
        validate_case_id(self.case_id)
        if (
            not isinstance(self.prepared_image, sitk.Image)
            or self.prepared_image.GetDimension() != 3
        ):
            raise TypeError(
                "Case reporting requires one three-dimensional prepared SimpleITK image."
            )
        if self.measurement_bundle.identity.case_id != self.case_id:
            raise ValueError("Report case_id differs from the measurement bundle case_id.")
        if self.vertebral_result.vertebral_body_labels is None:
            raise ValueError("Case reporting requires canonical vertebral-body labels.")
        labels = np.asarray(self.tissue_labels_zyx)
        expected_shape = tuple(reversed(self.prepared_image.GetSize()))
        if labels.ndim != 3 or labels.shape != expected_shape:
            raise ValueError(
                "Case reporting tissue-label array_zyx shape does not match the prepared CT."
            )
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise TypeError("Case reporting tissue labels must be non-negative integers.")
        metadata = dict(self.technical_metadata)
        unknown = sorted(set(metadata) - TECHNICAL_METADATA_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported report technical metadata fields: {unknown}.")
        for name, value in metadata.items():
            if value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > 160
            ):
                raise ValueError(
                    f"Report technical metadata {name} must be a non-empty string or null."
                )
        if metadata.get("input_format") not in {None, "dicom", "nifti", "unknown"}:
            raise ValueError("Report technical metadata input_format is unsupported.")
        object.__setattr__(self, "technical_metadata", metadata)

    @property
    def orientation_result(self) -> OrientationResult | Mapping[str, Any]:
        if isinstance(self.orientation, OrientationOutcome):
            return self.orientation.result
        return self.orientation


@dataclass(frozen=True)
class CaseReportResult:
    case_id: str
    report_id: str
    layout: str
    status: str
    pdf_path: Path
    manifest_path: Path
    pdf_sha256: str
    manual_review_required: bool
    review_entries: tuple[ReviewEntry, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_case_id(self.case_id)
        if self.status not in {"succeeded", "succeeded_with_warnings", "failed_page_generated"}:
            raise ValueError(f"Unsupported report status {self.status!r}.")
