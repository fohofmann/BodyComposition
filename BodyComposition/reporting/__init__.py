"""Deterministic, PHI-minimized PDF reports derived from canonical outputs."""

from BodyComposition.reporting.contracts import (
    CaseReportInput,
    CaseReportResult,
    ReportingSettings,
    ReportMeasurementData,
)
from BodyComposition.reporting.service import (
    collate_reports,
    render_case_report,
    validate_report_manifest,
)

__all__ = [
    "CaseReportInput",
    "CaseReportResult",
    "ReportMeasurementData",
    "ReportingSettings",
    "collate_reports",
    "render_case_report",
    "validate_report_manifest",
]
