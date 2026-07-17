"""Explicit immutable-manifest loading for post-hoc report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import SimpleITK as sitk

from BodyComposition.measurement.api import load_measurement_tables
from BodyComposition.measurement.contracts import MeasurementIdentity
from BodyComposition.reporting.contracts import (
    CaseReportInput,
    ReportMeasurementData,
    validate_case_id,
)
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.vertebral.contracts import (
    ExecutionStatus,
    QCFlag,
    QCSeverity,
    VertebralCentroid,
    VertebralResult,
)


CASE_INPUT_SCHEMA_VERSION = "1.0.0"


def _validate(payload: Mapping[str, Any], schema_name: str) -> None:
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as error:
        raise ValueError(f"Invalid reporting input manifest: {error.message}") from error


def _resolve(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Case manifest {name} must be a non-empty path string.")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if not path.exists():
        raise FileNotFoundError(f"Case manifest input is missing: {name}.")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.name}.")
    return value


def _qc_flag(value: Mapping[str, Any], *, default_stage: str) -> QCFlag:
    thresholds = value.get("thresholds", value.get("threshold", {}))
    return QCFlag(
        code=str(value["code"]),
        reason=str(value["reason"]),
        severity=QCSeverity(str(value.get("severity", "warning"))),
        stage=str(value.get("stage", default_stage)),
        observed=dict(value.get("observed", {})),
        thresholds=dict(thresholds or {}),
        suggested_review_action=str(
            value.get(
                "suggested_review_action",
                value.get("suggested_action", "Review the canonical QC artifact."),
            )
        ),
        reviewer=value.get("reviewer"),
        reviewed_at=value.get("reviewed_at"),
        adjudication=value.get("adjudication"),
        review_comment=value.get("review_comment"),
    )


def load_case_report_input(manifest_path: str | Path) -> CaseReportInput:
    """Load only the declared complete canonical bundle; never discover files."""

    path = Path(manifest_path)
    payload = _load_json(path)
    _validate(payload, "report_case_input.schema.json")
    if payload.get("schema_version") != CASE_INPUT_SCHEMA_VERSION:
        raise ValueError("Unsupported report case-manifest schema_version.")
    case_id = validate_case_id(payload.get("case_id"))
    root = path.parent
    prepared_path = _resolve(root, payload.get("prepared_ct"), "prepared_ct")
    body_path = _resolve(root, payload.get("vertebral_body_mask"), "vertebral_body_mask")
    vertebral_json_path = _resolve(
        root, payload.get("vertebral_result_json"), "vertebral_result_json"
    )
    measurement_directory = _resolve(
        root, payload.get("measurement_directory"), "measurement_directory"
    )
    orientation_path = _resolve(
        root, payload.get("orientation_report_json"), "orientation_report_json"
    )
    measurement_qc_path = _resolve(
        root, payload.get("measurement_qc_json"), "measurement_qc_json"
    )
    prepared = sitk.ReadImage(str(prepared_path))
    body_image = sitk.ReadImage(str(body_path))
    prepared_geometry = ImageGeometry.from_sitk(prepared)
    body_geometry = ImageGeometry.from_sitk(body_image)
    assert_same_physical_domain(
        prepared_geometry,
        body_geometry,
        reference_name="post-hoc prepared CT",
        candidate_name="post-hoc vertebral-body mask",
    )
    body_labels = sitk.GetArrayFromImage(body_image)
    vertebral_json = _load_json(vertebral_json_path)
    schema = {
        int(label): str(name)
        for label, name in vertebral_json.get("label_schema", {}).items()
    }
    centroids = tuple(
        VertebralCentroid(
            native_label=int(value["native_label"]),
            anatomical_label=str(value["anatomical_label"]),
            index_zyx=tuple(float(item) for item in value["index_zyx"]),
            physical_lps_xyz=tuple(float(item) for item in value["physical_lps_xyz"]),
            confidence=(
                float(value["confidence"])
                if value.get("confidence") is not None
                else None
            ),
        )
        for value in vertebral_json.get("centroids", [])
    )
    execution_status = ExecutionStatus(
        str(vertebral_json.get("execution_status", "succeeded"))
    )
    vertebral_result = VertebralResult(
        backend_id=str(vertebral_json["backend_id"]),
        execution_status=execution_status,
        geometry=prepared_geometry,
        whole_vertebra_labels=None,
        vertebral_body_labels=body_labels,
        centroids=centroids,
        label_schema=schema,
        provenance=dict(vertebral_json.get("provenance", {})),
        qc_flags=tuple(
            _qc_flag(value, default_stage="vertebral")
            for value in vertebral_json.get("qc_flags", [])
        ),
        error_summary=vertebral_json.get("error_summary"),
    )
    tables = load_measurement_tables(measurement_directory)
    first = tables["slices"].iloc[0]
    identity = MeasurementIdentity(
        case_id=str(first["case_id"]),
        run_id=str(first["run_id"]),
        analysis_id=str(first["analysis_id"]),
    )
    if identity.case_id != case_id:
        raise ValueError("Case manifest ID differs from canonical measurement stage table identity.")
    measurement_qc = _load_json(measurement_qc_path)
    if str(measurement_qc.get("analysis_id")) != identity.analysis_id:
        raise ValueError("Measurement QC analysis_id differs from canonical measurement tables.")
    report_bundle = ReportMeasurementData(
        identity=identity,
        slices=tables["slices"],
        vertebrae=tables["vertebrae"],
        summaries=tables["summaries"],
        provenance=dict(measurement_qc.get("provenance", {})),
        qc_flags=tuple(
            _qc_flag(value, default_stage="measurement")
            for value in measurement_qc.get("qc_flags", [])
        ),
    )
    orientation = _load_json(orientation_path)
    return CaseReportInput(
        case_id=case_id,
        prepared_image=prepared,
        vertebral_result=vertebral_result,
        measurement_bundle=report_bundle,
        orientation=orientation,
    )


def load_export_manifest(manifest_path: str | Path) -> tuple[str, list[Path]]:
    """Return an export ID and ordered explicit case-manifest paths."""

    path = Path(manifest_path)
    payload = _load_json(path)
    _validate(payload, "report_export_input.schema.json")
    export_id = validate_case_id(payload["export_id"])
    case_manifests = []
    for item in payload["cases"]:
        case_path = Path(item["case_manifest"])
        case_manifests.append(case_path if case_path.is_absolute() else path.parent / case_path)
    return export_id, case_manifests
