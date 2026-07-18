"""Atomic reporting service shared by the pipeline, Python API, and CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import jsonschema
import numpy as np
import pandas as pd
import pypdf
import reportlab
import SimpleITK as sitk
from pypdf import PdfReader, PdfWriter

from BodyComposition.measurement.physical import slice_geometry_table
from BodyComposition.reporting.contracts import (
    CASE_MANIFEST_NAME,
    CASE_REPORT_NAME,
    COMBINED_REPORT_NAME,
    MAX_REVIEW_SUMMARY_ROWS,
    MEASUREMENT_DEFINITIONS,
    REPORT_SCHEMA_VERSION,
    CaseReportInput,
    CaseReportResult,
    ReportingSettings,
    ReviewEntry,
    validate_case_id,
)
from BodyComposition.reporting.metrics import (
    aggregate_vertebral_measurements,
    collect_review_entries,
    public_text,
)
from BodyComposition.reporting.projection import (
    AxialSegmentationView,
    SagittalProjection,
    build_axial_segmentation_view,
    build_sagittal_projection,
)
from BodyComposition.reporting.render import (
    LAYOUT_REVISION,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    font_manifest,
    render_case_page,
    render_failure_page,
    render_review_summary_page,
)
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.vertebral.contracts import VertebralResult


class ReportValidationError(RuntimeError):
    """Raised when a report or manifest violates the released contract."""


_PRIVATE_PATH = re.compile(
    r"(?:^|[\s:=])(?:/[^\s;]+|~[/\\][^\s;]+|[A-Za-z]:\\[^\s;]+|file://[^\s;]+|\\\\[^\s;]+)",
    flags=re.IGNORECASE,
)


def _assert_manifest_privacy(value: Any, location: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_manifest_privacy(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_manifest_privacy(item, f"{location}[{index}]")
    elif isinstance(value, str) and _PRIVATE_PATH.search(value):
        raise ReportValidationError(
            f"Report manifest contains a forbidden local/source path at {location}."
        )


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _sitk_pixel_digest(image: sitk.Image) -> str:
    """Match the canonical orientation-stage digest over a SimpleITK array_zyx."""

    array_zyx = np.ascontiguousarray(sitk.GetArrayViewFromImage(image))
    digest = hashlib.sha256()
    digest.update(str(array_zyx.dtype).encode("ascii"))
    digest.update(np.asarray(array_zyx.shape, dtype=np.int64).tobytes())
    digest.update(array_zyx.tobytes())
    return digest.hexdigest()


def _table_digest(table: pd.DataFrame) -> str:
    payload = {
        "columns": list(table.columns),
        "dtypes": [str(value) for value in table.dtypes],
        "data": json.loads(
            table.to_json(
                orient="split",
                index=False,
                double_precision=15,
                date_format="iso",
            )
        )["data"],
    }
    return _digest_payload(payload)


def _orientation_data(case: CaseReportInput) -> Mapping[str, Any]:
    value = case.orientation_result
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    raise TypeError("Case orientation input must be a canonical result or mapping.")


def _validate_orientation_link(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    for field in ("prepared_pixel_sha256", "orientation_changed"):
        if field not in candidate:
            raise ReportValidationError(f"{source_name} lacks canonical orientation field {field}.")
        if candidate[field] != reference[field]:
            raise ReportValidationError(
                f"{source_name} disagrees with the supplied orientation result ({field})."
            )


def _validate_slice_geometry(case: CaseReportInput, geometry: ImageGeometry) -> None:
    observed = case.measurement_bundle.slices.reset_index(drop=True)
    expected = slice_geometry_table(geometry).drop(columns=["original_storage_index"])
    if len(observed) != len(expected):
        raise ReportValidationError(
            "Measurement slices do not retain exactly one row per orientation-prepared CT slice."
        )
    integer_columns = ("longitudinal_order", "slice_id", "slice_index_zyx_z")
    for column in integer_columns:
        actual = pd.to_numeric(observed[column], errors="coerce").to_numpy(dtype=float)
        wanted = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(actual, wanted):
            raise ReportValidationError(
                f"Measurement slices disagree with orientation-prepared CT geometry ({column})."
            )
    physical_columns = (
        "position_superior_mm",
        "slice_center_lps_x_mm",
        "slice_center_lps_y_mm",
        "slice_center_lps_z_mm",
        "slice_normal_lps_x",
        "slice_normal_lps_y",
        "slice_normal_lps_z",
        "slice_slab_inferior_mm",
        "slice_slab_superior_mm",
        "slice_thickness_normal_mm",
        "normal_mm_per_superior_mm",
        "in_plane_pixel_area_mm2",
    )
    for column in physical_columns:
        actual = pd.to_numeric(observed[column], errors="coerce").to_numpy(dtype=float)
        wanted = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(actual, wanted, rtol=1e-7, atol=1e-6, equal_nan=False):
            raise ReportValidationError(
                "Measurement slices disagree with orientation-prepared CT physical coordinates "
                f"({column})."
            )


def _validate_case_sources(case: CaseReportInput) -> None:
    """Reject mixed or stale orientation, vertebral, and measurement inputs."""

    orientation = _orientation_data(case)
    expected_digest = orientation.get("prepared_pixel_sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_digest):
        raise ReportValidationError(
            "The supplied orientation result lacks a valid prepared-pixel digest."
        )
    if _sitk_pixel_digest(case.prepared_image) != expected_digest:
        raise ReportValidationError(
            "The report CT pixels disagree with the supplied orientation result."
        )
    if "orientation_changed" not in orientation:
        raise ReportValidationError(
            "The supplied orientation result lacks orientation_changed."
        )

    prepared_geometry = ImageGeometry.from_sitk(case.prepared_image)
    if case.vertebral_result.geometry is None:
        raise ReportValidationError("The vertebral result lacks physical geometry.")
    assert_same_physical_domain(
        prepared_geometry,
        case.vertebral_result.geometry,
        reference_name="orientation-prepared CT",
        candidate_name="canonical vertebral-body labels",
    )
    _validate_slice_geometry(case, prepared_geometry)

    tissue_provenance = case.measurement_bundle.provenance.get("tissue")
    expected_tissue_digest = (
        tissue_provenance.get("label_sha256") if isinstance(tissue_provenance, Mapping) else None
    )
    if not isinstance(expected_tissue_digest, str) or not re.fullmatch(
        r"[a-f0-9]{64}", expected_tissue_digest
    ):
        raise ReportValidationError(
            "The measurement bundle lacks the canonical tissue-label digest."
        )
    if _array_digest(case.tissue_labels_zyx) != expected_tissue_digest:
        raise ReportValidationError(
            "The report tissue labels disagree with the supplied measurement bundle."
        )

    measurement_orientation = case.measurement_bundle.provenance.get("orientation")
    if not isinstance(measurement_orientation, Mapping):
        raise ReportValidationError(
            "The measurement bundle lacks canonical orientation provenance."
        )
    _validate_orientation_link(
        measurement_orientation,
        orientation,
        source_name="measurement provenance",
    )
    vertebral_orientation = case.vertebral_result.provenance.get("orientation")
    if isinstance(vertebral_orientation, Mapping):
        _validate_orientation_link(
            vertebral_orientation,
            orientation,
            source_name="vertebral provenance",
        )

    present_labels = {
        int(value) for value in np.unique(case.vertebral_result.vertebral_body_labels) if value != 0
    }
    table_labels = set(
        pd.to_numeric(case.measurement_bundle.vertebrae["native_label"], errors="raise").astype(int)
    )
    if present_labels != table_labels:
        raise ReportValidationError(
            "Vertebral-body labels and measurement territories contain different native labels."
        )
    for native_label, group in case.measurement_bundle.vertebrae.groupby(
        "native_label", sort=False
    ):
        levels = set(group["vertebral_level"].astype(str))
        expected_level = case.vertebral_result.label_schema.get(int(native_label))
        if len(levels) != 1 or expected_level not in levels:
            raise ReportValidationError(
                "The vertebral native-label schema and measurement level names disagree."
            )


def _prepared_image_digest(image: sitk.Image) -> str:
    geometry = {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "origin_lps_xyz": list(image.GetOrigin()),
        "direction_lps": list(image.GetDirection()),
    }
    return _digest_payload(
        {
            "pixels": _sitk_pixel_digest(image),
            "geometry": geometry,
        }
    )


def _prepared_digest(case: CaseReportInput) -> str:
    return _prepared_image_digest(case.prepared_image)


def _failure_projection_rows(
    result: VertebralResult,
    settings: ReportingSettings,
) -> list[dict[str, Any]]:
    present = {int(value) for value in np.unique(result.vertebral_body_labels) if value != 0}
    rows = []
    for centroid in result.centroids:
        if centroid.native_label not in present:
            continue
        level = result.label_schema.get(
            centroid.native_label,
            centroid.anatomical_label,
        )
        point = [float(value) for value in centroid.physical_lps_xyz]
        rows.append(
            {
                "vertebral_level": str(level),
                "native_label": int(centroid.native_label),
                "centroid_superior_mm": point[2],
                "centroid_lps_xyz": point,
                "territory_inferior_mm": None,
                "territory_superior_mm": None,
                "territory_complete": False,
                "territory_qc_status": "not_assessed",
                "territory_reason": "scientific_stage_incomplete",
                "anatomical_variant": str(level).upper() in {"T13", "L6"},
                "metrics": {
                    column: {
                        "value": None,
                        "valid": False,
                        "reason": "scientific_stage_incomplete",
                    }
                    for column in settings.measurement_columns
                },
                "source_audit": {},
            }
        )
    return sorted(
        rows,
        key=lambda row: float(row["centroid_superior_mm"]),
        reverse=True,
    )


def _partial_failure_overview(
    prepared_image: sitk.Image | None,
    vertebral_result: VertebralResult | None,
    settings: ReportingSettings,
) -> tuple[SagittalProjection | None, list[dict[str, Any]], list[dict[str, str]]]:
    """Return a safe existing-data overview, or a technical-page fallback."""

    if prepared_image is None or vertebral_result is None:
        return None, [], []
    if not isinstance(prepared_image, sitk.Image) or prepared_image.GetDimension() != 3:
        return None, [], []
    if not isinstance(vertebral_result, VertebralResult):
        return None, [], []
    if vertebral_result.geometry is None or vertebral_result.vertebral_body_labels is None:
        return None, [], []
    try:
        assert_same_physical_domain(
            ImageGeometry.from_sitk(prepared_image),
            vertebral_result.geometry,
            reference_name="orientation-prepared CT",
            candidate_name="canonical vertebral-body labels",
        )
        source_artifacts = [
            {
                "name": "prepared_ct_pixels_and_geometry",
                "sha256": _prepared_image_digest(prepared_image),
            },
            {
                "name": "vertebral_body_labels",
                "sha256": _array_digest(vertebral_result.vertebral_body_labels),
            },
            {
                "name": "vertebral_display_metadata",
                "sha256": _digest_payload(
                    {
                        "backend_id": vertebral_result.backend_id,
                        "label_schema": {
                            str(key): value
                            for key, value in sorted(vertebral_result.label_schema.items())
                        },
                        "centroids": [
                            {
                                "native_label": centroid.native_label,
                                "anatomical_label": centroid.anatomical_label,
                                "physical_lps_xyz": list(centroid.physical_lps_xyz),
                            }
                            for centroid in vertebral_result.centroids
                        ],
                    }
                ),
            },
        ]
        rows = _failure_projection_rows(vertebral_result, settings)
    except Exception:
        return None, [], []
    try:
        projection = build_sagittal_projection(
            prepared_image,
            vertebral_result,
            settings,
            centroids_lps_xyz=[row["centroid_lps_xyz"] for row in rows],
        )
    except Exception:
        return None, [], source_artifacts
    return projection, rows, source_artifacts


def renderer_manifest() -> dict[str, Any]:
    from BodyComposition import __version__

    return {
        "bodycomposition": __version__,
        "layout_revision": LAYOUT_REVISION,
        "reportlab": reportlab.Version,
        "pypdf": pypdf.__version__,
        "font": font_manifest(),
    }


def _source_artifacts(case: CaseReportInput) -> list[dict[str, str]]:
    bundle = case.measurement_bundle
    return [
        {"name": "prepared_ct_pixels_and_geometry", "sha256": _prepared_digest(case)},
        {
            "name": "vertebral_body_labels",
            "sha256": _array_digest(case.vertebral_result.vertebral_body_labels),
        },
        {
            "name": "tissue_labels",
            "sha256": _array_digest(case.tissue_labels_zyx),
        },
        {"name": "slices.parquet_content", "sha256": _table_digest(bundle.slices)},
        {"name": "vertebrae.parquet_content", "sha256": _table_digest(bundle.vertebrae)},
        {"name": "summaries.parquet_content", "sha256": _table_digest(bundle.summaries)},
        {"name": "orientation_report", "sha256": _digest_payload(_orientation_data(case))},
    ]


def _case_identity_payload(
    case: CaseReportInput,
    settings: ReportingSettings,
    source_artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "case_id": case.case_id,
        "analysis_id": case.measurement_bundle.identity.analysis_id,
        "layout": settings.layout,
        "configuration": settings.normalized(),
        "technical_metadata": dict(case.technical_metadata),
        "measurement_definitions": {
            name: MEASUREMENT_DEFINITIONS[name] for name in settings.measurement_columns
        },
        "renderer": renderer_manifest(),
        "source_artifacts": source_artifacts,
    }


def _validate_pdf(path: Path, *, expected_pages: int) -> None:
    try:
        reader = PdfReader(str(path), strict=True)
    except Exception as error:
        raise ReportValidationError(
            f"Generated PDF cannot be parsed: {error.__class__.__name__}."
        ) from error
    if len(reader.pages) != expected_pages:
        raise ReportValidationError(
            f"Generated PDF has {len(reader.pages)} pages; expected {expected_pages}."
        )
    for page in reader.pages:
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if abs(width - PAGE_WIDTH) > 0.2 or abs(height - PAGE_HEIGHT) > 0.2:
            raise ReportValidationError(
                f"Generated PDF page box is {width:.2f} x {height:.2f}, not A4 landscape."
            )


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    try:
        temporary.write_bytes(
            json.dumps(
                payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
            ).encode("ascii")
            + b"\n"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_pdf_and_manifest(
    *,
    temporary_pdf: Path,
    pdf_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Publish the PDF last, removing any stale completion marker first."""

    pdf_path.unlink(missing_ok=True)
    _atomic_json(manifest, manifest_path)
    os.replace(temporary_pdf, pdf_path)


def _manifest_schema() -> dict[str, Any]:
    schema_path = resources_schema_path("report_manifest.schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def resources_schema_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / name


def validate_report_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    pdf_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Validate schema, PDF digest, and page-count agreement when available."""

    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    _assert_manifest_privacy(payload)
    try:
        jsonschema.validate(payload, _manifest_schema())
    except jsonschema.ValidationError as error:
        raise ReportValidationError(f"Invalid report manifest: {error.message}") from error
    if pdf_path is not None:
        path = Path(pdf_path)
        expected_digest = payload.get("pdf_sha256") or payload.get("combined_pdf_sha256")
        if expected_digest != _sha256(path):
            raise ReportValidationError("Report PDF digest differs from its manifest.")
        _validate_pdf(path, expected_pages=int(payload["page_count"]))
    return payload


def load_case_report_result(manifest_path: str | Path) -> CaseReportResult:
    """Load and verify one immutable individual report for ordered collation."""

    manifest = Path(manifest_path)
    payload = validate_report_manifest(manifest)
    if payload.get("manifest_type") != "case":
        raise ReportValidationError("Expected an individual case report manifest.")
    pdf_name = payload.get("pdf_file")
    if pdf_name != CASE_REPORT_NAME:
        raise ReportValidationError("Case report manifest has an unsupported PDF name.")
    pdf_path = manifest.parent / str(pdf_name)
    validate_report_manifest(manifest, pdf_path=pdf_path)
    entries = tuple(
        ReviewEntry(
            case_id=str(value["case_id"]),
            domain=str(value["domain"]),
            code=str(value["code"]),
            reason=str(value["reason"]),
            automatic_action=str(value["automatic_action"]),
            review_status=str(value.get("review_status", "not_adjudicated")),
        )
        for value in payload.get("review_entries", ())
    )
    return CaseReportResult(
        case_id=str(payload["case_id"]),
        report_id=str(payload["report_id"]),
        layout=str(payload["layout"]),
        status=str(payload["render_status"]),
        pdf_path=pdf_path,
        manifest_path=manifest,
        pdf_sha256=str(payload["pdf_sha256"]),
        manual_review_required=bool(payload["manual_review_required"]),
        review_entries=entries,
        warnings=tuple(str(value) for value in payload.get("warnings", ())),
    )


def _validate_case_result_for_collation(
    item: CaseReportResult,
    settings: ReportingSettings,
) -> Mapping[str, Any]:
    manifest = validate_report_manifest(
        item.manifest_path,
        pdf_path=item.pdf_path,
    )
    expected = {
        "manifest_type": "case",
        "case_id": item.case_id,
        "report_id": item.report_id,
        "layout": item.layout,
        "render_status": item.status,
        "manual_review_required": item.manual_review_required,
        "pdf_sha256": item.pdf_sha256,
        "configuration": settings.normalized(),
        "review_entries": [entry.as_dict() for entry in item.review_entries],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ReportValidationError(
                f"Individual report result disagrees with its manifest ({field})."
            )
    return manifest


def render_case_report(
    case: CaseReportInput,
    output_directory: str | Path,
    settings: ReportingSettings | Mapping[str, Any],
) -> CaseReportResult:
    """Render and atomically publish one complete, one-page case report."""

    settings = (
        settings
        if isinstance(settings, ReportingSettings)
        else ReportingSettings.from_mapping(settings)
    )
    settings.validate()
    if not settings.enabled:
        raise ValueError("Explicit case rendering requires reporting.enabled=true.")
    _validate_case_sources(case)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    source_artifacts = _source_artifacts(case)
    identity_payload = _case_identity_payload(case, settings, source_artifacts)
    report_id = _digest_payload(identity_payload)
    rows = aggregate_vertebral_measurements(
        case.measurement_bundle,
        settings.measurement_columns,
    )
    review_entries = collect_review_entries(
        case.case_id,
        case.orientation_result,
        case.vertebral_result,
        case.measurement_bundle,
        case.extra_review_entries,
    )
    invalid_cells = sum(not metric["valid"] for row in rows for metric in row["metrics"].values())
    warnings = tuple(
        value
        for value, condition in (
            ("canonical_manual_review_required", bool(review_entries)),
            ("missing_or_invalid_display_measurement", invalid_cells > 0),
        )
        if condition
    )
    status = "succeeded_with_warnings" if warnings else "succeeded"
    projection = build_sagittal_projection(
        case.prepared_image,
        case.vertebral_result,
        settings,
        centroids_lps_xyz=[
            row["centroid_lps_xyz"]
            for row in rows
            if all(value is not None for value in row["centroid_lps_xyz"])
        ],
    )
    axial_view: AxialSegmentationView = build_axial_segmentation_view(
        case.prepared_image,
        case.tissue_labels_zyx,
        case.vertebral_result,
        rows,
        spacing_mm=min(float(settings.projection_spacing_mm), 1.5),
    )

    pdf_path = output / CASE_REPORT_NAME
    manifest_path = output / CASE_MANIFEST_NAME
    temporary_pdf = output / f".{CASE_REPORT_NAME}.{uuid4().hex}.partial"
    try:
        display_audit = render_case_page(
            temporary_pdf,
            case=case,
            settings=settings,
            report_id=report_id,
            projection=projection,
            axial_view=axial_view,
            rows=rows,
            review_entries=review_entries,
            warnings=warnings,
        )
        _validate_pdf(temporary_pdf, expected_pages=1)
        pdf_sha256 = _sha256(temporary_pdf)
        manifest = {
            **identity_payload,
            "manifest_type": "case",
            "report_id": report_id,
            "page_size": "A4_landscape",
            "page_count": 1,
            "render_status": status,
            "manual_review_required": bool(review_entries),
            "review_entries": [entry.as_dict() for entry in review_entries],
            "warnings": list(warnings),
            "display": {
                "ct_window_hu": list(settings.ct_window),
                "overlay_opacity": settings.overlay_opacity,
                "palette": "vertebral_region_v1_and_okabe_ito_tissue_v1",
                "measurement_columns": list(settings.measurement_columns),
                "numeric_precision": settings.numeric_precision,
                "missing_value_symbol": settings.missing_value_symbol,
                "audit": display_audit,
            },
            "pdf_file": CASE_REPORT_NAME,
            "pdf_sha256": pdf_sha256,
        }
        validate_report_manifest(manifest)
        # The PDF is the completion marker: publish its manifest first and the
        # validated PDF last so an interrupted render cannot look complete.
        _publish_pdf_and_manifest(
            temporary_pdf=temporary_pdf,
            pdf_path=pdf_path,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    finally:
        temporary_pdf.unlink(missing_ok=True)
    return CaseReportResult(
        case_id=case.case_id,
        report_id=report_id,
        layout=settings.layout,
        status=status,
        pdf_path=pdf_path,
        manifest_path=manifest_path,
        pdf_sha256=pdf_sha256,
        manual_review_required=bool(review_entries),
        review_entries=review_entries,
        warnings=warnings,
    )


def render_failed_case_report(
    *,
    case_id: str,
    analysis_id: str,
    output_directory: str | Path,
    settings: ReportingSettings | Mapping[str, Any],
    failure_stage: str,
    failure_code: str,
    prepared_image: sitk.Image | None = None,
    vertebral_result: VertebralResult | None = None,
) -> CaseReportResult:
    """Publish one explicit scientific-failure page using stable safe fields.

    Existing prepared-CT and vertebral-body outputs are displayed when their
    physical domains agree and the released projection remains constructible.
    """

    validate_case_id(case_id)
    settings = (
        settings
        if isinstance(settings, ReportingSettings)
        else ReportingSettings.from_mapping(settings)
    )
    settings.validate()
    projection, rows, source_artifacts = _partial_failure_overview(
        prepared_image,
        vertebral_result,
        settings,
    )
    partial_status = "rendered" if projection is not None else "unavailable"
    identity_payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "case_id": case_id,
        "analysis_id": analysis_id,
        "layout": settings.layout,
        "configuration": settings.normalized(),
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "partial_spine_overview": partial_status,
        "renderer": renderer_manifest(),
    }
    if source_artifacts:
        identity_payload["source_artifacts"] = source_artifacts
    report_id = _digest_payload(identity_payload)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / CASE_REPORT_NAME
    manifest_path = output / CASE_MANIFEST_NAME
    temporary_pdf = output / f".{CASE_REPORT_NAME}.{uuid4().hex}.partial"
    try:
        display_audit = render_failure_page(
            temporary_pdf,
            case_id=case_id,
            report_id=report_id,
            settings=settings,
            failure_stage=failure_stage,
            failure_code=failure_code,
            projection=projection,
            rows=rows,
        )
        _validate_pdf(temporary_pdf, expected_pages=1)
        pdf_sha256 = _sha256(temporary_pdf)
        failure_entry = ReviewEntry(
            case_id=case_id,
            domain="pipeline",
            code=public_text(failure_code, maximum=80),
            reason=(
                f"Scientific processing incomplete at {public_text(failure_stage, maximum=60)}."
            ),
            automatic_action="No complete scientific result was produced.",
        )
        warnings = [
            "scientific_analysis_incomplete",
            "measurements_unavailable",
            (
                "partial_spine_overview_only"
                if projection is not None
                else "spine_overview_unavailable"
            ),
        ]
        manifest = {
            **identity_payload,
            "manifest_type": "case",
            "report_id": report_id,
            "page_size": "A4_landscape",
            "page_count": 1,
            "render_status": "failed_page_generated",
            "manual_review_required": True,
            "review_entries": [failure_entry.as_dict()],
            "warnings": warnings,
            "display": {
                "ct_window_hu": list(settings.ct_window),
                "overlay_opacity": settings.overlay_opacity,
                "palette": "vertebral_region_v1_and_okabe_ito_tissue_v1",
                "measurement_columns": list(settings.measurement_columns),
                "numeric_precision": settings.numeric_precision,
                "missing_value_symbol": settings.missing_value_symbol,
                "audit": display_audit,
            },
            "pdf_file": CASE_REPORT_NAME,
            "pdf_sha256": pdf_sha256,
        }
        validate_report_manifest(manifest)
        _publish_pdf_and_manifest(
            temporary_pdf=temporary_pdf,
            pdf_path=pdf_path,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    finally:
        temporary_pdf.unlink(missing_ok=True)
    return CaseReportResult(
        case_id=case_id,
        report_id=report_id,
        layout=settings.layout,
        status="failed_page_generated",
        pdf_path=pdf_path,
        manifest_path=manifest_path,
        pdf_sha256=pdf_sha256,
        manual_review_required=True,
        review_entries=(failure_entry,),
        warnings=tuple(warnings),
    )


def collate_reports(
    case_reports: Sequence[CaseReportResult],
    *,
    export_id: str,
    output_directory: str | Path,
    settings: ReportingSettings | Mapping[str, Any],
) -> dict[str, Any]:
    """Collate explicit manifest order with a conditional first summary page."""

    validate_case_id(export_id)
    settings = (
        settings
        if isinstance(settings, ReportingSettings)
        else ReportingSettings.from_mapping(settings)
    )
    settings.validate()
    if not settings.combined_pdf:
        raise ValueError("Explicit report collation requires reporting.combined_pdf=true.")
    if not case_reports:
        raise ValueError("An explicit export must contain at least one case report.")
    if len({item.case_id for item in case_reports}) != len(case_reports):
        raise ValueError("An explicit report export cannot contain duplicate case IDs.")
    layouts = {item.layout for item in case_reports}
    if layouts != {settings.layout}:
        raise ValueError("Combined exports cannot mix layouts or settings.")
    flagged = [item for item in case_reports if item.manual_review_required]
    if len(flagged) > MAX_REVIEW_SUMMARY_ROWS:
        raise ValueError(
            "Manual-review summary exceeds 24 cases; split the explicit export manifest."
        )
    for item in case_reports:
        _validate_case_result_for_collation(item, settings)
    export_report_id = _digest_payload(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "export_id": export_id,
            "layout": settings.layout,
            "configuration": settings.normalized(),
            "ordered_case_reports": [item.report_id for item in case_reports],
            "renderer": renderer_manifest(),
        }
    )
    summary_offset = 1 if flagged else 0
    summary_rows = [
        {
            "case_id": item.case_id,
            "report_id": item.report_id,
            "combined_page_number": index + 1 + summary_offset,
            "review_entries": [entry.as_dict() for entry in item.review_entries],
            "render_status": item.status,
        }
        for index, item in enumerate(case_reports)
        if item.manual_review_required
    ]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / COMBINED_REPORT_NAME
    manifest_path = output / CASE_MANIFEST_NAME
    temporary_pdf = output / f".{COMBINED_REPORT_NAME}.{uuid4().hex}.partial"
    temporary_summary = output / f".manual-review-summary.{uuid4().hex}.partial.pdf"
    try:
        writer = PdfWriter()
        if flagged:
            render_review_summary_page(
                temporary_summary,
                export_report_id=export_report_id,
                total_cases=len(case_reports),
                rows=summary_rows,
            )
            _validate_pdf(temporary_summary, expected_pages=1)
            writer.append(
                str(temporary_summary), pages=(0, 1), outline_item="Manual review summary"
            )
        for item in case_reports:
            writer.append(str(item.pdf_path), pages=(0, 1), outline_item=item.case_id)
        writer.add_metadata(
            {
                "/Title": f"BodyComposition export {export_id}",
                "/Author": "BodyComposition",
                "/Subject": "Automated research and QC reports",
                "/Creator": "BodyComposition reporting",
                "/Producer": "BodyComposition reporting",
            }
        )
        with temporary_pdf.open("wb") as handle:
            writer.write(handle)
        expected_pages = len(case_reports) + summary_offset
        _validate_pdf(temporary_pdf, expected_pages=expected_pages)
        combined_sha256 = _sha256(temporary_pdf)
        cases = [
            {
                "case_id": item.case_id,
                "report_id": item.report_id,
                "render_status": item.status,
                "manual_review_required": item.manual_review_required,
                "combined_page_number": index + 1 + summary_offset,
                "individual_pdf_sha256": item.pdf_sha256,
            }
            for index, item in enumerate(case_reports)
        ]
        counts = Counter(
            (entry.domain, entry.code) for item in flagged for entry in item.review_entries
        )
        manifest = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "manifest_type": "export",
            "export_id": export_id,
            "report_id": export_report_id,
            "layout": settings.layout,
            "configuration": settings.normalized(),
            "renderer": renderer_manifest(),
            "page_size": "A4_landscape",
            "page_count": expected_pages,
            "render_status": "succeeded_with_warnings" if flagged else "succeeded",
            "manual_review_required": bool(flagged),
            "manual_review_summary": {
                "included": bool(flagged),
                "page_number": 1 if flagged else None,
                "case_count": len(flagged),
                "capacity": MAX_REVIEW_SUMMARY_ROWS,
                "counts": [
                    {"domain": domain, "code": code, "count": count}
                    for (domain, code), count in sorted(counts.items())
                ],
                "cases": summary_rows,
            },
            "cases": cases,
            "combined_pdf_file": COMBINED_REPORT_NAME,
            "combined_pdf_sha256": combined_sha256,
        }
        validate_report_manifest(manifest)
        _publish_pdf_and_manifest(
            temporary_pdf=temporary_pdf,
            pdf_path=pdf_path,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    finally:
        temporary_pdf.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
    return {"pdf_path": pdf_path, "manifest_path": manifest_path, "manifest": manifest}
