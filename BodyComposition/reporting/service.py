"""Atomic reporting service shared by the pipeline, Python API, and CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from io import BytesIO
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

from BodyComposition.measurement.contracts import (
    MEASUREMENT_SCHEMA_VERSION,
    VERTEBRAL_TERRITORY_SCHEMA_VERSION,
    validate_hu_distribution_contract,
)
from BodyComposition.measurement.physical import slice_geometry_table
from BodyComposition.reporting.contracts import (
    CASE_MANIFEST_NAME,
    CASE_REPORT_NAME,
    COMBINED_REPORT_NAME,
    MEASUREMENT_DEFINITIONS,
    REPORT_SCHEMA_VERSION,
    CaseReportInput,
    CaseReportResult,
    ReportingSettings,
    ReviewEntry,
    validate_case_id,
)
from BodyComposition.reporting.metrics import (
    ROUTINE_BOUNDARY_REVIEW_CODES,
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
    RUN_COVER_HIDDEN_MODEL_ROLES,
    RUN_COVER_MAX_CONFIGURATION_ROWS,
    RUN_COVER_MAX_PIPELINE_ROWS,
    RUN_COVER_REVISION,
    font_manifest,
    page_number_overlay,
    render_case_page,
    render_failure_page,
    render_run_cover_page,
)
from BodyComposition.utils.digests import array_sha256
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


def _sitk_pixel_digest(image: sitk.Image) -> str:
    """Match the canonical orientation-stage digest over a SimpleITK array_zyx."""

    return array_sha256(sitk.GetArrayViewFromImage(image))


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
    expected = slice_geometry_table(geometry)
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
        "slice_spacing_normal_mm",
        "slice_spacing_superior_mm",
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

    measurement_tables = [
        ("slices", case.measurement_bundle.slices),
        ("vertebrae", case.measurement_bundle.vertebrae),
        ("summaries", case.measurement_bundle.summaries),
        ("hu_distributions", case.measurement_bundle.hu_distributions),
    ]
    signature = getattr(case.measurement_bundle, "signature", None)
    if signature is not None:
        measurement_tables.append(("signature", signature))
    for table_name, table in measurement_tables:
        if (
            "schema_version" not in table
            or not table["schema_version"].eq(MEASUREMENT_SCHEMA_VERSION).all()
        ):
            raise ReportValidationError(
                f"Report table {table_name!r} does not use the current measurement schema "
                f"{MEASUREMENT_SCHEMA_VERSION}; regenerate measurements before reporting."
            )
    try:
        validate_hu_distribution_contract(
            case.measurement_bundle.hu_distributions,
            table_name="report hu_distributions",
        )
    except ValueError as error:
        raise ReportValidationError(
            "Report compartment HU distributions violate the current "
            "native-compartment "
            "contract; regenerate measurements before reporting."
        ) from error
    vertebrae = case.measurement_bundle.vertebrae
    if (
        "vertebral_territory_schema_version" not in vertebrae
        or not vertebrae["vertebral_territory_schema_version"]
        .eq(VERTEBRAL_TERRITORY_SCHEMA_VERSION)
        .all()
    ):
        raise ReportValidationError(
            "Report vertebral measurements use a stale territory policy; "
            "regenerate measurements before reporting."
        )

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
        raise ReportValidationError("The supplied orientation result lacks orientation_changed.")

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

    body_composition_provenance = case.measurement_bundle.provenance.get(
        "body_composition"
    )
    tissue_provenance = (
        body_composition_provenance.get("tissues")
        if isinstance(body_composition_provenance, Mapping)
        else None
    )
    expected_tissue_digest = (
        tissue_provenance.get("sha256")
        if isinstance(tissue_provenance, Mapping)
        else None
    )
    if not isinstance(expected_tissue_digest, str) or not re.fullmatch(
        r"[a-f0-9]{64}", expected_tissue_digest
    ):
        raise ReportValidationError(
            "The measurement bundle lacks the canonical tissue-label digest."
        )
    if array_sha256(case.tissue_labels_zyx) != expected_tissue_digest:
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
                "sha256": array_sha256(vertebral_result.vertebral_body_labels),
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
        "run_cover_revision": RUN_COVER_REVISION,
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
            "sha256": array_sha256(case.vertebral_result.vertebral_body_labels),
        },
        {
            "name": "tissue_labels",
            "sha256": array_sha256(case.tissue_labels_zyx),
        },
        {"name": "slices.parquet_content", "sha256": _table_digest(bundle.slices)},
        {"name": "vertebrae.parquet_content", "sha256": _table_digest(bundle.vertebrae)},
        {"name": "summaries.parquet_content", "sha256": _table_digest(bundle.summaries)},
        {
            "name": "hu_distributions.parquet_content",
            "sha256": _table_digest(bundle.hu_distributions),
        },
        {"name": "orientation_report", "sha256": _digest_payload(_orientation_data(case))},
    ]


def _case_identity_payload(
    case: CaseReportInput,
    settings: ReportingSettings,
    source_artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    patient_metadata = {
        key: value
        for key, value in sorted(case.patient_metadata.items())
        if value is not None
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "case_id": case.case_id,
        "analysis_id": case.measurement_bundle.identity.analysis_id,
        "layout": settings.layout,
        "configuration": settings.normalized(),
        "patient_header": {
            "present_fields": list(patient_metadata),
            "content_sha256": _digest_payload(patient_metadata),
        },
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
    publication = payload.get("publication")
    if (
        payload.get("manifest_type") == "export"
        and isinstance(publication, Mapping)
        and publication.get("snapshot_id") != payload.get("report_id")
    ):
        raise ReportValidationError(
            "Combined-report snapshot identity differs from its report identifier."
        )
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
                "palette": "vertebral_region_modern_v2_and_warm_tissue_v2",
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
                "palette": "vertebral_region_modern_v2_and_warm_tissue_v2",
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


def _joined_cover_values(values: Sequence[Any]) -> str | None:
    unique = [
        value
        for value in dict.fromkeys(
            str(item).strip() for item in values if item is not None and str(item).strip()
        )
        if value
    ]
    return " | ".join(unique) if unique else None


def _cover_number(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _cover_hu_range(
    tissue_definitions: Mapping[str, Any] | None,
    *,
    source_labels: set[str],
) -> str | None:
    if not tissue_definitions:
        return None
    ranges = {
        tuple(item["hu_range"])
        for item in tissue_definitions.values()
        if isinstance(item, Mapping)
        and item.get("enabled", True)
        and isinstance(item.get("source_labels"), Sequence)
        and source_labels.intersection(str(label) for label in item["source_labels"])
        and isinstance(item.get("hu_range"), Sequence)
        and len(item["hu_range"]) == 2
    }
    if len(ranges) != 1:
        return None
    lower, upper = next(iter(ranges))
    return f"{_cover_number(lower)}..{_cover_number(upper)}"


def run_cover_configuration_rows(
    settings: ReportingSettings,
    *,
    normalized_config: Mapping[str, Any] | None = None,
    orientation_policy: str | None = None,
    tissue_definitions: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return settings that affect results or scientific reproducibility."""

    rows: list[dict[str, str]] = []
    config = normalized_config if isinstance(normalized_config, Mapping) else {}

    def section(name: str) -> Mapping[str, Any]:
        value = config.get(name)
        return value if isinstance(value, Mapping) else {}

    def add(label: str, value: str | None) -> None:
        if value is not None and str(value).strip():
            rows.append({"label": label, "value": str(value)})

    def on_off(value: Any) -> str:
        return "On" if bool(value) else "Off"

    def percentage(value: Any) -> str:
        return f"{_cover_number(float(value) * 100.0)}%"

    analysis = section("analysis")
    scope = analysis.get("scope")
    if scope:
        scope_text = {
            "full_ct": "Full CT",
            "l3_vertebral_level": "L3 vertebral level",
        }.get(str(scope), str(scope).replace("_", " "))
        l3 = analysis.get("l3")
        if (
            scope == "l3_vertebral_level"
            and isinstance(l3, Mapping)
            and l3.get("inference_context_mm") is not None
        ):
            scope_text = (
                f"{scope_text} | context "
                f"{_cover_number(l3['inference_context_mm'])} mm"
            )
        add("Analysis scope", scope_text)

    orientation = section("orientation")
    if orientation_policy is None and orientation.get("policy") is not None:
        orientation_policy = str(orientation["policy"])
    if orientation_policy:
        policy = {
            "check_and_safe_repair": "Check and safe repair",
        }.get(str(orientation_policy), str(orientation_policy).replace("_", " "))
        add("Orientation policy", policy)

    confidence = orientation.get("confidence")
    if isinstance(confidence, Mapping):
        if (
            confidence.get("min_equivariant_votes") is not None
            and confidence.get("max_obliquity_deg") is not None
        ):
            add(
                "Orientation decision",
                (
                    f">={_cover_number(confidence['min_equivariant_votes'])} votes | "
                    f"obliquity <={_cover_number(confidence['max_obliquity_deg'])} deg"
                ),
            )
        if (
            confidence.get("body_threshold_hu") is not None
            and confidence.get("min_body_extent_mm") is not None
            and confidence.get("min_body_pixels_per_slice") is not None
        ):
            add(
                "Anatomy evidence",
                (
                    f">{_cover_number(confidence['body_threshold_hu'])} HU | "
                    f">={_cover_number(confidence['min_body_extent_mm'])} mm | "
                    f">={_cover_number(confidence['min_body_pixels_per_slice'])} px/slice"
                ),
            )
        if confidence.get("allow_axial_180_repair") is not None:
            add(
                "Axial 180 repair",
                on_off(confidence["allow_axial_180_repair"]),
            )

    body_surface = section("body_surface")
    if all(
        body_surface.get(name) is not None
        for name in ("threshold_hu", "closing_radius_mm", "smoothing_sigma_mm")
    ):
        add(
            "Body surface",
            (
                f"{_cover_number(body_surface['threshold_hu'])} HU | "
                f"close {_cover_number(body_surface['closing_radius_mm'])} mm | "
                f"smooth {_cover_number(body_surface['smoothing_sigma_mm'])} mm"
            ),
        )
    if all(
        body_surface.get(name) is not None
        for name in ("min_component_volume_mm3", "minimum_component_area_mm2")
    ):
        add(
            "Surface cleanup",
            (
                f">={_cover_number(body_surface['min_component_volume_mm3'])} mm3 | "
                f">={_cover_number(body_surface['minimum_component_area_mm2'])} mm2"
            ),
        )

    measurements = section("measurements")
    if tissue_definitions is None:
        value = measurements.get("tissue_definitions")
        if isinstance(value, Mapping):
            tissue_definitions = value
    if (
        measurements.get("full_coverage_tolerance") is not None
        and measurements.get("l3_slab_length_mm") is not None
    ):
        add(
            "Measurement coverage",
            (
                f"{percentage(measurements['full_coverage_tolerance'])} | "
                f"L3 slab {_cover_number(measurements['l3_slab_length_mm'])} mm"
            ),
        )
    vertebral_extent = measurements.get("vertebral_extent")
    if (
        isinstance(vertebral_extent, Mapping)
        and vertebral_extent.get("minimum_component_voxels") is not None
        and vertebral_extent.get("maximum_removed_fraction") is not None
    ):
        add(
            "Vertebral cleanup",
            (
                f">={_cover_number(vertebral_extent['minimum_component_voxels'])} voxels | "
                f"removed <={percentage(vertebral_extent['maximum_removed_fraction'])}"
            ),
        )
    sm_range = _cover_hu_range(
        tissue_definitions,
        source_labels={"SM"},
    )
    at_range = _cover_hu_range(
        tissue_definitions,
        source_labels={"SAT", "aVAT", "tVAT", "VAT"},
    )
    if sm_range is None and any(
        "hu_m29_150" in column for column in settings.measurement_columns
    ):
        sm_range = "-29..150"
    if at_range is None and any(
        "hu_m190_m30" in column for column in settings.measurement_columns
    ):
        at_range = "-190..-30"
    if sm_range is not None:
        add("SM HU filter", f"{sm_range} HU")
    if at_range is not None:
        add("AT HU filter", f"{at_range} HU")
    if sm_range is None and at_range is None:
        add("HU filter", "Inactive")

    runtime = section("runtime")
    if runtime.get("deterministic") is not None:
        add(
            "Deterministic execution",
            on_off(runtime["deterministic"]),
        )

    return rows


def _run_cover_quality_issues(
    case_reports: Sequence[CaseReportResult],
) -> list[dict[str, Any]]:
    """Aggregate non-technical review findings once per affected case."""

    counts: Counter[tuple[str, str]] = Counter()
    for report in case_reports:
        case_findings = {
            (entry.domain, entry.code)
            for entry in report.review_entries
            if entry.domain not in {"pipeline", "reporting"}
            and entry.code not in ROUTINE_BOUNDARY_REVIEW_CODES
        }
        counts.update(case_findings)
    return [
        {
            "domain": domain,
            "code": code,
            "count": count,
        }
        for (domain, code), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]


def _default_run_cover_summary(
    case_reports: Sequence[CaseReportResult],
    case_manifests: Sequence[Mapping[str, Any]],
    *,
    export_id: str,
    output: Path,
    settings: ReportingSettings,
) -> dict[str, Any]:
    case_counts = {
        "succeeded": sum(item.status != "failed_page_generated" for item in case_reports),
        "failed": sum(item.status == "failed_page_generated" for item in case_reports),
        "skipped_identical": 0,
        "cancelled": 0,
    }
    qc_counts = {
        "pass": sum(not item.manual_review_required for item in case_reports),
        "review": sum(
            item.manual_review_required and item.status != "failed_page_generated"
            for item in case_reports
        ),
        "fail": sum(item.status == "failed_page_generated" for item in case_reports),
        "not_assessed": 0,
    }
    technical_metadata = [
        manifest.get("technical_metadata", {})
        if isinstance(manifest.get("technical_metadata"), Mapping)
        else {}
        for manifest in case_manifests
    ]
    started_values = [
        metadata.get("analysis_started_at")
        for metadata in technical_metadata
        if metadata.get("analysis_started_at")
    ]
    error_counts = Counter(
        (entry.domain, entry.code)
        for item in case_reports
        for entry in item.review_entries
        if entry.domain in {"pipeline", "reporting"}
    )
    renderer = case_manifests[0].get("renderer", {})
    renderer = renderer if isinstance(renderer, Mapping) else {}
    return {
        "run_id": export_id,
        "document_state": "complete",
        "planned_case_count": len(case_reports),
        "included_case_count": len(case_reports),
        "remaining_case_count": 0,
        "included_case_ids": [item.case_id for item in case_reports],
        "case_counts": case_counts,
        "qc_counts": qc_counts,
        "manual_review_case_count": sum(
            item.manual_review_required for item in case_reports
        ),
        "mean_runtime_seconds": None,
        "runtime_case_count": 0,
        "started_at": min(started_values) if started_values else None,
        "ended_at": None,
        "run_duration_seconds": None,
        "updated_at": max(started_values) if started_values else None,
        "layout": settings.layout,
        "pipeline": {
            "name": "BodyComposition",
            "version": str(renderer.get("bodycomposition") or "Not recorded"),
            "configuration_sha256": _digest_payload(settings.normalized()),
            "source_revision": None,
        },
        "configuration_rows": run_cover_configuration_rows(settings),
        "models": [],
        "runtime": {
            "device": _joined_cover_values(
                [metadata.get("runtime_backend") for metadata in technical_metadata]
            ),
            "hardware": _joined_cover_values(
                [metadata.get("runtime_hardware") for metadata in technical_metadata]
            ),
            "cuda": None,
            "torch": None,
            "python": None,
            "strategy": None,
        },
        "technical_errors": [
            {"stage": domain, "code": code, "count": count}
            for (domain, code), count in sorted(error_counts.items())
        ],
        "quality_issues": _run_cover_quality_issues(case_reports),
        "paths": {
            "output_root": output.parent.name or ".",
            "run_folder": output.name,
            "input_directories": ["Not recorded"],
            "output_directory": output.name,
            "case_folder_pattern": "individual case report folders",
            "failed_folder_pattern": "individual failure report folders",
            "combined_pdf": COMBINED_REPORT_NAME,
        },
    }


def _normalize_run_cover_summary(
    value: Mapping[str, Any] | None,
    *,
    default: Mapping[str, Any],
    case_reports: Sequence[CaseReportResult],
) -> dict[str, Any]:
    summary = json.loads(json.dumps(dict(default), allow_nan=False))
    if value is not None:
        if not isinstance(value, Mapping):
            raise TypeError("run_summary must be a mapping.")
        unknown = sorted(set(value) - set(summary))
        if unknown:
            raise ValueError(f"Unknown run-summary fields: {unknown}.")
        for key, item in value.items():
            summary[key] = item
    case_ids = [item.case_id for item in case_reports]
    if summary["included_case_ids"] != case_ids:
        raise ValueError("Run summary case order differs from the collated report order.")
    if int(summary["included_case_count"]) != len(case_reports):
        raise ValueError("Run summary included-case count differs from the report pages.")
    planned = int(summary["planned_case_count"])
    if planned < len(case_reports):
        raise ValueError("Run summary planned-case count cannot be smaller than its snapshot.")
    summary["planned_case_count"] = planned
    summary["included_case_count"] = len(case_reports)
    summary["remaining_case_count"] = planned - len(case_reports)
    if summary["document_state"] not in {
        "in_progress",
        "complete",
        "completed_with_errors",
        "cancelled",
    }:
        raise ValueError("Run summary has an unsupported document state.")
    if summary["document_state"] == "in_progress" and not summary["remaining_case_count"]:
        raise ValueError("A complete run snapshot cannot remain marked in progress.")
    if summary["document_state"] == "in_progress" and (
        summary["ended_at"] is not None
        or summary["run_duration_seconds"] is not None
    ):
        raise ValueError("An in-progress run cannot have a final end time or duration.")
    if summary["ended_at"] is not None and not str(summary["ended_at"]).strip():
        raise ValueError("Run summary ended_at must be a non-empty timestamp or null.")
    run_duration = summary["run_duration_seconds"]
    if run_duration is not None and (
        isinstance(run_duration, bool)
        or not isinstance(run_duration, (int, float))
        or not np.isfinite(float(run_duration))
        or float(run_duration) < 0
    ):
        raise ValueError("Run summary duration must be finite and non-negative.")
    case_counts = dict(summary["case_counts"])
    if sum(int(case_counts.get(name, 0)) for name in (
        "succeeded",
        "failed",
        "skipped_identical",
        "cancelled",
    )) != len(case_reports):
        raise ValueError("Run summary execution counts do not equal the included case count.")
    qc_counts = dict(summary["qc_counts"])
    if sum(int(qc_counts.get(name, 0)) for name in (
        "pass",
        "review",
        "fail",
        "not_assessed",
    )) != len(case_reports):
        raise ValueError("Run summary QC counts do not equal the included case count.")
    if str(summary["run_id"]) != str(default["run_id"]):
        raise ValueError("Run summary run_id differs from the export ID.")
    if summary["layout"] != default["layout"]:
        raise ValueError("Run summary layout differs from the collated report layout.")
    configuration_rows = summary["configuration_rows"]
    if (
        not isinstance(configuration_rows, list)
        or len(configuration_rows) > RUN_COVER_MAX_CONFIGURATION_ROWS
    ):
        raise ValueError(
            "Run summary configuration exceeds the reader-facing cover capacity."
        )
    for item in configuration_rows:
        if (
            not isinstance(item, Mapping)
            or not str(item.get("label") or "").strip()
            or not str(item.get("value") or "").strip()
        ):
            raise ValueError("Run summary configuration rows require label and value.")
    for item in summary["models"]:
        if not isinstance(item, Mapping) or not item.get("role") or not item.get("identifier"):
            raise ValueError("Run summary models require role and identifier.")
    displayed_model_count = sum(
        str(item["role"]).strip().lower() not in RUN_COVER_HIDDEN_MODEL_ROLES
        for item in summary["models"]
    )
    if (
        len(configuration_rows) + displayed_model_count + 1
        > RUN_COVER_MAX_PIPELINE_ROWS
    ):
        raise ValueError("Run summary configuration and model rows exceed cover capacity.")
    for item in summary["technical_errors"]:
        if (
            not isinstance(item, Mapping)
            or not item.get("stage")
            or not item.get("code")
            or int(item.get("count", 0)) <= 0
        ):
            raise ValueError("Run summary technical errors require stage, code, and count.")
    quality_issues = summary["quality_issues"]
    if quality_issues != default["quality_issues"]:
        raise ValueError(
            "Run summary quality issues differ from the collated case-report findings."
        )
    seen_quality_issues: set[tuple[str, str]] = set()
    for item in quality_issues:
        if (
            not isinstance(item, Mapping)
            or item.get("domain") not in {"orientation", "vertebral", "measurement"}
            or not str(item.get("code") or "").strip()
            or int(item.get("count", 0)) <= 0
            or int(item["count"]) > len(case_reports)
        ):
            raise ValueError(
                "Run summary quality issues require domain, code, and a valid case count."
            )
        identity = (str(item["domain"]), str(item["code"]))
        if identity in seen_quality_issues:
            raise ValueError("Run summary quality issues must be unique by domain and code.")
        seen_quality_issues.add(identity)
    _assert_manifest_privacy(summary, "run_summary")
    return summary


def _publish_combined_snapshot(
    *,
    temporary_pdf: Path,
    pdf_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Publish one immutable PDF/manifest pair through an atomic pointer."""

    if pdf_path.parent != manifest_path.parent:
        raise ValueError("Combined PDF and manifest must share one output directory.")
    output = pdf_path.parent
    snapshots = output / ".snapshots"
    current = output / ".current"
    report_id = str(manifest.get("report_id") or "")
    if not re.fullmatch(r"[a-f0-9]{64}", report_id):
        raise ValueError("Combined snapshot publication requires a report SHA-256.")
    snapshots.mkdir(parents=True, exist_ok=True)
    staging = snapshots / f".{report_id}.{uuid4().hex}.partial"
    final_snapshot = snapshots / report_id
    try:
        staging.mkdir()
        staged_pdf = staging / pdf_path.name
        staged_manifest = staging / manifest_path.name
        os.replace(temporary_pdf, staged_pdf)
        _atomic_json(manifest, staged_manifest)
        validate_report_manifest(staged_manifest, pdf_path=staged_pdf)

        if final_snapshot.exists():
            existing_manifest = final_snapshot / manifest_path.name
            existing_pdf = final_snapshot / pdf_path.name
            existing = validate_report_manifest(existing_manifest, pdf_path=existing_pdf)
            if dict(existing) != dict(manifest):
                raise ReportValidationError(
                    "An immutable combined-report snapshot has conflicting content."
                )
            shutil.rmtree(staging)
        else:
            os.replace(staging, final_snapshot)

        if not os.path.lexists(current):
            _migrate_legacy_combined_pair(
                pdf_path=pdf_path,
                manifest_path=manifest_path,
                snapshots=snapshots,
                current=current,
            )
        if os.path.lexists(current):
            current_snapshot = _combined_snapshot_directory(
                current=current,
                snapshots=snapshots,
            )
            _validate_combined_snapshot(
                current_snapshot,
                pdf_name=pdf_path.name,
                manifest_name=manifest_path.name,
            )
            _ensure_combined_public_aliases(
                pdf_path=pdf_path,
                manifest_path=manifest_path,
                current_snapshot=current_snapshot,
            )

        _replace_relative_symlink(
            current,
            Path(".snapshots") / final_snapshot.name,
        )
        _ensure_combined_public_aliases(
            pdf_path=pdf_path,
            manifest_path=manifest_path,
            current_snapshot=final_snapshot,
            refresh=True,
        )
        validate_report_manifest(manifest_path, pdf_path=pdf_path)
        _prune_combined_snapshots(
            snapshots=snapshots,
            current=current,
            keep=2,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _replace_relative_symlink(path: Path, target: Path) -> None:
    """Atomically install one bounded relative symlink."""

    if target.is_absolute() or ".." in target.parts:
        raise ValueError("Combined-report symlink targets must be bounded and relative.")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial-link")
    try:
        temporary.symlink_to(target)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _combined_snapshot_directory(*, current: Path, snapshots: Path) -> Path:
    """Resolve and bound the current snapshot pointer."""

    if not current.is_symlink():
        raise ReportValidationError(
            "The combined-report snapshot pointer is not a symbolic link."
        )
    target = Path(os.readlink(current))
    if (
        target.is_absolute()
        or len(target.parts) != 2
        or target.parts[0] != snapshots.name
        or target.parts[1] in {"", ".", ".."}
    ):
        raise ReportValidationError(
            "The combined-report snapshot pointer has an invalid target."
        )
    directory = current.parent / target
    if not directory.is_dir():
        raise ReportValidationError(
            "The combined-report snapshot pointer target is unavailable."
        )
    return directory


def _validate_combined_snapshot(
    snapshot: Path,
    *,
    pdf_name: str,
    manifest_name: str,
) -> Mapping[str, Any]:
    """Validate both members of one immutable combined-report snapshot."""

    return validate_report_manifest(
        snapshot / manifest_name,
        pdf_path=snapshot / pdf_name,
    )


def _migrate_legacy_combined_pair(
    *,
    pdf_path: Path,
    manifest_path: Path,
    snapshots: Path,
    current: Path,
) -> None:
    """Place a pre-snapshot direct pair behind the pointer without changing it."""

    pdf_exists = os.path.lexists(pdf_path)
    manifest_exists = os.path.lexists(manifest_path)
    if not pdf_exists and not manifest_exists:
        return
    if pdf_exists != manifest_exists:
        raise ReportValidationError(
            "A legacy combined report is incomplete and cannot be migrated."
        )
    if pdf_path.is_symlink() or manifest_path.is_symlink():
        raise ReportValidationError(
            "Combined-report aliases exist without their snapshot pointer."
        )
    legacy_manifest = validate_report_manifest(manifest_path, pdf_path=pdf_path)
    legacy_report_id = str(legacy_manifest.get("report_id") or "")
    if not re.fullmatch(r"[a-f0-9]{64}", legacy_report_id):
        raise ReportValidationError(
            "The legacy combined report has no valid report identifier."
        )
    legacy_snapshot = snapshots / f"legacy-{legacy_report_id}"
    staging = snapshots / f".legacy-{legacy_report_id}.{uuid4().hex}.partial"
    try:
        if not legacy_snapshot.exists():
            staging.mkdir()
            shutil.copy2(pdf_path, staging / pdf_path.name)
            shutil.copy2(manifest_path, staging / manifest_path.name)
            _validate_combined_snapshot(
                staging,
                pdf_name=pdf_path.name,
                manifest_name=manifest_path.name,
            )
            os.replace(staging, legacy_snapshot)
        else:
            existing = _validate_combined_snapshot(
                legacy_snapshot,
                pdf_name=pdf_path.name,
                manifest_name=manifest_path.name,
            )
            if dict(existing) != dict(legacy_manifest):
                raise ReportValidationError(
                    "The legacy combined-report snapshot has conflicting content."
                )
        _replace_relative_symlink(
            current,
            Path(".snapshots") / legacy_snapshot.name,
        )
        _ensure_combined_public_aliases(
            pdf_path=pdf_path,
            manifest_path=manifest_path,
            current_snapshot=legacy_snapshot,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _ensure_combined_public_aliases(
    *,
    pdf_path: Path,
    manifest_path: Path,
    current_snapshot: Path,
    refresh: bool = False,
) -> None:
    """Expose stable public names and optionally refresh them for file watchers."""

    for public_path in (pdf_path, manifest_path):
        expected_target = Path(".current") / public_path.name
        snapshot_file = current_snapshot / public_path.name
        if not snapshot_file.is_file():
            raise ReportValidationError(
                "The current combined-report snapshot is incomplete."
            )
        if os.path.lexists(public_path):
            if (
                public_path.is_symlink()
                and Path(os.readlink(public_path)) == expected_target
                and not refresh
            ):
                continue
            if not public_path.is_file() or _sha256(public_path) != _sha256(
                snapshot_file
            ):
                raise ReportValidationError(
                    f"Cannot replace conflicting combined-report path {public_path.name}."
                )
        _replace_relative_symlink(public_path, expected_target)


def _prune_combined_snapshots(
    *,
    snapshots: Path,
    current: Path,
    keep: int,
) -> None:
    """Retain the active snapshot and one rollback generation."""

    if keep < 1:
        raise ValueError("At least one combined-report snapshot must be retained.")
    active = _combined_snapshot_directory(current=current, snapshots=snapshots)
    candidates = sorted(
        (
            path
            for path in snapshots.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    retained = {active}
    for candidate in candidates:
        if len(retained) >= keep:
            break
        retained.add(candidate)
    for candidate in candidates:
        if candidate not in retained:
            shutil.rmtree(candidate)


def collate_reports(
    case_reports: Sequence[CaseReportResult],
    *,
    export_id: str,
    output_directory: str | Path,
    settings: ReportingSettings | Mapping[str, Any],
    run_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collate explicit manifest order behind one always-present run cover."""

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
    case_manifests = [
        _validate_case_result_for_collation(item, settings) for item in case_reports
    ]
    flagged = [item for item in case_reports if item.manual_review_required]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    default_summary = _default_run_cover_summary(
        case_reports,
        case_manifests,
        export_id=export_id,
        output=output,
        settings=settings,
    )
    cover_summary = _normalize_run_cover_summary(
        run_summary,
        default=default_summary,
        case_reports=case_reports,
    )
    export_report_id = _digest_payload(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "export_id": export_id,
            "layout": settings.layout,
            "configuration": settings.normalized(),
            "ordered_case_reports": [item.report_id for item in case_reports],
            "run_summary": cover_summary,
            "renderer": renderer_manifest(),
        }
    )
    expected_pages = len(case_reports) + 1
    summary_rows = [
        {
            "case_id": item.case_id,
            "report_id": item.report_id,
            "combined_page_number": index + 2,
            "review_entries": [entry.as_dict() for entry in item.review_entries],
            "render_status": item.status,
        }
        for index, item in enumerate(case_reports)
        if item.manual_review_required
    ]
    pdf_path = output / COMBINED_REPORT_NAME
    manifest_path = output / CASE_MANIFEST_NAME
    temporary_pdf = output / f".{COMBINED_REPORT_NAME}.{uuid4().hex}.partial"
    temporary_cover = output / f".run-cover.{uuid4().hex}.partial.pdf"
    try:
        cover_audit = render_run_cover_page(
            temporary_cover,
            summary=cover_summary,
            page_count=expected_pages,
        )
        _validate_pdf(temporary_cover, expected_pages=1)
        writer = PdfWriter()
        writer.append(
            str(temporary_cover),
            pages=(0, 1),
            outline_item="Run summary",
        )
        for item in case_reports:
            writer.append(str(item.pdf_path), pages=(0, 1), outline_item=item.case_id)
        for page_index in range(1, expected_pages):
            overlay = PdfReader(
                BytesIO(page_number_overlay(page_index + 1, expected_pages)),
                strict=True,
            )
            writer.pages[page_index].merge_page(overlay.pages[0], over=True)
        writer.add_metadata(
            {
                "/Title": f"BodyComposition run report {export_id}",
                "/Author": "BodyComposition",
                "/Subject": "Run summary and automated research/QC case reports",
                "/Creator": "BodyComposition reporting",
                "/Producer": "BodyComposition reporting",
            }
        )
        with temporary_pdf.open("wb") as handle:
            writer.write(handle)
        _validate_pdf(temporary_pdf, expected_pages=expected_pages)
        combined_sha256 = _sha256(temporary_pdf)
        cases = [
            {
                "case_id": item.case_id,
                "report_id": item.report_id,
                "render_status": item.status,
                "manual_review_required": item.manual_review_required,
                "combined_page_number": index + 2,
                "individual_pdf_sha256": item.pdf_sha256,
            }
            for index, item in enumerate(case_reports)
        ]
        counts = Counter(
            (entry.domain, entry.code) for item in flagged for entry in item.review_entries
        )
        has_technical_errors = bool(cover_summary["technical_errors"])
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
            "render_status": (
                "succeeded_with_warnings"
                if flagged or has_technical_errors
                else "succeeded"
            ),
            "manual_review_required": bool(flagged),
            "cover_page": {
                "included": True,
                "page_number": 1,
                "summary": cover_summary,
                "audit": cover_audit,
            },
            "manual_review_summary": {
                "included": bool(flagged),
                "page_number": 1 if flagged else None,
                "placement": "cover_page",
                "case_count": len(flagged),
                "counts": [
                    {"domain": domain, "code": code, "count": count}
                    for (domain, code), count in sorted(counts.items())
                ],
                "cases": summary_rows,
            },
            "cases": cases,
            "combined_pdf_file": COMBINED_REPORT_NAME,
            "combined_pdf_sha256": combined_sha256,
            "publication": {
                "strategy": "atomic_snapshot_pointer_v1",
                "snapshot_id": export_report_id,
                "pointer": ".current",
            },
        }
        validate_report_manifest(manifest)
        _publish_combined_snapshot(
            temporary_pdf=temporary_pdf,
            pdf_path=pdf_path,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    finally:
        temporary_pdf.unlink(missing_ok=True)
        temporary_cover.unlink(missing_ok=True)
    return {"pdf_path": pdf_path, "manifest_path": manifest_path, "manifest": manifest}
