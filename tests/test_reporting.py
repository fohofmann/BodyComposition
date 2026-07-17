from dataclasses import replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pdfplumber
import jsonschema
from pypdf import PdfReader
import pytest
import SimpleITK as sitk

import BodyComposition.reporting.service as reporting_service
from BodyComposition.measurement.body_surface import body_surface_from_totalsegmentator
from BodyComposition.actions.measurement import ExportMeasurementBundle
from BodyComposition.actions.reporting import (
    CASE_REPORT_MANIFEST,
    CASE_REPORT_PDF,
    RenderCaseReport,
)
from BodyComposition.measurement.builder import build_measurement_bundle
from BodyComposition.measurement.contracts import MeasurementIdentity
from BodyComposition.reporting.contracts import (
    MAX_REVIEW_SUMMARY_ROWS,
    CaseReportInput,
    CaseReportResult,
    ReportingSettings,
    ReviewEntry,
)
from BodyComposition.reporting.metrics import aggregate_vertebral_measurements
from BodyComposition.reporting.projection import vertebral_color
from BodyComposition.reporting.projection import build_sagittal_projection
from BodyComposition.reporting.io import load_case_report_input
from BodyComposition.bin.reporting import build_parser
from BodyComposition.reporting.service import (
    ReportValidationError,
    collate_reports,
    render_case_report,
    render_failed_case_report,
    validate_report_manifest,
)
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import (
    ExecutionStatus,
    QCFlag,
    VertebralCentroid,
    VertebralResult,
)


def _report_case(config, case_id="case-001", *, orientation=None):
    shape = (64, 32, 34)
    geometry = ImageGeometry(
        size_xyz=tuple(reversed(shape)),
        spacing_xyz=(1.1, 1.2, 4.0),
        origin_lps_xyz=(-18.0, -40.0, -100.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    image = np.full(shape, -850, dtype=np.int16)
    body_label = np.zeros(shape, dtype=np.uint8)
    body_label[:, 3:-3, 3:-3] = 1
    image[:, 3:-3, 3:-3] = 20
    body_surface = body_surface_from_totalsegmentator(body_label, geometry)

    tissues = np.zeros(shape, dtype=np.uint8)
    tissues[:, 8:13, 8:14] = 1
    tissues[:, 13:16, 8:14] = 3
    tissues[:, 16:19, 9:14] = 4
    tissues[:, 19:22, 10:15] = 5
    tissues[:, 22:25, 11:16] = 8
    image[tissues == 1] = 42
    image[tissues == 3] = -100
    image[tissues == 4] = -85
    image[tissues == 5] = -70
    image[tissues == 8] = -45

    vertebral = np.zeros(shape, dtype=np.uint8)
    for label, start in ((19, 4), (17, 12), (16, 20), (15, 28), (14, 36), (13, 44), (12, 52)):
        vertebral[start : start + 5, 12:19, 14:21] = label
        image[start : start + 5, 12:19, 14:21] = 600
    result = VertebralResult(
        backend_id="synthetic_body_only",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=None,
        vertebral_body_labels=vertebral,
        label_schema=config["LBL_VERTEBRALBODIES"],
        provenance={"model": "synthetic", "local_path": "/Users/localuser/private/model"},
    )
    identity = MeasurementIdentity(case_id, "run-001", f"analysis-{case_id}")
    orientation = dict(orientation or {
        "state": "PASS_METADATA_MATCH",
        "execution_status": "succeeded",
        "qc_status": "pass",
        "manual_review_required": False,
        "orientation_changed": False,
        "review_flags": [],
    })
    pixel_digest = hashlib.sha256()
    pixel_digest.update(str(image.dtype).encode("ascii"))
    pixel_digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    pixel_digest.update(np.ascontiguousarray(image).tobytes())
    orientation["prepared_pixel_sha256"] = pixel_digest.hexdigest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        bundle = build_measurement_bundle(
            image_zyx=image,
            tissue_labels_zyx=tissues,
            geometry=geometry,
            tissue_label_schema=config["LBL_TISSUE"],
            tissue_backend_id="synthetic",
            tissue_preprocessing={"mode": "none"},
            body_surface=body_surface,
            vertebral_result=result,
            identity=identity,
            landmarks=None,
            settings=config["measurements"],
            orientation_changed=bool(orientation["orientation_changed"]),
            orientation_provenance=orientation,
        )
    # Keep this synthetic PDF fixture free of unrelated anthropometric-search
    # warnings; dedicated tests add the canonical review state explicitly.
    bundle = replace(bundle, qc_flags=())
    prepared = sitk.GetImageFromArray(image)
    prepared.SetSpacing(geometry.spacing_xyz)
    prepared.SetOrigin(geometry.origin_lps_xyz)
    prepared.SetDirection(geometry.direction_lps)
    return CaseReportInput(
        case_id=case_id,
        prepared_image=prepared,
        vertebral_result=result,
        measurement_bundle=bundle,
        orientation=orientation,
    )


def _settings(layout="spine_overview_v1"):
    return ReportingSettings.from_mapping({"enabled": True, "layout": layout})


def test_reporting_settings_and_palette_are_frozen():
    settings = _settings()
    assert settings.page_size == "A4_landscape"
    assert settings.measure_aggregation == "territory_mean"
    assert vertebral_color("T13") == vertebral_color("T13")
    assert vertebral_color("T13") != vertebral_color("L6")
    assert vertebral_color("SACRUM") != vertebral_color("L5")
    schema = json.loads(
        Path("BodyComposition/schemas/reporting_config.schema.json").read_text()
    )
    jsonschema.validate(settings.normalized(), schema)
    with pytest.raises(ValueError, match="one to six"):
        ReportingSettings.from_mapping(
            {"enabled": True, "measurement_columns": ["sm_mean_csa_cm2"] * 7}
        )
    with pytest.raises(ValueError, match="fixed ISO A4"):
        ReportingSettings.from_mapping({"enabled": True, "page_size": "A3"})
    with pytest.raises(ValueError, match="retains one individual"):
        ReportingSettings.from_mapping({"enabled": True, "individual_pdf": False})


@pytest.mark.parametrize(
    "direction,split_axis",
    [
        ((1.0, 0.0, 0.0, 0.0, 0.9659258263, -0.2588190451, 0.0, 0.2588190451, 0.9659258263), "z"),
        ((0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0), "x"),
    ],
)
def test_projection_preserves_label_centroids_under_oblique_and_permuted_geometry(direction, split_axis):
    shape = (22, 20, 18)
    geometry = ImageGeometry(
        size_xyz=tuple(reversed(shape)),
        spacing_xyz=(1.3, 1.7, 3.2),
        origin_lps_xyz=(-30.0, -40.0, -80.0),
        direction_lps=direction,
    )
    labels = np.zeros(shape, dtype=np.uint8)
    if split_axis == "z":
        labels[3:8, 7:13, 6:12] = 1
        labels[14:19, 7:13, 6:12] = 2
    else:
        labels[7:15, 7:13, 2:7] = 1
        labels[7:15, 7:13, 11:16] = 2
    centroids = []
    for native, anatomical in ((1, "T12"), (2, "L1")):
        center_zyx = np.argwhere(labels == native).mean(axis=0)
        physical = geometry.physical_point(center_zyx[::-1])
        centroids.append(
            VertebralCentroid(
                native_label=native,
                anatomical_label=anatomical,
                index_zyx=tuple(center_zyx),
                physical_lps_xyz=tuple(physical),
            )
        )
    result = VertebralResult(
        backend_id="geometry-test",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=None,
        vertebral_body_labels=labels,
        centroids=tuple(centroids),
        label_schema={1: "T12", 2: "L1"},
    )
    image = sitk.GetImageFromArray(np.where(labels, 500, -800).astype(np.int16))
    image.SetSpacing(geometry.spacing_xyz)
    image.SetOrigin(geometry.origin_lps_xyz)
    image.SetDirection(geometry.direction_lps)
    projection = build_sagittal_projection(image, result, _settings())
    assert set(np.unique(projection.label_projection)) >= {0, 1, 2}
    for centroid in centroids:
        raster_rows = np.flatnonzero(
            np.any(projection.label_projection == centroid.native_label, axis=1)
        )
        observed_superior = projection.superior_mm - (
            float(raster_rows.mean()) / (projection.label_projection.shape[0] - 1)
        ) * (projection.superior_mm - projection.inferior_mm)
        assert observed_superior == pytest.approx(
            centroid.physical_lps_xyz[2], abs=2 * _settings().projection_spacing_mm
        )


def test_whole_territory_report_values_reconstruct_canonical_bins(base_config):
    case = _report_case(base_config)
    rows = aggregate_vertebral_measurements(
        case.measurement_bundle,
        ("sm_mean_csa_cm2", "sm_mean_hu", "trunk_mean_circumference_cm"),
    )
    l3 = next(row for row in rows if row["vertebral_level"] == "L3")
    source = case.measurement_bundle.vertebrae.query("vertebral_level == 'L3'")
    weights = source["bin_integration_length_mm"].to_numpy(dtype=float)
    expected_csa = np.average(source["sm_mean_csa_cm2"], weights=weights)
    hu_weights = source["sm_mean_csa_cm2"].to_numpy(dtype=float) * weights
    expected_hu = np.average(source["sm_mean_hu"], weights=hu_weights)
    assert l3["metrics"]["sm_mean_csa_cm2"]["value"] == pytest.approx(expected_csa)
    assert l3["metrics"]["sm_mean_hu"]["value"] == pytest.approx(expected_hu)


def test_hu_aggregation_ignores_only_valid_zero_area_bins(base_config):
    case = _report_case(base_config)
    table = case.measurement_bundle.vertebrae.copy()
    target = table["vertebral_level"].eq("L3") & table["territory_bin"].eq(1)
    table.loc[target, "sm_mean_csa_cm2"] = 0.0
    table.loc[target, "sm_mean_csa_cm2_valid"] = True
    table.loc[target, "sm_mean_hu"] = np.nan
    table.loc[target, "sm_mean_hu_valid"] = False
    table.loc[target, "sm_mean_hu_reason"] = "empty_tissue"
    bundle = replace(case.measurement_bundle, vertebrae=table)
    rows = aggregate_vertebral_measurements(bundle, ("sm_mean_hu",))
    l3 = next(row for row in rows if row["vertebral_level"] == "L3")
    source = table.query("vertebral_level == 'L3' and territory_bin != 1")
    weights = (
        source["sm_mean_csa_cm2"].to_numpy(dtype=float)
        * source["bin_integration_length_mm"].to_numpy(dtype=float)
    )
    assert l3["metrics"]["sm_mean_hu"]["value"] == pytest.approx(
        np.average(source["sm_mean_hu"], weights=weights)
    )


def test_pdf_distinguishes_valid_zero_missing_and_native_variant(base_config, tmp_path):
    case = _report_case(base_config, "case-variant")
    table = case.measurement_bundle.vertebrae.copy()
    l3 = table["vertebral_level"].eq("L3")
    table.loc[l3, "sm_mean_csa_cm2"] = 0.0
    table.loc[l3, "sm_mean_csa_cm2_valid"] = True
    l1 = table["vertebral_level"].eq("L1")
    variant_native_label = int(table.loc[l1, "native_label"].iloc[0])
    table.loc[l1, "vertebral_level"] = "T13"
    table.loc[l1, "anatomical_variant"] = True
    table.loc[l1, "territory_complete"] = False
    bundle = replace(case.measurement_bundle, vertebrae=table)
    schema = dict(case.vertebral_result.label_schema)
    schema[variant_native_label] = "T13"
    vertebral_result = replace(case.vertebral_result, label_schema=schema)
    case = replace(
        case,
        measurement_bundle=bundle,
        vertebral_result=vertebral_result,
    )
    settings = ReportingSettings.from_mapping(
        {"enabled": True, "measurement_columns": ["sm_mean_csa_cm2"]}
    )
    result = render_case_report(case, tmp_path, settings)
    text = PdfReader(result.pdf_path).pages[0].extract_text()
    assert "L3 0.0" in " ".join(text.split())
    assert "NA*" in text
    assert "T13 V!" in " ".join(text.split())


@pytest.mark.parametrize("layout", ["spine_overview_v1", "spine_profile_v2"])
def test_each_layout_is_one_page_a4_embedded_and_text_extractable(base_config, tmp_path, layout):
    case = _report_case(base_config)
    result = render_case_report(
        case,
        tmp_path / layout,
        _settings(layout),
    )
    manifest = validate_report_manifest(result.manifest_path, pdf_path=result.pdf_path)
    reader = PdfReader(result.pdf_path)
    assert len(reader.pages) == 1
    assert float(reader.pages[0].mediabox.width) > float(reader.pages[0].mediabox.height)
    fonts = reader.pages[0]["/Resources"]["/Font"].get_object()
    embedded = [
        font.get_object()["/BaseFont"]
        for font in fonts.values()
        if font.get_object().get("/FontDescriptor") is not None
        and "/FontFile2" in font.get_object()["/FontDescriptor"].get_object()
    ]
    assert any("BitstreamVeraSans-Roman" in str(name) for name in embedded)
    assert any("BitstreamVeraSans-Bold" in str(name) for name in embedded)
    with pdfplumber.open(result.pdf_path) as document:
        text = document.pages[0].extract_text()
    assert "Sagittal thick-slab projection" in text
    assert "Whole-territory means from three physical bins" in text
    if layout == "spine_profile_v2":
        assert "Stacked tissue-area profile" in text
        profile = manifest["display"]["audit"]["profile"]
        assert profile["layers"] == [
            "sm", "imat", "sat", "avat", "tvat"
        ]
        positions = case.measurement_bundle.slices[
            "position_superior_mm"
        ].to_numpy(dtype=float)
        inferior_mm, superior_mm = profile["displayed_superior_range_mm"]
        displayed_valid = (
            np.isfinite(positions)
            & (positions >= inferior_mm)
            & (positions <= superior_mm)
        )
        layer_values = {}
        for layer, column in zip(
            profile["layers"], profile["source_columns"], strict=True
        ):
            values = case.measurement_bundle.slices[column].to_numpy(dtype=float)
            validity_column = column.replace("_cm2", "_valid")
            displayed_valid &= (
                np.isfinite(values)
                & (values >= 0)
                & case.measurement_bundle.slices[validity_column].to_numpy(dtype=bool)
            )
            layer_values[layer] = values
        for layer, column in zip(
            profile["layers"], profile["source_columns"], strict=True
        ):
            values = layer_values[layer]
            expected_cm3 = float(
                np.trapezoid(values[displayed_valid], positions[displayed_valid]) / 10.0
            )
            assert profile["displayed_layer_integrals_cm3"][layer] == pytest.approx(
                expected_cm3
            )
    raw = result.pdf_path.read_bytes() + result.manifest_path.read_bytes()
    assert b"/Users/localuser" not in raw
    assert b"localuser" not in raw
    assert all("/private/" not in str(value) for value in reader.metadata.values())
    unsafe = dict(manifest)
    unsafe["debug"] = "/data/clinical/patient-001/input.nii.gz"
    with pytest.raises(ReportValidationError, match="forbidden local/source path"):
        validate_report_manifest(unsafe)


def test_report_id_and_pdf_are_deterministic(base_config, tmp_path):
    case = _report_case(base_config)
    first = render_case_report(case, tmp_path / "one", _settings())
    second = render_case_report(case, tmp_path / "two", _settings())
    assert first.report_id == second.report_id
    assert first.pdf_sha256 == second.pdf_sha256
    changed = render_case_report(case, tmp_path / "three", _settings("spine_profile_v2"))
    assert changed.report_id != first.report_id


def test_report_rejects_mixed_preparation_vertebral_measurement_sources(base_config, tmp_path):
    case = _report_case(base_config, "case-mixed")

    wrong_orientation = dict(case.orientation)
    wrong_orientation["prepared_pixel_sha256"] = "0" * 64
    with pytest.raises(ReportValidationError, match="CT pixels disagree"):
        render_case_report(
            replace(case, orientation=wrong_orientation),
            tmp_path / "pixels",
            _settings(),
        )

    slices = case.measurement_bundle.slices.copy()
    slices.loc[slices.index[3], "position_superior_mm"] += 0.5
    geometry_mismatch = replace(case.measurement_bundle, slices=slices)
    with pytest.raises(ReportValidationError, match="physical coordinates"):
        render_case_report(
            replace(case, measurement_bundle=geometry_mismatch),
            tmp_path / "geometry",
            _settings(),
        )

    vertebrae = case.measurement_bundle.vertebrae.copy()
    vertebrae.loc[vertebrae["vertebral_level"].eq("L3"), "native_label"] = 99
    label_mismatch = replace(case.measurement_bundle, vertebrae=vertebrae)
    with pytest.raises(ReportValidationError, match="different native labels"):
        render_case_report(
            replace(case, measurement_bundle=label_mismatch),
            tmp_path / "labels",
            _settings(),
        )


def test_interrupted_rerender_removes_stale_pdf_completion_marker(
    base_config,
    tmp_path,
    monkeypatch,
):
    case = _report_case(base_config, "case-interrupted")
    output = tmp_path / "report"
    first = render_case_report(case, output, _settings())
    assert first.pdf_path.is_file()

    def fail_manifest_write(*args, **kwargs):
        raise OSError("simulated manifest publication failure")

    monkeypatch.setattr(reporting_service, "_atomic_json", fail_manifest_write)
    with pytest.raises(OSError, match="simulated manifest"):
        render_case_report(case, output, _settings())
    assert not first.pdf_path.exists()


def test_orientation_repair_is_annotated_and_summary_is_conditional(base_config, tmp_path):
    ordinary = render_case_report(
        _report_case(base_config, "case-ordinary"),
        tmp_path / "ordinary",
        _settings(),
    )
    orientation = {
        "state": "MISMATCH_REPAIRED",
        "execution_status": "succeeded",
        "qc_status": "review",
        "manual_review_required": True,
        "orientation_changed": True,
        "prepared_pixel_sha256": "a" * 64,
        "review_flags": [
            {
                "code": "severe_misorientation_repaired",
                "reason": "Metadata and anatomy differed; orientation was repaired.",
            }
        ],
    }
    repaired = render_case_report(
        _report_case(base_config, "case-repaired", orientation=orientation),
        tmp_path / "repaired",
        _settings(),
    )
    clean_export = collate_reports(
        [ordinary],
        export_id="export-clean",
        output_directory=tmp_path / "clean-export",
        settings=_settings(),
    )
    assert len(PdfReader(clean_export["pdf_path"]).pages) == 1
    reviewed_export = collate_reports(
        [ordinary, repaired],
        export_id="export-review",
        output_directory=tmp_path / "review-export",
        settings=_settings(),
    )
    reader = PdfReader(reviewed_export["pdf_path"])
    assert len(reader.pages) == 3
    assert [str(item.title) for item in reader.outline] == [
        "Manual review summary", "case-ordinary", "case-repaired"
    ]
    manifest = json.loads(reviewed_export["manifest_path"].read_text())
    assert manifest["manual_review_summary"]["included"]
    assert manifest["manual_review_summary"]["cases"][0]["case_id"] == "case-repaired"
    assert manifest["cases"][1]["combined_page_number"] == 3
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Lossless orientation transform applied" in text
    assert "input CT automatically reoriented before analysis" in text
    assert "not_adjudicated" in text


def test_collation_rejects_result_manifest_or_configuration_mismatch(
    base_config,
    tmp_path,
):
    source = render_case_report(
        _report_case(base_config, "case-source"),
        tmp_path / "source",
        _settings(),
    )
    with pytest.raises(ReportValidationError, match="case_id"):
        collate_reports(
            [replace(source, case_id="case-forged")],
            export_id="export-forged",
            output_directory=tmp_path / "forged",
            settings=_settings(),
        )
    changed_settings = ReportingSettings.from_mapping(
        {"enabled": True, "numeric_precision": 2}
    )
    with pytest.raises(ReportValidationError, match="configuration"):
        collate_reports(
            [source],
            export_id="export-config-mismatch",
            output_directory=tmp_path / "config-mismatch",
            settings=changed_settings,
        )


def test_uncertain_unchanged_orientation_has_distinct_action(base_config, tmp_path):
    orientation = {
        "state": "MISMATCH_UNCERTAIN",
        "execution_status": "succeeded",
        "qc_status": "review",
        "manual_review_required": True,
        "orientation_changed": False,
        "prepared_pixel_sha256": "b" * 64,
        "review_flags": [
            {
                "code": "orientation_mismatch_uncertain",
                "reason": "The candidate repair did not satisfy all safety gates.",
            }
        ],
    }
    result = render_case_report(
        _report_case(base_config, "case-uncertain", orientation=orientation),
        tmp_path,
        _settings(),
    )
    assert result.review_entries[0].automatic_action == "Continued without orientation repair."
    text = PdfReader(result.pdf_path).pages[0].extract_text()
    assert "processing continued without orientation repair" in text
    assert "automatically reoriented" not in text


def test_downstream_copies_of_orientation_qc_are_not_duplicated(
    base_config,
    tmp_path,
):
    orientation = {
        "state": "MISMATCH_REPAIRED",
        "execution_status": "succeeded",
        "qc_status": "review",
        "manual_review_required": True,
        "orientation_changed": True,
        "review_flags": [
            {
                "code": "severe_misorientation_repaired",
                "reason": "Canonical orientation review finding.",
            }
        ],
    }
    case = _report_case(base_config, "case-orientation-copy", orientation=orientation)
    copied_flag = QCFlag(
        code="orientation_changed",
        stage="orientation",
        reason="Downstream copy of the orientation state.",
    )
    case = replace(
        case,
        vertebral_result=replace(case.vertebral_result, qc_flags=(copied_flag,)),
        measurement_bundle=replace(case.measurement_bundle, qc_flags=(copied_flag,)),
    )
    result = render_case_report(case, tmp_path, _settings())
    assert [(entry.domain, entry.code) for entry in result.review_entries] == [
        ("orientation", "severe_misorientation_repaired")
    ]


def test_summary_capacity_overflow_fails_before_output(base_config, tmp_path):
    source = render_case_report(
        _report_case(base_config), tmp_path / "source", _settings()
    )
    reports = []
    for index in range(MAX_REVIEW_SUMMARY_ROWS + 1):
        case_id = f"case-{index:02d}"
        entry = ReviewEntry(
            case_id=case_id,
            domain="measurement",
            code="review_required",
            reason="Canonical review flag",
            automatic_action="Processing completed; inspect canonical QC.",
        )
        reports.append(
            CaseReportResult(
                case_id=case_id,
                report_id=hashlib.sha256(case_id.encode()).hexdigest(),
                layout=source.layout,
                status="succeeded_with_warnings",
                pdf_path=source.pdf_path,
                manifest_path=source.manifest_path,
                pdf_sha256=source.pdf_sha256,
                manual_review_required=True,
                review_entries=(entry,),
            )
        )
    destination = tmp_path / "overflow"
    with pytest.raises(ValueError, match="exceeds 24"):
        collate_reports(
            reports,
            export_id="export-overflow",
            output_directory=destination,
            settings=_settings(),
        )
    assert not (destination / "case_reports.pdf").exists()


def test_failure_page_is_explicit_and_one_page(tmp_path):
    result = render_failed_case_report(
        case_id="case-failed",
        analysis_id="analysis-unavailable",
        output_directory=tmp_path,
        settings=_settings(),
        failure_stage="SegmSpinepsVeridah",
        failure_code="RuntimeError",
    )
    assert result.status == "failed_page_generated"
    reader = PdfReader(result.pdf_path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert "ANALYSIS INCOMPLETE - FAILURE PAGE" in text
    assert "not evidence that the scientific analysis succeeded" in text


@pytest.mark.parametrize("layout", ["spine_overview_v1", "spine_profile_v2"])
def test_failure_page_preserves_available_vertebral_body_overview(
    base_config,
    tmp_path,
    layout,
):
    case = _report_case(base_config, f"case-partial-{layout[-2:]}")
    centroids = []
    labels = case.vertebral_result.vertebral_body_labels
    geometry = case.vertebral_result.geometry
    for native_label in sorted(int(value) for value in np.unique(labels) if value):
        center_zyx = np.argwhere(labels == native_label).mean(axis=0)
        centroids.append(
            VertebralCentroid(
                native_label=native_label,
                anatomical_label=case.vertebral_result.label_schema[native_label],
                index_zyx=tuple(float(value) for value in center_zyx),
                physical_lps_xyz=tuple(
                    float(value)
                    for value in geometry.physical_point(center_zyx[::-1])
                ),
            )
        )
    result_with_centroids = replace(
        case.vertebral_result,
        centroids=tuple(centroids),
    )
    result = render_failed_case_report(
        case_id=case.case_id,
        analysis_id=case.measurement_bundle.identity.analysis_id,
        output_directory=tmp_path / layout,
        settings=_settings(layout),
        failure_stage="BuildMeasurements",
        failure_code="measurement_contract_error",
        prepared_image=case.prepared_image,
        vertebral_result=result_with_centroids,
    )

    reader = PdfReader(result.pdf_path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert "ANALYSIS INCOMPLETE - FAILURE PAGE" in text
    assert "AVAILABLE SPINE OVERVIEW" in text
    assert "Sagittal thick-slab projection" in text
    assert "Measurements unavailable" in text
    assert "SM (cm2): NA*" in text
    if layout == "spine_profile_v2":
        assert "Stacked tissue-area profile" in text
        assert "measurement stage output incomplete" in text

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["partial_spine_overview"] == "rendered"
    assert manifest["display"]["audit"]["partial_spine_overview"] is True
    assert {item["name"] for item in manifest["source_artifacts"]} == {
        "prepared_ct_pixels_and_geometry",
        "vertebral_body_labels",
        "vertebral_display_metadata",
    }
    assert "partial_spine_overview_only" in result.warnings
    if layout == "spine_overview_v1":
        changed_schema = dict(result_with_centroids.label_schema)
        changed_schema[centroids[0].native_label] = "L6"
        changed = render_failed_case_report(
            case_id=case.case_id,
            analysis_id=case.measurement_bundle.identity.analysis_id,
            output_directory=tmp_path / "changed-display-metadata",
            settings=_settings(layout),
            failure_stage="BuildMeasurements",
            failure_code="measurement_contract_error",
            prepared_image=case.prepared_image,
            vertebral_result=replace(
                result_with_centroids,
                label_schema=changed_schema,
            ),
        )
        assert changed.report_id != result.report_id


def test_posthoc_manifest_loader_and_cli_surface_use_canonical_bundle(base_config, tmp_path):
    case = _report_case(base_config, "case-posthoc")
    workspace = tmp_path / "analysis"
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=1, device="cpu")
    )
    action(
        {
            "id": case.case_id,
            "workspace": workspace,
            "tmp/measurement_bundle": case.measurement_bundle,
        }
    )
    prepared_path = workspace / "prepared.nii.gz"
    body_path = workspace / "vertebral_bodies.nii.gz"
    sitk.WriteImage(case.prepared_image, str(prepared_path))
    body_image = sitk.GetImageFromArray(case.vertebral_result.vertebral_body_labels)
    body_image.CopyInformation(case.prepared_image)
    sitk.WriteImage(body_image, str(body_path))
    vertebral_path = workspace / "vertebral_result.json"
    vertebral_path.write_text(
        json.dumps(case.vertebral_result.summary(), indent=2), encoding="utf-8"
    )
    orientation_path = workspace / "orientation_report.json"
    orientation_path.write_text(json.dumps(case.orientation), encoding="utf-8")
    manifest_path = workspace / "case_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "case_id": case.case_id,
                "prepared_ct": prepared_path.name,
                "vertebral_body_mask": body_path.name,
                "vertebral_result_json": vertebral_path.name,
                "measurement_directory": f"tables/{case.case_id}",
                "orientation_report_json": orientation_path.name,
                "measurement_qc_json": f"qc/{case.case_id}_measurement-qc.json",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_case_report_input(manifest_path)
    assert loaded.case_id == case.case_id
    assert loaded.measurement_bundle.identity.analysis_id == case.measurement_bundle.identity.analysis_id
    result = render_case_report(loaded, tmp_path / "posthoc-report", _settings())
    assert len(PdfReader(result.pdf_path).pages) == 1
    assert build_parser().parse_args(["validate", "--manifest", str(result.manifest_path)]).command == "validate"


def test_pipeline_reporting_action_uses_same_service_and_declares_completion_markers(base_config, tmp_path):
    config = deepcopy(base_config)
    config["reporting"]["enabled"] = True
    action = RenderCaseReport(
        SimpleNamespace(config=config, timestamp=1, device="cpu")
    )
    case = _report_case(config, "case-action")
    memory = {
        "id": case.case_id,
        "workspace": tmp_path,
        "tmp/prepared_image": SimpleNamespace(prepared_image=case.prepared_image),
        "tmp/orientation_result": case.orientation,
        "tmp/vertebral_result": case.vertebral_result,
        "tmp/measurement_bundle": case.measurement_bundle,
    }
    action(memory)
    action.validate_outputs(memory)
    assert action.io_persisted_outputs == [CASE_REPORT_PDF, CASE_REPORT_MANIFEST]
    assert Path(memory[CASE_REPORT_PDF]).is_file()
    assert Path(memory[CASE_REPORT_MANIFEST]).is_file()
