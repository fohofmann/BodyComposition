import hashlib
import json
import warnings
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pandas as pd
import pdfplumber
import pytest
import SimpleITK as sitk
from pypdf import PdfReader

import BodyComposition.reporting.service as reporting_service
from BodyComposition.actions.measurement import ExportMeasurementBundle
from BodyComposition.actions.reporting import (
    CASE_REPORT_MANIFEST,
    CASE_REPORT_PDF,
    TISSUE_LABEL_MASK,
    RenderCaseReport,
)
from BodyComposition.cli import build_parser
from BodyComposition.measurement.body_surface import body_surface_from_totalsegmentator
from BodyComposition.measurement.builder import build_measurement_bundle
from BodyComposition.measurement.contracts import MeasurementIdentity
from BodyComposition.reporting.contracts import (
    MAX_REVIEW_SUMMARY_ROWS,
    MEASUREMENT_DEFINITIONS,
    CaseReportInput,
    CaseReportResult,
    ReportingSettings,
    ReviewEntry,
)
from BodyComposition.reporting.io import load_case_report_input
from BodyComposition.reporting.metrics import aggregate_vertebral_measurements
from BodyComposition.reporting.projection import (
    SAGITTAL_AP_CROP_MARGIN_MM,
    SAGITTAL_SI_CROP_MARGIN_MM,
    build_axial_segmentation_view,
    build_sagittal_projection,
    vertebral_color,
)
from BodyComposition.reporting.render import (
    ImageBox,
    _draw_table,
    _new_canvas,
    _separate_label_y_positions,
    _summary_value,
)
from BodyComposition.reporting.service import (
    ReportValidationError,
    collate_reports,
    load_case_report_result,
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
    image[tissues == 1] = 42
    image[tissues == 3] = -100
    image[tissues == 4] = -85
    image[tissues == 5] = -70
    image[0, 8, 8] = -100
    tissue_labels = tissues.copy()
    tissue_labels[0, 8, 8] = 0

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
        provenance={
            "model": "synthetic",
            "local_path": str(Path("/") / "Users" / "localuser" / "private" / "model"),
        },
    )
    identity = MeasurementIdentity(case_id, "run-001", f"analysis-{case_id}")
    orientation = dict(
        orientation
        or {
            "state": "PASS_METADATA_MATCH",
            "execution_status": "succeeded",
            "qc_status": "pass",
            "manual_review_required": False,
            "orientation_changed": False,
            "review_flags": [],
        }
    )
    orientation.setdefault("model", {"asset_id": "ctdeeprot_2d_v1"})
    pixel_digest = hashlib.sha256()
    pixel_digest.update(str(image.dtype).encode("ascii"))
    pixel_digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    pixel_digest.update(np.ascontiguousarray(image).tobytes())
    orientation["prepared_pixel_sha256"] = pixel_digest.hexdigest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        bundle = build_measurement_bundle(
            image_zyx=image,
            tissue_labels_zyx=tissue_labels,
            geometry=geometry,
            tissue_label_schema=config["LBL_TISSUE"],
            tissue_backend_id="synthetic",
            tissue_preprocessing={"mode": "none"},
            compartment_labels_zyx=tissues,
            compartment_label_schema=config["LBL_TISSUE_COMPARTMENTS"],
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
        tissue_labels_zyx=tissue_labels,
        orientation=orientation,
        technical_metadata={
            "analysis_started_at": "2026-07-18T12:30:00+02:00",
            "data_attribution": "Synthetic test data; no patient information.",
            "input_format": "dicom",
            "pipeline_version": "1.0.0rc1",
            "runtime_backend": "cuda",
            "runtime_hardware": "NVIDIA A100-SXM4-80GB",
            "scanner_manufacturer": "Research Test Imaging",
            "scanner_model": "Synthetic CT 1.0",
        },
    )


def _settings(layout="spine_overview_v1"):
    return ReportingSettings.from_mapping({"enabled": True, "layout": layout})


def test_reporting_settings_and_palette_are_frozen():
    settings = _settings()
    assert settings.page_size == "A4_landscape"
    assert settings.measure_aggregation == "territory_mean"
    assert settings.measurement_columns == (
        "skeletal_muscle_tissue_hu_m29_150_mean_csa_cm2",
        "vat_total_hu_m190_m30_mean_csa_cm2",
        "sat_total_hu_m190_m30_mean_csa_cm2",
        "trunk_mean_circumference_cm",
    )
    assert vertebral_color("T13") == vertebral_color("T13")
    assert vertebral_color("T13") != vertebral_color("L6")
    assert vertebral_color("SACRUM") != vertebral_color("L5")
    schema = json.loads(Path("BodyComposition/schemas/reporting_config.schema.json").read_text())
    jsonschema.validate(settings.normalized(), schema)


def test_projection_label_collision_handling_preserves_order_and_spacing():
    desired = [48.0, 45.0, 44.0, 43.0]
    placed = _separate_label_y_positions(desired, lower=10.0, upper=55.0)

    assert placed == sorted(placed, reverse=True)
    assert np.min(np.abs(np.diff(placed))) >= 11.0
    assert min(placed) >= 10.0
    assert max(placed) <= 55.0
    with pytest.raises(ValueError, match="one to six"):
        ReportingSettings.from_mapping(
            {"enabled": True, "measurement_columns": ["sm_mean_csa_cm2"] * 7}
        )
    with pytest.raises(ValueError, match="fixed ISO A4"):
        ReportingSettings.from_mapping({"enabled": True, "page_size": "A3"})
    with pytest.raises(ValueError, match="retains one individual"):
        ReportingSettings.from_mapping({"enabled": True, "individual_pdf": False})


def test_full_spine_table_fits_released_one_page_geometry(tmp_path):
    levels = [
        *(f"C{index}" for index in range(1, 8)),
        *(f"T{index}" for index in range(1, 14)),
        *(f"L{index}" for index in range(1, 7)),
        "SACRUM",
    ]
    settings = _settings("spine_profile_v2")
    rows = [
        {
            "vertebral_level": level,
            "native_label": native_label,
            "anatomical_variant": level in {"T13", "L6"},
            "territory_complete": True,
            "metrics": {
                column: {"valid": True, "value": float(native_label)}
                for column in settings.measurement_columns
            },
        }
        for native_label, level in enumerate(levels, start=1)
    ]
    projection = SimpleNamespace(native_labels=tuple(range(1, len(levels) + 1)))
    destination = tmp_path / "full-spine-table.pdf"
    pdf = _new_canvas(destination, case_id="full-spine-capacity")
    audit = _draw_table(
        pdf,
        rows,
        settings,
        projection,
        ImageBox(left=24.0, bottom=150.0, width=168.0, height=365.27559055118115),
    )
    pdf.showPage()
    pdf.save()

    assert len(levels) == 27
    assert audit["row_count"] == 27
    assert audit["row_capacity"] >= 27
    assert audit["row_height_pt"] >= 11.0
    assert audit["minimum_row_height_pt"] == 9.5
    assert len(PdfReader(destination).pages) == 1
    crowded_labels = _separate_label_y_positions(
        [140.0 + index * 0.1 for index in range(27)],
        lower=10.0,
        upper=284.0,
    )
    assert np.min(np.diff(sorted(crowded_labels))) >= 10.0


def test_anthropometry_display_distinguishes_missing_from_observed_ineligible():
    settings = _settings()
    observed_ineligible = pd.Series(
        {
            "ct_max_pelvic_circumference_cm": 101.2,
            "ct_max_pelvic_circumference_valid": True,
            "ct_max_pelvic_circumference_eligible": False,
        }
    )
    missing = pd.Series(
        {
            "ct_max_pelvic_circumference_cm": np.nan,
            "ct_max_pelvic_circumference_valid": False,
            "ct_max_pelvic_circumference_eligible": False,
        }
    )

    assert (
        _summary_value(
            observed_ineligible,
            "ct_max_pelvic_circumference_cm",
            settings,
            unit="cm",
        )
        == "101.2 cm*"
    )
    assert (
        _summary_value(
            missing,
            "ct_max_pelvic_circumference_cm",
            settings,
            unit="cm",
        )
        == "NA*"
    )


@pytest.mark.parametrize(
    "direction,split_axis",
    [
        ((1.0, 0.0, 0.0, 0.0, 0.9659258263, -0.2588190451, 0.0, 0.2588190451, 0.9659258263), "z"),
        ((0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0), "x"),
    ],
)
def test_projection_preserves_label_centroids_under_oblique_and_permuted_geometry(
    direction, split_axis
):
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


def test_projection_crops_display_plane_to_vertebral_body_bounds(base_config):
    case = _report_case(base_config)
    rows = aggregate_vertebral_measurements(
        case.measurement_bundle,
        _settings().measurement_columns,
    )
    projection = build_sagittal_projection(
        case.prepared_image,
        case.vertebral_result,
        _settings(),
        centroids_lps_xyz=[row["centroid_lps_xyz"] for row in rows],
    )

    # SimpleITK metadata and indices are xyz; the stored label array is zyx.
    geometry = case.vertebral_result.geometry
    labels_zyx = case.vertebral_result.vertebral_body_labels
    occupied_zyx = np.argwhere(labels_zyx != 0)
    lower_zyx = occupied_zyx.min(axis=0).astype(float) - 0.5
    upper_zyx = occupied_zyx.max(axis=0).astype(float) + 0.5
    ct_anterior = geometry.origin_lps_xyz[1] - 0.5 * geometry.spacing_xyz[1]
    ct_posterior = (
        geometry.origin_lps_xyz[1] + (geometry.size_xyz[1] - 0.5) * geometry.spacing_xyz[1]
    )
    ct_inferior = geometry.origin_lps_xyz[2] - 0.5 * geometry.spacing_xyz[2]
    ct_superior = (
        geometry.origin_lps_xyz[2] + (geometry.size_xyz[2] - 0.5) * geometry.spacing_xyz[2]
    )
    mask_anterior = geometry.origin_lps_xyz[1] + lower_zyx[1] * geometry.spacing_xyz[1]
    mask_posterior = geometry.origin_lps_xyz[1] + upper_zyx[1] * geometry.spacing_xyz[1]
    mask_inferior = geometry.origin_lps_xyz[2] + lower_zyx[0] * geometry.spacing_xyz[2]
    mask_superior = geometry.origin_lps_xyz[2] + upper_zyx[0] * geometry.spacing_xyz[2]

    assert projection.anterior_mm == pytest.approx(
        max(ct_anterior, mask_anterior - SAGITTAL_AP_CROP_MARGIN_MM)
    )
    assert projection.posterior_mm == pytest.approx(
        min(ct_posterior, mask_posterior + SAGITTAL_AP_CROP_MARGIN_MM)
    )
    assert projection.inferior_mm == pytest.approx(
        max(ct_inferior, mask_inferior - SAGITTAL_SI_CROP_MARGIN_MM)
    )
    assert projection.superior_mm == pytest.approx(
        min(ct_superior, mask_superior + SAGITTAL_SI_CROP_MARGIN_MM)
    )
    assert projection.posterior_mm - projection.anterior_mm < ct_posterior - ct_anterior
    assert projection.crop_basis == "vertebral_body_mask"
    assert projection.method == "sagittal_thick_slab_vertebral_crop_v2"


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


def test_whole_territory_report_retains_observed_mean_for_incomplete_level(base_config):
    case = _report_case(base_config)
    table = case.measurement_bundle.vertebrae.copy()
    l3_rows = table["vertebral_level"].eq("L3")
    table.loc[l3_rows, "territory_complete"] = False
    table.loc[l3_rows, "territory_reason"] = "truncated_vertebra"
    table.loc[l3_rows, "territory_qc_status"] = "review"
    second_bin = l3_rows & table["territory_bin"].eq(2)
    third_bin = l3_rows & table["territory_bin"].eq(3)
    table.loc[second_bin, "sm_mean_csa_cm2_coverage_fraction"] = 0.5
    table.loc[third_bin, "sm_mean_csa_cm2"] = np.nan
    table.loc[third_bin, "sm_mean_csa_cm2_valid"] = False
    table.loc[third_bin, "sm_mean_csa_cm2_reason"] = "outside_fov"
    table.loc[third_bin, "sm_mean_csa_cm2_coverage_fraction"] = 0.0
    table.loc[third_bin, "bin_valid"] = False
    table.loc[third_bin, "bin_missing_reason"] = "outside_fov"
    bundle = replace(case.measurement_bundle, vertebrae=table)

    rows = aggregate_vertebral_measurements(
        bundle,
        ("sm_mean_csa_cm2", "sm_mean_hu"),
    )
    l3 = next(row for row in rows if row["vertebral_level"] == "L3")
    source = table.loc[l3_rows]
    valid = source["sm_mean_csa_cm2_valid"] & source["bin_valid"]
    effective_lengths = (
        source["bin_integration_length_mm"]
        * source["sm_mean_csa_cm2_coverage_fraction"]
    )
    expected = np.average(
        source.loc[valid, "sm_mean_csa_cm2"],
        weights=effective_lengths.loc[valid],
    )

    assert not l3["territory_complete"]
    assert l3["territory_reason"] == "truncated_vertebra"
    assert l3["metrics"]["sm_mean_csa_cm2"]["valid"]
    assert l3["metrics"]["sm_mean_csa_cm2"]["value"] == pytest.approx(expected)
    assert l3["source_audit"]["sm_mean_csa_cm2"][
        "effective_integration_length_mm"
    ] == pytest.approx(
        effective_lengths.to_numpy(dtype=float)
    )
    hu_valid = (
        valid
        & source["sm_mean_hu_valid"]
        & source["sm_mean_csa_cm2"].gt(0)
    )
    hu_weights = (
        source["sm_mean_csa_cm2"] * effective_lengths
    )
    assert l3["metrics"]["sm_mean_hu"]["valid"]
    assert l3["metrics"]["sm_mean_hu"]["value"] == pytest.approx(
        np.average(
            source.loc[hu_valid, "sm_mean_hu"],
            weights=hu_weights.loc[hu_valid],
        )
    )


def test_axial_segmentation_view_uses_l3_and_explicit_xyz_zyx_boundary(base_config):
    case = _report_case(base_config)
    rows = aggregate_vertebral_measurements(
        case.measurement_bundle,
        _settings().measurement_columns,
    )
    view = build_axial_segmentation_view(
        case.prepared_image,
        case.tissue_labels_zyx,
        case.vertebral_result,
        rows,
    )

    assert view.vertebral_level == "L3"
    assert view.native_label == 15
    assert view.l3_available
    assert view.rgb.ndim == 3 and view.rgb.shape[2] == 3
    assert {"SM", "SAT", "aVAT", "tVAT"}.issubset(view.tissue_names)

    without_l3 = [row for row in rows if row["vertebral_level"] != "L3"]
    fallback = build_axial_segmentation_view(
        case.prepared_image,
        case.tissue_labels_zyx,
        case.vertebral_result,
        without_l3,
    )
    assert fallback.vertebral_level == "L2"
    assert not fallback.l3_available


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
    weights = source["sm_mean_csa_cm2"].to_numpy(dtype=float) * source[
        "bin_integration_length_mm"
    ].to_numpy(dtype=float)
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
    loaded = load_case_report_result(result.manifest_path)
    assert loaded.report_id == result.report_id
    assert loaded.pdf_sha256 == result.pdf_sha256
    reader = PdfReader(result.pdf_path)
    assert len(reader.pages) == 1
    assert reader.metadata.creator == "BodyComposition reporting"
    assert reader.metadata.producer == "BodyComposition reporting"
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
        page = document.pages[0]
        text = page.extract_text()
        words = page.extract_words()
        font_sizes = {round(float(char["size"]), 1) for char in page.chars}
        page_height = float(page.height)
    assert "Sagittal thick-slab projection" in text
    assert "Vertebral summary" in text
    assert "Whole-territory mean | three physical bins" in text
    expected_axial_title = (
        "Axial tissue segmentation | L3" if layout == "spine_overview_v1" else "Axial overlay | L3"
    )
    assert expected_axial_title in text
    assert "Key anthropometry" in text
    assert "Technical metadata" not in text
    assert "Notes" in text
    assert "Notes / QC" not in text
    assert "Review required" not in text
    assert "Source: Synthetic test data; no patient information." in text
    assert "2026-07-18 10:30 UTC" in text
    assert "Research Test Imaging Synthetic CT 1.0" in text
    assert "NVIDIA A100-SXM4-80GB" in text
    assert "DICOM 1.10 x 1.20 x 4.00 mm" in text
    assert "BodyComposition 1.0.0rc1" in text
    assert "ctdeeprot_2d_v1 | synthetic_body_only | synthetic" in text
    assert "CASE case-001 | analysis analysis-cas" in " ".join(text.split())
    assert "case page 1 of 1 | report" in " ".join(text.split())
    assert "Body composition analysis" not in text
    assert "Automated CT segmentation and quantitative QC summary" not in text
    assert "Automated research/QC report" not in text
    assert "SM HU" not in text
    assert "Muscle" not in text
    normalized_text = " ".join(text.split())
    assert "Min waist NA*" in normalized_text
    assert "Pelvic max 12.3 cm" in normalized_text
    assert "Waist / pelvic NA*" in normalized_text
    word_set = {word["text"] for word in words}
    assert {"SM", "SAT", "aVAT", "tVAT"}.issubset(word_set)
    assert not {"aV", "tV", "Muscle"}.intersection(word_set)
    assert not {
        "ANALYSIS",
        "INPUT",
        "PIPELINE",
        "RUNTIME",
        "SCANNER",
        "SLICE",
        "MODELS",
    }.intersection(word_set)
    audit = manifest["display"]["audit"]
    assert audit["layout_revision"] == "scientific_onepager_v16"
    assert audit["design"] == {
        "card_gap_pt": 8.0,
        "content_padding_pt": 10.0,
        "title_baseline_from_top_pt": 18.0,
        "subtitle_baseline_from_top_pt": 31.0,
        "main_card_top_pt": audit["design"]["main_card_top_pt"],
        "main_panel_style": "open_editorial_grid",
        "card_corner_radius_pt": 0.0,
        "rule_width_pt": 0.6,
    }
    assert audit["header"]["boxed"] is True
    assert audit["header"]["box_height_pt"] == 52
    assert audit["header"]["content_padding_pt"] == 10
    assert audit["header"]["technical_metadata_location"] == "header"
    assert audit["header"]["technical_metadata_rows"] == 2
    assert audit["header"]["technical_metadata_icons"] == [
        "analysis",
        "input",
        "pipeline",
        "runtime",
        "scanner",
        "slice",
        "models",
    ]
    assert audit["header"]["technical_metadata_icon_style"] == "monochrome_vector"
    assert audit["header"]["technical_metadata_truncated"] is False
    assert audit["notes_qc"]["box_width_pt"] > 790
    assert audit["notes_qc"]["box_height_pt"] == 100
    assert audit["notes_qc"]["maximum_columns"] == 3
    assert audit["notes_qc"]["content_padding_pt"] == 10
    assert audit["notes_qc"]["block_split"] is False
    assert audit["notes_qc"]["structured"] is True
    assert audit["notes_qc"]["status_visible"] is False
    assert audit["notes_qc"]["review_count"] == 0
    assert audit["notes_qc"]["displayed_review_count"] == 0
    assert audit["notes_qc"]["suppressed_review_codes"] == []
    assert audit["notes_qc"]["section_headings"] == ["Data and provenance"]
    assert not audit["notes_qc"]["truncated"]
    assert audit["anthropometry"]["separate_card"] is True
    assert audit["anthropometry"]["box_height_pt"] == 82
    assert audit["anthropometry"]["box_width_pt"] == (
        205 if layout == "spine_overview_v1" else 145
    )
    assert audit["anthropometry"]["content_padding_pt"] == 10
    assert audit["anthropometry"]["title_baseline_from_top_pt"] == 18
    assert audit["anthropometry"]["labels"] == [
        "Min waist",
        "Pelvic max",
        "Waist / pelvic",
    ]
    assert font_sizes == {8.0}
    metadata_word = next(word for word in words if word["text"] == "2026-07-18")
    assert float(metadata_word["top"]) < 50
    title_words = {
        "header": next(word for word in words if word["text"] == "CASE"),
        "main": next(word for word in words if word["text"] == "Spine"),
        "notes": next(word for word in words if word["text"] == "Notes"),
        "anthropometry": next(word for word in words if word["text"] == "Key"),
    }
    box_tops = {
        "header": audit["header"]["box_top_pt"],
        "main": audit["design"]["main_card_top_pt"],
        "notes": audit["notes_qc"]["box_top_pt"],
        "anthropometry": audit["anthropometry"]["box_top_pt"],
    }
    relative_title_tops = [
        float(title_words[name]["top"]) - (page_height - float(box_tops[name]))
        for name in title_words
    ]
    assert np.ptp(relative_title_tops) < 0.2
    assert not {"PASS", "REVIEW"}.intersection(word["text"] for word in words)
    assert audit["sagittal_projection"]["crop_basis"] == "vertebral_body_mask"
    assert audit["sagittal_projection"]["method"] == "sagittal_thick_slab_vertebral_crop_v2"
    assert audit["table"]["row_layout"] == "uniform_categorical_grid"
    assert audit["table"]["display_labels"] == ["SM", "VAT", "SAT", "Trunk"]
    assert "sm_mean_hu" not in audit["table"]["measurement_columns"]
    assert manifest["display"]["audit"]["axial_view"]["vertebral_level"] == "L3"
    assert manifest["display"]["audit"]["axial_view"]["tissue_names"] == [
        "SM",
        "SAT",
        "aVAT",
        "tVAT",
    ]
    assert "tissue_labels" in {artifact["name"] for artifact in manifest["source_artifacts"]}
    heading_x = {
        heading: next(float(word["x0"]) for word in words if word["text"] == heading)
        for heading in ("Spine", "Vertebral", "Axial")
    }
    if layout == "spine_profile_v2":
        assert "Stacked tissue-area profile" in text
        assert "Tissue mean HU" in text
        assert "white = NA" in text
        assert "Tissue area (cm2)" in text
        assert "Trunk area reference" in text
        heading_x["Stacked"] = next(
            float(word["x0"]) for word in words if word["text"] == "Stacked"
        )
        heading_x["Mean"] = min(
            float(word["x0"])
            for word in words
            if word["text"] == "mean" and float(word["x0"]) > heading_x["Stacked"]
        )
        assert (
            heading_x["Spine"]
            < heading_x["Stacked"]
            < heading_x["Mean"]
            < heading_x["Vertebral"]
            < heading_x["Axial"]
        )
        assert manifest["display"]["audit"]["panel_order"] == [
            "sagittal_spine",
            "tissue_area_profile",
            "tissue_hu_heatmap",
            "vertebral_summary",
            "axial_segmentation",
            "key_anthropometry",
        ]
        profile = manifest["display"]["audit"]["profile"]
        assert profile["layers"] == ["sm", "sat", "avat", "tvat"]
        assert profile["legend_labels"] == ["SM", "SAT", "aVAT", "tVAT"]
        assert profile["legend_rows"] == 1
        assert profile["hu_definition_caption"] == "SM -29..150 | fat -190..-30 HU"
        assert profile["trunk_area_reference_label"] == "Trunk area reference"
        heatmap = manifest["display"]["audit"]["hu_heatmap"]
        assert heatmap["tissues"] == ["sm", "sat", "avat", "tvat"]
        assert heatmap["display_labels"] == ["SM", "SAT", "aVAT", "tVAT"]
        assert heatmap["source_columns"] == [
            "skeletal_muscle_tissue_hu_m29_150_mean_hu",
            "sat_total_hu_m190_m30_mean_hu",
            "avat_hu_m190_m30_mean_hu",
            "tvat_hu_m190_m30_mean_hu",
        ]
        assert heatmap["color_scale_hu"] == [-190.0, 150.0]
        assert heatmap["color_scale_midpoint_hu"] == -20.0
        assert heatmap["aggregation"] == "per_slice_voxel_mean"
        assert heatmap["smoothing"] == "none"
        assert all(count > 0 for count in heatmap["valid_cell_counts"].values())
        assert not any(heatmap["clipped_cell_counts"].values())
        positions = case.measurement_bundle.slices["position_superior_mm"].to_numpy(dtype=float)
        inferior_mm, superior_mm = profile["displayed_superior_range_mm"]
        displayed_valid = (
            np.isfinite(positions) & (positions >= inferior_mm) & (positions <= superior_mm)
        )
        for layer, column in zip(profile["layers"], profile["source_columns"], strict=True):
            values = case.measurement_bundle.slices[column].to_numpy(dtype=float)
            validity_column = column.replace("_cm2", "_valid")
            layer_valid = displayed_valid & (
                np.isfinite(values)
                & (values >= 0)
                & case.measurement_bundle.slices[validity_column].to_numpy(dtype=bool)
            )
            expected_cm3 = float(
                np.trapezoid(values[layer_valid], positions[layer_valid]) / 10.0
            )
            assert profile["displayed_layer_integrals_cm3"][layer] == pytest.approx(expected_cm3)
            assert profile["valid_slice_counts"][layer] == int(np.count_nonzero(layer_valid))
    else:
        assert heading_x["Spine"] < heading_x["Vertebral"] < heading_x["Axial"]
        assert manifest["display"]["audit"]["panel_order"] == [
            "sagittal_spine",
            "vertebral_summary",
            "axial_segmentation",
            "key_anthropometry",
        ]
    table_left = heading_x["Vertebral"]
    table_right = heading_x["Axial"]
    table_levels = set(audit["table"]["displayed_levels"])
    level_words = [
        word
        for word in words
        if word["text"] in table_levels and table_left <= float(word["x0"]) < table_right
    ]
    assert len(level_words) == len(table_levels)
    level_tops = sorted(float(word["top"]) for word in level_words)
    assert np.ptp(np.diff(level_tops)) < 0.2
    assert np.diff(level_tops).mean() == pytest.approx(audit["table"]["row_height_pt"], abs=0.2)
    raw = result.pdf_path.read_bytes() + result.manifest_path.read_bytes()
    assert str(Path("/") / "Users" / "localuser").encode() not in raw
    assert b"localuser" not in raw
    assert all("/private/" not in str(value) for value in reader.metadata.values())
    unsafe = dict(manifest)
    unsafe["debug"] = str(Path("/") / "data" / "clinical" / "patient-001" / "input.nii.gz")
    with pytest.raises(ReportValidationError, match="forbidden local/source path"):
        validate_report_manifest(unsafe)


def test_review_findings_are_rendered_as_plain_language_notes(base_config, tmp_path):
    case_id = "case-qc-notes"
    reasons = {
        "disconnected_vertebral_body": "The segmentation contains disconnected components.",
        "vertebra_touches_cranial_fov": "A predicted vertebra reaches the upper image boundary.",
        "vertebral_body_extent_invalid": "A vertebral body is incomplete or truncated.",
        "slice_measurement_invalid": "One or more slice measurements are invalid.",
    }
    entries = tuple(
        ReviewEntry(
            case_id=case_id,
            domain=domain,
            code=code,
            reason=reasons[code],
            automatic_action="Processing completed; inspect the canonical QC artifact.",
        )
        for domain, code in (
            ("vertebral", "disconnected_vertebral_body"),
            ("vertebral", "vertebra_touches_cranial_fov"),
            ("measurement", "vertebral_body_extent_invalid"),
            ("measurement", "slice_measurement_invalid"),
        )
    )
    case = replace(
        _report_case(base_config, case_id),
        extra_review_entries=entries,
    )
    result = render_case_report(case, tmp_path, _settings("spine_profile_v2"))
    manifest = validate_report_manifest(result.manifest_path, pdf_path=result.pdf_path)
    with pdfplumber.open(result.pdf_path) as document:
        text = document.pages[0].extract_text()

    normalized = " ".join(text.split())
    assert "Review required" not in normalized
    for entry in entries:
        assert entry.code not in normalized
    assert "Disconnected vertebral body segmentation." in normalized
    assert "Incomplete or truncated vertebral body extent." in normalized
    assert "One or more slice measurements are invalid." in normalized
    assert reasons["vertebra_touches_cranial_fov"] not in normalized
    assert "Details: canonical quality-control record." in normalized
    notes_audit = manifest["display"]["audit"]["notes_qc"]
    assert notes_audit["columns_used"] == 3
    assert notes_audit["column_block_counts"] == [1, 1, 1]
    assert notes_audit["section_headings"] == [
        "Spine segmentation",
        "Measurements",
        "Data and provenance",
    ]
    assert notes_audit["review_count"] == 4
    assert notes_audit["displayed_review_count"] == 3
    assert notes_audit["suppressed_review_codes"] == ["vertebra_touches_cranial_fov"]
    assert notes_audit["status_visible"] is False
    assert notes_audit["block_split"] is False
    assert not notes_audit["truncated"]


def test_dense_plain_language_notes_are_grouped_without_truncation(base_config, tmp_path):
    case_id = "case-dense-qc-notes"
    reasons = {
        "disconnected_vertebral_body": "The segmentation contains disconnected components.",
        "vertebra_touches_cranial_fov": "A predicted vertebra reaches the upper image boundary.",
        "vertebral_body_extent_invalid": "A vertebral body is incomplete or truncated.",
        "trunk_contour_touches_fov": "The trunk contour reaches the image boundary.",
        "pelvic_maximum_unavailable": "The pelvic maximum could not be measured.",
    }
    entries = tuple(
        ReviewEntry(
            case_id=case_id,
            domain=domain,
            code=code,
            reason=reasons[code],
            automatic_action="Processing completed; inspect the canonical QC artifact.",
        )
        for domain, code in (
            ("vertebral", "disconnected_vertebral_body"),
            ("vertebral", "vertebra_touches_cranial_fov"),
            ("measurement", "vertebral_body_extent_invalid"),
            ("measurement", "trunk_contour_touches_fov"),
            ("measurement", "pelvic_maximum_unavailable"),
        )
    )
    case = replace(
        _report_case(base_config, case_id),
        extra_review_entries=entries,
    )
    result = render_case_report(case, tmp_path, _settings("spine_profile_v2"))
    manifest = validate_report_manifest(result.manifest_path, pdf_path=result.pdf_path)
    with pdfplumber.open(result.pdf_path) as document:
        text = " ".join(document.pages[0].extract_text().split())

    assert "Review required" not in text
    assert "Details: canonical quality-control record." in text
    for entry in entries:
        assert entry.code not in text
    assert reasons["vertebra_touches_cranial_fov"] not in text
    assert "Disconnected vertebral body segmentation." in text
    assert "Incomplete or truncated vertebral body extent." in text
    assert "Trunk contour reaches the image boundary." in text
    assert "Pelvic maximum could not be measured." in text
    notes_audit = manifest["display"]["audit"]["notes_qc"]
    assert notes_audit["columns_used"] == 3
    assert notes_audit["section_headings"] == [
        "Spine segmentation",
        "Measurements",
        "Data and provenance",
    ]
    assert notes_audit["review_count"] == 5
    assert notes_audit["displayed_review_count"] == 4
    assert notes_audit["suppressed_review_codes"] == ["vertebra_touches_cranial_fov"]
    assert notes_audit["status_visible"] is False
    assert notes_audit["block_split"] is False
    assert not notes_audit["truncated"]


def test_more_than_eight_notes_compact_without_machine_codes_or_overflow(
    base_config,
    tmp_path,
):
    case_id = "case-many-notes"
    entries = tuple(
        ReviewEntry(
            case_id=case_id,
            domain="measurement",
            code=f"measurement_observation_{index}",
            reason=f"Readable measurement observation {index + 1}.",
            automatic_action="Processing completed; inspect the canonical record.",
        )
        for index in range(9)
    )
    case = replace(
        _report_case(base_config, case_id),
        extra_review_entries=entries,
    )
    result = render_case_report(case, tmp_path, _settings("spine_profile_v2"))
    manifest = validate_report_manifest(result.manifest_path, pdf_path=result.pdf_path)
    with pdfplumber.open(result.pdf_path) as document:
        text = " ".join(document.pages[0].extract_text().split())

    assert "Readable measurement observation 1." in text
    assert "Readable measurement observation 2." not in text
    assert "8 additional findings in the canonical quality-control record." in text
    assert "measurement_observation_" not in text
    notes_audit = manifest["display"]["audit"]["notes_qc"]
    assert notes_audit["displayed_review_count"] == 9
    assert notes_audit["truncated"] is False


def test_profile_keeps_valid_tissues_when_one_layer_has_a_missing_slice(
    base_config,
    tmp_path,
):
    case = _report_case(base_config, "case-profile-partial-layer")
    slices = case.measurement_bundle.slices.copy()
    source = "sat_total_hu_m190_m30_area_cm2"
    target = slices.index[len(slices) // 2]
    slices.loc[target, source] = np.nan
    slices.loc[target, source.replace("_cm2", "_valid")] = False
    bundle = replace(case.measurement_bundle, slices=slices)

    result = render_case_report(
        replace(case, measurement_bundle=bundle),
        tmp_path,
        _settings("spine_profile_v2"),
    )
    manifest = validate_report_manifest(result.manifest_path, pdf_path=result.pdf_path)
    profile = manifest["display"]["audit"]["profile"]

    assert profile["valid_slice_counts"]["sat"] + 1 == profile["valid_slice_counts"]["sm"]
    assert profile["valid_slice_counts"]["sm"] == profile["valid_slice_counts"]["avat"]
    assert profile["displayed_layer_integrals_cm3"]["sm"] > 0
    assert profile["displayed_layer_integrals_cm3"]["avat"] > 0


def test_report_id_and_pdf_are_deterministic(base_config, tmp_path):
    case = _report_case(base_config)
    first = render_case_report(case, tmp_path / "one", _settings())
    second = render_case_report(case, tmp_path / "two", _settings())
    assert first.report_id == second.report_id
    assert first.pdf_sha256 == second.pdf_sha256
    changed = render_case_report(case, tmp_path / "three", _settings("spine_profile_v2"))
    assert changed.report_id != first.report_id


def test_report_rejects_mixed_orientation_vertebral_measurement_sources(base_config, tmp_path):
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

    changed_tissue = case.tissue_labels_zyx.copy()
    changed_tissue[0, 0, 0] = 1
    with pytest.raises(ReportValidationError, match="tissue labels disagree"):
        render_case_report(
            replace(case, tissue_labels_zyx=changed_tissue),
            tmp_path / "tissue-labels",
            _settings(),
        )


def test_report_label_view_is_distinct_from_raw_measurement_compartments(
    base_config,
    tmp_path,
):
    case = _report_case(base_config, "case-dual-tissue-source")
    tissue_provenance = case.measurement_bundle.provenance["tissue"]

    assert tissue_provenance["label_sha256"] != tissue_provenance["compartment_sha256"]
    processed_sm_count = int(np.count_nonzero(case.tissue_labels_zyx[0] == 1))
    assert case.measurement_bundle.slices.iloc[0]["sm_voxel_count"] == (processed_sm_count + 1)

    result = render_case_report(case, tmp_path / "dual-source", _settings())
    assert result.pdf_path.is_file()


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
        "Manual review summary",
        "case-ordinary",
        "case-repaired",
    ]
    manifest = json.loads(reviewed_export["manifest_path"].read_text())
    assert manifest["manual_review_summary"]["included"]
    assert manifest["manual_review_summary"]["cases"][0]["case_id"] == "case-repaired"
    assert manifest["cases"][1]["combined_page_number"] == 3
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized_text = " ".join(text.split())
    assert "Lossless orientation transform applied" in text
    assert "input CT automatically reoriented before analysis" in normalized_text
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
        {
            "enabled": True,
            "layout": source.layout,
            "numeric_precision": 2,
        }
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
    assert "processing continued without orientation repair" in " ".join(text.split())
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
    source = render_case_report(_report_case(base_config), tmp_path / "source", _settings())
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
                    float(value) for value in geometry.physical_point(center_zyx[::-1])
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
    for column in _settings(layout).measurement_columns:
        definition = MEASUREMENT_DEFINITIONS[column]
        assert f"{definition['short_label']} ({definition['unit']}): NA*" in text
    if layout == "spine_profile_v2":
        assert "Stacked tissue-area profile" in text
        assert "Measurement output incomplete" in text

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
    action = ExportMeasurementBundle(SimpleNamespace(config=base_config, timestamp=1, device="cpu"))
    action(
        {
            "id": case.case_id,
            "workspace": workspace,
            "tmp/measurement_bundle": case.measurement_bundle,
        }
    )
    prepared_path = workspace / "prepared.nii.gz"
    tissue_path = workspace / "tissue_labels.nii.gz"
    body_path = workspace / "vertebral_bodies.nii.gz"
    sitk.WriteImage(case.prepared_image, str(prepared_path))
    tissue_image = sitk.GetImageFromArray(case.tissue_labels_zyx)
    tissue_image.CopyInformation(case.prepared_image)
    sitk.WriteImage(tissue_image, str(tissue_path))
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
                "tissue_label_mask": tissue_path.name,
                "vertebral_body_mask": body_path.name,
                "vertebral_result_json": vertebral_path.name,
                "measurement_directory": "tables",
                "orientation_report_json": orientation_path.name,
                "measurement_qc_json": "qc/qc.json",
                "technical_metadata": dict(case.technical_metadata),
            }
        ),
        encoding="utf-8",
    )
    loaded = load_case_report_input(manifest_path)
    assert loaded.case_id == case.case_id
    assert (
        loaded.measurement_bundle.identity.analysis_id
        == case.measurement_bundle.identity.analysis_id
    )
    result = render_case_report(loaded, tmp_path / "posthoc-report", _settings())
    assert len(PdfReader(result.pdf_path).pages) == 1
    args = build_parser().parse_args(["reports", "inspect", str(result.manifest_path)])
    assert args.command == "reports"
    assert args.reports_command == "inspect"


def test_pipeline_reporting_action_uses_same_service_and_declares_completion_markers(
    base_config, tmp_path
):
    config = deepcopy(base_config)
    config["reporting"]["enabled"] = True
    action = RenderCaseReport(SimpleNamespace(config=config, timestamp=1, device="cpu"))
    case = _report_case(config, "case-action")
    memory = {
        "id": case.case_id,
        "workspace": tmp_path,
        "tmp/prepared_image": SimpleNamespace(prepared_image=case.prepared_image),
        "tmp/orientation_result": case.orientation,
        "tmp/vertebral_result": case.vertebral_result,
        "tmp/measurement_bundle": case.measurement_bundle,
        TISSUE_LABEL_MASK: SimpleNamespace(data=case.tissue_labels_zyx),
        "tmp/report_metadata": dict(case.technical_metadata),
    }
    action(memory)
    action.validate_outputs(memory)
    assert action.io_persisted_outputs == [CASE_REPORT_PDF, CASE_REPORT_MANIFEST]
    assert Path(memory[CASE_REPORT_PDF]).is_file()
    assert Path(memory[CASE_REPORT_MANIFEST]).is_file()
