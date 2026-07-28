import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import jsonschema
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import SimpleITK as sitk

from BodyComposition.actions.measurement import (
    CreateBodySurface,
    ExportMeasurementBundle,
    MeasureCanonicalBodyComposition,
    _scientific_measurement_configuration,
)
from BodyComposition.actions.segm_int import SegmIntBodyComposition, _SegmInternal
from BodyComposition.actions.segm_totalsegmentator import SegmTotalSegmentator
from BodyComposition.actions.vertebral import SPINEPS_BODY_MASK
from BodyComposition.config import PipelineConfig
from BodyComposition.measurement import totalsegmentator_assets
from BodyComposition.measurement.api import (
    TABLE_NAMES,
    l3_measurements,
    load_measurement_tables,
    load_signature,
    range_measurements,
    signature_measurements,
)
from BodyComposition.measurement.body_surface import (
    TISSUE_ENVELOPE_BACKEND,
    body_surface_from_totalsegmentator,
)
from BodyComposition.measurement.builder import (
    build_measurement_bundle,
    measurement_analysis_id,
)
from BodyComposition.measurement.contracts import Landmark, LandmarkSet, MeasurementIdentity
from BodyComposition.measurement.review import write_measurement_review
from BodyComposition.model_manager import required_model_ids
from BodyComposition.pipelines.bodycomposition import canonical_actions
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import (
    ExecutionStatus,
    QCFlag,
    QCSeverity,
    QCStatus,
    VertebralResult,
)


def make_measurement_inputs(config, *, empty_vertebrae=False):
    shape = (80, 24, 28)
    geometry = ImageGeometry(
        size_xyz=tuple(reversed(shape)),
        spacing_xyz=(0.9, 1.2, 4.0),
        origin_lps_xyz=(10.0, -20.0, 0.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    image = np.zeros(shape, dtype=np.int16)
    tissues = np.zeros(shape, dtype=np.uint8)
    tissues[:, 5:9, 5:9] = 1
    tissues[:, 9:11, 6:10] = 3
    tissues[:, 11:13, 7:9] = 4
    tissues[:, 13:15, 8:10] = 5
    image[tissues == 1] = 40
    image[tissues == 3] = -100
    image[tissues == 4] = -90
    image[tissues == 5] = -70

    body_labels = np.zeros(shape, dtype=np.uint8)
    body_labels[:, 2:-2, 2:-2] = 1
    body_labels[20:60, 7:11, 22:26] = 2
    body_surface = body_surface_from_totalsegmentator(body_labels, geometry)

    vertebral = np.zeros(shape, dtype=np.uint8)
    if not empty_vertebrae:
        for label, start in (
            (19, 5),
            (17, 15),
            (16, 22),
            (15, 30),
            (14, 38),
            (13, 46),
            (12, 54),
            (10, 65),
        ):
            vertebral[start : start + 5, 9:15, 11:17] = label
    vertebral_result = VertebralResult(
        backend_id="synthetic_body_only",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=vertebral.copy(),
        vertebral_body_labels=vertebral,
        label_schema=config["LBL_VERTEBRALBODIES"],
        provenance={"model": "synthetic"},
    )
    return image, tissues, geometry, body_surface, vertebral_result


def make_bundle(config, *, empty_vertebrae=False):
    image, tissues, geometry, body_surface, vertebral_result = make_measurement_inputs(
        config,
        empty_vertebrae=empty_vertebrae,
    )
    identity = MeasurementIdentity("case-001", "run-001", "analysis-001")
    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=tissues,
        geometry=geometry,
        tissue_label_schema=config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        compartment_labels_zyx=tissues,
        compartment_label_schema=config["LBL_TISSUE_COMPARTMENTS"],
        body_surface=body_surface,
        vertebral_result=vertebral_result,
        identity=identity,
        landmarks=None,
        settings=config["measurements"],
        orientation_provenance={
            "state": "PASS_METADATA_MATCH",
            "qc_status": "pass",
            "manual_review_required": False,
            "orientation_changed": False,
        },
    )
    return bundle, image, geometry, tissues, vertebral_result


def sitk_image(array_zyx, geometry):
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(geometry.spacing_xyz)
    image.SetOrigin(geometry.origin_lps_xyz)
    image.SetDirection(geometry.direction_lps)
    return image


def test_complete_bundle_validates_three_native_territory_bins(base_config):
    bundle, image, _, _, _ = make_bundle(base_config)

    assert len(bundle.slices) == image.shape[0]
    assert np.all(np.diff(bundle.slices["position_superior_mm"]) > 0)
    assert bundle.vertebrae.groupby("vertebral_level").size().eq(3).all()
    assert set(bundle.vertebrae["territory_bin"]) == {1, 2, 3}
    assert "SACRUM" in set(bundle.vertebrae["vertebral_level"])
    assert len(bundle.signature) == 100
    assert bundle.signature["signature_bin"].tolist() == list(range(100))
    assert bundle.signature["bin_width_mm"].eq(20.0).all()
    assert bundle.signature["reference_alignment_method"].eq("direct_l3").all()
    assert bundle.summaries.iloc[0]["l3_200mm_slab_valid"]
    assert "orientation_changed" not in bundle.slices
    assert bundle.provenance["orientation"]["state"] == "PASS_METADATA_MATCH"
    assert set(TABLE_NAMES) == {
        "slices",
        "vertebrae",
        "summaries",
        "signature",
        "hu_distributions",
    }
    assert bundle.hu_distributions["tissue_key"].drop_duplicates().tolist() == [
        "sm",
        "sat",
        "avat",
        "tvat",
    ]
    assert bundle.hu_distributions.groupby("tissue_key").size().eq(68).all()
    assert bundle.hu_distributions["distribution_scope"].eq("analyzed_volume").all()


def test_hu_distributions_use_native_compartments_not_filtered_masks(base_config):
    image, compartments, geometry, body_surface, vertebral_result = make_measurement_inputs(
        base_config
    )
    filtered_labels = compartments.copy()
    native_sm_voxels = ((0, 5, 5), (0, 5, 6), (0, 5, 7))
    assert all(compartments[voxel] == 1 for voxel in native_sm_voxels)
    image[native_sm_voxels[0]] = -100
    image[native_sm_voxels[1]] = -250
    image[native_sm_voxels[2]] = 160
    for voxel in native_sm_voxels:
        filtered_labels[voxel] = 0

    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=filtered_labels,
        geometry=geometry,
        tissue_label_schema=base_config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        compartment_labels_zyx=compartments,
        compartment_label_schema=base_config["LBL_TISSUE_COMPARTMENTS"],
        body_surface=body_surface,
        vertebral_result=vertebral_result,
        identity=MeasurementIdentity("case-native-hu", "run-001", "analysis-001"),
        landmarks=None,
        settings=base_config["measurements"],
    )

    sm = bundle.hu_distributions.loc[bundle.hu_distributions["tissue_key"].eq("sm")]
    native_count = int(np.count_nonzero(compartments == 1))
    filtered_count = int(np.count_nonzero(filtered_labels == 1))
    assert native_count == filtered_count + len(native_sm_voxels)
    assert sm["source_semantics"].eq("model_native_compartment").all()
    assert sm["source_label_ids"].eq("1").all()
    assert sm["total_voxel_count"].eq(native_count).all()
    out_of_filtered_range_bin = sm.loc[sm["bin_lower_hu"].eq(-100.0) & sm["bin_upper_hu"].eq(-95.0)]
    assert out_of_filtered_range_bin["voxel_count"].tolist() == [1]
    assert sm["below_histogram_voxel_count"].eq(1).all()
    assert sm["above_histogram_voxel_count"].eq(1).all()
    assert sm["in_histogram_voxel_count"].eq(native_count - 2).all()
    assert sm["mean_hu"].eq(float(np.mean(image[compartments == 1]))).all()
    assert sm["total_volume_cm3"].eq(native_count * geometry.voxel_volume_mm3 / 1000.0).all()


def test_body_fov_contact_is_distinct_from_trunk_contour_contact(base_config):
    image, tissues, geometry, _, vertebral_result = make_measurement_inputs(base_config)
    body_labels = np.zeros(image.shape, dtype=np.uint8)
    body_labels[:, 2:-2, 2:-2] = 1
    body_labels[:, 10:12, 0] = 2
    body_surface = body_surface_from_totalsegmentator(body_labels, geometry)

    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=tissues,
        geometry=geometry,
        tissue_label_schema=base_config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        compartment_labels_zyx=tissues,
        compartment_label_schema=base_config["LBL_TISSUE_COMPARTMENTS"],
        body_surface=body_surface,
        vertebral_result=vertebral_result,
        identity=MeasurementIdentity("case-body-fov", "run-001", "analysis-001"),
        landmarks=None,
        settings=base_config["measurements"],
    )

    summary = bundle.summaries.iloc[0]
    assert bundle.slices["body_touching_fov"].all()
    assert not bundle.slices["trunk_touching_fov"].any()
    assert summary["body_surface_touches_fov"]
    assert summary["body_surface_touches_fov_slice_count"] == len(bundle.slices)
    assert not summary["trunk_contour_touches_fov"]
    assert any(flag.code == "body_surface_touches_fov" for flag in bundle.qc_flags)


def test_cropped_pelvic_value_keeps_selected_slice_provenance(base_config):
    image, tissues, geometry, _, vertebral_result = make_measurement_inputs(base_config)
    body_labels = np.zeros(image.shape, dtype=np.uint8)
    body_labels[:, 2:-2, :-2] = 1
    body_surface = body_surface_from_totalsegmentator(body_labels, geometry)

    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=tissues,
        geometry=geometry,
        tissue_label_schema=base_config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        compartment_labels_zyx=tissues,
        compartment_label_schema=base_config["LBL_TISSUE_COMPARTMENTS"],
        body_surface=body_surface,
        vertebral_result=vertebral_result,
        identity=MeasurementIdentity("case-cropped-pelvis", "run-001", "analysis-001"),
        landmarks=None,
        settings=base_config["measurements"],
    )

    summary = bundle.summaries.iloc[0]
    assert summary["ct_max_pelvic_circumference_value_is_fov_cropped"]
    assert summary["ct_max_pelvic_circumference_valid"]
    assert not summary["ct_max_pelvic_circumference_eligible"]
    assert any(flag.code == "pelvic_maximum_fov_cropped" for flag in bundle.qc_flags)

    corrupted_slices = bundle.slices.copy()
    selected_slice_id = int(summary["ct_max_pelvic_circumference_slice_id"])
    corrupted_slices.loc[
        corrupted_slices["slice_id"].eq(selected_slice_id),
        "trunk_touching_fov",
    ] = False
    corrupted_summary = bundle.summaries.copy()
    corrupted_summary.loc[
        :,
        "trunk_contour_touches_fov_slice_count",
    ] -= 1
    with pytest.raises(ValueError, match="source slice"):
        replace(
            bundle,
            slices=corrupted_slices,
            summaries=corrupted_summary,
        )


def test_oblique_longitudinal_allocation_is_recorded_and_flagged(base_config):
    axial, image, geometry, tissues, vertebral_result = make_bundle(base_config)
    assert not any(
        flag.code == "oblique_longitudinal_allocation_approximate" for flag in axial.qc_flags
    )
    assert axial.provenance["measurement"]["longitudinal_allocation"] == {
        "method": "native_slice_center_v1",
        "in_plane_superior_span_mm": 0.0,
        "maximum_unflagged_in_plane_superior_span_mm": 10.0,
        "review_required": False,
    }

    angle = np.deg2rad(30.0)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    oblique_geometry = replace(
        geometry,
        direction_lps=(1.0, 0.0, 0.0, 0.0, cosine, -sine, 0.0, sine, cosine),
    )
    body_surface = replace(axial.body_surface, geometry=oblique_geometry)
    oblique_vertebral = replace(vertebral_result, geometry=oblique_geometry)
    oblique = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=tissues,
        geometry=oblique_geometry,
        tissue_label_schema=base_config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        compartment_labels_zyx=tissues,
        compartment_label_schema=base_config["LBL_TISSUE_COMPARTMENTS"],
        body_surface=body_surface,
        vertebral_result=oblique_vertebral,
        identity=MeasurementIdentity("case-001", "run-001", "analysis-oblique"),
        landmarks=None,
        settings=base_config["measurements"],
    )

    assert any(
        flag.code == "oblique_longitudinal_allocation_approximate" for flag in oblique.qc_flags
    )
    allocation = oblique.provenance["measurement"]["longitudinal_allocation"]
    assert allocation["in_plane_superior_span_mm"] == pytest.approx(
        geometry.size_xyz[1] * geometry.spacing_xyz[1] * sine
    )
    assert allocation["review_required"]


@pytest.mark.parametrize(
    "column",
    [
        "l3_territory_valid",
        "l3_territory_complete",
        "l3_territory_reason",
        "l3_territory_coverage_fraction",
        "ct_min_trunk_circumference_t10_l5_position_superior_mm",
        "ct_min_trunk_circumference_t10_l5_slice_id",
        "ct_min_trunk_circumference_t10_l5_search_acquisition_coverage_fraction",
        "ct_min_trunk_circumference_t10_l5_search_anatomically_complete",
        "ct_min_trunk_circumference_t10_l5_at_search_boundary",
        "ct_max_pelvic_circumference_position_superior_mm",
        "ct_max_pelvic_circumference_slice_id",
        "ct_max_pelvic_circumference_search_acquisition_coverage_fraction",
        "ct_max_pelvic_circumference_search_anatomically_complete",
        "ct_max_pelvic_circumference_at_search_boundary",
        "ct_max_pelvic_circumference_search_touches_fov",
        "ct_max_pelvic_circumference_value_is_fov_cropped",
        "ct_max_pelvic_search_definition",
        "body_surface_touches_fov",
        "body_surface_touches_fov_slice_count",
        "trunk_contour_touches_fov",
        "trunk_contour_touches_fov_slice_count",
    ],
)
def test_summary_contract_rejects_missing_safety_fields(base_config, column):
    bundle, *_ = make_bundle(base_config)

    with pytest.raises(ValueError, match="summaries is missing columns"):
        replace(bundle, summaries=bundle.summaries.drop(columns=[column]))


def test_summary_contract_rejects_eligibility_that_contradicts_qc(base_config):
    bundle, *_ = make_bundle(base_config)
    summaries = bundle.summaries.copy()
    prefix = "ct_min_trunk_circumference_t10_l5"
    summaries.loc[0, f"{prefix}_valid"] = True
    summaries.loc[0, f"{prefix}_eligible"] = True
    summaries.loc[0, f"{prefix}_search_anatomically_complete"] = True
    summaries.loc[0, f"{prefix}_search_acquisition_coverage_fraction"] = 1.0
    summaries.loc[0, f"{prefix}_search_valid_contour_coverage_fraction"] = 1.0
    summaries.loc[0, f"{prefix}_at_search_boundary"] = True

    with pytest.raises(ValueError, match="eligibility contradicts"):
        replace(bundle, summaries=summaries)


def test_hu_distribution_contract_rejects_null_native_source_metadata(base_config):
    bundle, *_ = make_bundle(base_config)
    distributions = bundle.hu_distributions.copy()
    distributions.loc[distributions["tissue_key"].eq("sm"), "source_label_ids"] = pd.NA

    with pytest.raises(ValueError, match="inconsistent metadata"):
        replace(bundle, hu_distributions=distributions)


def test_slice_table_is_the_longitudinal_csa_and_hu_feature_source(base_config):
    bundle, _, _, _, _ = make_bundle(base_config)
    slices = bundle.slices

    assert np.all(np.diff(slices["position_superior_mm"].to_numpy(dtype=float)) > 0)
    assert slices["assigned_vertebral_level"].notna().any()
    for tissue in (
        "sm",
        "sat",
        "avat",
        "tvat",
        "total_vat",
        "skeletal_muscle_tissue_hu_m29_150",
        "sat_total_hu_m190_m30",
        "avat_hu_m190_m30",
        "tvat_hu_m190_m30",
    ):
        required = {
            f"{tissue}_area_cm2",
            f"{tissue}_area_valid",
            f"{tissue}_area_reason",
            f"{tissue}_mean_hu",
            f"{tissue}_hu_valid",
            f"{tissue}_hu_reason",
        }
        assert required.issubset(slices.columns)

    assert slices.loc[slices["sm_hu_valid"], "sm_mean_hu"].notna().all()


def test_named_tissue_profile_appends_signature_features_without_changing_core():
    config = PipelineConfig.load(
        Path("config/tissue_profiles/derstine_2022_vat_m150_m50.yaml")
    ).to_runtime_dict()

    bundle, *_ = make_bundle(config)

    assert "vat_total_hu_m150_m50_mean_csa_cm2" in bundle.signature
    assert "vat_total_hu_m150_m50_mean_hu" in bundle.signature
    assert "vat_total_hu_m150_m50_mean_csa_fraction_of_trunk" in bundle.signature
    assert "sm_mean_csa_cm2" in bundle.signature
    assert "imat_hu_m190_m30_mean_csa_cm2" not in bundle.signature


def test_empty_vertebral_segmentation_preserves_slices_and_explicit_qc(base_config):
    bundle, image, _, _, _ = make_bundle(base_config, empty_vertebrae=True)

    assert len(bundle.slices) == image.shape[0]
    assert bundle.vertebrae.empty
    assert bundle.slices["assigned_vertebral_level"].isna().all()
    assert set(bundle.slices["vertebral_assignment_status"]) == {"no_valid_territory"}
    assert len(bundle.signature) == 100
    assert not bundle.signature["reference_alignment_valid"].any()
    assert not bundle.signature["bin_valid"].any()
    assert bundle.signature["bin_reason"].eq("missing_anchor").all()
    assert any(flag.code == "vertebral_body_segmentation_empty" for flag in bundle.qc_flags)


def test_bundle_surfaces_fragmented_trunk_and_backend_neutral_variant_qc(base_config):
    image, tissues, geometry, _, vertebral_result = make_measurement_inputs(base_config)
    body_labels = np.zeros(image.shape, dtype=np.uint8)
    body_labels[:, 3:-3, 3:-3] = 1
    body_labels[10, 1:2, 8:11] = 1
    body_surface = body_surface_from_totalsegmentator(body_labels, geometry)
    vertebral = vertebral_result.vertebral_body_labels.copy()
    vertebral[72:77, 9:15, 11:17] = 18
    vertebral_result = replace(
        vertebral_result,
        whole_vertebra_labels=vertebral.copy(),
        vertebral_body_labels=vertebral,
    )

    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=tissues,
        geometry=geometry,
        tissue_label_schema=base_config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        compartment_labels_zyx=tissues,
        compartment_label_schema=base_config["LBL_TISSUE_COMPARTMENTS"],
        body_surface=body_surface,
        vertebral_result=vertebral_result,
        identity=MeasurementIdentity("case-001", "run-001", "analysis-001"),
        landmarks=None,
        settings=base_config["measurements"],
    )

    codes = {flag.code for flag in bundle.qc_flags}
    assert "trunk_surface_fragmented_slices" in codes
    assert "anatomical_variant_present" in codes
    fragmented = bundle.slices.loc[bundle.slices["trunk_mask_fragmented"]]
    assert fragmented["slice_qc_status"].eq("review").all()
    assert not fragmented["trunk_area_valid"].any()


def test_ambiguous_sequence_variant_stays_continuous_and_is_flagged(
    base_config,
):
    image, tissues, geometry, body_surface, vertebral_result = make_measurement_inputs(base_config)
    vertebral = vertebral_result.vertebral_body_labels.copy()
    vertebral[vertebral == 17] = 0
    vertebral_result = replace(
        vertebral_result,
        whole_vertebra_labels=vertebral.copy(),
        vertebral_body_labels=vertebral,
        qc_flags=(),
    )

    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=tissues,
        geometry=geometry,
        tissue_label_schema=base_config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        compartment_labels_zyx=tissues,
        compartment_label_schema=base_config["LBL_TISSUE_COMPARTMENTS"],
        body_surface=body_surface,
        vertebral_result=vertebral_result,
        identity=MeasurementIdentity("case-001", "run-001", "analysis-001"),
        landmarks=None,
        settings=base_config["measurements"],
    )

    assert "L5" not in set(bundle.vertebrae["vertebral_level"])
    l4 = bundle.vertebrae.loc[bundle.vertebrae["vertebral_level"].eq("L4")]
    assert l4["bin_valid"].all()
    assert l4["sequence_gap_caudal"].all()
    assert l4["territory_qc_status"].eq("review").all()
    assert any(flag.code == "anatomical_variant_present" for flag in bundle.qc_flags)
    assert any(flag.code == "vertebral_territory_sequence_gap" for flag in bundle.qc_flags)


def test_parquet_export_round_trip_and_api_views(base_config, tmp_path):
    bundle, _, _, _, _ = make_bundle(base_config)
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/measurement_bundle": bundle,
    }

    action(memory)
    action.validate_outputs(memory)

    table_directory = tmp_path / "tables"
    tables = load_measurement_tables(table_directory)
    assert set(tables) == set(TABLE_NAMES)
    assert len(tables["slices"]) == len(bundle.slices)
    assert tables["vertebrae"][["vertebral_level", "territory_bin"]].to_numpy().tolist() == (
        bundle.vertebrae[["vertebral_level", "territory_bin"]].to_numpy().tolist()
    )
    distributions = tables["hu_distributions"]
    assert distributions.groupby("tissue_key").size().eq(68).all()
    assert distributions["bin_width_hu"].eq(5.0).all()
    assert distributions["histogram_min_hu"].eq(-190.0).all()
    assert distributions["histogram_max_hu"].eq(150.0).all()
    for tissue, expected_median in {
        "sm": 40.0,
        "sat": -100.0,
        "avat": -90.0,
        "tvat": -70.0,
    }.items():
        rows = distributions.loc[distributions["tissue_key"].eq(tissue)]
        assert rows["distribution_valid"].all()
        assert rows["median_hu"].eq(expected_median).all()
        assert rows["voxel_count"].sum() == rows["total_voxel_count"].iloc[0]
        assert rows["voxel_fraction"].sum() == pytest.approx(1.0)
    signature = signature_measurements(table_directory)
    assert signature["signature_bin"].tolist() == list(range(100))
    assert signature["signature_profile_id"].nunique() == 1
    observed = signature.loc[signature["coverage_fraction"].fillna(0).gt(0)]
    outside = signature.loc[signature["coverage_fraction"].eq(0)]
    assert not observed.empty
    assert not outside.empty
    assert observed["sm_mean_csa_cm2_valid"].all()
    assert observed["sm_mean_hu_valid"].all()
    assert observed["heart_mean_csa_cm2"].eq(0).all()
    assert not observed["heart_mean_hu_valid"].any()
    assert outside["sm_mean_csa_cm2"].isna().all()
    assert outside["sm_mean_csa_cm2_reason"].eq("outside_fov").all()
    assert outside["sm_mean_csa_cm2_coverage_fraction"].eq(0).all()
    assert outside["sm_mean_hu_reason"].eq("outside_fov").all()
    assert outside["trunk_mean_csa_cm2_reason"].eq("outside_fov").all()
    signature_components = load_signature(table_directory)
    assert signature_components.longitudinal.equals(signature)
    assert signature_components.tissue_hu_distributions.equals(distributions)
    for component in (
        signature_components.longitudinal,
        signature_components.tissue_hu_distributions,
    ):
        assert component["case_id"].eq(bundle.identity.case_id).all()
        assert component["run_id"].eq(bundle.identity.run_id).all()
        assert component["analysis_id"].eq(bundle.identity.analysis_id).all()
    provenance_components = bundle.provenance["measurement"]["signature_components"]
    assert [component["table"] for component in provenance_components] == [
        "signature.parquet",
        "hu_distributions.parquet",
    ]
    assert provenance_components[1]["source_semantics"] == ("model_native_compartment")
    assert (
        l3_measurements(table_directory, aggregation="territory_mean").iloc[0]["aggregation"]
        == "territory_mean"
    )
    assert l3_measurements(table_directory, aggregation="slice").iloc[0]["aggregation"] == "slice"
    indexed_l3 = l3_measurements(
        table_directory,
        aggregation="slice",
        height_m=2.0,
    ).iloc[0]
    assert indexed_l3["smi_skeletal_muscle_tissue_hu_m29_150_cm2_m2"] == pytest.approx(
        indexed_l3["skeletal_muscle_tissue_hu_m29_150_area_cm2"] / 4.0
    )
    assert indexed_l3["height_source"] == "caller_supplied_measured_height"
    named = range_measurements(
        table_directory,
        start_level="T12",
        end_level="L5",
    )
    assert named.iloc[0]["range_valid"]
    assert not named.iloc[0]["range_eligible"]
    assert named.iloc[0]["range_eligibility_reason"] == "sequence_gap"
    assert named.iloc[0]["analysis_id"] == bundle.identity.analysis_id
    assert named.iloc[0]["aggregation"] == "physical_range"
    indexed_range = range_measurements(
        table_directory,
        start_level="T12",
        end_level="L5",
        height_m=2.0,
    ).iloc[0]
    assert indexed_range["skeletal_muscle_tissue_hu_m29_150_volume_index_cm3_m2"] == pytest.approx(
        indexed_range["skeletal_muscle_tissue_hu_m29_150_volume_cm3"] / 4.0
    )
    with pytest.raises(ValueError, match="positive"):
        l3_measurements(table_directory, height_m=0.0)
    qc = json.loads((tmp_path / "qc" / "qc.json").read_text())
    schema = json.loads(Path("BodyComposition/schemas/measurement_qc.schema.json").read_text())
    jsonschema.validate(qc, schema)
    assert qc["provenance"]["vertebral"]["backend_id"] == "synthetic_body_only"
    assert qc["provenance"]["orientation"]["state"] == "PASS_METADATA_MATCH"

    assert not list(tmp_path.rglob("*.partial*"))

    summaries_path = table_directory / "summaries.parquet"
    parquet_schema = pq.read_schema(summaries_path)
    metadata = parquet_schema.metadata
    summaries = pd.read_parquet(summaries_path)
    summaries.loc[:, "analysis_id"] = "analysis-from-another-case"
    corrupted = pa.Table.from_pandas(
        summaries,
        schema=parquet_schema.remove_metadata(),
        preserve_index=False,
        safe=True,
    )
    pq.write_table(corrupted.replace_schema_metadata(metadata), summaries_path)
    with pytest.raises(ValueError, match="inconsistent analysis_id"):
        load_measurement_tables(table_directory)


def test_parquet_reader_rejects_missing_vertebral_territory_bin(base_config, tmp_path):
    bundle, _, _, _, _ = make_bundle(base_config)
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    action(
        {
            "id": "case-001",
            "workspace": tmp_path,
            "tmp/measurement_bundle": bundle,
        }
    )

    vertebrae_path = tmp_path / "tables" / "vertebrae.parquet"
    vertebrae = pq.read_table(vertebrae_path)
    frame = vertebrae.to_pandas()
    removed_row = frame.index[frame["vertebral_level"].eq("L3") & frame["territory_bin"].eq(2)][0]
    corrupted = vertebrae.take(
        pa.array(
            [index for index in range(len(frame)) if index != removed_row],
            type=pa.int64(),
        )
    )
    pq.write_table(corrupted, vertebrae_path)

    with pytest.raises(ValueError, match="unstable bins for 'L3'"):
        load_measurement_tables(vertebrae_path.parent)


def test_parquet_reader_rejects_schema_metadata_drift(base_config, tmp_path):
    bundle, _, _, _, _ = make_bundle(base_config)
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    action(
        {
            "id": "case-001",
            "workspace": tmp_path,
            "tmp/measurement_bundle": bundle,
        }
    )

    slices_path = tmp_path / "tables" / "slices.parquet"
    slices = pq.read_table(slices_path)
    metadata = dict(slices.schema.metadata or {})
    metadata[b"bodycomposition.measurement_schema_version"] = b"0.0.0"
    pq.write_table(slices.replace_schema_metadata(metadata), slices_path)

    with pytest.raises(ValueError, match="inconsistent measurement schema metadata"):
        load_measurement_tables(slices_path.parent)


def test_parquet_reader_rejects_shifted_signature_grid(base_config, tmp_path):
    bundle, _, _, _, _ = make_bundle(base_config)
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    action(
        {
            "id": "case-001",
            "workspace": tmp_path,
            "tmp/measurement_bundle": bundle,
        }
    )

    signature_path = tmp_path / "tables" / "signature.parquet"
    parquet_schema = pq.read_schema(signature_path)
    metadata = parquet_schema.metadata
    signature = pd.read_parquet(signature_path)
    signature.loc[50, "reference_center_mm"] = 1.0
    corrupted = pa.Table.from_pandas(
        signature,
        schema=parquet_schema.remove_metadata(),
        preserve_index=False,
        safe=True,
    )
    pq.write_table(corrupted.replace_schema_metadata(metadata), signature_path)

    with pytest.raises(ValueError, match="fixed L3-centred grid"):
        load_measurement_tables(signature_path.parent)


def test_parquet_reader_rejects_shifted_hu_distribution_grid(base_config, tmp_path):
    bundle, _, _, _, _ = make_bundle(base_config)
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    action(
        {
            "id": "case-001",
            "workspace": tmp_path,
            "tmp/measurement_bundle": bundle,
        }
    )

    distribution_path = tmp_path / "tables" / "hu_distributions.parquet"
    parquet_schema = pq.read_schema(distribution_path)
    metadata = parquet_schema.metadata
    distributions = pd.read_parquet(distribution_path)
    distributions.loc[0, "bin_lower_hu"] = -189.0
    corrupted = pa.Table.from_pandas(
        distributions,
        schema=parquet_schema.remove_metadata(),
        preserve_index=False,
        safe=True,
    )
    pq.write_table(
        corrupted.replace_schema_metadata(metadata),
        distribution_path,
    )

    with pytest.raises(ValueError, match="fixed HU grid"):
        load_measurement_tables(distribution_path.parent)


def test_missing_anatomy_keeps_identical_nullable_parquet_schemas(
    base_config,
    tmp_path,
):
    complete, *_ = make_bundle(base_config)
    missing, *_ = make_bundle(base_config, empty_vertebrae=True)
    schemas = {}
    for state, bundle in (("complete", complete), ("missing", missing)):
        workspace = tmp_path / state
        action = ExportMeasurementBundle(
            SimpleNamespace(config=base_config, timestamp=123, device="cpu")
        )
        action(
            {
                "id": "case-001",
                "workspace": workspace,
                "tmp/measurement_bundle": bundle,
            }
        )
        table_directory = workspace / "tables"
        schemas[state] = {
            name: pq.read_schema(table_directory / f"{name}.parquet") for name in TABLE_NAMES
        }
        load_measurement_tables(table_directory)

    for name in TABLE_NAMES:
        assert schemas["complete"][name].equals(
            schemas["missing"][name],
            check_metadata=True,
        )
    assert schemas["missing"]["vertebrae"].field("territory_bin").type == pa.int64()
    assert schemas["missing"]["slices"].field("assigned_vertebral_level").type == pa.string()
    assert "sm_volume_cm3" not in missing.vertebrae
    assert "sm_mean_csa_cm2" in missing.vertebrae


def test_range_api_reuses_the_analysis_coverage_tolerance(base_config, tmp_path):
    config = deepcopy(base_config)
    config["measurements"]["full_coverage_tolerance"] = 0.99
    bundle, *_ = make_bundle(config)
    slices = bundle.slices.copy()
    vertebrae = bundle.vertebrae.copy()
    acquisition_lower = float(slices["slice_slab_inferior_mm"].min())
    acquisition_upper = float(slices["slice_slab_superior_mm"].max())
    observed_length = acquisition_upper - acquisition_lower
    target_coverage = 0.995
    extension = observed_length / target_coverage - observed_length
    vertebrae.loc[
        vertebrae["vertebral_level"].eq("L5"),
        "territory_inferior_mm",
    ] = acquisition_lower - extension
    vertebrae.loc[
        vertebrae["vertebral_level"].eq("T12"),
        "territory_superior_mm",
    ] = acquisition_upper
    bundle = replace(bundle, slices=slices, vertebrae=vertebrae)
    action = ExportMeasurementBundle(SimpleNamespace(config=config, timestamp=123, device="cpu"))
    action(
        {
            "id": "case-001",
            "workspace": tmp_path,
            "tmp/measurement_bundle": bundle,
        }
    )

    result = range_measurements(
        tmp_path / "tables",
        start_level="L5",
        end_level="T12",
    ).iloc[0]

    assert result["coverage_fraction"] == pytest.approx(target_coverage)
    assert result["full_coverage_tolerance"] == pytest.approx(0.99)
    assert result["range_valid"]


def test_informational_coverage_flag_does_not_request_manual_review(
    base_config,
    tmp_path,
):
    bundle, _, _, _, _ = make_bundle(base_config)
    bundle = replace(
        bundle,
        qc_flags=(
            QCFlag(
                code="expected_outside_fov",
                stage="measurement",
                severity=QCSeverity.INFO,
                reason="Optional vertebral territories are outside the acquired FOV.",
            ),
        ),
    )
    assert bundle.qc_status == QCStatus.PASS
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/measurement_bundle": bundle,
    }

    action(memory)

    qc = json.loads((tmp_path / "qc" / "qc.json").read_text())
    assert qc["qc_status"] == "pass"
    assert not qc["manual_review_required"]
    assert qc["qc_flags"][0]["severity"] == "info"


def test_measurement_review_contains_axial_and_longitudinal_panels(base_config, tmp_path):
    bundle, image, geometry, _, _ = make_bundle(base_config)
    output = write_measurement_review(
        sitk_image(image, geometry),
        bundle,
        tmp_path / "measurement-review.png",
    )

    rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert rendered is not None
    assert rendered.shape == (790, 1280, 3)
    assert rendered.std() > 5


def test_measurement_analysis_identity_is_order_stable_and_content_sensitive(base_config):
    image, tissues, _, body_surface, vertebral_result = make_measurement_inputs(base_config)
    first = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
    )
    reordered = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"beta": 2, "alpha": 1},
    )
    fortran_ordered = measurement_analysis_id(
        np.asfortranarray(image),
        np.asfortranarray(tissues),
        replace(
            body_surface,
            body_mask_zyx=np.asfortranarray(body_surface.body_mask_zyx),
            trunk_mask_zyx=np.asfortranarray(body_surface.trunk_mask_zyx),
        ),
        replace(
            vertebral_result,
            vertebral_body_labels=np.asfortranarray(vertebral_result.vertebral_body_labels),
        ),
        {"alpha": 1, "beta": 2},
    )
    changed_tissues = tissues.copy()
    changed_tissues[0, 0, 0] = 1
    changed = measurement_analysis_id(
        image,
        changed_tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
    )
    changed_provenance = measurement_analysis_id(
        image,
        tissues,
        replace(body_surface, provenance={"asset_sha256": "changed"}),
        vertebral_result,
        {"alpha": 1, "beta": 2},
    )
    landmarks = LandmarkSet(
        lowest_rib_inferior=Landmark("rib", 250.0, "synthetic", True),
        iliac_crest_superior=Landmark("crest", 150.0, "synthetic", True),
    )
    with_landmarks = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
        landmarks=landmarks,
    )
    with_tissue_contract = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
        tissue_label_schema={1: "SM", 3: "SAT"},
        tissue_preprocessing={"filter_size": False},
    )
    with_orientation_provenance = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
        orientation_provenance={
            "state": "PASS_METADATA_MATCH",
            "orientation_changed": False,
        },
    )
    compartments = tissues.copy()
    with_compartments = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
        compartment_labels_zyx=compartments,
        compartment_label_schema=base_config["LBL_TISSUE_COMPARTMENTS"],
    )
    changed_compartments = compartments.copy()
    changed_compartments[0, 0, 0] = 2
    with_changed_compartments = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
        compartment_labels_zyx=changed_compartments,
        compartment_label_schema=base_config["LBL_TISSUE_COMPARTMENTS"],
    )

    assert first == reordered
    assert first == fortran_ordered
    assert first.startswith("analysis-")
    assert first != changed
    assert first != changed_provenance
    assert first != with_landmarks
    assert first != with_tissue_contract
    assert first != with_orientation_provenance
    assert with_compartments != with_changed_compartments


def test_measurement_identity_configuration_excludes_output_and_review_policy(base_config):
    settings = json.loads(json.dumps(base_config["measurements"]))
    output_only = json.loads(json.dumps(settings))
    output_only["enabled"] = not output_only["enabled"]
    output_only["review"]["enabled"] = not output_only["review"]["enabled"]
    output_only["export"]["parquet"] = not output_only["export"]["parquet"]
    output_only["body_surface"]["save_mask"] = not output_only["body_surface"]["save_mask"]

    assert _scientific_measurement_configuration(settings) == _scientific_measurement_configuration(
        output_only
    )

    output_only["body_surface"]["smoothing_sigma_mm"] += 1
    assert _scientific_measurement_configuration(settings) != _scientific_measurement_configuration(
        output_only
    )


def test_canonical_pipeline_uses_full_prepared_domain_and_body_label_source(pipeline_stub):
    actions = canonical_actions(pipeline_stub)
    names = [type(action).__name__ for action in actions]

    assert "CreateBoundingBox" not in names
    assert "ApplyBoundingBox" not in names
    tissue = next(action for action in actions if isinstance(action, SegmIntBodyComposition))
    measurement = next(
        action for action in actions if isinstance(action, MeasureCanonicalBodyComposition)
    )
    surface = next(action for action in actions if isinstance(action, CreateBodySurface))
    assert tissue.input_image_name == "tmp/index"
    assert tissue.model_preset == "ResEncL"
    assert measurement.vertebral_body_source_name == SPINEPS_BODY_MASK
    assert "tmp/vertebral_result" in measurement.io_inputs
    assert "masks/tissue_compartments.nii.gz" in measurement.io_inputs
    assert "masks/tissue_labels.nii.gz" in measurement.io_inputs
    assert surface.backend == TISSUE_ENVELOPE_BACKEND
    assert names.index("MasksInternalTissue") < names.index("CreateBodySurface")


def test_tissue_envelope_pipeline_can_run_without_totalsegmentator(
    pipeline_stub,
):
    pipeline_stub.config["measurements"]["landmarks"]["enabled"] = False

    actions = canonical_actions(pipeline_stub)
    names = [type(action).__name__ for action in actions]

    assert "SegmTotalSegmentatorConfig" not in names
    assert "SegmTotalSegmentator" not in names
    assert "CreateMeasurementLandmarks" not in names
    assert "CreateBodySurface" in names
    assert "MeasureCanonicalBodyComposition" in names


def test_measurement_support_action_resolves_pinned_nnunet_directory(
    tmp_path,
    pipeline_stub,
    monkeypatch,
):
    model_root = tmp_path / "models"
    model_root.mkdir()
    pipeline_stub.config["paths"]["weights"]["totalsegmentator"] = model_root
    action = SegmTotalSegmentator(
        pipeline_stub,
        image="tmp/index",
        task="body_landmarks",
    )

    assert action.task_id == 297
    assert action.model_folds == [0]
    assert action.model_path == (
        model_root.resolve()
        / "Dataset297_TotalSegmentator_total_3mm_1559subj"
        / "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
    )
    assert list(model_root.iterdir()) == []


def test_measurement_totalsegmentator_assets_are_checked_without_download(
    tmp_path,
    monkeypatch,
):
    files = {
        "trainer/dataset.json": b"dataset",
        "trainer/plans.json": b"plans",
        "trainer/fold_0/checkpoint_final.pth": b"checkpoint",
    }
    manifest = {
        "bodytrunk": {
            "task_id": 299,
            "model_title": "TotalSegmentator-body",
            "directory": "Dataset299_body_1559subj",
            "files": {
                relative: {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                for relative, content in files.items()
            },
        }
    }
    monkeypatch.setattr(
        totalsegmentator_assets,
        "MEASUREMENT_MODEL_MANIFEST",
        manifest,
    )
    totalsegmentator_assets._check_cached.cache_clear()

    with pytest.raises(FileNotFoundError, match="bodycomposition models sync"):
        totalsegmentator_assets.require_measurement_model("bodytrunk", tmp_path)

    model_directory = tmp_path / "Dataset299_body_1559subj"
    for relative, content in files.items():
        path = model_directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    totalsegmentator_assets.require_measurement_model("bodytrunk", tmp_path)

    (model_directory / "trainer/plans.json").write_bytes(b"corrupted")
    report = totalsegmentator_assets.check_measurement_model("bodytrunk", tmp_path)
    assert not report.ready
    assert report.errors == ("size_mismatch:trainer/plans.json",)


def test_measurement_support_verifies_asset_before_predictor_initialization(
    tmp_path,
    pipeline_stub,
    monkeypatch,
):
    pipeline_stub.config["paths"]["weights"]["totalsegmentator"] = tmp_path
    events = []
    report = object()
    predictor = object()
    pipeline_stub.prepare_exclusive_model = lambda action: events.append(("exclusive", action.task))
    monkeypatch.setattr(
        "BodyComposition.actions.segm_totalsegmentator.require_measurement_model",
        lambda task, root: events.append(("verify", task, Path(root))) or report,
    )
    monkeypatch.setattr(
        _SegmInternal,
        "_get_predictor",
        lambda self: events.append(("initialize", self.model_path)) or predictor,
    )

    action = SegmTotalSegmentator(
        pipeline_stub,
        image="tmp/index",
        task="body_landmarks",
    )
    assert action._get_predictor() is predictor
    assert action.asset_report is report
    assert events[0] == ("exclusive", "body_landmarks")
    assert events[1] == ("verify", "body_landmarks", tmp_path)
    assert events[2][0] == "initialize"


def test_default_model_sync_omits_optional_body_model_but_keeps_landmarks():
    models = required_model_ids(PipelineConfig.load())
    assert "totalsegmentator_body_task299_v1" not in models
    assert "totalsegmentator_total_task297_landmarks_v1" in models


def test_configured_model_sync_respects_optional_measurement_support():
    config = PipelineConfig.model_validate({"measurements": {"landmarks": {"enabled": False}}})
    models = required_model_ids(config)
    assert "totalsegmentator_body_task299_v1" not in models
    assert "totalsegmentator_total_task297_landmarks_v1" not in models

    value = config.normalized()
    value["body_surface"]["backend"] = "totalsegmentator_body_task299_v1"
    models = required_model_ids(PipelineConfig.model_validate(value))
    assert "totalsegmentator_body_task299_v1" in models
    assert "totalsegmentator_total_task297_landmarks_v1" not in models

    value["measurements"]["landmarks"]["enabled"] = True
    models = required_model_ids(PipelineConfig.model_validate(value))
    assert "totalsegmentator_body_task299_v1" in models
    assert "totalsegmentator_total_task297_landmarks_v1" in models
