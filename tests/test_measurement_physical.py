import math

import numpy as np
import pandas as pd
import pytest

from BodyComposition.measurement.aggregation import aggregate_physical_range
from BodyComposition.measurement.body_surface import (
    DETERMINISTIC_BODY_BACKEND,
    TISSUE_ENVELOPE_BACKEND,
    TOTALSEGMENTATOR_BODY_BACKEND,
    body_surface_from_totalsegmentator,
    deterministic_body_surface,
    tissue_segmentation_envelope,
)
from BodyComposition.measurement.contours import external_contour_measurement
from BodyComposition.measurement.contracts import MeasurementIdentity
from BodyComposition.measurement.landmarks import landmarks_from_totalsegmentator
from BodyComposition.measurement.physical import (
    slice_geometry_table,
    validate_array_zyx,
)
from BodyComposition.measurement.slices import (
    annotate_longitudinal_circumference_qc,
    calculate_canonical_slice_measurements,
)
from BodyComposition.utils.geometry import ImageGeometry


def make_geometry(
    shape_zyx,
    *,
    spacing_xyz=(1.0, 1.0, 2.0),
    origin_lps_xyz=(0.0, 0.0, 0.0),
    direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
):
    return ImageGeometry(
        size_xyz=tuple(reversed(shape_zyx)),
        spacing_xyz=spacing_xyz,
        origin_lps_xyz=origin_lps_xyz,
        direction_lps=direction_lps,
    )


def test_slice_geometry_is_sorted_inferior_to_superior_for_flipped_storage_axis():
    geometry = make_geometry(
        (4, 5, 6),
        origin_lps_xyz=(7.0, -2.0, 20.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    )

    table = slice_geometry_table(geometry)

    assert table["longitudinal_order"].tolist() == [0, 1, 2, 3]
    assert table["slice_id"].tolist() == [3, 2, 1, 0]
    assert table["slice_index_zyx_z"].tolist() == [3, 2, 1, 0]
    assert table["position_superior_mm"].tolist() == pytest.approx([14, 16, 18, 20])
    assert np.all(np.diff(table["position_superior_mm"]) > 0)


def test_oblique_slice_geometry_uses_full_affine_and_projected_thickness():
    angle = math.radians(30)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    geometry = make_geometry(
        (3, 8, 9),
        spacing_xyz=(0.7, 1.3, 4.0),
        origin_lps_xyz=(2.0, -3.0, 10.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, cosine, -sine, 0.0, sine, cosine),
    )

    table = slice_geometry_table(geometry)

    assert table["in_plane_pixel_area_mm2"].iloc[0] == pytest.approx(0.91)
    assert table["slice_thickness_normal_mm"].iloc[0] == pytest.approx(4.0)
    assert table["slice_thickness_superior_mm"].iloc[0] == pytest.approx(4.0 * cosine)
    assert np.diff(table["position_superior_mm"]).tolist() == pytest.approx(
        [4.0 * cosine, 4.0 * cosine]
    )


def test_xyz_shaped_array_is_rejected_at_the_simpleitk_zyx_boundary():
    geometry = make_geometry((2, 3, 4))
    with pytest.raises(ValueError, match=r"array_zyx shape \(2, 3, 4\)"):
        validate_array_zyx(np.zeros((4, 3, 2)), geometry, "input")


def test_prepared_slice_ids_do_not_claim_an_original_axis_after_orientation_repair():
    shape = (3, 8, 10)
    geometry = make_geometry(shape)
    image = np.zeros(shape, dtype=np.int16)
    tissue = np.zeros(shape, dtype=np.uint8)
    body_labels = np.zeros(shape, dtype=np.uint8)
    body_labels[:, 1:-1, 1:-1] = 1
    body = body_surface_from_totalsegmentator(body_labels, geometry)

    table = calculate_canonical_slice_measurements(
        image,
        tissue,
        geometry,
        {1: "SM"},
        body,
        MeasurementIdentity("case", "run", "analysis"),
        tissue_backend_id="synthetic",
        compartment_labels_zyx=tissue,
        compartment_label_schema={1: "SM"},
        orientation_changed=True,
    )

    assert "original_storage_index" not in table
    assert table["slice_index_zyx_z"].tolist() == [0, 1, 2]
    assert table["slice_id"].tolist() == [0, 1, 2]


def test_external_contours_exclude_holes_and_distinguish_all_from_primary_components():
    angle = math.radians(20)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    geometry = make_geometry(
        (1, 80, 100),
        spacing_xyz=(0.8, 1.4, 3.0),
        direction_lps=(cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine),
    )
    mask = np.zeros((80, 100), dtype=bool)
    mask[15:55, 20:70] = True
    mask[28:38, 35:48] = False
    mask[60:66, 80:88] = True
    filled = mask.copy()
    filled[28:38, 35:48] = True

    all_components = external_contour_measurement(
        mask,
        geometry,
        0,
        component_mode="all",
    )
    primary = external_contour_measurement(
        mask,
        geometry,
        0,
        component_mode="primary",
    )
    filled_primary = external_contour_measurement(
        filled,
        geometry,
        0,
        component_mode="primary",
    )

    assert all_components.component_count == 2
    assert primary.component_count == 2
    assert all_components.valid and primary.valid
    assert primary.perimeter_cm == pytest.approx(filled_primary.perimeter_cm, abs=0.05)
    assert all_components.perimeter_cm > primary.perimeter_cm


def test_external_rectangle_contour_matches_independent_physical_reference():
    geometry = make_geometry(
        (1, 60, 70),
        spacing_xyz=(0.8, 1.4, 3.0),
        direction_lps=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    mask = np.zeros((60, 70), dtype=bool)
    height_pixels = 24
    width_pixels = 32
    mask[10 : 10 + height_pixels, 15 : 15 + width_pixels] = True

    observed = external_contour_measurement(
        mask,
        geometry,
        0,
        component_mode="primary",
    )

    # The 0.5-level marching contour has straight spans between the outer
    # pixel centers and four half-pixel diagonal corner transitions.
    expected_mm = (
        2 * (width_pixels - 1) * geometry.spacing_xyz[0]
        + 2 * (height_pixels - 1) * geometry.spacing_xyz[1]
        + 2
        * math.hypot(
            geometry.spacing_xyz[0],
            geometry.spacing_xyz[1],
        )
    )
    assert observed.valid
    assert observed.perimeter_cm * 10 == pytest.approx(expected_mm, abs=0.5)


def test_oblique_anisotropic_elliptical_solid_matches_voxel_volume_reference():
    shape = (5, 96, 120)
    angle = math.radians(18)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    geometry = make_geometry(
        shape,
        spacing_xyz=(0.7, 1.2, 2.5),
        direction_lps=(1.0, 0.0, 0.0, 0.0, cosine, -sine, 0.0, sine, cosine),
    )
    y, x = np.ogrid[: shape[1], : shape[2]]
    ellipse_yx = (
        ((x - 60.0) / 31.0) ** 2 + ((y - 48.0) / 23.0) ** 2
    ) <= 1.0
    labels = np.broadcast_to(ellipse_yx, shape).astype(np.uint8).copy()
    image = np.full(shape, 35, dtype=np.int16)
    body_labels = np.zeros(shape, dtype=np.uint8)
    body_labels[:, 8:88, 10:110] = 1
    body = body_surface_from_totalsegmentator(body_labels, geometry)

    table = calculate_canonical_slice_measurements(
        image,
        labels,
        geometry,
        {1: "SM"},
        body,
        MeasurementIdentity("case", "run", "analysis"),
        tissue_backend_id="synthetic",
        compartment_labels_zyx=labels,
        compartment_label_schema={1: "SM"},
    )
    voxel_count_per_slice = int(np.count_nonzero(ellipse_yx))
    expected_area_cm2 = voxel_count_per_slice * geometry.in_plane_area_mm2 / 100.0
    assert table["sm_area_cm2"].to_numpy() == pytest.approx(
        np.full(shape[0], expected_area_cm2),
        rel=1e-12,
    )

    aggregate = aggregate_physical_range(
        table,
        float(table["slice_slab_inferior_mm"].min()),
        float(table["slice_slab_superior_mm"].max()),
    )
    expected_volume_cm3 = (
        int(np.count_nonzero(labels)) * geometry.voxel_volume_mm3 / 1000.0
    )
    assert aggregate["range_valid"]
    assert aggregate["sm_volume_cm3"] == pytest.approx(
        expected_volume_cm3,
        rel=1e-12,
    )


def test_slice_measurements_are_concise_and_preserve_native_and_total_vat_areas():
    shape = (2, 12, 14)
    geometry = make_geometry(shape, spacing_xyz=(1.0, 1.0, 2.0))
    image = np.zeros(shape, dtype=np.int16)
    labels = np.zeros(shape, dtype=np.uint8)
    body_labels = np.zeros(shape, dtype=np.uint8)
    body_labels[0, 2:10, 2:11] = 1
    body_labels[0, 4:8, 11:13] = 2
    labels[0, 4:8, 11:13] = 1
    labels[0, 3:5, 3:7] = 3
    labels[0, 5:7, 4:6] = 4
    labels[0, 7:9, 5:7] = 5
    labels[1, 5, 5] = 1
    image[labels == 1] = 40
    image[labels == 3] = -100
    image[labels == 4] = -90
    image[labels == 5] = -70
    body = body_surface_from_totalsegmentator(body_labels, geometry)

    table = calculate_canonical_slice_measurements(
        image,
        labels,
        geometry,
        {
            1: "SM",
            3: "SAT",
            4: "aVAT",
            5: "tVAT",
        },
        body,
        MeasurementIdentity("case", "run", "analysis"),
        tissue_backend_id="synthetic",
        compartment_labels_zyx=labels,
        compartment_label_schema={
            1: "SM",
            3: "SAT",
            4: "aVAT",
            5: "tVAT",
        },
    )

    first = table.iloc[0]
    assert first["sm_voxel_count"] == 8
    assert first["total_segmented_tissue_voxel_count"] == 24
    assert first["total_segmented_tissue_area_cm2"] == pytest.approx(0.24)
    assert first["total_vat_voxel_count"] == 8
    assert first["total_vat_area_cm2"] == pytest.approx(
        first["avat_area_cm2"] + first["tvat_area_cm2"]
    )
    assert first["total_vat_source"] == "avat_plus_tvat"
    assert first["sm_mean_hu"] == pytest.approx(40.0)
    assert "sm_fraction_body_area" not in table
    assert "body_outer_perimeter_cm" not in table
    second = table.iloc[1]
    assert not second["sm_area_valid"]
    assert second["sm_area_reason"] == "invalid_measurement"
    assert second["tissue_outside_body_voxel_count"] == 1
    assert not second["slice_measurement_valid"]


def test_body_surface_backends_are_explicit_and_never_substitute_each_other():
    shape = (4, 16, 18)
    geometry = make_geometry(shape)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[:, 3:13, 3:15] = 1
    labels[1:3, 5:9, 16:18] = 2
    upstream = body_surface_from_totalsegmentator(labels, geometry)
    image = np.full(shape, -1000, dtype=np.int16)
    image[:, 3:13, 3:15] = 30
    deterministic = deterministic_body_surface(
        image,
        geometry,
        min_component_volume_mm3=1,
    )
    derived = tissue_segmentation_envelope(
        labels,
        geometry,
        minimum_component_area_mm2=1,
        closing_radius_mm=2,
        smoothing_sigma_mm=1,
    )

    assert upstream.backend_id == TOTALSEGMENTATOR_BODY_BACKEND
    assert deterministic.backend_id == DETERMINISTIC_BODY_BACKEND
    assert derived.backend_id == TISSUE_ENVELOPE_BACKEND
    assert upstream.provenance["task_id"] == 299
    assert np.all(derived.body_mask_zyx[labels != 0])
    assert np.all(derived.trunk_mask_zyx <= derived.body_mask_zyx)
    assert np.all(derived.body_mask_zyx[labels == 2])
    assert not np.any(derived.trunk_mask_zyx[labels == 2])
    assert any(
        flag.code == "deterministic_body_surface_selected"
        for flag in deterministic.qc_flags
    )


def test_exact_range_overlap_weights_area_volume_and_hu_with_variable_thickness():
    slices = pd.DataFrame(
        {
            "slice_id": [0, 1, 2],
            "slice_slab_inferior_mm": [0.0, 2.0, 5.0],
            "slice_slab_superior_mm": [2.0, 5.0, 9.0],
            "slice_thickness_normal_mm": [2.0, 3.0, 4.0],
            "normal_mm_per_superior_mm": [1.0, 1.0, 1.0],
            "sm_area_cm2": [10.0, 20.0, 30.0],
            "sm_body_compartment_area_cm2": [10.0, 20.0, 30.0],
            "sm_trunk_compartment_area_cm2": [10.0, 20.0, 30.0],
            "body_area_cm2": [100.0, 100.0, 100.0],
            "trunk_area_cm2": [80.0, 80.0, 80.0],
            "sm_voxel_count": [2, 4, 8],
            "sm_mean_hu": [10.0, 20.0, 40.0],
        }
    )

    result = aggregate_physical_range(slices, 1.0, 7.0)

    assert result["range_valid"]
    assert result["coverage_fraction"] == pytest.approx(1.0)
    assert result["sm_mean_csa_cm2"] == pytest.approx((10 * 1 + 20 * 3 + 30 * 2) / 6)
    assert result["sm_volume_cm3"] == pytest.approx((10 * 1 + 20 * 3 + 30 * 2) / 10)
    assert result["sm_mean_hu"] == pytest.approx((10 * 1 + 20 * 4 + 40 * 4) / 9)
    assert result["contributing_slice_ids"] == [0, 1, 2]

    strict = aggregate_physical_range(slices, 0.0, 10.0)
    partial = aggregate_physical_range(slices, 0.0, 10.0, allow_partial=True)
    assert not strict["range_valid"]
    assert strict["range_missing_reason"] == "partial_fov"
    assert np.isnan(strict["sm_volume_cm3"])
    assert strict["sm_volume_cm3_reason"] == "partial_fov"
    assert partial["range_valid"]
    assert partial["coverage_fraction"] == pytest.approx(0.9)
    assert partial["sm_volume_cm3"] == pytest.approx(20.0)


def test_range_aggregation_honors_metric_specific_surface_and_tissue_validity():
    slices = pd.DataFrame(
        {
            "slice_id": [0, 1],
            "slice_slab_inferior_mm": [0.0, 2.0],
            "slice_slab_superior_mm": [2.0, 4.0],
            "slice_thickness_normal_mm": [2.0, 2.0],
            "normal_mm_per_superior_mm": [1.0, 1.0],
            "body_area_cm2": [100.0, 90.0],
            "body_area_valid": [True, False],
            "trunk_area_cm2": [80.0, 70.0],
            "trunk_area_valid": [True, False],
            "sm_area_cm2": [10.0, 20.0],
            "sm_area_valid": [True, True],
            "sm_body_compartment_area_cm2": [10.0, 20.0],
            "sm_trunk_compartment_area_cm2": [10.0, 20.0],
            "body_outer_perimeter_cm": [120.0, 115.0],
            "body_contour_valid": [True, False],
            "trunk_circumference_cm": [90.0, 85.0],
            "trunk_contour_valid": [True, False],
            "sm_voxel_count": [10, 20],
            "sm_mean_hu": [30.0, 40.0],
            "sm_hu_valid": [True, True],
        }
    )

    strict = aggregate_physical_range(slices, 0.0, 4.0)
    partial = aggregate_physical_range(slices, 0.0, 4.0, allow_partial=True)

    assert strict["range_valid"]
    assert strict["sm_mean_csa_cm2"] == pytest.approx(15.0)
    assert strict["sm_mean_hu"] == pytest.approx((30 * 10 + 40 * 20) / 30)
    assert not strict["body_mean_csa_cm2_valid"]
    assert not strict["trunk_mean_circumference_cm_valid"]
    assert np.isnan(strict["trunk_mean_circumference_cm"])
    assert partial["body_mean_csa_cm2"] == pytest.approx(100.0)
    assert partial["body_mean_csa_cm2_coverage_fraction"] == pytest.approx(0.5)
    assert partial["trunk_mean_circumference_cm"] == pytest.approx(90.0)


def test_strict_pooled_hu_rejects_invalid_nonempty_slices_but_not_empty_tissue():
    base = {
        "slice_id": [0, 1],
        "slice_slab_inferior_mm": [0.0, 2.0],
        "slice_slab_superior_mm": [2.0, 4.0],
        "slice_thickness_normal_mm": [2.0, 2.0],
        "normal_mm_per_superior_mm": [1.0, 1.0],
        "sm_area_cm2": [10.0, 20.0],
    }
    invalid_nonempty = pd.DataFrame(
        {
            **base,
            "sm_voxel_count": [10, 20],
            "sm_mean_hu": [30.0, 40.0],
            "sm_hu_valid": [True, False],
        }
    )

    strict = aggregate_physical_range(invalid_nonempty, 0.0, 4.0)
    partial = aggregate_physical_range(
        invalid_nonempty,
        0.0,
        4.0,
        allow_partial=True,
    )

    assert not strict["sm_mean_hu_valid"]
    assert strict["sm_mean_hu_reason"] == "invalid_measurement"
    assert np.isnan(strict["sm_mean_hu"])
    assert partial["sm_mean_hu_valid"]
    assert partial["sm_mean_hu"] == pytest.approx(30.0)
    assert partial["sm_mean_hu_coverage_fraction"] == pytest.approx(0.5)

    empty_second_slice = invalid_nonempty.copy()
    empty_second_slice.loc[1, "sm_voxel_count"] = 0
    empty_second_slice.loc[1, "sm_mean_hu"] = np.nan
    empty_second_slice.loc[1, "sm_hu_valid"] = False
    pooled = aggregate_physical_range(empty_second_slice, 0.0, 4.0)

    assert pooled["sm_mean_hu_valid"]
    assert pooled["sm_mean_hu"] == pytest.approx(30.0)
    assert pooled["sm_mean_hu_coverage_fraction"] == pytest.approx(0.5)


def test_fragmented_trunk_is_observed_but_ineligible_for_canonical_summaries():
    shape = (1, 14, 16)
    geometry = make_geometry(shape)
    image = np.zeros(shape, dtype=np.int16)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[0, 3:5, 3:5] = 1
    image[labels == 1] = 40
    body_labels = np.zeros(shape, dtype=np.uint8)
    body_labels[0, 2:8, 2:8] = 1
    body_labels[0, 10:12, 11:14] = 1
    body = body_surface_from_totalsegmentator(body_labels, geometry)

    table = calculate_canonical_slice_measurements(
        image,
        labels,
        geometry,
        {1: "SM"},
        body,
        MeasurementIdentity("case", "run", "analysis"),
        tissue_backend_id="synthetic",
        compartment_labels_zyx=labels,
        compartment_label_schema={1: "SM"},
    )

    row = table.iloc[0]
    assert row["trunk_voxel_count"] > 0
    assert row["trunk_area_cm2"] > 0
    assert row["trunk_mask_fragmented"]
    assert not row["trunk_area_valid"]
    assert row["trunk_area_reason"] == "fragmented_mask"
    assert not row["trunk_contour_valid"]
    assert row["trunk_contour_reason"] == "fragmented_mask"
    assert not row["slice_measurement_valid"]
    assert row["slice_qc_status"] == "review"


def test_internal_body_surface_gap_and_empty_end_slices_are_explicitly_invalid():
    shape = (5, 14, 16)
    geometry = make_geometry(shape)
    image = np.zeros(shape, dtype=np.int16)
    labels = np.zeros(shape, dtype=np.uint8)
    body_labels = np.zeros(shape, dtype=np.uint8)
    body_labels[1, 3:11, 3:13] = 1
    body_labels[3, 3:11, 3:13] = 1
    body = body_surface_from_totalsegmentator(body_labels, geometry)

    table = calculate_canonical_slice_measurements(
        image,
        labels,
        geometry,
        {1: "SM"},
        body,
        MeasurementIdentity("case", "run", "analysis"),
        tissue_backend_id="synthetic",
        compartment_labels_zyx=labels,
        compartment_label_schema={1: "SM"},
    ).set_index("slice_id")

    assert table.loc[2, "body_mask_internal_gap"]
    assert table.loc[2, "trunk_mask_internal_gap"]
    assert table.loc[2, "body_surface_invalid"]
    assert not table.loc[2, "trunk_area_valid"]
    assert table.loc[2, "trunk_area_reason"] == "empty_mask"
    assert not table.loc[2, "slice_measurement_valid"]
    assert table.loc[2, "slice_qc_status"] == "review"
    for end_slice in (0, 4):
        assert not table.loc[end_slice, "body_mask_internal_gap"]
        assert not table.loc[end_slice, "trunk_area_valid"]
        assert table.loc[end_slice, "trunk_area_cm2"] == pytest.approx(0.0)


def test_longitudinal_qc_flags_only_large_close_adjacent_jumps():
    slices = pd.DataFrame(
        {
            "position_superior_mm": [0.0, 3.0, 6.0, 30.0],
            "trunk_circumference_cm": [100.0, 102.0, 140.0, 80.0],
            "trunk_contour_valid": [True, True, True, True],
            "slice_qc_status": ["pass", "pass", "pass", "pass"],
        }
    )
    output = annotate_longitudinal_circumference_qc(
        slices,
        maximum_gap_mm=10,
        minimum_absolute_jump_cm=15,
        minimum_relative_jump=0.15,
    )

    assert output["trunk_circumference_jump_flag"].tolist() == [False, False, True, False]
    assert output["slice_qc_status"].tolist() == ["pass", "pass", "review", "pass"]


def test_landmark_adapter_uses_paired_lowest_rib_and_bilateral_iliac_crest():
    shape = (60, 20, 24)
    geometry = make_geometry(shape, spacing_xyz=(1.0, 1.0, 1.0))
    labels = np.zeros(shape, dtype=np.uint8)
    schema = {1: "rib_left_12", 2: "rib_right_12", 3: "hip_left", 4: "hip_right"}
    labels[40:43, 6:9, 4:7] = 1
    labels[42:45, 6:9, 16:19] = 2
    labels[10:21, 8:13, 3:8] = 3
    labels[12:23, 8:13, 15:20] = 4

    landmarks = landmarks_from_totalsegmentator(
        labels,
        geometry,
        schema,
        minimum_voxels=5,
        maximum_side_disagreement_mm=10,
    )

    assert landmarks.lowest_rib_inferior.valid
    assert landmarks.iliac_crest_superior.valid
    assert landmarks.lowest_rib_inferior.position_superior_mm == pytest.approx(40.5)
    assert landmarks.lowest_rib_inferior.uncertainty_mm == pytest.approx(1.0)
    assert landmarks.iliac_crest_superior.position_superior_mm == pytest.approx(21.5)
    assert landmarks.iliac_crest_superior.uncertainty_mm == pytest.approx(1.0)


def test_landmark_side_threshold_applies_to_full_bilateral_disagreement():
    shape = (80, 20, 24)
    geometry = make_geometry(shape, spacing_xyz=(1.0, 1.0, 1.0))
    labels = np.zeros(shape, dtype=np.uint8)
    schema = {1: "rib_left_12", 2: "rib_right_12", 3: "hip_left", 4: "hip_right"}
    labels[50:53, 6:9, 4:7] = 1
    labels[70:73, 6:9, 16:19] = 2
    labels[10:15, 8:13, 3:8] = 3
    labels[30:35, 8:13, 15:20] = 4

    landmarks = landmarks_from_totalsegmentator(
        labels,
        geometry,
        schema,
        minimum_voxels=5,
        maximum_side_disagreement_mm=15,
    )

    assert not landmarks.lowest_rib_inferior.valid
    assert landmarks.lowest_rib_inferior.reason == "uncertain_landmark"
    assert landmarks.lowest_rib_inferior.uncertainty_mm == pytest.approx(10.0)
    assert not landmarks.iliac_crest_superior.valid
    assert landmarks.iliac_crest_superior.reason == "uncertain_landmark"
