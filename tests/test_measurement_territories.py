import numpy as np
import pandas as pd
import pytest

from BodyComposition.measurement.aggregation import (
    aggregate_named_range,
    aggregate_physical_range,
    annotate_slice_anatomy,
    build_vertebra_table,
    derive_vertebral_extents,
    derive_vertebral_territories,
    select_l3_view,
)
from BodyComposition.measurement.contracts import (
    BINS_PER_VERTEBRAL_TERRITORY,
    MeasurementIdentity,
    VertebralExtent,
)
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import ExecutionStatus, VertebralResult


IDENTITY = MeasurementIdentity("case", "run", "analysis")


def make_geometry(shape_zyx, spacing_xyz=(1.0, 1.0, 1.0)):
    return ImageGeometry(
        size_xyz=tuple(reversed(shape_zyx)),
        spacing_xyz=spacing_xyz,
        origin_lps_xyz=(0.0, 0.0, 0.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def make_extent(
    level,
    inferior,
    superior,
    *,
    native_label,
    complete=True,
    valid=True,
    reason=None,
):
    centroid = (
        (inferior + superior) / 2
        if inferior is not None and superior is not None
        else None
    )
    return VertebralExtent(
        native_label=native_label,
        anatomical_label=level,
        inferior_mm=inferior,
        superior_mm=superior,
        centroid_lps_xyz=(0.0, 0.0, centroid) if centroid is not None else None,
        centroid_superior_mm=centroid,
        voxel_count=100,
        retained_voxel_count=100,
        removed_voxel_fraction=0.0,
        component_count=1,
        touches_fov=not complete,
        complete=complete,
        valid=valid,
        missing_reason=reason,
    )


def make_slice_table(lower_edges, upper_edges, *, area=None, hu=None):
    lower = np.asarray(lower_edges, dtype=float)
    upper = np.asarray(upper_edges, dtype=float)
    thickness = upper - lower
    center = (lower + upper) / 2
    area = np.full(len(lower), 10.0) if area is None else np.asarray(area, dtype=float)
    hu = np.full(len(lower), 30.0) if hu is None else np.asarray(hu, dtype=float)
    return pd.DataFrame(
        {
            **IDENTITY.as_columns(),
            "longitudinal_order": np.arange(len(lower), dtype=int),
            "slice_id": np.arange(len(lower), dtype=int),
            "slice_index_zyx_z": np.arange(len(lower), dtype=int),
            "position_superior_mm": center,
            "slice_center_lps_x_mm": np.zeros(len(lower)),
            "slice_center_lps_y_mm": np.zeros(len(lower)),
            "slice_center_lps_z_mm": center,
            "slice_normal_lps_x": np.zeros(len(lower)),
            "slice_normal_lps_y": np.zeros(len(lower)),
            "slice_normal_lps_z": np.ones(len(lower)),
            "slice_slab_inferior_mm": lower,
            "slice_slab_superior_mm": upper,
            "slice_thickness_normal_mm": thickness,
            "normal_mm_per_superior_mm": np.ones(len(lower)),
            "trunk_contour_valid": np.ones(len(lower), dtype=bool),
            "trunk_circumference_cm": np.full(len(lower), 100.0),
            "trunk_area_cm2": np.full(len(lower), 80.0),
            "trunk_area_valid": np.ones(len(lower), dtype=bool),
            "sm_area_cm2": area,
            "sm_area_valid": np.ones(len(lower), dtype=bool),
            "sm_voxel_count": np.full(len(lower), 10),
            "sm_mean_hu": hu,
            "sm_hu_valid": np.ones(len(lower), dtype=bool),
        }
    )


def test_vertebral_extents_use_body_labels_and_remove_isolated_voxels():
    shape = (20, 12, 14)
    geometry = make_geometry(shape)
    body = np.zeros(shape, dtype=np.uint8)
    body[5:10, 4:8, 5:9] = 15
    body[15, 6, 7] = 15
    whole = np.zeros(shape, dtype=np.uint8)
    whole[2:18, 2:10, 3:11] = 15
    result = VertebralResult(
        backend_id="synthetic",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=whole,
        vertebral_body_labels=body,
        label_schema={15: "L3"},
    )

    extent = derive_vertebral_extents(result, maximum_removed_fraction=0.05)["L3"]

    assert extent.valid and extent.complete
    assert extent.component_count == 2
    assert extent.removed_voxel_fraction == pytest.approx(1 / 81)
    assert extent.inferior_mm == pytest.approx(4.5)
    assert extent.superior_mm == pytest.approx(9.5)


def test_midpoint_territories_assign_intervertebral_slices_without_stretching():
    extents = {
        "L2": make_extent("L2", 50.0, 60.0, native_label=21),
        "L3": make_extent("L3", 30.0, 40.0, native_label=22),
        "L4": make_extent("L4", 10.0, 20.0, native_label=23),
    }
    territories = derive_vertebral_territories(extents)

    assert territories["L3"].inferior_mm == pytest.approx(25.0)
    assert territories["L3"].superior_mm == pytest.approx(45.0)
    assert territories["L3"].complete
    assert not territories["L2"].complete
    assert not territories["L4"].complete

    slices = make_slice_table([0, 20, 30, 40, 50, 60], [20, 30, 40, 50, 60, 70])
    annotated = annotate_slice_anatomy(slices, territories)
    assigned = annotated.set_index("slice_id")["assigned_vertebral_level"].to_dict()
    assert assigned[2] == "L3"
    assert assigned[3] == "L2"
    assert annotated.iloc[0]["vertebral_assignment_status"] == "assigned_partial_edge"
    assert annotated.iloc[-1]["vertebral_assignment_status"] == "unassigned_edge"


def test_midpoint_boundary_tie_is_stably_assigned_to_the_cranial_territory():
    extents = {
        "L2": make_extent("L2", 50.0, 60.0, native_label=21),
        "L3": make_extent("L3", 30.0, 40.0, native_label=22),
        "L4": make_extent("L4", 10.0, 20.0, native_label=23),
    }
    forward = derive_vertebral_territories(extents)
    reverse = derive_vertebral_territories(dict(reversed(list(extents.items()))))
    boundary_slice = make_slice_table([44.0], [46.0])

    forward_level = annotate_slice_anatomy(
        boundary_slice,
        forward,
    ).iloc[0]["assigned_vertebral_level"]
    reverse_level = annotate_slice_anatomy(
        boundary_slice,
        reverse,
    ).iloc[0]["assigned_vertebral_level"]

    assert forward_level == reverse_level == "L2"


def test_t13_and_l6_are_native_continuous_territories_and_sacrum_is_included():
    extents = {
        "T12": make_extent("T12", 90.0, 100.0, native_label=19),
        "T13": make_extent("T13", 70.0, 80.0, native_label=28),
        "L1": make_extent("L1", 50.0, 60.0, native_label=20),
        "L5": make_extent("L5", 30.0, 40.0, native_label=24),
        "L6": make_extent("L6", 10.0, 20.0, native_label=25),
        "SACRUM": make_extent("SACRUM", -20.0, 0.0, native_label=26),
    }
    territories = derive_vertebral_territories(extents)

    assert territories["T13"].complete
    assert territories["T13"].superior_mm == pytest.approx(85.0)
    assert territories["T13"].inferior_mm == pytest.approx(65.0)
    assert territories["L6"].complete
    assert territories["SACRUM"].complete
    assert territories["SACRUM"].inferior_mm == pytest.approx(-20.0)

    slices = make_slice_table(np.arange(-20.0, 110.0, 2.0), np.arange(-18.0, 112.0, 2.0))
    table = build_vertebra_table(slices, extents, territories, IDENTITY)
    assert set(table["vertebral_level"]) == set(extents)
    assert table.groupby("vertebral_level").size().eq(3).all()
    assert table.loc[table["vertebral_level"].isin(["T13", "L6"]), "anatomical_variant"].all()
    assert table.loc[table["vertebral_level"].eq("SACRUM"), "bin_valid"].all()


def test_sequence_gap_keeps_contour_continuous_but_marks_review():
    extents = {
        "T11": make_extent("T11", 70.0, 80.0, native_label=18),
        "L1": make_extent("L1", 50.0, 60.0, native_label=20),
        "L2": make_extent("L2", 30.0, 40.0, native_label=21),
    }
    territories = derive_vertebral_territories(extents)
    assert territories["L1"].sequence_gap_cranial
    assert territories["L1"].superior_mm == pytest.approx(65.0)
    slices = make_slice_table([40.0, 50.0, 60.0], [50.0, 60.0, 70.0])
    annotated = annotate_slice_anatomy(slices, territories)
    assert annotated.iloc[1]["assigned_vertebral_level"] == "L1"
    assert "sequence_gap" in annotated.iloc[1]["vertebral_assignment_status"]


def test_three_bins_split_thick_slices_exactly_and_reconstruct_volume():
    extents = {
        "L2": make_extent("L2", 40.0, 50.0, native_label=21),
        "L3": make_extent("L3", 20.0, 30.0, native_label=22),
        "L4": make_extent("L4", 0.0, 10.0, native_label=23),
    }
    territories = derive_vertebral_territories(extents)
    slices = make_slice_table(
        [10.0, 19.0, 31.0, 40.0],
        [19.0, 31.0, 40.0, 49.0],
        area=[10.0, 20.0, 30.0, 40.0],
        hu=[10.0, 20.0, 30.0, 40.0],
    )

    table = build_vertebra_table(slices, extents, territories, IDENTITY)
    l3 = table.loc[table["vertebral_level"].eq("L3")]
    assert l3["territory_bin"].tolist() == [1, 2, 3]
    assert l3["bin_valid"].all()
    assert l3["bin_height_mm"].tolist() == pytest.approx([20 / 3] * 3)
    assert l3["bin_integration_length_mm"].tolist() == pytest.approx(
        [20 / 3] * 3
    )

    whole = aggregate_physical_range(slices, 15.0, 35.0)
    reconstructed = float(
        np.sum(l3["sm_mean_csa_cm2"] * l3["bin_height_mm"] / 10.0)
    )
    assert reconstructed == pytest.approx(whole["sm_volume_cm3"])


def test_oblique_bin_volume_reconstruction_uses_integration_length():
    extents = {
        "L2": make_extent("L2", 40.0, 50.0, native_label=21),
        "L3": make_extent("L3", 20.0, 30.0, native_label=22),
        "L4": make_extent("L4", 0.0, 10.0, native_label=23),
    }
    territories = derive_vertebral_territories(extents)
    slices = make_slice_table(
        np.arange(0.0, 50.0, 2.0),
        np.arange(2.0, 52.0, 2.0),
        area=np.linspace(10.0, 30.0, 25),
    )
    slices["normal_mm_per_superior_mm"] = 1.25
    slices["slice_thickness_normal_mm"] *= 1.25

    table = build_vertebra_table(slices, extents, territories, IDENTITY)
    l3 = table.loc[table["vertebral_level"].eq("L3")]
    whole = aggregate_physical_range(slices, 15.0, 35.0)
    reconstructed = float(
        np.sum(
            l3["sm_mean_csa_cm2"]
            * l3["bin_integration_length_mm"]
            / 10.0
        )
    )

    assert l3["bin_integration_length_mm"].tolist() == pytest.approx(
        [25 / 3] * 3
    )
    assert reconstructed == pytest.approx(whole["sm_volume_cm3"])


def test_equal_height_slices_reduce_to_arithmetic_mean_and_variable_height_is_weighted():
    equal = make_slice_table([0, 2, 4], [2, 4, 6], area=[10, 20, 40])
    variable = make_slice_table([0, 1, 4], [1, 4, 6], area=[10, 20, 40])

    equal_result = aggregate_physical_range(equal, 0.0, 6.0)
    variable_result = aggregate_physical_range(variable, 0.0, 6.0)

    assert equal_result["sm_mean_csa_cm2"] == pytest.approx(np.mean([10, 20, 40]))
    assert variable_result["sm_mean_csa_cm2"] == pytest.approx((10 + 60 + 80) / 6)


def test_partial_fov_edge_territories_have_null_bins_but_internal_level_remains_valid():
    extents = {
        "L2": make_extent("L2", 50.0, 60.0, native_label=21),
        "L3": make_extent("L3", 30.0, 40.0, native_label=22),
        "L4": make_extent("L4", 10.0, 20.0, native_label=23),
    }
    territories = derive_vertebral_territories(extents)
    slices = make_slice_table(np.arange(10.0, 60.0, 2.0), np.arange(12.0, 62.0, 2.0))
    table = build_vertebra_table(slices, extents, territories, IDENTITY)

    assert table.loc[table["vertebral_level"].eq("L3"), "bin_valid"].all()
    edge = table.loc[table["vertebral_level"].isin(["L2", "L4"])]
    assert not edge["bin_valid"].any()
    assert set(edge["bin_missing_reason"]) == {"missing_neighbor"}


def test_l3_centroid_slice_uses_full_point_to_plane_distance():
    angle = np.deg2rad(30.0)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    slices = make_slice_table([0, 4, 8, 12, 16], [4, 8, 12, 16, 20])
    slices["slice_center_lps_y_mm"] = [0, -2, -4, -6, -8]
    slices["slice_center_lps_z_mm"] = [0, 3, 6, 9, 12]
    slices["slice_normal_lps_y"] = -sine
    slices["slice_normal_lps_z"] = cosine
    extents = {
        "L2": make_extent("L2", 14.0, 18.0, native_label=21),
        "L3": make_extent("L3", 8.0, 12.0, native_label=22),
        "L4": make_extent("L4", 2.0, 6.0, native_label=23),
    }
    extents["L3"] = VertebralExtent(
        **{
            **extents["L3"].__dict__,
            "centroid_lps_xyz": (0.0, -4.0, 6.0),
        }
    )
    territories = derive_vertebral_territories(extents)
    vertebrae = build_vertebra_table(slices, extents, territories, IDENTITY)

    view = select_l3_view(slices, vertebrae, aggregation="slice")

    assert view.iloc[0]["slice_id"] == 2
    assert view.iloc[0]["centroid_slice_plane_distance_mm"] == pytest.approx(0.0)


def test_named_range_uses_territory_bounds_and_explicit_partial_mode():
    extents = {
        "L2": make_extent("L2", 40.0, 50.0, native_label=21),
        "L3": make_extent("L3", 20.0, 30.0, native_label=22),
        "L4": make_extent("L4", 0.0, 10.0, native_label=23),
    }
    territories = derive_vertebral_territories(extents)
    slices = make_slice_table(np.arange(0.0, 50.0, 2.0), np.arange(2.0, 52.0, 2.0))

    strict = aggregate_named_range(slices, territories, "L2", "L4")
    partial = aggregate_named_range(
        slices,
        territories,
        "L2",
        "L4",
        allow_partial=True,
    )

    assert not strict["range_valid"]
    assert strict["range_missing_reason"] == "missing_anchor"
    assert partial["range_valid"]
    assert partial["allow_partial"]
    assert BINS_PER_VERTEBRAL_TERRITORY == 3
