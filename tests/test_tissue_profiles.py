from pathlib import Path

import numpy as np
import pytest
import yaml

from BodyComposition.config import (
    CANONICAL_TISSUE_DEFINITIONS,
    CONSENSUS_TISSUE_PROFILE_ID,
    PipelineConfig,
)
from BodyComposition.measurement.aggregation import aggregate_physical_range
from BodyComposition.measurement.contracts import (
    BodySurfaceResult,
    MeasurementIdentity,
)
from BodyComposition.measurement.slices import calculate_canonical_slice_measurements
from BodyComposition.measurement.tissues import (
    derive_configured_tissue_masks,
    iter_configured_tissue_masks,
)
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.utils.masks import fill_small_holes, remove_small_objects


def test_every_shipped_tissue_profile_is_a_valid_downstream_override():
    profiles = sorted(Path("config/tissue_profiles").glob("*.yaml"))
    assert profiles
    profile_ids = set()
    for profile in profiles:
        override = yaml.safe_load(profile.read_text(encoding="utf-8"))
        config = PipelineConfig.model_validate(override)
        normalized = config.normalized()
        profile_id = normalized["measurements"]["tissue_profile_id"]
        assert profile_id not in profile_ids
        profile_ids.add(profile_id)
        assert normalized["tissue"] == {
            "backend": "bodycomposition_resenc_l_v1"
        }
        for name, definition in CANONICAL_TISSUE_DEFINITIONS.items():
            assert normalized["measurements"]["tissue_definitions"][name] == definition


def test_default_is_the_consensus_profile():
    default = PipelineConfig.model_validate({}).normalized()

    assert (
        default["measurements"]["tissue_profile_id"]
        == CONSENSUS_TISSUE_PROFILE_ID
    )


def test_native_and_consensus_visualization_label_sets_are_stable(base_config):
    expected = {
        1: "SM",
        2: "BONE",
        3: "SAT",
        4: "aVAT",
        5: "tVAT",
        6: "HEART",
        7: "LUNG",
    }
    assert base_config["LBL_TISSUE_COMPARTMENTS"] == expected
    assert base_config["LBL_TISSUE"] == expected

    for profile in Path("config/tissue_profiles").glob("*.yaml"):
        runtime = PipelineConfig.load(profile).to_runtime_dict()
        assert runtime["LBL_TISSUE_COMPARTMENTS"] == expected
        assert runtime["LBL_TISSUE"] == expected


def test_unknown_label_is_rejected_as_a_model_native_compartment(base_config):
    image = np.full((1, 1, 1), -100, dtype=np.int16)
    invalid_compartments = np.full((1, 1, 1), 8, dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match="outside the model-native schema: \\[8\\]",
    ):
        derive_configured_tissue_masks(
            image,
            invalid_compartments,
            base_config["LBL_TISSUE_COMPARTMENTS"],
            base_config["measurements"]["tissue_definitions"],
        )


def test_pixel_component_threshold_and_connectivity_are_reproducible():
    diagonal = np.zeros((1, 5, 5), dtype=bool)
    diagonal[0, 1, 1] = True
    diagonal[0, 2, 2] = True

    four_connected = diagonal.copy()
    remove_small_objects(
        four_connected,
        (0.8, 1.2, 3.0),
        "2D",
        limit_size_2D=2,
        size_unit="voxel",
        connectivity=4,
    )
    assert not four_connected.any()

    eight_connected = diagonal.copy()
    remove_small_objects(
        eight_connected,
        (0.8, 1.2, 3.0),
        "2D",
        limit_size_2D=2,
        size_unit="voxel",
        connectivity=8,
    )
    assert np.array_equal(eight_connected, diagonal)


def test_hole_filling_is_bounded_and_does_not_fill_exterior_background():
    mask = np.zeros((1, 7, 7), dtype=bool)
    mask[0, 1:6, 1:6] = True
    mask[0, 3, 3] = False
    mask[0, 1, 3] = False

    fill_small_holes(
        mask,
        (1.0, 1.0, 2.0),
        "2D",
        limit_size_2D=2,
        size_unit="voxel",
        connectivity=8,
    )

    assert mask[0, 3, 3]
    assert not mask[0, 1, 3]


def test_definition_cleanup_never_fills_outside_anatomical_support():
    image = np.full((1, 7, 7), -100, dtype=np.int16)
    support = np.zeros((1, 7, 7), dtype=np.uint8)
    support[0, 1:6, 1:6] = 1
    support[0, 3, 3] = 0
    definitions = {
        "test_hole_fill": {
            "enabled": True,
            "source_labels": ["SM"],
            "hu_range": [-190, -30],
            "cleanup": {
                "fill_small_holes": {
                    "dimensionality": "2D",
                    "threshold": 2,
                    "unit": "voxel",
                    "connectivity": 8,
                }
            },
        }
    }

    mask = derive_configured_tissue_masks(
        image,
        support,
        {1: "SM"},
        definitions,
        spacing_xyz=(1.0, 1.0, 2.0),
    )["test_hole_fill"]

    assert not mask[0, 3, 3]
    assert np.all(mask <= (support == 1))


@pytest.mark.parametrize(
    "profile_name",
    [
        "boa_median_3x3.yaml",
        "ahmad_2023_adaptive_median_sensitivity.yaml",
        "lee_2018_fat_sensitivity.yaml",
    ],
)
def test_profile_preprocessing_is_downstream_and_does_not_mutate_raw_ct(
    profile_name,
):
    config = PipelineConfig.load(
        Path("config/tissue_profiles") / profile_name
    ).to_runtime_dict()
    image = np.zeros((1, 9, 9), dtype=np.int16)
    image[0, 4, 4] = 500
    original = image.copy()
    compartments = np.ones(image.shape, dtype=np.uint8)

    definitions = {
        name: definition
        for name, definition in config["measurements"]["tissue_definitions"].items()
        if "preprocessing" in definition
        and "SM" in definition["source_labels"]
    }
    if definitions:
        derived = derive_configured_tissue_masks(
            image,
            compartments,
            {1: "SM"},
            definitions,
            spacing_xyz=(0.8, 0.8, 3.0),
        )
        assert all(mask.shape == image.shape for mask in derived.values())

    assert np.array_equal(image, original)


def test_consensus_phenotypes_use_raw_compartments_and_raw_hu(base_config):
    shape = (1, 10, 10)
    geometry = ImageGeometry(
        size_xyz=tuple(reversed(shape)),
        spacing_xyz=(1.0, 1.0, 2.0),
        origin_lps_xyz=(0.0, 0.0, 0.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    image = np.zeros(shape, dtype=np.int16)
    compartments = np.zeros(shape, dtype=np.uint8)

    muscle_points = [
        (3, 2, -100),
        (3, 3, 0),
        (3, 4, 40),
        (3, 5, 160),
        (4, 2, -80),
    ]
    for y, x, hu in muscle_points:
        compartments[0, y, x] = 1
        image[0, y, x] = hu
    compartments[0, 5, 2] = 3
    image[0, 5, 2] = -100
    compartments[0, 6, 2:4] = 4
    image[0, 6, 2] = -100
    image[0, 6, 3] = -40
    compartments[0, 7, 2:4] = 2
    image[0, 7, 2] = 200
    image[0, 7, 3] = 100

    body = np.ones(shape, dtype=bool)
    trunk = np.zeros(shape, dtype=bool)
    trunk[:, 1:-1, 1:-1] = True
    body_surface = BodySurfaceResult(
        backend_id="synthetic",
        geometry=geometry,
        body_mask_zyx=body,
        trunk_mask_zyx=trunk,
    )
    table = calculate_canonical_slice_measurements(
        image,
        compartments,
        geometry,
        base_config["LBL_TISSUE_COMPARTMENTS"],
        body_surface,
        MeasurementIdentity("case", "run", "analysis"),
        compartment_labels_zyx=compartments,
        compartment_label_schema=base_config["LBL_TISSUE_COMPARTMENTS"],
        tissue_definitions=base_config["measurements"]["tissue_definitions"],
    )
    row = table.iloc[0]

    assert row["sm_voxel_count"] == 5
    assert row["bone_voxel_count"] == 2
    assert row["skeletal_muscle_tissue_hu_m29_150_voxel_count"] == 2
    assert row["sat_total_hu_m190_m30_voxel_count"] == 1
    assert row["avat_hu_m190_m30_voxel_count"] == 2
    assert row["tvat_hu_m190_m30_voxel_count"] == 0
    assert row["vat_total_hu_m190_m30_voxel_count"] == 2
    assert row["sm_mean_hu"] == pytest.approx(4.0)
    assert "imat_hu_m190_m30_voxel_count" not in table
    assert "bone_tissue_hu_152_1000_voxel_count" not in table
    assert "vat_total_hu_m150_m50_voxel_count" not in table

    aggregate = aggregate_physical_range(
        table,
        float(table["slice_slab_inferior_mm"].iloc[0]),
        float(table["slice_slab_superior_mm"].iloc[0]),
    )
    assert aggregate["vat_to_sat_ratio_hu_m190_m30"] == pytest.approx(2.0)


def test_literature_window_profile_adds_a_named_definition(base_config):
    optional = PipelineConfig.load(
        Path("config/tissue_profiles/derstine_2022_vat_m150_m50.yaml")
    ).to_runtime_dict()
    image = np.array([[[-170, -100, -45]]], dtype=np.int16)
    labels = np.full(image.shape, 4, dtype=np.uint8)

    masks = derive_configured_tissue_masks(
        image,
        labels,
        base_config["LBL_TISSUE_COMPARTMENTS"],
        optional["measurements"]["tissue_definitions"],
        spacing_xyz=(1.0, 1.0, 1.0),
    )

    assert masks["vat_total_hu_m190_m30"].sum() == 3
    assert masks["vat_total_hu_m150_m50"].sum() == 1
    streamed = dict(
        iter_configured_tissue_masks(
            image,
            labels,
            base_config["LBL_TISSUE_COMPARTMENTS"],
            optional["measurements"]["tissue_definitions"],
            spacing_xyz=(1.0, 1.0, 1.0),
        )
    )
    assert streamed.keys() == masks.keys()
    assert all(np.array_equal(streamed[name], masks[name]) for name in masks)


def test_generic_derivation_accepts_validated_anatomical_subcompartments():
    image = np.full((1, 2, 4), -100, dtype=np.int16)
    labels = np.array([[[1, 2, 3, 4], [0, 0, 0, 0]]], dtype=np.uint8)
    definitions = {
        "ssat_hu_m190_m30": {
            "enabled": True,
            "source_labels": ["SSAT"],
            "hu_range": [-190, -30],
        },
        "rpat_hu_m190_m30": {
            "enabled": True,
            "source_labels": ["RPAT"],
            "hu_range": [-190, -30],
        },
    }
    output = derive_configured_tissue_masks(
        image,
        labels,
        {1: "SSAT", 2: "DSAT", 3: "IPAT", 4: "RPAT"},
        definitions,
    )

    assert output["ssat_hu_m190_m30"].sum() == 1
    assert output["rpat_hu_m190_m30"].sum() == 1
