from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from BodyComposition.measurement.aggregation import aggregate_physical_range
from BodyComposition.measurement.contracts import BodySurfaceResult, MeasurementIdentity
from BodyComposition.measurement.slices import calculate_canonical_slice_measurements
from BodyComposition.measurement.tissues import derive_configured_tissue_masks
from BodyComposition.tissue import cleanup_tissue_mask, prepare_classification_images
from BodyComposition.utils.config import update_config, validate_config
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.utils.masks import fill_small_holes, remove_small_objects


def test_every_shipped_tissue_profile_is_a_valid_override():
    profiles = sorted(Path("config/tissue_profiles").glob("*.yaml"))
    assert profiles
    profile_ids = set()
    for profile in profiles:
        config = update_config({}, Path("config/config.yaml"))
        config = update_config(config, Path("config/labels.yaml"))
        config = update_config(config, profile)
        validate_config(config)
        profile_id = config["tissue"]["profile_id"]
        assert profile_id not in profile_ids
        profile_ids.add(profile_id)


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
    mask[0, 1, 3] = False  # connects to exterior through the top background

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


def test_cleanup_never_fills_outside_anatomical_support(base_config):
    support = np.zeros((1, 7, 7), dtype=bool)
    support[0, 1:6, 1:6] = True
    support[0, 3, 3] = False
    mask = support.copy()
    settings = deepcopy(base_config["tissue"]["imat"])
    settings.update(
        {
            "filter_size": False,
            "fill_holes": True,
            "fill_holes_version": "2D",
            "fill_holes_2D": 2,
            "fill_holes_unit": "voxel",
            "fill_holes_connectivity": 8,
        }
    )

    cleanup_tissue_mask(
        mask,
        settings,
        (1.0, 1.0, 2.0),
        support_mask=support,
    )

    assert not mask[0, 3, 3]
    assert np.all(mask <= support)


@pytest.mark.parametrize("method", ["adaptive_median", "curvature_anisotropic_diffusion"])
def test_optional_noise_methods_are_selectable_without_mutating_raw_hu(
    base_config,
    method,
):
    settings = deepcopy(base_config["tissue"]["hu_denoise"])
    settings.update(
        {
            "method": method,
            "apply_to": ["vat"],
            "filter_median": False,
        }
    )
    image = np.zeros((1, 9, 9), dtype=np.int16)
    image[0, 4, 4] = 500
    original = image.copy()

    classified = prepare_classification_images(
        image,
        (0.8, 0.8, 3.0),
        settings,
    )

    assert np.array_equal(image, original)
    assert np.shares_memory(classified["sm"], image)
    assert classified["vat"].shape == image.shape
    assert np.isfinite(classified["vat"]).all()


def test_default_phenotypes_are_derived_from_raw_compartments_and_raw_hu(base_config):
    shape = (1, 10, 10)
    geometry = ImageGeometry(
        size_xyz=tuple(reversed(shape)),
        spacing_xyz=(1.0, 1.0, 2.0),
        origin_lps_xyz=(0.0, 0.0, 0.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    image = np.zeros(shape, dtype=np.int16)
    compartments = np.zeros(shape, dtype=np.uint8)
    postprocessed = np.zeros(shape, dtype=np.uint8)

    muscle_points = [(3, 2, -100), (3, 3, 0), (3, 4, 40), (3, 5, 160)]
    for y, x, hu in muscle_points:
        compartments[0, y, x] = 1
        image[0, y, x] = hu
    compartments[0, 4, 2] = 8
    image[0, 4, 2] = -80
    compartments[0, 5, 2] = 3
    image[0, 5, 2] = -100
    compartments[0, 6, 2:4] = 4
    image[0, 6, 2] = -100
    image[0, 6, 3] = -40
    compartments[0, 7, 2:4] = 2
    image[0, 7, 2] = 200
    image[0, 7, 3] = 100

    postprocessed[0, 3, 2] = 8
    postprocessed[0, 3, 3:5] = 1
    postprocessed[0, 4, 2] = 8
    postprocessed[0, 5, 2] = 3
    postprocessed[0, 6, 2:4] = 4

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
        postprocessed,
        geometry,
        base_config["LBL_TISSUE"],
        body_surface,
        MeasurementIdentity("case", "run", "analysis"),
        tissue_backend_id="synthetic",
        compartment_labels_zyx=compartments,
        compartment_label_schema=base_config["LBL_TISSUE"],
        tissue_definitions=base_config["measurements"]["tissue_definitions"],
    )
    row = table.iloc[0]

    assert row["muscle_compartment_voxel_count"] == 5
    assert row["learned_imat_voxel_count"] == 1
    assert row["whole_bone_anatomical_voxel_count"] == 2
    assert row["bone_tissue_hu_152_1000_voxel_count"] == 1
    assert row["skeletal_muscle_tissue_hu_m29_150_voxel_count"] == 2
    assert row["lama_hu_m29_29_voxel_count"] == 1
    assert row["nama_hu_30_150_voxel_count"] == 1
    assert row["imat_ct_hu_m190_m30_voxel_count"] == 2
    assert row["vat_total_hu_m190_m30_voxel_count"] == 2
    assert row["vat_total_hu_m150_m50_voxel_count"] == 1
    assert row["muscle_compartment_mean_hu"] == pytest.approx(4.0)
    assert row["lama_fraction_of_skeletal_muscle_tissue_hu_m29_150"] == pytest.approx(0.5)
    assert row["imat_fraction_of_sm_plus_imat_hu_m190_m30"] == pytest.approx(0.5)

    aggregate = aggregate_physical_range(
        table,
        float(table["slice_slab_inferior_mm"].iloc[0]),
        float(table["slice_slab_superior_mm"].iloc[0]),
    )
    assert aggregate["lama_fraction_of_skeletal_muscle_tissue_hu_m29_150"] == pytest.approx(0.5)
    assert aggregate["vat_to_sat_ratio_hu_m190_m30"] == pytest.approx(2.0)


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
