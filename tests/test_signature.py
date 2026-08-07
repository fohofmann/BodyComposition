from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from BodyComposition.measurement.contracts import (
    SIGNATURE_CORE_CHANNELS,
    MeasurementIdentity,
    VertebralExtent,
    VertebralTerritory,
    validate_signature_contract,
)
from BodyComposition.measurement.signature import (
    _dominant_territory,
    build_longitudinal_signature,
    estimate_reference_alignment,
)


def _extent(level: str, position_superior_mm: float) -> VertebralExtent:
    return VertebralExtent(
        native_label=1,
        anatomical_label=level,
        inferior_mm=position_superior_mm - 5.0,
        superior_mm=position_superior_mm + 5.0,
        centroid_lps_xyz=(0.0, 0.0, position_superior_mm),
        centroid_superior_mm=position_superior_mm,
        voxel_count=100,
        retained_voxel_count=100,
        removed_voxel_fraction=0.0,
        component_count=1,
        touches_fov=False,
        complete=True,
        valid=True,
    )


def _territory(level: str, position_superior_mm: float) -> VertebralTerritory:
    return VertebralTerritory(
        native_label=15,
        anatomical_label=level,
        inferior_mm=position_superior_mm - 20.0,
        superior_mm=position_superior_mm + 20.0,
        centroid_lps_xyz=(0.0, 0.0, position_superior_mm),
        centroid_superior_mm=position_superior_mm,
        extent_valid=True,
        complete=True,
    )


def _signature_slices(position_offset_mm: float) -> pd.DataFrame:
    centres = np.asarray([-15.0, -5.0, 5.0, 15.0]) + position_offset_mm
    table = pd.DataFrame(
        {
            "slice_id": np.arange(4),
            "slice_slab_inferior_mm": centres - 5.0,
            "slice_slab_superior_mm": centres + 5.0,
            "slice_spacing_normal_mm": np.full(4, 10.0),
            "normal_mm_per_superior_mm": np.ones(4),
            "trunk_area_cm2": np.full(4, 100.0),
            "trunk_area_valid": np.ones(4, dtype=bool),
            "trunk_circumference_cm": np.full(4, 80.0),
            "trunk_contour_valid": np.ones(4, dtype=bool),
        }
    )
    for _, source in SIGNATURE_CORE_CHANNELS:
        if source == "sm_compartment":
            area = np.asarray([10.0, 20.0, 30.0, 40.0])
            mean_hu = np.asarray([20.0, 30.0, 40.0, 50.0])
            count = np.asarray([10, 20, 30, 40])
            valid = np.ones(4, dtype=bool)
        else:
            area = np.zeros(4)
            mean_hu = np.full(4, np.nan)
            count = np.zeros(4, dtype=int)
            valid = np.zeros(4, dtype=bool)
        table[f"{source}_area_cm2"] = area
        table[f"{source}_area_valid"] = True
        table[f"{source}_mean_hu"] = mean_hu
        table[f"{source}_hu_valid"] = valid
        table[f"{source}_voxel_count"] = count
    return table


def test_default_signature_channels_preserve_public_semantics():
    assert SIGNATURE_CORE_CHANNELS == (
        ("sm_compartment", "sm_compartment"),
        (
            "skeletal_muscle_tissue_hu_m29_150",
            "skeletal_muscle_tissue_hu_m29_150",
        ),
        ("sat_tissue_hu_m190_m30", "sat_tissue_hu_m190_m30"),
        ("avat_tissue_hu_m190_m30", "avat_tissue_hu_m190_m30"),
        ("tvat_tissue_hu_m190_m30", "tvat_tissue_hu_m190_m30"),
        ("bone_compartment", "bone_compartment"),
        ("heart_compartment", "heart_compartment"),
        ("lung_compartment", "lung_compartment"),
    )


def test_reference_alignment_uses_observed_l3_without_rescaling():
    alignment = estimate_reference_alignment(
        {
            "L2": _extent("L2", 130.0),
            "L3": _extent("L3", 100.0),
            "L4": _extent("L4", 70.0),
        },
        {},
    )

    assert alignment.valid
    assert alignment.method == "direct_l3"
    assert alignment.origin_position_superior_mm == pytest.approx(100.0)
    assert alignment.slope_mm_per_level is None
    assert not alignment.review_required


def test_direct_l3_keeps_native_variant_review_visible():
    alignment = estimate_reference_alignment(
        {
            "T13": _extent("T13", 220.0),
            "L3": _extent("L3", 100.0),
        },
        {},
    )

    assert alignment.method == "direct_l3"
    assert alignment.origin_position_superior_mm == pytest.approx(100.0)
    assert alignment.variant_sequence == "T13"
    assert alignment.review_required


def test_reference_alignment_infers_missing_l3_from_patient_centroids():
    # The synthetic subject has a measured 32-mm centroid pitch. The
    # alignment should recover only the missing origin; it must not replace
    # that pitch with a population-normalized scale.
    origin = 250.0
    slope = -32.0
    relative_indices = {
        "T10": -5,
        "T11": -4,
        "T12": -3,
        "L1": -2,
        "L2": -1,
    }
    extents = {
        level: _extent(level, origin + slope * relative_index)
        for level, relative_index in relative_indices.items()
    }

    alignment = estimate_reference_alignment(extents, {})

    assert alignment.valid
    assert alignment.method == "multi_anchor_inferred"
    assert alignment.origin_position_superior_mm == pytest.approx(origin)
    assert alignment.slope_mm_per_level == pytest.approx(slope)
    assert alignment.residual_mm == pytest.approx(0.0)
    assert alignment.confidence == "high"


def test_reference_alignment_keeps_t13_in_the_native_sequence():
    origin = 100.0
    slope = -30.0
    # With T13 present, it occupies a real level between T12 and L1. L3 is
    # therefore four native steps caudal to T12, not three.
    relative_indices = {
        "T12": -4,
        "T13": -3,
        "L1": -2,
        "L2": -1,
    }
    extents = {
        level: _extent(level, origin + slope * relative_index)
        for level, relative_index in relative_indices.items()
    }

    alignment = estimate_reference_alignment(extents, {})

    assert alignment.valid
    assert alignment.origin_position_superior_mm == pytest.approx(origin)
    assert alignment.slope_mm_per_level == pytest.approx(slope)
    assert alignment.variant_sequence == "T13"
    assert alignment.review_required


def test_reference_alignment_keeps_l6_in_the_native_sequence():
    origin = 100.0
    slope = -30.0
    extents = {
        level: _extent(level, origin + slope * relative_index)
        for level, relative_index in {"L4": 1, "L5": 2, "L6": 3}.items()
    }

    alignment = estimate_reference_alignment(extents, {})

    assert alignment.valid
    assert alignment.origin_position_superior_mm == pytest.approx(origin)
    assert alignment.anchor_levels == ("L4", "L5", "L6")
    assert alignment.variant_sequence == "L6"
    assert alignment.review_required


def test_distant_cervical_anchors_produce_a_reviewable_low_confidence_origin():
    origin = 100.0
    slope = -25.0
    relative_indices = {
        "C1": -21,
        "C2": -20,
        "C3": -19,
        "C4": -18,
    }
    extents = {
        level: _extent(level, origin + slope * relative_index)
        for level, relative_index in relative_indices.items()
    }

    alignment = estimate_reference_alignment(extents, {})

    assert alignment.valid
    assert alignment.origin_position_superior_mm == pytest.approx(origin)
    assert alignment.confidence == "low"
    assert alignment.review_required


def test_sacrum_is_an_annotation_but_not_a_latent_l3_alignment_anchor():
    alignment = estimate_reference_alignment(
        {"SACRUM": _extent("SACRUM", 50.0)},
        {"SACRUM": _territory("SACRUM", 50.0)},
    )

    assert not alignment.valid
    assert alignment.method == "unresolved"
    assert alignment.reason == "missing_anchor"
    assert alignment.anchor_levels == ()


def test_single_anchor_alignment_is_available_but_explicitly_low_confidence():
    alignment = estimate_reference_alignment(
        {"L2": _extent("L2", 130.0)},
        {},
    )

    assert alignment.valid
    assert alignment.method == "single_anchor_inferred"
    assert alignment.confidence == "low"
    assert alignment.review_required
    assert alignment.origin_position_superior_mm == pytest.approx(100.0)


def test_dominant_territory_tie_chooses_cranial_level_at_zero_coordinate():
    territories = {
        "L3": VertebralTerritory(
            native_label=15,
            anatomical_label="L3",
            inferior_mm=-15.0,
            superior_mm=-5.0,
            centroid_lps_xyz=(0.0, 0.0, -10.0),
            centroid_superior_mm=-10.0,
            extent_valid=True,
            complete=True,
        ),
        "L2": VertebralTerritory(
            native_label=14,
            anatomical_label="L2",
            inferior_mm=-5.0,
            superior_mm=5.0,
            centroid_lps_xyz=(0.0, 0.0, 0.0),
            centroid_superior_mm=0.0,
            extent_valid=True,
            complete=True,
        ),
    }

    level, fraction, status = _dominant_territory(
        -10.0,
        0.0,
        territories,
    )

    assert level == "L2"
    assert fraction == pytest.approx(0.5)
    assert status == "assigned"


def test_fixed_signature_removes_scanner_offset_without_stretching_profile():
    identity = MeasurementIdentity("case-001", "run-001", "analysis-001")

    signatures = []
    for offset in (0.0, 500.0):
        signature, alignment = build_longitudinal_signature(
            _signature_slices(offset),
            {"L3": _extent("L3", offset)},
            {"L3": _territory("L3", offset)},
            identity,
            tissue_profile_id="test_profile_v1",
            full_coverage_tolerance=0.999,
        )
        assert alignment.origin_position_superior_mm == pytest.approx(offset)
        signatures.append(signature)

    baseline, shifted = signatures
    for column in (
        "reference_inferior_mm",
        "reference_center_mm",
        "reference_superior_mm",
        "coverage_fraction",
        "sm_compartment_mean_csa_cm2",
        "sm_compartment_mean_hu",
        "sm_compartment_mean_csa_fraction_of_trunk",
    ):
        pd.testing.assert_series_equal(
            baseline[column],
            shifted[column],
            check_names=False,
        )
    np.testing.assert_allclose(
        shifted["physical_inferior_position_superior_mm"]
        - baseline["physical_inferior_position_superior_mm"],
        500.0,
        equal_nan=True,
    )
    assert baseline.loc[50, "sm_compartment_mean_csa_cm2"] == pytest.approx(25.0)
    assert baseline.loc[50, "sm_compartment_mean_hu"] == pytest.approx(36.0)


def test_signature_retains_observed_values_for_incomplete_dominant_territory():
    identity = MeasurementIdentity("case-001", "run-001", "analysis-001")
    territory = VertebralTerritory(
        native_label=15,
        anatomical_label="L3",
        inferior_mm=-20.0,
        superior_mm=20.0,
        centroid_lps_xyz=(0.0, 0.0, 0.0),
        centroid_superior_mm=0.0,
        extent_valid=True,
        complete=False,
        missing_reason="truncated_vertebra",
    )

    signature, _alignment = build_longitudinal_signature(
        _signature_slices(0.0),
        {"L3": _extent("L3", 0.0)},
        {"L3": territory},
        identity,
        tissue_profile_id="test_profile_v1",
        full_coverage_tolerance=0.999,
    )

    observed = signature.loc[50]
    assert observed["dominant_vertebral_level"] == "L3"
    assert observed["vertebral_assignment_status"] == "assigned_partial_edge"
    assert observed["sm_compartment_mean_csa_cm2_valid"]
    assert observed["sm_compartment_mean_csa_cm2"] == pytest.approx(25.0)
    assert observed["sm_compartment_mean_hu_valid"]
    assert observed["sm_compartment_mean_hu"] == pytest.approx(36.0)
    outside = signature.loc[52]
    assert not outside["bin_valid"]
    assert pd.isna(outside["sm_compartment_mean_csa_cm2"])
    assert outside["sm_compartment_mean_csa_cm2_reason"] == "outside_fov"


def test_signature_contract_rejects_non_translation_physical_bounds():
    identity = MeasurementIdentity("case-001", "run-001", "analysis-001")
    signature, _ = build_longitudinal_signature(
        _signature_slices(100.0),
        {"L3": _extent("L3", 100.0)},
        {"L3": _territory("L3", 100.0)},
        identity,
        tissue_profile_id="test_profile_v1",
        full_coverage_tolerance=0.999,
    )
    signature.loc[50, "physical_inferior_position_superior_mm"] += 1.0

    with pytest.raises(ValueError, match="translation-only reference"):
        validate_signature_contract(signature)


def test_trunk_fraction_is_invalid_when_numerator_and_denominator_coverage_differ():
    identity = MeasurementIdentity("case-001", "run-001", "analysis-001")
    slices = _signature_slices(0.0)
    slices.loc[1, "trunk_area_valid"] = False
    signature, _ = build_longitudinal_signature(
        slices,
        {"L3": _extent("L3", 0.0)},
        {"L3": _territory("L3", 0.0)},
        identity,
        tissue_profile_id="test_profile_v1",
        full_coverage_tolerance=0.999,
    )

    row = signature.loc[50]
    assert row["sm_compartment_mean_csa_cm2_valid"]
    assert row["trunk_mean_csa_cm2_valid"]
    assert row["sm_compartment_mean_csa_cm2_coverage_fraction"] != row["trunk_mean_csa_cm2_coverage_fraction"]
    assert not row["sm_compartment_mean_csa_fraction_of_trunk_valid"]
    assert row["sm_compartment_mean_csa_fraction_of_trunk_reason"] == "invalid_measurement"
