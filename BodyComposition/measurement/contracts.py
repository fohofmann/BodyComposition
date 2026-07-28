"""Stable data contracts for canonical measurement outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import QCFlag, QCStatus

MEASUREMENT_SCHEMA_VERSION = "3.2.0"
VERTEBRAL_TERRITORY_SCHEMA_VERSION = "native-physical-territories-v2-bins3"
BINS_PER_VERTEBRAL_TERRITORY = 3
SIGNATURE_SCHEMA_VERSION = "vertebral-reference-fixed-mm-v1"
REFERENCE_ALIGNMENT_VERSION = "robust-vertebral-linear-reference-v1"
SIGNATURE_BIN_COUNT = 100
SIGNATURE_BIN_WIDTH_MM = 20.0
SIGNATURE_REFERENCE_LEVEL = "L3"
TISSUE_HU_DISTRIBUTION_SCHEMA_VERSION = "fixed-bin-native-compartment-hu-v1"
HU_DISTRIBUTION_SCOPE = "analyzed_volume"
HU_DISTRIBUTION_MIN_HU = -190.0
HU_DISTRIBUTION_MAX_HU = 150.0
HU_DISTRIBUTION_BIN_WIDTH_HU = 5.0
HU_DISTRIBUTION_TISSUES: tuple[tuple[str, str, str], ...] = (
    ("sm", "SM", "sm"),
    ("sat", "SAT", "sat"),
    ("avat", "aVAT", "avat"),
    ("tvat", "tVAT", "tvat"),
)
SIGNATURE_CORE_CHANNELS: tuple[tuple[str, str], ...] = (
    ("sm", "sm"),
    (
        "skeletal_muscle_tissue_hu_m29_150",
        "skeletal_muscle_tissue_hu_m29_150",
    ),
    ("sat_total_hu_m190_m30", "sat_total_hu_m190_m30"),
    ("avat_hu_m190_m30", "avat_hu_m190_m30"),
    ("tvat_hu_m190_m30", "tvat_hu_m190_m30"),
    ("bone", "bone"),
    ("heart", "heart"),
    ("lung", "lung"),
)

SLICE_REQUIRED_COLUMNS = {
    "schema_version",
    "run_id",
    "analysis_id",
    "case_id",
    "slice_id",
    "longitudinal_order",
    "slice_index_zyx_z",
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
    "full_coverage_tolerance",
    "assigned_vertebral_level",
    "vertebral_assignment_status",
    "vertebral_territory_bin",
    "total_segmented_tissue_area_cm2",
    "trunk_area_cm2",
    "trunk_area_valid",
    "trunk_area_reason",
    "trunk_circumference_cm",
    "trunk_contour_valid",
    "trunk_contour_reason",
    "trunk_contour_closed",
    "body_touching_fov",
    "trunk_touching_fov",
    "trunk_mask_fragmented",
    "slice_measurement_valid",
    "slice_qc_status",
}
VERTEBRA_REQUIRED_COLUMNS = {
    "schema_version",
    "vertebral_territory_schema_version",
    "run_id",
    "analysis_id",
    "case_id",
    "native_label",
    "vertebral_level",
    "vertebral_extent_valid",
    "anatomical_variant",
    "territory_inferior_mm",
    "territory_superior_mm",
    "territory_height_mm",
    "territory_complete",
    "territory_reason",
    "territory_qc_status",
    "territory_bin",
    "bins_per_territory",
    "bin_inferior_mm",
    "bin_superior_mm",
    "bin_height_mm",
    "bin_integration_length_mm",
    "coverage_fraction",
    "bin_valid",
    "bin_missing_reason",
    "aggregation",
    "trunk_mean_circumference_cm_value_is_fov_cropped",
}
SUMMARY_REQUIRED_COLUMNS = {
    "schema_version",
    "run_id",
    "analysis_id",
    "case_id",
    "l3_200mm_slab_valid",
    "l3_200mm_slab_reason",
    "l3_territory_valid",
    "l3_territory_complete",
    "l3_territory_reason",
    "l3_territory_coverage_fraction",
    "ct_min_trunk_circumference_t10_l5_cm",
    "ct_min_trunk_circumference_t10_l5_valid",
    "ct_min_trunk_circumference_t10_l5_eligible",
    "ct_min_trunk_circumference_t10_l5_reason",
    "ct_min_trunk_circumference_t10_l5_position_superior_mm",
    "ct_min_trunk_circumference_t10_l5_slice_id",
    "ct_min_trunk_circumference_t10_l5_at_search_boundary",
    "ct_min_trunk_circumference_t10_l5_search_acquisition_coverage_fraction",
    "ct_min_trunk_circumference_t10_l5_search_valid_contour_coverage_fraction",
    "ct_min_trunk_circumference_t10_l5_search_anatomically_complete",
    "ct_midwaist_circumference_cm",
    "ct_midwaist_circumference_valid",
    "ct_midwaist_circumference_reason",
    "ct_midwaist_lowest_rib_inferior_position_superior_mm",
    "ct_midwaist_lowest_rib_inferior_valid",
    "ct_midwaist_lowest_rib_inferior_reason",
    "ct_midwaist_iliac_crest_superior_position_superior_mm",
    "ct_midwaist_iliac_crest_superior_valid",
    "ct_midwaist_iliac_crest_superior_reason",
    "ct_max_pelvic_circumference_cm",
    "ct_max_pelvic_circumference_valid",
    "ct_max_pelvic_circumference_eligible",
    "ct_max_pelvic_circumference_reason",
    "ct_max_pelvic_circumference_position_superior_mm",
    "ct_max_pelvic_circumference_slice_id",
    "ct_max_pelvic_circumference_at_search_boundary",
    "ct_max_pelvic_circumference_search_acquisition_coverage_fraction",
    "ct_max_pelvic_circumference_search_valid_contour_coverage_fraction",
    "ct_max_pelvic_circumference_search_anatomically_complete",
    "ct_max_pelvic_circumference_search_touches_fov",
    "ct_max_pelvic_circumference_value_is_fov_cropped",
    "ct_max_pelvic_search_definition",
    "ct_min_waist_to_pelvic_ratio",
    "ct_min_waist_to_pelvic_ratio_valid",
    "ct_min_waist_to_pelvic_ratio_eligible",
    "ct_min_waist_to_pelvic_ratio_reason",
    "ct_min_waist_to_pelvic_ratio_value_is_fov_cropped",
    "ct_midwaist_to_pelvic_ratio",
    "ct_midwaist_to_pelvic_ratio_valid",
    "ct_midwaist_to_pelvic_ratio_eligible",
    "ct_midwaist_to_pelvic_ratio_reason",
    "ct_midwaist_to_pelvic_ratio_value_is_fov_cropped",
    "body_surface_touches_fov",
    "body_surface_touches_fov_slice_count",
    "trunk_contour_touches_fov",
    "trunk_contour_touches_fov_slice_count",
}
SIGNATURE_REQUIRED_COLUMNS = {
    "schema_version",
    "signature_schema_version",
    "signature_profile_id",
    "reference_alignment_version",
    "run_id",
    "analysis_id",
    "case_id",
    "signature_bin",
    "relative_bin_index",
    "reference_inferior_mm",
    "reference_center_mm",
    "reference_superior_mm",
    "bin_width_mm",
    "physical_inferior_position_superior_mm",
    "physical_superior_position_superior_mm",
    "coverage_fraction",
    "contributing_slice_count",
    "bin_valid",
    "bin_reason",
    "dominant_vertebral_level",
    "dominant_vertebral_fraction",
    "vertebral_assignment_status",
    "reference_level",
    "reference_origin_position_superior_mm",
    "reference_alignment_valid",
    "reference_alignment_method",
    "reference_alignment_confidence",
    "reference_alignment_anchor_count",
    "reference_alignment_anchor_levels",
    "reference_alignment_slope_mm_per_level",
    "reference_alignment_residual_mm",
    "reference_alignment_review_required",
    "reference_variant_sequence",
    "trunk_mean_csa_cm2",
    "trunk_mean_csa_cm2_valid",
    "trunk_mean_csa_cm2_reason",
    "trunk_mean_csa_cm2_coverage_fraction",
    "trunk_mean_circumference_cm",
    "trunk_mean_circumference_cm_valid",
    "trunk_mean_circumference_cm_reason",
    "trunk_mean_circumference_cm_coverage_fraction",
}
HU_DISTRIBUTION_REQUIRED_COLUMNS = {
    "schema_version",
    "run_id",
    "analysis_id",
    "case_id",
    "distribution_schema_version",
    "distribution_scope",
    "tissue_key",
    "display_label",
    "source_compartment",
    "source_label_ids",
    "source_semantics",
    "histogram_min_hu",
    "histogram_max_hu",
    "bin_width_hu",
    "bin_semantics",
    "bin_index",
    "bin_lower_hu",
    "bin_upper_hu",
    "bin_upper_inclusive",
    "voxel_count",
    "voxel_fraction",
    "total_voxel_count",
    "finite_voxel_count",
    "nonfinite_voxel_count",
    "in_histogram_voxel_count",
    "below_histogram_voxel_count",
    "above_histogram_voxel_count",
    "below_histogram_voxel_fraction",
    "above_histogram_voxel_fraction",
    "contributing_slice_count",
    "voxel_volume_mm3",
    "total_volume_cm3",
    "distribution_valid",
    "distribution_reason",
    "mean_hu",
    "standard_deviation_hu",
    "q1_hu",
    "median_hu",
    "q3_hu",
    "quantile_method",
    "scope_inferior_position_superior_mm",
    "scope_superior_position_superior_mm",
}
for _destination, _ in SIGNATURE_CORE_CHANNELS:
    SIGNATURE_REQUIRED_COLUMNS.update(
        {
            f"{_destination}_mean_csa_cm2",
            f"{_destination}_mean_csa_cm2_valid",
            f"{_destination}_mean_csa_cm2_reason",
            f"{_destination}_mean_csa_cm2_coverage_fraction",
            f"{_destination}_mean_hu",
            f"{_destination}_mean_hu_valid",
            f"{_destination}_mean_hu_reason",
            f"{_destination}_mean_hu_coverage_fraction",
            f"{_destination}_mean_csa_fraction_of_trunk",
            f"{_destination}_mean_csa_fraction_of_trunk_valid",
            f"{_destination}_mean_csa_fraction_of_trunk_reason",
        }
    )
MISSING_REASONS = {
    "outside_fov",
    "partial_fov",
    "partial_contour_coverage",
    "missing_vertebra",
    "truncated_vertebra",
    "segmentation_failure",
    "invalid_extent",
    "invalid_measurement",
    "anatomical_variant",
    "missing_anchor",
    "missing_landmark",
    "uncertain_landmark",
    "boundary_extremum",
    "empty_tissue",
    "zero_denominator",
    "empty_mask",
    "contour_not_found",
    "open_contour",
    "touching_image_boundary",
    "unassigned_edge",
    "missing_neighbor",
    "sequence_gap",
    "definition_disabled",
    "missing_compartment",
}


_STRING_COLUMNS = {
    "schema_version",
    "vertebral_territory_schema_version",
    "signature_schema_version",
    "signature_profile_id",
    "reference_alignment_version",
    "distribution_schema_version",
    "distribution_scope",
    "tissue_key",
    "display_label",
    "tissue_definition",
    "source_compartment",
    "source_label_ids",
    "source_semantics",
    "bin_semantics",
    "quantile_method",
    "run_id",
    "analysis_id",
    "case_id",
    "assigned_vertebral_level",
    "vertebral_assignment_status",
    "tissue_preprocessing",
    "aggregation",
    "range_name",
    "start_level",
    "end_level",
    "total_vat_source",
    "orientation_state",
    "orientation_qc_status",
    "territory_qc_status",
    "reference_level",
    "reference_alignment_method",
    "reference_alignment_confidence",
    "reference_alignment_anchor_levels",
    "reference_variant_sequence",
    "dominant_vertebral_level",
}
_BOOLEAN_COLUMNS = {
    "allow_partial",
    "anatomical_variant",
    "tissue_exceeds_body_area",
    "trunk_mask_fragmented",
    "body_mask_internal_gap",
    "trunk_mask_internal_gap",
    "body_surface_invalid",
    "trunk_surface_invalid",
    "territory_complete",
    "reference_alignment_valid",
    "reference_alignment_review_required",
    "bin_valid",
    "bin_upper_inclusive",
    "distribution_valid",
    "body_surface_touches_fov",
    "ct_max_pelvic_circumference_value_is_fov_cropped",
    "trunk_mean_circumference_cm_value_is_fov_cropped",
    "ct_min_waist_to_pelvic_ratio_value_is_fov_cropped",
    "ct_midwaist_to_pelvic_ratio_value_is_fov_cropped",
    "sequence_gap_cranial",
    "sequence_gap_caudal",
}
_INTEGER_COLUMNS = {
    "slice_id",
    "longitudinal_order",
    "slice_index_zyx_z",
    "native_label",
    "territory_bin",
    "bins_per_territory",
    "vertebral_territory_bin",
    "contributing_slice_count",
    "signature_bin",
    "relative_bin_index",
    "reference_alignment_anchor_count",
    "bin_index",
    "voxel_count",
    "total_voxel_count",
    "finite_voxel_count",
    "nonfinite_voxel_count",
    "in_histogram_voxel_count",
    "below_histogram_voxel_count",
    "above_histogram_voxel_count",
}


def _storage_kind(column: str) -> str:
    """Return the canonical nullable storage family for one table column."""

    if column == "contributing_slice_ids" or column.endswith("_contributing_slice_ids"):
        return "list_int64"
    if column in _STRING_COLUMNS or column.endswith(
        (
            "_reason",
            "_status",
            "_backend",
            "_source",
            "_definition",
            "_preprocessing",
            "_level",
            "_sha256",
        )
    ):
        return "string"
    if (
        column in _BOOLEAN_COLUMNS
        or column.endswith(
            (
                "_valid",
                "_complete",
                "_eligible",
                "_flag",
                "_closed",
                "_touching_fov",
                "_touches_fov",
                "_at_search_boundary",
                "_present",
                "_fragmented",
                "_surface_invalid",
                "_allow_partial",
            )
        )
        or "_valid_at_" in column
        or "_complete_at_" in column
    ):
        return "boolean"
    if column in _INTEGER_COLUMNS or column.endswith(
        ("_count", "_voxel_count", "_slice_id", "_index", "_order")
    ):
        return "int64"
    return "float64"


def normalize_measurement_table_storage(table: pd.DataFrame) -> pd.DataFrame:
    """Give nullable and empty tables the same deterministic Arrow types.

    Pandas otherwise serializes an all-null string as Arrow ``null`` and an
    all-empty list column as ``list<null>``. That makes per-case Parquet
    schemas depend on anatomy/FOV availability and prevents reliable cohort
    concatenation.
    """

    output = table.copy()
    for column in output.columns:
        kind = _storage_kind(str(column))
        if kind == "list_int64":
            values: list[list[int] | None] = []
            for value in output[column].tolist():
                if value is None or value is pd.NA:
                    values.append(None)
                    continue
                if not isinstance(value, (list, tuple, np.ndarray)):
                    raise TypeError(f"{column} must contain lists of integer slice IDs.")
                normalized: list[int] = []
                for item in value:
                    if isinstance(item, (bool, np.bool_)) or not isinstance(
                        item,
                        (int, np.integer),
                    ):
                        raise TypeError(f"{column} must contain lists of integer slice IDs.")
                    normalized.append(int(item))
                values.append(normalized)
            output[column] = pd.Series(
                values,
                index=output.index,
                dtype="object",
            )
        elif kind == "string":
            output[column] = output[column].astype("string")
        elif kind == "boolean":
            output[column] = output[column].astype("boolean")
        elif kind == "int64":
            output[column] = pd.to_numeric(output[column], errors="raise").astype("Int64")
        else:
            output[column] = pd.to_numeric(output[column], errors="raise").astype("Float64")
    return output


def canonical_arrow_schema(table: pd.DataFrame) -> pa.Schema:
    """Return deterministic physical Arrow types for one canonical table."""

    arrow_types = {
        "list_int64": pa.list_(pa.int64()),
        "string": pa.string(),
        "boolean": pa.bool_(),
        "int64": pa.int64(),
        "float64": pa.float64(),
    }
    return pa.schema(
        [
            pa.field(str(column), arrow_types[_storage_kind(str(column))])
            for column in table.columns
        ],
        metadata={
            b"bodycomposition.measurement_schema_version": (
                MEASUREMENT_SCHEMA_VERSION.encode("ascii")
            )
        },
    )


def validate_signature_contract(
    table: pd.DataFrame,
    *,
    table_name: str = "signature",
) -> None:
    """Validate the immutable fixed-mm signature identity and alignment fields."""

    if len(table) != SIGNATURE_BIN_COUNT:
        raise ValueError(
            f"{table_name} must contain exactly {SIGNATURE_BIN_COUNT} fixed physical bins."
        )
    expected_bins = np.arange(SIGNATURE_BIN_COUNT, dtype=np.int64)
    expected_relative = expected_bins - (SIGNATURE_BIN_COUNT // 2)
    expected_centres = expected_relative.astype(float) * SIGNATURE_BIN_WIDTH_MM
    expected_inferior = expected_centres - SIGNATURE_BIN_WIDTH_MM / 2.0
    expected_superior = expected_centres + SIGNATURE_BIN_WIDTH_MM / 2.0

    if not np.array_equal(
        table["signature_bin"].to_numpy(dtype=np.int64),
        expected_bins,
    ):
        raise ValueError(f"{table_name} must contain ordered bin identities 0..99.")
    if not np.array_equal(
        table["relative_bin_index"].to_numpy(dtype=np.int64),
        expected_relative,
    ):
        raise ValueError(f"{table_name} relative_bin_index differs from the fixed L3 grid.")
    for column, expected in (
        ("reference_inferior_mm", expected_inferior),
        ("reference_center_mm", expected_centres),
        ("reference_superior_mm", expected_superior),
        (
            "bin_width_mm",
            np.full(SIGNATURE_BIN_COUNT, SIGNATURE_BIN_WIDTH_MM, dtype=float),
        ),
    ):
        if not np.allclose(
            table[column].to_numpy(dtype=float),
            expected,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"{table_name} {column} differs from the fixed L3-centred grid.")
    if not table["signature_schema_version"].eq(SIGNATURE_SCHEMA_VERSION).all():
        raise ValueError(f"{table_name} has an unsupported signature_schema_version.")
    if not table["reference_alignment_version"].eq(REFERENCE_ALIGNMENT_VERSION).all():
        raise ValueError(f"{table_name} has an unsupported reference_alignment_version.")
    if not table["reference_level"].eq(SIGNATURE_REFERENCE_LEVEL).all():
        raise ValueError(f"{table_name} reference_level must be {SIGNATURE_REFERENCE_LEVEL}.")
    profile_ids = table["signature_profile_id"].dropna().astype(str).unique()
    if len(profile_ids) != 1 or not table["signature_profile_id"].notna().all():
        raise ValueError(f"{table_name} must contain one signature_profile_id.")
    for column in (
        "reference_alignment_valid",
        "reference_alignment_method",
        "reference_alignment_confidence",
        "reference_alignment_anchor_count",
        "reference_alignment_anchor_levels",
        "reference_origin_position_superior_mm",
        "reference_alignment_slope_mm_per_level",
        "reference_alignment_residual_mm",
        "reference_alignment_review_required",
        "reference_variant_sequence",
    ):
        if table[column].nunique(dropna=False) != 1:
            raise ValueError(f"{table_name} has inconsistent {column} across fixed bins.")

    anchor_count = int(table["reference_alignment_anchor_count"].iloc[0])
    anchor_levels = str(table["reference_alignment_anchor_levels"].iloc[0])
    observed_anchor_count = len([level for level in anchor_levels.split(",") if level])
    if anchor_count < 0 or observed_anchor_count != anchor_count:
        raise ValueError(f"{table_name} has inconsistent reference alignment anchors.")

    alignment_valid = bool(table["reference_alignment_valid"].iloc[0])
    origin = pd.to_numeric(
        table["reference_origin_position_superior_mm"],
        errors="coerce",
    ).to_numpy(dtype=float)
    physical_inferior = pd.to_numeric(
        table["physical_inferior_position_superior_mm"],
        errors="coerce",
    ).to_numpy(dtype=float)
    physical_superior = pd.to_numeric(
        table["physical_superior_position_superior_mm"],
        errors="coerce",
    ).to_numpy(dtype=float)
    coverage = pd.to_numeric(
        table["coverage_fraction"],
        errors="coerce",
    ).to_numpy(dtype=float)
    contributing_slices = pd.to_numeric(
        table["contributing_slice_count"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(contributing_slices)) or np.any(contributing_slices < 0):
        raise ValueError(f"{table_name} has invalid contributing_slice_count values.")

    if alignment_valid:
        if not np.all(np.isfinite(origin)):
            raise ValueError(f"{table_name} has a valid alignment without a finite origin.")
        if not np.allclose(
            physical_inferior,
            origin + expected_inferior,
            rtol=0.0,
            atol=1e-9,
        ) or not np.allclose(
            physical_superior,
            origin + expected_superior,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{table_name} physical bounds do not match its translation-only reference."
            )
        if not np.all(np.isfinite(coverage)) or np.any((coverage < 0.0) | (coverage > 1.0)):
            raise ValueError(f"{table_name} coverage_fraction must be finite and within [0, 1].")
        bin_valid = table["bin_valid"].fillna(False).to_numpy(dtype=bool)
        if not np.array_equal(bin_valid, coverage > 0.0):
            raise ValueError(f"{table_name} bin_valid does not match acquired coverage.")
        outside_fov = coverage == 0.0
        if np.any(contributing_slices[outside_fov] != 0):
            raise ValueError(f"{table_name} has contributing slices outside the acquired FOV.")
        if not table.loc[outside_fov, "bin_reason"].eq("outside_fov").all():
            raise ValueError(f"{table_name} must mark zero-coverage bins as outside_fov.")
    else:
        if (
            np.any(np.isfinite(origin))
            or np.any(np.isfinite(physical_inferior))
            or np.any(np.isfinite(physical_superior))
            or np.any(np.isfinite(coverage))
        ):
            raise ValueError(f"{table_name} has physical coordinates despite unresolved alignment.")
        if table["bin_valid"].fillna(False).any() or np.any(contributing_slices != 0):
            raise ValueError(f"{table_name} has acquired bins despite unresolved alignment.")
        if not bool(table["reference_alignment_review_required"].iloc[0]):
            raise ValueError(f"{table_name} unresolved alignment must require review.")


def validate_hu_distribution_contract(
    table: pd.DataFrame,
    *,
    table_name: str = "hu_distributions",
) -> None:
    """Validate the fixed-bin, raw-voxel tissue HU distribution contract."""

    missing = sorted(HU_DISTRIBUTION_REQUIRED_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}.")
    expected_bin_count = int(
        round((HU_DISTRIBUTION_MAX_HU - HU_DISTRIBUTION_MIN_HU) / HU_DISTRIBUTION_BIN_WIDTH_HU)
    )
    expected_row_count = len(HU_DISTRIBUTION_TISSUES) * expected_bin_count
    if len(table) != expected_row_count:
        raise ValueError(
            f"{table_name} must contain exactly {expected_row_count} fixed tissue/bin rows."
        )
    expected_tissue_sequence = [
        tissue_key
        for tissue_key, _display_label, _definition in HU_DISTRIBUTION_TISSUES
        for _ in range(expected_bin_count)
    ]
    if table["tissue_key"].astype(str).tolist() != expected_tissue_sequence:
        raise ValueError(f"{table_name} must retain the fixed SM, SAT, aVAT, tVAT tissue order.")
    if (
        table["distribution_schema_version"].isna().any()
        or not table["distribution_schema_version"].eq(TISSUE_HU_DISTRIBUTION_SCHEMA_VERSION).all()
    ):
        raise ValueError(f"{table_name} has an unsupported distribution_schema_version.")
    if (
        table["distribution_scope"].isna().any()
        or not table["distribution_scope"].eq(HU_DISTRIBUTION_SCOPE).all()
    ):
        raise ValueError(f"{table_name} distribution_scope must be {HU_DISTRIBUTION_SCOPE}.")
    if (
        table["bin_semantics"].isna().any()
        or not table["bin_semantics"].eq("left_closed_right_open_final_closed").all()
    ):
        raise ValueError(f"{table_name} has unsupported histogram bin semantics.")
    if table["quantile_method"].isna().any() or not table["quantile_method"].eq("linear").all():
        raise ValueError(f"{table_name} has an unsupported quantile method.")
    if table[["tissue_key", "bin_index"]].duplicated().any():
        raise ValueError(f"{table_name} contains duplicate tissue/bin identities.")

    scope_inferior = pd.to_numeric(
        table["scope_inferior_position_superior_mm"],
        errors="coerce",
    ).to_numpy(dtype=float)
    scope_superior = pd.to_numeric(
        table["scope_superior_position_superior_mm"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(scope_inferior))
        or not np.all(np.isfinite(scope_superior))
        or not np.allclose(scope_inferior, scope_inferior[0], rtol=0.0, atol=1e-9)
        or not np.allclose(scope_superior, scope_superior[0], rtol=0.0, atol=1e-9)
        or not scope_inferior[0] < scope_superior[0]
    ):
        raise ValueError(f"{table_name} has inconsistent analyzed-volume bounds.")

    expected_edges = np.arange(
        HU_DISTRIBUTION_MIN_HU,
        HU_DISTRIBUTION_MAX_HU + HU_DISTRIBUTION_BIN_WIDTH_HU,
        HU_DISTRIBUTION_BIN_WIDTH_HU,
        dtype=float,
    )
    expected_indices = np.arange(expected_bin_count, dtype=np.int64)
    expected_inclusive = np.zeros(expected_bin_count, dtype=bool)
    expected_inclusive[-1] = True
    tissue_metadata = {
        tissue_key: (display_label, source_compartment)
        for tissue_key, display_label, source_compartment in HU_DISTRIBUTION_TISSUES
    }
    allowed_invalid_reasons = {
        "empty_tissue",
        "invalid_measurement",
        "missing_compartment",
    }

    for tissue_key, group in table.groupby("tissue_key", sort=False):
        group = group.reset_index(drop=True)
        display_label, source_compartment = tissue_metadata[str(tissue_key)]
        if (
            group[
                [
                    "display_label",
                    "source_compartment",
                    "source_label_ids",
                    "source_semantics",
                ]
            ]
            .isna()
            .any()
            .any()
            or not group["display_label"].eq(display_label).all()
            or not group["source_compartment"].eq(source_compartment).all()
            or not group["source_semantics"].eq("model_native_compartment").all()
        ):
            raise ValueError(f"{table_name} has inconsistent metadata for {tissue_key}.")
        if not np.array_equal(
            pd.to_numeric(group["bin_index"], errors="coerce").to_numpy(dtype=np.int64),
            expected_indices,
        ):
            raise ValueError(f"{table_name} has unstable bin identities for {tissue_key}.")
        for column, expected in (
            ("bin_lower_hu", expected_edges[:-1]),
            ("bin_upper_hu", expected_edges[1:]),
            (
                "bin_width_hu",
                np.full(expected_bin_count, HU_DISTRIBUTION_BIN_WIDTH_HU),
            ),
            (
                "histogram_min_hu",
                np.full(expected_bin_count, HU_DISTRIBUTION_MIN_HU),
            ),
            (
                "histogram_max_hu",
                np.full(expected_bin_count, HU_DISTRIBUTION_MAX_HU),
            ),
        ):
            observed = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(observed, expected, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"{table_name} {column} differs from the fixed HU grid for {tissue_key}."
                )
        if not group["bin_upper_inclusive"].notna().all() or not np.array_equal(
            group["bin_upper_inclusive"].to_numpy(dtype=bool),
            expected_inclusive,
        ):
            raise ValueError(f"{table_name} has inconsistent final-bin inclusion for {tissue_key}.")
        for column in (
            "source_label_ids",
            "total_voxel_count",
            "finite_voxel_count",
            "nonfinite_voxel_count",
            "in_histogram_voxel_count",
            "below_histogram_voxel_count",
            "above_histogram_voxel_count",
            "below_histogram_voxel_fraction",
            "above_histogram_voxel_fraction",
            "contributing_slice_count",
            "voxel_volume_mm3",
            "total_volume_cm3",
            "distribution_valid",
            "distribution_reason",
            "mean_hu",
            "standard_deviation_hu",
            "q1_hu",
            "median_hu",
            "q3_hu",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"{table_name} has inconsistent {column} for {tissue_key}.")

        counts = pd.to_numeric(group["voxel_count"], errors="coerce").to_numpy(dtype=float)
        fractions = pd.to_numeric(
            group["voxel_fraction"],
            errors="coerce",
        ).to_numpy(dtype=float)
        if (
            not np.all(np.isfinite(counts))
            or np.any(counts < 0)
            or not np.all(counts == np.floor(counts))
            or not np.all(np.isfinite(fractions))
            or np.any((fractions < 0.0) | (fractions > 1.0))
        ):
            raise ValueError(f"{table_name} has invalid histogram values for {tissue_key}.")
        totals = {
            column: int(group[column].iloc[0])
            for column in (
                "total_voxel_count",
                "finite_voxel_count",
                "nonfinite_voxel_count",
                "in_histogram_voxel_count",
                "below_histogram_voxel_count",
                "above_histogram_voxel_count",
                "contributing_slice_count",
            )
        }
        if any(value < 0 for value in totals.values()):
            raise ValueError(f"{table_name} has negative counts for {tissue_key}.")
        if (
            totals["finite_voxel_count"] + totals["nonfinite_voxel_count"]
            != totals["total_voxel_count"]
            or totals["in_histogram_voxel_count"]
            + totals["below_histogram_voxel_count"]
            + totals["above_histogram_voxel_count"]
            != totals["finite_voxel_count"]
            or int(counts.sum()) != totals["in_histogram_voxel_count"]
        ):
            raise ValueError(f"{table_name} count accounting differs for {tissue_key}.")
        expected_fractions = (
            counts / totals["total_voxel_count"]
            if totals["total_voxel_count"]
            else np.zeros(expected_bin_count, dtype=float)
        )
        if not np.allclose(fractions, expected_fractions, rtol=0.0, atol=1e-12):
            raise ValueError(f"{table_name} voxel fractions differ for {tissue_key}.")
        below_fraction = float(group["below_histogram_voxel_fraction"].iloc[0])
        above_fraction = float(group["above_histogram_voxel_fraction"].iloc[0])
        expected_below_fraction = (
            totals["below_histogram_voxel_count"] / totals["total_voxel_count"]
            if totals["total_voxel_count"]
            else 0.0
        )
        expected_above_fraction = (
            totals["above_histogram_voxel_count"] / totals["total_voxel_count"]
            if totals["total_voxel_count"]
            else 0.0
        )
        if not np.isclose(
            below_fraction,
            expected_below_fraction,
            rtol=0.0,
            atol=1e-12,
        ) or not np.isclose(
            above_fraction,
            expected_above_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{table_name} overflow fractions differ for {tissue_key}.")
        voxel_volume_mm3 = float(group["voxel_volume_mm3"].iloc[0])
        total_volume_cm3 = float(group["total_volume_cm3"].iloc[0])
        if (
            not np.isfinite(voxel_volume_mm3)
            or voxel_volume_mm3 <= 0
            or not np.isclose(
                total_volume_cm3,
                totals["total_voxel_count"] * voxel_volume_mm3 / 1000.0,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise ValueError(f"{table_name} has inconsistent physical volume for {tissue_key}.")

        valid = bool(group["distribution_valid"].iloc[0])
        reason_value = group["distribution_reason"].iloc[0]
        reason = None if pd.isna(reason_value) else str(reason_value)
        quantiles = np.asarray(
            [
                group["q1_hu"].iloc[0],
                group["median_hu"].iloc[0],
                group["q3_hu"].iloc[0],
            ],
            dtype=float,
        )
        moments = np.asarray(
            [
                group["mean_hu"].iloc[0],
                group["standard_deviation_hu"].iloc[0],
            ],
            dtype=float,
        )
        source_label_ids = str(group["source_label_ids"].iloc[0])
        parsed_source_label_ids: tuple[int, ...] = ()
        if source_label_ids:
            try:
                parsed_source_label_ids = tuple(int(value) for value in source_label_ids.split(","))
            except ValueError as error:
                raise ValueError(
                    f"{table_name} has invalid source_label_ids for {tissue_key}."
                ) from error
            if any(value <= 0 for value in parsed_source_label_ids) or len(
                set(parsed_source_label_ids)
            ) != len(parsed_source_label_ids):
                raise ValueError(f"{table_name} has invalid source_label_ids for {tissue_key}.")
        if valid:
            if (
                totals["total_voxel_count"] <= 0
                or totals["nonfinite_voxel_count"] != 0
                or totals["contributing_slice_count"] <= 0
                or not parsed_source_label_ids
                or reason is not None
                or not np.all(np.isfinite(moments))
                or moments[1] < 0
                or not np.all(np.isfinite(quantiles))
                or not quantiles[0] <= quantiles[1] <= quantiles[2]
            ):
                raise ValueError(
                    f"{table_name} has inconsistent valid statistics for {tissue_key}."
                )
        elif reason not in allowed_invalid_reasons:
            raise ValueError(f"{table_name} has an unsupported invalid reason for {tissue_key}.")
        if totals["total_voxel_count"] == 0 and (
            totals["contributing_slice_count"] != 0
            or np.any(np.isfinite(moments))
            or np.any(np.isfinite(quantiles))
        ):
            raise ValueError(f"{table_name} has statistics for an empty tissue {tissue_key}.")
        if reason == "missing_compartment" and parsed_source_label_ids:
            raise ValueError(f"{table_name} marks {tissue_key} missing despite source labels.")


def _validate_mask(mask_zyx: np.ndarray, geometry: ImageGeometry, name: str) -> np.ndarray:
    mask = np.asarray(mask_zyx)
    expected_shape = tuple(reversed(geometry.size_xyz))
    if mask.ndim != 3 or mask.shape != expected_shape:
        raise ValueError(f"{name} must have array_zyx shape {expected_shape}, got {mask.shape}.")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool, copy=False)
    return mask


@dataclass(frozen=True)
class MeasurementIdentity:
    """Identifiers attached to every canonical measurement row."""

    case_id: str
    run_id: str
    analysis_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("case_id", self.case_id),
            ("run_id", self.run_id),
            ("analysis_id", self.analysis_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

    def as_columns(self) -> dict[str, str]:
        return {
            "schema_version": MEASUREMENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "analysis_id": self.analysis_id,
            "case_id": self.case_id,
        }


@dataclass(frozen=True)
class BodySurfaceResult:
    """Aligned full-body and trunk masks from one explicitly selected backend."""

    backend_id: str
    geometry: ImageGeometry
    body_mask_zyx: np.ndarray
    trunk_mask_zyx: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)
    qc_flags: tuple[QCFlag, ...] = ()

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("body-surface backend_id must not be empty.")
        body = _validate_mask(self.body_mask_zyx, self.geometry, "body_mask_zyx")
        trunk = _validate_mask(self.trunk_mask_zyx, self.geometry, "trunk_mask_zyx")
        if np.any(trunk & ~body):
            raise ValueError("The trunk mask must be a subset of the full body mask.")
        object.__setattr__(self, "body_mask_zyx", body)
        object.__setattr__(self, "trunk_mask_zyx", trunk)

    @property
    def qc_status(self) -> QCStatus:
        if any(flag.severity.value == "error" for flag in self.qc_flags):
            return QCStatus.FAIL
        if any(flag.severity.value == "warning" for flag in self.qc_flags):
            return QCStatus.REVIEW
        return QCStatus.PASS


@dataclass(frozen=True)
class Landmark:
    """One validated anatomical superior-axis coordinate."""

    name: str
    position_superior_mm: float | None
    backend_id: str
    valid: bool
    uncertainty_mm: float | None = None
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.backend_id.strip():
            raise ValueError("Landmark name and backend_id must not be empty.")
        if self.valid:
            if self.position_superior_mm is None or not np.isfinite(self.position_superior_mm):
                raise ValueError("A valid landmark requires a finite superior coordinate.")
            if self.reason is not None:
                raise ValueError("A valid landmark cannot have a missing reason.")
        elif self.reason is None:
            raise ValueError("An invalid landmark requires a reason.")
        if self.uncertainty_mm is not None and (
            not np.isfinite(self.uncertainty_mm) or self.uncertainty_mm < 0
        ):
            raise ValueError("Landmark uncertainty must be finite and non-negative.")


@dataclass(frozen=True)
class LandmarkSet:
    """Landmarks required for anatomical mid-waist localization."""

    lowest_rib_inferior: Landmark
    iliac_crest_superior: Landmark
    provenance: Mapping[str, Any] = field(default_factory=dict)
    qc_flags: tuple[QCFlag, ...] = ()


@dataclass(frozen=True)
class VertebralExtent:
    """Cleaned physical superior/inferior extent of one vertebral body."""

    native_label: int
    anatomical_label: str
    inferior_mm: float | None
    superior_mm: float | None
    centroid_lps_xyz: tuple[float, float, float] | None
    centroid_superior_mm: float | None
    voxel_count: int
    retained_voxel_count: int
    removed_voxel_fraction: float
    component_count: int
    touches_fov: bool
    complete: bool
    valid: bool
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if self.native_label <= 0 or not self.anatomical_label.strip():
            raise ValueError("A vertebral extent requires a positive label and anatomical name.")
        if self.voxel_count < 0 or self.retained_voxel_count < 0:
            raise ValueError("Vertebral voxel counts must be non-negative.")
        if not 0 <= self.removed_voxel_fraction <= 1:
            raise ValueError("removed_voxel_fraction must be between zero and one.")
        if self.valid:
            if self.inferior_mm is None or self.superior_mm is None:
                raise ValueError("A valid vertebral extent requires physical bounds.")
            if not self.inferior_mm < self.superior_mm:
                raise ValueError("Vertebral inferior bound must be below its superior bound.")
        if not self.valid and self.missing_reason is None:
            raise ValueError("An invalid vertebral extent requires a missing reason.")

    @property
    def height_mm(self) -> float | None:
        if self.inferior_mm is None or self.superior_mm is None:
            return None
        return self.superior_mm - self.inferior_mm


@dataclass(frozen=True)
class VertebralTerritory:
    """Native level territory bounded by adjacent physical centroids."""

    native_label: int
    anatomical_label: str
    inferior_mm: float | None
    superior_mm: float | None
    centroid_lps_xyz: tuple[float, float, float] | None
    centroid_superior_mm: float | None
    extent_valid: bool
    complete: bool
    sequence_gap_cranial: bool = False
    sequence_gap_caudal: bool = False
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if self.native_label <= 0 or not self.anatomical_label.strip():
            raise ValueError("A vertebral territory requires a positive native label and name.")
        has_bounds = self.inferior_mm is not None and self.superior_mm is not None
        if has_bounds and not float(self.inferior_mm) < float(self.superior_mm):
            raise ValueError("Vertebral territory inferior bound must be below its superior bound.")
        if self.complete and (not self.extent_valid or not has_bounds):
            raise ValueError("A complete vertebral territory requires a valid extent and bounds.")
        if not self.complete and self.missing_reason is None:
            raise ValueError("An incomplete vertebral territory requires a reason.")

    @property
    def height_mm(self) -> float | None:
        if self.inferior_mm is None or self.superior_mm is None:
            return None
        return float(self.superior_mm - self.inferior_mm)


@dataclass(frozen=True)
class MeasurementBundle:
    """Canonical measurement tables and the source objects used to derive them."""

    identity: MeasurementIdentity
    slices: pd.DataFrame
    vertebrae: pd.DataFrame
    summaries: pd.DataFrame
    signature: pd.DataFrame
    hu_distributions: pd.DataFrame
    body_surface: BodySurfaceResult
    vertebral_extents: Mapping[str, VertebralExtent]
    vertebral_territories: Mapping[str, VertebralTerritory]
    landmarks: LandmarkSet | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    qc_flags: tuple[QCFlag, ...] = ()
    paths: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "slices",
            "vertebrae",
            "summaries",
            "signature",
            "hu_distributions",
        ):
            value = getattr(self, name)
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"Measurement bundle {name} must be a pandas DataFrame.")
            object.__setattr__(
                self,
                name,
                normalize_measurement_table_storage(value),
            )
        if len(self.summaries) != 1:
            raise ValueError("summaries must contain exactly one case row.")
        required_by_table = {
            "slices": SLICE_REQUIRED_COLUMNS,
            "vertebrae": VERTEBRA_REQUIRED_COLUMNS,
            "summaries": SUMMARY_REQUIRED_COLUMNS,
            "signature": SIGNATURE_REQUIRED_COLUMNS,
            "hu_distributions": HU_DISTRIBUTION_REQUIRED_COLUMNS,
        }
        for name, required in required_by_table.items():
            missing = sorted(required - set(getattr(self, name).columns))
            if missing:
                raise ValueError(f"Measurement table {name} is missing columns: {missing}.")
        if self.slices.empty:
            raise ValueError("slices must retain every acquired CT slice and cannot be empty.")
        validate_signature_contract(self.signature)
        validate_hu_distribution_contract(self.hu_distributions)
        if self.slices["slice_id"].duplicated().any():
            raise ValueError("slice_id must be unique within a case.")
        position = self.slices["position_superior_mm"].to_numpy(dtype=float)
        if len(position) > 1 and not np.all(np.diff(position) > 0):
            raise ValueError("slice_measurements must be strictly ordered inferior to superior.")
        bin_identity = ["vertebral_level", "territory_bin"]
        if self.vertebrae[bin_identity].duplicated().any():
            raise ValueError("vertebrae must contain unique native level/bin identities.")
        counts = self.vertebrae.groupby("vertebral_level", dropna=False).size()
        if not counts.empty and not counts.eq(BINS_PER_VERTEBRAL_TERRITORY).all():
            raise ValueError("Every detected vertebral territory must contain exactly three rows.")
        for level, group in self.vertebrae.groupby("vertebral_level", sort=False):
            observed = group["territory_bin"].to_numpy(dtype=int).tolist()
            expected = list(range(1, BINS_PER_VERTEBRAL_TERRITORY + 1))
            if observed != expected:
                raise ValueError(f"Vertebral territory {level!r} has unstable bin identities.")
        expected_identity = self.identity.as_columns()
        for name in (
            "slices",
            "vertebrae",
            "summaries",
            "signature",
            "hu_distributions",
        ):
            table = getattr(self, name)
            for column, expected in expected_identity.items():
                if not table[column].eq(expected).all():
                    raise ValueError(f"Measurement table {name} has inconsistent {column} values.")
        summary = self.summaries.iloc[0]
        body_touching_fov = self.slices["body_touching_fov"].fillna(False).astype(bool)
        if bool(summary["body_surface_touches_fov"]) != bool(body_touching_fov.any()) or int(
            summary["body_surface_touches_fov_slice_count"]
        ) != int(body_touching_fov.sum()):
            raise ValueError("Case-level body FOV flags disagree with slices.parquet.")
        trunk_touching_fov = self.slices["trunk_touching_fov"].fillna(False).astype(bool)
        if bool(summary["trunk_contour_touches_fov"]) != bool(trunk_touching_fov.any()) or int(
            summary["trunk_contour_touches_fov_slice_count"]
        ) != int(trunk_touching_fov.sum()):
            raise ValueError("Case-level trunk FOV flags disagree with slices.parquet.")
        tolerance_values = (
            pd.to_numeric(
                self.slices["full_coverage_tolerance"],
                errors="coerce",
            )
            .dropna()
            .unique()
        )
        if len(tolerance_values) != 1:
            raise ValueError("slices must contain one full_coverage_tolerance.")
        full_coverage_tolerance = float(tolerance_values[0])
        for prefix in (
            "ct_min_trunk_circumference_t10_l5",
            "ct_max_pelvic_circumference",
        ):
            valid = bool(summary[f"{prefix}_valid"])
            eligible = bool(summary[f"{prefix}_eligible"])
            if eligible and not valid:
                raise ValueError(f"{prefix} cannot be eligible when it is invalid.")
            if not eligible:
                continue
            acquisition_coverage = float(summary[f"{prefix}_search_acquisition_coverage_fraction"])
            contour_coverage = float(summary[f"{prefix}_search_valid_contour_coverage_fraction"])
            if (
                not bool(summary[f"{prefix}_search_anatomically_complete"])
                or bool(summary[f"{prefix}_at_search_boundary"])
                or not np.isfinite(acquisition_coverage)
                or acquisition_coverage < full_coverage_tolerance
                or not np.isfinite(contour_coverage)
                or contour_coverage < full_coverage_tolerance
            ):
                raise ValueError(
                    f"{prefix} eligibility contradicts anatomy, coverage, or boundary QC."
                )
        for prefix in (
            "ct_min_waist_to_pelvic_ratio",
            "ct_midwaist_to_pelvic_ratio",
        ):
            if bool(summary[f"{prefix}_eligible"]) and not bool(summary[f"{prefix}_valid"]):
                raise ValueError(f"{prefix} cannot be eligible when it is invalid.")
        pelvic_fov_cropped = bool(summary["ct_max_pelvic_circumference_value_is_fov_cropped"])
        if pelvic_fov_cropped and (
            not bool(summary["ct_max_pelvic_circumference_valid"])
            or bool(summary["ct_max_pelvic_circumference_eligible"])
            or not bool(summary["ct_max_pelvic_circumference_search_touches_fov"])
            or str(summary["ct_max_pelvic_circumference_reason"]) != "touching_image_boundary"
        ):
            raise ValueError("FOV-cropped pelvic circumference has contradictory validity or QC.")
        if pelvic_fov_cropped:
            pelvic_slice_id = int(summary["ct_max_pelvic_circumference_slice_id"])
            selected_slices = self.slices.loc[self.slices["slice_id"].eq(pelvic_slice_id)]
            if len(selected_slices) != 1:
                raise ValueError(
                    "FOV-cropped pelvic circumference lacks one selected source slice."
                )
            selected_slice = selected_slices.iloc[0]
            selected_circumference = float(selected_slice["trunk_circumference_cm"])
            if (
                not bool(selected_slice["trunk_touching_fov"])
                or not bool(selected_slice["trunk_contour_closed"])
                or bool(selected_slice["trunk_mask_fragmented"])
                or str(selected_slice["trunk_contour_reason"]) != "touching_image_boundary"
                or not np.isclose(
                    selected_circumference,
                    float(summary["ct_max_pelvic_circumference_cm"]),
                    rtol=0.0,
                    atol=1e-9,
                )
            ):
                raise ValueError(
                    "FOV-cropped pelvic circumference disagrees with its source slice."
                )
        for prefix in (
            "ct_min_waist_to_pelvic_ratio",
            "ct_midwaist_to_pelvic_ratio",
        ):
            ratio_fov_cropped = bool(summary[f"{prefix}_value_is_fov_cropped"])
            if ratio_fov_cropped and (
                not pelvic_fov_cropped
                or not bool(summary[f"{prefix}_valid"])
                or bool(summary[f"{prefix}_eligible"])
            ):
                raise ValueError(f"{prefix} has contradictory FOV-cropped denominator QC.")
        if bool(summary["ct_min_waist_to_pelvic_ratio_eligible"]) and not (
            bool(summary["ct_min_trunk_circumference_t10_l5_eligible"])
            and bool(summary["ct_max_pelvic_circumference_eligible"])
        ):
            raise ValueError(
                "Minimum waist-to-pelvic ratio eligibility requires eligible components."
            )
        if bool(summary["ct_midwaist_to_pelvic_ratio_eligible"]) and not (
            bool(summary["ct_midwaist_circumference_valid"])
            and bool(summary["ct_max_pelvic_circumference_eligible"])
        ):
            raise ValueError(
                "Mid-waist-to-pelvic ratio eligibility requires valid, eligible components."
            )
        fov_cropped_trunk = self.vertebrae[
            "trunk_mean_circumference_cm_value_is_fov_cropped"
        ].fillna(False).astype(bool)
        trunk_valid = self.vertebrae["trunk_mean_circumference_cm_valid"].fillna(
            False
        ).astype(bool)
        trunk_values = pd.to_numeric(
            self.vertebrae["trunk_mean_circumference_cm"],
            errors="coerce",
        )
        if (fov_cropped_trunk & (~trunk_valid | ~np.isfinite(trunk_values))).any():
            raise ValueError(
                "FOV-cropped vertebral trunk circumference must retain a valid numeric value."
            )
        for _, row in self.vertebrae.loc[fov_cropped_trunk].iterrows():
            overlaps = (
                np.minimum(
                    self.slices["slice_slab_superior_mm"].to_numpy(dtype=float),
                    float(row["bin_superior_mm"]),
                )
                - np.maximum(
                    self.slices["slice_slab_inferior_mm"].to_numpy(dtype=float),
                    float(row["bin_inferior_mm"]),
                )
            ) > 0
            source_evidence = (
                overlaps
                & self.slices["trunk_touching_fov"].fillna(False).to_numpy(dtype=bool)
                & self.slices["trunk_contour_closed"].fillna(False).to_numpy(dtype=bool)
                & ~self.slices["trunk_mask_fragmented"].fillna(False).to_numpy(dtype=bool)
                & ~self.slices["trunk_mask_internal_gap"].fillna(False).to_numpy(dtype=bool)
                & self.slices["trunk_contour_reason"]
                .fillna("")
                .astype(str)
                .eq("touching_image_boundary")
                .to_numpy(dtype=bool)
                & np.isfinite(
                    pd.to_numeric(
                        self.slices["trunk_circumference_cm"],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                )
            )
            if not np.any(source_evidence):
                raise ValueError(
                    "FOV-cropped vertebral trunk circumference lacks supporting source slices."
                )
        invalid = ~self.vertebrae["bin_valid"].fillna(False).astype(bool)
        reasons = set(self.vertebrae.loc[invalid, "bin_missing_reason"].dropna().astype(str))
        unknown_reasons = sorted(reasons - MISSING_REASONS)
        if unknown_reasons:
            raise ValueError(
                f"Vertebral territories have unknown missing reasons: {unknown_reasons}."
            )

    @property
    def qc_status(self) -> QCStatus:
        if any(flag.severity.value == "error" for flag in self.qc_flags):
            return QCStatus.FAIL
        if any(flag.severity.value == "warning" for flag in self.qc_flags):
            return QCStatus.REVIEW
        return QCStatus.PASS
