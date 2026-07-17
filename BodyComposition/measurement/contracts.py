"""Stable data contracts for measurement outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa

from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import QCFlag, QCStatus


MEASUREMENT_SCHEMA_VERSION = "2.1.0"
VERTEBRAL_TERRITORY_SCHEMA_VERSION = "native-physical-territories-v1-bins3"
BINS_PER_VERTEBRAL_TERRITORY = 3

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
}
SUMMARY_REQUIRED_COLUMNS = {
    "schema_version",
    "run_id",
    "analysis_id",
    "case_id",
    "l3_200mm_slab_valid",
    "l3_200mm_slab_reason",
    "ct_min_trunk_circumference_t10_l5_cm",
    "ct_min_trunk_circumference_t10_l5_valid",
    "ct_min_trunk_circumference_t10_l5_eligible",
    "ct_min_trunk_circumference_t10_l5_reason",
    "ct_min_trunk_circumference_t10_l5_search_valid_contour_coverage_fraction",
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
    "ct_max_pelvic_circumference_search_valid_contour_coverage_fraction",
    "ct_min_waist_to_pelvic_ratio",
    "ct_min_waist_to_pelvic_ratio_valid",
    "ct_min_waist_to_pelvic_ratio_reason",
    "ct_midwaist_to_pelvic_ratio",
    "ct_midwaist_to_pelvic_ratio_valid",
    "ct_midwaist_to_pelvic_ratio_reason",
}
MISSING_REASONS = {
    "outside_fov",
    "partial_fov",
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
}


_STRING_COLUMNS = {
    "schema_version",
    "vertebral_territory_schema_version",
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
    "sequence_gap_cranial",
    "sequence_gap_caudal",
}
_INTEGER_COLUMNS = {
    "slice_id",
    "longitudinal_order",
    "slice_index_zyx_z",
    "original_storage_index",
    "native_label",
    "territory_bin",
    "bins_per_territory",
    "vertebral_territory_bin",
    "contributing_slice_count",
}


def _storage_kind(column: str) -> str:
    """Return the canonical nullable storage family for one table column."""

    if column == "contributing_slice_ids" or column.endswith(
        "_contributing_slice_ids"
    ):
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
                        raise TypeError(
                            f"{column} must contain lists of integer slice IDs."
                        )
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
            output[column] = pd.to_numeric(output[column], errors="raise").astype(
                "Int64"
            )
        else:
            output[column] = pd.to_numeric(output[column], errors="raise").astype(
                "Float64"
            )
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
    body_surface: BodySurfaceResult
    vertebral_extents: Mapping[str, VertebralExtent]
    vertebral_territories: Mapping[str, VertebralTerritory]
    landmarks: LandmarkSet | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    qc_flags: tuple[QCFlag, ...] = ()
    paths: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("slices", "vertebrae", "summaries"):
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
        }
        for name, required in required_by_table.items():
            missing = sorted(required - set(getattr(self, name).columns))
            if missing:
                raise ValueError(f"Measurement table {name} is missing columns: {missing}.")
        if self.slices.empty:
            raise ValueError("slices must retain every acquired CT slice and cannot be empty.")
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
        for name in ("slices", "vertebrae", "summaries"):
            table = getattr(self, name)
            for column, expected in expected_identity.items():
                if not table[column].eq(expected).all():
                    raise ValueError(f"Measurement table {name} has inconsistent {column} values.")
        invalid = ~self.vertebrae["bin_valid"].fillna(False).astype(bool)
        reasons = set(self.vertebrae.loc[invalid, "bin_missing_reason"].dropna().astype(str))
        unknown_reasons = sorted(reasons - MISSING_REASONS)
        if unknown_reasons:
            raise ValueError(f"Vertebral territories have unknown missing reasons: {unknown_reasons}.")

    @property
    def qc_status(self) -> QCStatus:
        if any(flag.severity.value == "error" for flag in self.qc_flags):
            return QCStatus.FAIL
        if any(flag.severity.value == "warning" for flag in self.qc_flags):
            return QCStatus.REVIEW
        return QCStatus.PASS
