"""Construction and deterministic identity of a complete measurement bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from BodyComposition import __version__
from BodyComposition.config import CANONICAL_TISSUE_DEFINITIONS
from BodyComposition.measurement.aggregation import (
    annotate_slice_anatomy,
    build_case_summaries,
    build_vertebra_table,
    derive_vertebral_extents,
    derive_vertebral_territories,
    inferred_anatomical_variants,
)
from BodyComposition.measurement.contracts import (
    MEASUREMENT_SCHEMA_VERSION,
    SIGNATURE_BIN_WIDTH_MM,
    VERTEBRAL_TERRITORY_SCHEMA_VERSION,
    BodySurfaceResult,
    LandmarkSet,
    MeasurementBundle,
    MeasurementIdentity,
)
from BodyComposition.measurement.distributions import build_tissue_hu_distributions
from BodyComposition.measurement.physical import geometry_digest, in_plane_superior_span_mm
from BodyComposition.measurement.signature import build_longitudinal_signature
from BodyComposition.measurement.slices import (
    annotate_longitudinal_circumference_qc,
    calculate_canonical_slice_measurements,
)
from BodyComposition.utils.digests import array_sha256
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.vertebral.contracts import QCFlag, QCSeverity, VertebralResult


def measurement_analysis_id(
    image_zyx: np.ndarray,
    tissue_labels_zyx: np.ndarray,
    body_surface: BodySurfaceResult,
    vertebral_result: VertebralResult,
    configuration: Mapping[str, Any],
    *,
    landmarks: LandmarkSet | None = None,
    tissue_label_schema: Mapping[int, str] | None = None,
    tissue_preprocessing: Mapping[str, Any] | None = None,
    compartment_labels_zyx: np.ndarray | None = None,
    compartment_label_schema: Mapping[int, str] | None = None,
    orientation_provenance: Mapping[str, Any] | None = None,
) -> str:
    """Return an identity covering every measurement input and setting."""

    if vertebral_result.vertebral_body_labels is None:
        raise ValueError("Vertebral body labels are required for measurement identity.")
    payload = {
        "schema": f"bodycomposition-measurement-analysis-v{MEASUREMENT_SCHEMA_VERSION}",
        "bodycomposition_version": __version__,
        "geometry_sha256": geometry_digest(body_surface.geometry),
        "image_sha256": array_sha256(image_zyx),
        "tissue_sha256": array_sha256(tissue_labels_zyx),
        "compartment_sha256": (
            array_sha256(compartment_labels_zyx) if compartment_labels_zyx is not None else None
        ),
        "body_sha256": array_sha256(body_surface.body_mask_zyx),
        "trunk_sha256": array_sha256(body_surface.trunk_mask_zyx),
        "vertebral_body_sha256": array_sha256(vertebral_result.vertebral_body_labels),
        "body_surface_backend": body_surface.backend_id,
        "body_surface_provenance": dict(body_surface.provenance),
        "vertebral_backend": vertebral_result.backend_id,
        "vertebral_label_schema": {
            str(label): name for label, name in sorted(vertebral_result.label_schema.items())
        },
        "vertebral_provenance": dict(vertebral_result.provenance),
        "tissue_label_schema": (
            {str(label): name for label, name in sorted(tissue_label_schema.items())}
            if tissue_label_schema is not None
            else None
        ),
        "compartment_label_schema": (
            {str(label): name for label, name in sorted(compartment_label_schema.items())}
            if compartment_label_schema is not None
            else None
        ),
        "tissue_preprocessing": dict(tissue_preprocessing or {}),
        "orientation_provenance": dict(orientation_provenance or {}),
        "landmarks": (
            {
                "lowest_rib_inferior": {
                    "position_superior_mm": landmarks.lowest_rib_inferior.position_superior_mm,
                    "valid": landmarks.lowest_rib_inferior.valid,
                    "uncertainty_mm": landmarks.lowest_rib_inferior.uncertainty_mm,
                    "reason": landmarks.lowest_rib_inferior.reason,
                    "details": dict(landmarks.lowest_rib_inferior.details),
                },
                "iliac_crest_superior": {
                    "position_superior_mm": landmarks.iliac_crest_superior.position_superior_mm,
                    "valid": landmarks.iliac_crest_superior.valid,
                    "uncertainty_mm": landmarks.iliac_crest_superior.uncertainty_mm,
                    "reason": landmarks.iliac_crest_superior.reason,
                    "details": dict(landmarks.iliac_crest_superior.details),
                },
                "provenance": dict(landmarks.provenance),
            }
            if landmarks is not None
            else None
        ),
        "measurement_configuration": dict(configuration),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return f"analysis-{hashlib.sha256(encoded).hexdigest()}"


def build_measurement_bundle(
    *,
    image_zyx: np.ndarray,
    tissue_labels_zyx: np.ndarray,
    geometry: ImageGeometry,
    tissue_label_schema: Mapping[int, str],
    tissue_backend_id: str,
    tissue_preprocessing: Mapping[str, Any],
    compartment_labels_zyx: np.ndarray,
    compartment_label_schema: Mapping[int, str],
    body_surface: BodySurfaceResult,
    vertebral_result: VertebralResult,
    identity: MeasurementIdentity,
    landmarks: LandmarkSet | None,
    settings: Mapping[str, Any],
    orientation_provenance: Mapping[str, Any] | None = None,
) -> MeasurementBundle:
    """Build all canonical tables from one immutable set of prepared inputs."""

    if vertebral_result.geometry is None:
        raise ValueError("The vertebral result has no geometry.")
    assert_same_physical_domain(
        geometry,
        vertebral_result.geometry,
        reference_name="orientation-prepared CT",
        candidate_name="canonical vertebral-body labels",
    )
    assert_same_physical_domain(
        geometry,
        body_surface.geometry,
        reference_name="orientation-prepared CT",
        candidate_name="body-surface result",
    )
    extent_settings = settings["vertebral_extent"]
    full_coverage_tolerance = float(settings["full_coverage_tolerance"])
    extents = derive_vertebral_extents(
        vertebral_result,
        minimum_component_voxels=int(extent_settings["minimum_component_voxels"]),
        maximum_removed_fraction=float(extent_settings["maximum_removed_fraction"]),
    )
    territories = derive_vertebral_territories(extents)
    slices = calculate_canonical_slice_measurements(
        image_zyx,
        tissue_labels_zyx,
        geometry,
        tissue_label_schema,
        body_surface,
        identity,
        compartment_labels_zyx=compartment_labels_zyx,
        compartment_label_schema=compartment_label_schema,
        tissue_definitions=settings.get("tissue_definitions"),
    )
    orientation = dict(orientation_provenance or {})
    slices["full_coverage_tolerance"] = full_coverage_tolerance
    circumference_qc = settings["qc"]["circumference_jump"]
    slices = annotate_longitudinal_circumference_qc(
        slices,
        maximum_gap_mm=float(circumference_qc["maximum_gap_mm"]),
        minimum_absolute_jump_cm=float(circumference_qc["minimum_absolute_jump_cm"]),
        minimum_relative_jump=float(circumference_qc["minimum_relative_jump"]),
    )
    slices = annotate_slice_anatomy(slices, territories)
    vertebrae = build_vertebra_table(
        slices,
        extents,
        territories,
        identity,
        full_coverage_tolerance=full_coverage_tolerance,
    )
    summaries = build_case_summaries(
        slices,
        extents,
        territories,
        identity,
        landmarks=landmarks,
        slab_length_mm=float(settings["l3_slab_length_mm"]),
        full_coverage_tolerance=full_coverage_tolerance,
    )
    definitions = settings.get("tissue_definitions", {})
    hu_distributions = build_tissue_hu_distributions(
        image_zyx=image_zyx,
        compartment_labels_zyx=compartment_labels_zyx,
        compartment_label_schema=compartment_label_schema,
        geometry=geometry,
        identity=identity,
    )
    extra_signature_definitions = tuple(
        name
        for name, definition in definitions.items()
        if bool(definition.get("enabled", False)) and name not in CANONICAL_TISSUE_DEFINITIONS
    )
    signature, signature_alignment = build_longitudinal_signature(
        slices,
        extents,
        territories,
        identity,
        tissue_profile_id=str(settings["tissue_profile_id"]),
        extra_tissue_definitions=extra_signature_definitions,
        full_coverage_tolerance=full_coverage_tolerance,
    )

    flags: list[QCFlag] = [*body_surface.qc_flags, *vertebral_result.qc_flags]
    in_plane_span_mm = in_plane_superior_span_mm(geometry)
    maximum_unflagged_span_mm = SIGNATURE_BIN_WIDTH_MM / 2.0
    if in_plane_span_mm > maximum_unflagged_span_mm:
        flags.append(
            QCFlag(
                code="oblique_longitudinal_allocation_approximate",
                stage="measurement",
                severity=QCSeverity.WARNING,
                reason=(
                    "A native slice spans a material distance along the patient superior "
                    "axis; slice-centre longitudinal allocation is approximate."
                ),
                observed={"in_plane_superior_span_mm": in_plane_span_mm},
                thresholds={
                    "maximum_unflagged_in_plane_superior_span_mm": (maximum_unflagged_span_mm)
                },
                suggested_review_action=(
                    "Review fixed-mm and vertebral-bin profiles before comparison, or "
                    "resample the CT and all masks together onto a validated axial grid."
                ),
            )
        )
    if not extents:
        flags.append(
            QCFlag(
                code="vertebral_body_segmentation_empty",
                stage="measurement",
                severity=QCSeverity.ERROR,
                reason="No labeled vertebral-body instance is available for aggregation.",
            )
        )
    elif not signature_alignment.valid:
        flags.append(
            QCFlag(
                code="signature_reference_unresolved",
                stage="measurement",
                severity=QCSeverity.WARNING,
                reason=(
                    "The detected vertebral bodies could not define the "
                    "versioned longitudinal reference coordinate."
                ),
                observed={
                    "method": signature_alignment.method,
                    "reason": signature_alignment.reason,
                    "anchor_levels": list(signature_alignment.anchor_levels),
                    "slope_mm_per_level": (signature_alignment.slope_mm_per_level),
                },
                suggested_review_action=(
                    "Review vertebral enumeration and centroids before using "
                    "the longitudinal signature."
                ),
            )
        )
    elif signature_alignment.confidence == "low":
        flags.append(
            QCFlag(
                code="signature_reference_low_confidence",
                stage="measurement",
                severity=QCSeverity.WARNING,
                reason=(
                    "The longitudinal reference coordinate was inferred from "
                    "insufficient, distant, or inconsistent vertebral anchors."
                ),
                observed={
                    "method": signature_alignment.method,
                    "anchor_levels": list(signature_alignment.anchor_levels),
                    "slope_mm_per_level": (signature_alignment.slope_mm_per_level),
                    "residual_mm": signature_alignment.residual_mm,
                },
                suggested_review_action=(
                    "Confirm vertebral enumeration and the inferred L3 origin "
                    "before comparing the longitudinal signature."
                ),
            )
        )
    if landmarks is not None:
        flags.extend(landmarks.qc_flags)
    for extent in extents.values():
        if extent.valid and extent.complete:
            continue
        flags.append(
            QCFlag(
                code="vertebral_body_extent_invalid",
                stage="measurement",
                severity=(QCSeverity.ERROR if not extent.valid else QCSeverity.WARNING),
                reason="A detected vertebral-body extent is invalid or truncated.",
                observed={
                    "vertebral_level": extent.anatomical_label,
                    "missing_reason": extent.missing_reason,
                    "touches_fov": extent.touches_fov,
                    "removed_voxel_fraction": extent.removed_voxel_fraction,
                },
                thresholds={
                    "minimum_component_voxels": int(extent_settings["minimum_component_voxels"]),
                    "maximum_removed_fraction": float(extent_settings["maximum_removed_fraction"]),
                },
                suggested_review_action=(
                    "Review the vertebral-body mask and confirm truncation or segmentation failure."
                ),
            )
        )
    variant_labels = inferred_anatomical_variants(extents)
    if variant_labels and not any("variant" in flag.code for flag in flags):
        flags.append(
            QCFlag(
                code="anatomical_variant_present",
                stage="measurement",
                reason="A native transitional vertebral label is present and was not compressed.",
                observed={"vertebral_levels": variant_labels},
                suggested_review_action=(
                    "Confirm the native vertebral enumeration before fixed-range analysis."
                ),
            )
        )
    if bool(slices["tissue_exceeds_body_area"].any()):
        flags.append(
            QCFlag(
                code="tissue_exceeds_body_area",
                stage="measurement",
                severity=QCSeverity.ERROR,
                reason="At least one tissue-label voxel lies outside the aligned body mask.",
                observed={
                    "slice_count": int(slices["tissue_exceeds_body_area"].sum()),
                    "voxel_count": int(slices["tissue_outside_body_voxel_count"].sum()),
                },
            )
        )
    body_fov_count = int(slices["body_touching_fov"].sum())
    if body_fov_count:
        flags.append(
            QCFlag(
                code="body_surface_touches_fov",
                stage="measurement",
                reason=(
                    "The derived body surface touches the lateral image boundary "
                    "on one or more slices."
                ),
                observed={"slice_count": body_fov_count},
            )
        )
    trunk_fov_count = int(slices["trunk_touching_fov"].sum())
    if trunk_fov_count:
        flags.append(
            QCFlag(
                code="trunk_contour_touches_fov",
                stage="measurement",
                reason="The primary trunk contour touches the image boundary on one or more slices.",
                observed={"slice_count": trunk_fov_count},
            )
        )
    fragmented_slice_count = int(
        (slices["trunk_mask_fragmented"] & slices["trunk_voxel_count"].gt(0)).sum()
    )
    if fragmented_slice_count:
        flags.append(
            QCFlag(
                code="trunk_surface_fragmented_slices",
                stage="measurement",
                reason=(
                    "The trunk label has multiple axial components on one or more slices; "
                    "their area and circumference are ineligible for strict summaries."
                ),
                observed={"slice_count": fragmented_slice_count},
                suggested_review_action=(
                    "Review the body-surface mask for arm, device, table, or segmentation "
                    "components."
                ),
            )
        )
    for column, code, description in (
        (
            "body_mask_internal_gap",
            "body_surface_internal_gap",
            "The body mask is empty between non-empty body slices.",
        ),
        (
            "trunk_mask_internal_gap",
            "trunk_surface_internal_gap",
            "The trunk mask is empty between non-empty trunk slices.",
        ),
    ):
        gap_count = int(slices[column].sum())
        if gap_count:
            flags.append(
                QCFlag(
                    code=code,
                    stage="measurement",
                    severity=QCSeverity.ERROR,
                    reason=description,
                    observed={"slice_count": gap_count},
                    suggested_review_action=(
                        "Review the body-surface segmentation for a missing internal "
                        "slice or disconnected longitudinal support."
                    ),
                )
            )
    contour_failure = (
        slices["trunk_voxel_count"].gt(0)
        & ~slices["trunk_contour_valid"]
        & ~slices["trunk_touching_fov"]
        & ~slices["trunk_mask_fragmented"]
    )
    if bool(contour_failure.any()):
        flags.append(
            QCFlag(
                code="body_surface_contour_invalid",
                stage="measurement",
                reason=("A non-empty trunk mask did not yield a valid closed external contour."),
                observed={"slice_count": int(contour_failure.sum())},
                suggested_review_action="Review the affected axial contour overlays.",
            )
        )
    jump_count = int(slices["trunk_circumference_jump_flag"].sum())
    if jump_count:
        flags.append(
            QCFlag(
                code="trunk_circumference_discontinuity",
                stage="measurement",
                reason="The trunk circumference has an implausible adjacent-slice jump.",
                observed={"slice_count": jump_count},
                thresholds={
                    "maximum_gap_mm": float(circumference_qc["maximum_gap_mm"]),
                    "minimum_absolute_jump_cm": float(circumference_qc["minimum_absolute_jump_cm"]),
                    "minimum_relative_jump": float(circumference_qc["minimum_relative_jump"]),
                },
            )
        )
    summary = summaries.iloc[0]
    if not bool(summary["l3_200mm_slab_valid"]):
        flags.append(
            QCFlag(
                code="l3_200mm_slab_unavailable",
                stage="measurement",
                severity=(
                    QCSeverity.INFO
                    if summary["l3_200mm_slab_reason"] in {"outside_fov", "partial_fov"}
                    else QCSeverity.WARNING
                ),
                reason="The complete L3-centered 200-mm slab could not be measured.",
                observed={"reason": summary["l3_200mm_slab_reason"]},
                suggested_review_action=(
                    "Review L3 completeness and cranio-caudal acquisition coverage."
                ),
            )
        )
    for prefix, code, description in (
        (
            "ct_min_trunk_circumference_t10_l5",
            "minimum_waist_observed_ineligible",
            (
                "An observed T10-L5 minimum circumference is available, but its "
                "search is not eligible for unqualified comparison."
            ),
        ),
        (
            "ct_max_pelvic_circumference",
            "pelvic_maximum_observed_ineligible",
            (
                "An observed sacral pelvic maximum is available, but its search "
                "is not eligible for unqualified comparison."
            ),
        ),
    ):
        value_is_fov_cropped = bool(
            prefix == "ct_max_pelvic_circumference"
            and summary.get(f"{prefix}_value_is_fov_cropped", False)
        )
        if (
            not bool(summary.get(f"{prefix}_valid", False))
            or bool(summary.get(f"{prefix}_eligible", False))
            or (
                bool(summary.get(f"{prefix}_at_search_boundary", False))
                and not value_is_fov_cropped
            )
        ):
            continue
        if value_is_fov_cropped:
            code = "pelvic_maximum_fov_cropped"
            description = (
                "A numeric sacral pelvic observation is available from an "
                "FOV-cropped contour and may underestimate the true maximum."
            )
        flags.append(
            QCFlag(
                code=code,
                stage="measurement",
                severity=QCSeverity.INFO,
                reason=description,
                observed={
                    "reason": summary.get(f"{prefix}_reason"),
                    "acquisition_coverage_fraction": summary.get(
                        f"{prefix}_search_acquisition_coverage_fraction"
                    ),
                    "valid_contour_coverage_fraction": summary.get(
                        f"{prefix}_search_valid_contour_coverage_fraction"
                    ),
                },
                suggested_review_action=(
                    "Retain the eligibility flag and review anatomy, coverage, and "
                    "contours before using this observed extremum."
                ),
            )
        )
    for prefix, code, description in (
        (
            "ct_min_trunk_circumference_t10_l5",
            "minimum_waist_unavailable",
            "No valid observed T10-L5 minimum circumference could be measured.",
        ),
        (
            "ct_max_pelvic_circumference",
            "pelvic_maximum_unavailable",
            "No valid observed sacral pelvic maximum could be measured.",
        ),
    ):
        if bool(summary.get(f"{prefix}_valid", False)):
            continue
        reason = str(summary.get(f"{prefix}_reason") or "invalid_measurement")
        coverage = summary.get(f"{prefix}_search_valid_contour_coverage_fraction")
        coverage = None if pd.isna(coverage) else float(coverage)
        flags.append(
            QCFlag(
                code=code,
                stage="measurement",
                severity=(
                    QCSeverity.INFO
                    if reason in {"missing_anchor", "outside_fov", "partial_fov"}
                    else QCSeverity.WARNING
                ),
                reason=description,
                observed={
                    "reason": reason,
                    "valid_contour_coverage_fraction": coverage,
                },
                suggested_review_action=(
                    "Review the vertebral anchors, acquisition coverage, and axial "
                    "trunk contours before using this anthropometric summary."
                ),
            )
        )
    for prefix, code, description in (
        (
            "ct_min_trunk_circumference_t10_l5",
            "minimum_waist_at_search_boundary",
            "The T10-L5 minimum circumference occurs at the search boundary.",
        ),
        (
            "ct_max_pelvic_circumference",
            "pelvic_maximum_at_search_boundary",
            "The sacral pelvic maximum occurs at the search boundary.",
        ),
    ):
        if bool(summary.get(f"{prefix}_at_search_boundary", False)):
            flags.append(
                QCFlag(
                    code=code,
                    stage="measurement",
                    reason=description,
                    observed={
                        "position_superior_mm": summary.get(f"{prefix}_position_superior_mm"),
                        "slice_id": summary.get(f"{prefix}_slice_id"),
                    },
                    suggested_review_action=(
                        "Review acquisition coverage and do not use the extremum as an "
                        "unqualified anthropometric measure."
                    ),
                )
            )
    if (
        landmarks is not None
        and all(
            landmark.valid
            for landmark in (
                landmarks.lowest_rib_inferior,
                landmarks.iliac_crest_superior,
            )
        )
        and not bool(summary["ct_midwaist_circumference_valid"])
    ):
        flags.append(
            QCFlag(
                code="midwaist_measurement_unavailable",
                stage="measurement",
                reason="Valid anatomical landmarks did not yield a valid CT mid-waist plane.",
                observed={"reason": summary["ct_midwaist_circumference_reason"]},
                suggested_review_action="Review landmark coverage and the trunk contour.",
            )
        )
    incomplete_territories = [
        territory.anatomical_label for territory in territories.values() if not territory.complete
    ]
    if incomplete_territories:
        flags.append(
            QCFlag(
                code="vertebral_territory_incomplete",
                stage="measurement",
                severity=QCSeverity.INFO,
                reason=(
                    "One or more native vertebral territories are incomplete at an "
                    "acquisition edge or invalid vertebral extent."
                ),
                observed={"vertebral_levels": incomplete_territories},
                suggested_review_action=(
                    "Retain territory completeness and coverage when using observed "
                    "measurements from incomplete edge levels."
                ),
            )
        )
    sequence_gap_levels = [
        territory.anatomical_label
        for territory in territories.values()
        if territory.sequence_gap_cranial or territory.sequence_gap_caudal
    ]
    if sequence_gap_levels:
        flags.append(
            QCFlag(
                code="vertebral_territory_sequence_gap",
                stage="measurement",
                reason=(
                    "Centroid-midpoint assignment was retained across a native label "
                    "gap; the affected levels require enumeration review."
                ),
                observed={"vertebral_levels": sequence_gap_levels},
                suggested_review_action="Review the native vertebral enumeration.",
            )
        )
    return MeasurementBundle(
        identity=identity,
        slices=slices,
        vertebrae=vertebrae,
        summaries=summaries,
        signature=signature,
        hu_distributions=hu_distributions,
        body_surface=body_surface,
        vertebral_extents=extents,
        vertebral_territories=territories,
        landmarks=landmarks,
        provenance={
            "orientation": orientation,
            "measurement": {
                "schema_version": MEASUREMENT_SCHEMA_VERSION,
                "vertebral_territory_schema_version": (VERTEBRAL_TERRITORY_SCHEMA_VERSION),
                "signature_schema_version": str(signature["signature_schema_version"].iloc[0]),
                "signature_profile_id": str(signature["signature_profile_id"].iloc[0]),
                "tissue_hu_distribution_schema_version": str(
                    hu_distributions["distribution_schema_version"].iloc[0]
                ),
                "signature_components": [
                    {
                        "component": "longitudinal_anatomy",
                        "table": "signature.parquet",
                        "schema_version": str(signature["signature_schema_version"].iloc[0]),
                        "rows": len(signature),
                    },
                    {
                        "component": "global_tissue_hu",
                        "table": "hu_distributions.parquet",
                        "schema_version": str(
                            hu_distributions["distribution_schema_version"].iloc[0]
                        ),
                        "scope": str(hu_distributions["distribution_scope"].iloc[0]),
                        "source_semantics": str(hu_distributions["source_semantics"].iloc[0]),
                        "tissues": (
                            hu_distributions["tissue_key"].drop_duplicates().astype(str).tolist()
                        ),
                    },
                ],
                "signature_reference": {
                    "alignment_version": str(signature["reference_alignment_version"].iloc[0]),
                    "level": str(signature["reference_level"].iloc[0]),
                    "method": signature_alignment.method,
                    "confidence": signature_alignment.confidence,
                    "anchor_levels": list(signature_alignment.anchor_levels),
                    "origin_position_superior_mm": (
                        signature_alignment.origin_position_superior_mm
                    ),
                    "slope_mm_per_level": (signature_alignment.slope_mm_per_level),
                    "residual_mm": signature_alignment.residual_mm,
                    "review_required": signature_alignment.review_required,
                    "variant_sequence": signature_alignment.variant_sequence,
                    "reason": signature_alignment.reason,
                },
                "longitudinal_allocation": {
                    "method": "native_slice_center_v1",
                    "in_plane_superior_span_mm": in_plane_span_mm,
                    "maximum_unflagged_in_plane_superior_span_mm": (maximum_unflagged_span_mm),
                    "review_required": in_plane_span_mm > maximum_unflagged_span_mm,
                },
                "settings": dict(settings),
            },
            "tissue": {
                "backend_id": tissue_backend_id,
                "label_sha256": array_sha256(tissue_labels_zyx),
                "label_schema": {
                    str(label): name for label, name in sorted(tissue_label_schema.items())
                },
                "preprocessing": dict(tissue_preprocessing),
                "compartment_sha256": array_sha256(compartment_labels_zyx),
                "compartment_schema": {
                    str(label): name for label, name in sorted(compartment_label_schema.items())
                },
                "compartment_source": "raw_model_labels",
                "definitions": dict(settings.get("tissue_definitions", {})),
            },
            "body_surface": {
                "backend_id": body_surface.backend_id,
                "provenance": dict(body_surface.provenance),
            },
            "landmarks": (
                {
                    "backend_id": landmarks.lowest_rib_inferior.backend_id,
                    "provenance": dict(landmarks.provenance),
                    "lowest_rib_inferior": {
                        "position_superior_mm": (
                            landmarks.lowest_rib_inferior.position_superior_mm
                        ),
                        "valid": landmarks.lowest_rib_inferior.valid,
                        "uncertainty_mm": landmarks.lowest_rib_inferior.uncertainty_mm,
                        "reason": landmarks.lowest_rib_inferior.reason,
                    },
                    "iliac_crest_superior": {
                        "position_superior_mm": (
                            landmarks.iliac_crest_superior.position_superior_mm
                        ),
                        "valid": landmarks.iliac_crest_superior.valid,
                        "uncertainty_mm": landmarks.iliac_crest_superior.uncertainty_mm,
                        "reason": landmarks.iliac_crest_superior.reason,
                    },
                }
                if landmarks is not None
                else None
            ),
            "vertebral": {
                "backend_id": vertebral_result.backend_id,
                "provenance": dict(vertebral_result.provenance),
            },
        },
        qc_flags=tuple(flags),
    )
