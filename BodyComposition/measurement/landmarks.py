"""Anatomical waist landmarks derived from explicit segmentation labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from BodyComposition.measurement.contracts import Landmark, LandmarkSet
from BodyComposition.measurement.physical import (
    clean_largest_component,
    mask_physical_extent,
    validate_array_zyx,
)
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import QCFlag, QCSeverity


TOTALSEGMENTATOR_LANDMARK_BACKEND = "totalsegmentator_total_task297_landmarks_v1"


def _valid_component_bounds(
    labels_zyx: np.ndarray,
    label: int | None,
    geometry: ImageGeometry,
    minimum_voxels: int,
) -> tuple[float, float] | None:
    if label is None:
        return None
    component = clean_largest_component(labels_zyx == label)
    if component.retained_voxel_count < minimum_voxels or component.touches_fov:
        return None
    inferior, superior, _, _ = mask_physical_extent(component.mask_zyx, geometry)
    return inferior, superior


def landmarks_from_totalsegmentator(
    labels_zyx: np.ndarray,
    geometry: ImageGeometry,
    label_schema: Mapping[int, str],
    *,
    minimum_voxels: int = 20,
    maximum_side_disagreement_mm: float = 30.0,
    provenance: Mapping[str, Any] | None = None,
) -> LandmarkSet:
    """Locate the lowest paired rib margin and superior iliac crest."""

    if minimum_voxels <= 0:
        raise ValueError("minimum_voxels must be positive.")
    if maximum_side_disagreement_mm <= 0:
        raise ValueError("maximum_side_disagreement_mm must be positive.")
    labels = validate_array_zyx(labels_zyx, geometry, "landmark_labels_zyx")
    inverse = {str(name).lower(): int(label) for label, name in label_schema.items()}
    flags: list[QCFlag] = []

    rib_details: dict[str, Any] = {}
    rib_position = None
    rib_uncertainty = None
    for rib_number in range(12, 0, -1):
        left_name = f"rib_left_{rib_number}"
        right_name = f"rib_right_{rib_number}"
        left = _valid_component_bounds(labels, inverse.get(left_name), geometry, minimum_voxels)
        right = _valid_component_bounds(labels, inverse.get(right_name), geometry, minimum_voxels)
        if left is None or right is None:
            continue
        side_positions = (left[0], right[0])
        side_disagreement = float(abs(side_positions[0] - side_positions[1]))
        rib_position = float(np.mean(side_positions))
        rib_uncertainty = side_disagreement / 2.0
        rib_details = {
            "rib_number": rib_number,
            "left_inferior_mm": side_positions[0],
            "right_inferior_mm": side_positions[1],
            "side_disagreement_mm": side_disagreement,
        }
        break

    rib_valid = rib_position is not None and rib_uncertainty is not None
    if rib_valid and rib_details["side_disagreement_mm"] > maximum_side_disagreement_mm:
        rib_valid = False
        rib_reason = "uncertain_landmark"
    elif rib_valid:
        rib_reason = None
    else:
        rib_reason = "missing_landmark"
    if not rib_valid:
        flags.append(
            QCFlag(
                code="lowest_rib_landmark_invalid",
                stage="measurement",
                severity=(
                    QCSeverity.INFO
                    if rib_reason == "missing_landmark"
                    else QCSeverity.WARNING
                ),
                reason="A complete paired lowest-rib inferior landmark could not be established.",
                observed=rib_details,
            )
        )
    rib = Landmark(
        name="lowest_valid_rib_inferior_margin",
        position_superior_mm=rib_position,
        backend_id=TOTALSEGMENTATOR_LANDMARK_BACKEND,
        valid=rib_valid,
        uncertainty_mm=rib_uncertainty,
        reason=rib_reason,
        details=rib_details,
    )

    hip_positions: dict[str, float] = {}
    for side in ("left", "right"):
        bounds = _valid_component_bounds(
            labels,
            inverse.get(f"hip_{side}"),
            geometry,
            minimum_voxels,
        )
        if bounds is not None:
            hip_positions[side] = bounds[1]
    if len(hip_positions) == 2:
        hip_disagreement = float(abs(hip_positions["left"] - hip_positions["right"]))
        hip_position = float(np.mean(tuple(hip_positions.values())))
        hip_uncertainty = hip_disagreement / 2.0
        hip_valid = hip_disagreement <= maximum_side_disagreement_mm
        hip_reason = None if hip_valid else "uncertain_landmark"
    else:
        hip_position = None
        hip_uncertainty = None
        hip_valid = False
        hip_reason = "missing_landmark"
    if not hip_valid:
        flags.append(
            QCFlag(
                code="iliac_crest_landmark_invalid",
                stage="measurement",
                severity=(
                    QCSeverity.INFO
                    if hip_reason == "missing_landmark"
                    else QCSeverity.WARNING
                ),
                reason="A complete bilateral superior iliac-crest landmark could not be established.",
                observed=hip_positions,
            )
        )
    hip = Landmark(
        name="superior_iliac_crest",
        position_superior_mm=hip_position,
        backend_id=TOTALSEGMENTATOR_LANDMARK_BACKEND,
        valid=hip_valid,
        uncertainty_mm=hip_uncertainty,
        reason=hip_reason,
        details={
            **hip_positions,
            **(
                {"side_disagreement_mm": hip_disagreement}
                if len(hip_positions) == 2
                else {}
            ),
        },
    )
    if (
        rib.valid
        and hip.valid
        and rib.position_superior_mm is not None
        and hip.position_superior_mm is not None
        and rib.position_superior_mm <= hip.position_superior_mm
    ):
        rib = replace(rib, valid=False, reason="uncertain_landmark")
        hip = replace(hip, valid=False, reason="uncertain_landmark")
        flags.append(
            QCFlag(
                code="midwaist_landmark_order_invalid",
                stage="measurement",
                reason=(
                    "The lowest-rib inferior margin is not superior to the iliac-crest "
                    "landmark, so an anatomical mid-waist cannot be defined."
                ),
                observed={
                    "lowest_rib_inferior_mm": rib.position_superior_mm,
                    "iliac_crest_superior_mm": hip.position_superior_mm,
                },
            )
        )
    return LandmarkSet(
        lowest_rib_inferior=rib,
        iliac_crest_superior=hip,
        provenance={
            "upstream_project": "TotalSegmentator",
            "task": "total",
            "task_id": 297,
            "resampling_mm": 3.0,
            **dict(provenance or {}),
        },
        qc_flags=tuple(flags),
    )
