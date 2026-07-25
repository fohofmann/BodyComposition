"""Adapt pinned SPINEPS/VERIDAH arrays to the common vertebral contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from BodyComposition.utils.geometry import GeometryError, ImageGeometry, assert_same_physical_domain
from BodyComposition.vertebral.contracts import ExecutionStatus, QCFlag, QCSeverity, VertebralResult
from BodyComposition.vertebral.spineps_manifest import (
    MODEL_WEIGHT_REDISTRIBUTION_MODE,
    SPINEPS_BACKEND_ID,
    SPINEPS_MODEL_ASSETS,
    SPINEPS_NATIVE_LABELS,
    SPINEPS_SOURCE_COMMIT,
    SPINEPS_VERSION,
    SPINEPS_WHEEL_SHA256,
    TPTBOX_COMPAT_VERSION,
    TPTBOX_LICENSE_STATUS,
    VIBESEG_CROP_ASSETS,
    VIBESEG_CROP_DATASET_ID,
    VIBESEG_CROP_RELEASE,
    VIBESEG_WEIGHT_LICENSE_STATUS,
)
from BodyComposition.vertebral.spineps_qc import (
    centroids_from_body_labels,
    construct_vertebral_body_labels,
    evaluate_spineps_qc,
    vertebra_only_instance_labels,
)

SUCCESS_CODES = {"OK", "ALL_DONE"}


def _geometry_from_sitk(image: sitk.Image) -> ImageGeometry:
    if image.GetDimension() != 3:
        raise GeometryError(f"SPINEPS output must be three-dimensional, got {image.GetDimension()}D.")
    return ImageGeometry(
        size_xyz=tuple(int(value) for value in image.GetSize()),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_lps_xyz=tuple(float(value) for value in image.GetOrigin()),
        direction_lps=tuple(float(value) for value in image.GetDirection()),
    )


def _base_provenance() -> dict[str, Any]:
    return {
        "backend_id": SPINEPS_BACKEND_ID,
        "spineps_version": SPINEPS_VERSION,
        "spineps_source_commit": SPINEPS_SOURCE_COMMIT,
        "spineps_wheel_sha256": SPINEPS_WHEEL_SHA256,
        "tptbox_compat_version": TPTBOX_COMPAT_VERSION,
        "tptbox_license_status": TPTBOX_LICENSE_STATUS,
        "model_assets": {
            asset.model_id: {
                "release": asset.release,
                "asset": asset.asset_name,
                "sha256": asset.sha256,
                "redistribution_mode": MODEL_WEIGHT_REDISTRIBUTION_MODE,
            }
            for asset in SPINEPS_MODEL_ASSETS
        },
        "required_crop_model": {
            "dataset_id": VIBESEG_CROP_DATASET_ID,
            "release": VIBESEG_CROP_RELEASE,
            "weight_license_status": VIBESEG_WEIGHT_LICENSE_STATUS,
            "redistribution_mode": MODEL_WEIGHT_REDISTRIBUTION_MODE,
            "archives": [
                {
                    "asset": asset.asset_name,
                    "sha256": asset.sha256,
                    "bytes": asset.bytes,
                }
                for asset in VIBESEG_CROP_ASSETS
            ],
        },
    }


def adapt_spineps_outputs(
    semantic_labels_zyx: np.ndarray | None,
    vertebra_labels_zyx: np.ndarray | None,
    geometry: ImageGeometry | None,
    *,
    upstream_error_code: str = "OK",
    native_outputs: Mapping[str, Path] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> VertebralResult:
    """Create a deterministic backend result without mutating upstream arrays."""

    combined_provenance = _base_provenance()
    combined_provenance.update(dict(provenance or {}))
    combined_provenance["upstream_error_code"] = upstream_error_code

    if upstream_error_code not in SUCCESS_CODES:
        return VertebralResult(
            backend_id=SPINEPS_BACKEND_ID,
            execution_status=ExecutionStatus.FAILED,
            geometry=geometry,
            whole_vertebra_labels=None,
            vertebral_body_labels=None,
            native_outputs=dict(native_outputs or {}),
            provenance=combined_provenance,
            qc_flags=(
                QCFlag(
                    code="upstream_spineps_failure",
                    reason="SPINEPS returned a non-success error code.",
                    severity=QCSeverity.ERROR,
                    observed={"error_code": upstream_error_code},
                ),
            ),
            error_summary=f"SPINEPS failed with error code {upstream_error_code}.",
        )
    if semantic_labels_zyx is None or vertebra_labels_zyx is None or geometry is None:
        return VertebralResult(
            backend_id=SPINEPS_BACKEND_ID,
            execution_status=ExecutionStatus.FAILED,
            geometry=geometry,
            whole_vertebra_labels=None,
            vertebral_body_labels=None,
            native_outputs=dict(native_outputs or {}),
            provenance=combined_provenance,
            qc_flags=(
                QCFlag(
                    code="missing_upstream_outputs",
                    reason="SPINEPS reported success without all required arrays and geometry.",
                    severity=QCSeverity.ERROR,
                ),
            ),
            error_summary="SPINEPS success response omitted required outputs.",
        )

    whole, removed_auxiliary_labels = vertebra_only_instance_labels(
        np.asarray(vertebra_labels_zyx),
        geometry,
    )
    combined_provenance["instance_label_policy"] = {
        "output": "vertebrae_only",
        "removed_auxiliary_labels": list(removed_auxiliary_labels),
        "native_outputs_preserved": True,
    }
    body = construct_vertebral_body_labels(np.asarray(semantic_labels_zyx), whole, geometry)
    flags = evaluate_spineps_qc(
        whole,
        body,
        geometry,
        semantic_labels_zyx=np.asarray(semantic_labels_zyx),
    )
    centroids = centroids_from_body_labels(body, geometry)
    return VertebralResult(
        backend_id=SPINEPS_BACKEND_ID,
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=whole,
        vertebral_body_labels=body,
        centroids=centroids,
        label_schema=SPINEPS_NATIVE_LABELS,
        native_outputs=dict(native_outputs or {}),
        provenance=combined_provenance,
        qc_flags=flags,
    )


def adapt_spineps_sitk_outputs(
    semantic_image: sitk.Image,
    vertebra_image: sitk.Image,
    reference_geometry: ImageGeometry,
    *,
    upstream_error_code: str = "OK",
    native_outputs: Mapping[str, Path] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> VertebralResult:
    """Convert SPINEPS images through the explicit SimpleITK ``zyx`` boundary."""

    try:
        semantic_geometry = _geometry_from_sitk(semantic_image)
        vertebra_geometry = _geometry_from_sitk(vertebra_image)
        assert_same_physical_domain(
            reference_geometry,
            semantic_geometry,
            reference_name="prepared CT",
            candidate_name="SPINEPS semantic output",
        )
        assert_same_physical_domain(
            reference_geometry,
            vertebra_geometry,
            reference_name="prepared CT",
            candidate_name="SPINEPS vertebra output",
        )
    except GeometryError as error:
        combined_provenance = _base_provenance()
        combined_provenance.update(dict(provenance or {}))
        combined_provenance["upstream_error_code"] = upstream_error_code
        return VertebralResult(
            backend_id=SPINEPS_BACKEND_ID,
            execution_status=ExecutionStatus.FAILED,
            geometry=reference_geometry,
            whole_vertebra_labels=None,
            vertebral_body_labels=None,
            native_outputs=dict(native_outputs or {}),
            provenance=combined_provenance,
            qc_flags=(
                QCFlag(
                    code="physical_domain_mismatch",
                    reason="A SPINEPS output does not share the prepared CT physical domain.",
                    severity=QCSeverity.ERROR,
                    observed={"error": str(error)},
                ),
            ),
            error_summary=str(error),
        )

    semantic_zyx = sitk.GetArrayFromImage(semantic_image)
    vertebra_zyx = sitk.GetArrayFromImage(vertebra_image)
    return adapt_spineps_outputs(
        semantic_zyx,
        vertebra_zyx,
        reference_geometry,
        upstream_error_code=upstream_error_code,
        native_outputs=native_outputs,
        provenance=provenance,
    )
