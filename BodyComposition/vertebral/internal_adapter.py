"""Common-contract adapter for selectable internal vertebral-body models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import (
    ExecutionStatus,
    QCFlag,
    QCSeverity,
    VertebralCentroid,
    VertebralResult,
)

INTERNAL_BACKEND_IDS = {
    "ResEncL": "vertebral_bodies_resenc_l",
    "ResEncM": "vertebral_bodies_resenc_m",
}


def adapt_internal_vertebral_bodies(
    labels_zyx: np.ndarray,
    geometry: ImageGeometry,
    label_schema: Mapping[int, str],
    *,
    model_preset: str,
    native_outputs: Mapping[str, Path] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> VertebralResult:
    """Adapt a body-only internal result without inventing whole-vertebra labels."""

    if model_preset not in INTERNAL_BACKEND_IDS:
        raise ValueError(f"Unsupported internal vertebral model preset: {model_preset}.")
    labels = np.asarray(labels_zyx)
    expected_shape = tuple(reversed(geometry.size_xyz))
    if labels.ndim != 3 or labels.shape != expected_shape:
        raise ValueError(
            f"Internal vertebral labels shape {labels.shape} does not match {expected_shape}."
        )
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise TypeError("Internal vertebral labels must be non-negative integers.")

    flags: list[QCFlag] = []
    present = sorted(int(value) for value in np.unique(labels) if value != 0)
    unknown = [value for value in present if value not in label_schema]
    if not present:
        flags.append(
            QCFlag(
                code="empty_vertebra_mask",
                reason="The internal vertebral backend returned no labeled bodies.",
                severity=QCSeverity.ERROR,
            )
        )
    if unknown:
        flags.append(
            QCFlag(
                code="unexpected_native_labels",
                reason="The internal vertebral backend returned unknown native labels.",
                severity=QCSeverity.ERROR,
                observed={"labels": unknown},
            )
        )

    centroids: list[VertebralCentroid] = []
    structure = ndimage.generate_binary_structure(3, 1)
    for label in present:
        mask = labels == label
        center_zyx = tuple(float(value) for value in ndimage.center_of_mass(mask))
        center_xyz = tuple(reversed(center_zyx))
        physical = tuple(float(value) for value in geometry.physical_point(center_xyz))
        centroids.append(
            VertebralCentroid(
                native_label=label,
                anatomical_label=label_schema.get(label, f"UNKNOWN_{label}"),
                index_zyx=center_zyx,
                physical_lps_xyz=physical,
            )
        )
        _, components = ndimage.label(mask, structure=structure)
        if components > 1:
            flags.append(
                QCFlag(
                    code="disconnected_vertebral_body",
                    reason="A labeled vertebral body contains multiple components.",
                    observed={"label": label, "component_count": int(components)},
                )
            )

    return VertebralResult(
        backend_id=INTERNAL_BACKEND_IDS[model_preset],
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=None,
        vertebral_body_labels=labels,
        centroids=tuple(centroids),
        label_schema=dict(label_schema),
        native_outputs=dict(native_outputs or {}),
        provenance={
            "backend_id": INTERNAL_BACKEND_IDS[model_preset],
            "model_preset": model_preset,
            **dict(provenance or {}),
        },
        qc_flags=tuple(flags),
    )
