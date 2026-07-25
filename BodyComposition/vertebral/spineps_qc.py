"""Pure-array construction and QC for native SPINEPS/VERIDAH outputs."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import QCFlag, QCSeverity, VertebralCentroid
from BodyComposition.vertebral.spineps_manifest import (
    SACRUM_BODY_SEMANTIC_LABEL,
    SPINEPS_AUXILIARY_INSTANCE_LABELS,
    SPINEPS_CRANIO_CAUDAL_ORDER,
    SPINEPS_NATIVE_LABELS,
    VERTEBRA_CORPUS_SEMANTIC_LABEL,
)

DEFAULT_MIN_BODY_VOLUME_MM3 = 500.0
DEFAULT_MAX_NEIGHBOR_VOLUME_RATIO = 2.5


def _validate_label_array(name: str, array: np.ndarray, geometry: ImageGeometry) -> None:
    expected_shape = tuple(reversed(geometry.size_xyz))
    if array.ndim != 3 or array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match geometry {expected_shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer labels.")
    if np.any(array < 0):
        raise ValueError(f"{name} contains negative labels.")


def construct_vertebral_body_labels(
    semantic_labels_zyx: np.ndarray,
    vertebra_labels_zyx: np.ndarray,
    geometry: ImageGeometry,
) -> np.ndarray:
    """Intersect native labels with the corresponding vertebral-body region."""

    _validate_label_array("semantic_labels_zyx", semantic_labels_zyx, geometry)
    _validate_label_array("vertebra_labels_zyx", vertebra_labels_zyx, geometry)
    output = np.zeros_like(vertebra_labels_zyx)
    corpus = (
        (semantic_labels_zyx == VERTEBRA_CORPUS_SEMANTIC_LABEL)
        & (vertebra_labels_zyx != 26)
    ) | (
        (semantic_labels_zyx == SACRUM_BODY_SEMANTIC_LABEL)
        & (vertebra_labels_zyx == 26)
    )
    output[corpus] = vertebra_labels_zyx[corpus]
    return output


def vertebra_only_instance_labels(
    instance_labels_zyx: np.ndarray,
    geometry: ImageGeometry,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Remove documented SPINEPS disc/endplate instances from a copied array."""

    _validate_label_array("instance_labels_zyx", instance_labels_zyx, geometry)
    present = {
        int(value)
        for value in np.unique(instance_labels_zyx)
        if value != 0
    }
    removed = tuple(sorted(present & SPINEPS_AUXILIARY_INSTANCE_LABELS))
    output = np.array(instance_labels_zyx, copy=True)
    if removed:
        output[np.isin(output, removed)] = 0
    return output, removed


def centroids_from_body_labels(
    vertebral_body_labels_zyx: np.ndarray,
    geometry: ImageGeometry,
) -> tuple[VertebralCentroid, ...]:
    _validate_label_array("vertebral_body_labels_zyx", vertebral_body_labels_zyx, geometry)
    result = []
    for label in sorted(int(value) for value in np.unique(vertebral_body_labels_zyx) if value != 0):
        centroid_zyx = tuple(float(value) for value in ndimage.center_of_mass(vertebral_body_labels_zyx == label))
        index_xyz = tuple(reversed(centroid_zyx))
        physical_lps = tuple(float(value) for value in geometry.physical_point(index_xyz))
        result.append(
            VertebralCentroid(
                native_label=label,
                anatomical_label=SPINEPS_NATIVE_LABELS.get(label, f"UNKNOWN_{label}"),
                index_zyx=centroid_zyx,
                physical_lps_xyz=physical_lps,
            )
        )
    return tuple(result)


def _flag(
    code: str,
    reason: str,
    *,
    severity: QCSeverity = QCSeverity.WARNING,
    observed: dict | None = None,
    thresholds: dict | None = None,
) -> QCFlag:
    return QCFlag(
        code=code,
        reason=reason,
        severity=severity,
        observed=observed or {},
        thresholds=thresholds or {},
    )


def _superior_boundary_labels(
    mask_zyx: np.ndarray,
    geometry: ImageGeometry,
) -> tuple[set[int], set[int]]:
    direction = geometry.direction_matrix_lps
    index_axis_xyz = int(np.argmax(np.abs(direction[2, :])))
    numpy_axis = 2 - index_axis_xyz
    low_plane = np.take(mask_zyx, 0, axis=numpy_axis)
    high_plane = np.take(mask_zyx, mask_zyx.shape[numpy_axis] - 1, axis=numpy_axis)
    low_labels = {int(value) for value in np.unique(low_plane) if value != 0}
    high_labels = {int(value) for value in np.unique(high_plane) if value != 0}
    if direction[2, index_axis_xyz] > 0:
        return high_labels, low_labels
    return low_labels, high_labels


def evaluate_spineps_qc(
    whole_vertebra_labels_zyx: np.ndarray,
    vertebral_body_labels_zyx: np.ndarray,
    geometry: ImageGeometry,
    *,
    semantic_labels_zyx: np.ndarray | None = None,
    min_body_volume_mm3: float = DEFAULT_MIN_BODY_VOLUME_MM3,
    max_neighbor_volume_ratio: float = DEFAULT_MAX_NEIGHBOR_VOLUME_RATIO,
) -> tuple[QCFlag, ...]:
    """Evaluate native labels without rewriting anatomical variants."""

    _validate_label_array("whole_vertebra_labels_zyx", whole_vertebra_labels_zyx, geometry)
    _validate_label_array("vertebral_body_labels_zyx", vertebral_body_labels_zyx, geometry)
    if semantic_labels_zyx is not None:
        _validate_label_array("semantic_labels_zyx", semantic_labels_zyx, geometry)
    if np.any((vertebral_body_labels_zyx != 0) & (vertebral_body_labels_zyx != whole_vertebra_labels_zyx)):
        raise ValueError("Vertebral-body labels are not a subset of whole vertebra labels.")

    flags: list[QCFlag] = []
    labels = {int(value) for value in np.unique(whole_vertebra_labels_zyx) if value != 0}
    body_labels = {int(value) for value in np.unique(vertebral_body_labels_zyx) if value != 0}
    unexpected = sorted(labels - set(SPINEPS_NATIVE_LABELS))
    if unexpected:
        flags.append(
            _flag(
                "unexpected_native_labels",
                "SPINEPS returned labels outside the pinned native CT schema.",
                severity=QCSeverity.ERROR,
                observed={"labels": unexpected},
            )
        )
    if not labels:
        flags.append(
            _flag(
                "empty_vertebra_mask",
                "SPINEPS returned no vertebral labels.",
                severity=QCSeverity.ERROR,
            )
        )
        return tuple(flags)

    if semantic_labels_zyx is not None:
        semantic_corpus = (
            semantic_labels_zyx == VERTEBRA_CORPUS_SEMANTIC_LABEL
        ) | (
            semantic_labels_zyx == SACRUM_BODY_SEMANTIC_LABEL
        )
        unlabeled_corpus = semantic_corpus & (whole_vertebra_labels_zyx == 0)
        unlabeled_count = int(np.count_nonzero(unlabeled_corpus))
        if unlabeled_count:
            corpus_count = int(np.count_nonzero(semantic_corpus))
            flags.append(
                _flag(
                    "semantic_corpus_without_vertebra_label",
                    "Semantic corpus voxels are not assigned to a native vertebral label.",
                    observed={
                        "voxels": unlabeled_count,
                        "fraction_of_semantic_corpus": unlabeled_count / corpus_count,
                    },
                )
            )

    missing_corpus = sorted(labels - body_labels)
    if missing_corpus:
        flags.append(
            _flag(
                "empty_corpus_intersection",
                "One or more native vertebrae have no semantic corpus intersection.",
                observed={"labels": missing_corpus},
            )
        )

    body_volumes = {}
    small_bodies = {}
    disconnected = {}
    structure = ndimage.generate_binary_structure(3, 1)
    for label in sorted(body_labels):
        label_mask = vertebral_body_labels_zyx == label
        volume_mm3 = float(np.count_nonzero(label_mask) * geometry.voxel_volume_mm3)
        body_volumes[label] = volume_mm3
        if volume_mm3 < min_body_volume_mm3:
            small_bodies[label] = volume_mm3
        _, components = ndimage.label(label_mask, structure=structure)
        if components > 1:
            disconnected[label] = int(components)
    if small_bodies:
        flags.append(
            _flag(
                "small_corpus_intersection",
                "One or more vertebral-body intersections are implausibly small.",
                observed={"volume_mm3_by_label": small_bodies},
                thresholds={"minimum_volume_mm3": min_body_volume_mm3},
            )
        )
    if disconnected:
        flags.append(
            _flag(
                "disconnected_vertebral_body",
                "One or more labeled vertebral bodies contain multiple components.",
                observed={"component_count_by_label": disconnected},
            )
        )

    centroids = centroids_from_body_labels(vertebral_body_labels_zyx, geometry)
    cranial_to_caudal = sorted(centroids, key=lambda centroid: centroid.physical_lps_xyz[2], reverse=True)
    active_order = [
        label
        for label in SPINEPS_CRANIO_CAUDAL_ORDER
        if label not in {28, 25} or label in body_labels
    ]
    rank = {label: index for index, label in enumerate(active_order)}
    ranked_labels = [centroid.native_label for centroid in cranial_to_caudal if centroid.native_label in rank]
    ranked_positions = [rank[label] for label in ranked_labels]
    if any(
        right <= left
        for left, right in zip(ranked_positions, ranked_positions[1:], strict=False)
    ):
        flags.append(
            _flag(
                "non_monotonic_vertebral_order",
                "Native labels are not monotonic along the physical superior axis.",
                severity=QCSeverity.ERROR,
                observed={"cranial_to_caudal_labels": ranked_labels},
            )
        )
    gaps = [
        (ranked_labels[index], ranked_labels[index + 1])
        for index in range(len(ranked_positions) - 1)
        if ranked_positions[index + 1] - ranked_positions[index] > 1
    ]
    if gaps:
        flags.append(
            _flag(
                "internal_label_gap",
                "The native vertebral sequence contains one or more internal gaps.",
                observed={"neighbor_pairs": gaps},
            )
        )

    implausibly_large = {}
    for index in range(1, len(ranked_labels) - 1):
        previous_label = ranked_labels[index - 1]
        label = ranked_labels[index]
        next_label = ranked_labels[index + 1]
        if (
            rank[label] - rank[previous_label] != 1
            or rank[next_label] - rank[label] != 1
        ):
            continue
        neighbor_reference = max(
            body_volumes[previous_label],
            body_volumes[next_label],
        )
        ratio = body_volumes[label] / neighbor_reference
        if ratio > max_neighbor_volume_ratio:
            implausibly_large[label] = ratio
    if implausibly_large:
        flags.append(
            _flag(
                "implausibly_large_corpus",
                "A vertebral-body volume is much larger than both adjacent native levels; a merged label is possible.",
                observed={"neighbor_volume_ratio_by_label": implausibly_large},
                thresholds={"maximum_neighbor_volume_ratio": max_neighbor_volume_ratio},
            )
        )

    if 28 in labels:
        flags.append(_flag("variant_t13", "VERIDAH labeled a T13 vertebra.", observed={"label": 28}))
    if 25 in labels:
        flags.append(_flag("variant_l6", "VERIDAH labeled an L6 vertebra.", observed={"label": 25}))
    if 26 in labels and 23 in labels and not ({24, 25} & labels):
        flags.append(
            _flag(
                "variant_four_lumbar_or_missing_l5",
                "The sequence reaches L4 and sacrum without an L5/L6 label; this is compatible with four lumbar vertebrae or a missing label.",
                observed={"labels": sorted(labels)},
            )
        )
    if 18 in labels and 20 in labels and not ({19, 28} & labels):
        flags.append(
            _flag(
                "variant_eleven_thoracic_or_missing_t12",
                "The sequence reaches T11 and L1 without T12/T13; this is compatible with eleven thoracic vertebrae or a missing label.",
                observed={"labels": sorted(labels)},
            )
        )
    if 26 in labels and not ({24, 25} & labels):
        flags.append(
            _flag(
                "incomplete_caudal_anchor_set",
                "A sacral label is present without an adjacent L5/L6 caudal anchor.",
                observed={"expected_any": [24, 25]},
            )
        )
    cervical = labels & set(range(1, 8))
    thoracic = labels & ({*range(8, 20), 28})
    lumbar = labels & set(range(20, 26))
    if cervical and thoracic and not {7, 8}.issubset(labels):
        flags.append(
            _flag(
                "incomplete_cranial_anchor_set",
                "The cervical-thoracic transition lacks the native C7/T1 anchor pair.",
                observed={"expected": [7, 8]},
            )
        )
    if thoracic and lumbar and not (20 in labels and bool({19, 28} & labels)):
        flags.append(
            _flag(
                "incomplete_thoracolumbar_anchor_set",
                "The thoracic-lumbar transition lacks a T12/T13-to-L1 anchor pair.",
                observed={"expected_cranial_any": [19, 28], "expected_caudal": 20},
            )
        )
    if 22 not in labels and {21, 23}.issubset(labels):
        flags.append(
            _flag(
                "l3_missing_between_adjacent_levels",
                "L3 is absent although both L2 and L4 are present.",
                severity=QCSeverity.ERROR,
            )
        )

    cranial_labels, caudal_labels = _superior_boundary_labels(whole_vertebra_labels_zyx, geometry)
    if cranial_labels:
        flags.append(
            _flag(
                "vertebra_touches_cranial_fov",
                "A vertebral prediction touches the cranial field-of-view boundary.",
                observed={"labels": sorted(cranial_labels)},
            )
        )
    if caudal_labels:
        flags.append(
            _flag(
                "vertebra_touches_caudal_fov",
                "A vertebral prediction touches the caudal field-of-view boundary.",
                observed={"labels": sorted(caudal_labels)},
            )
        )
    if 22 in cranial_labels or 22 in caudal_labels:
        flags.append(
            _flag(
                "l3_touches_fov_boundary",
                "The predicted L3 touches a cranial or caudal field-of-view boundary.",
                severity=QCSeverity.ERROR,
            )
        )
    return tuple(flags)
