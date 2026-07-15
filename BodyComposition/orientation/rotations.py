"""Rotation conventions used by the CTDeepRot 2D model.

The rotation table and projection transform reproduce the CTDeepRot reference
implementation at commit 492114b8f9f3a7f058d4e97c0dd3643fb8d39649.
That project is Copyright (c) 2020 Jakubicek Roman and distributed under the
BSD 3-Clause license; see THIRD_PARTY_NOTICES.md.

Model arrays are explicitly ordered ``(y, x, z)``. SimpleITK arrays are
ordered ``(z, y, x)`` and are converted only at named boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class RotationClass:
    """One of the 24 proper axis-aligned rotations used by CTDeepRot."""

    index: int
    angles_deg: tuple[int, int, int]
    projection_permutation: tuple[int, int, int]
    projection_flips: tuple[bool, bool, bool]
    projection_quarter_turns: tuple[int, int, int]


# Columns are the upstream CSV fields:
# angles, projection permutation, projection flips, and 2D quarter turns.
_ROTATION_ROWS = (
    (180, 0, 0, 1, 2, 3, 0, 1, 1, 2, 0, 0),
    (180, 0, 90, 2, 1, 3, 1, 1, 1, 0, 0, 1),
    (180, 0, 180, 1, 2, 3, 1, 0, 1, 0, 2, 2),
    (180, 0, 270, 2, 1, 3, 0, 0, 1, 2, 2, 3),
    (180, 180, 0, 1, 2, 3, 1, 1, 0, 2, 2, 2),
    (180, 180, 90, 2, 1, 3, 1, 0, 0, 2, 0, 3),
    (180, 180, 180, 1, 2, 3, 0, 0, 0, 0, 0, 0),
    (180, 180, 270, 2, 1, 3, 0, 1, 0, 0, 2, 1),
    (270, 0, 0, 1, 3, 2, 0, 1, 0, 3, 0, 0),
    (270, 0, 90, 3, 1, 2, 1, 1, 0, 0, 3, 1),
    (270, 0, 180, 1, 3, 2, 1, 0, 0, 3, 2, 2),
    (270, 0, 270, 3, 1, 2, 0, 0, 0, 2, 3, 3),
    (270, 90, 0, 2, 3, 1, 1, 1, 0, 1, 1, 0),
    (270, 90, 90, 3, 2, 1, 1, 0, 0, 1, 1, 1),
    (270, 90, 180, 2, 3, 1, 0, 0, 0, 1, 1, 2),
    (270, 90, 270, 3, 2, 1, 0, 1, 0, 1, 1, 3),
    (270, 180, 0, 1, 3, 2, 1, 1, 1, 1, 2, 2),
    (270, 180, 90, 3, 1, 2, 1, 0, 1, 2, 1, 3),
    (270, 180, 180, 1, 3, 2, 0, 0, 1, 1, 0, 0),
    (270, 180, 270, 3, 1, 2, 0, 1, 1, 0, 1, 1),
    (270, 270, 0, 2, 3, 1, 0, 1, 1, 3, 3, 2),
    (270, 270, 90, 3, 2, 1, 1, 1, 1, 3, 3, 3),
    (270, 270, 180, 2, 3, 1, 1, 0, 1, 3, 3, 0),
    (270, 270, 270, 3, 2, 1, 0, 0, 1, 3, 3, 1),
)


ROTATION_CLASSES = tuple(
    RotationClass(
        index=index,
        angles_deg=tuple(int(value) for value in row[:3]),
        projection_permutation=tuple(int(value) - 1 for value in row[3:6]),
        projection_flips=tuple(bool(value) for value in row[6:9]),
        projection_quarter_turns=tuple(int(value) for value in row[9:12]),
    )
    for index, row in enumerate(_ROTATION_ROWS)
)


def rotation_class(index: int) -> RotationClass:
    try:
        return ROTATION_CLASSES[index]
    except IndexError as exc:
        raise ValueError(f"Rotation class must be in [0, 23], got {index}.") from exc


def apply_volume_rotation(array_yxz: np.ndarray, class_index: int) -> np.ndarray:
    """Apply an upstream CTDeepRot rotation to a ``(y, x, z)`` array."""

    if array_yxz.ndim != 3:
        raise ValueError("CTDeepRot volume rotation requires a three-dimensional array_yxz.")
    a, b, c = rotation_class(class_index).angles_deg
    output = np.rot90(array_yxz, a // 90, axes=(1, 2))
    output = np.rot90(output, b // 90, axes=(0, 2))
    return np.rot90(output, c // 90, axes=(0, 1))


def apply_projection_rotation(features_hwc: np.ndarray, class_index: int) -> np.ndarray:
    """Rotate one mean/max/std projection triplet as in CTDeepRot training."""

    if features_hwc.ndim != 3 or features_hwc.shape[2] != 3:
        raise ValueError("Projection features must have shape (height, width, 3).")
    spec = rotation_class(class_index)
    channels = features_hwc[:, :, spec.projection_permutation].copy()
    output = []
    for channel_index in range(3):
        channel = channels[:, :, channel_index]
        if spec.projection_flips[channel_index]:
            channel = np.fliplr(channel)
        output.append(np.rot90(channel, spec.projection_quarter_turns[channel_index]))
    return np.stack(output, axis=2)


def _array_key(array: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    contiguous = np.ascontiguousarray(array)
    return contiguous.shape, contiguous.tobytes()


@lru_cache(maxsize=1)
def _group_tables() -> tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    probe = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5)
    keys = {
        _array_key(apply_volume_rotation(probe, class_index)): class_index
        for class_index in range(24)
    }
    if len(keys) != 24:
        raise RuntimeError("CTDeepRot rotation table does not contain 24 unique rotations.")

    identity_key = _array_key(probe)
    if identity_key not in keys:
        raise RuntimeError("CTDeepRot rotation table has no identity operation.")
    identity_index = keys[identity_key]

    composition = []
    for outer in range(24):
        row = []
        for inner in range(24):
            transformed = apply_volume_rotation(
                apply_volume_rotation(probe, inner),
                outer,
            )
            row.append(keys[_array_key(transformed)])
        composition.append(tuple(row))

    inverse = []
    for class_index in range(24):
        candidates = [
            candidate
            for candidate in range(24)
            if composition[candidate][class_index] == identity_index
            and composition[class_index][candidate] == identity_index
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Rotation class {class_index} has no unique inverse.")
        inverse.append(candidates[0])
    return identity_index, tuple(inverse), tuple(composition)


def identity_class_index() -> int:
    return _group_tables()[0]


def inverse_class_index(class_index: int) -> int:
    rotation_class(class_index)
    return _group_tables()[1][class_index]


def compose_class_indices(outer: int, inner: int) -> int:
    """Return the class for applying ``inner`` first and ``outer`` second."""

    rotation_class(outer)
    rotation_class(inner)
    return _group_tables()[2][outer][inner]


def recover_unaugmented_class(augmentation: int, prediction: int) -> int:
    """Map a prediction on an augmented volume back to the original frame."""

    return compose_class_indices(inverse_class_index(augmentation), prediction)


@lru_cache(maxsize=48)
def array_transform_spec(
    class_index: int,
    inverse: bool = False,
) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    """Return output-axis permutation and flips for a model-array rotation."""

    selected = inverse_class_index(class_index) if inverse else class_index
    shape = (2, 3, 5)
    coordinates = np.indices(shape, dtype=np.int16)
    rotated = [apply_volume_rotation(coordinate, selected) for coordinate in coordinates]
    output_shape = rotated[0].shape
    permutation = tuple(shape.index(length) for length in output_shape)
    flips = []
    origin = (0, 0, 0)
    for output_axis, input_axis in enumerate(permutation):
        if output_shape[output_axis] < 2:
            flips.append(False)
            continue
        adjacent = [0, 0, 0]
        adjacent[output_axis] = 1
        flips.append(
            rotated[input_axis][tuple(adjacent)]
            < rotated[input_axis][origin]
        )
    return permutation, tuple(bool(value) for value in flips)


def signed_permutation_matrix(
    class_index: int,
    inverse: bool = False,
) -> np.ndarray:
    """Return output-axis to input-axis signed permutation in model order."""

    permutation, flips = array_transform_spec(class_index, inverse=inverse)
    matrix = np.zeros((3, 3), dtype=int)
    for output_axis, input_axis in enumerate(permutation):
        matrix[output_axis, input_axis] = -1 if flips[output_axis] else 1
    return matrix


def is_axial_half_turn(class_index: int) -> bool:
    """Return whether a class is the SI-preserving 180-degree in-plane turn.

    This rotation reverses both transverse axes while leaving the
    cranio-caudal axis in place.  It is anatomically more ambiguous than the
    other proper rotations and is handled by a separate repair gate.
    """

    permutation, flips = array_transform_spec(class_index)
    return permutation == (0, 1, 2) and flips == (True, True, False)


def sitk_transform_spec(
    class_index: int,
    inverse: bool = False,
) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    """Convert a model ``(y,x,z)`` transform to SimpleITK ``(x,y,z)`` axes."""

    model_permutation, model_flips = array_transform_spec(class_index, inverse=inverse)
    model_to_sitk = (1, 0, 2)
    order = [0, 0, 0]
    flips = [False, False, False]
    for output_model_axis, input_model_axis in enumerate(model_permutation):
        output_sitk_axis = model_to_sitk[output_model_axis]
        order[output_sitk_axis] = model_to_sitk[input_model_axis]
        flips[output_sitk_axis] = model_flips[output_model_axis]
    return tuple(order), tuple(flips)
