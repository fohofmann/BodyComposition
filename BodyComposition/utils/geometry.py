from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


GEOMETRY_ATOL = 1e-5
ORTHONORMALITY_ATOL = 1e-4


class GeometryError(ValueError):
    """Raised when image geometry is invalid or physically incompatible."""


def _as_tuple(values: Iterable, length: int, name: str, cast=float) -> tuple:
    result = tuple(cast(value) for value in values)
    if len(result) != length:
        raise GeometryError(f"{name} must contain {length} values, got {len(result)}.")
    return result


@dataclass(frozen=True)
class ImageGeometry:
    """Physical geometry for a three-dimensional image in SimpleITK order."""

    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_lps_xyz: tuple[float, float, float]
    direction_lps: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "size_xyz", _as_tuple(self.size_xyz, 3, "size_xyz", int))
        object.__setattr__(self, "spacing_xyz", _as_tuple(self.spacing_xyz, 3, "spacing_xyz"))
        object.__setattr__(self, "origin_lps_xyz", _as_tuple(self.origin_lps_xyz, 3, "origin_lps_xyz"))
        object.__setattr__(self, "direction_lps", _as_tuple(self.direction_lps, 9, "direction_lps"))
        self.validate()

    @classmethod
    def from_container(cls, container) -> "ImageGeometry":
        shape_zyx = container.shape
        if shape_zyx is None or len(shape_zyx) != 3:
            raise GeometryError("A three-dimensional image array is required.")
        return cls(
            size_xyz=tuple(int(value) for value in reversed(shape_zyx)),
            spacing_xyz=container.spacing,
            origin_lps_xyz=container.origin,
            direction_lps=container.direction,
        )

    @property
    def direction_matrix_lps(self) -> np.ndarray:
        return np.asarray(self.direction_lps, dtype=float).reshape(3, 3)

    @property
    def basis_lps(self) -> np.ndarray:
        """Physical basis vectors as columns for x, y, and z image indices."""
        return self.direction_matrix_lps @ np.diag(self.spacing_xyz)

    @property
    def affine_lps(self) -> np.ndarray:
        affine = np.eye(4, dtype=float)
        affine[:3, :3] = self.basis_lps
        affine[:3, 3] = self.origin_lps_xyz
        return affine

    @property
    def in_plane_area_mm2(self) -> float:
        basis = self.basis_lps
        return float(np.linalg.norm(np.cross(basis[:, 0], basis[:, 1])))

    @property
    def voxel_volume_mm3(self) -> float:
        return float(abs(np.linalg.det(self.basis_lps)))

    def physical_point(self, index_xyz: Sequence[float]) -> np.ndarray:
        index = np.asarray(index_xyz, dtype=float)
        if index.shape[-1] != 3:
            raise GeometryError("index_xyz must end in three coordinates.")
        return np.asarray(self.origin_lps_xyz) + index @ self.basis_lps.T

    def validate(self) -> None:
        size = np.asarray(self.size_xyz)
        spacing = np.asarray(self.spacing_xyz, dtype=float)
        origin = np.asarray(self.origin_lps_xyz, dtype=float)
        direction = self.direction_matrix_lps

        if np.any(size <= 0):
            raise GeometryError(f"Image size must be positive, got {self.size_xyz}.")
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise GeometryError(f"Image spacing must be finite and positive, got {self.spacing_xyz}.")
        if not np.all(np.isfinite(origin)):
            raise GeometryError(f"Image origin must be finite, got {self.origin_lps_xyz}.")
        if not np.all(np.isfinite(direction)):
            raise GeometryError("Image direction contains non-finite values.")
        if abs(np.linalg.det(direction)) <= np.finfo(float).eps:
            raise GeometryError("Image direction matrix is not invertible.")

        residual = np.max(np.abs(direction.T @ direction - np.eye(3)))
        if residual > ORTHONORMALITY_ATOL:
            raise GeometryError(
                "Image direction matrix is not orthonormal "
                f"(maximum residual {residual:.3g}, tolerance {ORTHONORMALITY_ATOL})."
            )

    def equivalent_to(self, other: "ImageGeometry", atol: float = GEOMETRY_ATOL) -> bool:
        return (
            self.size_xyz == other.size_xyz
            and np.allclose(self.spacing_xyz, other.spacing_xyz, atol=atol, rtol=0)
            and np.allclose(self.origin_lps_xyz, other.origin_lps_xyz, atol=atol, rtol=0)
            and np.allclose(self.direction_lps, other.direction_lps, atol=atol, rtol=0)
        )


def assert_same_physical_domain(
    reference: ImageGeometry,
    candidate: ImageGeometry,
    *,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
    atol: float = GEOMETRY_ATOL,
) -> None:
    if reference.size_xyz != candidate.size_xyz:
        raise GeometryError(
            f"{reference_name} and {candidate_name} sizes differ: "
            f"{reference.size_xyz} != {candidate.size_xyz}."
        )
    for field in ("spacing_xyz", "origin_lps_xyz", "direction_lps"):
        left = getattr(reference, field)
        right = getattr(candidate, field)
        if not np.allclose(left, right, atol=atol, rtol=0):
            raise GeometryError(
                f"{reference_name} and {candidate_name} {field} differ: {left} != {right}."
            )
