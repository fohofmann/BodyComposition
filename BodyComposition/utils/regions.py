"""Explicit, affine-aware image regions without implicit container state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import SimpleITK as sitk

from BodyComposition.measurement.physical import validate_array_zyx
from BodyComposition.utils.geometry import (
    ImageGeometry,
    assert_same_physical_domain,
)


@dataclass(frozen=True)
class ImageRegion:
    """One rectangular image-index region expressed in SimpleITK xyz order."""

    source_geometry: ImageGeometry
    start_xyz: tuple[int, int, int]
    size_xyz: tuple[int, int, int]

    def __post_init__(self) -> None:
        start = tuple(int(value) for value in self.start_xyz)
        size = tuple(int(value) for value in self.size_xyz)
        if len(start) != 3 or len(size) != 3:
            raise ValueError("Image regions require three xyz start and size values.")
        if any(value < 0 for value in start):
            raise ValueError("Image-region starts must be non-negative.")
        if any(value <= 0 for value in size):
            raise ValueError("Image-region sizes must be positive.")
        stop = tuple(
            start_value + size_value
            for start_value, size_value in zip(start, size, strict=True)
        )
        if any(
            stop_value > source_size
            for stop_value, source_size in zip(
                stop,
                self.source_geometry.size_xyz,
                strict=True,
            )
        ):
            raise ValueError(
                "Image region exceeds the source domain: "
                f"start={start}, size={size}, source={self.source_geometry.size_xyz}."
            )
        object.__setattr__(self, "start_xyz", start)
        object.__setattr__(self, "size_xyz", size)

    @property
    def stop_xyz(self) -> tuple[int, int, int]:
        return (
            self.start_xyz[0] + self.size_xyz[0],
            self.start_xyz[1] + self.size_xyz[1],
            self.start_xyz[2] + self.size_xyz[2],
        )

    @property
    def slices_zyx(self) -> tuple[slice, slice, slice]:
        start_zyx = tuple(reversed(self.start_xyz))
        stop_zyx = tuple(reversed(self.stop_xyz))
        return (
            slice(start_zyx[0], stop_zyx[0]),
            slice(start_zyx[1], stop_zyx[1]),
            slice(start_zyx[2], stop_zyx[2]),
        )

    @property
    def geometry(self) -> ImageGeometry:
        origin = self.source_geometry.physical_point(self.start_xyz)
        return ImageGeometry(
            size_xyz=self.size_xyz,
            spacing_xyz=self.source_geometry.spacing_xyz,
            origin_lps_xyz=(
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
            ),
            direction_lps=self.source_geometry.direction_lps,
        )

    def extract_array(self, array_zyx: np.ndarray) -> np.ndarray:
        source = validate_array_zyx(
            array_zyx,
            self.source_geometry,
            "source_array_zyx",
        )
        return np.array(source[self.slices_zyx], copy=True)

    def extract_image(self, image: sitk.Image) -> sitk.Image:
        source_geometry = ImageGeometry.from_sitk(image)
        assert_same_physical_domain(
            self.source_geometry,
            source_geometry,
            reference_name="region source",
            candidate_name="input image",
        )
        cropped = sitk.RegionOfInterest(
            image,
            size=list(self.size_xyz),
            index=list(self.start_xyz),
        )
        assert_same_physical_domain(
            self.geometry,
            ImageGeometry.from_sitk(cropped),
            reference_name="calculated region",
            candidate_name="SimpleITK region",
        )
        return cropped

    def restore_array(
        self,
        region_array_zyx: np.ndarray,
        *,
        fill_value: int | float = 0,
    ) -> np.ndarray:
        array = np.asarray(region_array_zyx)
        expected = tuple(reversed(self.size_xyz))
        if array.ndim != 3 or array.shape != expected:
            raise ValueError(
                f"Region array must have zyx shape {expected}, got {array.shape}."
            )
        restored = np.full(
            tuple(reversed(self.source_geometry.size_xyz)),
            fill_value,
            dtype=array.dtype,
        )
        restored[self.slices_zyx] = array
        return restored

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_xyz": list(self.start_xyz),
            "size_xyz": list(self.size_xyz),
            "stop_xyz_exclusive": list(self.stop_xyz),
            "source_size_xyz": list(self.source_geometry.size_xyz),
            "spacing_xyz": list(self.source_geometry.spacing_xyz),
            "origin_lps_xyz": list(self.geometry.origin_lps_xyz),
            "direction_lps": list(self.source_geometry.direction_lps),
            "operation": "index_crop_no_resampling",
        }


@dataclass(frozen=True)
class L3TissueAnalysisRegion:
    """L3 target slices plus the larger region supplied to the 3-D model."""

    inference_region: ImageRegion
    analyzed_slice_indices_z: tuple[int, ...]
    target_inferior_mm: float
    target_superior_mm: float
    inference_context_mm: float
    territory_complete: bool
    territory_reason: str | None

    def __post_init__(self) -> None:
        indices = tuple(int(value) for value in self.analyzed_slice_indices_z)
        if not indices:
            raise ValueError("An L3 analysis region requires at least one target slice.")
        if len(set(indices)) != len(indices):
            raise ValueError("L3 target slice indices must be unique.")
        size_z = self.inference_region.source_geometry.size_xyz[2]
        if any(index < 0 or index >= size_z for index in indices):
            raise ValueError("An L3 target slice lies outside the source image.")
        if not self.target_inferior_mm < self.target_superior_mm:
            raise ValueError("The L3 target interval must have positive physical height.")
        if not np.isfinite(self.inference_context_mm) or self.inference_context_mm < 0:
            raise ValueError("L3 inference context must be finite and non-negative.")
        inference_z = self.inference_region.slices_zyx[0]
        if any(
            index < int(inference_z.start) or index >= int(inference_z.stop)
            for index in indices
        ):
            raise ValueError("Every analyzed L3 slice must lie inside the inference region.")
        object.__setattr__(self, "analyzed_slice_indices_z", indices)

    @property
    def analyzed_slices_mask_z(self) -> np.ndarray:
        mask: np.ndarray = np.zeros(
            self.inference_region.source_geometry.size_xyz[2],
            dtype=bool,
        )
        mask[list(self.analyzed_slice_indices_z)] = True
        return mask

    def restore_target_labels(self, labels_zyx: np.ndarray) -> np.ndarray:
        restored = self.inference_region.restore_array(labels_zyx)
        restored[~self.analyzed_slices_mask_z] = 0
        return restored

    def as_dict(self) -> dict[str, Any]:
        indices = self.analyzed_slice_indices_z
        return {
            "scope": "l3_vertebral_level",
            "target_level": "L3",
            "target_inferior_position_superior_mm": float(self.target_inferior_mm),
            "target_superior_position_superior_mm": float(self.target_superior_mm),
            "analyzed_slice_count": len(indices),
            "analyzed_slice_index_zyx_z_min": min(indices),
            "analyzed_slice_index_zyx_z_max": max(indices),
            "inference_context_mm_each_side": float(self.inference_context_mm),
            "territory_complete": bool(self.territory_complete),
            "territory_reason": self.territory_reason,
            "inference_region": self.inference_region.as_dict(),
            "prediction_restore": "full_prepared_domain_zero_outside_l3_target",
        }
