"""Explicit analysis-region preparation for scoped tissue inference."""

from __future__ import annotations

from pathlib import Path

from BodyComposition.measurement.aggregation import (
    derive_vertebral_extents,
    derive_vertebral_territories,
)
from BodyComposition.measurement.physical import slice_geometry_table
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer
from BodyComposition.utils.regions import ImageRegion, L3TissueAnalysisRegion

L3_TISSUE_INPUT = "tmp/l3_tissue_input"
L3_ANALYSIS_REGION = "tmp/tissue_analysis_region"


class L3RegionUnavailableError(RuntimeError):
    """Raised when the requested L3-only analysis cannot be localized."""


def _overlapping_slice_indices(
    table,
    inferior_mm: float,
    superior_mm: float,
) -> tuple[int, ...]:
    overlaps = (
        table["slice_slab_superior_mm"].to_numpy(dtype=float) > inferior_mm
    ) & (
        table["slice_slab_inferior_mm"].to_numpy(dtype=float) < superior_mm
    )
    return tuple(
        int(value)
        for value in table.loc[overlaps, "slice_index_zyx_z"].to_numpy(dtype=int)
    )


class PrepareL3TissueRegion(PipelineAction):
    """Prepare a full-axial-FOV z crop after vertebral localization."""

    def __init__(self, pipeline):
        super().__init__(pipeline)
        settings = self.config["analysis"]
        if settings["scope"] != "l3_vertebral_level":
            raise ValueError(
                "PrepareL3TissueRegion requires analysis.scope=l3_vertebral_level."
            )
        self.context_mm = float(settings["l3"]["inference_context_mm"])
        self.extent_settings = self.config["measurements"]["vertebral_extent"]
        self.io_inputs = ["tmp/index", "tmp/prepared_image", "tmp/vertebral_result"]
        self.io_outputs = [L3_TISSUE_INPUT, L3_ANALYSIS_REGION]

    def __call__(self, memory):
        super().__call__(memory)
        source = memory["tmp/index"]
        if not isinstance(source, NiftiDataContainer):
            raise TypeError("tmp/index must be a NiftiDataContainer.")
        source.validate()
        prepared_geometry = ImageGeometry.from_sitk(
            memory["tmp/prepared_image"].prepared_image
        )
        assert_same_physical_domain(
            prepared_geometry,
            source.geometry,
            reference_name="orientation-prepared CT",
            candidate_name="L3 region source",
        )

        result = memory["tmp/vertebral_result"]
        if result.geometry is None:
            raise L3RegionUnavailableError(
                "L3-only analysis requires a successful vertebral result with geometry."
            )
        assert_same_physical_domain(
            source.geometry,
            result.geometry,
            reference_name="orientation-prepared CT",
            candidate_name="vertebral-body result",
        )
        extents = derive_vertebral_extents(
            result,
            minimum_component_voxels=int(
                self.extent_settings["minimum_component_voxels"]
            ),
            maximum_removed_fraction=float(
                self.extent_settings["maximum_removed_fraction"]
            ),
        )
        l3_extent = extents.get("L3")
        if (
            l3_extent is None
            or not l3_extent.valid
            or l3_extent.inferior_mm is None
            or l3_extent.superior_mm is None
        ):
            raise L3RegionUnavailableError(
                "The low-resource preset could not identify a valid L3 vertebral body; "
                "tissue inference was not run on another level or the complete CT."
            )
        territory = derive_vertebral_territories(extents).get("L3")
        if (
            territory is not None
            and territory.inferior_mm is not None
            and territory.superior_mm is not None
        ):
            target_inferior = float(territory.inferior_mm)
            target_superior = float(territory.superior_mm)
            territory_complete = bool(territory.complete)
            territory_reason = territory.missing_reason
        else:
            target_inferior = float(l3_extent.inferior_mm)
            target_superior = float(l3_extent.superior_mm)
            territory_complete = False
            territory_reason = "missing_neighbor"

        slices = slice_geometry_table(source.geometry)
        target_indices = _overlapping_slice_indices(
            slices,
            target_inferior,
            target_superior,
        )
        if not target_indices:
            raise L3RegionUnavailableError(
                "The localized L3 territory does not overlap an acquired CT slice."
            )
        inference_indices = _overlapping_slice_indices(
            slices,
            target_inferior - self.context_mm,
            target_superior + self.context_mm,
        )
        if not inference_indices:
            raise L3RegionUnavailableError(
                "No acquired CT slices overlap the requested L3 inference context."
            )
        start_z = min(inference_indices)
        stop_z = max(inference_indices) + 1
        region = ImageRegion(
            source_geometry=source.geometry,
            start_xyz=(0, 0, start_z),
            size_xyz=(
                source.geometry.size_xyz[0],
                source.geometry.size_xyz[1],
                stop_z - start_z,
            ),
        )
        analysis_region = L3TissueAnalysisRegion(
            inference_region=region,
            analyzed_slice_indices_z=tuple(sorted(target_indices)),
            target_inferior_mm=target_inferior,
            target_superior_mm=target_superior,
            inference_context_mm=self.context_mm,
            territory_complete=territory_complete,
            territory_reason=territory_reason,
        )
        cropped = NiftiDataContainer(
            Path(memory["workspace"]) / "tmp" / "l3_tissue_input.nii.gz"
        )
        cropped.img = region.extract_image(source.img)
        cropped.validate()
        assert_same_physical_domain(
            region.geometry,
            cropped.geometry,
            reference_name="calculated L3 inference region",
            candidate_name="cropped L3 model input",
        )
        memory[L3_TISSUE_INPUT] = cropped
        memory[L3_ANALYSIS_REGION] = analysis_region
