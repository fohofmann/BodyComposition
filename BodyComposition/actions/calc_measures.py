from __future__ import annotations

import logging
from time import time

from BodyComposition.measurements import calculate_slice_measurements
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import assert_same_physical_domain


class CalcMeasures(PipelineAction):
    """Calculate per-slice tissue measurements from aligned image and masks."""

    def __init__(
        self,
        pipeline,
        mask: str,
        image: str | None = None,
        contour_mask: str | None = None,
        calculate_hu: bool = False,
        calculate_contours: bool = False,
    ):
        super().__init__(pipeline)
        if calculate_hu and image is None:
            raise ValueError("image is required when calculate_hu is enabled.")

        self.input_mask_name = mask
        self.input_image_name = image
        self.input_contour_mask_name = contour_mask or mask
        self.calculate_hu = calculate_hu
        self.calculate_contours = calculate_contours
        self.labels = pipeline.config["LBL_TISSUE"]

        self.io_inputs = [mask]
        if calculate_hu:
            self.io_inputs.append(image)
        if calculate_contours and self.input_contour_mask_name not in self.io_inputs:
            self.io_inputs.append(self.input_contour_mask_name)
        self.io_outputs = ["tmp/tissue_values", "tmp/tissue_geometry"]

    def __call__(self, memory):
        super().__call__(memory)
        time_start = time()

        input_mask = memory[self.input_mask_name].as_closest_canonical()
        input_mask.validate()
        geometry = input_mask.geometry

        input_image = None
        if self.calculate_hu:
            input_image = memory[self.input_image_name].as_closest_canonical()
            input_image.validate()
            assert_same_physical_domain(
                geometry,
                input_image.geometry,
                reference_name="tissue mask",
                candidate_name="CT image",
            )

        contour_mask = None
        if self.calculate_contours:
            contour_container = memory[self.input_contour_mask_name].as_closest_canonical()
            contour_container.validate()
            assert_same_physical_domain(
                geometry,
                contour_container.geometry,
                reference_name="tissue mask",
                candidate_name="contour mask",
            )
            contour_mask = contour_container.data

        result = calculate_slice_measurements(
            input_mask.data,
            geometry,
            self.labels,
            image_zyx=None if input_image is None else input_image.data,
            contour_mask_zyx=contour_mask,
        )

        memory["tmp/tissue_values"] = result
        memory["tmp/tissue_geometry"] = geometry
        logging.info(
            " output: memory:tmp/tissue_values, shape %s (%.2fs)",
            result.shape,
            time() - time_start,
        )
