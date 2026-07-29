"""Materialize the canonical tissue-label view from model compartments."""

import logging
from pathlib import Path
from time import time

import numpy as np

from BodyComposition.config import CANONICAL_TISSUE_DEFINITIONS
from BodyComposition.measurement.tissues import derive_configured_tissue_masks
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer


class MasksInternalTissue(PipelineAction):
    """Materialize the stable consensus label view from raw compartments."""

    def __init__(self, pipeline, image: str):
        super().__init__(pipeline)

        self.input_label_tissue_name = "masks/tissue_compartments.nii.gz"
        self.output_mask_name = "masks/tissue_labels.nii.gz"
        self.io_inputs = [image, self.input_label_tissue_name]
        self.io_outputs = [self.output_mask_name]
        self.io_reset_outputs = [self.output_mask_name]

        self.input_image_name = image
        self.config_tissue = self.config["tissue"]
        self.io_persisted_outputs = []
        if self.config_tissue["save_mask"]:
            self.io_persisted_outputs.append(self.output_mask_name)
        if self.config_tissue["save_compartment_mask"]:
            self.io_persisted_outputs.append(self.input_label_tissue_name)
            self.io_reset_outputs.append(self.input_label_tissue_name)

        self.LBL_COMPARTMENT = pipeline.config["LBL_TISSUE_COMPARTMENTS"]
        self.LBL_COMPARTMENT_R = {v: k for k, v in self.LBL_COMPARTMENT.items()}

    def __call__(self, memory):
        """Create the consensus label view for one case."""

        super().__call__(memory)
        time_start = time()

        output_mask_path = Path(memory["workspace"]) / self.output_mask_name.format(
            caseid=memory["id"]
        )
        output_mask = memory[self.output_mask_name] = NiftiDataContainer(
            output_mask_path,
            dtype=np.uint8,
        )

        if output_mask.exists() and self.config["run"]["skip"]:
            input_label_tissue = memory[self.input_label_tissue_name]
            if self.config_tissue["save_compartment_mask"] and not input_label_tissue.path.exists():
                input_label_tissue.validate()
                input_label_tissue.save_to_file()
                logging.info(" persisted raw anatomical compartment labels")
            logging.info(" tissue mask already available, skipping")
            return

        input_label_tissue = memory[self.input_label_tissue_name]
        input_image = memory[self.input_image_name]
        input_label_tissue.validate()
        input_image.validate()
        assert_same_physical_domain(
            input_image.geometry,
            input_label_tissue.geometry,
            reference_name="input image",
            candidate_name="internal tissue label",
        )
        logging.info(" load tissue segmentation")
        logging.debug(
            "  tissue: origin=%s, shape=%s",
            input_label_tissue.origin,
            input_label_tissue.shape,
        )
        unknown_labels = sorted(
            int(label)
            for label in np.unique(input_label_tissue.data)
            if int(label) != 0 and int(label) not in self.LBL_COMPARTMENT
        )
        if unknown_labels:
            raise ValueError(
                "Model-native compartments contain labels outside the "
                f"0-7 contract: {unknown_labels}."
            )
        if self.config_tissue["save_compartment_mask"] and not input_label_tissue.path.exists():
            input_label_tissue.save_to_file()
            logging.info(" persisted raw anatomical compartment labels")

        # The default label view is fixed to the consensus definitions. Named
        # sensitivity profiles add downstream measurements; they never mutate
        # this artifact or the model-native source.
        definitions = {
            name: self.config["measurements"]["tissue_definitions"][name]
            for name in (
                "skeletal_muscle_tissue_hu_m29_150",
                "sat_tissue_hu_m190_m30",
                "avat_tissue_hu_m190_m30",
                "tvat_tissue_hu_m190_m30",
            )
        }
        expected = {name: CANONICAL_TISSUE_DEFINITIONS[name] for name in definitions}
        if definitions != expected:
            raise ValueError(
                "Consensus tissue-label materialization requires the immutable "
                "canonical tissue definitions."
            )
        derived = derive_configured_tissue_masks(
            input_image.data,
            input_label_tissue.data,
            self.LBL_COMPARTMENT,
            definitions,
            spacing_xyz=tuple(float(value) for value in input_image.spacing),
        )
        raw = input_label_tissue.data
        output_np = np.zeros_like(raw, dtype=np.uint8)
        consensus_by_label = {
            self.LBL_COMPARTMENT_R["SM"]: derived["skeletal_muscle_tissue_hu_m29_150"],
            self.LBL_COMPARTMENT_R["SAT"]: derived["sat_tissue_hu_m190_m30"],
            self.LBL_COMPARTMENT_R["aVAT"]: derived["avat_tissue_hu_m190_m30"],
            self.LBL_COMPARTMENT_R["tVAT"]: derived["tvat_tissue_hu_m190_m30"],
        }
        for label, consensus_mask in consensus_by_label.items():
            output_np[(raw == label) & consensus_mask] = label
        output_mask.meta = input_label_tissue.meta

        output_mask.data = output_np
        logging.info(" tissue mask complete (%.2fs)", time() - time_start)

        if self.config_tissue["save_mask"]:
            output_mask.save_to_file()
            logging.info("  file saved")
