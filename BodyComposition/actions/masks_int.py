# libraries
import logging
from time import time
import numpy as np

from BodyComposition.pipeline import PipelineAction
from BodyComposition.tissue import cleanup_tissue_mask, prepare_classification_images
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer
from BodyComposition.utils.masks import filter_hu


def _apply_configured_cleanup(mask, settings, spacing_xyz, compartment):
    return cleanup_tissue_mask(
        mask,
        settings,
        spacing_xyz,
        support_mask=compartment,
    )

# action class
class MasksInternalTissue(PipelineAction):
    """Action class for postprocessing tissue segmentations derived from internal body composition segmetation."""

    def __init__(self, pipeline, image: str):
        super().__init__(pipeline)

        # io names & to pipeline
        self.input_label_tissue_name = 'labels/{caseid}_int-bodycomposition.nii.gz'
        self.output_mask_name = 'masks/{caseid}_int-bodycomposition.nii.gz'
        self.io_inputs = [image, self.input_label_tissue_name]
        self.io_outputs = [self.output_mask_name]
        self.io_reset_outputs = [self.output_mask_name]

        # input
        self.input_image_name = image

        # load config
        self.config_tissue = self.config['tissue']
        self.io_persisted_outputs = []
        if self.config_tissue['save_mask']:
            self.io_persisted_outputs.append(self.output_mask_name)
        if self.config_tissue['save_compartment_mask']:
            self.io_persisted_outputs.append(self.input_label_tissue_name)
            self.io_reset_outputs.append(self.input_label_tissue_name)

        # overwrite default mappings
        self.LBL_TISSUE = pipeline.config['LBL_TISSUE']
        self.LBL_TISSUE_R = {v: k for k, v in self.LBL_TISSUE.items()}

    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory)
        time_start = time()

        # create output: mask, = empty dc + header from input
        output_mask_path = memory['workspace']/self.output_mask_name.format(caseid=memory['id'])
        output_mask = memory[self.output_mask_name] = NiftiDataContainer(output_mask_path)

        # if mask already available, skip all
        if output_mask.exists() and self.config['run']['skip']:
            input_label_tissue = memory[self.input_label_tissue_name]
            if (
                self.config_tissue['save_compartment_mask']
                and not input_label_tissue.path.exists()
            ):
                input_label_tissue.validate()
                input_label_tissue.save_to_file()
                logging.info(" persisted raw anatomical compartment labels")
            logging.info(f' output: {output_mask.path} available, skipping')
            return

        # load input: label, tissue segmentation
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
        logging.info(f' load {input_label_tissue}')
        logging.debug(f'  tissue: origin={input_label_tissue.origin}, shape={input_label_tissue.shape}')
        if (
            self.config_tissue['save_compartment_mask']
            and not input_label_tissue.path.exists()
        ):
            input_label_tissue.save_to_file()
            logging.info(" persisted raw anatomical compartment labels")

        # copy content and header from input
        output_np = input_label_tissue.data.copy()
        output_mask.meta = input_label_tissue.meta

        # if any HU-based filter active:
        if any([self.config_tissue[tissue]['filter_hu'] for tissue in ['imat', 'sm', 'vat', 'sat']]):
            
            # load image np
            image_by_tissue = prepare_classification_images(
                input_image.data,
                input_image.spacing,
                self.config_tissue['hu_denoise'],
            )
            logging.debug(f'  HU filter(s) active, loaded image')
            logging.debug(f'   image: origin={input_image.origin}, shape={input_image.shape}')

        # filter: intermuscular adipose tissue IMAT; special case as introducing extra label
        if self.config_tissue['imat']['filter_hu']:
            logging.info(f" HU filter muscle compartment")
            compartment = output_np == self.LBL_TISSUE_R['SM']
            mask_tmp = compartment & filter_hu(
                image_by_tissue['imat'],
                self.config_tissue['imat']['filter_hu_range'],
            )
            _apply_configured_cleanup(
                mask_tmp,
                self.config_tissue['imat'],
                input_image.spacing,
                compartment,
            )
            output_np[mask_tmp] = self.LBL_TISSUE_R['IMAT']
            logging.debug(f"  identified everything in IMAT HU-range within label SM as IMAT (={self.LBL_TISSUE_R['IMAT']})")


        # filter: skeletal muscle SM
        if self.config_tissue['sm']['filter_hu']:
            logging.info(f" HU filter muscle compartment(s)")
            compartment = output_np == self.LBL_TISSUE_R['SM']
            mask_tmp = compartment & filter_hu(
                image_by_tissue['sm'],
                self.config_tissue['sm']['filter_hu_range'],
            )
            _apply_configured_cleanup(
                mask_tmp,
                self.config_tissue['sm'],
                input_image.spacing,
                compartment,
            )
            output_np[compartment & ~mask_tmp] = 0
            logging.debug(f"  removed everything out of SM HU-range from label SM")


        # filter: visceral adipose tissue VAT 
        if self.config_tissue['vat']['filter_hu']:
            logging.info(f" HU filter visceral compartment(s)")
            hu_mask = filter_hu(
                image_by_tissue['vat'],
                self.config_tissue['vat']['filter_hu_range'],
            )
            for label_name in ('aVAT', 'tVAT'):
                compartment = output_np == self.LBL_TISSUE_R[label_name]
                mask_tmp = compartment & hu_mask
                _apply_configured_cleanup(
                    mask_tmp,
                    self.config_tissue['vat'],
                    input_image.spacing,
                    compartment,
                )
                output_np[compartment & ~mask_tmp] = 0
            logging.debug(f"  removed everything out of VAT HU-range from label aVAT, tVAT")


        # filter: subcutaneous adipose tissue SAT
        if self.config_tissue['sat']['filter_hu']:
            logging.info(f" HU filter subcutaneous compartment")
            compartment = output_np == self.LBL_TISSUE_R['SAT']
            mask_tmp = compartment & filter_hu(
                image_by_tissue['sat'],
                self.config_tissue['sat']['filter_hu_range'],
            )
            _apply_configured_cleanup(
                mask_tmp,
                self.config_tissue['sat'],
                input_image.spacing,
                compartment,
            )
            output_np[compartment & ~mask_tmp] = 0
            logging.debug(f"  removed everything out of SAT HU-range from label SAT")

        # logging
        output_mask.data = output_np
        logging.info(f' output: memory:{output_mask} ({time()-time_start:.2f}s)')

        # save mask if active
        if self.config_tissue['save_mask']:
            output_mask.save_to_file()
            logging.info(f'  file saved')
