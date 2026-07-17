# libraries
import logging
from BodyComposition.pipeline import PipelineAction
from BodyComposition.tissue import cleanup_tissue_mask, prepare_classification_images
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time
from BodyComposition.utils.masks import filter_hu
import numpy as np


# action class
class MasksStanfordSpine(PipelineAction):
    """Action class for postprocessing spine segmentations derived from nnUNetv1|Stanford"""

    def __init__(self, pipeline):
        super().__init__(pipeline)

        # io names & to pipeline
        self.input_label_name = 'labels/{caseid}_stanford-spine.nii.gz'
        self.output_mask_name = 'masks/{caseid}_stanford-spine.nii.gz'
        self.io_inputs = [self.input_label_name]
        self.io_outputs = [self.output_mask_name]
        self.io_reset_outputs = [self.output_mask_name]
        if self.config['vertebrae']['save_mask']:
            self.io_persisted_outputs = [self.output_mask_name]

        # create mappings to internal labels
        LBL_VERTEBRALBODIES = pipeline.config['LBL_VERTEBRALBODIES']
        LBL_VERTEBRALBODIES_R = {v: k for k, v in LBL_VERTEBRALBODIES.items()}
        LBL_stanford = {8:'T1',9:'T2',10:'T3',11:'T4',12:'T5',13:'T6',14:'T7',15:'T8',16:'T9',17:'T10',18:'T11',19:'T12',
                        20:'L1',21:'L2',22:'L3',23:'L4',24:'L5'}
        self.mapping_vertebralbodies = {k: LBL_VERTEBRALBODIES_R[v] for k, v in LBL_stanford.items() if v in LBL_VERTEBRALBODIES_R}

    def __call__(self, memory):
        """Do."""
        super().__call__(memory)
        time_start = time()

        # create output: mask, = empty dc + header from input
        output_mask_path = memory['workspace']/self.output_mask_name.format(caseid=memory['id'])
        output_mask = memory[self.output_mask_name] = NiftiDataContainer(output_mask_path)

        # if mask already available, skip all
        if output_mask.exists() and self.config['run']['skip']:
            logging.info(f' output: {output_mask.path} available, skipping')
            return
        
        # load input: label, spine segmentation
        input_label = memory[self.input_label_name]
        input_label.validate()
        output_mask.meta = input_label.meta
        output_mask.data = input_label.data
        output_mask.remap(mapping=self.mapping_vertebralbodies)
        logging.info(f' loaded {input_label} and remapped output labels')

        # save output
        logging.info(f' output: memory:{output_mask} ({time()-time_start:.2f}s)')

         # save mask if active
        if self.config['vertebrae']['save_mask']:
            output_mask.save_to_file()
            logging.info(f'  file saved')



# action class
class MasksStanfordTissue(PipelineAction):
    """Action class for postprocessing tissue segmentations derived from Stanford Tissue model"""

    def __init__(self, pipeline, image: str):
        super().__init__(pipeline)

        # io names & to pipeline
        self.input_label_tissue_name = 'labels/{caseid}_stanford-tissue.nii.gz'
        self.output_mask_name = 'masks/{caseid}_stanford-tissue.nii.gz'
        self.io_inputs = [image, self.input_label_tissue_name]
        self.io_outputs = [self.output_mask_name]
        self.io_reset_outputs = [self.output_mask_name]

        # input
        self.input_image_name = image

        # load config
        self.config_tissue = self.config['tissue']
        if self.config_tissue['save_mask']:
            self.io_persisted_outputs = [self.output_mask_name]

        # overwrite default mappings
        self.LBL_TISSUE = pipeline.config['LBL_TISSUE'] = pipeline.config['LBL_TISSUE_STANFORD']
        self.LBL_TISSUE_R = {v: k for k, v in self.LBL_TISSUE.items()}
        logging.info(f'  tissue labels are mapped to `LBL_TISSUE_STANFORD` mapping: {self.LBL_TISSUE}')

    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory)
        time_start = time()

        # create output: mask, = empty dc + header from input
        output_mask_path = memory['workspace']/self.output_mask_name.format(caseid=memory['id'])
        output_mask = memory[self.output_mask_name] = NiftiDataContainer(output_mask_path)

        # if mask already available, skip all
        if output_mask.exists() and self.config['run']['skip']:
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
            candidate_name="Stanford tissue label",
        )
        logging.info(f' load {input_label_tissue}')
        logging.debug(f'  tissue: origin={input_label_tissue.origin}, shape={input_label_tissue.shape}')

        # copy content and header from input
        output_np = input_label_tissue.data.copy()
        output_mask.meta = input_label_tissue.meta

        # combine IMAT and SM, as segmentation of IMAT does work worse than thresholding
        mask_tmp = np.isin(output_np, [self.LBL_TISSUE_R['IMAT']])
        output_np[mask_tmp] = self.LBL_TISSUE_R['SM']

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
            cleanup_tissue_mask(
                mask_tmp,
                self.config_tissue['imat'],
                input_image.spacing,
                support_mask=compartment,
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
            cleanup_tissue_mask(
                mask_tmp,
                self.config_tissue['sm'],
                input_image.spacing,
                support_mask=compartment,
            )
            output_np[compartment & ~mask_tmp] = 0
            logging.debug(f"  removed everything out of SM HU-range from label SM")


        # filter: visceral adipose tissue VAT 
        if self.config_tissue['vat']['filter_hu']:
            logging.info(f" HU filter visceral compartment")
            compartment = output_np == self.LBL_TISSUE_R['VAT']
            mask_tmp = compartment & filter_hu(
                image_by_tissue['vat'],
                self.config_tissue['vat']['filter_hu_range'],
            )
            cleanup_tissue_mask(
                mask_tmp,
                self.config_tissue['vat'],
                input_image.spacing,
                support_mask=compartment,
            )
            output_np[compartment & ~mask_tmp] = 0
            logging.debug(f"  removed everything out of VAT HU-range from label VAT")


        # filter: subcutaneous adipose tissue SAT
        if self.config_tissue['sat']['filter_hu']:
            logging.info(f" HU filter subcutaneous compartment")
            compartment = output_np == self.LBL_TISSUE_R['SAT']
            mask_tmp = compartment & filter_hu(
                image_by_tissue['sat'],
                self.config_tissue['sat']['filter_hu_range'],
            )
            cleanup_tissue_mask(
                mask_tmp,
                self.config_tissue['sat'],
                input_image.spacing,
                support_mask=compartment,
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
