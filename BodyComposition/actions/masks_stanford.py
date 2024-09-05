# libraries
import logging
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time
from BodyComposition.utils.masks import filter_hu, remove_small_objects, filter_keep_largest, fill_holes
from scipy.ndimage import median_filter as ndi_median_filter
import numpy as np


# action class
class MasksStanfordSpine(PipelineAction):
    """Action class for postprocessing spine segmentations derived from nnUNetv1|Stanford"""

    def __init__(self, pipeline, reduce_to_vb: bool = True):
        super().__init__(pipeline)

        # io names & to pipeline
        self.input_label_name = 'labels/{caseid}_stanford-spine.nii.gz'
        self.output_mask_name = 'masks/{caseid}_stanford-vertebrae.nii.gz'
        self.io_inputs = [self.input_label_name]
        self.io_outputs = [self.output_mask_name]

        # reduce to vertebral bodies
        if reduce_to_vb:
            self.input_label_vb_name = 'labels/{caseid}_tseg-vertebralbodies.nii.gz'
            self.io_inputs.append(self.input_label_vb_name)
        self.reduce_to_vb = reduce_to_vb

        # create mappings to internal labels
        LBL_VERTEBRALBODIES = pipeline.config['LBL_VERTEBRALBODIES']
        LBL_VERTEBRALBODIES_R = {v: k for k, v in LBL_VERTEBRALBODIES.items()}
        LBL_stanford = {8:'T1',9:'T2',10:'T3',11:'T4',12:'T5',13:'T6',14:'T7',15:'T8',16:'T9',17:'T10',18:'T11',19:'T12',
                        20:'L1',21:'L2',22:'L3',23:'L4',24:'L5'}
        self.mapping_vertebralbodies = {k: LBL_VERTEBRALBODIES_R[v] for k, v in LBL_stanford.items() if v in LBL_VERTEBRALBODIES_R}
        self.LBL_VERTEBRALBODIESONLY = 1

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
        input_label.remap(mapping=self.mapping_vertebralbodies)
        logging.info(f' loaded and remapped {input_label}')

        # copy content and header from input
        output_mask.data_np = input_label.data_np
        output_mask.meta = input_label.meta

        # remove all but vertebral bodies
        if self.reduce_to_vb:
            input_label_vb = memory[self.input_label_vb_name]
            output_mask.data_np[input_label_vb.data_np != self.LBL_VERTEBRALBODIESONLY] = 0 # set everything outside of vertebralbodies to zero
            logging.info(f'  removed everything except vertebral bodies')

        # save output
        logging.info(f' output: memory:{output_mask} ({time()-time_start:.2f}s)')

         # save mask if active
        if self.config['vertebrae']['save_mask']:
            output_mask.save_to_file()
            logging.info(f'  file saved')
