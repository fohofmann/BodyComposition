# libraries
import logging
from BodyComposition.pipeline import PipelineAction
from time import time
import numpy as np

# action class
class CalcCSA(PipelineAction):
    """
    Action class for calculation of cross-sectional areas
    Argument: input_mask (str) - identifier of the input mask
    """

    def __init__(self, pipeline, mask: str):
        super().__init__(pipeline)

        # io to pipeline
        self.io_inputs = [mask]
        self.io_outputs = ['tmp/tissue_values', 'tmp/tissue_meta']
        self.input_mask_name = mask

        # load labels as constants
        self.LBL_TISSUE = pipeline.config['LBL_TISSUE']

    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory)
        time_start = time()

        # load mask, reorientate
        input_mask = memory[self.input_mask_name]
        input_mask.as_closest_canonical()
        logging.info(f' loaded {input_mask}, reorientated to canonical')

        # load spacing
        spacing = input_mask.spacing # RAS+
        pix_area = spacing[0] * spacing[1]
        memory['slicethickness'] = spacing[2]
        logging.info(f' spacing: {spacing}')

        # RAS+ format: 2 = -1 = inferior to superior, starting w 0
        # create empty output array
        res_csa_np = np.zeros(shape=(input_mask.shape[-1], len(self.LBL_TISSUE)), dtype=np.uint32)

        # loop through tissue labels, run vectorized operation
        for i, key in enumerate(self.LBL_TISSUE):
            res_csa_np[:, i] = np.round(np.sum(input_mask.data_np == key, axis=(0, 1)) * pix_area)

        # save data to pipeline
        memory['tmp/tissue_values'] = res_csa_np
        memory['tmp/tissue_meta'] = input_mask.meta
        logging.info(f' output: memory:tmp/tissue_values, shape {res_csa_np.shape} ({time()-time_start:.2f}s)') 
