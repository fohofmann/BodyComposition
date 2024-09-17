# libraries
import logging
from BodyComposition.pipeline import PipelineAction
from time import time
import numpy as np
import cv2

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
        spacing = input_mask.spacing # SITK: z, y, x
        pix_area = spacing[1] * spacing[2]
        memory['slicethickness'] = spacing[0]
        logging.info(f' spacing: {spacing}')

        # SITK: SAR+ format: 0 = inferior to superior, starting w 0
        # create empty output array
        res_csa_np = np.zeros(shape=(input_mask.shape[0], len(self.LBL_TISSUE)), dtype=np.uint32)

        # loop through tissue labels, run vectorized operation
        for i, key in enumerate(self.LBL_TISSUE):
            res_csa_np[:, i] = np.round(np.sum(input_mask.data == key, axis=(1, 2)) * pix_area)

        # save data to pipeline
        memory['tmp/tissue_values'] = res_csa_np
        memory['tmp/tissue_meta'] = input_mask.meta
        logging.info(f' output: memory:tmp/tissue_values, shape {res_csa_np.shape} ({time()-time_start:.2f}s)')



# action class
class CalcCRI(PipelineAction):
    """
    Action class for calculation of circumference and area of the tissue contour
    Argument: input_mask (str) - identifier of the input mask
    """

    def __init__(self, pipeline, mask: str):
        super().__init__(pipeline)

        # io to pipeline
        self.io_inputs = [mask]
        self.input_mask_name = mask
        self.io_outputs = ['tmp/tissue_contour']

    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory)
        time_start = time()

        # load mask, reorientate
        input_mask = memory[self.input_mask_name]
        input_mask.as_closest_canonical()
        logging.info(f' loaded {input_mask}, reorientated to canonical')

        # load spacing
        spacing = input_mask.spacing # SITK: z, y, x
        logging.info(f' spacing: {spacing}')

        # timer, all but background, define range, create empty np
        time_start = time()
        binarymask = input_mask.data != 0
        z_dim = binarymask.shape[0]
        res_contours_np = np.zeros(shape=(z_dim, 2), dtype=np.float32)

        # calculate contour for each slice
        # SITK: zyx, LPS+ -> 0 = inferior to superior, starting w 0
        for z in range(z_dim):
            slice_mask = binarymask[z, :, :]
            contours, _ = cv2.findContours(slice_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:  # Ensure at least one contour is found
                largest_contour = max(contours, key=cv2.contourArea) # Find the largest contour by area
                scaled_contour = largest_contour.astype(np.float32) # Adjust the contour points by voxel size to calculate circumference in physical units
                scaled_contour[:, 0, 0] *= spacing[2]  # Adjust x-coordinates (columns) by x_spacing
                scaled_contour[:, 0, 1] *= spacing[1]  # Adjust y-coordinates (rows) by y_spacing
                res_contours_np[z,0] = round(cv2.arcLength(scaled_contour, True)) # calculate length of the adjusted contour, mm
                res_contours_np[z,1] = round(cv2.contourArea(scaled_contour)) # calculate area of the adjusted contour, mm^2
            else:
                res_contours_np[z] = np.nan

        # save data to pipeline
        memory['tmp/tissue_contour'] = res_contours_np
        logging.info(f' output: memory:tmp/tissue_contour, shape {res_contours_np.shape} ({time()-time_start:.2f}s)')
