# general libraries
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time
import logging
import numpy as np
from math import ceil

# class
class CreateBoundingBox(PipelineAction):
    """Class to crop images to ROI"""

    def __init__(self, pipeline, label, task):
        super().__init__(pipeline)

        # define io
        self.input_label_name = label
        self.io_inputs = [label]
        self.io_outputs = [label]

        # load task specific parameters
        if task not in self.config['crop']:
            raise AssertionError(f'Unknown task: {task}')

        logging.info(f'  task: {task}')
        self.task = task
        self.task_config = self.config['crop'][task]
    
    def __call__(self, memory):
        """
        Function crops image using label and task, saves a bounding box to the pipeline.
        The bounding box is applied by the following function.
        If the image is loaded directly within a action (e.g. by using the file path), the bounding box is not used.
        """
        super().__call__(memory)
        time_start = time()

        # load label, get metadata
        input_label = memory[self.input_label_name]
        spacing = input_label.spacing
        shape = input_label.shape
        logging.info(f' label: {input_label}, shape {shape}, spacing {spacing}')

        # transform margins from mm to voxels
        margin = [ceil(float(self.task_config['margin'][i]) / spacing[i//2]) for i in range(6)]
        logging.info(f' margins: {self.task_config["margin"]}mm -> {margin}vx')

        # create bounding box for cropping
        label_roi = np.isin(input_label.data_np, self.task_config['roi']) # only roi

        # create bounding box
        bbox = []
        for dim in range(3):
            # adapt axis to current dim
            imask = np.any(label_roi, axis=tuple(i for i in range(3) if i != dim))
            iindex = np.where(imask)[0]

            # get min and max
            imin = iindex[0] - margin[dim*2] if iindex.size > 0 else 0
            imax = iindex[-1] + margin[dim*2+1] if iindex.size > 0 else shape[dim]

            # check bounds
            imin = max(imin, 0) if self.task_config['axes'][dim*2] else 0
            imax = min(imax, shape[dim]) if self.task_config['axes'][dim*2+1] else shape[dim]

            # append to bbox
            bbox.append(imin)
            bbox.append(imax)

        # save bbox to pipeline, logging
        memory['bbox'] = bbox
        logging.info(f' bounding box: {bbox}')
        logging.info(f' saved to pipeline ({time() - time_start:.2f}s)')


# class
class ApplyBoundingBox(PipelineAction):
    """Class to apply the bounding box to image or mask"""

    def __init__(self, pipeline, input, output=None):
        super().__init__(pipeline)

        # define io
        self.input_name = input
        self.io_inputs = [input]

        # output
        self.output_name = output
        if output is None:
            self.io_outputs = [input]
        else:
            self.io_outputs = [output]

    
    def __call__(self, memory):
        """Applies the bounding box to the image or mask"""
        super().__call__(memory)

        # check
        if 'bbox' not in memory:
            raise AssertionError('Bounding box not available')
        if self.input_name not in memory:
            raise AssertionError(f'Input {self.input_name} not available')

        # create new container if necessary
        if self.output_name is None:
            self.output_name = self.input_name
        elif self.output_name not in memory:
            output_path = memory['workspace']/self.output_name.format(caseid=memory['id'])
            output = memory[self.output_name] = NiftiDataContainer(output_path)
            output.meta = memory[self.input_name].meta
            output.data_np = memory[self.input_name].data_np
            logging.info(f' copied container: {memory[self.input_name]} -> {output}')

        # apply bounding box, log
        memory[self.output_name].bbox = memory['bbox']
        logging.info(f' applied bounding box to {memory[self.output_name]}')