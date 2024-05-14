# general libraries
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time
import logging

# specific libraries
from totalsegmentator.python_api import totalsegmentator
from BodyComposition.utils.logging import LoggingWriter, log_gpu_usage
import contextlib
import os

# class
class SegmTotalSegmentator(PipelineAction):
    """Segmentation class"""

    def __init__(self, pipeline, image: str, task: str):
        super().__init__(pipeline, task)

        # define io
        self.input_image_name = image
        self.output_label_name = 'labels/{{caseid}}_tseg-{task}.nii.gz'.format(task=task)
        self.io_inputs = [image]
        self.io_outputs = [self.output_label_name]

        # set tseg paths if defined in config
        path_tseg_config = str(self.config['paths']['tseg_config'])
        path_tseg_weights = str(self.config['paths']['weights']['totalsegmentator'])
        if path_tseg_config not in ('None', ''):
            os.environ["TOTALSEG_HOME_DIR"] = path_tseg_config
        if path_tseg_weights not in ('None', ''):
            os.environ["TOTALSEG_WEIGHTS_PATH"] = path_tseg_weights

        # dictionary for task settings
        tasks = {
            'iliopsoas': {
                'task': 'total',
                'fast': self.config['segmentation']['tseg_fast'],
                'roi_subset': ["iliopsoas_left", "iliopsoas_right"],
                'license_warning': False,
            },
            'spine': {
                'task': 'total',
                'fast': self.config['segmentation']['tseg_fast'],
                'roi_subset': ["sacrum", "vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
                               "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6",
                               "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1",
                               "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1"],
                'license_warning': False,
            },
            'bodytrunk': {
                'task': 'body',
                'fast': self.config['segmentation']['tseg_fast'],
                'roi_subset': None,
                'license_warning': False,
            },
            'tissue': {
                'task': 'tissue_types',
                'fast': False, # not available
                'roi_subset': None,
                'license_warning': True,
            },
            'vertebralbodies': {
                'task': 'vertebrae_body',
                'fast': False, # not available
                'roi_subset': None,
                'license_warning': True,
            },
        }

        if task not in tasks:
            raise AssertionError(f'Unknown task: {task}')
        
        self.task = task
        self.task_config = tasks[task]
        if self.task_config['license_warning']:
            logging.warning(f'Using TotalSegmentator/{task}: RESPECT THE LICENSE! Info: https://github.com/wasserth/TotalSegmentator / J. Wasserthal')


    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory, task=self.task)
        time_start = time()

        # create case specific nifti data container for output
        output_label_path = memory['workspace']/self.output_label_name.format(caseid=memory['id'])
        output_label = memory[self.output_label_name] = NiftiDataContainer(output_label_path)

        # check if segmentation already available
        if output_label.exists() and self.config['run']['skip']:
            logging.info(f' output: {output_label} available, skipping')
        else:
            # load input container, log
            input_image = memory[self.input_image_name]
            logging.info(f' input: {input_image}')

            # do segmentation, redirect stdout and stderr to logging
            logging.info(f' running segmentation using totalsegmentator')
            sl = LoggingWriter(logging.DEBUG)
            with contextlib.redirect_stdout(sl), contextlib.redirect_stderr(sl):
                output_label.data_nib = totalsegmentator(input=input_image.data_nib,
                                                         output=None,
                                                         ml=True,
                                                         nr_thr_resamp=1,
                                                         nr_thr_saving=6,
                                                         fast=self.task_config['fast'],
                                                         nora_tag="None",
                                                         preview=False,
                                                         task=self.task_config['task'],
                                                         roi_subset=self.task_config['roi_subset'],
                                                         statistics=False,
                                                         radiomics=False,
                                                         crop_path=None,
                                                         body_seg=False,
                                                         force_split=False,
                                                         output_type="nifti",
                                                         quiet=True,
                                                         verbose=True,
                                                         test=0,
                                                         skip_saving=True,
                                                         device="gpu",
                                                         license_number=None,
                                                         statistics_exclude_masks_at_border=True,
                                                         no_derived_masks=False,
                                                         v1_order=False)
                log_gpu_usage()

            # logging
            logging.info(f' finished segmentation ({time() - time_start:.2f}s)')
            logging.info(f' output: memory:{output_label.path}')

            # saving
            if self.config['segmentation']['save']:
                output_label.save_to_file()
                logging.info(f' saved file')