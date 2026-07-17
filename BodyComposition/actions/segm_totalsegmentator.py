# general libraries
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer
from time import time
import logging

# specific libraries
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.config import setup_nnunet, setup_totalseg
from BodyComposition.utils.logging import LoggingWriter, log_gpu_usage
import contextlib
import os
import threading


MEASUREMENT_TASK_IDS = {
    "bodytrunk": 299,
    "body_landmarks": 297,
}
_MEASUREMENT_INFERENCE_LOCK = threading.RLock()


def _require_measurement_model(task: str) -> None:
    """Prevent inference-time downloads for measurement stage's TotalSegmentator assets."""

    if task not in {"bodytrunk", "body_landmarks"}:
        return
    from BodyComposition.measurement.totalsegmentator_assets import (
        require_measurement_model,
    )

    require_measurement_model(task)


@contextlib.contextmanager
def _measurement_model_download_guard(task: str):
    """Make measurement stage TotalSegmentator inference cache-only and task-pinned."""

    if task not in MEASUREMENT_TASK_IDS:
        with _MEASUREMENT_INFERENCE_LOCK:
            yield
        return

    import totalsegmentator.python_api as upstream_api

    _require_measurement_model(task)
    expected_task_id = MEASUREMENT_TASK_IDS[task]
    with _MEASUREMENT_INFERENCE_LOCK:
        upstream_download = upstream_api.download_pretrained_weights
        upstream_usage_stats = upstream_api.send_usage_stats

        def require_cached(task_id) -> None:
            requested = task_id if isinstance(task_id, list) else [task_id]
            unexpected = [int(value) for value in requested if int(value) != expected_task_id]
            if unexpected:
                raise RuntimeError(
                    "TotalSegmentator requested an unexpected model during cache-only "
                    f"{task} inference: {unexpected}."
                )
            _require_measurement_model(task)

        upstream_api.download_pretrained_weights = require_cached
        upstream_api.send_usage_stats = lambda config, params: None
        try:
            yield
        finally:
            upstream_api.download_pretrained_weights = upstream_download
            upstream_api.send_usage_stats = upstream_usage_stats

# class
class SegmTotalSegmentatorConfig(PipelineAction):
    """Must run before other nnUNet actions to set up TotalSegmentator."""

    def __init__(self, pipeline):
        super().__init__(pipeline)

        # set tseg paths if defined in config
        path_tseg_config = str(self.config['paths']['totalsegmentator_config'])
        path_tseg_weights = str(self.config['paths']['weights']['totalsegmentator'])
        if path_tseg_config not in ('None', ''):
            os.environ["TOTALSEG_HOME_DIR"] = path_tseg_config
        if path_tseg_weights not in ('None', ''):
            os.environ["TOTALSEG_WEIGHTS_PATH"] = path_tseg_weights
        setup_nnunet()
        setup_totalseg()

    def __call__(self, memory):
        # nothing to do
        pass


# class
class SegmTotalSegmentator(PipelineAction):
    """Segmentation class"""

    def __init__(self, pipeline, image: str, task: str, fast: bool = False):
        super().__init__(pipeline, task)
        pipeline_device = getattr(pipeline.device, "type", str(pipeline.device))
        self.device = "gpu" if pipeline_device == "cuda" else pipeline_device

        # define io
        self.input_image_name = image
        self.output_label_name = 'labels/{{caseid}}_tseg-{task}.nii.gz'.format(task=task)
        self.io_inputs = [image]
        self.io_outputs = [self.output_label_name]
        self.io_reset_outputs = [self.output_label_name]
        if self.config['segmentation']['save_label']:
            self.io_persisted_outputs = [self.output_label_name]
        self.licenses = ['totalsegmentator', 'nnunet']

        # dictionary for task settings
        tasks = {
            'iliopsoas': {
                'task': 'total',
                'fast': fast,
                'roi_subset': ["iliopsoas_left", "iliopsoas_right"],
                'license_nc': False,
            },
            'spine': {
                'task': 'total',
                'fast': fast,
                'roi_subset': ["sacrum", "vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
                               "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6",
                               "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1",
                               "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1"],
                'license_nc': False,
            },
            'bodytrunk': {
                'task': 'body',
                'fast': fast,
                'roi_subset': None,
                'license_nc': False,
            },
            'body_landmarks': {
                'task': 'total',
                'fast': True,
                # roi_subset would make upstream run and potentially download
                # an additional rough task-298 model. Run the pinned fast total
                # task once; the landmark adapter consumes only hips and ribs.
                'roi_subset': None,
                'license_nc': False,
            },
            'tissue': {
                'task': 'tissue_types',
                'fast': False, # not available
                'roi_subset': None,
                'license_nc': True,
            },
            'vertebralbodies': {
                'task': 'vertebrae_body',
                'fast': False, # not available
                'roi_subset': None,
                'license_nc': True,
            },
        }

        if task not in tasks:
            raise AssertionError(f'Unknown task: {task}')
        
        self.task = task
        self.task_config = tasks[task]
        if self.task_config['license_nc']:
            self.licenses.append('totalsegmentator_nc')


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
            input_image.validate()
            logging.info(f' input: {input_image}')

            # do segmentation, redirect stdout and stderr to logging
            logging.info(f' running segmentation using totalsegmentator')
            sl = LoggingWriter(logging.DEBUG)
            with (
                _measurement_model_download_guard(self.task),
                contextlib.redirect_stdout(sl),
                contextlib.redirect_stderr(sl),
            ):
                output_label.img = totalsegmentator(input=input_image.imgNifti1,
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
                                                    device=self.device,
                                                    license_number=None,
                                                    statistics_exclude_masks_at_border=True,
                                                    no_derived_masks=False,
                                                    v1_order=False)
                log_gpu_usage()
            output_label.validate()
            assert_same_physical_domain(
                input_image.geometry,
                output_label.geometry,
                reference_name="input image",
                candidate_name=f"TotalSegmentator {self.task} label",
            )

            # logging
            logging.info(f' finished segmentation ({time() - time_start:.2f}s)')
            logging.info(f' output: memory:{output_label.path}')

            # saving
            if self.config['segmentation']['save_label']:
                output_label.save_to_file()
                logging.info(f' saved file')
