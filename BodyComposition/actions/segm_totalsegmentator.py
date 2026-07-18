# general libraries
import contextlib
import logging
import os
import threading
from pathlib import Path
from time import time

from totalsegmentator.config import setup_nnunet, setup_totalseg

# specific libraries
from totalsegmentator.python_api import totalsegmentator

from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.logging import LoggingWriter, log_gpu_usage
from BodyComposition.utils.nifti import NiftiDataContainer

MEASUREMENT_TASK_IDS = {
    "bodytrunk": 299,
    "body_landmarks": 297,
}
MEASUREMENT_OUTPUTS = {
    "bodytrunk": "masks/totalsegmentator_body.nii.gz",
    "body_landmarks": "masks/totalsegmentator_landmarks.nii.gz",
}
_MEASUREMENT_INFERENCE_LOCK = threading.RLock()


def _require_measurement_model(task: str) -> None:
    """Prevent inference-time downloads for measurement-support model assets."""

    if task not in {"bodytrunk", "body_landmarks"}:
        return
    from BodyComposition.measurement.totalsegmentator_assets import (
        require_measurement_model,
    )

    require_measurement_model(task)


@contextlib.contextmanager
def _measurement_model_download_guard(task: str):
    """Make measurement-support inference cache-only and task-pinned."""

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
    """Configure TotalSegmentator when execution starts, not during planning."""

    def __init__(self, pipeline):
        super().__init__(pipeline)

        # set tseg paths if defined in config
        path_tseg_config = str(self.config['paths']['totalsegmentator_config'])
        path_tseg_weights = str(self.config['paths']['weights']['totalsegmentator'])
        self.path_tseg_config = path_tseg_config
        self.path_tseg_weights = path_tseg_weights

    def __call__(self, memory):
        # Pipeline construction and dry planning must remain side-effect free.
        # TotalSegmentator creates its local configuration directory here, only
        # after the service has verified every required model asset.
        for value in (self.path_tseg_config, self.path_tseg_weights):
            if value not in ("None", ""):
                Path(value).mkdir(parents=True, exist_ok=True)
        if self.path_tseg_config not in ("None", ""):
            os.environ["TOTALSEG_HOME_DIR"] = self.path_tseg_config
        if self.path_tseg_weights not in ("None", ""):
            os.environ["TOTALSEG_WEIGHTS_PATH"] = self.path_tseg_weights
        setup_nnunet()
        setup_totalseg()


# class
class SegmTotalSegmentator(PipelineAction):
    """Segmentation class"""

    def __init__(self, pipeline, image: str, task: str, fast: bool = False):
        super().__init__(pipeline, task)
        pipeline_device = getattr(pipeline.device, "type", str(pipeline.device))
        self.device = "gpu" if pipeline_device == "cuda" else pipeline_device
        self.cpu_threads = max(1, int(getattr(pipeline, "cpu_threads", 1)))

        # define io
        self.input_image_name = image
        try:
            self.output_label_name = MEASUREMENT_OUTPUTS[task]
        except KeyError as error:
            raise ValueError(
                f"Unsupported release TotalSegmentator task: {task!r}."
            ) from error
        self.io_inputs = [image]
        self.io_outputs = [self.output_label_name]
        self.io_reset_outputs = [self.output_label_name]
        if self.config['segmentation']['save_label']:
            self.io_persisted_outputs = [self.output_label_name]

        # dictionary for task settings
        tasks = {
            'bodytrunk': {
                'task': 'body',
                'fast': fast,
                'roi_subset': None,
            },
            'body_landmarks': {
                'task': 'total',
                'fast': True,
                # roi_subset would make upstream run and potentially download
                # an additional rough task-298 model. Run the pinned fast total
                # task once; the landmark adapter consumes only hips and ribs.
                'roi_subset': None,
            },
        }

        if task not in tasks:
            raise ValueError(f'Unknown release TotalSegmentator task: {task}')
        
        self.task = task
        self.task_config = tasks[task]
    def __call__(self, memory):
        """Segment case."""
        super().__call__(memory, task=self.task)
        time_start = time()

        # create case specific nifti data container for output
        output_label_path = memory['workspace']/self.output_label_name.format(caseid=memory['id'])
        output_label = memory[self.output_label_name] = NiftiDataContainer(output_label_path)

        # check if segmentation already available
        if output_label.exists() and self.config['run']['skip']:
            logging.info(" TotalSegmentator output already available, skipping")
        else:
            # load input container, log
            input_image = memory[self.input_image_name]
            input_image.validate()
            logging.info(" input image validated")

            # do segmentation, redirect stdout and stderr to logging
            logging.info(' running segmentation using totalsegmentator')
            sl = LoggingWriter(logging.DEBUG)
            with (
                _measurement_model_download_guard(self.task),
                contextlib.redirect_stdout(sl),
                contextlib.redirect_stderr(sl),
            ):
                output_label.img = totalsegmentator(input=input_image.imgNifti1,
                                                    output=None,
                                                    ml=True,
                                                    nr_thr_resamp=self.cpu_threads,
                                                    nr_thr_saving=self.cpu_threads,
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
            logging.info(" TotalSegmentator output validated")

            # saving
            if self.config['segmentation']['save_label']:
                output_label.save_to_file()
                logging.info(' saved file')
