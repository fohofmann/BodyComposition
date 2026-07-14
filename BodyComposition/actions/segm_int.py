import contextlib
import logging
from pathlib import Path
from time import time

from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.logging import LoggingWriter, log_gpu_usage
from BodyComposition.utils.nifti import NiftiDataContainer


class _SegmInternal(PipelineAction):
    output_label_name: str
    weight_key: str
    model_title: str
    licenses: list[str]

    def __init__(self, pipeline, image: str, model: str = "ResEncM"):
        super().__init__(pipeline)
        self.input_image_name = image
        self.io_inputs = [image]
        self.io_outputs = [self.output_label_name]
        self.io_reset_outputs = [self.output_label_name]
        if self.config["segmentation"]["save_label"]:
            self.io_persisted_outputs = [self.output_label_name]
        self.device = pipeline.device
        self.predictor = None

        model_settings = {
            "ResEncM": (
                "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
                "all",
            ),
            "ResEncL": (
                "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
                [0, 1, 2, 3, 4],
            ),
        }
        if model not in model_settings:
            raise ValueError(f"Unknown internal model preset: {model}.")
        model_directory, self.model_folds = model_settings[model]
        self.model_path = Path(self.config["paths"]["weights"][self.weight_key]) / model_directory
        logging.info(" using %s model preset %s", self.model_title, model)

    def _get_predictor(self):
        if self.predictor is not None:
            return self.predictor
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}.")

        stream = LoggingWriter(logging.DEBUG)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=self.device,
                verbose=True,
                verbose_preprocessing=True,
                allow_tqdm=False,
            )
            predictor.initialize_from_trained_model_folder(
                str(self.model_path),
                use_folds=self.model_folds,
                checkpoint_name="checkpoint_final.pth",
            )
        self.predictor = predictor
        return predictor

    def _predict(self, predictor, input_image, spacing_zyx):
        arguments = (
            input_image.data[None],
            {"spacing": spacing_zyx},
            None,
            None,
            False,
        )
        try:
            return predictor.predict_single_npy_array(*arguments)
        except RuntimeError as error:
            cudnn_engine_error = any(
                message in str(error)
                for message in (
                    "FIND was unable to find an engine to execute this computation",
                    "GET was unable to find an engine to execute this computation",
                )
            )
            if getattr(self.device, "type", self.device) != "cuda" or not cudnn_engine_error:
                raise

            import torch

            if not torch.backends.cudnn.enabled:
                raise
            logging.warning(
                " cuDNN could not execute this model; retrying with cuDNN disabled "
                "for the remainder of the process."
            )
            torch.backends.cudnn.enabled = False
            return predictor.predict_single_npy_array(*arguments)

    def __call__(self, memory):
        super().__call__(memory)
        time_start = time()
        output_path = Path(memory["workspace"]) / self.output_label_name.format(caseid=memory["id"])
        output_label = memory[self.output_label_name] = NiftiDataContainer(output_path)

        if output_label.exists() and self.config["run"]["skip"]:
            logging.info(" output: %s available, skipping", output_label)
            return

        input_image = memory[self.input_image_name]
        input_image.validate()
        output_label.meta = input_image.meta
        predictor = self._get_predictor()
        spacing_zyx = tuple(reversed(input_image.spacing))

        logging.info(" running segmentation using nnUNetv2|%s", self.model_title)
        stream = LoggingWriter(logging.DEBUG)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            output_label.data = self._predict(predictor, input_image, spacing_zyx)
            log_gpu_usage(self.device)
        output_label.validate()
        assert_same_physical_domain(
            input_image.geometry,
            output_label.geometry,
            reference_name="input image",
            candidate_name=f"{self.model_title} label",
        )

        logging.info(" finished segmentation (%.2fs)", time() - time_start)
        if self.config["segmentation"]["save_label"]:
            output_label.save_to_file()
            logging.info(" saved file: %s", output_label.path)


class SegmIntVertebrae(_SegmInternal):
    """Segment thoracic and lumbar vertebral bodies with an internal nnU-Net model."""

    output_label_name = "labels/{caseid}_int-vertebrae.nii.gz"
    weight_key = "int-vertebrae"
    model_title = "VertebralBodiesCT"
    licenses = ["nnunet", "nnunet_resenc", "intvertebrae", "verse", "boa", "totalsegmentator"]


class SegmIntBodyComposition(_SegmInternal):
    """Segment tissue compartments with an internal nnU-Net model."""

    output_label_name = "labels/{caseid}_int-bodycomposition.nii.gz"
    weight_key = "int-bodycomposition"
    model_title = "BodyCompositionCT"
    licenses = ["nnunet", "nnunet_resenc", "intbodycomposition", "boa", "totalsegmentator"]
