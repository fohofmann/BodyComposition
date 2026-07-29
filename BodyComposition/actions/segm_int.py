import contextlib
import logging
from pathlib import Path
from time import time

import numpy as np

from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import assert_same_physical_domain
from BodyComposition.utils.logging import LoggingWriter, log_gpu_usage
from BodyComposition.utils.nifti import NiftiDataContainer


class _SegmInternal(PipelineAction):
    output_label_name: str
    weight_key: str
    model_title: str
    licenses: list[str]

    def __init__(
        self,
        pipeline,
        image: str,
        model: str = "ResEncM",
        *,
        model_directory: str | Path | None = None,
        model_folds: str | list[int] | None = None,
        analysis_region: str | None = None,
        restore_reference: str | None = None,
    ):
        super().__init__(pipeline)
        self.input_image_name = image
        self.io_inputs = [image]
        self.io_outputs = [self.output_label_name]
        self.io_reset_outputs = [self.output_label_name]
        if self.config["segmentation"]["save_label"]:
            self.io_persisted_outputs = [self.output_label_name]
        self.device = pipeline.device
        self.predictor = None
        self.model_preset = model
        self.analysis_region_name = analysis_region
        self.restore_reference_name = restore_reference
        if (analysis_region is None) != (restore_reference is None):
            raise ValueError(
                "analysis_region and restore_reference must be configured together."
            )
        if analysis_region is not None:
            self.io_inputs.extend((analysis_region, restore_reference))

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
        if model_directory is None:
            if model not in model_settings:
                raise ValueError(f"Unknown internal model preset: {model}.")
            preset_directory, self.model_folds = model_settings[model]
            self.model_path = (
                Path(self.config["paths"]["weights"][self.weight_key])
                / preset_directory
            )
        else:
            if model_folds is None:
                raise ValueError("Explicit nnUNet model directories require model_folds.")
            self.model_path = Path(model_directory)
            self.model_folds = model_folds
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
                perform_everything_on_device=(
                    getattr(self.device, "type", self.device) == "cuda"
                ),
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

    def release_model(self) -> None:
        self.predictor = None

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
        output_label = memory[self.output_label_name] = NiftiDataContainer(
            output_path,
            dtype=np.uint8,
        )

        if output_label.exists() and self.config["run"]["skip"]:
            logging.info(" internal segmentation already available, skipping")
            return

        input_image = memory[self.input_image_name]
        input_image.validate()
        predictor = self._get_predictor()
        spacing_zyx = tuple(reversed(input_image.spacing))

        logging.info(" running segmentation using nnUNetv2|%s", self.model_title)
        stream = LoggingWriter(logging.DEBUG)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            prediction = self._predict(predictor, input_image, spacing_zyx)
            log_gpu_usage(self.device)
        if self.analysis_region_name is None:
            output_label.meta = input_image.meta
            output_label.data = prediction
            output_reference = input_image
        else:
            analysis_region = memory[self.analysis_region_name]
            restore_reference = memory[self.restore_reference_name]
            restore_reference.validate()
            assert_same_physical_domain(
                analysis_region.inference_region.geometry,
                input_image.geometry,
                reference_name="configured tissue inference region",
                candidate_name="nnUNet tissue input",
            )
            output_label.meta = restore_reference.meta
            output_label.data = analysis_region.restore_target_labels(prediction)
            output_reference = restore_reference
        output_label.validate()
        assert_same_physical_domain(
            output_reference.geometry,
            output_label.geometry,
            reference_name="segmentation output reference",
            candidate_name=f"{self.model_title} label",
        )

        logging.info(" finished segmentation (%.2fs)", time() - time_start)
        if self.config["segmentation"]["save_label"]:
            output_label.save_to_file()
            logging.info(" saved internal segmentation")


class SegmIntVertebrae(_SegmInternal):
    """Segment thoracic and lumbar vertebral bodies with an internal nnU-Net model."""

    output_label_name = "masks/vertebral_bodies.nii.gz"
    weight_key = "int-vertebrae"
    model_title = "VertebralBodiesCT"
    def __init__(self, pipeline, image: str, model: str = "ResEncM"):
        super().__init__(pipeline, image=image, model=model)
        self.io_outputs.append("tmp/vertebral_result")

    def __call__(self, memory):
        super().__call__(memory)
        from dataclasses import replace

        from BodyComposition.vertebral.internal_adapter import adapt_internal_vertebral_bodies
        from BodyComposition.vertebral.spineps_backend import orientation_qc_flags

        output = memory[self.output_label_name]
        output.validate()
        native_outputs = {"vertebral_body_labels": output.path} if output.path.is_file() else {}
        result = adapt_internal_vertebral_bodies(
            output.data,
            output.geometry,
            self.config["LBL_VERTEBRALBODIES"],
            model_preset=self.model_preset,
            native_outputs=native_outputs,
            provenance={"model_family": self.model_title},
        )
        prepared = memory.get("tmp/prepared_image")
        if prepared is not None:
            result = replace(
                result,
                provenance={
                    **dict(result.provenance),
                    "orientation": prepared.result.to_dict(),
                    "orientation_changed": prepared.result.orientation_changed,
                    "orientation_manual_review_required": prepared.result.manual_review_required,
                },
                qc_flags=tuple(result.qc_flags) + orientation_qc_flags(prepared),
            )
        memory["tmp/vertebral_result"] = result


class SegmIntBodyComposition(_SegmInternal):
    """Segment anatomical compartments with an internal nnU-Net model."""

    output_label_name = "masks/tissue_compartments.nii.gz"
    weight_key = "int-bodycomposition"
    model_title = "BodyCompositionCT"
