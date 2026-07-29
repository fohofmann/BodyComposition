"""Pinned TotalSegmentator measurement models executed with upstream nnUNet."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from BodyComposition.actions.segm_int import _SegmInternal
from BodyComposition.measurement.totalsegmentator_assets import (
    MEASUREMENT_MODEL_MANIFEST,
    TotalSegmentatorAssetReport,
    measurement_model_directory,
    require_measurement_model,
)

MAX_RESAMPLED_LOGIT_BYTES = 512 * 1024**2

MEASUREMENT_TASK_IDS = {
    "bodytrunk": 299,
    "body_landmarks": 297,
}
MEASUREMENT_OUTPUTS = {
    "bodytrunk": "masks/totalsegmentator_body.nii.gz",
    "body_landmarks": "masks/totalsegmentator_landmarks.nii.gz",
}


def _as_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()
    return np.asarray(array)


def _bounded_multiclass_segmentation(
    predicted_logits,
    plans_manager,
    configuration_manager,
    label_manager,
    properties: dict,
    *,
    maximum_resampled_bytes: int = MAX_RESAMPLED_LOGIT_BYTES,
) -> np.ndarray:
    """Match nnU-Net's multiclass export without a full target-grid logit tensor."""

    if maximum_resampled_bytes <= 0:
        raise ValueError("maximum_resampled_bytes must be positive.")
    if label_manager.has_regions:
        raise RuntimeError(
            "Bounded TotalSegmentator export requires a non-region multiclass model."
        )
    number_of_channels = int(predicted_logits.shape[0])
    labels = tuple(int(value) for value in label_manager.all_labels)
    if number_of_channels != int(label_manager.num_segmentation_heads):
        raise RuntimeError("TotalSegmentator logits do not match the pinned label schema.")
    if labels != tuple(range(number_of_channels)):
        raise RuntimeError(
            "Bounded TotalSegmentator export requires consecutive labels beginning at zero."
        )

    target_shape = tuple(
        int(value) for value in properties["shape_after_cropping_and_before_resampling"]
    )
    target_voxels = int(np.prod(target_shape, dtype=np.int64))
    bytes_per_channel = target_voxels * np.dtype(np.float32).itemsize
    channels_per_chunk = max(
        1,
        min(number_of_channels, maximum_resampled_bytes // bytes_per_channel),
    )
    spacing_transposed = [properties["spacing"][index] for index in plans_manager.transpose_forward]
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing) == len(target_shape)
        else [spacing_transposed[0], *configuration_manager.spacing]
    )
    target_spacing = [properties["spacing"][index] for index in plans_manager.transpose_forward]

    output_dtype = np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16
    best_values: np.ndarray | None = None
    best_labels = np.zeros(target_shape, dtype=output_dtype)
    for start in range(0, number_of_channels, channels_per_chunk):
        stop = min(number_of_channels, start + channels_per_chunk)
        resampled = _as_numpy(
            configuration_manager.resampling_fn_probabilities(
                predicted_logits[start:stop],
                target_shape,
                current_spacing,
                target_spacing,
            )
        )
        if resampled.shape != (stop - start, *target_shape):
            raise RuntimeError(
                "TotalSegmentator probability resampling returned an unexpected shape."
            )
        for offset, channel in enumerate(range(start, stop)):
            values = resampled[offset]
            if best_values is None:
                best_values = values.copy()
                continue
            better = values > best_values
            np.copyto(best_values, values, where=better)
            np.copyto(best_labels, channel, where=better)
        del resampled

    if best_values is None:
        raise RuntimeError("TotalSegmentator returned no output channels.")
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

    restored = np.zeros(
        tuple(int(value) for value in properties["shape_before_cropping"]),
        dtype=output_dtype,
    )
    restored[bounding_box_to_slice(properties["bbox_used_for_cropping"])] = best_labels
    return restored.transpose(plans_manager.transpose_backward)


def _predict_bounded_multiclass(predictor, input_array, spacing_zyx) -> np.ndarray:
    """Use pinned nnU-Net preprocessing/inference and a bounded export boundary."""

    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy

    iterator = PreprocessAdapterFromNpy(
        [input_array],
        [None],
        [{"spacing": spacing_zyx}],
        [None],
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        num_threads_in_multithreaded=1,
        verbose=predictor.verbose,
    )
    prepared = next(iterator)
    predicted_logits = predictor.predict_logits_from_preprocessed_data(prepared["data"]).cpu()
    return _bounded_multiclass_segmentation(
        predicted_logits,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.label_manager,
        prepared["data_properties"],
    )


class SegmTotalSegmentator(_SegmInternal):
    """Run one verified official TotalSegmentator task with nnUNetPredictor.

    The official task archives are already nnUNet result directories. Calling
    the pinned nnUNet inference engine directly avoids mutable runtime state and
    inference-time downloads while preserving the upstream model and plans.
    """

    weight_key = "totalsegmentator"
    licenses = ["Apache-2.0"]

    def __init__(self, pipeline, image: str, task: str):
        try:
            manifest = MEASUREMENT_MODEL_MANIFEST[task]
            self.output_label_name = MEASUREMENT_OUTPUTS[task]
        except KeyError as error:
            raise ValueError(
                f"Unsupported measurement-support task: {task!r}."
            ) from error
        weights_root = self.configured_weights_root(pipeline)
        self.task = task
        self.task_id = int(manifest["task_id"])
        self.model_title = str(manifest["model_title"])
        self.weights_root = weights_root
        self.asset_report: TotalSegmentatorAssetReport | None = None
        self.pipeline = pipeline
        super().__init__(
            pipeline,
            image=image,
            model=f"task-{self.task_id}",
            model_directory=measurement_model_directory(task, weights_root),
            model_folds=[0],
        )

    @staticmethod
    def configured_weights_root(pipeline) -> Path:
        value = pipeline.config["paths"]["weights"]["totalsegmentator"]
        if value in (None, ""):
            raise RuntimeError(
                "The measurement-support model root is not configured."
            )
        return Path(value)

    def _get_predictor(self):
        if self.predictor is None:
            prepare = getattr(self.pipeline, "prepare_exclusive_model", None)
            if callable(prepare):
                prepare(self)
            self.asset_report = require_measurement_model(
                self.task,
                self.weights_root,
            )
        return super()._get_predictor()

    def _predict_once(self, predictor, input_image, spacing_zyx):
        return _predict_bounded_multiclass(
            predictor,
            input_image.data[None],
            spacing_zyx,
        )

    def __call__(self, memory, task=None) -> None:
        del task
        try:
            super().__call__(memory)
        finally:
            release = getattr(self.pipeline, "release_models", None)
            if callable(release):
                release()
            else:
                self.release_model()
