"""Pinned TotalSegmentator measurement models executed with upstream nnUNet."""

from __future__ import annotations

from pathlib import Path

from BodyComposition.actions.segm_int import _SegmInternal
from BodyComposition.measurement.totalsegmentator_assets import (
    MEASUREMENT_MODEL_MANIFEST,
    TotalSegmentatorAssetReport,
    measurement_model_directory,
    require_measurement_model,
)

MEASUREMENT_TASK_IDS = {
    "bodytrunk": 299,
    "body_landmarks": 297,
}
MEASUREMENT_OUTPUTS = {
    "bodytrunk": "masks/totalsegmentator_body.nii.gz",
    "body_landmarks": "masks/totalsegmentator_landmarks.nii.gz",
}


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
