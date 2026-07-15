"""Pipeline action for CT orientation integrity and preparation."""

from __future__ import annotations

import copy
import csv
import logging
from pathlib import Path

from BodyComposition.orientation.core import (
    OrientationSettings,
    assess_orientation,
    write_orientation_failure_artifacts,
)
from BodyComposition.orientation.ctdeeprot import CTDeepRotPredictor
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.nifti import NiftiDataContainer


class AssessOrientation(PipelineAction):
    """Run CTDeepRot before segmentation and replace only a repaired input."""

    report_name = "orientation/{caseid}/orientation_report.json"
    review_name = "orientation/{caseid}/orientation_review.png"
    corrected_name = "orientation/{caseid}/corrected_input.nii.gz"

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.settings = OrientationSettings.from_mapping(pipeline.config)
        self.settings.validate()
        self.io_inputs = ["tmp/index"]
        self.io_outputs = ["tmp/index", "tmp/orientation_result", "tmp/prepared_image"]
        if self.settings.report_enabled:
            self.io_outputs.extend((self.report_name, self.review_name))
            self.io_persisted_outputs = [self.report_name, self.review_name]
        self.io_reset_outputs = [self.report_name, self.review_name, self.corrected_name]
        self.licenses = ["ctdeeprot"]
        self._predictor = None

    def _get_predictor(self) -> CTDeepRotPredictor:
        if self._predictor is None:
            self._predictor = CTDeepRotPredictor(
                self.settings.checkpoint_path,
                device=self.settings.device,
                batch_size=self.settings.batch_size,
            )
        return self._predictor

    def __call__(self, memory):
        super().__call__(memory)
        source = memory["tmp/index"]
        if not isinstance(source, NiftiDataContainer):
            raise TypeError("tmp/index must be a NiftiDataContainer before orientation assessment.")

        output_directory = (
            Path(memory["workspace"])
            / "orientation"
            / str(memory["id"])
        )
        case_config = self.config
        metadata_path = (
            source.path.parent.parent
            / "metadata"
            / f"{memory['id']}.csv"
        )
        if metadata_path.is_file():
            with metadata_path.open(newline="", encoding="utf-8") as handle:
                metadata_keys = {
                    row[0]
                    for row in csv.reader(handle)
                    if row
                }
            if "ImageOrientationPatient" not in metadata_keys:
                case_config = copy.deepcopy(self.config)
                case_config["orientation"]["header_uncertain_reasons"] = [
                    *case_config["orientation"].get("header_uncertain_reasons", []),
                    "DICOM conversion provenance has no ImageOrientationPatient value.",
                ]
        try:
            outcome = assess_orientation(
                source.img,
                config=case_config,
                predictor=self._get_predictor(),
                output_directory=(output_directory if self.settings.report_enabled else None),
                case_id=str(memory["id"]),
            )
        except Exception as exc:
            if self.settings.report_enabled:
                write_orientation_failure_artifacts(
                    output_directory=output_directory,
                    case_id=str(memory["id"]),
                    error=exc,
                )
            raise

        memory["tmp/original_index"] = source
        memory["tmp/orientation_result"] = outcome.result
        # Keep the prepared-image handoff immutable for downstream adapters.
        # It keeps the prepared image, orientation provenance, transform, and
        # review state together instead of asking later stages to reconstruct it.
        memory["tmp/prepared_image"] = outcome
        if outcome.report_json is not None:
            memory[self.report_name] = outcome.report_json
        if outcome.review_png is not None:
            memory[self.review_name] = outcome.review_png

        if outcome.result.orientation_changed:
            corrected_path = (
                outcome.corrected_input
                or (
                    Path(memory["workspace"])
                    / self.corrected_name.format(caseid=memory["id"])
                )
            )
            corrected = NiftiDataContainer(corrected_path)
            if corrected_path.exists():
                corrected.load_from_file()
            else:
                corrected.img = outcome.prepared_image
                corrected.save_to_file()
            corrected.validate()
            memory["tmp/index"] = corrected
            memory[self.corrected_name] = corrected_path
            logging.warning(
                "  orientation changed for %s; downstream segmentation uses %s and manual review is required",
                memory["id"],
                corrected_path,
            )
        else:
            logging.info(
                "  orientation retained for %s (%s, QC=%s)",
                memory["id"],
                outcome.result.state.value,
                outcome.result.qc_status,
            )
