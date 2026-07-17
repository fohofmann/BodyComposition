"""measurement stage body-surface, landmark, measurement, and canonical export actions."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from time import time
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import SimpleITK as sitk

from BodyComposition.measurement.body_surface import (
    DETERMINISTIC_BODY_BACKEND,
    TISSUE_ENVELOPE_BACKEND,
    TOTALSEGMENTATOR_BODY_BACKEND,
    body_surface_from_totalsegmentator,
    deterministic_body_surface,
    tissue_segmentation_envelope,
)
from BodyComposition.measurement.builder import (
    build_measurement_bundle,
    measurement_analysis_id,
)
from BodyComposition.measurement.contracts import (
    MeasurementIdentity,
    canonical_arrow_schema,
)
from BodyComposition.measurement.landmarks import (
    TOTALSEGMENTATOR_LANDMARK_BACKEND,
    landmarks_from_totalsegmentator,
)
from BodyComposition.measurement.review import write_measurement_review
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer


BODY_SURFACE_MASK = "masks/{caseid}_body-surface.nii.gz"
MEASUREMENT_REVIEW = "qc/{caseid}_measurement-review.png"
MEASUREMENT_QC = "qc/{caseid}_measurement-qc.json"
SLICE_TABLE = "tables/{caseid}/slices.parquet"
VERTEBRA_TABLE = "tables/{caseid}/vertebrae.parquet"
SUMMARY_TABLE = "tables/{caseid}/summaries.parquet"
TOTALSEG_BODY_LABEL = "labels/{caseid}_tseg-bodytrunk.nii.gz"
TOTALSEG_LANDMARK_LABEL = "labels/{caseid}_tseg-body_landmarks.nii.gz"


def _path(memory: dict, template: str) -> Path:
    return Path(memory["workspace"]) / template.format(caseid=memory["id"])


def _scientific_measurement_configuration(settings: dict) -> dict:
    """Exclude output/debug policy from the measurement-stage identity."""

    configuration = {
        key: value
        for key, value in settings.items()
        if key not in {"enabled", "export", "review"}
    }
    body_surface = settings["body_surface"]
    backend_parameters = {
        TISSUE_ENVELOPE_BACKEND: (
            "minimum_component_area_mm2",
            "closing_radius_mm",
            "smoothing_sigma_mm",
        ),
        DETERMINISTIC_BODY_BACKEND: (
            "threshold_hu",
            "min_component_volume_mm3",
        ),
        TOTALSEGMENTATOR_BODY_BACKEND: (),
    }
    configuration["body_surface"] = {
        "backend": body_surface["backend"],
        **{
            key: body_surface[key]
            for key in backend_parameters[body_surface["backend"]]
        },
    }
    return configuration


def _atomic_json(payload: object, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _atomic_parquet(
    table: pd.DataFrame,
    destination: Path,
    *,
    table_name: str,
    identity: Mapping[str, str],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial.parquet")
    try:
        arrow_table = pa.Table.from_pandas(
            table,
            schema=canonical_arrow_schema(table),
            preserve_index=False,
            safe=True,
        )
        metadata = {
            key: value
            for key, value in (arrow_table.schema.metadata or {}).items()
            if key != b"pandas"
        }
        metadata[b"bodycomposition.table"] = table_name.encode("ascii")
        for key, value in identity.items():
            metadata[f"bodycomposition.{key}".encode("ascii")] = str(value).encode(
                "utf-8"
            )
        arrow_table = arrow_table.replace_schema_metadata(metadata)
        pq.write_table(arrow_table, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class CreateBodySurface(PipelineAction):
    """Build aligned body/trunk masks with no implicit backend substitution."""

    def __init__(self, pipeline, *, tissue_mask: str):
        super().__init__(pipeline)
        self.settings = self.config["measurements"]["body_surface"]
        self.backend = self.settings["backend"]
        self.tissue_mask_name = tissue_mask
        if self.backend == TOTALSEGMENTATOR_BODY_BACKEND:
            self.input_name = TOTALSEG_BODY_LABEL
        elif self.backend == DETERMINISTIC_BODY_BACKEND:
            self.input_name = "tmp/prepared_image"
        elif self.backend == TISSUE_ENVELOPE_BACKEND:
            self.input_name = tissue_mask
        else:
            raise ValueError(f"Unknown body-surface backend {self.backend!r}.")
        self.io_inputs = [self.input_name, "tmp/prepared_image"]
        self.io_outputs = ["tmp/body_surface_result", BODY_SURFACE_MASK]
        self.io_reset_outputs = [BODY_SURFACE_MASK]
        if self.settings["save_mask"]:
            self.io_persisted_outputs = [BODY_SURFACE_MASK]
        if self.backend == TOTALSEGMENTATOR_BODY_BACKEND:
            self.licenses = ["totalsegmentator", "nnunet"]

    def __call__(self, memory):
        super().__call__(memory)
        started = time()
        prepared = memory["tmp/prepared_image"].prepared_image
        geometry = ImageGeometry.from_sitk(prepared)
        if self.backend == TOTALSEGMENTATOR_BODY_BACKEND:
            from BodyComposition.measurement.totalsegmentator_assets import (
                require_measurement_model,
            )

            asset_report = require_measurement_model("bodytrunk")
            label = memory[self.input_name]
            label.validate()
            assert_same_physical_domain(
                geometry,
                label.geometry,
                reference_name="prepared CT",
                candidate_name="TotalSegmentator body label",
            )
            result = body_surface_from_totalsegmentator(
                label.data,
                geometry,
                provenance={
                    "software_version": self.config["measurements"][
                        "totalsegmentator_version"
                    ],
                    "inference_mode": "body_1p5mm_full_volume_cache_only",
                    "model_assets": asset_report.provenance(),
                },
            )
        elif self.backend == DETERMINISTIC_BODY_BACKEND:
            result = deterministic_body_surface(
                sitk.GetArrayFromImage(prepared),
                geometry,
                threshold_hu=float(self.settings["threshold_hu"]),
                min_component_volume_mm3=float(self.settings["min_component_volume_mm3"]),
            )
        else:
            tissue = memory[self.input_name]
            tissue.validate()
            assert_same_physical_domain(
                geometry,
                tissue.geometry,
                reference_name="prepared CT",
                candidate_name="tissue segmentation used for body envelope",
            )
            result = tissue_segmentation_envelope(
                tissue.data,
                geometry,
                minimum_component_area_mm2=float(
                    self.settings["minimum_component_area_mm2"]
                ),
                closing_radius_mm=float(self.settings["closing_radius_mm"]),
                smoothing_sigma_mm=float(self.settings["smoothing_sigma_mm"]),
                provenance={"tissue_mask_input": "canonical_pipeline_tissue_mask"},
            )

        encoded = np.where(result.body_mask_zyx, 2, 0).astype(np.uint8)
        encoded[result.trunk_mask_zyx] = 1
        image = sitk.GetImageFromArray(encoded)
        image.CopyInformation(prepared)
        output = NiftiDataContainer(_path(memory, BODY_SURFACE_MASK))
        output.img = image
        output.validate()
        if self.settings["save_mask"]:
            output.save_to_file()
        memory[BODY_SURFACE_MASK] = output
        memory["tmp/body_surface_result"] = result
        logging.info(" output: body/trunk surface (%s, %.2fs)", result.backend_id, time() - started)


class CreateMeasurementLandmarks(PipelineAction):
    """Adapt explicit rib/iliac labels into physical mid-waist landmarks."""

    def __init__(self, pipeline):
        super().__init__(pipeline)
        settings = self.config["measurements"]["landmarks"]
        self.backend = settings["backend"]
        self.minimum_voxels = settings["minimum_voxels"]
        self.maximum_side_disagreement_mm = settings["maximum_side_disagreement_mm"]
        if self.backend != TOTALSEGMENTATOR_LANDMARK_BACKEND:
            raise ValueError(f"Unknown measurement-landmark backend {self.backend!r}.")
        self.io_inputs = [TOTALSEG_LANDMARK_LABEL, "tmp/prepared_image"]
        self.io_outputs = ["tmp/measurement_landmarks"]
        self.licenses = ["totalsegmentator", "nnunet"]

    def __call__(self, memory):
        super().__call__(memory)
        from totalsegmentator.map_to_binary import class_map
        from BodyComposition.measurement.totalsegmentator_assets import (
            require_measurement_model,
        )

        prepared = memory["tmp/prepared_image"].prepared_image
        geometry = ImageGeometry.from_sitk(prepared)
        label = memory[TOTALSEG_LANDMARK_LABEL]
        label.validate()
        assert_same_physical_domain(
            geometry,
            label.geometry,
            reference_name="prepared CT",
            candidate_name="TotalSegmentator landmark label",
        )
        memory["tmp/measurement_landmarks"] = landmarks_from_totalsegmentator(
            label.data,
            geometry,
            class_map["total"],
            minimum_voxels=int(self.minimum_voxels),
            maximum_side_disagreement_mm=float(self.maximum_side_disagreement_mm),
            provenance={
                "software_version": self.config["measurements"][
                    "totalsegmentator_version"
                ],
                "inference_mode": "fast_total_3mm_full_volume_cache_only",
                "upstream_roi_subset_used": False,
                "consumed_labels": "bilateral_hips_and_ribs",
                "model_assets": require_measurement_model(
                    "body_landmarks"
                ).provenance(),
            },
        )


class MeasureCanonicalBodyComposition(PipelineAction):
    """Create canonical measurement tables from aligned prepared-image products."""

    def __init__(
        self,
        pipeline,
        *,
        tissue_mask: str,
        tissue_backend_id: str,
        vertebral_body_source: str,
        compartment_mask: str | None = None,
    ):
        super().__init__(pipeline)
        self.tissue_mask_name = tissue_mask
        self.tissue_backend_id = tissue_backend_id
        self.compartment_mask_name = compartment_mask
        self.vertebral_body_source_name = vertebral_body_source
        self.settings = self.config["measurements"]
        self.timestamp = pipeline.timestamp
        self.io_inputs = [
            "tmp/prepared_image",
            tissue_mask,
            "tmp/vertebral_result",
            "tmp/body_surface_result",
        ]
        if compartment_mask is not None:
            self.io_inputs.append(compartment_mask)
        if self.settings["landmarks"]["enabled"]:
            self.io_inputs.append("tmp/measurement_landmarks")
        self.io_outputs = [
            "tmp/measurement_bundle",
            "tmp/slice_measurements",
            "tmp/vertebra_measurements",
            "tmp/case_summaries",
        ]

    def __call__(self, memory):
        super().__call__(memory)
        prepared = memory["tmp/prepared_image"].prepared_image
        geometry = ImageGeometry.from_sitk(prepared)
        image_zyx = sitk.GetArrayFromImage(prepared)
        tissue = memory[self.tissue_mask_name]
        tissue.validate()
        assert_same_physical_domain(
            geometry,
            tissue.geometry,
            reference_name="prepared CT",
            candidate_name="tissue mask",
        )
        compartment = (
            memory[self.compartment_mask_name]
            if self.compartment_mask_name is not None
            else tissue
        )
        compartment.validate()
        assert_same_physical_domain(
            geometry,
            compartment.geometry,
            reference_name="prepared CT",
            candidate_name="raw tissue-compartment labels",
        )
        body_surface = memory["tmp/body_surface_result"]
        vertebral_result = memory["tmp/vertebral_result"]
        orientation_result = memory["tmp/prepared_image"].result
        measurement_config = _scientific_measurement_configuration(self.settings)
        tissue_preprocessing = {
            "profile_id": self.config["tissue"]["profile_id"],
            "hu_denoise": self.config["tissue"]["hu_denoise"],
            "tissue_rules": {
                name: self.config["tissue"][name]
                for name in ("imat", "sm", "vat", "sat")
            },
        }
        analysis_id = memory.get("analysis_id") or measurement_analysis_id(
            image_zyx,
            tissue.data,
            body_surface,
            vertebral_result,
            measurement_config,
            landmarks=memory.get("tmp/measurement_landmarks"),
            tissue_label_schema=self.config["LBL_TISSUE"],
            tissue_preprocessing=tissue_preprocessing,
            compartment_labels_zyx=compartment.data,
            compartment_label_schema=self.config["LBL_TISSUE"],
            orientation_provenance=orientation_result.to_dict(),
        )
        identity = MeasurementIdentity(
            case_id=str(memory["id"]),
            run_id=str(memory.get("run_id", f"run-{self.timestamp}")),
            analysis_id=str(analysis_id),
        )
        bundle = build_measurement_bundle(
            image_zyx=image_zyx,
            tissue_labels_zyx=tissue.data,
            geometry=geometry,
            tissue_label_schema=self.config["LBL_TISSUE"],
            tissue_backend_id=self.tissue_backend_id,
            tissue_preprocessing=tissue_preprocessing,
            body_surface=body_surface,
            vertebral_result=vertebral_result,
            identity=identity,
            landmarks=memory.get("tmp/measurement_landmarks"),
            settings=self.settings,
            orientation_changed=bool(orientation_result.orientation_changed),
            orientation_provenance=orientation_result.to_dict(),
            compartment_labels_zyx=compartment.data,
            compartment_label_schema=self.config["LBL_TISSUE"],
        )
        memory["analysis_id"] = identity.analysis_id
        memory["run_id"] = identity.run_id
        memory["tmp/measurement_bundle"] = bundle
        memory["tmp/slice_measurements"] = bundle.slices
        memory["tmp/vertebra_measurements"] = bundle.vertebrae
        memory["tmp/case_summaries"] = bundle.summaries


class WriteMeasurementReview(PipelineAction):
    """Render the identifier-minimized measurement stage contour/curve review artifact."""

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.io_inputs = ["tmp/prepared_image", "tmp/measurement_bundle"]
        self.io_outputs = [MEASUREMENT_REVIEW]
        self.io_persisted_outputs = [MEASUREMENT_REVIEW]
        self.io_reset_outputs = [MEASUREMENT_REVIEW]

    def __call__(self, memory):
        super().__call__(memory)
        bundle = memory["tmp/measurement_bundle"]
        path = write_measurement_review(
            memory["tmp/prepared_image"].prepared_image,
            bundle,
            _path(memory, MEASUREMENT_REVIEW),
        )
        memory[MEASUREMENT_REVIEW] = path


class ExportMeasurementBundle(PipelineAction):
    """Atomically persist the three authoritative Parquet tables and QC JSON."""

    table_templates = {
        "slices": SLICE_TABLE,
        "vertebrae": VERTEBRA_TABLE,
        "summaries": SUMMARY_TABLE,
    }

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.io_inputs = ["tmp/measurement_bundle"]
        self.io_outputs = [*self.table_templates.values(), MEASUREMENT_QC, "tmp/return"]
        self.io_persisted_outputs = [*self.table_templates.values(), MEASUREMENT_QC]
        self.io_reset_outputs = [*self.table_templates.values(), MEASUREMENT_QC]

    def __call__(self, memory):
        super().__call__(memory)
        bundle = memory["tmp/measurement_bundle"]
        paths = {
            name: _atomic_parquet(
                getattr(bundle, name),
                _path(memory, template),
                table_name=name,
                identity=bundle.identity.as_columns(),
            )
            for name, template in self.table_templates.items()
        }
        qc_path = _atomic_json(
            {
                **bundle.identity.as_columns(),
                "qc_status": bundle.qc_status.value,
                "manual_review_required": bundle.qc_status.value in {"review", "fail"},
                "body_surface_backend": bundle.body_surface.backend_id,
                "provenance": dict(bundle.provenance),
                "qc_flags": [flag.as_dict() for flag in bundle.qc_flags],
            },
            _path(memory, MEASUREMENT_QC),
        )
        paths["qc"] = qc_path
        for template in self.table_templates.values():
            memory[template] = _path(memory, template)
        memory[MEASUREMENT_QC] = qc_path
        bundle = replace(bundle, paths=paths)
        memory["tmp/measurement_bundle"] = bundle
        memory["tmp/return"] = bundle


def measurement_support_actions(
    pipeline,
    *,
    tissue_mask: str,
) -> list[PipelineAction]:
    """Return explicit body/landmark actions; never insert a silent fallback."""

    settings = pipeline.config["measurements"]
    body_backend = settings["body_surface"]["backend"]
    landmarks_enabled = settings["landmarks"]["enabled"]
    needs_totalsegmentator = body_backend == TOTALSEGMENTATOR_BODY_BACKEND or landmarks_enabled
    actions: list[PipelineAction] = []
    if needs_totalsegmentator:
        from BodyComposition.actions.segm_totalsegmentator import (
            SegmTotalSegmentator,
            SegmTotalSegmentatorConfig,
        )

        actions.append(SegmTotalSegmentatorConfig(pipeline))
        if body_backend == TOTALSEGMENTATOR_BODY_BACKEND:
            actions.append(SegmTotalSegmentator(pipeline, image="tmp/index", task="bodytrunk"))
        if landmarks_enabled:
            if settings["landmarks"]["backend"] != TOTALSEGMENTATOR_LANDMARK_BACKEND:
                raise ValueError(
                    f"Unknown landmark backend {settings['landmarks']['backend']!r}."
                )
            actions.append(
                SegmTotalSegmentator(
                    pipeline,
                    image="tmp/index",
                    task="body_landmarks",
                    fast=True,
                )
            )
    actions.append(CreateBodySurface(pipeline, tissue_mask=tissue_mask))
    if landmarks_enabled:
        actions.append(CreateMeasurementLandmarks(pipeline))
    return actions
