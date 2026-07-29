"""Body-surface, landmark, measurement, and canonical export actions."""

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
from BodyComposition.measurement.csv_export import write_csv_table
from BodyComposition.measurement.landmarks import (
    TOTALSEGMENTATOR_LANDMARK_BACKEND,
    landmarks_from_totalsegmentator,
)
from BodyComposition.measurement.review import write_measurement_review
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.utils.nifti import NiftiDataContainer

BODY_SURFACE_MASK = "masks/body_surface.nii.gz"
MEASUREMENT_REVIEW = "qc/measurement_review.png"
MEASUREMENT_QC = "qc/qc.json"
SLICE_TABLE = "tables/slices.parquet"
VERTEBRA_TABLE = "tables/vertebrae.parquet"
SUMMARY_TABLE = "tables/summaries.parquet"
SIGNATURE_TABLE = "tables/signature.parquet"
HU_DISTRIBUTION_TABLE = "tables/hu_distributions.parquet"
SLICE_TABLE_CSV = "tables/slices.csv"
VERTEBRA_TABLE_CSV = "tables/vertebrae.csv"
SUMMARY_TABLE_CSV = "tables/summaries.csv"
SIGNATURE_TABLE_CSV = "tables/signature.csv"
HU_DISTRIBUTION_TABLE_CSV = "tables/hu_distributions.csv"
TISSUE_LABEL_MASK = "masks/tissue_labels.nii.gz"
TOTALSEG_BODY_LABEL = "masks/totalsegmentator_body.nii.gz"
TOTALSEG_LANDMARK_LABEL = "masks/totalsegmentator_landmarks.nii.gz"


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

    def __init__(self, pipeline, *, compartment_mask: str):
        super().__init__(pipeline)
        self.settings = self.config["measurements"]["body_surface"]
        self.backend = self.settings["backend"]
        self.compartment_mask_name = compartment_mask
        if self.backend == TOTALSEGMENTATOR_BODY_BACKEND:
            self.input_name = TOTALSEG_BODY_LABEL
        elif self.backend == DETERMINISTIC_BODY_BACKEND:
            self.input_name = "tmp/prepared_image"
        elif self.backend == TISSUE_ENVELOPE_BACKEND:
            self.input_name = compartment_mask
        else:
            raise ValueError(f"Unknown body-surface backend {self.backend!r}.")
        self.io_inputs = [self.input_name, "tmp/prepared_image"]
        self.io_outputs = ["tmp/body_surface_result", BODY_SURFACE_MASK]
        self.io_reset_outputs = [BODY_SURFACE_MASK]
        if self.settings["save_mask"]:
            self.io_persisted_outputs = [BODY_SURFACE_MASK]

    def __call__(self, memory):
        super().__call__(memory)
        started = time()
        prepared = memory["tmp/prepared_image"].prepared_image
        geometry = ImageGeometry.from_sitk(prepared)
        if self.backend == TOTALSEGMENTATOR_BODY_BACKEND:
            from BodyComposition.measurement.totalsegmentator_assets import (
                require_measurement_model,
            )

            model_root = self.config["paths"]["weights"]["totalsegmentator"]
            asset_report = require_measurement_model("bodytrunk", model_root)
            label = memory[self.input_name]
            label.validate()
            assert_same_physical_domain(
                geometry,
                label.geometry,
                reference_name="orientation-prepared CT",
                candidate_name="TotalSegmentator body label",
            )
            result = body_surface_from_totalsegmentator(
                label.data,
                geometry,
                provenance={
                    "inference_engine": "nnUNetv2",
                    "inference_engine_version": self.config["measurements"][
                        "measurement_support_nnunet_version"
                    ],
                    "inference_mode": "pinned_task299_full_volume_cache_only",
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
                reference_name="orientation-prepared CT",
                candidate_name="model-native compartments used for body envelope",
            )
            result = tissue_segmentation_envelope(
                tissue.data,
                geometry,
                minimum_component_area_mm2=float(
                    self.settings["minimum_component_area_mm2"]
                ),
                closing_radius_mm=float(self.settings["closing_radius_mm"]),
                smoothing_sigma_mm=float(self.settings["smoothing_sigma_mm"]),
                provenance={"tissue_mask_input": "raw_model_compartments"},
            )

        encoded = np.where(result.body_mask_zyx, 2, 0).astype(np.uint8)
        encoded[result.trunk_mask_zyx] = 1
        image = sitk.GetImageFromArray(encoded)
        image.CopyInformation(prepared)
        output = NiftiDataContainer(
            _path(memory, BODY_SURFACE_MASK),
            dtype=np.uint8,
        )
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

    def __call__(self, memory):
        super().__call__(memory)
        from BodyComposition.measurement.totalsegmentator_assets import (
            measurement_label_schema,
            require_measurement_model,
        )

        model_root = self.config["paths"]["weights"]["totalsegmentator"]
        asset_report = require_measurement_model("body_landmarks", model_root)
        prepared = memory["tmp/prepared_image"].prepared_image
        geometry = ImageGeometry.from_sitk(prepared)
        label = memory[TOTALSEG_LANDMARK_LABEL]
        label.validate()
        assert_same_physical_domain(
            geometry,
            label.geometry,
            reference_name="orientation-prepared CT",
            candidate_name="TotalSegmentator landmark label",
        )
        memory["tmp/measurement_landmarks"] = landmarks_from_totalsegmentator(
            label.data,
            geometry,
            measurement_label_schema("body_landmarks", model_root),
            minimum_voxels=int(self.minimum_voxels),
            maximum_side_disagreement_mm=float(self.maximum_side_disagreement_mm),
            provenance={
                "inference_engine": "nnUNetv2",
                "inference_engine_version": self.config["measurements"][
                    "measurement_support_nnunet_version"
                ],
                "inference_mode": "pinned_task297_full_volume_cache_only",
                "upstream_roi_subset_used": False,
                "consumed_labels": "bilateral_hips_and_ribs",
                "model_assets": asset_report.provenance(),
            },
        )


class MeasureCanonicalBodyComposition(PipelineAction):
    """Create canonical measurement tables from aligned prepared-image products."""

    def __init__(
        self,
        pipeline,
        *,
        compartment_mask: str,
        tissue_backend_id: str,
        vertebral_body_source: str,
    ):
        super().__init__(pipeline)
        self.tissue_backend_id = tissue_backend_id
        self.compartment_mask_name = compartment_mask
        self.vertebral_body_source_name = vertebral_body_source
        self.settings = self.config["measurements"]
        self.analysis_scope = self.config["analysis"]["scope"]
        self.timestamp = pipeline.timestamp
        self.io_inputs = [
            "tmp/prepared_image",
            compartment_mask,
            TISSUE_LABEL_MASK,
            "tmp/vertebral_result",
            "tmp/body_surface_result",
        ]
        if self.settings["landmarks"]["enabled"]:
            self.io_inputs.append("tmp/measurement_landmarks")
        if self.analysis_scope == "l3_vertebral_level":
            self.io_inputs.append("tmp/tissue_analysis_region")
        self.io_outputs = [
            "tmp/measurement_bundle",
            "tmp/slice_measurements",
            "tmp/vertebra_measurements",
            "tmp/case_summaries",
            "tmp/longitudinal_signature",
            "tmp/compartment_hu_distributions",
        ]

    def __call__(self, memory):
        super().__call__(memory)
        prepared = memory["tmp/prepared_image"].prepared_image
        geometry = ImageGeometry.from_sitk(prepared)
        image_zyx = sitk.GetArrayFromImage(prepared)
        compartment = memory[self.compartment_mask_name]
        compartment.validate()
        assert_same_physical_domain(
            geometry,
            compartment.geometry,
            reference_name="orientation-prepared CT",
            candidate_name="model-native compartment labels",
        )
        tissue_labels = memory[TISSUE_LABEL_MASK]
        tissue_labels.validate()
        assert_same_physical_domain(
            geometry,
            tissue_labels.geometry,
            reference_name="orientation-prepared CT",
            candidate_name="postprocessed tissue-label view",
        )
        body_surface = memory["tmp/body_surface_result"]
        vertebral_result = memory["tmp/vertebral_result"]
        orientation_result = memory["tmp/prepared_image"].result
        analysis_region = memory.get("tmp/tissue_analysis_region")
        analyzed_slices_z = (
            analysis_region.analyzed_slices_mask_z
            if analysis_region is not None
            else None
        )
        analysis_region_provenance = (
            analysis_region.as_dict() if analysis_region is not None else {}
        )
        measurement_config = _scientific_measurement_configuration(self.settings)
        tissue_preprocessing = {
            "source": "raw_model_compartments",
            "profile_id": self.settings["tissue_profile_id"],
            "analysis_scope": self.analysis_scope,
            "analysis_region": analysis_region_provenance,
        }
        analysis_id = memory.get("analysis_id") or measurement_analysis_id(
            image_zyx,
            tissue_labels.data,
            body_surface,
            vertebral_result,
            measurement_config,
            landmarks=memory.get("tmp/measurement_landmarks"),
            tissue_label_schema=self.config["LBL_TISSUE"],
            tissue_preprocessing=tissue_preprocessing,
            compartment_labels_zyx=compartment.data,
            compartment_label_schema=self.config["LBL_TISSUE_COMPARTMENTS"],
            orientation_provenance=orientation_result.to_dict(),
        )
        identity = MeasurementIdentity(
            case_id=str(memory["id"]),
            run_id=str(memory.get("run_id", f"run-{self.timestamp}")),
            analysis_id=str(analysis_id),
        )
        bundle = build_measurement_bundle(
            image_zyx=image_zyx,
            tissue_labels_zyx=tissue_labels.data,
            geometry=geometry,
            tissue_label_schema=self.config["LBL_TISSUE"],
            tissue_backend_id=self.tissue_backend_id,
            tissue_preprocessing=tissue_preprocessing,
            compartment_labels_zyx=compartment.data,
            compartment_label_schema=self.config["LBL_TISSUE_COMPARTMENTS"],
            body_surface=body_surface,
            vertebral_result=vertebral_result,
            identity=identity,
            landmarks=memory.get("tmp/measurement_landmarks"),
            settings=self.settings,
            orientation_provenance=orientation_result.to_dict(),
            analysis_scope=self.analysis_scope,
            analyzed_slices_z=analyzed_slices_z,
            analysis_region_provenance=analysis_region_provenance,
        )
        memory["analysis_id"] = identity.analysis_id
        memory["run_id"] = identity.run_id
        memory["tmp/measurement_bundle"] = bundle
        memory["tmp/slice_measurements"] = bundle.slices
        memory["tmp/vertebra_measurements"] = bundle.vertebrae
        memory["tmp/case_summaries"] = bundle.summaries
        memory["tmp/longitudinal_signature"] = bundle.signature
        memory["tmp/compartment_hu_distributions"] = bundle.hu_distributions


class WriteMeasurementReview(PipelineAction):
    """Render the identifier-minimized contour and curve review artifact."""

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
    """Persist canonical Parquet tables, optional CSV mirrors, and QC JSON."""

    table_templates = {
        "slices": SLICE_TABLE,
        "vertebrae": VERTEBRA_TABLE,
        "summaries": SUMMARY_TABLE,
        "signature": SIGNATURE_TABLE,
        "hu_distributions": HU_DISTRIBUTION_TABLE,
    }
    csv_table_templates = {
        "slices": SLICE_TABLE_CSV,
        "vertebrae": VERTEBRA_TABLE_CSV,
        "summaries": SUMMARY_TABLE_CSV,
        "signature": SIGNATURE_TABLE_CSV,
        "hu_distributions": HU_DISTRIBUTION_TABLE_CSV,
    }

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.csv_enabled = bool(self.config["measurements"]["export"]["csv"])
        csv_outputs = (
            list(self.csv_table_templates.values()) if self.csv_enabled else []
        )
        self.io_inputs = ["tmp/measurement_bundle"]
        self.io_outputs = [
            *self.table_templates.values(),
            *csv_outputs,
            MEASUREMENT_QC,
            "tmp/return",
        ]
        self.io_persisted_outputs = [
            *self.table_templates.values(),
            *csv_outputs,
            MEASUREMENT_QC,
        ]
        self.io_reset_outputs = [
            *self.table_templates.values(),
            *csv_outputs,
            MEASUREMENT_QC,
        ]

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
        if self.csv_enabled:
            paths.update(
                {
                    f"{name}_csv": write_csv_table(
                        getattr(bundle, name),
                        _path(memory, template),
                        overwrite=True,
                    )
                    for name, template in self.csv_table_templates.items()
                }
            )
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
        if self.csv_enabled:
            for template in self.csv_table_templates.values():
                memory[template] = _path(memory, template)
        memory[MEASUREMENT_QC] = qc_path
        bundle = replace(bundle, paths=paths)
        memory["tmp/measurement_bundle"] = bundle
        memory["tmp/return"] = bundle


def measurement_support_actions(
    pipeline,
    *,
    compartment_mask: str,
) -> list[PipelineAction]:
    """Return explicit body/landmark actions; never insert a silent fallback."""

    settings = pipeline.config["measurements"]
    body_backend = settings["body_surface"]["backend"]
    landmarks_enabled = settings["landmarks"]["enabled"]
    needs_measurement_models = (
        body_backend == TOTALSEGMENTATOR_BODY_BACKEND or landmarks_enabled
    )
    actions: list[PipelineAction] = []
    if needs_measurement_models:
        from BodyComposition.actions.segm_totalsegmentator import SegmTotalSegmentator

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
                )
            )
    actions.append(
        CreateBodySurface(pipeline, compartment_mask=compartment_mask)
    )
    if landmarks_enabled:
        actions.append(CreateMeasurementLandmarks(pipeline))
    return actions
