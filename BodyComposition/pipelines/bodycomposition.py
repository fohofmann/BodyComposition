"""Canonical internal-model pipelines with measurement stage physical measurements."""

from __future__ import annotations


def _canonical_bodycomposition_pipeline(pipeline, *, tissue_model: str):
    from BodyComposition.actions.measurement import (
        ExportMeasurementBundle,
        MeasureCanonicalBodyComposition,
        WriteMeasurementReview,
        measurement_support_actions,
    )
    from BodyComposition.actions.masks_int import MasksInternalTissue
    from BodyComposition.actions.segm_int import SegmIntBodyComposition
    from BodyComposition.actions.vertebral import vertebral_backend_actions

    if not pipeline.config["measurements"]["enabled"]:
        raise ValueError("The canonical BodyComposition pipelines require measurements.enabled=true.")
    if not pipeline.config["orientation"]["enabled"]:
        raise ValueError(
            "The canonical BodyComposition pipelines require orientation.enabled=true "
            "so every downstream stage consumes the prepared-image object."
        )
    vertebral_actions, vertebral_body_source = vertebral_backend_actions(pipeline)
    tissue_mask = "masks/{caseid}_int-bodycomposition.nii.gz"
    measurement_actions = measurement_support_actions(
        pipeline,
        tissue_mask=tissue_mask,
    )
    tissue_backend_id = f"bodycomposition_resenc_{tissue_model[-1].lower()}_v1"
    actions = [
        *vertebral_actions,
        SegmIntBodyComposition(pipeline, image="tmp/index", model=tissue_model),
        MasksInternalTissue(pipeline, image="tmp/index"),
        *measurement_actions,
        MeasureCanonicalBodyComposition(
            pipeline,
            tissue_mask=tissue_mask,
            tissue_backend_id=tissue_backend_id,
            vertebral_body_source=vertebral_body_source,
        ),
    ]
    if pipeline.config["measurements"]["review"]["enabled"]:
        actions.append(WriteMeasurementReview(pipeline))
    actions.append(ExportMeasurementBundle(pipeline))
    return actions


def BodyComposition(pipeline):
    """Full-resolution internal tissue model with canonical measurement stage outputs."""

    return _canonical_bodycomposition_pipeline(pipeline, tissue_model="ResEncL")


def BodyCompositionFast(pipeline):
    """Faster internal tissue model on the full prepared CT physical domain."""

    return _canonical_bodycomposition_pipeline(pipeline, tissue_model="ResEncM")
