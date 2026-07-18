"""Single canonical action composition used by the release service."""

from __future__ import annotations


def canonical_actions(pipeline):
    from BodyComposition.actions.masks_int import MasksInternalTissue
    from BodyComposition.actions.measurement import (
        ExportMeasurementBundle,
        MeasureCanonicalBodyComposition,
        WriteMeasurementReview,
        measurement_support_actions,
    )
    from BodyComposition.actions.segm_int import SegmIntBodyComposition
    from BodyComposition.actions.vertebral import vertebral_backend_actions

    if not pipeline.config["orientation"]["enabled"]:
        raise ValueError(
            "The canonical pipeline requires orientation.enabled=true "
            "so every downstream stage consumes the orientation-prepared image object."
        )
    tissue_backend = pipeline.public_config.tissue_backend
    tissue_models = {
        "bodycomposition_resenc_l_v1": "ResEncL",
        "bodycomposition_resenc_m_v1": "ResEncM",
    }
    try:
        tissue_model = tissue_models[tissue_backend]
    except KeyError as error:
        raise ValueError(f"Unknown tissue backend {tissue_backend!r}.") from error
    vertebral_actions, vertebral_body_source = vertebral_backend_actions(pipeline)
    tissue_mask = "masks/tissue_labels.nii.gz"
    compartment_mask = "masks/tissue_compartments.nii.gz"
    measurement_actions = measurement_support_actions(
        pipeline,
        tissue_mask=tissue_mask,
    )
    actions = [
        *vertebral_actions,
        SegmIntBodyComposition(pipeline, image="tmp/index", model=tissue_model),
        MasksInternalTissue(pipeline, image="tmp/index"),
        *measurement_actions,
        MeasureCanonicalBodyComposition(
            pipeline,
            tissue_mask=tissue_mask,
            tissue_backend_id=tissue_backend,
            vertebral_body_source=vertebral_body_source,
            compartment_mask=compartment_mask,
        ),
    ]
    if pipeline.config["measurements"]["review"]["enabled"]:
        actions.append(WriteMeasurementReview(pipeline))
    actions.append(ExportMeasurementBundle(pipeline))
    if pipeline.config["reporting"]["enabled"]:
        from BodyComposition.actions.reporting import RenderCaseReport

        actions.append(RenderCaseReport(pipeline))
    return actions
