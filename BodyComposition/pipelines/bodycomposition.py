def BodyComposition(pipeline):
    """Full internal-model body-composition pipeline."""
    from BodyComposition.actions.calc_measures import CalcMeasures
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.data_loading import LoadMetadata
    from BodyComposition.actions.data_postprocessing import DataCombine, DataExport
    from BodyComposition.actions.masks_int import MasksInternalTissue
    from BodyComposition.actions.segm_int import SegmIntBodyComposition, SegmIntVertebrae

    return [
        SegmIntVertebrae(pipeline, image="tmp/index", model="ResEncL"),
        SegmIntBodyComposition(pipeline, image="tmp/index", model="ResEncL"),
        MasksInternalTissue(pipeline, image="tmp/index"),
        CalcVertebralLevel(pipeline, mask="labels/{caseid}_int-vertebrae.nii.gz"),
        CalcMeasures(
            pipeline,
            mask="masks/{caseid}_int-bodycomposition.nii.gz",
            contour_mask="labels/{caseid}_int-bodycomposition.nii.gz",
            calculate_contours=True,
        ),
        LoadMetadata(pipeline, input="metadata/{caseid}.csv"),
        DataCombine(pipeline),
        DataExport(pipeline, file="exports/{caseid}_raw.csv", add_metadata=True),
        DataExport(pipeline, file="exports/all.csv", append=True, add_metadata=True),
    ]


def BodyCompositionFast(pipeline):
    """Cropped internal-model pipeline for L3-focused analysis."""
    from BodyComposition.actions.calc_measures import CalcMeasures
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.crop import ApplyBoundingBox, CreateBoundingBox
    from BodyComposition.actions.data_loading import LoadMetadata
    from BodyComposition.actions.data_postprocessing import (
        DataAggregate,
        DataCombine,
        DataExport,
        DataSubset,
    )
    from BodyComposition.actions.masks_int import MasksInternalTissue
    from BodyComposition.actions.segm_int import SegmIntBodyComposition, SegmIntVertebrae

    return [
        SegmIntVertebrae(pipeline, image="tmp/index", model="ResEncM"),
        CreateBoundingBox(
            pipeline,
            label="labels/{caseid}_int-vertebrae.nii.gz",
            task="L234CranioCaudal",
        ),
        ApplyBoundingBox(pipeline, input="tmp/index"),
        ApplyBoundingBox(pipeline, input="labels/{caseid}_int-vertebrae.nii.gz"),
        SegmIntBodyComposition(pipeline, image="tmp/index", model="ResEncM"),
        MasksInternalTissue(pipeline, image="tmp/index"),
        CalcVertebralLevel(pipeline, mask="labels/{caseid}_int-vertebrae.nii.gz"),
        CalcMeasures(pipeline, mask="masks/{caseid}_int-bodycomposition.nii.gz"),
        LoadMetadata(pipeline, input="metadata/{caseid}.csv"),
        DataCombine(pipeline),
        DataExport(pipeline),
        DataSubset(pipeline, ref="Level", level=["L3"]),
        DataAggregate(pipeline, method="mean", ref="Level"),
        DataExport(
            pipeline,
            file="exports/all_L3Mean.csv",
            append=True,
            add_metadata=True,
        ),
    ]
