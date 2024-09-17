import os

def BodyComposition(pipeline):
    """Pipeline definition for body composition analysis."""
    from BodyComposition.actions.segm_int import SegmIntVertebrae, SegmIntBodyComposition
    from BodyComposition.actions.masks_int import MasksInternalTissue
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.calc_measures import CalcMeasures
    from BodyComposition.actions.data_postprocessing import DataCombine, DataExport
    from BodyComposition.actions.data_loading import LoadMetadata
   
    # pipeline step definition
    pipeline_definition = [
        # segmentations
        SegmIntVertebrae(pipeline, image='tmp/index', model='ResEncL'),
        SegmIntBodyComposition(pipeline, image='tmp/index', model='ResEncL'),

        # postprocessing mask
        MasksInternalTissue(pipeline, image='tmp/index'),

        # calculations
        CalcVertebralLevel(pipeline, mask='labels/{caseid}_int-vertebrae.nii.gz'),
        CalcMeasures(pipeline, mask='masks/{caseid}_int-bodycomposition.nii.gz'),

        # postprocessing and export
        LoadMetadata(pipeline, input='metadata/{caseid}.csv'),
        DataCombine(pipeline),
        DataExport(pipeline, file='exports/{caseid}_raw.csv', append=False, add_metadata=True),
        DataExport(pipeline, file='exports/all.csv', append=True, add_metadata=True),
    ]
    return pipeline_definition



def BodyCompositionFast(pipeline):
    """Pipeline definition for body composition analysis."""
    from BodyComposition.actions.segm_int import SegmIntVertebrae, SegmIntBodyComposition
    from BodyComposition.actions.masks_int import MasksInternalTissue
    from BodyComposition.actions.crop import CreateBoundingBox, ApplyBoundingBox
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.calc_measures import CalcMeasures
    from BodyComposition.actions.data_postprocessing import DataCombine, DataSubset, DataAggregate, DataExport
    from BodyComposition.actions.data_loading import LoadMetadata
   
    # pipeline step definition
    pipeline_definition = [
        # segmentation to localize L3
        SegmIntVertebrae(pipeline, image='tmp/index', model='ResEncM'),

        # crop image to L234
        CreateBoundingBox(pipeline, label='labels/{caseid}_int-vertebrae.nii.gz', task='L234CranioCaudal'),
        ApplyBoundingBox(pipeline, input='tmp/index'),
        ApplyBoundingBox(pipeline, input='labels/{caseid}_int-vertebrae.nii.gz'),

        # segmentation area
        SegmIntBodyComposition(pipeline, image='tmp/index', model='ResEncM'),

        # postprocessing masks
        MasksInternalTissue(pipeline, image='tmp/index'),

        # calculations
        CalcVertebralLevel(pipeline, mask='labels/{caseid}_int-vertebrae.nii.gz'),
        CalcMeasures(pipeline, mask='masks/{caseid}_int-bodycomposition.nii.gz'),

        # postprocessing and export
        LoadMetadata(pipeline, input='metadata/{caseid}.csv'),
        DataCombine(pipeline),
        DataExport(pipeline),
        DataSubset(pipeline, ref='Level', level=['L3']),
        DataAggregate(pipeline, method='mean', ref='Level'),
        DataExport(pipeline, file='exports/all_L3Mean.csv', append=True, add_metadata=True),
    ]
    return pipeline_definition
