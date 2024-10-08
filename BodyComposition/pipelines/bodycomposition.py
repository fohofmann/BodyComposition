import os

def BodyComposition(pipeline):
    """Pipeline definition for body composition analysis."""
    from BodyComposition.actions.segm_totalsegmentator import SegmTotalSegmentatorConfig, SegmTotalSegmentator
    from BodyComposition.actions.segm_int_vertebrae import SegmIntVertebrae
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.calc_csa import CalcCSA
    from BodyComposition.actions.data_postprocessing import DataCombine, DataExport
    from BodyComposition.actions.masks_totalsegmentator import MasksTotalSegmentatorTissue
    from BodyComposition.actions.data_loading import LoadMetadata
   
    # pipeline step definition
    pipeline_definition = [
        # segmentations
        SegmTotalSegmentatorConfig(pipeline),
        SegmIntVertebrae(pipeline, image='tmp/index', model='ResEncL'),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='bodytrunk'),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='tissue'),

        # postprocessing TotalSegmentator mask
        MasksTotalSegmentatorTissue(pipeline, image='tmp/index', iliopsoas=False, bodytrunk=True),

        # calculations
        CalcVertebralLevel(pipeline, mask='labels/{caseid}_int-vertebrae.nii.gz'),
        CalcCSA(pipeline, mask='masks/{caseid}_tseg-tissue.nii.gz'),

        # postprocessing and export
        LoadMetadata(pipeline, input='metadata/{caseid}.csv'),
        DataCombine(pipeline),
        DataExport(pipeline, file='exports/{caseid}.csv', append=False, add_metadata=True),
        DataExport(pipeline, file='exports/all.csv', append=True, add_metadata=True),
    ]
    return pipeline_definition



def BodyCompositionFast(pipeline):
    """Pipeline definition for body composition analysis."""
    from BodyComposition.actions.segm_int_vertebrae import SegmIntVertebrae
    from BodyComposition.actions.segm_totalsegmentator import SegmTotalSegmentatorConfig, SegmTotalSegmentator
    from BodyComposition.actions.crop import CreateBoundingBox, ApplyBoundingBox
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.calc_csa import CalcCSA
    from BodyComposition.actions.data_postprocessing import DataCombine, DataSubset, DataAggregate, DataExport
    from BodyComposition.actions.masks_totalsegmentator import MasksTotalSegmentatorTissue
    from BodyComposition.actions.data_loading import LoadMetadata
   
    # pipeline step definition
    pipeline_definition = [
        # load TotalSegmentator config
        SegmTotalSegmentatorConfig(pipeline),

        # segmentation to localize L3
        SegmIntVertebrae(pipeline, image='tmp/index', model='ResEncM'),

        # as
        CreateBoundingBox(pipeline, label='labels/{caseid}_int-vertebrae.nii.gz', task='L234CranioCaudal'),
        ApplyBoundingBox(pipeline, input='tmp/index'),
        ApplyBoundingBox(pipeline, input='labels/{caseid}_int-vertebrae.nii.gz'),

        # segmentation area
        SegmTotalSegmentator(pipeline, image='tmp/index', task='tissue', fast=True),

        # postprocessing TotalSegmentator masks
        MasksTotalSegmentatorTissue(pipeline, image='tmp/index', iliopsoas=False, bodytrunk=False),

        # calculations
        CalcVertebralLevel(pipeline, mask='labels/{caseid}_int-vertebrae.nii.gz'),
        CalcCSA(pipeline, mask='masks/{caseid}_tseg-tissue.nii.gz'),

        # postprocessing and export
        LoadMetadata(pipeline, input='metadata/{caseid}.csv'),
        DataCombine(pipeline),
        DataExport(pipeline),
        DataSubset(pipeline, ref='Level', level=['L3']),
        DataAggregate(pipeline, method='mean', ref='Level'),
        DataExport(pipeline, file='exports/all_L3Mean.csv', append=True, add_metadata=True),
    ]
    return pipeline_definition
