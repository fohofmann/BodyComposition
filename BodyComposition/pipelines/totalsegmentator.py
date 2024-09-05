import os

def SarcopeniaTotalSegmentator(pipeline):
    """Pipeline definition for body composition analysis."""
    from BodyComposition.actions.segm_totalsegmentator import SegmTotalSegmentatorConfig, SegmTotalSegmentator
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.calc_csa import CalcCSA
    from BodyComposition.actions.data_postprocessing import DataCombine, DataExport
    from BodyComposition.actions.masks_totalsegmentator import MasksTotalSegmentatorSpine, MasksTotalSegmentatorTissue
    from BodyComposition.actions.data_loading import LoadMetadata
   
    # pipeline step definition
    pipeline_definition = [
        SegmTotalSegmentatorConfig(pipeline),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='spine'),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='vertebralbodies'),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='bodytrunk'),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='tissue'),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='iliopsoas'),
        
        # postprocessing TotalSegmentator masks
        MasksTotalSegmentatorSpine(pipeline, reduce_to_vb=True),
        MasksTotalSegmentatorTissue(pipeline, image='tmp/index', iliopsoas=True, bodytrunk=True),

        # calculations
        CalcVertebralLevel(pipeline, mask='masks/{caseid}_tseg-vertebrae.nii.gz'),
        CalcCSA(pipeline, mask='masks/{caseid}_tseg-tissue.nii.gz'),

        # postprocessing and export
        LoadMetadata(pipeline, input='metadata/{caseid}.csv'),
        DataCombine(pipeline),
        DataExport(pipeline, file='exports/{caseid}.csv', append=False, add_metadata=True),
        DataExport(pipeline, file='exports/all.csv', append=True, add_metadata=True),
    ]
    return pipeline_definition



def SarcopeniaTotalSegmentatorFast(pipeline):
    """Pipeline definition for body composition analysis using Cropping."""
    from BodyComposition.actions.segm_totalsegmentator import SegmTotalSegmentatorConfig, SegmTotalSegmentator
    from BodyComposition.actions.crop import CreateBoundingBox, ApplyBoundingBox
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.calc_csa import CalcCSA
    from BodyComposition.actions.data_postprocessing import DataCombine, DataSubset, DataAggregate, DataExport
    from BodyComposition.actions.masks_totalsegmentator import MasksTotalSegmentatorSpine, MasksTotalSegmentatorTissue
    from BodyComposition.actions.data_loading import LoadMetadata

    # pipeline step definition
    pipeline_definition = [
        # localization vertebrae
        SegmTotalSegmentatorConfig(pipeline),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='spine', fast=True),

        # crop to L2-4
        MasksTotalSegmentatorSpine(pipeline, reduce_to_vb=False),
        CreateBoundingBox(pipeline, label='masks/{caseid}_tseg-vertebrae.nii.gz', task='L234CranioCaudal'),
        ApplyBoundingBox(pipeline, input='tmp/index'),
        ApplyBoundingBox(pipeline, input='masks/{caseid}_tseg-vertebrae.nii.gz'),
        
        # segmentation area
        SegmTotalSegmentator(pipeline, image='tmp/index', task='tissue', fast=True),

        # postprocessing TotalSegmentator masks
        MasksTotalSegmentatorTissue(pipeline, image='tmp/index', iliopsoas=False, bodytrunk=False),

        # calculations
        CalcVertebralLevel(pipeline, mask='masks/{caseid}_tseg-vertebrae.nii.gz'),
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
