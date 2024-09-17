import os

def BodyAndOrganAnalysis(pipeline):
    """Pipeline definition for body composition analysis."""
    from BodyComposition.actions.segm_totalsegmentator import SegmTotalSegmentatorConfig, SegmTotalSegmentator
    from BodyComposition.actions.segm_boa import SegmBOA
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.calc_measures import CalcMeasures
    from BodyComposition.actions.data_postprocessing import DataCombine, DataExport
    from BodyComposition.actions.masks_totalsegmentator import MasksTotalSegmentatorSpine
    from BodyComposition.actions.masks_boa import MasksBoaTissue
    from BodyComposition.actions.data_loading import LoadMetadata
   
    # pipeline step definition
    pipeline_definition = [
        SegmTotalSegmentatorConfig(pipeline),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='spine'),
        SegmBOA(pipeline, image='tmp/index'),
        
        # postprocessing TotalSegmentator masks
        MasksTotalSegmentatorSpine(pipeline, reduce_to_vb=False),
        MasksBoaTissue(pipeline, image='tmp/index'),

        # calculations
        CalcVertebralLevel(pipeline, mask='masks/{caseid}_tseg-vertebrae.nii.gz'),
        CalcMeasures(pipeline, mask='masks/{caseid}_boa.nii.gz'),

        # postprocessing and export
        LoadMetadata(pipeline, input='metadata/{caseid}.csv'),
        DataCombine(pipeline),
        DataExport(pipeline, file='exports/{caseid}_raw.csv', append=False, add_metadata=True),
        DataExport(pipeline, file='exports/all.csv', append=True, add_metadata=True),
    ]
    return pipeline_definition


