
def SarcopeniaStanford(pipeline):
    """Pipeline definition for body composition analysis."""
    from BodyComposition.actions.segm_totalsegmentator import SegmTotalSegmentatorConfig, SegmTotalSegmentator
    from BodyComposition.actions.segm_stanford_spine import SegmStanfordSpine
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.calc_csa import CalcCSA
    from BodyComposition.actions.data_postprocessing import DataCombine, DataExport
    from BodyComposition.actions.masks_stanford import MasksStanfordSpine
    from BodyComposition.actions.masks_totalsegmentator import MasksTotalSegmentatorTissue
    from BodyComposition.actions.data_loading import LoadMetadata
   
    # pipeline step definition
    pipeline_definition = [
        # segmentations
        SegmTotalSegmentatorConfig(pipeline),
        SegmStanfordSpine(pipeline, image='tmp/index'),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='vertebralbodies'),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='bodytrunk'),
        SegmTotalSegmentator(pipeline, image='tmp/index', task='tissue'),

        # postprocessing TotalSegmentator mask
        MasksStanfordSpine(pipeline, reduce_to_vb=True),
        MasksTotalSegmentatorTissue(pipeline, image='tmp/index', iliopsoas=False, bodytrunk=True),

        # calculations
        CalcVertebralLevel(pipeline, mask='masks/{caseid}_stanford-vertebrae.nii.gz'),
        CalcCSA(pipeline, mask='masks/{caseid}_tseg-tissue.nii.gz'),

        # postprocessing and export
        LoadMetadata(pipeline, input='metadata/{caseid}.csv'),
        DataCombine(pipeline),
        DataExport(pipeline, file='exports/{caseid}.csv', append=False, add_metadata=True),
        DataExport(pipeline, file='exports/all.csv', append=True, add_metadata=True),
    ]
    return pipeline_definition
