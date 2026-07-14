def SarcopeniaStanfordFast(pipeline):
    """Pipeline definition for body composition analysis."""
    from BodyComposition.actions.segm_stanford import SegmStanfordSpine, SegmStanfordTissue
    from BodyComposition.actions.crop import CreateBoundingBox, ApplyBoundingBox
    from BodyComposition.actions.masks_stanford import MasksStanfordSpine, MasksStanfordTissue
    from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
    from BodyComposition.actions.calc_measures import CalcMeasures
    from BodyComposition.actions.data_postprocessing import DataCombine, DataSubset, DataAggregate, DataExport
    from BodyComposition.actions.data_loading import LoadMetadata
   
    # pipeline step definition
    pipeline_definition = [
        # segmentation to localize L3
        SegmStanfordSpine(pipeline, image='tmp/index'),

        # crop image to L234
        MasksStanfordSpine(pipeline),
        CreateBoundingBox(pipeline, label='masks/{caseid}_stanford-spine.nii.gz', task='L234CranioCaudal'),
        ApplyBoundingBox(pipeline, input='tmp/index'),
        ApplyBoundingBox(pipeline, input='masks/{caseid}_stanford-spine.nii.gz'),

        # segmentation tissue
        SegmStanfordTissue(pipeline, image='tmp/index'),

        # postprocessing Stanford tissue mask
        MasksStanfordTissue(pipeline, image='tmp/index'),

        # calculations
        CalcVertebralLevel(pipeline, mask='masks/{caseid}_stanford-spine.nii.gz'),
        CalcMeasures(pipeline, mask='masks/{caseid}_stanford-tissue.nii.gz'),

        # postprocessing and export
        LoadMetadata(pipeline, input='metadata/{caseid}.csv'),
        DataCombine(pipeline),
        DataExport(pipeline),
        DataSubset(pipeline, ref='Level', level=['L3']),
        DataAggregate(pipeline, method='mean', ref='Level'),
        DataExport(pipeline, file='exports/all_L3Mean.csv', append=True, add_metadata=True),
    ]
    return pipeline_definition
