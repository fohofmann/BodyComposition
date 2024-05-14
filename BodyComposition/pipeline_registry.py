# This file is used to register the available pipelines
# It is used by the pipeline builder to create a pipeline instance by name
# The pipeline build is used to process each case
# different executers can be used to process the pipeline

# pipeline definitions import
from BodyComposition.pipelines.totalsegmentator import SarcopeniaTotalSegmentator, SarcopeniaTotalSegmentatorFast

# add other pipelines to import here

# dictionary of pipeline definitions
pipeline_registry = {
    'SarcopeniaTotalSegmentatorFast': SarcopeniaTotalSegmentatorFast,
    'SarcopeniaTotalSegmentator': SarcopeniaTotalSegmentator,

    # add new pipelines here
}