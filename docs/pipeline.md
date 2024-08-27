# Pipeline

This tool can be used to run different pipelines.
Pipelines are registred in the [pipeline_registry.py](../BodyComposition/pipelines/pipeline_registry.py) file.
If a registred pipeline is run, the specific pipeline is loaded from the [pipelines directory](../BodyComposition/pipelines/).

The pipeline is a sequence of stages that are executed in order as defined in the respective pipeline file.
Each stage is defined as `action`, which are classes that are derived from the `PipelineAction` class.

At the start of the pipeline, each action class is initialized.
The pipeline selects all cases, where all required inputs (defined by the individual actions) are available.
Then, for each input image, all actions are executed in order.
Outputs of actions are available to the next actions in the pipeline using a shared dictionary.

## Actions

### Segmentation
- **SegmIntVertebrae**: Segments the vertebral body using nnU-Net v2, with models described [here](https://huggingface.co/collections/fhofmann/vertebralbodies-66bb7ff2131e1dbc2143ff20). Requires `image` being a [NIfTI data container](../BodyComposition/utils/nifti.py) and `model` being a string defining either the `ResEncM` or `ResEncL` model. Returns (and optionally saves) the segmentation as a [NIfTI data container](../BodyComposition/utils/nifti.py).
- **SegmStanfordSpine**: Segments the vertebral body using the [Comp2Comp Spine Segmentation model](https://huggingface.co/louisblankemeier/stanford_spine). Requires `image` being a [NIfTI data container](../BodyComposition/utils/nifti.py). Returns (and optionally saves) the segmentation as a [NIfTI data container](../BodyComposition/utils/nifti.py).
- **SegmTotalSegmentator**: Segments any structures using the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator). Requires `image` being a [NIfTI data container](../BodyComposition/utils/nifti.py), `task` being a string defining the [task to be performed](../BodyComposition/actions/segm_totalsegmentator.py) (e.g., spine, bodytrunk, tissue, vertebralbodies, iliospoas), and `fast` being a boolean defining whether TotalSegmentator's *fast* function should be used. Before running this action, TotalSegmentator must be initialized using the **SegmTotalSegmentatorConfig** action. Returns (and optionally saves) the segmentation as a [NIfTI data container](../BodyComposition/utils/nifti.py).

### Masks Spine
- **MasksTotalSegmentatorSpine**: Relabels the TotalSegmentator labels to the [standard labels](labels.md) used in the pipeline. If `reduce_to_vb` is set to `True`, the labels are reduced to the vertebral bodies using TotalSegmentator's `vertebral_body` segmentation. Returns the relabeled masks as a [NIfTI data container](../BodyComposition/utils/nifti.py).
- **MasksStanfordSpine**: Relabels the Comp2Comp Spine labels to the [standard labels](labels.md) used in the pipeline. If `reduce_to_vb` is set to `True`, the labels are reduced to the vertebral bodies using TotalSegmentator's `vertebral_body` segmentation. Returns the relabeled masks as a [NIfTI data container](../BodyComposition/utils/nifti.py).

### Masks Tissue
During the processing of tissue masks, filters based on Hounsfield units and 2D or 3D size properties are be applied to subsegment `labels` and generate `masks` as explained [here](labels.md) and defined in the pipeline's [configuration](config.md). 

- **MasksTotalSegmentatorTissue**: Relabels the TotalSegmentator labels to the [standard labels](labels.md) used in the pipeline. If `iliopsoas` is set to `True`, the a separate label of the iliopsoas muscle is returned using TotalSegmentator's `iliopsoas` segmentation. If `bodytrunk` is set to `True`, the labels are reduced to the body trunk using TotalSegmentator's `bodytrunk` segmentation. Returns the relabeled masks as a [NIfTI data container](../BodyComposition/utils/nifti.py).

### Bounding Boxes
- **CreateBoundingBox**: Creates bounding boxes around specific labels. The segmentation `label` must be provided as a [NIfTI data container](../BodyComposition/utils/nifti.py). The bounding box is defined in the pipeline's [configuration](config.md), and the specific `task` must be defined as argument. The function then creates a bounding box around the specific label and saves it (`bbox`) to the memory dictionary.
- **ApplyBoundingBox**: Applies a bounding box to a image, label or mask. The segmentation `label` must be provided as [NIfTI data containers](../BodyComposition/utils/nifti.py), and the bounding box `bbox` must be available within the memory dictionary. If the NIfTI data container should not be changed, but saved separately, define its name using the `output` argument. The function then applies the bounding box to the input NIfTI. If the changed NIfTI data container is used later on, only values within the bounding box are returned, changed or saved.

### Postprocessing
- **CalcVertebralLevel**: Calculates the vertebral levels based based on a (reorientated) `mask` refering to a [NIfTI data container](../BodyComposition/utils/nifti.py) containing the (postprocessed) vertebral body segmentations. For each slice, the dominating vertebral body (most pixels) is determined using settings as defined in the pipeline's [configuration](config.md). Returns a numpy array containing the vertebral levels (`tmp/vertebrae_values`) to the memory dictionary.
- **CalcCSA**: Calculates the cross-sectional area (CSA) of the tissues based on a (reorientated) `mask` refering to a [NIfTI data container](../BodyComposition/utils/nifti.py) containing the (postprocessed) tissue segmentations. The CSA is calculated for each label in cm², considering the settings as defined in the pipeline's [configuration](config.md). Returns a numpy array containing the CSA values (`tmp/tissue_values`) to the memory dictionary.


### Data Handling
- **LoadMetadata**: Trys to load metadata. The path is given as an argument, with the placeholder `{caseid}` being replaced by the current cases id. Can be both, a *csv (containing DICOM metadata) or a *dcm file. The metadata is saved to the memory dictionary as `tmp/metadata`.
- **DataCombine**: Trys to combine tissue measurements (CSA) and vertebral levels. Checks whether affine, spacing and other metadata match. Returns a pandas dataframe containing the combined data (`tmp/bodycomposition`) to the memory dictionary.
- **DataSubset**: Can be used to create a subset of `tmp/bodycomposition` (or an other df as defined as `input_df` argument) for later aggregation. The subset is defined by a reference (Center, Level, Centroid, Tag) corresponding to the vertebral levels created by **CalcVertebralLevel** and a specific vertebral level (`ALL` for all vertebrae, `L` for all lumbar vertebrae, or a string or list defining specific vertebrae). The subset is saved to the memory dictionary as `tmp/bodycomposition` or a specific name defined by the `output_df` argument.
- **DataAggregate**: Aggregates the data in `tmp/bodycomposition` (or an other df as defined as `input_df` argument). Groups are defined by a reference `ref` (Center, Level, Centroid, Tag) corresponding to the vertebral levels created by **CalcVertebralLevel**. If individual groups are required, individual groups can be defined using the `tag_mapping` dictionary that should map the values from `ref` to new, individual groups "tags". The method of aggregation is defined by `method`, currently mean, median and sum are supported. The aggregated data is saved to the memory dictionary as `tmp/bodycomposition` or a specific name defined by the `output_df` argument.
- **DataExport**: Saves data from a pandas dataframe (defined as argument `input`) to a csv file (defined as argument `file`, using placeholder `{caseid}`). If `add_metadata` is set to `True`, the metadata imported by **LoadMetadata** is concatenated. If `append` is set to `True`, the data is appended to the file which can be useful to generate summary files of multiple cases. If `add_header` is set to `True`, the header is added to the exported data. If `add_index` is set to `True`, the index is added to the exported data.