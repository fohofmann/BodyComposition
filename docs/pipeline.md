# Pipeline

The BodyComposition package provides several registered pipelines.
Pipelines are registered in [pipeline_registry.py](../BodyComposition/pipeline_registry.py), with their ordered action definitions in the [pipelines directory](../BodyComposition/pipelines/).

Each pipeline is a sequence of stages that are executed in order as defined in the respective pipeline file.
Each stage is defined as `action`, which are classes that are derived from the `PipelineAction` class.

At the start of the pipeline, each action class is initialized.
The pipeline selects all cases, where all required inputs (defined by the individual actions) are available.
Then, for each case, all actions are executed in order.
Outputs of actions are available to subsequent actions through a shared memory dictionary. Each action declares its inputs and outputs. Missing inputs and outputs fail at the responsible action rather than surfacing later as an unrelated key error.

Action outputs, persisted completion markers, and reset targets are separate contracts. An output may exist only in memory when saving is disabled; only files that an action actually writes are used to decide whether a batch case is complete. Reset targets also include shared append exports so a requested clean run does not retain earlier rows. A pipeline with no persisted completion markers is processed normally instead of being skipped vacuously.

A case failure is logged with its original traceback. Batch execution continues with the remaining cases and then raises `PipelineExecutionError`, so CLI processes return a non-zero exit status instead of reporting success for a partial run.

Internal nnU-Net inference normally uses cuDNN on CUDA devices. If cuDNN reports its specific `FIND` or `GET` engine-selection failure, the action logs a warning, disables cuDNN for the remainder of that process, and retries once through PyTorch's CUDA fallback. Other runtime errors are not retried.

Images use explicit geometry conventions: SimpleITK size, spacing, origin, and indices are in x-y-z order; arrays returned by SimpleITK are in z-y-x order. Images and masks must have the same size and physical domain before measurements are combined.

Unless `orientation.enabled` is false, `PipelineBuilder` prepends the same
orientation action to every registered pipeline. This makes orientation
assessment part of both the Python API and CLI execution path rather than a
DICOM conversion option.

## Pipelines
While pipelines can be customized by the user, the following methods are readily implemented:
- **[BodyCompositionFast](../BodyComposition/pipelines/bodycomposition.py) (default)**: Uses the internal ResEncM vertebral and tissue models, crops in the cranio-caudal axis to L2-L4, and exports per-slice and mean-L3 measurements.
- **[BodyComposition](../BodyComposition/pipelines/bodycomposition.py)**: Uses the internal ResEncL vertebral and tissue models and exports full-range per-slice measurements plus the available outer tissue contour.
- **[SarcopeniaTotalSegmentatorFast](../BodyComposition/pipelines/totalsegmentator.py)**: Segments the spine using [TotalSegmentator](models.md), crops the image in cranio-caudal axis to L2-L4, segments tissue using [TotalSegmentator](models.md), and exports slice-wise measures for each case as well as mean L3 measures for all cases.
- **[SarcopeniaTotalSegmentator](../BodyComposition/pipelines/totalsegmentator.py)**: Segments spine, vertebral bodies, body trunk, tissue and psoas muscle using [TotalSegmentator](models.md), reduces the labels to the vertebral bodies, and exports slice-wise measures for each case.
- **[SarcopeniaStanfordFast](../BodyComposition/pipelines/stanford.py)**: Uses the Comp2Comp spine and tissue models in a cropped L2-L4 workflow.
- **[BodyAndOrganAnalysis](../BodyComposition/pipelines/boa.py)**: Combines TotalSegmentator spine localization with the BOA tissue model and exports per-slice measurements.

## Actions

### Orientation

- **AssessOrientation**: Validates the original pixels and physical geometry,
  converts the SimpleITK z-y-x array explicitly to CTDeepRot y-x-z convention,
  obtains equivariance votes across all 24 proper rotations, and compares the
  anatomy result with the metadata interpretation. Matching inputs are not
  canonicalized. A mismatch is repaired only when all configured confidence,
  coverage, obliquity, artefact, and class-specific safety gates pass. The
  SI-preserving 180-degree in-plane class is advisory-only by default because
  it produced a confident false-positive candidate in the public discrepancy
  audit. Repair uses only
  `SimpleITK.PermuteAxes` and `SimpleITK.Flip`, with no interpolation, and must
  pass an exact inverse-array and physical-point round trip. LPS is only the
  internal comparison frame; the final derivative is losslessly re-expressed
  in the input's original discrete orientation convention. Readable uncertain
  cases continue unchanged with QC `review`; a changed case always requires
  manual review. The action writes a structured JSON report and an
  identifier-free scout image. The scout's top row shows the metadata
  interpretation; the lower row shows the actual prepared image, or a clearly
  labeled, unapplied model candidate when a mismatch is gated to review. The
  action preserves the original container as `tmp/original_index`.

### Segmentation
- **SegmIntVertebrae**: Segments the vertebral body using nnU-Net with models described [here](models.md). Requires `image` being a [NIfTI data container](../BodyComposition/utils/nifti.py) and `model` being a string defining either the `ResEncM` or `ResEncL` model. Returns (and optionally saves) the segmentation as a [NIfTI data container](../BodyComposition/utils/nifti.py).
- **SegmStanfordSpine**: Segments the vertebral body using the [Comp2Comp Spine Segmentation model](models.md). Requires `image` being a [NIfTI data container](../BodyComposition/utils/nifti.py). Returns (and optionally saves) the segmentation as a [NIfTI data container](../BodyComposition/utils/nifti.py).
- **SegmTotalSegmentator**: Segments any structures using the [TotalSegmentator](models.md). Requires `image` being a [NIfTI data container](../BodyComposition/utils/nifti.py), `task` being a string defining the [task to be performed](../BodyComposition/actions/segm_totalsegmentator.py) (e.g., spine, bodytrunk, tissue, vertebralbodies, iliospoas), and `fast` being a boolean defining whether TotalSegmentator's *fast* function should be used. Before running this action, TotalSegmentator must be initialized using the **SegmTotalSegmentatorConfig** action. Returns (and optionally saves) the segmentation as a [NIfTI data container](../BodyComposition/utils/nifti.py).

### Masks Spine
- **MasksTotalSegmentatorSpine**: Maps the TotalSegmentator labels to the [standard labels](labels.md) used in the pipeline. If `reduce_to_vb` is set to `True`, the labels are reduced to the vertebral bodies using TotalSegmentator's `vertebral_body` segmentation. Returns the remapped masks as a [NIfTI data container](../BodyComposition/utils/nifti.py).
- **MasksStanfordSpine**: Maps the Comp2Comp Spine labels to the [standard labels](labels.md) used in the pipeline and returns the remapped mask as a [NIfTI data container](../BodyComposition/utils/nifti.py).

### Masks Tissue
*During the processing of tissue masks, filters based on Hounsfield units and 2D or 3D size properties are applied to subsegment `labels` and generate `masks` as explained [here](labels.md) and defined in the pipeline's [configuration](config.md).*

- **MasksTotalSegmentatorTissue**: Maps the TotalSegmentator labels to the [standard labels](labels.md) used in the pipeline. If `iliopsoas` is set to `True`, the a separate label of the iliopsoas muscle is returned using TotalSegmentator's `iliopsoas` segmentation. If `bodytrunk` is set to `True`, the labels are reduced to the body trunk using TotalSegmentator's `bodytrunk` segmentation. Returns the remapped masks as a [NIfTI data container](../BodyComposition/utils/nifti.py).

### Bounding Boxes
- **CreateBoundingBox**: Creates bounding boxes around specific labels. The segmentation `label` must be provided as a [NIfTI data container](../BodyComposition/utils/nifti.py). The bounding box is defined in the pipeline's [configuration](config.md), and the specific `task` must be defined as argument. The function then creates a bounding box around the specific label and saves it (`bbox`) to the memory dictionary.
- **ApplyBoundingBox**: Applies a bounding box to a image, label or mask. The segmentation `label` must be provided as [NIfTI data containers](../BodyComposition/utils/nifti.py), and the bounding box `bbox` must be available within the memory dictionary. If the NIfTI data container should not be changed, but saved separately, define its name using the `output` argument. The function then applies the bounding box to the input NIfTI. If the changed NIfTI data container is used later on, only values within the bounding box are returned, changed or saved.

### Postprocessing
- **CalcVertebralLevel**: Canonicalizes and validates a vertebral mask, then produces a DataFrame with one row per prepared CT slice and the dominating level, center, centroid, tag, and status. An empty vertebral mask returns the same schema with `VertebraStatus=empty_mask` instead of omitting the output.
- **CalcMeasures**: Canonicalizes and validates the tissue mask and any optional CT/contour mask. It returns named per-slice voxel counts and CSA in cm² for every native label. Optional HU columns include an explicit empty-tissue status. Optional contour perimeter and area are calculated after transforming contour points through the physical affine.

### Data Handling
- **LoadMetadata**: Trys to load metadata. The path is given as an argument, with the placeholder `{caseid}` being replaced by the current cases id. Can be both, a *csv (containing DICOM metadata) or a *dcm file. The metadata is saved to the memory dictionary as `tmp/metadata`.
- **DataCombine**: Combines tissue measurements and vertebral levels only after size, spacing, origin, direction, and row count agree. It returns `tmp/bodycomposition` in ascending prepared-slice order.
- **L3MeanCSA**: Produces a one-row mean-L3 CSA table. If L3 is absent, it still produces the declared schema with `status=not_available` and `reason=missing_L3`.
- **DataSubset**: Can be used to create a subset of `tmp/bodycomposition` (or an other df as defined as `input_df` argument) for later aggregation. The subset is defined by a reference (Center, Level, Centroid, Tag) corresponding to the vertebral levels created by **CalcVertebralLevel** and a specific vertebral level (`ALL` for all vertebrae, `L` for all lumbar vertebrae, or a string or list defining specific vertebrae). The subset is saved to the memory dictionary as `tmp/bodycomposition` or a specific name defined by the `output_df` argument.
- **DataAggregate**: Aggregates the data in `tmp/bodycomposition` (or an other df as defined as `input_df` argument). Groups are defined by a reference `ref` (Center, Level, Centroid, Tag) corresponding to the vertebral levels created by **CalcVertebralLevel**. If individual groups are required, individual groups can be defined using the `tag_mapping` dictionary that should map the values from `ref` to new, individual groups "tags". The method of aggregation is defined by `method`, currently mean, median and sum are supported. The aggregated data is saved to the memory dictionary as `tmp/bodycomposition` or a specific name defined by the `output_df` argument.
- **DataExport**: Saves the DataFrame selected by `input_df` to the CSV path selected by `file`, including the `{caseid}` placeholder. `add_metadata=True` prepends loaded metadata; `append=True` appends rows and writes a header only when the target does not yet exist. CSV is the current interoperability output for the baseline pipeline.
