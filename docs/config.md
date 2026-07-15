# Configuration File

Frequently changing parameters can be adapted using configuration files or dictionaries. The pipeline is loading
- `config/config.yaml`, and
- the `config/*.yaml` file with the same name as the current pipeline (e.g., `BodyCompositionFast.yaml`), and
- the argument `config`, which can either be a path to a file (e.g. `bodycomposition --config path/to/new/config.yaml`) or a dictionary (e.g. `bodycomposition --config '{"segmentation": {"save_label": True}"}'`).

Parameters are replaced in the following order: `config/config.yaml` < `config/*.yaml` < `--config` argument. This means that parameters in the `--config`/`-c` argument will overwrite parameters defined in the other files.

Configuration is validated before a pipeline is built. Boolean fields must be YAML booleans (`true`/`false`), numeric fields must have the documented numeric type, and optional paths use YAML `null`. Typographical strings are rejected at startup rather than interpreted as truthy values later in processing.

## Paramenters

### Paths
- `workspace`: Path to the workspace where the data (e.g., the directories `labels`, `masks`, `exports`) is stored. The path can include placeholders for `method` (= name of the current pipeline), `filter` (= regex filter used for input filtering), and `timestamp` (= int time). If `None`, the parent directory of the input data is used.
- `logs`: Path to the log file. The path can include placeholders for `method` (= name of the current pipeline), `filter` (= regex filter used for input filtering), and `timestamp` (= int time).
- `weights`: Path to the weights of the nnU-Net models. For `totalsegmentator`, the path should be the directory that includes all different models ("tasks" in TotalSegementator), as TotalSegmentator identifies the subdirectory by itself.
- `totalsegmentator_config`: Path to the directory that includes the TotalSegmentator configuration file, including your license number.

### Logging
`logging_level`: Logging level for the logging file and console. The levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.

### Run
- `reset`: If `True`, all outputs are removed at initialization.
- `skip`: If `True`, segmentations and mask generation are skipped if already present.
- `timeout`: Timeout in seconds for **each case** in the pipeline. Can be used to prevent the pipeline from getting stuck on a single case.

### Orientation

Orientation assessment runs as the first API/pipeline stage, before
segmentation, cropping, canonicalization, or resampling.

- `enabled`: Enables the shared stage for every registered pipeline. Default:
  `true`.
- `confidence.min_equivariant_votes`: Minimum agreeing CTDeepRot votes out of
  24 for automatic proper-rotation repair. Default: `23`.
- `confidence.min_body_extent_mm`: Minimum inferred cranio-caudal body extent
  for repair. Default: `250` mm.
- `confidence.max_obliquity_deg`: Maximum residual header obliquity for a
  discrete repair. Similar metadata below this threshold are retained rather
  than flattened. Default: `15` degrees.
- `confidence.body_threshold_hu` and
  `confidence.min_body_pixels_per_slice`: Rules used only for the coverage
  safety gate.
- `confidence.allow_axial_180_repair`: Allows automatic correction of the
  SI-preserving 180-degree in-plane class. It defaults to `false` because the
  public discrepancy audit found a confident false-positive candidate for
  this anatomically ambiguous rotation. The default keeps the input unchanged,
  shows the proposed correction, and requires manual review.
- `artifact_rules.metal_threshold_hu` and `max_metal_fraction`: High-density
  artefact gate. `review_on_possible_truncation` optionally prevents repair
  when the estimated transverse body touches the field-of-view boundary.
- `model.checkpoint_path`: Local cache path. Environment variable
  `BODYCOMPOSITION_CTDEEPROT_CHECKPOINT` overrides it.
- The source commit, download URL, byte size, and SHA-256 are locked in the
  adapter rather than configurable. Run
  `bodycomposition_download_models --model CTDeepRot-2D` once to synchronize
  the pinned asset. Inference never downloads models.
- `model.device`: `cpu`, `cuda`, or `auto`; the default `cpu` works without
  GPU-specific setup. `batch_size` controls how many of the 24 views are
  inferred together.
- `report.enabled`: Writes the per-case JSON and PNG QC artifacts. Changed
  cases additionally persist the exact downstream derivative.
- `header_uncertain_reasons` and `force_review_reasons`: Normally empty lists
  populated by trusted ingestion/runtime provenance. Any entry prevents
  automatic repair and forces review.

Every reoriented case has `orientation_changed=true`, QC `review`, and an
`ORIENTATION_CHANGED` flag. CTDeepRot does not detect left-right reflections;
reflection-like concerns must be supplied as a forced-review reason.

### Segmentation
- `save_label`: If `True`, the segmentation labels are saved. If `False`, the labels are just saved to the temporary pipeline memory.

### Vertebrae

- `backend`: Exactly one backend. The default is
  `spineps_veridah_ct_v1`; retained alternatives are
  `vertebral_bodies_resenc_l` and `vertebral_bodies_resenc_m`. A failure never
  causes an automatic fallback.
- `spineps.device`: `auto`, `cpu`, or `cuda`.
- `spineps.save_native_outputs`: Retains the native semantic mask, labeled
  vertebra mask, centroid/POI JSON, snapshot, crop mask, and other named safe
  upstream outputs. The canonical vertebral-body mask is always persisted.
- `spineps.review_enabled`: Writes the sagittal/coronal vertebral QC image.
- `save_mask`: If `True`, the vertebrae masks are saved. If `False`, the masks are just saved to the temporary pipeline memory.
- `min_voxels_per_vertebra`: Minimum number of voxels per vertebra. If (the dominating) vertebra in a slice has less voxels, the label is ignored. This reduces the number of artifacts.
- `fill_undefined_levels`: If `True`, slices between vertebra, in which no dominating vertebra was detected, are filled considering the neighbouring vertebrae in stepwise growing windows along the cranio-caudal axis.
- `correct_monotonicity`: If `True`, the labels are corrected to be monotonically increasing along the cranio-caudal axis. To do so, the labels are corrected by using stepwise growing windows along the cranio-caudal axis.
- `correct_monotonicity_max_windowsize`: Maximum window size for the monotonicity correction.
- `center_of_mass`: If `True`, the center of mass of each vertebra is calculated and returned in the output.
- `deprioritize_labels`: We determine the *dominating vertebra* in each slice, which is the vertebra with the highest number of voxels in the slice. Due to its size, the cranial parts of the sacrum can dominate the lower lumbar vertebrae. Therefore, this option allows to deprioritize the sacrum.
- `deprioritize_anatomical`: Backend-independent anatomical names used when a
  common `VertebralResult` is available. The default is `[SACRUM]`, avoiding
  the collision between legacy label 19 and SPINEPS label 19 (T12).

### Tissue
- `save_mask`: If `True`, the [tissue masks](labels.md) are saved. If `False`, the masks are just saved to the temporary pipeline memory.

#### Filter: HU Denoise
- `filter_outliers`: If `True`, outliers are clipped. If `False`, no clipping is performed.
- `filter_outliers_range`: Range for clipping outliers, e.g. metal implants or noisy outliers [-1024, 3071]
- `filter_median`: If `True`, a median filter is applied. If `False`, no median filter is applied.
- `filter_median_kernel`: Kernel size in NumPy z-y-x order, e.g. `[1,3,3]` for in-plane filtering without smoothing across slices.

#### Filter: IMAT
- `filter_hu`: If `True`, IMAT is determined by using a HU range of voxels within the [muscle compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining IMAT, e.g. [-190, -30].
- `filter_size`: If `True`, the IMAT (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Size for the 2D filter, e.g. 10 mm^2 (calculated based on voxel spacing). Only areas larger than this are considered as IMAT.
- `filter_size_3D`: Size for the 3D filter, e.g. 80 mm^3 (calculated based on voxel spacing). Only volumes larger than this are considered as IMAT.

#### Filter: SM
- `filter_hu`: If `True`, SM is determined by using a HU range of voxels within the [muscle compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining SM tissue, e.g. [-29, 150].
- `filter_size`: If `True`, the SM (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Size for the 2D filter, e.g. 10 mm^2 (calculated based on voxel spacing). Only areas larger than this are considered as SM.
- `filter_size_3D`: Size for the 3D filter, e.g. 80 mm^3 (calculated based on voxel spacing). Only volumes larger than this are considered as SM.

#### Filter: VAT
- `filter_hu`: If `True`, VAT is determined by using a HU range of voxels within the [visceral compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining VAT, e.g. [-190, -30].
- `filter_size`: If `True`, the VAT (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Size for the 2D filter, e.g. 10 mm^2 (calculated based on voxel spacing). Only areas larger than this are considered as VAT.
- `filter_size_3D`: Size for the 3D filter, e.g. 80 mm^3 (calculated based on voxel spacing). Only volumes larger than this are considered as VAT.

#### Filter: SAT
- `filter_hu`: If `True`, SAT is determined by using a HU range of voxels within the [subcutaneous compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining SAT, e.g. [-190, -30].
- `filter_size`: If `True`, the SAT (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Size for the 2D filter, e.g. 10 mm^2 (calculated based on voxel spacing). Only areas larger than this are considered as SAT.
- `filter_size_3D`: Size for the 3D filter, e.g. 80 mm^3 (calculated based on voxel spacing). Only volumes larger than this are considered as SAT.

### Crop
For the fast versions of the pipeline, we segment the vertebrae, and then crop the image to the region of interest for all further analyses. The region of interest can be defined in this section of the configuration file, and is then used by the [CreateBoundingBox](../BodyComposition/actions/crop.py) class. For each region of interest, the following parameters can be defined:

- `roi`: List of labels that define the region of interest.
- `roi_anatomical`: Preferred backend-independent names. These are resolved
  through the selected backend's native label schema; native labels are never
  rewritten.
- `axes`: Six booleans defining whether each crop boundary is active. The order is [inferior, superior, posterior, anterior, left, right] for the RAS-oriented prepared inputs, matching array z-y-x boundary pairs. The orientation stage either retains a trusted input or supplies the reviewed lossless derivative before this action runs.
- `margin`: List of integers that define the margin in mm along each axis. The margin is transformed to pixels later in the pipeline.

```yaml
  L234CranioCaudal:
    roi: [14,15,16] # labels
    roi_anatomical: [L2, L3, L4]
    axes: [True, True, False, False, False, False]
    margin: [0, 0, 0, 0, 0, 0] # inferior, superior, posterior, anterior, left, right
```
