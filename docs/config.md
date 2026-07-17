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
- `profile_id`: Stable lowercase identifier included in scientific provenance
  and `analysis_id`. The default is `hofmann_2025_canonical_raw_v1`.
- `save_mask`: If `True`, the [tissue masks](labels.md) are saved. If `False`, the masks are just saved to the temporary pipeline memory.
- `save_compartment_mask`: Persists the untouched model compartment labels
  even when the speed preset disables general segmentation-label storage. The
  canonical default is `true` because later HU definitions must remain
  auditable and reproducible.

The canonical defaults use calibrated, unsmoothed HU values and do not delete
small SM, IMAT, VAT, or SAT islands. The configurable size thresholds remain
available for prespecified sensitivity analyses, but enabling one changes the
measurement identity and must be reported.

#### Filter: HU Denoise
- `method`: `none` (default), `median`, `adaptive_median`, or
  `curvature_anisotropic_diffusion`. This image is used only to assign a
  compatibility mask; canonical attenuation outputs always use the untouched
  prepared CT.
- `apply_to`: Unique subset of `imat`, `sm`, `vat`, and `sat`. This prevents a
  fat-specific literature method from silently changing muscle classification.
- `filter_outliers`: If `True`, outliers are clipped. If `False`, no clipping is performed.
- `filter_outliers_range`: Range for clipping outliers, e.g. metal implants or noisy outliers [-1024, 3071]
- `filter_median`: Legacy compatibility flag. It must be `true` exactly when
  `method: median` and `false` otherwise.
- `filter_median_kernel`: Kernel size in NumPy z-y-x order, e.g. `[1,3,3]` for in-plane filtering without smoothing across slices.
- `adaptive_median_min_kernel` and `adaptive_median_max_kernel`: Positive odd
  z-y-x kernels for the standard two-stage adaptive median algorithm.
- `anisotropic_diffusion`: Dimensionality, iterations, stable time step, and
  conductance for SimpleITK curvature anisotropic diffusion. Published profiles
  with incompletely reported parameters are explicitly labeled sensitivity
  implementations.

Every IMAT/SM/VAT/SAT block uses the same optional cleanup controls. Hole
filling is performed before small-component removal.

- `filter_size_unit`: `physical` means mm² in 2D or mm³ in 3D; `voxel` means
  pixels in 2D or voxels in 3D.
- `filter_size_connectivity`: 4 or 8 in 2D; 6, 18, or 26 in 3D.
- `fill_holes`: Enables bounded filling of enclosed background components.
- `fill_holes_version`, `fill_holes_2D`, `fill_holes_3D`,
  `fill_holes_unit`, and `fill_holes_connectivity`: Explicit dimension,
  threshold, unit, and connectivity. Background connected to the image border
  is never filled.

#### Filter: IMAT
- `filter_hu`: If `True`, IMAT is determined by using a HU range of voxels within the [muscle compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining IMAT, e.g. [-190, -30].
- `filter_size`: If `True`, the IMAT (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Minimum 2D area or pixel count, according to `filter_size_unit`.
- `filter_size_3D`: Minimum 3D volume or voxel count, according to `filter_size_unit`.

#### Filter: SM
- `filter_hu`: If `True`, SM is determined by using a HU range of voxels within the [muscle compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining SM tissue, e.g. [-29, 150].
- `filter_size`: If `True`, the SM (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Minimum 2D area or pixel count, according to `filter_size_unit`.
- `filter_size_3D`: Minimum 3D volume or voxel count, according to `filter_size_unit`.

#### Filter: VAT
- `filter_hu`: If `True`, VAT is determined by using a HU range of voxels within the [visceral compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining VAT, e.g. [-190, -30].
- `filter_size`: If `True`, the VAT (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Minimum 2D area or pixel count, according to `filter_size_unit`.
- `filter_size_3D`: Minimum 3D volume or voxel count, according to `filter_size_unit`.

#### Filter: SAT
- `filter_hu`: If `True`, SAT is determined by using a HU range of voxels within the [subcutaneous compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining SAT, e.g. [-190, -30].
- `filter_size`: If `True`, the SAT (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Minimum 2D area or pixel count, according to `filter_size_unit`.
- `filter_size_3D`: Minimum 3D volume or voxel count, according to `filter_size_unit`.

Named overrides for every reproducibly specified literature window and cleanup
branch are in [`config/tissue_profiles`](../config/tissue_profiles). The
evidence, default choice, exact operation order, and limits of the sensitivity
implementations are documented in
[tissue_definitions.md](tissue_definitions.md).

### Measurements

- `enabled`: Required for the canonical `BodyComposition` pipelines. Other
  registered pipelines may run without the measurement bundle.
- `totalsegmentator_version`: Pinned optional body/landmark adapter version. It
  must be `2.15.0` when either TotalSegmentator support backend is enabled.
- `full_coverage_tolerance`: Minimum physical interval coverage for a strict
  range or vertebral-territory bin. The default is `0.999`.
- `l3_slab_length_mm`: Fixed L3-centred comparator length. The canonical value
  is `200` mm.
- `tissue_definitions`: Named raw-compartment derivations. Each definition has
  `enabled`, a non-empty `source_labels` list, and either a two-number
  `hu_range` or `null` for the full anatomical compartment. Required canonical
  definitions cannot be disabled; additional validated anatomy or HU windows
  can be added without changing code. Names become output-column prefixes and
  must therefore use lowercase snake case.
- `body_surface.backend`: The default
  `tissue_segmentation_envelope_v1` derives body/trunk envelopes from the
  existing postprocessed tissue labels. Explicit alternatives are
  `totalsegmentator_body_task299_v1` and `deterministic_body_mask_v1`. A
  failure never causes backend substitution.
- `body_surface.save_mask`: Persists the encoded body/trunk QC mask. Trunk is
  label 1; full body is the union of labels 1 and 2.
- `body_surface.minimum_component_area_mm2`, `closing_radius_mm`, and
  `smoothing_sigma_mm`: Physical parameters of the tissue-derived envelope.
- `body_surface.threshold_hu` and `min_component_volume_mm3`: Parameters used
  only by the deterministic alternative.
- `landmarks.enabled`: Runs the anatomical mid-waist landmark stage. Disabling
  it leaves mid-waist null; it does not substitute the minimum waist.
- `landmarks.backend`: Pinned task-297 rib/hip adapter.
- `landmarks.minimum_voxels`: Minimum retained landmark component.
- `landmarks.maximum_side_disagreement_mm`: Maximum bilateral position
  disagreement before a landmark is uncertain.
- `vertebral_extent.minimum_component_voxels`: Minimum largest vertebral-body
  component.
- `vertebral_extent.maximum_removed_fraction`: Maximum fraction that may be
  removed as smaller isolated components while retaining a valid extent.
- `qc.circumference_jump.maximum_gap_mm`: Largest adjacent physical gap on
  which longitudinal jump QC is evaluated.
- `qc.circumference_jump.minimum_absolute_jump_cm` and
  `minimum_relative_jump`: Both thresholds must be exceeded to flag an
  implausible adjacent-slice circumference discontinuity.
- `review.enabled`: Writes the identifier-free measurement review PNG.
- `export.parquet`: Must remain `true` for the canonical bundle.
- `export.csv`: Reserved interoperability setting; canonical measurement CSV
  duplication is disabled and must remain `false`.

Definitions, units, physical aggregation, territories, and missing reasons are
documented in [measurements.md](measurements.md).

### Reporting

Reporting is a derived, optional final stage of the canonical pipelines. It
does not change `analysis_id` and never replaces the Parquet, NIfTI, or JSON
outputs.

- `enabled`: Default `false`. When true, retain one individual PDF and report
  manifest for every selected case.
- `layout`: `spine_overview_v1` or `spine_profile_v2`. Both use fixed ISO A4
  landscape pages. The second layout adds the native-mm stacked tissue-area
  profile.
- `individual_pdf`: Must remain `true` in released layouts.
- `combined_pdf`: Collate batch reports when all cases share one explicit
  workspace. Explicit post-hoc exports use an ordered export manifest.
- `page_size`: Fixed `A4_landscape`.
- `spine_view`: Fixed `sagittal_thick_slab_v1`. This is a projection for
  research/QC, not a diagnostic MPR.
- `measure_aggregation`: Fixed `territory_mean`. CSA and circumference are
  reconstructed from the three measurement stage bins with
  `bin_integration_length_mm`; HU uses the corresponding tissue-volume
  weights.
- `measurement_columns`: Ordered selection of one to six released measures.
  Defaults to SM CSA, SM HU, IMAT CSA, total VAT CSA, SAT CSA, and trunk
  circumference.
- `vertebral_range`: Currently fixed `detected`; T13, L6, and sacrum remain
  native labels.
- `include_qc_flags`: Must remain `true`.
- `manual_review_summary`: Fixed `auto`; the summary is absent when no case is
  canonically flagged.
- `numeric_precision`: Display precision from zero to three decimals. Source
  values remain unrounded in the report manifest.
- `locale`: Currently `en`; `missing_value_symbol` must be ASCII.
- `ct_window`: Frozen lower/upper HU display window.
- `overlay_opacity`: Vertebral-body overlay opacity from zero to one.
- `sagittal_slab_margin_mm`: Physical lateral margin around valid vertebral
  centroids.
- `projection_spacing_mm`: Canonical physical projection resolution, at most
  3 mm.

See [reporting.md](reporting.md) for layouts, output paths, privacy, and the
explicit post-hoc manifest format.

### Crop
Legacy/noncanonical pipelines may segment vertebrae and crop to a configured
region of interest. The canonical `BodyCompositionFast` pipeline intentionally
does not crop: the native-mm longitudinal table and vertebral territories
require the complete prepared CT. Crop definitions remain available to actions that
explicitly use [CreateBoundingBox](../BodyComposition/actions/crop.py). For
each region of interest, the following parameters can be defined:

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
