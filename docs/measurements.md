# Canonical physical measurements

The canonical pipeline writes one outcome-blind measurement bundle per case.
Schema `3.5.0` preserves every acquired slice, compact native vertebral
summaries, established case summaries, and a comparable fixed-millimetre
longitudinal signature. Its second, identity-linked component stores four
fixed native-compartment HU-distribution channels, with explicit missingness
when a selected backend has no homologous compartment. No
longitudinal curve is stretched and no unscanned anatomy is imputed.

## Authoritative files

- `tables/slices.parquet`: one row for every prepared CT slice;
- `tables/vertebrae.parquet`: three physical bins for every detected native
  vertebral territory, including sacrum;
- `tables/summaries.parquet`: one deterministic case-level row;
- `tables/signature.parquet`: 100 fixed 20-mm bins in a versioned
  vertebral-reference coordinate;
- `tables/hu_distributions.parquet`: fixed 5-HU voxel histograms plus exact
  mean, standard deviation, median, and quartiles for SM, SAT, aVAT, and tVAT
  over the analyzed volume when the native source is available; and
- `qc/qc.json`: provenance, validity, QC, and review status.

`slices.parquet` remains the lossless longitudinal measurement source.
`signature.parquet` is a compact comparison view derived from it. Every table
row carries `schema_version`, `case_id`, `run_id`, and `analysis_id`; Parquet
metadata repeats the identity and table name. The reader rejects missing
tables/columns, incompatible field types, mixed identities, invalid fixed bins,
and unsupported schema versions.

By default, every Parquet table also has a same-name `.csv` mirror with
identical columns and rows. These files make ordinary inspection and analysis
easier but do not replace the typed, identity-validated Parquet contract. Add
`--no-csv` to `analyze` or `batch` when duplicate representations are not
wanted. `bodycomposition results export-csv` can create the mirrors later from
one validated case bundle into a separate directory.

## Geometry and units

- SimpleITK metadata, sizes, and indices are `x-y-z`.
- Arrays returned by SimpleITK are `z-y-x`.
- Physical coordinates use DICOM LPS millimetres.
- Slice rows are ordered inferior to superior by
  `position_superior_mm`, independently of storage order.
- Areas are cm², volumes are cm³, lengths are millimetres unless a name ends
  in `_cm`, and attenuation is HU.
- `slice_spacing_normal_mm` is the distance between acquired slice centres
  along the image slice axis; `slice_spacing_superior_mm` is its projection
  onto the patient superior axis. These are the sampling distances used for
  physical integration, not acquisition slice thickness. When DICOM
  `SliceThickness (0018,0050)` is available and consistent, it is preserved
  separately in input provenance and report metadata.

For one binary label on an acquired plane:

`area_cm2 = voxel_count * in_plane_pixel_area_mm2 / 100`

Range and bin aggregation uses exact overlap between each acquired slice slab
and the requested physical superior-axis interval. Mean CSA is overlap-
weighted where a boundary cuts a slice. Mean HU is pooled by contributing mask
voxels and overlap fraction; it is not an unweighted average of slice means.

## Tissue measurements

The raw model compartments and default HU definitions are documented in
[tissue_definitions.md](tissue_definitions.md). Native outputs use a
`<name>_compartment` prefix; HU-filtered outputs use a `<name>_tissue_hu_*`
prefix. Each prefix in `slices.parquet` has:

- `<name>_voxel_count`;
- `<name>_area_cm2`, `<name>_area_valid`, and `<name>_area_reason`; and
- `<name>_mean_hu`, `<name>_hu_valid`, and `<name>_hu_reason`.

Raw `sm_compartment_mean_hu` measures the complete muscle compartment. The filtered
`skeletal_muscle_tissue_hu_m29_150_*` fields are the conventional
-29-to-150 HU subset. The default does not calculate IMAT, LAMA, or NAMA.

A native compartment that is absent on a valid scanned slice has CSA zero and
mean HU null with reason `empty_compartment`; an absent filtered tissue uses
`empty_tissue`. An invalid measurement is null with its reason. Unscanned
signature bins are null with coverage zero. These states must not be merged
during analysis.

Every slice also records `analysis_scope` and
`body_composition_analysis_available`. In the standard `full_ct` scope, every acquired
slice is available. In the explicit `l3_vertebral_level` scope, rows outside
the L3 target remain in the table for geometry and anatomical registration,
but tissue measurements are invalid with reason `outside_analysis_region`;
zero label counts on those rows are not biological zero measurements.

## `hu_distributions.parquet`

This table is the authoritative source for the PDF native-compartment HU
distributions.
It contains 68 ordered bins from -190 to 150 HU for each of `SM`, `SAT`,
`aVAT`, and `tVAT`; bins are 5 HU wide, left-closed/right-open, with the final
bin closed at 150 HU. Compartment membership comes from the unfiltered
model-native labels, not the configured HU-restricted tissue definitions.
Histogram values and exact linear-method quartiles come from the unchanged
prepared CT voxels.

The four rows remain present for a backend that lacks a homologous native
compartment. Such rows contain no source label IDs, have
`distribution_valid=false`, and use reason `missing_compartment`. They are not
silently populated from a broader anatomical region. In particular, BOA's
ABDOMINAL_CAVITY and THORACIC_CAVITY are not native aVAT/tVAT compartments;
BOA can still produce separately named HU-filtered tissue measurements from
those regions.

The scope is always recorded as `analyzed_volume`, with the analyzed physical
inferior and superior voxel-cell bounds and the number of contributing slices.
This remains correct for oblique images because the bounds use all eight
physical image corners rather than only slice-centre positions. Counts
below or above the fixed display range and non-finite values are recorded
separately, never silently discarded. `voxel_fraction` uses all selected
compartment voxels as its denominator, so clipped/non-finite counts remain
accountable.

For the standard scope the analyzed volume is the complete prepared CT. For
the low-resource scope it is the retained L3 target, not the larger model
context crop.

## Body/trunk support

The default `tissue_segmentation_envelope_v1` builds a deterministic
measurement envelope from the raw model compartments. It does not rewrite the
tissue mask. It retains all source compartment voxels, uses physical closing and
smoothing only for the envelope, and identifies the primary trunk component
per slice.

`trunk_area_cm2` and `trunk_circumference_cm` include explicit contour,
fragmentation, field-of-view, and longitudinal-gap QC. Circumference uses only
the closed external contour; internal holes are not added. A contour touching
the image boundary remains numerically observable in `slices.parquet`, with
`valid=false` and reason `touching_image_boundary`, and is excluded from strict
means because the value may be underestimated. Full-body and primary-trunk
lateral FOV contact are recorded separately per slice and at case level. A
closed, nonfragmented sacral contour may additionally provide a numeric pelvic
observation under the explicit cropped-value policy below. A fragmented trunk
is never promoted to that observed pelvic value.

## Native vertebral territories

Only canonical vertebral-body/corpus labels define downstream anatomy. Full
vertebrae and posterior elements are never used.

For every detected cervical, thoracic, lumbar, or sacral level:

1. retain the largest cleaned vertebral-body component;
2. calculate its physical centroid and extent;
3. bound each territory at the midpoint of adjacent detected centroids; and
4. divide the territory into three equal physical superior-axis bins.

Intervertebral slices are therefore assigned to the closest adjacent detected
level. Slices above the first supported territory or below the last remain
unassigned. Internal label gaps remain continuous but are flagged. T13 and L6
remain native levels and are never compressed. Sacrum is handled like a
vertebral territory and receives three rows.

## `vertebrae.parquet`

The table contains exactly three ordered rows per detected supported level.
`territory_bin=1` is superior and `territory_bin=3` is inferior. It includes:

- native label, anatomical level, variants, and sequence-gap flags;
- vertebral-body centroid and extent;
- territory/bin physical bounds, completeness, coverage, and QC;
- mean CSA and pooled HU for every available compartment and filtered tissue;
- trunk mean CSA and circumference.

Volume is not duplicated. For a fully covered valid bin:

`volume_cm3 = mean_csa_cm2 * bin_integration_length_mm / 10`

For a partially observed bin, use the metric-specific coverage:

`observed_volume_cm3 = mean_csa_cm2 * bin_integration_length_mm * mean_csa_coverage_fraction / 10`

The integration length is the target through-plane length and preserves oblique
geometry. The coverage term prevents unobserved anatomy from being
extrapolated. Whole-territory observed means and volumes can be reconstructed
from the three rows.

An incomplete edge or truncated vertebral territory remains marked
`territory_complete=false` with its specific reason, but its observed bins are
still summarized when physical bounds and acquired slices are available.
`bin_valid` describes whether the observed bin can be measured; it does not
claim that the full anatomical territory was acquired. Downstream comparisons
must retain territory completeness and bin coverage.

For trunk circumference, a finite closed and nonfragmented contour that only
fails strict validity because it touches the image boundary is retained as an
observed, potentially underestimated value. The contributing bin is marked
`trunk_mean_circumference_cm_value_is_fov_cropped=true`; the PDF displays `!`
beside any whole-territory mean that includes such a bin. Contours with an
internal gap, fragmentation, or no finite circumference remain invalid rather
than being rescued by this rule.

## `summaries.parquet`

The single case row contains deterministic views:

- a complete 200-mm L3-centred slab when available;
- a whole-L3-territory view;
- minimum observed valid trunk circumference across the available T10-to-L5
  search;
- optional anatomical mid-waist circumference when task 297 rib and iliac
  landmarks are explicitly enabled and valid;
- maximum observed valid circumference in the bounded sacral territory; and
- explicitly named waist-to-pelvic ratios.

The standard anthropometric summary is the T10-to-L5 minimum, sacral maximum,
and their ratio. It does not run task 297. The separate `ct_midwaist_*` fields
remain unavailable unless the optional landmark extension is enabled.

Numerical validity and scientific eligibility are separate. The minimum waist
uses the observed bounded part of the T10-to-L5 range when either boundary
level lies outside the scan; its value remains numeric but is marked
`eligible=false`, reason `missing_anchor`, and retains the actual physical
search bounds. It is eligible for an unqualified comparison only when both
anchors, the anatomical sequence, acquired interval, and valid-contour
coverage are complete and the extremum is not at a search boundary.
Waist-to-pelvic ratios are calculated when both component values are
numerically valid and carry their own eligibility field. A scan without any
bounded T10-to-L5 territory remains null.

The pelvic maximum requires an observed sacral territory and at least one
strictly valid contour or one closed, nonfragmented boundary-cropped contour.
If the selected maximum comes from a cropped contour, its value and derived
ratios remain numeric but are marked `value_is_fov_cropped=true`,
`eligible=false`, and `reason=touching_image_boundary`. The separate
`search_touches_fov` flag records any contact in the sacral search even when
the selected maximum itself is strict. These observations may underestimate
the true circumference and are not asserted to be mathematical lower bounds.
The sacral maximum is called pelvic, not hip, until that clinical
interpretation is separately validated.

## Default signature

The default comparison signature has two components linked by
`schema_version`, `case_id`, `run_id`, and `analysis_id`:

1. `signature.parquet` contains compact longitudinal anatomy with explicitly
   named native-compartment and HU-filtered-tissue channels; and
2. `hu_distributions.parquet` contains the global native-compartment attenuation
   shapes described above.

This preserves both anatomical position and tissue-quality shape without
copying 272 global histogram rows into each of 100 longitudinal bins.
`load_signature()` returns both validated components;
`signature_measurements()` remains the convenience reader for the
longitudinal component only.

### `signature.parquet`

The default signature is a wide table with 100 rows. Each row spans exactly
20 mm; `reference_center_mm` runs from -1000 to +980 mm in ascending physical
superior direction. Bin 50 is centred on zero. The coordinate is a
translation-only vertebral reference:

- zero is the L3 vertebral-body centroid when reliable L3 is visible;
- when L3 is absent, a robust linear fit of the visible native vertebral
  centroids estimates the latent L3 position;
- a single visible level uses a versioned 30-mm nominal pitch and is explicitly
  low confidence;
- multi-anchor confidence is reduced when the nearest observed level is more
  than three native steps from L3, and becomes low beyond six steps, so long
  extrapolations are never presented as high-confidence alignment;
- the fused sacral label remains a dominant-level annotation but is not used
  alone to extrapolate an L3 origin;
- no CT curve, vertebral interval, or tissue measurement is scaled; and
- no values are filled into unscanned bins.

The robust fit uses patient-specific measured vertebral pitch only to estimate
the missing origin. This accommodates patient size without normalizing away
the physical length of the tissue profile. T13 and L6 are inserted into the
native reference sequence when detected. Alignment method, confidence,
anchors, fitted pitch, residual, variant sequence, and review requirement are
stored in every row and in QC provenance.

Each bin includes:

- reference and original physical bounds;
- coverage and contributing-slice count;
- dominant vertebral territory and overlap fraction;
- alignment and bin validity;
- trunk mean CSA and circumference;
- mean CSA and pooled mean HU for `sm_compartment`, `bone_compartment`,
  `heart_compartment`, and `lung_compartment`;
- mean CSA and pooled mean HU for
  `skeletal_muscle_tissue_hu_m29_150`, `sat_tissue_hu_m190_m30`,
  `avat_tissue_hu_m190_m30`, and `tvat_tissue_hu_m190_m30`; and
- CSA divided by trunk CSA for each measurement channel.

Raw values and normalized fractions are stored together. This keeps patient
size information available while providing a simple within-trunk comparison.
One fraction is valid only when its tissue numerator and trunk denominator
cover the same physical part of the bin; differing valid-support coverage is
reported as an invalid measurement rather than mixed silently.
The redundant `vat_tissue_hu_m190_m30` union is omitted from the default
signature because filtered aVAT and tVAT remain separate; it is still
available in the slice, vertebral, range, and summary tables.

If a named tissue profile enables additional definitions, corresponding CSA,
HU, and trunk-fraction channels are appended to the signature. The fixed grid
and default channels do not change. `signature_profile_id` records the exact
profile.

### Missingness

- `coverage_fraction=0`: the bin is outside the acquired CT; measurements are
  null.
- `0 < coverage_fraction < full_coverage_tolerance`: the observed portion is
  summarized and the bin is marked `partial_fov`.
- scanned bin with no supported native compartment or filtered tissue: CSA is
  valid zero and mean HU is null with `empty_compartment` or `empty_tissue`,
  respectively.
- failed tissue/trunk measurement: value is null with the specific QC reason.
- unresolved vertebral reference: physical bounds and measurements are null
  with `missing_anchor` or `invalid_extent`.

Downstream cohort code must retain coverage and validity instead of replacing
structural missingness with zero.

## Reader views

`BodyComposition.measurement.api` validates the five tables and provides:

- `load_measurement_tables`;
- `load_signature`;
- `signature_measurements`;
- `l3_measurements`; and
- `range_measurements`.

The L3 and range views are read-only derivations. Caller-supplied measured
height may be used for explicitly named indices; the pipeline never infers
patient height from the CT.

## Validation boundary

Automated tests cover array/metadata order, affine geometry, anisotropic and
oblique integration, contours, native territories, T13/L6/sacrum, fixed-mm
signature alignment, profile extensions, missingness, schema stability, API
views, and atomic export. Clinical use still requires prespecified
segmentation, attenuation/protocol, circumference, alignment, cohort coverage,
and missingness validation.
