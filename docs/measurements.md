# Canonical physical measurements

`BodyComposition` and `BodyCompositionFast` write one outcome-blind
measurement bundle per case. The bundle preserves the acquired longitudinal
profile and compact anatomical summaries. It does not create a learned or
template-normalized signature, impute missing coverage, or use clinical
outcomes.

## Authoritative files

- `tables/{caseid}/slices.parquet`: one row for every prepared CT slice;
- `tables/{caseid}/vertebrae.parquet`: three physical bins for every detected
  native vertebral territory, including sacrum;
- `tables/{caseid}/summaries.parquet`: one row of deterministic case-level
  summaries; and
- `qc/{caseid}_measurement-qc.json`: provenance, QC, and review status.

`qc/{caseid}_measurement-review.png` is an optional visual derivative. The
three Parquet tables remain authoritative. There is deliberately no
`signatures.parquet`: cohort-specific longitudinal representations are derived
later from the native-mm tables after a coverage audit.

Every table row carries `schema_version`, `case_id`, `run_id`, and
`analysis_id`. Parquet metadata repeats the table and identity fields. The
reader rejects missing columns, mixed identities, changed field order/types,
and unsupported schema versions.

## Geometry and units

- SimpleITK metadata, sizes, and indices are `x-y-z`.
- Arrays returned by SimpleITK are `z-y-x`.
- Physical coordinates use DICOM LPS millimetres.
- Rows in `slices.parquet` are ordered inferior to superior by
  `position_superior_mm`, independently of array storage direction.
- Areas are cm², volumes are cm³, lengths are millimetres unless a column ends
  in `_cm`, and attenuation is in HU.

For a binary label on one acquired plane:

`area_cm2 = voxel_count * in_plane_pixel_area_mm2 / 100`

The in-plane area and contours use the complete direction matrix. Range and
bin aggregation uses exact overlap between the acquired slice slab and the
requested physical superior-axis interval.

## Tissue-derived body and trunk envelope

The release default is `tissue_segmentation_envelope_v1`. It requires no
separate body model:

1. start from every non-zero voxel in the postprocessed tissue segmentation;
2. find axial connected components and discard components smaller than the
   configured physical area only for envelope construction;
3. apply a physical closing, convex envelope, and Gaussian edge smoothing to
   each retained component;
4. define the body as the union of all original tissue voxels and retained
   component envelopes; and
5. define the trunk as the envelope of the largest retained component on each
   slice.

The default parameters are a 100 mm² component threshold, 6 mm closing radius,
and 2 mm smoothing sigma. They enter the deterministic analysis identity.
The body always contains every source tissue voxel. A clearly separate arm or
device can remain in the body while being excluded from the primary trunk.
Connected arms, sparse tissue predictions, external devices, and lateral
field-of-view truncation remain clinical-validation cases; the algorithm does
not claim a true skin surface before that gate passes.

`totalsegmentator_body_task299_v1` and `deterministic_body_mask_v1` remain
explicit alternatives. A backend failure never triggers another backend.
Disabling measurement landmarks permits a canonical run without any
TotalSegmentator stage. Enabling anatomical mid-waist landmarks still uses the
pinned TotalSegmentator task-297 adapter.

## `slices.parquet`

Every acquired slice remains present, including empty, cropped, or invalid
slices. Missing or ineligible values are null with a validity flag and reason;
a truly absent tissue has a valid area of zero.

### Geometry and anatomical assignment

Important fields include:

- `slice_id`, `slice_index_zyx_z`, and `longitudinal_order`;
- `position_superior_mm`, physical slice centre and normal;
- `slice_slab_inferior_mm`, `slice_slab_superior_mm`, native through-plane
  thickness, and in-plane pixel area;
- `assigned_vertebral_level`, `vertebral_territory_bin`, and
  `vertebral_assignment_status`.

The level/bin annotation identifies the territory containing the slice centre
and is useful for plotting and tables. At an exact midpoint tie, the cranial
territory is selected deterministically. Exact bin summaries use physical slab
overlap rather than this centre annotation.

### Tissue measurements

For every released tissue `<name>`:

- `<name>_voxel_count`;
- `<name>_area_cm2`, `<name>_area_valid`, and `<name>_area_reason`;
- `<name>_mean_hu`, `<name>_hu_valid`, and `<name>_hu_reason`.

Native aVAT and tVAT remain separate. `total_vat` is their deterministic union
when both are present, or the native VAT label when that is the source schema.
`total_segmented_tissue_area_cm2` is the union of all non-zero tissue labels;
it is not called body or trunk CSA.

The canonical slice export intentionally omits redundant compartment
intersections, fractions, body perimeter, vertebral-overlap lists, and local
centroid coordinates. These can be derived for a prespecified study if they
later become necessary.

### Trunk and QC fields

- `trunk_area_cm2` and its validity/reason;
- `trunk_circumference_cm` from the primary closed external contour;
- component, closure, boundary-contact, fragmentation, and internal-gap QC;
- adjacent-slice circumference jump measures; and
- `slice_measurement_valid` plus `slice_qc_status`.

Internal contour holes are not added to circumference. A contour touching the
image edge or a fragmented primary trunk is observable but ineligible for
strict summaries.

## Native vertebral territories

Only the canonical vertebral-body/corpus labels from vertebral stage define vertebral
anatomy. Whole-vertebra masks and posterior elements are never used.

For every detected cervical, thoracic, lumbar, or sacral label:

1. retain the largest cleaned vertebral-body component and calculate its
   physical centroid and extent;
2. sort valid centroids cranial to caudal;
3. place the boundary between adjacent detected levels at the physical
   midpoint of their centroids; and
4. divide each resulting territory into three equal superior-axis bins:
   superior, middle, and inferior.

The midpoint territories assign intervertebral slices to the closest adjacent
level without stretching the scan or normalizing patient height. A complete C1
uses its observed superior body boundary; a complete sacrum uses its observed
inferior boundary. When the acquisition starts below C1 or ends above the
sacrum, the outer detected territory is marked incomplete and values beyond
its observed body extent remain unassigned. Internal label gaps remain
continuous to avoid breaking the longitudinal contour, but affected levels are
flagged for enumeration review.

T13 and L6 remain ordinary native levels and are not compressed. Sacrum is
treated like every other supported level and contributes three rows.

## `vertebrae.parquet`

The table contains exactly three ordered rows per detected supported level.
`territory_bin=1` is superior and `territory_bin=3` is inferior. Important
fields include:

- native label, native level, variant and sequence-gap flags;
- vertebral-body centroid/extent fields;
- territory and bin physical bounds, height, completeness, coverage, and QC;
- mean CSA for every available tissue and for the trunk;
- mean trunk circumference; and
- pooled tissue HU.

Mean CSA is the ordinary arithmetic mean when complete contributing slices
have equal height. Exact physical overlap weights are used only where needed,
such as partial boundary slices, varying spacing, or obliquity. HU is pooled by
contributing tissue voxels and fractional slice overlap.

Volume columns are intentionally not duplicated in this table. For any valid
bin, exact volume can be reconstructed as:

`volume_cm3 = mean_csa_cm2 * bin_integration_length_mm / 10`

`bin_height_mm` is the anatomical superior-axis span used for longitudinal
position. `bin_integration_length_mm` is the through-plane physical length
needed for exact volume reconstruction and differs only for oblique
acquisitions. A whole-territory mean or volume is the length-weighted
combination/sum of its three bins.

## `summaries.parquet`

The single case row contains deterministic, interpretable views:

- exact 200-mm L3-centred slab measurements when L3 and the full slab are
  covered;
- a whole-L3-territory view derived from the slice table;
- minimum valid trunk circumference across the complete T10-to-L5 search
  interval;
- anatomical mid-waist circumference when both rib and iliac landmarks are
  valid;
- maximum trunk circumference in the complete sacral territory; and
- explicitly named minimum-waist-to-pelvic and midwaist-to-pelvic ratios.

A search extremum at its interval boundary remains visible but is ineligible
for an unqualified waist/pelvic ratio. The sacral maximum is called pelvic,
not hip, until separately validated.

## Range and L3 views

The read-only API/CLI provides:

- `territory_mean`: a whole-L3 physical-range view reconstructed from
  `slices.parquet` and the L3 territory bounds;
- `slice`: the acquired plane closest to the L3 vertebral-body centroid; and
- strict or explicitly partial named physical ranges.

Strict range mode requires complete acquisition and metric-specific validity.
Partial mode returns observed length and coverage and never labels a partial
result complete.

```bash
bodycomposition_measurements \
  --tables /workspace/tables/case-001 \
  --view l3 \
  --aggregation territory_mean

bodycomposition_measurements \
  --tables /workspace/tables/case-001 \
  --view range \
  --start-level T12 \
  --end-level L5
```

## Longitudinal analyses

The full native-mm longitudinal result is `slices.parquet`. A study may model
each tissue curve with a vertical offset, functions, splines, functional data
analysis, or a fixed-mm grid, but that representation is not frozen in the
production package. Patient size and CT coverage remain explicit through
physical positions, anatomical assignments, coverage, and missingness. Scans
are never stretched and absent anatomy is never imputed by the pipeline.

The three-bin territory table is a compact, interpretable comparison view; it
does not replace the native longitudinal source.

## Validation boundary

Automated tests cover array/metadata ordering, affine geometry, anisotropic and
oblique volumes, contour calculations, deterministic body-envelope behavior,
midpoint territories, T13/L6/sacrum, three-bin overlap, equal-height arithmetic
means, missing edges, schema stability, API/CLI views, and atomic export.

Clinical release claims still require the prespecified contour, waist/pelvic,
and cohort-coverage audits. Until those pass, derived trunk circumference and
anthropometric extrema must retain their QC/research-use framing.
