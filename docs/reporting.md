# Optional PDF reporting

Reporting is a derived research/QC view. Canonical NIfTI, Parquet, JSON, and
QC artifacts remain authoritative; rendering never reruns orientation,
segmentation, or measurement.

## Enable during analysis

```yaml
reporting:
  enabled: true
  layout: spine_profile_v2
  individual_pdf: true
  combined_pdf: true
```

Every selected case receives exactly one A4-landscape page at:

```text
reports/case_report.pdf
reports/report_manifest.json
```

within its immutable case bundle. A multi-case batch with `combined_pdf=true`
maintains the current run order at:

```text
aggregate/reports/<run-id>/case_reports.pdf
aggregate/reports/<run-id>/report_manifest.json
```

“Current run” means the explicit immutable batch manifest, never directory
discovery. The combined document always starts with one A4-landscape run cover,
followed directly by all currently finished individual case pages in immutable
manifest order. The cover summarizes planned and included case counts, mean
finished-case runtime, the complete run's start, finish, and elapsed wall time,
pipeline and model identifiers, runtime context, execution and case-level
quality-control issue counts, and privacy-safe input/output directory labels.
In-progress covers explicitly
show that the finish time and duration remain in progress. The
cover does not print case/failure folder templates, combined-PDF paths, or
ordering implementation details; those remain in the manifest. Internal
worker execution strategy is likewise audit-only. Instead of printing an
opaque configuration digest, the cover uses a compact key/value table for a
curated subset of active settings. A row is eligible only when the setting is
user-configurable and either changes scientific results or is important for
reproducibility. The cover therefore includes analysis scope; orientation
policy and decision thresholds; primary prediction models; body-surface
derivation and cleanup; measurement coverage and vertebral cleanup; separate
SM and AT filters; and deterministic execution. It intentionally excludes
display-only settings such as overlay, projection, layout, and numeric
formatting; throughput controls such as device, batch size, workers, and
timeout; output/persistence switches; and routine QC-only thresholds.
The complete digest and normalized configuration remain in the manifest and
`normalized_config.yaml` for exhaustive reproducibility. The raw derived
body-surface backend identifier and optional rib/iliac landmark backend remain
recorded but are not printed; the actual body-surface parameters are printed
instead. Reader-facing model and configuration rows follow execution order.
The cover body uses three
vertical zones: pipeline/configuration on the left; compact runtime, directory,
and progress sections in the middle; and a full-height Issues column on the
right. The Issues column combines execution failures with case-level
quality-control findings. Each finding is reported as affected cases and the
percentage of included cases; the column header states the total included
denominator as `N`. Finding types may overlap within one case.
Routine cranial or caudal vertebral contact with the field-of-view boundary is
not printed there, but remains available in the canonical review records.

After every terminal case, the service rebuilds the combined document in a
temporary file and validates the PDF together with its audit manifest. The
pair is then installed in an immutable hidden version directory and a single
relative `.current` symlink is replaced atomically. The two stable public names
resolve through that pointer:

```text
case_reports.pdf      -> .current/case_reports.pdf
report_manifest.json  -> .current/report_manifest.json
.current              -> .snapshots/<report-id>
```

The active snapshot and one rollback generation are retained. Existing
pre-snapshot direct files are validated and migrated automatically on the next
refresh. A stopped worker therefore leaves the prior complete pair in place;
it cannot expose a newly rendered PDF with an older manifest. Programmatic
readers that require a transactionally fixed pair should resolve `.current`
once, read both immutable files from that selected directory, and verify the
manifest digest. The stable aliases remain convenient for normal PDF readers.
Restarting the same immutable run discovers terminal case manifests and
refreshes the document before continuing. The cover is marked `IN PROGRESS`
until every planned case has a terminal result. Individual pages are always
retained. If collation fails, the run manifest records the reporting failure
and keeps links to any previously validated combined snapshot; the
scientifically complete case bundles remain authoritative.

## Layouts

`spine_overview_v1` is a lean scientific dashboard. It contains a
physical-space sagittal thick-slab CT projection with color-coded and
text-labeled vertebral-body instances. The displayed sagittal field of view is
cropped to the vertebral-body mask with fixed physical margins so the spine is
legible without changing its geometry. It is followed from left to right by the
vertebral summary table and an axial filtered-tissue view at L3 (or an
explicitly labeled nearest-level fallback). Compact technical-metadata and
case identity share one compact header; a full-width Notes region completes the
page. The rightmost column is an aligned stack containing the axial
visualization, key anthropometry, and technical context.

The header and Notes region plus the right-column sections use a common 10-point
content inset and 18-point title baseline offset. The analytical body uses an open
editorial grid: aligned headings, whitespace, and 0.6-point neutral rules
separate the panels without repeated rounded dashboard cards. The header and
Notes region have no outer border; Notes uses whitespace below its title and
retains only the column separators. Notes is a flat bullet register without
subheadings. Every
printable plain-language observation is retained and balanced across three
columns, or four when the three-column layout cannot fit. A register beyond
that audited capacity fails rather than being truncated or summarized.

`spine_profile_v2` is the default when reporting is enabled. It retains the
sagittal and axial views and adds an unsmoothed `Tissue area` profile plus four
voxel-level native-compartment HU histograms under `HU distribution`. The area
profile is aligned to
the sagittal view's physical superior-coordinate axis and
stacks conventional skeletal-muscle tissue, SAT, aVAT, and tVAT; trunk area is
an unfilled reference curve. Sections with a strictly valid trunk-area
measurement are solid. Numeric observations whose contour reaches the image
boundary are still shown, but dotted to identify possible field-of-view
underestimation. Missing or otherwise invalid observations remain genuine gaps
and are never interpolated. The HU panel uses the unfiltered model-native
SM, SAT, aVAT, and tVAT compartments with the unchanged prepared CT values;
the configured HU-restricted downstream masks do not determine histogram
membership. The `HU distribution` detail line therefore uses the standard
funnel and `HU filter inactive`. It shows a fixed histogram resolution over
-190 to 150 HU, exact
voxel-level median and interquartile range, and one peak-normalized distribution
shape per compartment over the complete analyzed volume. Exact median and IQR values
remain in the row summary, while only the median is drawn as a reference line;
there is no shaded IQR/background band and no vertical threshold guide. The
fixed -30 HU tick and the filter definition above the tissue-area profile carry
the relevant threshold context without adding an ambiguous grey region. The normalization makes
shape legible but deliberately does not encode cross-compartment voxel-count
magnitude. The peak normalization is audit-only and has no separate printed
caption. Exact counts and any clipped values remain in
`hu_distributions.parquet` and the report audit. Values outside the fixed
display range remain in that audit but do not add a star or axis annotation to
the printed panel. Its fixed left-to-right order is sagittal spine, tissue-area
profile, HU distribution, measurements, and axial example. The `Measurements`
panel prints the active HU ranges on its first detail line and identifies the
whole-level mean on the second. The table
uses a regular categorical grid with equal-height rows because anatomical
position is already shown by the labeled spine and must not make the numerical
table irregular.

Plots, legends, and the axial overlay share one explicit warm scientific
palette: skeletal muscle is dark burgundy, while SAT, aVAT, and tVAT use
progressively deeper gold-to-ochre tones. Vertebral bodies use distinct modern
region families for cervical, thoracic, lumbar, and sacral anatomy. Labels and
boundaries provide a non-color distinction.

The default one-page table does not duplicate the HU-distribution channels as
attenuation columns. It contains whole-territory SM, VAT, and SAT areas plus
trunk circumference; exact per-channel slice attenuation remains in
`slices.parquet`, while voxel distributions are in
`hu_distributions.parquet`. Tissue abbreviations are rendered consistently as `SM`,
`SAT`, `aVAT`, and `tVAT`; the table's `VAT` column is the explicit union of
the filtered aVAT and tVAT tissue views.

HU-filter state uses one shared visual convention: a plain funnel followed by
an explicit text state. Active measurements show the exact ranges
(`SM -29..150 | AT -190..-30 HU`), where `AT` means adipose tissue. Inactive
measurements show `HU filter inactive`; the icon is not struck through. The
default tissue-area profile and measurements table use the active ranges; the
native-compartment HU panel is explicitly inactive. If a custom measurements
table contains only unfiltered compartment metrics, the heading remains
`Measurements` and the inactive caption is used. Mixed custom tables retain the
applied ranges and are marked `mixed`.

The sagittal localization view adds directly labeled horizontal markers at the
physical positions used for `Waist min` and `Pelvic max`. Both use the same
solid, white-keylined rule and the same left-aligned label treatment. Labels
sit fully below the rule, with an above-rule fallback only at an image edge.
The key-anthropometry section uses the same nomenclature.
Its footnote defines `*` as an incomplete anatomical range or a contour that
reaches the field-of-view edge. Vertebral level
names are compact translucent white badges centered on their vertebral
centroids. They use collision-only vertical separation and no leader lines,
reducing clutter while preserving an immediate label-to-body relationship.
The sagittal projection method and the sagittal and axial CT windows remain in
the report manifest's display audit for reproducibility, but are not printed in
the reader-facing panels. Area units use the compact baseline notation `cm2` to
avoid disrupting the fixed one-page layout.

Both layouts:

- render only canonical vertebral bodies, never posterior elements;
- preserve T13, L6, and sacrum as native labels;
- use native millimetres without stretching vertebral intervals;
- use the three physical vertebral-territory bins to calculate whole-territory display rows;
- retain observed means for incomplete or truncated territories and mark the
  level with `!`; territory incompleteness alone does not produce `NA`;
- retain a numeric trunk circumference when the only contour limitation is a
  closed, nonfragmented contour touching the image boundary, record
  `trunk_mean_circumference_cm_value_is_fov_cropped=true`, and mark the displayed
  value with `!`;
- refuse measurement tables from an older measurement or vertebral-territory
  schema instead of rendering them as a current report;
- retain missing/invalid values as `NA` with a marker and mark observed but
  ineligible anthropometric values with the same review marker; a minimum
  waist over a partially observed T10-to-L5 range is therefore numeric but
  ineligible, while a closed, nonfragmented pelvic contour cropped by the
  image boundary remains numeric, explicitly FOV-cropped, and ineligible; and
- store unrounded audit values in the report manifest.

The case-page background is white for print use. The unboxed header contains
the patient name and, when locally available, date of birth, scan date, and sex,
plus the case ID. Its right edge shows the page position in the complete opened
document. `SACRUM` remains the canonical machine-readable anatomy but is
displayed as `S` in the spine view and vertebral table. The deterministic
report ID remains in the manifest for exact-content linking and collation but
is not printed because its short form is not useful reader-facing information.
The unboxed Notes region uses whitespace below its title and retains only its
column separators.
The analysis date, input format and complete voxel size, pipeline version,
runtime, scanner, and model identifiers are consolidated in the right-column
technical-context section. The complete voxel size and measured slice-normal
thickness are both retained because they may differ in nonuniform or
reconstructed series. Stable monochrome vector icons
identify the fields without repeating uppercase labels; the corresponding
values remain selectable PDF text. Runtime backend, framework, hardware, and
individual model identifiers occupy separate lines without slash or pipe
separators. The right-column sections share the same
0.6-point horizontal rules as the main analytical grid. Repeated report titles,
a metadata subheader, and a case-page footer are intentionally omitted.

The axial panel uses the stable processed tissue-label view generated in the
same analysis. Scientific measurements remain derived from the separate raw
compartment mask. The measurement provenance records both digests, and the
report verifies the processed-view digest before rendering so the overlay and
displayed measurements cannot come from different analyses. L3 is preferred.
If L3 is unavailable, the selected nearest native vertebral level and the
fallback are stated on the page and in the report manifest; no anatomical
level is inferred silently.

Technical metadata is deliberately allowlisted. It may include the analysis
date, input format, package/pipeline version, runtime backend and hardware,
scanner manufacturer/model, voxel spacing, slice thickness, and orientation,
vertebral, and compartment-model identifiers. The displayed date is the analysis
timestamp. The separate header scan date prefers the DICOM acquisition date,
then series, study, or content date. Public demonstration
cases may also carry an allowlisted data-attribution note. These fields are
retained in the report manifest, but source attribution is not printed in the
Notes section because it is not expected for most operational cases.
The full-width Notes register uses three or four columns. It shows every
printable observation as an individual bullet rather than internal machine
codes and does not assign a visible review decision to the reader. An unusually
dense finding set fails the fixed-page capacity check instead of being clipped
or silently compacted.

The table adapts its row height to the detected spine. The released A4 geometry
fits the complete 27-level canonical range (C1-C7, T1-T13, L1-L6, and sacrum)
at 8-point type with at least 11 points of row height. The manifest records the
actual row count, row height, and calculated capacity. An input beyond the
audited capacity fails instead of producing a clipped table.

## Run cover and review summary

Canonical orientation, vertebral, measurement, pipeline, and reporting flags
are copied without adjudication. A repaired orientation states that the CT was
automatically reoriented before analysis. Individual pages list relevant
observations as plain-language bullets in the full-width Notes register, with
no subheadings; the page intentionally has no `Review required` badge. Generic cranial or
caudal vertebral-boundary observations are omitted from the printed Notes
register because they are common acquisition-boundary findings and are not
independently actionable. They remain unchanged in the report manifest and
canonical quality-control record. Boundary findings that can invalidate a
measurement, such as a trunk contour reaching the image edge, remain visible.

For `N` currently included cases, the combined PDF always has exactly `N + 1`
pages: one run cover followed by the ordered case pages. Review findings are
represented by the review-case count on the cover and retained in full,
including per-code counts, in the combined report manifest. Large flagged
cohorts therefore do not require a separate review-index page or split solely
because of cover-page capacity. Every combined page is stamped `Page x of y`
for the current validated snapshot.

## Failure pages

When scientific processing fails and reporting is enabled, the service creates
an explicit one-page status report where possible. If a prepared CT and valid
vertebral-body result already exist in the same physical domain, the page may
retain that partial overview while marking measurements unavailable. Invalid
inputs and cases cancelled before inference receive a text-only status page.
No failure page implies scientific success.

## Post-hoc report export

Post-hoc rendering uses a dedicated immutable report-input manifest. It does
not accept an arbitrary case directory or discover files. This explicit
boundary verifies the prepared CT pixel digest, orientation/vertebral physical domain,
measurement row coordinates and identity, orientation provenance, and native-label
mapping before rendering.

Case input example:

```json
{
  "schema_version": "1.0.0",
  "case_id": "case-001",
  "prepared_ct": "prepared.nii.gz",
  "tissue_label_mask": "masks/tissue_labels.nii.gz",
  "vertebral_body_mask": "masks/vertebral_bodies.nii.gz",
  "vertebral_result_json": "qc/vertebral_result.json",
  "measurement_directory": "tables",
  "orientation_report_json": "orientation/orientation_report.json",
  "measurement_qc_json": "qc/qc.json",
  "patient_metadata": {
    "patient_name": "Example Patient",
    "date_of_birth": "1984-12-03",
    "scan_date": "2026-07-18",
    "sex": "F"
  },
  "technical_metadata": {
    "analysis_started_at": "2026-07-18T10:30:00+00:00",
    "input_format": "dicom",
    "pipeline_version": "1.0.0rc1",
    "runtime_backend": "cuda",
    "runtime_hardware": "NVIDIA A100-SXM4-80GB",
    "scanner_manufacturer": "SIEMENS",
    "scanner_model": "SOMATOM Definition AS+"
  }
}
```

```bash
bodycomposition reports render \
  case-report-input.json \
  -o reports/case-001 \
  --layout spine_profile_v2 --json

bodycomposition reports inspect \
  reports/case-001/report_manifest.json \
  --pdf reports/case-001/case_report.pdf --json
```

An ordered export manifest is:

```json
{
  "export_id": "cohort-export-001",
  "cases": [
    {"case_manifest": "case-001/report-input.json"},
    {"case_manifest": "case-002/report-input.json"}
  ]
}
```

```bash
bodycomposition reports collate \
  export.json \
  -o aggregate/reports/cohort-export-001 \
  --layout spine_profile_v2 --json
```

The Python equivalents are `render_report`, `collate_report_export`, and
`inspect_report` from the top-level `BodyComposition` package.

## Privacy and interpretation

Pipeline-derived case IDs are content-based and pseudonymous. For a selected
DICOM series, standard patient name, birth date, study/scan date, and sex tags
are read locally only for the requested PDF header. Their values are not copied
into the canonical case or report manifest; the report manifest stores only
the present-field list and a content digest so report identity remains
deterministic. Explicit post-hoc `patient_metadata` is also rendered locally
and must be handled as PHI. Caller-supplied case IDs, attribution, and review
text still require appropriate governance. Path filtering and field
allowlisting are not PHI detection. Source paths, usernames, and hostnames are
not included in report metadata, but input deidentification and burned-in-text
review remain the caller's responsibility.

The sagittal projection is a deterministic technical overview, not a
diagnostic image or clinical report. Use canonical images and review JSON for
adjudication.
