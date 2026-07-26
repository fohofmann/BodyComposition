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
collates the current run order to:

```text
aggregate/reports/<run-id>/case_reports.pdf
aggregate/reports/<run-id>/report_manifest.json
```

“Current run” means the explicit immutable batch manifest, never directory
discovery. Individual pages are retained. If collation fails, the run manifest
records the reporting failure and the scientifically complete case bundles
remain authoritative.

## Layouts

`spine_overview_v1` is a lean scientific dashboard. It contains a
physical-space sagittal thick-slab CT projection with color-coded and
text-labeled vertebral-body instances. The displayed sagittal field of view is
cropped to the vertebral-body mask with fixed physical margins so the spine is
legible without changing its geometry. It is followed from left to right by the
vertebral summary table and an axial tissue-segmentation view at L3 (or an
explicitly labeled nearest-level fallback). Compact technical-metadata and
case identity share one boxed header; a full-width Notes/QC card completes the
page. Key anthropometry is displayed in its own bordered card immediately
below the axial visualization rather than inside the axial card.

The header, Notes/QC, and key-anthropometry box use a common 10-point content
inset and 18-point title baseline offset. The analytical body uses an open
editorial grid: aligned headings, whitespace, and 0.6-point neutral rules
separate the panels without repeated rounded dashboard cards. Notes/QC is a
structured review summary. Domain headings group concise plain-language
findings, while a separate data/provenance section carries source attribution,
display missingness, orientation actions, and the pointer to the complete
canonical QC record. Dense review registers are summarized rather than allowed
to overflow the page.

`spine_profile_v2` is the default when reporting is enabled. It retains the
sagittal and axial views and adds an unsmoothed
stacked tissue-area profile plus a per-slice tissue-HU heatmap. Both are aligned
to the sagittal view's physical superior-coordinate axis. The area profile
stacks conventional skeletal-muscle tissue, SAT, aVAT, and tVAT; trunk area is
an unfilled reference curve. The heatmap uses those same HU-restricted masks,
shows their voxel means on one fixed -190-to-150-HU scale, and leaves invalid
cells white. Its fixed left-to-right order is sagittal spine, tissue-area
profile, tissue-HU heatmap, vertebral summary, and axial example. The table
uses a regular categorical grid with equal-height rows because anatomical
position is already shown by the labeled spine and must not make the numerical
table irregular.

The default one-page table does not duplicate the heatmap channels as
attenuation columns. It contains whole-territory SM, VAT, and SAT areas plus
trunk circumference; exact per-tissue slice attenuation remains in
`slices.parquet`. Tissue abbreviations are rendered consistently as `SM`,
`SAT`, `aVAT`, and `tVAT`; the table's `VAT` column is the explicit total of
the two visceral compartments.

Both layouts:

- render only canonical vertebral bodies, never posterior elements;
- preserve T13, L6, and sacrum as native labels;
- use native millimetres without stretching vertebral intervals;
- use the three physical vertebral-territory bins to calculate whole-territory display rows;
- retain missing/invalid values as `NA` with a marker and mark observed but
  ineligible anthropometric values with the same review marker; and
- store unrounded audit values in the report manifest.

The case-page background is white for print use. Square hairline boxes are
reserved for the compact header, Notes, and key anthropometry. The header
contains the pseudonymous case ID, analysis ID, page position, report ID, and
two rows of allowlisted technical metadata. Stable monochrome vector icons
identify the analysis date, input, pipeline, runtime, scanner, slice thickness,
and models without repeating uppercase field labels; the corresponding values
remain selectable PDF text. The icon semantic keys are recorded in the report
manifest display audit. Repeated report titles and a case-page footer are
intentionally omitted.

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
vertebral, and tissue model identifiers. The displayed date is the analysis
timestamp, not a patient, acquisition, or study date. Public demonstration
cases may also carry an allowlisted data-attribution note. These fields are
integrated into the boxed header rather than consuming a separate lower card.
The full-width Notes card uses up to three text columns when required. It shows
short, plain-language observations rather than internal machine codes and does
not assign a visible review decision to the reader. More than eight printable
observations are compacted by domain with a count and canonical-record pointer,
so an unusually dense finding set cannot overflow the fixed page.

The table adapts its row height to the detected spine. The released A4 geometry
fits the complete 27-level canonical range (C1-C7, T1-T13, L1-L6, and sacrum)
at 8-point type with at least 11 points of row height. The manifest records the
actual row count, row height, and calculated capacity. An input beyond the
audited capacity fails instead of producing a clipped table.

## Manual-review summary

Canonical orientation, vertebral, measurement, pipeline, and reporting flags
are copied without adjudication. A repaired orientation states that the CT was
automatically reoriented before analysis. Individual pages group relevant
observations under plain-language headings in the full-width Notes register;
the page intentionally has no `Review required` badge. Generic cranial or
caudal vertebral-boundary observations are omitted from the printed Notes
register because they are common acquisition-boundary findings and are not
independently actionable. They remain unchanged in the report manifest and
canonical quality-control record. Boundary findings that can invalidate a
measurement, such as a trunk contour reaching the image edge, remain visible.

For `N` cases, the combined PDF has:

- exactly `N` pages when no case is flagged; or
- exactly `N + 1` pages when any case is flagged, with the manual-review
  summary first and case bookmarks/page numbers shifted accordingly.

The summary is limited to 24 affected cases. A larger flagged export fails
atomically and must be split; cases are never silently omitted.

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

Reports use only path-safe pseudonymous case IDs. Manifests/PDF metadata do not
include source paths, usernames, hostnames, MRNs, accession numbers, or dates
of birth. Acquisition and study dates are not displayed. The supplied pixels
may still contain burned-in identifiers; input de-identification remains
mandatory.

The sagittal projection is a deterministic technical overview, not a
diagnostic image or clinical report. Use canonical images and review JSON for
adjudication.
