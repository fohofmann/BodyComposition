# Optional PDF reporting

Reporting is a derived research/QC view. Canonical NIfTI, Parquet, JSON, and
QC artifacts remain authoritative; rendering never reruns orientation,
segmentation, or measurement.

## Enable during analysis

```yaml
reporting:
  enabled: true
  layout: spine_overview_v1
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
notes/QC cards complete the page.

`spine_profile_v2` retains the sagittal and axial views and adds an unsmoothed
stacked tissue-area profile aligned to the same physical superior-coordinate
axis. It stacks mutually exclusive SM, IMAT, SAT, aVAT, and tVAT when present;
total VAT is never stacked together with aVAT/tVAT. Trunk area is an unfilled
reference curve. Its fixed left-to-right order is sagittal spine, tissue-area
profile, vertebral summary, and axial example. The adjacent spine and profile
share one physical superior-coordinate scale. The table uses a regular
categorical grid with equal-height rows because anatomical position is already
shown by the labeled spine and must not make the numerical table irregular.

Both layouts:

- render only canonical vertebral bodies, never posterior elements;
- preserve T13, L6, and sacrum as native labels;
- use native millimetres without stretching vertebral intervals;
- use the three physical vertebral-territory bins to calculate whole-territory display rows;
- retain missing/invalid values as `NA` with a marker; and
- store unrounded audit values in the report manifest.

The axial panel uses the same processed tissue-label mask as the measurement stage. The report
verifies its digest before rendering so the image and displayed measurements
cannot come from different analyses. L3 is preferred. If L3 is unavailable,
the selected nearest native vertebral level and the fallback are stated on the
page and in the report manifest; no anatomical level is inferred silently.

Technical metadata is deliberately allowlisted. It may include the analysis
date, input format, package/pipeline version, runtime backend and hardware,
scanner manufacturer/model, voxel spacing, slice thickness, and orientation,
vertebral, and tissue model identifiers. The displayed date is the analysis
timestamp, not a patient, acquisition, or study date. Public demonstration
cases may also carry an allowlisted data-attribution note.

## Manual-review summary

Canonical orientation, vertebral, measurement, pipeline, and reporting flags
are copied without adjudication. A repaired orientation states that the CT was
automatically reoriented before analysis and requires manual review. Every
affected individual page has a compact warning annotation.

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
