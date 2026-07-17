# Optional PDF reporting

The reporting service creates derived research/QC pages from a complete
analysis. Canonical NIfTI, Parquet, JSON, and QC artifacts remain authoritative.
Reporting never reruns CTDeepRot, vertebral/tissue inference, or measurement stage scientific
measurements.

## Layouts

Both layouts are exactly one ISO A4 landscape page per case and use embedded
Bitstream Vera fonts at 8 pt or larger.

- `spine_overview_v1` shows a physical-space sagittal thick-slab CT projection
  with colored, text-labeled vertebral-body instances and one whole-territory
  measurement row per native vertebral level.
- `spine_profile_v2` uses the same physical superior-coordinate axis for the
  sagittal projection, an unsmoothed stacked tissue-area profile, and the
  per-level table. The stack uses mutually exclusive SM, IMAT, SAT, aVAT, and
  tVAT when available. It never stacks total VAT together with aVAT/tVAT.
  Trunk area is an unfilled reference curve.

Neither layout stretches individual vertebral intervals or creates a fixed
longitudinal signature. T13, L6, and sacrum remain native detected labels. Only
the canonical vertebral-body mask is rendered; full vertebra/posterior-element
segmentations are not accepted by the reporting action.

The table uses the three physical measurement stage bins. CSA and circumference are averaged
with `bin_integration_length_mm`; SM HU is weighted by SM area times integration
length. Missing or invalid values remain missing and include a marker. Display
rounding does not modify the unrounded audit values in `report_manifest.json`.

## Enable reports during analysis

```yaml
reporting:
  enabled: true
  layout: spine_overview_v1
```

Per-case outputs are:

```text
reports/<case_id>/case_report.pdf
reports/<case_id>/report_manifest.json
```

For a batch with a common workspace, the pipeline also writes:

```text
aggregate/reports/run-<run_timestamp>/case_reports.pdf
aggregate/reports/run-<run_timestamp>/report_manifest.json
```

The individual PDF is the completion marker and is published only after its
page and manifest validate. Before a rerender is published, any stale PDF
completion marker is removed; an interrupted publication therefore cannot look
current merely because an older PDF remains. A scientific-stage failure
receives an explicit one-page failure/status report if possible. When a valid
prepared CT and vertebral-body result already exist in the same physical
domain, that page retains the available sagittal overview and marks every
measurement (and the Phase B profile) unavailable. If that view cannot be
constructed safely, the page falls back to the technical stage/reason only.
Neither failure layout implies analysis success. A report-stage failure leaves
the scientific outputs valid but makes a report-enabled export incomplete and
the command exits nonzero.

## Manual-review behavior

Canonical orientation, vertebral, and measurement review flags are copied into
the report without clearing or adjudicating them. A changed orientation states
that the CT was automatically reoriented before analysis and requires manual
review. An unchanged uncertain case states that processing continued without
orientation repair.

Individual PDFs remain one page and show a compact warning banner. A combined
PDF has exactly `N` pages when no selected case is flagged and `N + 1` pages
when any case is flagged. In the latter case the review summary is page 1 and
case bookmarks/page numbers account for the offset. One summary page holds at
most 24 affected cases; a larger explicit export fails before producing a
combined PDF and must be split.

## Post-hoc Python API and CLI

The Python surface is available from `BodyComposition.reporting`:

```python
from BodyComposition.reporting import render_case_report
from BodyComposition.reporting.io import load_case_report_input

case = load_case_report_input("case_manifest.json")
result = render_case_report(
    case,
    "output/reports/case-001",
    {"enabled": True, "layout": "spine_profile_v2"},
)
```

The post-hoc case manifest is explicit and immutable. Before rendering, the
service verifies the preparation stage prepared-pixel digest, the preparation stage/vertebral stage physical domain,
the per-slice measurement stage physical coordinates and row count, the orientation links in
vertebral stage/measurement stage provenance, and the vertebral stage/measurement stage native-label mapping. A mixed or stale
bundle is rejected even when its case IDs happen to match. Paths may be
absolute or relative to the manifest, but paths are never copied into output
PDFs or report manifests:

```json
{
  "schema_version": "1.0.0",
  "case_id": "case-001",
  "prepared_ct": "prepared.nii.gz",
  "vertebral_body_mask": "vertebral_bodies.nii.gz",
  "vertebral_result_json": "vertebral_result.json",
  "measurement_directory": "tables/case-001",
  "orientation_report_json": "orientation/case-001/orientation_report.json",
  "measurement_qc_json": "qc/case-001_measurement-qc.json"
}
```

Render or validate it with:

```bash
bodycomposition_reports case \
  --manifest case_manifest.json \
  --output reports/case-001 \
  --layout spine_profile_v2

bodycomposition_reports validate \
  --manifest reports/case-001/report_manifest.json \
  --pdf reports/case-001/case_report.pdf
```

An explicit export manifest preserves its `cases` order:

```json
{
  "export_id": "cohort-export-001",
  "cases": [
    {"case_manifest": "case-001/case_manifest.json"},
    {"case_manifest": "case-002/case_manifest.json"}
  ]
}
```

```bash
bodycomposition_reports export \
  --manifest export_manifest.json \
  --output aggregate/reports/cohort-export-001
```

## Privacy and interpretation

Case IDs must be path-safe pseudonymous identifiers. Reports do not include
patient names, dates of birth, MRNs, accession numbers, source paths, hostnames,
usernames, DICOM text overlays, or temporary directories. PDF metadata is set
explicitly. The manifest stores logical artifact names and SHA-256 digests,
not local paths.

The sagittal view is a deterministic thick-slab projection for rapid technical
review, not a diagnostic image or clinical report. Canonical review JSON and
source images must be used for adjudication.
