# PDF reporting

PDF reporting is an optional research/QC view derived from completed pipeline
artifacts. It never reruns orientation, segmentation, or measurement. NIfTI,
Parquet, JSON, and QC artifacts remain authoritative.

## Enable reporting

Reporting is disabled by default. Generate a configuration, set
`reporting.enabled: true`, and pass it to `analyze` or `batch`:

```bash
bodycomposition config show-default -o bodycomposition.yaml
bodycomposition analyze CT.nii.gz --config bodycomposition.yaml
```

The relevant defaults are:

```yaml
reporting:
  enabled: true
  layout: spine_profile_v2
  individual_pdf: true
  combined_pdf: true
```

| Output | Location |
| --- | --- |
| Individual page | `reports/case_report.pdf` inside the case bundle |
| Individual audit | `reports/report_manifest.json` |
| Ordered batch PDF | `aggregate/reports/<run-id>/case_reports.pdf` |
| Batch report audit | `aggregate/reports/<run-id>/report_manifest.json` |

A combined report contains one run cover followed by one page per included
case. It is rebuilt from the immutable batch order, validated, and published
atomically. An interrupted refresh leaves the previous valid PDF/manifest pair
available. A completed `N`-case report therefore has `N + 1` pages.

## Layouts

| Layout | Content |
| --- | --- |
| `spine_profile_v2` | Default: spine localization, aligned tissue-area profile, native-compartment HU distributions, measurements, and axial overlay |
| `spine_overview_v1` | Lean view without the longitudinal area/HU panels |

Every case page is A4 landscape. The default page contains:

| Panel | Meaning |
| --- | --- |
| Spine localization | Sagittal thick-slab CT cropped around labeled vertebral bodies; waist and pelvic positions are annotated when available |
| Tissue area | Unsmoothed SM, SAT, aVAT, and tVAT HU-filtered areas along the same physical superior axis as the spine |
| HU distribution | Whole-analyzed-volume 5-HU distributions from the unfiltered native SM, SAT, aVAT, and tVAT compartments |
| Measurements | Whole-territory SM, VAT, SAT, and trunk-circumference summaries with equal-height anatomical rows |
| Axial overlay | Tissue overlay at L3 or an explicitly labeled nearest-level fallback |
| Technical context | Pipeline/model identity, runtime, scanner, voxel size, and measured slice-normal thickness when available |
| Notes | Plain-language warnings and errors, including automatic orientation repair |

Only canonical vertebral corpora are displayed and measured. T13, L6, and
sacrum retain their native names. Vertebral intervals are not stretched.

## Display and missingness rules

- Numeric measurements from an incomplete or scan-boundary vertebral territory
  remain visible and are marked `!`; territory truncation alone does not create
  `NA`.
- A closed, nonfragmented trunk contour that touches the lateral image boundary
  remains numeric, is marked `!`, and is drawn dotted in the longitudinal plot.
  Other invalid or missing observations remain gaps and are never interpolated.
- Waist and pelvic observations may be numeric but scientifically ineligible;
  their coverage, reason, and FOV-cropped status remain in the tables and audit.
- HU histograms use unchanged CT values and native compartment membership, not
  the configured HU-filtered tissue masks. The display range is −190 to 150 HU;
  exact counts, tails, summaries, and validity remain in
  `hu_distributions.parquet`.
- The page has no status badge. Actionable findings appear in Notes; routine
  cranial/caudal vertebral contact remains in canonical QC but is not printed
  as an independent warning.
- Unrounded values, layout choices, panel coordinates, artifact identities,
  and printable/non-printable flags remain in `report_manifest.json`.

The run cover summarizes included/failed/review cases, issue counts, timing,
model IDs, and selected reproducibility settings. It does not adjudicate a
case. Use the canonical review queue and linked case artifacts for that task.

## Configuration

The generated default configuration is the field-level reference. Common
options are:

| Key | Purpose |
| --- | --- |
| `reporting.layout` | Select `spine_profile_v2` or `spine_overview_v1` |
| `individual_pdf` / `combined_pdf` | Select case and batch outputs |
| `measurement_columns` | Choose one to six supported whole-territory table measures; four are enabled by default |
| `vertebral_range` | Fixed to the detected native levels in this release |
| `numeric_precision` | Printed decimal precision |
| `ct_window` / `overlay_opacity` | CT and overlay display only |

Reporting changes presentation and artifact identity, not the scientific
measurement identity. It requires `output.save_tissue_labels: true` so the
axial view can be reproduced from the immutable bundle.

## Post-hoc export

Post-hoc rendering accepts an explicit report-input manifest; it does not
discover or mix files from a directory. Required fields are defined by the
packaged
[`report_case_input.schema.json`](../BodyComposition/schemas/report_case_input.schema.json).
A minimal manifest is:

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
  "measurement_qc_json": "qc/qc.json"
}
```

```bash
bodycomposition reports render case-report-input.json \
  --output reports/case-001 --layout spine_profile_v2 --json

bodycomposition reports inspect reports/case-001/report_manifest.json \
  --pdf reports/case-001/case_report.pdf --json
```

For ordered collation, use a
[`report_export_input.schema.json`](../BodyComposition/schemas/report_export_input.schema.json)
manifest and `bodycomposition reports collate`. Python callers use
`render_report`, `collate_report_export`, and `inspect_report`.

## Privacy and interpretation

DICOM patient name, birth date, scan date, and sex may be read locally for an
enabled report header. Their values are not copied into canonical case or
report manifests; the report audit stores only the present-field list and a
content digest. Explicit post-hoc patient metadata is still PHI.

Generated PDFs and PNGs remain sensitive medical data and may contain
burned-in text from the input CT. BodyComposition does not deidentify input or
scrub pixels. The report is a technical research summary, not a diagnostic
image or clinical report.
