# Output and manifest schema

Every run is an immutable ordered plan identified by its resolved cases and
configuration. The canonical layout is:

```text
<output>/
├── .bodycomposition/
│   ├── preflight/<request-digest>.json        # shared path-free input summaries
│   └── execution/<run-id>/                    # shared worker state
└── runs/<run-id>/
    ├── normalized_config.yaml                 # queue-compatible config projection
    ├── run_manifest.json
    ├── cases/<case-id>/<analysis-id>/
    │   ├── case_manifest.json
    │   ├── orientation/
    │   │   ├── orientation_report.json
    │   │   ├── orientation_review.png
    │   │   └── corrected_input.nii.gz         # only when repaired and retained
    │   ├── masks/
    │   │   ├── vertebral_bodies.nii.gz
    │   │   ├── tissue_compartments.nii.gz
    │   │   ├── tissue_labels.nii.gz
    │   │   └── body_surface.nii.gz
    │   ├── tables/
    │   │   ├── slices.parquet + slices.csv
    │   │   ├── vertebrae.parquet + vertebrae.csv
    │   │   ├── summaries.parquet + summaries.csv
    │   │   ├── signature.parquet + signature.csv
    │   │   └── hu_distributions.parquet + hu_distributions.csv
    │   ├── qc/
    │   ├── reports/                           # optional
    │   └── logs/stages.jsonl
    ├── failed/<case-id>/<attempt-id>/
    └── aggregate/
        ├── cases.parquet + cases.csv
        ├── failures.parquet + failures.csv
        └── review_queue.parquet + review_queue.csv
```

Some optional upstream/QC masks exist only when their backend or output toggle
is enabled. The manifest artifact inventory, rather than this illustrative
tree, is authoritative for a case. CSV files are enabled by default and are
absent when `--no-csv` or `output.save_csv_tables=false` is selected.

## Input manifest

`batch_input.schema.json` defines an ordered non-empty list:

```json
{
  "schema_version": "1.0.0",
  "cases": [
    {"case_id": "case-001", "input_path": "images/case-001.nii.gz"},
    {
      "case_id": "case-002",
      "input_path": "dicom/case-002",
      "series_uid": "1.2.826.0.1.3680043.10.123.456"
    }
  ]
}
```

`input_path`, optional pseudonymous `case_id`, and optional DICOM
`series_uid` are accepted. `series_uid` is needed when one manifest case points
at an input containing multiple CT series. Relative paths resolve against the
manifest directory.
Case IDs use 1–64 ASCII letters, digits, dot, underscore, or hyphen and must
begin with a letter or digit.

Directory-output DICOM conversion writes `bodycomposition-batch.json` using
this schema and content-derived pseudonymous IDs. It also writes
`bodycomposition-conversion.json`, validated by
`dicom_conversion_batch.schema.json`, with counts, safe output filenames,
content identities, and privacy-minimized failures. `analyze` discovers the
generated batch manifest automatically; a governed run should review and
freeze it before execution.

## Case manifest

`case_manifest.schema.json` records terminal execution and QC status,
content-based input identity, scientific status, flags, path-free provenance,
artifact byte sizes/hashes, timing, and a sanitized failure object. Input
format is `nifti`, `dicom`, or `unreadable_or_missing`. DICOM provenance adds a
hashed Series Instance UID, instance count, CT modality, orientation/position
metadata-completeness flags, and converter identity/version. It does not
record the source file path, hostname, username, raw DICOM identifiers, or
model cache path.

A NIfTI created by `bodycomposition convert` can carry the same safe DICOM
provenance through its adjacent `.bodycomposition.json` sidecar. The manifest
still records the analyzed input format as `nifti`; its `prestage` object
records the hash-bound DICOM source and sidecar identity, while the `dicom`
object retains the technical fields used by orientation QC and reporting.

`execution_status` is one of `succeeded`, `failed`, `skipped_identical`, or
`cancelled`. `qc_status` is independent: `pass`, `review`, `fail`, or
`not_assessed`. A scientifically successful case may still require review.

`inspect_result` and `bodycomposition results inspect` validate the schema,
resolve paths inside the bundle only, require every declared artifact, verify
size and SHA-256, and reject unlisted files. The case bundle should therefore
be treated as immutable after publication.

## Run manifest and aggregates

`run_manifest.schema.json` preserves the requested case order and links each
case to one manifest. It records configuration identity, terminal status,
counts, and aggregate paths. Reusing a `run_id` with another ordered plan or
configuration is rejected.

The aggregate tables never silently drop failures:

- `cases.parquet` has one ordered row per requested case;
- `failures.parquet` has one row per failed/cancelled case with sanitized stage,
  code, and summary; and
- `review_queue.parquet` has one row per warning/error QC flag, including
  orientation repair. Informational flags remain in the immutable case
  manifest and stage QC JSON but do not request adjudication.

Default runs include same-name CSV mirrors of all three aggregates.
Regeneration uses the output setting stored with the run, so it cannot silently
change the run's selected artifact set.

Regenerate them deterministically with:

```bash
bodycomposition results aggregate output/runs/<run-id> --json
bodycomposition results review-queue output/runs/<run-id> --json
```

## Scientific tables

`slices.parquet`, `vertebrae.parquet`, `summaries.parquet`,
`signature.parquet`, and `hu_distributions.parquet` use measurement schema
`3.5.0`; their fields and units are
documented in [measurements.md](measurements.md). The signature is a
two-component, identity-linked view: `signature.parquet` preserves
translation-only fixed-millimetre anatomy, physical scale, coverage, and null
unscanned bins; `hu_distributions.parquet` preserves analyzed-volume native
compartment attenuation without repeating global bins across longitudinal rows.
Its four fixed channel slots remain present with reason `missing_compartment`
when the selected backend has no homologous native source.

Parquet is the authoritative representation. It preserves Arrow field types,
nullable semantics, embedded table/identity metadata, and strict reader
validation. Default CSV files are deterministic UTF-8, comma-separated mirrors
with the same row and column order, a decimal point, no index, and blank missing
values. List-valued fields are compact JSON arrays. CSV does not carry Parquet
schema metadata and is therefore a convenience analysis export rather than a
replacement result contract.

Create CSV copies later from one verified Parquet-only case without modifying
the immutable source bundle:

```bash
bodycomposition results export-csv \
  output/runs/<run-id>/cases/<case-id>/<analysis-id>/case_manifest.json \
  --output csv/<case-id>
```

## Schemas as package data

The JSON schemas in `BodyComposition/schemas/` ship in the wheel. The complete
strict configuration schema is generated by
`BodyComposition.configuration_schema()` because its defaults and enumerations
are derived from the same implementation used for validation.
