# Using BodyComposition with an agent

This file is the entry point for an automated agent asked to run
BodyComposition. Use the public command line or Python API described here.
Do not construct a separate runner from internal action classes.

## Safety

- Process only the CTs and output locations explicitly placed in scope.
- Never upload a patient scan, model output, or identifying metadata to an
  external service.
- BodyComposition accepts one three-dimensional NIfTI CT or one DICOM CT
  series. It does not deidentify DICOM, remove burned-in text, or determine
  whether a scan may be used for a particular study.
- If a DICOM directory contains more than one CT series, do not guess. Report
  the available series and require an explicit Series Instance UID.
- Omit `--case-id` unless the user supplied a path-safe pseudonym. Otherwise
  the pipeline derives a content-based identifier that does not expose the
  input filename.
- Do not commit CTs, DICOM files, masks, model weights, generated results,
  credentials, or identifying paths.

## Prepare the environment

From the repository root:

```bash
uv sync --frozen
uv run bodycomposition models verify --json
uv run bodycomposition doctor --json
```

Continue only when `operational_ready` is `true` and every required model has
`ready: true`. If verified assets are missing, synchronize the pinned assets
from their declared original sources:

```bash
uv run bodycomposition models sync
```

Inference never downloads models. Do not find substitute weights, copy weights
into the package or container, or silently select another backend.

## Analyze one CT

The standard workflow needs only an input:

```bash
uv run bodycomposition analyze /absolute/path/to/scan.nii.gz --json
```

One unambiguous DICOM CT series can be analyzed directly:

```bash
uv run bodycomposition analyze /absolute/path/to/dicom-directory --json
```

Use `-o /absolute/path/to/output` only when another result root is wanted.
Device selection defaults to `auto`.

On success, read `manifest_path`, `execution_status`, `qc_status`, and
`manual_review_required` from the JSON response. Validate the returned bundle:

```bash
uv run bodycomposition results inspect \
  /absolute/path/to/case_manifest.json --json
```

A successful case with `qc_status: review` still requires human review before
research use. Orientation changes, uncertain vertebral enumeration, cropped
measurements, and other review findings must remain visible.

## Pre-stage DICOM for transfer

To convert without running segmentation:

```bash
uv run bodycomposition convert \
  /absolute/path/to/dicom \
  /absolute/path/to/ct.nii.gz
```

The command creates `ct.nii.gz` and `ct.bodycomposition.json`. Transfer both
files together. The NIfTI preserves CT pixels and physical geometry; the
hash-bound sidecar preserves allowlisted technical DICOM provenance. On the
destination machine, use the ordinary analysis command:

```bash
uv run bodycomposition analyze /absolute/path/to/ct.nii.gz --json
```

The sidecar is found and verified automatically. Conversion does not perform
orientation repair.

## Low-resource L3 analysis

Use the low-resource preset only when the user explicitly requests an L3-only
analysis for a memory-limited machine:

```bash
uv run bodycomposition models sync --low-resource
uv run bodycomposition models verify --low-resource --json
uv run bodycomposition doctor --low-resource --json
uv run bodycomposition analyze /absolute/path/to/scan.nii.gz \
  --low-resource --json
```

This mode still assesses orientation and localizes L3 on the complete CT. It
then uses the ResEncM models on an L3-centred z region without resizing the
in-plane image and restores masks to the prepared CT grid. Its measurements
cover L3 only; do not present it as a full longitudinal analysis.

## Batch and scheduler execution

Use `bodycomposition batch` with an input manifest for an ordered collection.
Do not write a private loop around internal pipeline objects.

Independent scheduler tasks may run the same command with `--worker`. The
filesystem queue assigns complete cases without overlap, and a surplus worker
exits when every remaining case is owned by another live worker. The scheduler
controls array size, CPU, memory, time, and visible GPU resources. See
[docs/execution.md](docs/execution.md) for the generic Slurm example.

## Outputs

The manifest inventory is authoritative. Important default masks are:

- `masks/vertebral_bodies.nii.gz`;
- `masks/tissue_compartments.nii.gz`;
- `masks/tissue_labels.nii.gz`; and
- `masks/body_surface.nii.gz`.

Canonical tables are:

- `slices.parquet`;
- `vertebrae.parquet`;
- `summaries.parquet`;
- `signature.parquet`; and
- `hu_distributions.parquet`.

CSV mirrors are written by default for convenient inspection. Parquet remains
the typed source of truth. Use `--no-csv` only when duplicate representations
are not wanted. CSV mirrors can later be created from a verified result:

```bash
uv run bodycomposition results export-csv \
  /absolute/path/to/case_manifest.json \
  -o /absolute/path/to/csv-output
```

The default comparison signature consists of the fixed 100-by-20-mm
vertebral-reference view and whole-analyzed-volume native-compartment 5-HU
distributions. Preserve coverage, alignment confidence, histogram tails, and
missingness in downstream comparisons.

## Python API

```python
from BodyComposition import analyze_case

result = analyze_case("/absolute/path/to/scan.nii.gz")
if not result.succeeded:
    raise RuntimeError(result.failure)

tissue_mask = result.output_path / "masks" / "tissue_labels.nii.gz"
vertebral_bodies = result.output_path / "masks" / "vertebral_bodies.nii.gz"
```

Use `analyze_batch` for an ordered collection. Advanced callers may pass
`output_root`, `config`, `case_id`, `run_id`, and, for ambiguous DICOM input,
`series_uid` explicitly.

## Configuration and geometry

The code-defined defaults are the supported standard workflow. For an explicit
override:

```bash
uv run bodycomposition config show-default -o bodycomposition.yaml
uv run bodycomposition config validate bodycomposition.yaml --json
uv run bodycomposition analyze /absolute/path/to/scan.nii.gz \
  -c bodycomposition.yaml --json
```

Never guess configuration keys, reorient an input independently, transpose a
SimpleITK array implicitly, relabel an anatomical variant, or fall back to
another model. SimpleITK metadata and indices use `x-y-z`; arrays returned by
SimpleITK use `z-y-x`. The orientation-prepared image owns the physical domain
for all downstream stages.

More detail is available in [docs/pipeline.md](docs/pipeline.md),
[docs/models.md](docs/models.md), [docs/dicom.md](docs/dicom.md),
[docs/output-schema.md](docs/output-schema.md), and
[docs/qc-review.md](docs/qc-review.md).
