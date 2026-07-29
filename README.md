# BodyComposition

BodyComposition is a research pipeline for reproducible analysis of CT body
composition. One validated service drives both the Python API and the
`bodycomposition` command. The default workflow checks CT orientation with
CTDeepRot, identifies vertebral bodies with SPINEPS/VERIDAH, segments tissue
compartments, and exports native-millimetre per-slice and vertebral summaries,
QC artifacts, provenance, and optional one-page PDF reports.

This software is for research use only. It is not a medical device and must
not be used for diagnosis or treatment. Orientation repair, vertebral labels,
segmentations, anthropometry, and derived phenotypes require validation and
human review for the intended cohort.

## Release-candidate status

Version `1.0.0rc1` defines one supported configuration, service, Python API,
and command-line interface. See the [changelog](CHANGELOG.md) for migration
details.

The default ResEncL and selectable ResEncM tissue models are synchronized from
their original public Hugging Face repositories at immutable revisions. Every
required file has a pinned byte size and SHA-256 digest. No model weights are
included in Git, the wheel, the source archive, or the container.

## Local installation

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen
uv run bodycomposition --version
```

The lockfile is the sole dependency source of truth. Do not layer a separate
`pip install` over this environment.

Model assets live outside the project. Set the cache explicitly when desired:

```bash
export BODYCOMPOSITION_MODEL_ROOT="$HOME/.cache/bodycomposition/models"
uv run bodycomposition models list --json
uv run bodycomposition models sync
uv run bodycomposition models verify --json
uv run bodycomposition doctor --json
```

Synchronization uses exact original upstream URLs or immutable repository
revisions, verifies byte size and SHA-256, and installs atomically. Inference
never downloads a missing model and never falls back to another backend. See
[Model setup and licensing](docs/models.md) before hydrating assets.

## Analyze one CT

Input is one explicit 3-D NIfTI file or one DICOM CT series. A DICOM file or a
directory containing exactly one CT series can be analyzed directly:

```bash
uv run bodycomposition analyze ./input/case-001.nii.gz
uv run bodycomposition analyze ./input/case-002-dicom
```

That is the complete standard command. It uses the validated configuration,
selects CPU or CUDA automatically, derives a content-based pseudonymous case
ID, and writes results below `./output`. DICOM conversion preserves the GDCM
physical domain; the orientation stage still performs all orientation
assessment and any supported repair. When a directory contains multiple CT
series, select one explicitly:

```bash
uv run bodycomposition analyze ./input/case-002-dicom \
  --series 1.2.826.0.1.3680043.10.123.456
```

Use other explicit options only when needed:

```bash
uv run bodycomposition analyze ./input/case-001.nii.gz \
  -o ./another-output -c bodycomposition.yaml \
  --case-id case-001 --device cuda --json
```

Supplied case IDs must already be pseudonymous and path-safe. Machine-readable
JSON is written to stdout; logs are written to stderr. Exit codes are `0`
success, `2` usage/configuration, `3` environment or model readiness, and `4`
execution failure.

For conversion without analysis, use the same validated reader:

```bash
uv run bodycomposition convert ./input/case-002-dicom ./input/case-002.nii.gz
```

The converter is not a DICOM de-identification tool and does not detect or
remove burned-in pixel annotations. See [DICOM input and conversion](docs/dicom.md).

## Analyze an ordered batch

```json
{
  "schema_version": "1.0.0",
  "cases": [
    {"case_id": "case-001", "input_path": "input/case-001.nii.gz"},
    {
      "case_id": "case-002",
      "input_path": "input/case-002-dicom",
      "series_uid": "1.2.826.0.1.3680043.10.123.456"
    }
  ]
}
```

Paths are resolved relative to the manifest. Case order is preserved.

```bash
uv run bodycomposition batch cohort.json
```

Each process executes one complete case at a time. Multiple identical commands
can share the filesystem-backed case queue; the launcher assigns GPUs and
starts processes. See [Execution and accelerator policy](docs/execution.md).

## Python API

```python
from BodyComposition import analyze_case, convert_dicom, inspect_result

result = analyze_case("input/case-001.nii.gz")
converted = convert_dicom("input/case-002-dicom", "input/case-002.nii.gz")

assert result.succeeded
verified = inspect_result(result.manifest_path)
tissue_mask = result.output_path / "masks" / "tissue_labels.nii.gz"
```

The API uses the same defaults and `./output` result root as the CLI. Advanced
callers can pass `output_root`, `config`, `case_id`, and `run_id` explicitly;
`config` accepts a `PipelineConfig`, a strict mapping, or a YAML path. Pass
`series_uid` only when a DICOM input contains multiple CT series.

`CaseResult` and `BatchResult` are immutable terminal results. Inspection
validates the manifest schema and every declared artifact size and SHA-256.
The public API never exposes the pipeline's internal action-memory dictionary.

## Container

Build the container locally:

```bash
LOCK_SHA256=$(shasum -a 256 uv.lock | cut -d' ' -f1)
docker build --pull \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  --build-arg LOCK_SHA256="$LOCK_SHA256" \
  -t bodycomposition:1.0.0rc1 .
```

The image runs as UID/GID `10001`, contains no weights, and expects writable
mounts at `/models` and `/output`:

```bash
mkdir -p models output
docker run --rm \
  -v "$PWD/models:/models" \
  bodycomposition:1.0.0rc1 models sync

docker run --rm --gpus all \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  -v "$PWD/models:/models:ro" \
  bodycomposition:1.0.0rc1 analyze \
  /input/case-001.nii.gz \
  --case-id case-001 \
  --device cuda --json
```

On Linux, ensure the mounted writable directories permit UID/GID `10001`.
Validated and pending platform targets are listed in
[known limitations](docs/known-limitations.md).

## Outputs and review

Successful cases are immutable content-addressed bundles below
`runs/<run-id>/cases/<case-id>/<analysis-id>/`. Failed attempts are retained
separately. Canonical tables are:

- `tables/slices.parquet`: one row per physical CT slice;
- `tables/vertebrae.parquet`: native vertebral territories with three physical
  bins per detected vertebra, including T13, L6, and sacrum when present;
- `tables/summaries.parquet`: established case-level summaries;
- `tables/signature.parquet`: 100 fixed 20-mm bins with CSA, mean HU,
  vertebral anchoring, coverage, and alignment confidence; and
- `tables/hu_distributions.parquet`: whole-volume 5-HU distributions and exact
  attenuation summaries for native SM, SAT, aVAT, and tVAT.

Together, `signature.parquet` and `hu_distributions.parquet` form the default
comparison signature. They share the case, run, analysis, and measurement
schema identity; the global histogram rows are not duplicated into every
longitudinal bin.

Only vertebral-body masks feed downstream measurement and reporting. Full
vertebra/posterior-element masks are retained only as optional upstream/QC
artifacts. Manual-review cases appear in `aggregate/review_queue.parquet` and
in optional PDF Notes/QC annotations and batch review summaries.

See the [output schema](docs/output-schema.md), [measurement definitions](docs/measurements.md),
[QC guide](docs/qc-review.md), and [reporting guide](docs/reporting.md).

## Development and release checks

```bash
uv sync --frozen --extra test --extra release
uv run ruff check BodyComposition tests scripts
uv run mypy
uv run pytest
uv run pytest --cov
uv run python scripts/release_checks.py --allow-dirty
```

The final clean-source gate omits `--allow-dirty`. Model integration and the
checksum-pinned public CT test are opt-in because they require external assets
and controlled hardware. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

- [Configuration](docs/config.md)
- [Pipeline and geometry](docs/pipeline.md)
- [DICOM input and conversion](docs/dicom.md)
- [Models, synchronization, and licenses](docs/models.md)
- [Measurements and tissue definitions](docs/measurements.md)
- [Output and manifest schemas](docs/output-schema.md)
- [QC and manual review](docs/qc-review.md)
- [PDF reporting](docs/reporting.md)
- [Reproducible cohort execution](docs/reproducibility.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Model/data cards and known limitations](docs/known-limitations.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Citation and license

BodyComposition source is Apache-2.0. Model weights retain their own terms and
are never relicensed by this project. Cite BodyComposition and every enabled
model listed in the run provenance; see [CITATION.cff](CITATION.cff) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Parts of this code were implemented using Codex and GPT-5.6.
