

# BodyComposition

BodyComposition is a research pipeline for reproducible CT body-composition
analysis. It accepts one 3-D NIfTI/DICOM CT or a directory of cases and produces
vertebral-body localization, anatomical compartment masks, HU-filtered tissue
measurements, longitudinal summaries, quality-control artifacts, and optional
one-page PDF reports.

![BodyComposition overview showing sagittal vertebral-body localization, longitudinal tissue areas, an axial tissue overlay, and native-compartment HU distributions](docs/assets/pipeline-overview.png)

*Example based on CT-ORG volume 0: Rister et al. (2019), [The Cancer Imaging
Archive](https://doi.org/10.7937/TCIA.2019.TT7F4V7O),
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The overlays and
measurements are automated outputs, not dataset ground truth. See the
[complete report](docs/assets/pipeline-report-example.png) and
[asset attribution](docs/assets/README.md).*

Version `1.0.0rc1` is a breaking pre-release. Python 3.11 and
[uv](https://docs.astral.sh/uv/) are required.

## Quick start

```bash
git clone https://github.com/fohofmann/BodyComposition.git
cd BodyComposition
uv sync --frozen

# Model weights are downloaded from their pinned original sources.
uv run bodycomposition models sync
uv run bodycomposition doctor --json

# One CT file, one DICOM series, or a converted/DICOM cohort directory
uv run bodycomposition analyze /absolute/path/to/CT.nii.gz --json
```

The final command prints `manifest_path`, `execution_status`, `qc_status`, and
`manual_review_required`. Validate the immutable result before using it:

```bash
uv run bodycomposition results inspect /path/to/case_manifest.json --json
```

The standard command uses automatic CPU/CUDA selection and writes below
`./output`. Inference never downloads a model and never falls back to another
backend. Use an explicitly pseudonymous `--case-id` only when needed;
otherwise a content-based identifier is generated.

## Default workflow

| Stage | Released default |
| --- | --- |
| Orientation | [CTDeepRot](https://github.com/JakubicekRoman/CTDeepRot) assessment; metadata is retained unless a supported safe repair is justified |
| Vertebral bodies | Upstream [SPINEPS 2.0.0](https://github.com/Hendrik-code/spineps/tree/ad622b87d9e4b81fb6a88df2a8050bd40f7046d5) with its [VERIDAH](https://arxiv.org/abs/2601.14066) labeling model and [TPTBox 0.7.5](https://github.com/Hendrik-code/TPTBox/tree/acaaf16f74fb0fe8fc555b23cf4e0230efc49753)/[VibeSeg](https://github.com/robert-graf/VIBESegmentator) |
| Compartments | [BodyCompositionCT-ResEncL](https://huggingface.co/fhofmann/BodyCompositionCT-ResEncL/tree/b355aa7254f5307b6d18005dfbabae8c439d24fb) |
| Anthropometry | Minimum waist over the observed T10–L5 range, sacral pelvic maximum, and their ratio |
| Measurements | Native compartments plus the declared SM −29 to 150 HU and adipose −190 to −30 HU tissue views |
| Export | Typed Parquet tables, CSV mirrors, NIfTI masks, JSON provenance/QC, optional PDF |

Only vertebral corpora—not posterior elements—feed downstream measurements.
Model weights remain outside Git, packages, and containers. See
[models and licenses](docs/models.md), [labels](docs/labels.md), and
[pipeline geometry](docs/pipeline.md).

## Common workflows

| Goal | Command or next step |
| --- | --- |
| Analyze one CT or a directory of cases | `uv run bodycomposition analyze INPUT --json` |
| Analyze only one DICOM series | add `--series SERIES_INSTANCE_UID` |
| Convert DICOM for transfer | `uv run bodycomposition convert /path/to/dicom CT.nii.gz` and transfer the adjacent `CT.bodycomposition.json` sidecar too |
| Convert a DICOM cohort for transfer | `uv run bodycomposition convert /path/to/import /path/to/converted` |
| Run L3-only on limited hardware | synchronize and analyze with `--low-resource` |
| Use the optional BOA tissue backend | run `models sync --tissue-backend boa`, then analyze with `--tissue-backend boa` |
| Enable PDF reports | generate a configuration and set `reporting.enabled: true` |
| Run a governed or scheduler cohort | `uv run bodycomposition batch cohort.json` |
| Omit CSV mirrors | add `--no-csv`; Parquet remains canonical |

Detailed commands and constraints are in the [DICOM](docs/dicom.md),
[low-resource](docs/low-resource.md), [configuration](docs/config.md), and
[execution](docs/execution.md) guides.

### Low-resource L3 analysis

```bash
uv run bodycomposition models sync --low-resource
uv run bodycomposition doctor --low-resource --json
uv run bodycomposition analyze CT.nii.gz --low-resource --json
```

This preset uses the ResEncM vertebral-body and tissue models and restricts
tissue inference to an L3-centred z region without resizing the axial field of
view. It is an L3 phenotype, not a faster equivalent of the full longitudinal
workflow.

### Python API

```python
from BodyComposition import analyze, inspect_result

result = analyze("CT.nii.gz")

verified = inspect_result(result.manifest_path)
```

`analyze` returns a case result for one resolved CT and a batch result for a
directory containing several cases. `analyze_case`, `analyze_batch`, and
`convert_dicom` remain available when the caller wants an explicit contract.

## Main outputs

Successful cases are immutable, content-addressed bundles under
`output/runs/<run-id>/cases/<case-id>/<analysis-id>/`.

| Location | Purpose |
| --- | --- |
| `case_manifest.json` | Authoritative status, identities, provenance, QC, and artifact hashes |
| `masks/vertebral_bodies.nii.gz` | Canonical vertebral corpora and sacral body |
| `masks/tissue_compartments.nii.gz` | Model-native anatomical output |
| `masks/tissue_labels.nii.gz` | Default HU-filtered tissue view |
| `tables/slices.parquet` | Complete physical per-slice measurements |
| `tables/vertebrae.parquet` | Three physical bins per detected vertebral territory |
| `tables/summaries.parquet` | Case-level L3 and anthropometric summaries |
| `tables/signature.parquet` | Fixed 100-by-20-mm vertebral-reference signature |
| `tables/hu_distributions.parquet` | Whole-analyzed-volume native-compartment 5-HU distributions |
| `reports/case_report.pdf` | Optional one-page research/QC report |

CSV mirrors of all canonical tables are written by default for convenient use
in R, spreadsheets, and simple scripts. Parquet remains the typed source of
truth. Preserve coverage, missingness, backend identity, alignment confidence,
and QC flags in downstream analyses.

## Containers and clusters

The Linux `amd64`/`arm64` release image is the ordinary container entry point:

```bash
docker pull ghcr.io/fohofmann/bodycomposition:1.0.0rc1
```

Verify that the selected release tag is available before deployment. Mount
model weights at `/models`, inputs read-only at `/input`, and results at
`/output`; see [containers](docs/containers.md) and
[parallel execution](docs/execution.md).

## Performance

Historical measurements are retained in [performance observations](docs/performance.md),
but are not an accelerator ranking. The frozen current default will be
benchmarked again with optional task 297 disabled.

## Documentation

| I want to… | Read |
| --- | --- |
| get oriented | [documentation index](docs/README.md) and [pipeline](docs/pipeline.md) |
| set up or change models | [models and licensing](docs/models.md) |
| analyze or pre-stage DICOM | [DICOM](docs/dicom.md) |
| understand labels and tissue definitions | [labels](docs/labels.md) and [tissue definitions](docs/tissue_definitions.md) |
| interpret tables and signatures | [measurements](docs/measurements.md) and [output schema](docs/output-schema.md) |
| review QC or PDF reports | [QC](docs/qc-review.md) and [reporting](docs/reporting.md) |
| configure or troubleshoot a run | [configuration](docs/config.md) and [troubleshooting](docs/troubleshooting.md) |
| run cohorts, containers, or Slurm | [reproducibility](docs/reproducibility.md), [containers](docs/containers.md), and [execution](docs/execution.md) |
| understand limitations | [known limitations](docs/known-limitations.md) |

Developer setup and release checks are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation, license, and use

Publications should cite the software release, every enabled model listed in
the run provenance, and:

> Hofmann FO, Heiliger C, Tschaidse T, et al. Validation of body composition
> parameters extracted via deep learning-based segmentation from routine
> computed tomographies. *Scientific Reports*. 2025;15:11909.
> [doi:10.1038/s41598-025-96238-6](https://doi.org/10.1038/s41598-025-96238-6)

The paper validates the published measurement approach, not every optional
backend in this release. BodyComposition source is Apache-2.0. External code,
models, fonts, and data retain their own terms; see [CITATION.cff](CITATION.cff)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

BodyComposition is research software, not a medical device. It is not intended
for diagnosis, treatment, or patient management; outputs may contain errors
and require independent validation and human review.

Parts of this code were implemented using Codex and GPT-5.6.
