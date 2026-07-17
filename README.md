# BodyComposition v0.3
This package combines existing models of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) or the [Comp2Comp pipeline](https://huggingface.co/louisblankemeier/stanford_spine) or [modified models](https://huggingface.co/fhofmann) (based on labels from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/) and [VerSe](https://github.com/anjany/verse)), with pre- and postprocessing steps to calculate the cross-sectional areas of skeletal muscle (SM<sub>total</sub>), psoas muscle (SM<sub>psoas</sub>), visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT) and intermuscular adipose tissue (IMAT) from routine computed tomography (CT) scans.

## Installation

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/fohofmann/BodyComposition.git
   cd BodyComposition
   ```
2. Create the pinned Python 3.11 environment with `uv`:
   ```bash
   uv sync --frozen
   ```
3. Make sure the required model weights are available.

## Model weights

The default vertebral backend is pinned SPINEPS 2.0.0 plus its VERIDAH
`ct_labeling` model. VibeSeg Dataset100, called through pinned TPTBox 0.7.5,
provides the required crop mask. The internal ResEncL and ResEncM vertebral
models remain explicitly selectable. Details are in
[docs/models.md](docs/models.md).

The canonical `BodyComposition` pipelines derive the default body/trunk
envelope from the existing tissue segmentation. TotalSegmentator's open fast
`total` task is used only when rib/iliac landmarks are enabled, and its open
`body` task remains an explicit alternative body-surface backend. The upstream
project lists these tasks as openly available for any usage under Apache-2.0.
Legacy pipelines may additionally use separately licensed, non-commercial
tasks such as `tissue_types` or `vertebrae_body`; obtain and use those only
under their applicable terms. Disabling landmarks while retaining the default
tissue envelope removes TotalSegmentator inference from the canonical run.

1. Configure a TotalSegmentator license only when selecting one of its
   separately licensed subtasks. The optional measurement stage `body` and `total` paths do
   not require that license. If needed, use the
   [upstream instructions](https://github.com/wasserth/TotalSegmentator#subtasks)
   and point `paths/totalsegmentator_config` to the directory containing the
   resulting TotalSegmentator configuration.

2. Set TotalSegmentator model weights path.
  - **If you are already using TotalSegmentator in your current environment**, skip this. As we use TotalSegmentator's python API, the pipeline should find TotalSegmentators configuration file and model weights automatically.
  - **Alternativly, you can provide paths of the model weights manually** by editing `config/config.yaml` and setting `paths/weights/totalsegmentator` to the nnUNet_results directory used by TotalSegmentator (e.g., `./models/`).

3. Set other models weights path.
  - By default, the pipeline uses the `./models` directory to store the model weights other than TotalSegmentator. You can set individual paths by editing `config/config.yaml` and setting `paths/weights`. 

4. Download model weights.
  - Download the exact models required by a pipeline using `bodycomposition_download_models`:
    ```bash
    bodycomposition_download_models --pipeline BodyCompositionFast
    ```
    This script stores models in the directories defined in `config/config.yaml`.
    Default inference never downloads SPINEPS, VERIDAH, VibeSeg, CTDeepRot, or
    TotalSegmentator assets implicitly.

The orientation-enabled pipelines also require the approximately 43 MiB
CTDeepRot 2D checkpoint. Synchronize it explicitly from the pinned upstream
commit before inference; the model manager verifies both its byte size and
SHA-256 digest before atomically promoting it into the versioned model cache:

```bash
bodycomposition_download_models --model CTDeepRot-2D
```

Inference never downloads model assets. A missing or modified checkpoint stops
the case with an instruction to run the synchronization command. Set
`BODYCOMPOSITION_CTDEEPROT_CHECKPOINT` only when a verified checkpoint is kept
at a controlled alternative path.

The default vertebral backend additionally requires three pinned SPINEPS model
archives and all three VibeSeg Dataset100 archives (about 3.2 GB combined):

```bash
bodycomposition_download_models --model SPINEPS-VERIDAH
bodycomposition_download_models --model SPINEPS-VERIDAH --check
```

The synchronization is checksum-verified, atomic, and idempotent. Model weights
are fetched from their exact original upstream release URLs and are never
redistributed in the source, wheel, or public container. SPINEPS, VERIDAH,
TPTBox, VibeSeg, and their papers must be acknowledged as described in
`THIRD_PARTY_NOTICES.md`.

With the released defaults, the canonical measurement stage additionally uses
TotalSegmentator task 297 for anatomical mid-waist landmarks. Task 299 is
synchronized only when its alternative body-surface backend is selected. The
model manager honors the active configuration, fetches exact original GitHub
release archives, verifies pinned byte sizes and SHA-256 digests before
extraction, and atomically promotes verified contents into the mounted model
directory:

```bash
bodycomposition_download_models --model TotalSegmentator-body
bodycomposition_download_models --model TotalSegmentator-body --check
bodycomposition_download_models --model TotalSegmentator-total-fast
bodycomposition_download_models --model TotalSegmentator-total-fast --check
```

The task-297 landmark source runs once as the upstream fast `total` model; the
adapter then reads only bilateral hip and rib labels. The pipeline does not use
TotalSegmentator's `roi_subset` path because that path requests an additional
task-298 model. A cache-only guard turns any unexpected model request into a
clear error instead of an inference-time download.

The container uses a mounted model directory. Hydrate it with the same
BodyComposition command; no separate SPINEPS or VibeSeg installation is needed:

```bash
docker run --rm \
  --entrypoint bodycomposition_download_models \
  --mount type=bind,src=/absolute/path/to/models,dst=/app/models \
  bodycomposition:0.3 \
  --pipeline BodyCompositionFast
```

Reuse the same host directory at `/app/models` for every inference container.
The image itself remains weight-free.

## Model-free checks

The geometry, measurement, action-contract, configuration, CLI, and synthetic-pipeline tests do not download model weights:

```bash
uv sync --frozen --extra test
uv run --frozen python -m pytest -q
```

## Opt-in public CT integration test

The repository defines a real-world smoke test using CT-ORG volume 0, licensed
under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The scan is
not bundled with this Apache-2.0 package. The test obtains it on demand from an
immutable mirror revision, checks its exact byte count and SHA-256 digest, and
runs the default `BodyCompositionFast` pipeline twice to verify anatomy/geometry
invariants and safe resume behavior.

Run the test only in a CUDA-capable controlled environment with the pinned model
directories available (for example, in a CUDA-enabled Docker container):

```bash
BODYCOMPOSITION_RUN_REAL_WORLD_TEST=1 \
BODYCOMPOSITION_MODEL_ROOT=/absolute/path/to/models \
BODYCOMPOSITION_TEST_DATA_CACHE=/absolute/path/to/test-cache \
uv run python -m pytest -m real_world tests/test_public_real_world.py -v
```

The model root follows the default cache layout: `Dataset611_BodyComposition`,
`SPINEPS/spineps-veridah-ct-v1`,
`Dataset297_TotalSegmentator_total_3mm_1559subj`, and
`CTDeepRot/492114b8f9f3a7f058d4e97c0dd3643fb8d39649/net2d.pt`. An explicitly
mounted CTDeepRot checkpoint can instead be selected with
`BODYCOMPOSITION_CTDEEPROT_CHECKPOINT`.

To use an already downloaded copy instead of allowing network access, set
`BODYCOMPOSITION_PUBLIC_CT_PATH=/absolute/path/to/ct_org_volume-0_0000.nii.gz`.
The local file is accepted only when its pinned size and digest match. See
[`tests/fixtures/public_ct/README.md`](tests/fixtures/public_ct/README.md) for source,
license, attribution, citations, and the important CT-ORG orientation caveat.

## CT orientation integrity

Every registered analysis pipeline starts with an orientation-integrity stage
by default. It compares the image geometry with CTDeepRot anatomy inference at
the API level, before segmentation or cropping. DICOM-to-NIfTI conversion
preserves reader geometry and does not repair orientation.

CTDeepRot is the appropriate and required anatomy-based rotation model whenever
this default stage is enabled; it is not presented as work developed by the
BodyComposition authors. Analyses using the orientation stage must acknowledge
CTDeepRot and cite Jakubicek, Vicar, and Chmelik as listed below. Disabling
`orientation.enabled` explicitly removes this model-dependent check.

- Matching metadata are retained, including modest obliquity.
- A high-confidence proper-rotation mismatch produces a losslessly permuted or
  flipped derivative using SimpleITK; the source image remains unchanged. The
  anatomically ambiguous SI-preserving 180-degree in-plane class is
  advisory-only by default after a public-cohort false-positive audit.
- Every changed case continues through segmentation but is marked for manual
  review.
- Low-confidence, short-field-of-view, unusually oblique, or otherwise gated
  readable cases keep the metadata interpretation, continue through the
  pipeline, and receive review status.
- Invalid pixels or physical geometry stop that case.

Each enabled run writes `orientation_report.json` and
`orientation_review.png`; changed cases additionally receive
`corrected_input.nii.gz`. The report states one of
`PASS_METADATA_MATCH`, `PASS_METADATA_UNCERTAIN`, `MISMATCH_REPAIRED`,
`MISMATCH_UNCERTAIN`, `HEADER_UNCERTAIN`, or `ORIENTATION_FAILED`.
CTDeepRot models proper rotations only and cannot establish whether a scan is
left-right reflected, so the review image is a QC aid rather than orientation
ground truth. See [the pipeline guide](docs/pipeline.md) and
[configuration guide](docs/config.md).

## Usage

After installation of the pipeline and TotalSegmentator, you can use the pipeline via the command line:

```bash
bodycomposition -i ./data/images -f '^ct_.*\.nii\.gz$' -m BodyCompositionFast
```

The following flags are available:

- `--input` / `-i`: Path to input, either directory (e.g., `data/images`) or datalist file (*.json) or single NiFTI file
- `--filter` / `-f`: Regex string to filter and subset input files (e.g., `'^ct_.*\.nii\.gz$'`)
- `--config` / `-c`: Path to configuration file (*.yaml), or dictionary. Can be used to update the default configuration. For options, see [docs/config.md](docs/config.md).
- `--method` / `-m`: Name of pipeline method to be run, as defined in the [pipeline_registry.py](BodyComposition/pipeline_registry.py). Currently available options are described in [docs/pipeline.md](docs/pipeline.md). `BodyCompositionFast` uses SPINEPS/VERIDAH by default; select `vertebral_bodies_resenc_l` or `vertebral_bodies_resenc_m` through `vertebrae.backend` in a configuration override when required. A backend failure never triggers another model automatically.

*`bin/run_batch.py` is just an command line access point to `python_api.py`. You can also use this API directly from your scripts. For details, [have a look at the file](BodyComposition/python_api.py).*

`BodyComposition` and `BodyCompositionFast` write the canonical full-volume
measurement bundle to `tables/{case_id}/`: `slices.parquet`,
`vertebrae.parquet`, and `summaries.parquet`. The slice table preserves the
native-mm longitudinal profile; the vertebral table contains three physical
bins per detected native level, including sacrum. Read L3 or named-range
convenience views without creating duplicate exports:

```bash
bodycomposition_measurements \
  --tables ./data/output/BodyCompositionFast/tables/case-001 \
  --view l3 \
  --aggregation territory_mean
```

Definitions, units, anatomical territories, missingness, and volume
reconstruction are in
[docs/measurements.md](docs/measurements.md).
The literature review, default SAT/VAT/SM/IMAT/LAMA/NAMA definitions,
rationale, and executable compatibility profiles are in
[docs/tissue_definitions.md](docs/tissue_definitions.md).

Optional one-page PDF research/QC reports can be enabled without changing the
scientific analysis identity:

```yaml
reporting:
  enabled: true
  layout: spine_overview_v1  # or spine_profile_v2
```

Every case report is exactly one A4 landscape page. Batch runs with a common
workspace also produce an ordered combined PDF; a manual-review summary is
inserted as its first page only when at least one canonical review flag is
present. `spine_profile_v2` adds the unsmoothed native-mm stacked tissue-area
profile defined by measurement stage. Reports use only vertebral-body segmentations and the
canonical measurement tables. See [docs/reporting.md](docs/reporting.md) for the
configuration, post-hoc manifest API/CLI, outputs, and QC limitations.

## More
- A detailed description of the pipeline and the integrated actions can be found in [docs/pipeline.md](docs/pipeline.md).
- A short description of the integrated models (and underlying labels) can be found in [docs/models.md](docs/models.md).
- The differentiation between segmentations (= `labels`) and postprocessed segmentations (= `masks`) are described in [docs/labels.md](docs/labels.md).
- Tissue-compartment definitions, raw-HU defaults, evidence, and literature sensitivity profiles are documented in [docs/tissue_definitions.md](docs/tissue_definitions.md).
- For more information regarding the [config/](config/)-files and available options, see [docs/config.md](docs/config.md).
- Optional derived PDF reports and post-hoc collation are documented in [docs/reporting.md](docs/reporting.md).

## Citations

If you use this code, you should cite the following repositories and papers:

[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

> Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024

[CTDeepRot](https://github.com/JakubicekRoman/CTDeepRot)

> Jakubicek, R., Vicar, T., & Chmelik, J. (2021). A Tool for Automatic
> Estimation of Patient Position in Spinal CT Data. IFMBE Proceedings, 80,
> 51–56. https://doi.org/10.1007/978-3-030-64610-3_7

**Depending on the frameworks, [datasets and models](docs/models.md) used, you should also cite the respective sources! The pipeline logs the used resources at initialization. Please reference this repository, when using it.**
