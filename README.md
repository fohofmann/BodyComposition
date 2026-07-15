# BodyComposition v0.3
This package combines existing models of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) or the [Comp2Comp pipeline](https://huggingface.co/louisblankemeier/stanford_spine) or [modified models](https://huggingface.co/fhofmann) (based on labels from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/) and [VerSe](https://github.com/anjany/verse)), with pre- and postprocessing steps to calculate the cross-sectional areas of skeletal muscle (SM<sub>total</sub>), psoas muscle (SM<sub>psoas</sub>), visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT) and intermuscular adipose tissue (IMAT) from routine computed tomography (CT) scans.

## Installation

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/fohofmann/BodyComposition.git
   cd BodyComposition
   ```
2. Create a separate environment and install the project. Local development uses `uv`:
   ```bash
   uv venv
   uv pip install -e .
   ```
3. Make sure the required model weights are available.

## Model weights

This pipeline uses model weights derived from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for tissue segmentation, and model weights from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) or [Comp2Comp](https://huggingface.co/louisblankemeier/stanford_spine) or [an modified model](https://huggingface.co/fhofmann), based on labels from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/) and [VerSe](https://github.com/anjany/verse), for spine segmentation. The pipeline combines these models with pre- and postprocessing steps. Details on the models can be found in [docs/models.md](docs/models.md), and how to combine them in [docs/pipeline.md](docs/pipeline.md).

**Pipelines that use TotalSegmentator require the `tissue_types` and, for some configurations, `vertebrae_body` tasks. We do not provide these weights directly; obtain and use them under the applicable TotalSegmentator license.**

1. Set your TotalSegmentator license.
  - **If you are already using TotalSegmentator in your current environment**, and have set a license, skip this.
  - **If you have not obtained a license so far**, you have to do so using the [provided form](https://github.com/wasserth/TotalSegmentator#subtasks) and [set your license](https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#other-commands) (e.g. `totalseg_set_license -l aca_12345678910`). *Please note and respect TotalSegmentator's license conditions, especially regarding the non-commercial use!*
  - **If you have a license already available in an other environment**, you can also provide it manually by editing `config/config.yaml`: Set `paths/totalsegmentator_config` to the home directory of TotalSegmentator that includes TotalSegmentator's configuration file (e.g., `./models/totalsegmentator_config`).

2. Set TotalSegmentator model weights path.
  - **If you are already using TotalSegmentator in your current environment**, skip this. As we use TotalSegmentator's python API, the pipeline should find TotalSegmentators configuration file and model weights automatically.
  - **Alternativly, you can provide paths of the model weights manually** by editing `config/config.yaml` and setting `paths/weights/totalsegmentator` to the nnUNet_results directory used by TotalSegmentator (e.g., `./models/`).

3. Set other models weights path.
  - By default, the pipeline uses the `./models` directory to store the model weights other than TotalSegmentator. You can set individual paths by editing `config/config.yaml` and setting `paths/weights`. 

4. Download model weights.
  - **If you use TotalSegmentator models only,** you can skip this. TotalSegmentator will download the models automatically, if not available.
  - **Alternatively, you can download the model weights using `bodycomposition_download_models`**:
    ```bash
    bodycomposition_download_models --pipeline BodyCompositionFast
    ```
    This script will download the models and store them in the directories as defined in `config/config.yaml`. You can download single models (using `--model`), or all models required for a specific pipeline (using `--pipeline`).

The orientation-enabled pipelines also require the approximately 43 MiB
CTDeepRot 2D checkpoint. It is fetched automatically from a pinned upstream
commit on first use and accepted only when its SHA-256 digest matches. To
hydrate it ahead of an offline run, use:

```bash
bodycomposition_download_models --model CTDeepRot-2D
```

## Model-free checks

The geometry, measurement, action-contract, configuration, CLI, and synthetic-pipeline tests do not download model weights:

```bash
uv pip install -e '.[test]'
uv run python -m pytest -q
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
- `--method` / `-m`: Name of pipeline method to be run, as defined in the [pipeline_registry.py](BodyComposition/pipeline_registry.py). Currently available options are described in [docs/pipeline.md](docs/pipeline.md). Default pipeline is `BodyCompositionFast`, which uses TotalSegmentator for tissue segmentation and [an modified model](https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncM), based on labels from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/) and [VerSe](https://github.com/anjany/verse), for vertebral body segmentation.

*`bin/run_batch.py` is just an command line access point to `python_api.py`. You can also use this API directly from your scripts. For details, [have a look at the file](BodyComposition/python_api.py).*

## More
- A detailed description of the pipeline and the integrated actions can be found in [docs/pipeline.md](docs/pipeline.md).
- A short description of the integrated models (and underlying labels) can be found in [docs/models.md](docs/models.md).
- The differentiation between segmentations (= `labels`) and postprocessed segmentations (= `masks`) are described in [docs/labels.md](docs/labels.md).
- For more information regarding the [config/](config/)-files and available options, see [docs/config.md](docs/config.md).

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
