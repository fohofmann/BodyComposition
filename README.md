# BodyComposition v0.1
This package combines existing models of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) or the [Comp2Comp pipeline](https://huggingface.co/louisblankemeier/stanford_spine) or [modified models](https://huggingface.co/fhofmann) (based on labels from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/) and [VerSe](https://github.com/anjany/verse)), with pre- and postprocessing steps to calculate the cross-sectional areas of skeletal muscle (SM<sub>total</sub>), psoas muscle (SM<sub>psoas</sub>), visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT) and intermuscular adipose tissue (IMAT) from routine computed tomography (CT) scans.

## Installation

1. We recommend to install this pipeline in a separate python environment.
2. Navigate to the directory you would like to install the pipeline in.
3. Install the pipeline
   ```b
   git clone https://github.com/fohofmann/BodyComposition.git
   cd BodyComposition
   pip install --no-cache-dir -e .
   ```
4. Make sure, the TotalSegmentator model weights are available:

## Model weights

This pipeline uses model weights derived from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for tissue segmentation, amd model weights from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) or [Comp2Comp](https://huggingface.co/louisblankemeier/stanford_spine) or [an modified model](https://huggingface.co/fhofmann), based on labels from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/) and [VerSe](https://github.com/anjany/verse), for spine segmentation. The pipeline combines these models with pre- and postprocessing steps. Details on the models can be found in [docs/models.md](docs/models.md), and how to combine them in [docs/pipeline.md](docs/pipeline.md).

**BodyComposition v0.2 requires the `tissue_types` (and `vertebrae_body`) tasks from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator). We do not provide these weights directly, instead you must acquire a license available from [J. Wasserthal / TotalSegmentator](https://github.com/wasserth/TotalSegmentator).**

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
    bodycomposition_download_models -pipeline BodyCompositionFast
    ```
    This script will download the models and store them in the directories as defined in `config/config.yaml`. You can download single models (using `-model`), or all models required for a specific pipeline (using `-pipeline`).
  

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

**Depending on the frameworks, [datasets and models](docs/models.md) used, you should also cite the respective sources! The pipeline logs the used resources at initialization. Please reference this repository, when using it.**