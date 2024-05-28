# BodyComposition v0.1
This package combines model weights of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) with pre- and postprocessing steps to calculate the cross-sectional areas of skeletal muscle (SM<sub>total</sub>), psoas muscle (SM<sub>psoas</sub>), visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT) and intermuscular adipose tissue (IMAT) from routine computed tomographies.

## Installation

1. We recommend to install this pipeline in a separate python environment.

2. Navigate to the directory you would like to install the pipeline in.

3. Install the pipeline
   ```b
   git clone https://github.com/fohofmann/BodyComposition.git
   cd BodyComposition
   pip install --no-cache-dir -e .
   ```

4. Make sure the TotalSegmentator model weights are available:

## Model weights

Currently, this repository does not provide own model weights. Instead, we combine model weights of  [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) with pre- and postprocessing steps. BodyComposition v0.1 uses the following [task classes](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py) from TotalSegmentator: `total`, `body`, `vertebrae_body`, and `tissue_types`. The tasks `total` and `body` are [available](https://github.com/wasserth/TotalSegmentator/releases/tag/v2.0.0-weights) under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode), whereas `vertebrae_body` and `tissue_types` require a license available from [J. Wasserthal / TotalSegmentator](https://github.com/wasserth/TotalSegmentator).

- **If you are already using TotalSegmentator in your current environment**, you can skip the following steps. As we use TotalSegmentator's python API, the pipeline should just work.

- **If you have not used Totalsegmentator in your current environment so far**, you have to obtain the model weights. To do so, please refer to [TotalSegmentator](https://github.com/wasserth/TotalSegmentator):

  - Obtain a license using the [provided form](https://github.com/wasserth/TotalSegmentator#subtasks). *Please note and respect TotalSegmentator's license conditions, especially regarding the non-commercial use!*

  - [Set your license](https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#other-commands): `totalseg_set_license -l aca_12345678910`.

  - [Download the model weights](https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#other-commands): 

    ````bash
    totalseg_download_weights -t total
    totalseg_download_weights -t body
    totalseg_download_weights -t vertebrae_body
    totalseg_download_weights -t tissue_types
    ````

- **If you have a license and model weights already available in an other environment**, you can also provide its localization manually by editing `config/general.yaml`:

  - `weights/totalsegmentator` sets the environment variable `TOTALSEG_WEIGHTS_PATH` = path to the nnUNet_results directory used by TotalSegmentator

  - `tseg_config` sets the environment variable `TOTALSEG_HOME_DIR` = path to the home directory of TotalSegmentator, e.g. containing the configuration file 8including your license key)

  This method can also be used to mount a model directory when you are using the provided docker. We can not and do not want to integrate TotalSegmentator's (tissue and vertebral bodies) model weights into the container directly.

## Usage

After installation of the pipeline and TotalSegmentator, you can use the pipeline via the command line:

```bash
bin/RunBatch -i ./data/images -f '^ct_.*\.nii\.gz$' -m SarcopeniaTotalSegmentator
```

The following flags are available:

- `--input` / `-i`: Path to input, either directory (e.g., `data/images`) or datalist file (*.json) or single NiFTI file
- `--filter` / `-f`: Regex string to filter and subset input files (e.g., `'^ct_.*\.nii\.gz$'`)
- `--config` / `-c`: Path to configuration file (*.yaml), or dictionary. Can be used to update the default configuration
- `--method` / `-m`: Name of pipeline method to be run, as defined in the [pipeline_registry.py](BodyComposition/pipeline_registry.py). Currently available options are `SarcopeniaTotalSegmentator` and `SarcopeniaTotalSegmentatorFast` 

*`bin/RunBatch` is just an comand line access point to `python_api.py`. You can also use this API directly from your scripts. For details, [have a look at the file](BodyComposition/python_api.py).*

## More

- A detailed description of the pipeline and it's logic can be found in [docs/pipeline.md](docs/pipeline.md).
- The labels returned as `labels` and `masks` are described in [docs/labels.md](docs/labels.md).
- For more information regarding the [config/](config/)-files and available options, see [docs/config.md](docs/config.md)

## Citations

If you use this code, you cite the following repositories and papers:

[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

> Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024
