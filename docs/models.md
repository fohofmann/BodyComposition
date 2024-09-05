# Models

The pipeline is flexible and can be used with different deep-learning models used for spine / vertebral body as well as tissue / body compartment segmentation. The [pipeline](pipeline.md) combines these models with pre- and postprocessing steps. All integrated segmentation models are based on the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework, some in combination with [Residual Encoder presets](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md). An overview of the resources used in the pipeline is also given [here](../BodyComposition/utils/licenses.yaml).

## Spine segmentation

| Model (Repository) | Description | License | Download | Citation |
| --- | --- | --- | --- | --- |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | spine (Task 292) | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [GitHub](https://github.com/wasserth/TotalSegmentator/releases/tag/v2.0.0-weights) / [Zenodo](https://zenodo.org/record/6802358/) | 1 |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | vertebrae_body (Task 302) | [non-commercial](https://backend.totalsegmentator.com/license-academic/) | [backend totalsegmentator](https://backend.totalsegmentator.com/license-academic/) | 1 |
| [Comp2Comp](https://huggingface.co/louisblankemeier/stanford_spine) | spine | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) | [HuggingFace](https://huggingface.co/louisblankemeier/stanford_spine) | 2 |
| [VertebralBodiesCT-ResEncM](https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncM) | vertebral bodies, residual encoder presets M | [CC BY SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [HuggingFace](https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncM) | 3 |
| [VertebralBodiesCT-ResEncL](https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncL) | vertebral bodies, residual encoder presets L| [CC BY SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [HuggingFace](https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncL) | 3 |


## Tissue segmentation

| Model (Repository) | Description | License | Download | Citation |
| --- | --- | --- | --- | --- |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | muscles (Task 294) | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [GitHub](https://github.com/wasserth/TotalSegmentator/releases/tag/v2.0.0-weights) / [Zenodo](https://zenodo.org/record/6802366/) | 1 |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | body (Task 299) | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [GitHub](https://github.com/wasserth/TotalSegmentator/releases/tag/v2.0.0-weights) / [Zenodo](https://zenodo.org/records/7510286) | 1 |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | tissue_types (Task 481) | [non-commercial](https://backend.totalsegmentator.com/license-academic/) | [backend totalsegmentator](https://backend.totalsegmentator.com/license-academic/) | 1 |

## Citations
1. Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024
2. Blankemeier, L., Desai, A., Zambrano Chaves, J.M., Wentland, A., Yao, S., Reis, E., Jensen, M., Bahl, B., Arora, K., Patel, B.N., Lenchik, L., Willis, M., Boutin, R.D., Chaudhari, A.S., 2023. Comp2Comp: Open-Source Body Composition Assessment on Computed Tomography. arXiv preprint arXiv:2302.06568. https://doi.org/10.48550/arXiv.2302.06568
3. Hofmann F.O. et al. Thoracic & lumbar vertebral body labels corresponding to 1460 public CT scans. https://huggingface.co/datasets/fhofmann/VertebralBodiesCT-Labels/
