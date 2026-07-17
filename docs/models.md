# Models

The pipeline is flexible and can be used with different deep-learning models used for spine / vertebral body as well as tissue / body compartment segmentation. The [pipeline](pipeline.md) combines these models with pre- and postprocessing steps. All integrated segmentation models are based on the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework, some in combination with [Residual Encoder presets](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md). An overview of the resources used in the pipeline is also given [here](../BodyComposition/utils/licenses.yaml).

## Orientation assessment

| Model (Repository) | Description | License | Provisioning | Citation |
| --- | --- | --- | --- | --- |
| [CTDeepRot](https://github.com/JakubicekRoman/CTDeepRot) | 2D ResNet18, nine projection channels, 24 proper-rotation classes | BSD-3-Clause code; no separate checkpoint terms found, so redistribution is not assumed | The explicit model-sync command fetches `net2d.pt` from upstream commit `492114b8f9f3a7f058d4e97c0dd3643fb8d39649`, verifies 44,890,653 bytes and SHA-256 `a2feb521cfe49c367c4e594f38fa76ba9982ba009e114fbf15fb2aae06c06d96`, and atomically promotes it to the versioned cache. Inference never downloads, and the checkpoint is not stored in this repository or source distribution. | 4 |

The adapter reuses only the pinned architecture, preprocessing, and rotation
conventions needed by the pipeline. It does not add the CTDeepRot demo package
as a runtime dependency. CTDeepRot is the appropriate and required
anatomy-based model for the default orientation-integrity stage and must be
acknowledged and cited for analyses that use that stage. This dependency
statement does not imply endorsement by the CTDeepRot authors. CTDeepRot
covers proper rotations, not reflections;
its output is therefore combined with explicit safety gates and review states.
The raw model class is calibrated to the adapter's explicit SimpleITK LPS to
model `(y, x, z)` boundary: raw class 0 is the pinned upright reference, while
reports and correction decisions expose rotations relative to that reference
(group identity class 6 means agreement with the metadata).

## Spine segmentation

| Model (Repository) | Description | License | Download | Citation |
| --- | --- | --- | --- | --- |
| [SPINEPS 2.0.0](https://github.com/Hendrik-code/spineps) | Default CT semantic and instance spine segmentation | Apache-2.0 code; model archives synchronized from their original upstream releases and not redistributed | Exact pinned release archives via `bodycomposition_download_models --model SPINEPS-VERIDAH` | 5 |
| [VERIDAH](https://arxiv.org/abs/2601.14066) | Default anomaly-aware native vertebral labeling, supplied as SPINEPS `ct_labeling` | SPINEPS Apache-2.0 code; model archive synchronized from its original upstream release and not redistributed | Exact pinned SPINEPS archive; not a separate Python package | 6 |
| [VIBESegmentator Dataset100](https://github.com/robert-graf/VIBESegmentator) via [TPTBox 0.7.5](https://github.com/Hendrik-code/TPTBox) | Explicit crop segmentation required by the pinned CT semantic model | Apache-2.0 projects; Dataset100 synchronized from its original upstream release and not redistributed; TPTBox wheel metadata discrepancy retained as non-blocking provenance | Three exact pinned archives via the same model-sync command | 7 |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | spine (Task 292) | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [GitHub](https://github.com/wasserth/TotalSegmentator/releases/tag/v2.0.0-weights) / [Zenodo](https://zenodo.org/record/6802358/) | 1 |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | vertebrae_body (Task 302) | [non-commercial](https://backend.totalsegmentator.com/license-academic/) | [backend totalsegmentator](https://backend.totalsegmentator.com/license-academic/) | 1 |
| [Comp2Comp](https://huggingface.co/louisblankemeier/stanford_spine) | spine | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) | [HuggingFace](https://huggingface.co/louisblankemeier/stanford_spine) | 2 |
| [VertebralBodiesCT-ResEncM](https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncM) | vertebral bodies, residual encoder presets M | [CC BY SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [HuggingFace](https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncM) | 3 |
| [VertebralBodiesCT-ResEncL](https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncL) | vertebral bodies, residual encoder presets L| [CC BY SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [HuggingFace](https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncL) | 3 |


## Tissue segmentation

| Model (Repository) | Description | License | Download | Citation |
| --- | --- | --- | --- | --- |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | muscles (Task 294) | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [GitHub](https://github.com/wasserth/TotalSegmentator/releases/tag/v2.0.0-weights) / [Zenodo](https://zenodo.org/record/6802366/) | 1 |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | tissue_types (Task 481) | [non-commercial](https://backend.totalsegmentator.com/license-academic/) | [backend totalsegmentator](https://backend.totalsegmentator.com/license-academic/) | 1 |

## Measurement support

The canonical `BodyComposition` pipelines pin `TotalSegmentator==2.15.0` for
optional measurement stage support masks. The default body/trunk envelope is derived from
the existing tissue segmentation and needs no additional model. The upstream
`body` and `total` tasks remain explicitly selectable support backends. The
official project lists both tasks as openly available for any usage under
Apache-2.0. This does not change the separate terms for other subtasks such as
`tissue_types` and `vertebrae_body`.

| Model (Repository) | Description | License | Provisioning | Citation |
| --- | --- | --- | --- | --- |
| No additional model | `tissue_segmentation_envelope_v1`: smoothed body/trunk envelope from existing tissue labels (default) | BodyComposition implementation | No weights | — |
| [TotalSegmentator 2.15.0](https://github.com/wasserth/TotalSegmentator) | `body` task 299: explicit alternative trunk and extremity labels | Apache-2.0 open task | Exact `v2.0.0-weights` upstream archive through `bodycomposition_download_models --model TotalSegmentator-body`; archive and required extracted files are checksum-verified | 1 |
| [TotalSegmentator 2.15.0](https://github.com/wasserth/TotalSegmentator) | fast `total` task 297; BodyComposition consumes only bilateral hip and rib labels for mid-waist landmarks | Apache-2.0 open task | Exact `v2.0.0-weights` upstream archive through `bodycomposition_download_models --model TotalSegmentator-total-fast`; archive and required extracted files are checksum-verified | 1 |

Inference never downloads these assets. They live only in the configured,
mounted model directory and are not included in the repository, Python
distribution, or container image.

Task 297 intentionally runs without TotalSegmentator's `roi_subset` option:
that upstream path invokes an additional rough task-298 model. The package
instead runs the pinned fast model once and filters semantically in the
landmark adapter. A cache-only inference guard rejects any unexpected upstream
model request rather than downloading it.

## Citations
1. Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024
2. Blankemeier, L., Desai, A., Zambrano Chaves, J.M., Wentland, A., Yao, S., Reis, E., Jensen, M., Bahl, B., Arora, K., Patel, B.N., Lenchik, L., Willis, M., Boutin, R.D., Chaudhari, A.S., 2023. Comp2Comp: Open-Source Body Composition Assessment on Computed Tomography. arXiv preprint arXiv:2302.06568. https://doi.org/10.48550/arXiv.2302.06568
3. Hofmann F.O. et al. Thoracic & lumbar vertebral body labels corresponding to 1460 public CT scans. https://huggingface.co/datasets/fhofmann/VertebralBodiesCT-Labels/
4. Jakubicek R, Vicar T, Chmelik J. A Tool for Automatic Estimation of Patient Position in Spinal CT Data. IFMBE Proceedings. 2021;80:51–56. https://doi.org/10.1007/978-3-030-64610-3_7
5. Möller H, Graf R, Schmitt J, et al. SPINEPS—automatic whole spine segmentation of T2-weighted MR images using a two-phase approach to multi-class semantic and instance segmentation. European Radiology. 2024. https://doi.org/10.1007/s00330-024-11155-y
6. Möller H, Schoen H, Graf R, et al. VERIDAH: Solving Enumeration Anomaly Aware Vertebra Labeling across Imaging Sequences. arXiv:2601.14066. 2026.
7. Graf R, Platzek P, Riedel EO, et al. VIBESegmentator: full body MRI segmentation for the NAKO and UK Biobank. European Radiology. 2025. https://doi.org/10.1007/s00330-025-12035-9
