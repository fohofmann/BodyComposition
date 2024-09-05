# Configuration File

Frequently changing parameters can be adapted using configuration files or dictionaries. The pipeline is loading
- `config/config.yaml`, and
- the `config/*.yaml` file with the same name as the current pipeline (e.g., `BodyCompositionFast.yaml`), and
- the argument `config`, which can either be a path to a file (e.g. `bodycomposition --config path/to/new/config.yaml`) or a dictionary (e.g. `bodycomposition --config '{"segmentation": {"save_label": True}"}'`).

Parameters are replaced in the following order: `config/config.yaml` < `config/*.yaml` < `--config` argument. This means that parameters in the `--config`/`-c` argument will overwrite parameters defined in the other files.

## Paramenters

### Paths
- `workspace`: Path to the workspace where the data (e.g., the directories `labels`, `masks`, `exports`) is stored. The path can include placeholders for `method` (= name of the current pipeline), `filter` (= regex filter used for input filtering), and `timestamp` (= int time). If `None`, the parent directory of the input data is used.
- `logs`: Path to the log file. The path can include placeholders for `method` (= name of the current pipeline), `filter` (= regex filter used for input filtering), and `timestamp` (= int time).
- `weights`: Path to the weights of the nnU-Net models. For `totalsegmentator`, the path should be the directory that includes all different models ("tasks" in TotalSegementator), as TotalSegmentator identifies the subdirectory by itself.
- `totalsegmentator_config`: Path to the directory that includes the TotalSegmentator configuration file, including your license number.

### Logging
`logging_level`: Logging level for the logging file and console. The levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.

### Run
- `reset`: If `True`, all outputs are removed at initialization.
- `skip`: If `True`, segmentations and mask generation are skipped if already present.
- `timeout`: Timeout in seconds for **each case** in the pipeline. Can be used to prevent the pipeline from getting stuck on a single case.

### Segmentation
- `save_label`: If `True`, the segmentation labels are saved. If `False`, the labels are just saved to the temporary pipeline memory.

### Vertebrae

- `save_mask`: If `True`, the vertebrae masks are saved. If `False`, the masks are just saved to the temporary pipeline memory.
- `min_voxels_per_vertebra`: Minimum number of voxels per vertebra. If (the dominating) vertebra in a slice has less voxels, the label is ignored. This reduces the number of artifacts.
- `fill_undefined_levels`: If `True`, slices between vertebra, in which no dominating vertebra was detected, are filled considering the neighbouring vertebrae in stepwise growing windows along the cranio-caudal axis.
- `correct_monotonicity`: If `True`, the labels are corrected to be monotonically increasing along the cranio-caudal axis. To do so, the labels are corrected by using stepwise growing windows along the cranio-caudal axis.
- `correct_monotonicity_max_windowsize`: Maximum window size for the monotonicity correction.
- `center_of_mass`: If `True`, the center of mass of each vertebra is calculated and returned in the output.
- `deprioritize_labels`: We determine the *dominating vertebra* in each slice, which is the vertebra with the highest number of voxels in the slice. Due to its size, the cranial parts of the sacrum can dominate the lower lumbar vertebrae. Therefore, this option allows to deprioritize the sacrum.

### Tissue
- `save_mask`: If `True`, the [tissue masks](labels.md) are saved. If `False`, the masks are just saved to the temporary pipeline memory.

#### Filter: HU Denoise
- `filter_outliers`: If `True`, outliers are clipped. If `False`, no clipping is performed.
- `filter_outliers_range`: Range for clipping outliers, e.g. metal implants or noisy outliers [-1024, 3071]
- `filter_median`: If `True`, a median filter is applied. If `False`, no median filter is applied.
- `filter_median_kernel`: Kernel size for the median filter, e.g. [3,3,1] (RAS+).

#### Filter: IMAT
- `filter_hu`: If `True`, IMAT is determined by using a HU range of voxels within the [muscle compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining IMAT, e.g. [-190, -30].
- `filter_size`: If `True`, the IMAT (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Size for the 2D filter, e.g. 10 mm^2 (calculated based on voxel spacing). Only areas larger than this are considered as IMAT.
- `filter_size_3D`: Size for the 3D filter, e.g. 80 mm^3 (calculated based on voxel spacing). Only volumes larger than this are considered as IMAT.

#### Filter: SM
- `filter_hu`: If `True`, SM is determined by using a HU range of voxels within the [muscle compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining SM tissue, e.g. [-29, 150].
- `filter_size`: If `True`, the SM (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Size for the 2D filter, e.g. 10 mm^2 (calculated based on voxel spacing). Only areas larger than this are considered as SM.
- `filter_size_3D`: Size for the 3D filter, e.g. 80 mm^3 (calculated based on voxel spacing). Only volumes larger than this are considered as SM.

#### Filter: VAT
- `filter_hu`: If `True`, VAT is determined by using a HU range of voxels within the [visceral compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining VAT, e.g. [-190, -30].
- `filter_size`: If `True`, the VAT (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Size for the 2D filter, e.g. 10 mm^2 (calculated based on voxel spacing). Only areas larger than this are considered as VAT.
- `filter_size_3D`: Size for the 3D filter, e.g. 80 mm^3 (calculated based on voxel spacing). Only volumes larger than this are considered as VAT.

#### Filter: SAT
- `filter_hu`: If `True`, SAT is determined by using a HU range of voxels within the [subcutaneous compartment](labels.md). If `False`, the filter is not applied.
- `filter_hu_range`: Range for determining SAT, e.g. [-190, -30].
- `filter_size`: If `True`, the SAT (determined by HU range) is filtered by size. If `False`, the filter is not applied.
- `filter_size_version`: Version of the filter. Options are `2D` and `3D`.
- `filter_size_2D`: Size for the 2D filter, e.g. 10 mm^2 (calculated based on voxel spacing). Only areas larger than this are considered as SAT.
- `filter_size_3D`: Size for the 3D filter, e.g. 80 mm^3 (calculated based on voxel spacing). Only volumes larger than this are considered as SAT.

### Crop
For the fast versions of the pipeline, we segment the vertebrae, and then crop the image to the region of interest for all further analyses. The region of interest can be defined in this section of the configuration file, and is then used by the [CreateBoundingBox](../BodyComposition/actions/crop.py) class. For each region of interest, the following parameters can be defined:

- `roi`: List of labels that define the region of interest.
- `axes`: List of boolean values that define the axes along which the cropping is performed. The order is [left, right, posterior, anterior, inferior, superior].
- `margin`: List of integers that define the margin in mm along each axis. The margin is transformed to pixels later in the pipeline.

```yaml
  L234CranioCaudal:
    roi: [14,15,16] # labels
    axes: [False, False, False, False, True, True] # RAS+: left -> right, posterior -> anterior, inferor -> superior
    margin: [0, 0, 0, 0, 0, 0] # RAS+: left -> right, posterior -> anterior, inferor -> superior
```