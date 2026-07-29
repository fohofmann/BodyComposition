# Low-resource L3 analysis

The `--low-resource` shortcut is a distinct, recorded analysis scope for
machines that cannot efficiently run the default full-CT models:

```bash
bodycomposition models sync --low-resource
bodycomposition doctor --low-resource --json
bodycomposition analyze CT.nii.gz --low-resource
```

It selects the ResEncM vertebral-body and tissue models, disables the optional
full-volume landmark model, and unloads each model before the next stage. The
orientation check and L3 localization still use the complete prepared CT. If
a valid L3 cannot be identified, the case fails explicitly; another level or
the full-CT tissue model is never selected silently.

After localization, the tissue model receives the complete axial field of
view and a z crop spanning the L3 territory plus 120 mm of context on each
side. This context is needed by the three-dimensional nnU-Net; predictions
outside the L3 territory are discarded. Cropping is index-based and does not
resample or resize the CT. nnU-Net performs its normal pinned-spacing
preprocessing, returns the prediction at the cropped input shape, and the
pipeline restores it to the exact orientation-prepared CT geometry.

The exported masks therefore remain aligned with the full CT, while
`body_composition_analysis_available` in `slices.parquet` identifies the L3 slices that
were actually analyzed. Areas and HU values outside that scope are invalid,
not biological zeros. Vertebral and fixed-mm tables retain explicit coverage,
and the native-compartment HU table describes the analyzed L3 volume only. Full
longitudinal profiles, waist, pelvic maximum, and waist-to-pelvic ratios
require the standard full-CT analysis.

Python callers can use the same preset without constructing a configuration:

```python
from BodyComposition import analyze_case

result = analyze_case("CT.nii.gz", low_resource=True)
```

For a reviewed customization, start with `low_resource_config()` and change
only explicit settings such as the model-context distance:

```python
from BodyComposition import PipelineConfig, analyze_case, low_resource_config

config = low_resource_config()
value = config.normalized()
value["analysis"]["l3"]["inference_context_mm"] = 100.0
result = analyze_case("CT.nii.gz", config=PipelineConfig.model_validate(value))
```

## Illustrative runtime

The following existing validation runs used the same public CT
(`512 × 512 × 75`, 5-mm slice spacing), an NVIDIA GB10, CUDA, and isolated
Docker containers:

| Pipeline | Complete case time | Relative time |
| --- | ---: | ---: |
| Standard defaults | 220.1 s | 1.00 |
| `--low-resource` | 52.5 s | 0.24 |

For this case, the low-resource preset was approximately 4.2 times faster,
reducing elapsed time by about 76%. The principal differences were:

- vertebral localization: SPINEPS/VERIDAH took 97.2 s, compared with 33.3 s
  for VertebralBodiesCT-ResEncM;
- tissue segmentation: the five-fold ResEncL model took 30.0 s, compared with
  11.9 s for the L3-cropped, single-model ResEncM run; and
- the standard pipeline spent another 81.6 s on full-volume
  TotalSegmentator landmarks, which the L3 preset does not require.

This is an illustrative single-case measurement from adjacent development
candidates, not a formal benchmark or performance guarantee. The advantage
depends on scan length, slice spacing, hardware, model caching, and storage
speed. More importantly, the outputs are not equivalent: the standard pipeline
provides full longitudinal and anthropometric analysis, whereas the
low-resource preset provides L3-focused tissue phenotyping.
