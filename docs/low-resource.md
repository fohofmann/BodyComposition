# Low-resource L3 analysis

The `--low-resource` shortcut is a distinct, recorded analysis scope for
machines that cannot efficiently run the default full-CT models:

```bash
bodycomposition models sync --low-resource
bodycomposition doctor --low-resource --json
bodycomposition analyze CT.nii.gz --low-resource
```

It selects the ResEncM vertebral-body and tissue models, keeps the optional
full-volume landmark extension disabled, and unloads each model before the
next stage. The orientation check and L3 localization still use the complete
prepared CT. If a valid L3 cannot be identified, the case fails explicitly;
another level or the full-CT tissue model is never selected silently.

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

## Historical observed runtime

Complete-case measurements for the same public 512 × 512 × 75 CT show the
expected within-platform reduction, but the workflows do not produce
equivalent outputs:

| Platform | Standard | Low-resource |
| --- | ---: | ---: |
| Apple M1 Max CPU | 69.4 min | 7.3 min |
| NVIDIA GB10 | 220.1 s | 52.5 s |
| NVIDIA A100 40 GB | 398.6 s | 110.6 s |
| NVIDIA B200 | 329.6 s | 96.4 s |

Scan length, storage, caching, and hardware affect runtime. The standard
pipeline provides full longitudinal and anthropometric analysis; the
low-resource preset provides L3-focused tissue phenotyping. Exact revisions,
memory observations, repeated-run variation, and benchmark conditions are in
[performance.md](performance.md). The platforms used different environments,
so the rows must not be used as an accelerator ranking. Both workflows will be
remeasured from the frozen current default before release.
