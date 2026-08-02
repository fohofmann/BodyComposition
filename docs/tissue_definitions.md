# CT compartments and tissue definitions

This document defines the tissue contract for measurement schema `3.4.0`.
The default is intentionally small: preserve the model-native anatomical
compartments, measure their raw CT attenuation, and add only the conventional
muscle and adipose HU filters needed for broad clinical comparability.

## Input contract

The measurement stage expects a calibrated three-dimensional CT in Hounsfield
units. DICOM CT is read through SimpleITK/GDCM, including the modality
rescale encoded by the source series. A NIfTI input is expected to retain the
CT voxel values in HU. The package cannot reconstruct lost calibration from a
NIfTI file whose values were normalized, windowed, or otherwise transformed.

Contrast phase, scanner, reconstruction, dose, and slice thickness can change
the interpretation and comparability of attenuation. They do not make the
voxel values cease to be HU. The pipeline therefore stores measured HU and
available acquisition provenance; it does not guess a harmonized value or
silently discard a scan because optional DICOM fields are unavailable.

## Anatomical source

`masks/tissue_compartments.nii.gz` is the immutable model output:

| Label | Compartment |
| ---: | --- |
| 1 | SM, the complete skeletal-muscle compartment |
| 2 | bone |
| 3 | subcutaneous adipose compartment (SAT) |
| 4 | abdominal visceral adipose compartment (aVAT) |
| 5 | thoracic visceral adipose compartment (tVAT) |
| 6 | heart |
| 7 | lung |

Every native compartment uses a `<name>_compartment_*` export prefix and
receives per-slice voxel count, CSA, mean HU, validity, and missing-reason
columns. In particular, `sm_compartment_mean_hu` is the continuous attenuation
of the complete predicted muscle compartment and is the default measure for
analyses of fatty muscle change.

## Default consensus profile

The default profile is
`consensus_hu_muscle_m29_150_adipose_m190_m30_v1`. It has exactly two
inclusive raw-HU rules: -29 to 150 HU for the SM compartment and -190 to -30
HU for adipose compartments. Those two rules create these named views:

| Output prefix | Anatomical support | Inclusive HU range |
| --- | --- | ---: |
| `skeletal_muscle_tissue_hu_m29_150` | SM | -29 to 150 |
| `sat_tissue_hu_m190_m30` | SAT | -190 to -30 |
| `avat_tissue_hu_m190_m30` | aVAT | -190 to -30 |
| `tvat_tissue_hu_m190_m30` | tVAT | -190 to -30 |
| `vat_tissue_hu_m190_m30` | aVAT union tVAT | -190 to -30 |

The VAT tissue union is convenient for conventional summaries and the
VAT-to-SAT tissue ratio. It is derived from the filtered aVAT and tVAT views,
not a third HU rule. It is not duplicated in the default longitudinal
signature, which retains aVAT and tVAT separately.

The default applies no denoising, hole filling, connected-component removal,
or minimum-size rule to these tissue definitions. Mean HU is always measured
from the untouched prepared CT, even when an optional profile uses a filtered
image to classify a mask.

The default does not calculate IMAT, LAMA, NAMA, a binary myosteatosis class,
or a sarcopenia cutpoint. Those terms have heterogeneous definitions and are
not required to study muscle quality: the complete-compartment mean HU,
conventional muscle-tissue CSA/HU, and their longitudinal distribution remain
available as continuous measurements.

## Configuration of an additional definition

Additional definitions belong in a named profile under
`measurements.tissue_definitions`. A minimal example is:

```yaml
measurements:
  tissue_profile_id: study_specific_muscle_filter_v1
  tissue_definitions:
    filter_muscle_tissue_hu_m190_m30:
      enabled: true
      source_labels: [SM]
      hu_range: [-190, -30]
```

This creates explicitly named measurement columns but does not add a new
model-native label or change the standard visualization mask. A profile may
also declare:

- `preprocessing.method`: `none`, `median`, `adaptive_median`, or
  `curvature_anisotropic_diffusion`;
- `preprocessing.clip_hu_range`;
- `cleanup.fill_small_holes`; and
- `cleanup.remove_small_objects`.

Cleanup must state `2D` or `3D`, a threshold, `physical` or `voxel` units, and
connectivity. Physical thresholds mean mm² in 2D and mm³ in 3D. A different
HU range or morphology rule requires a different definition and profile ID;
the operation must never change behind an existing column name.

Enabled non-default definitions are appended to `slices.parquet`, the
vertebral/range aggregates, and `signature.parquet`. The fixed signature grid
and default core channels remain unchanged, so a study can use the common core
and add prespecified profile-specific features without redefining it.
`hu_distributions.parquet` is deliberately independent of these filters: it
uses the complete model-native source for each fixed SM, SAT, aVAT, and tVAT
channel and the unchanged prepared CT values. If the selected backend has no
homologous native compartment, the corresponding channel remains explicitly
missing; a broader region is not relabeled merely to fill the histogram.

## Shipped sensitivity profiles

The files in `config/tissue_profiles/` are strict configuration overrides for
published HU windows or explicitly parameterized preprocessing/cleanup
sensitivity analyses. They are not clinical phenotypes. The default is
[`consensus_hu.yaml`](../config/tissue_profiles/consensus_hu.yaml).

Examples include the Brown/Derstine VAT windows, the optional BOA-style median
classification filter, and explicitly parameterized physical or pixel cleanup
experiments. Every shipped profile is validated in tests and retains the
default definitions alongside its additional outputs.

## Interpretation

The -29-to-150 HU muscle and -190-to--30 HU adipose windows are widely used
compatibility definitions, not universal biological boundaries. The exact
bounds remain in the output names. Sarcopenia, myosteatosis, and outcome-based
cutpoints belong to a governed downstream analysis with the relevant cohort,
anatomical view, acquisition protocol, and uncertainty specified.

Whole-bone mean HU is not bone mineral density or a validated trabecular ROI.
Similarly, an HU filter inside the muscle compartment is a technical
measurement, not microscopic intramyocellular fat.
