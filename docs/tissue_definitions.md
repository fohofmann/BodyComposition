# CT body compartments, tissue classes, and executable defaults

This document is the implementation contract for measurement schema `2.1.0`.
It translates the [source-backed literature review](research/body_composition_definitions_review.md),
[paper matrix](research/body_composition_paper_matrix.csv), and
[feature dictionary](research/body_composition_default_features.csv) into
pipeline defaults and named sensitivity profiles.

## Technical summary

The default pipeline preserves the learned anatomical compartments, classifies
tissue from untouched calibrated HU, and reports exact physical slice and range
measurements. It does not smooth or delete small tissue components by default.
Alternative published HU windows, denoisers, connected-component rules, and
hole-filling rules are executable as named profiles, while unavailable learned
anatomical models are represented as explicit input contracts rather than
reimplemented from guesses.

## The distinction the pipeline enforces

Body-composition studies often give the same abbreviation to three different
objects:

1. an anatomical compartment bounded by fascia, body wall, or cavity;
2. an attenuation-defined tissue class inside that compartment; and
3. an outcome-derived phenotype or cutpoint.

The pipeline keeps them separate. The untouched model output in
`labels/{caseid}_int-bodycomposition.nii.gz` is the anatomical source. The
postprocessed compatibility mask in `masks/{caseid}_int-bodycomposition.nii.gz`
contains the selected HU classes. Canonical derived measurements use the raw
compartment labels and the untouched prepared CT, not a denoised intensity
image and not a thresholded mask from which excluded voxels cannot be
recovered.

This separation is part of `analysis_id`: the compartment array, processed
tissue array, label schema, definition map, HU preprocessing, cleanup settings,
and `tissue.profile_id` are hashed with the other scientific inputs.

## Default definitions

| Output prefix | Anatomical support | HU rule | Default | Reason |
|---|---|---:|---|---|
| `muscle_compartment` | raw `SM` plus raw learned `IMAT` | none | yes | Preserves the anatomical muscle envelope, including fatty replacement and connective tissue |
| `learned_imat` | raw learned `IMAT` label | none | yes | Keeps model-predicted IMAT distinct from HU-derived IMAT |
| `whole_bone_anatomical` | raw `BONE` compartment | none | yes | Makes anatomical bone area/volume available without calling it bone mass or BMD |
| `bone_tissue_hu_152_1000` | raw `BONE` compartment | 152 to 1000 | yes | Reproduces the foundational Mitsiopoulos CT tissue window; remains distinct from trabecular attenuation |
| `skeletal_muscle_tissue_hu_m29_150` | muscle compartment | -29 to 150 | yes | Foundational Mitsiopoulos window and dominant oncology/local compatibility definition |
| `lama_hu_m29_29` | muscle compartment | -29 to 29 | yes | Standard low-attenuation muscle partition; it is a phenotype, not noise |
| `nama_hu_30_150` | muscle compartment | 30 to 150 | yes | Standard normal-attenuation muscle partition |
| `imat_ct_hu_m190_m30` | muscle compartment | -190 to -30 | yes | Operational CT-visible intermuscular adipose tissue; not microscopic intramyocellular lipid |
| `sat_total_hu_m190_m30` | raw SAT compartment | -190 to -30 | yes | Foundational, BOA, and local-paper compatibility |
| `vat_total_hu_m190_m30` | union of raw aVAT/tVAT, or native VAT | -190 to -30 | yes | Primary broad adipose compatibility definition |
| `vat_total_hu_m150_m50` | same VAT anatomy | -150 to -50 | yes | Important C-SCANS/Brown compatibility output; threshold-sensitivity studies show it must remain separately named |

Every prefix receives voxel count, physical area, area validity/reason, raw mean
HU, and HU validity/reason per acquired slice. Physical range aggregation adds
mean CSA, exact integrated volume, pooled raw HU, coverage, and validity. The
same source columns therefore support the L3 centroid slice, mean across the L3
territory, the fixed 200-mm L3-centred slab, arbitrary named vertebral ranges
such as T12-L5, and the complete observed native-mm profile without stretching
the scan.

The following ratios are default continuous outputs:

- LAMA divided by -29-to-150 HU skeletal-muscle tissue;
- CT-IMAT divided by skeletal-muscle tissue plus CT-IMAT;
- VAT/SAT under the matched -190-to-30 HU rule; and
- VAT divided by VAT + SAT + CT-IMAT under the matched rule.

Range ratios are calculated from integrated component volumes. Slice ratios
are never averaged to make a multilevel ratio. A zero or invalid denominator
returns null with a reason.

SMI is not calculated from an inferred CT height. Supply a measured height to
the read-only view:

```bash
bodycomposition_measurements \
  --tables ./tables/case-001 \
  --view l3 \
  --aggregation slice \
  --height-m 1.78
```

This adds `smi_skeletal_muscle_tissue_hu_m29_150_cm2_m2` with explicit height
provenance. Named-range views similarly add volume/height² indices. Raw values
remain the authoritative persisted measurements.

## Why these defaults

The -190/-30 HU adipose and -29/150 HU muscle windows originate in cadaver
validation by [Mitsiopoulos et al.](https://doi.org/10.1152/jappl.1998.85.1.115),
remain common in clinical L3 work, and match the local validation pipeline in
[Hofmann et al.](https://doi.org/10.1038/s41598-025-96238-6). They are
compatibility boundaries, not universal biological constants. That is why the
exact bounds are encoded in each derived name.

[Aubrey et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4309522/) and the
muscle-fat terminology review by
[Engelke et al.](https://doi.org/10.1016/j.jot.2018.10.004) support separating
LAMA, NAMA, macroscopic fat inside the muscle compartment, and mean attenuation
instead of making one opaque “myosteatosis” label. The current methods standard
by [Prado et al.](https://doi.org/10.1016/j.ajcnut.2026.101283) likewise treats
CT outputs at the tissue/organ level and emphasizes acquisition, calibration,
analysis, and QC provenance.

VAT windows vary even in high-impact cohorts. Brown et al. used -150/-50 HU,
whereas the five-window paired analysis by
[Derstine et al.](https://doi.org/10.1038/s41598-022-06232-5) found materially
different areas for -205/-51, -150/-50, -195/-45, -190/-30, and -250/-50 HU.
The pipeline therefore exports the two most important compatibility measures
by default and makes all five runnable under distinct names.

No tissue-component threshold has comparable evidential convergence. Small
fat islands inside muscle can be the signal of interest. The default therefore
uses calibrated raw HU, no smoothing, no hole filling, and no minimum SM,
IMAT, VAT, or SAT component deletion. Anatomical body/trunk cleanup is a
separate measurement-support operation and does not rewrite tissue classes.

The bone outputs deliberately stop at anatomy and the published 152-to-1000 HU
tissue class. The influential L1 biomarker is attenuation in a validated
anterior trabecular ROI, not mean HU of the whole bone mask. The reviewed
automated implementation by
[Pickhardt et al.](https://doi.org/10.1259/bjr.20180726) uses customized
vertebral partitioning and adaptive cortical erosion whose complete parameters
and public implementation are not available. The pipeline therefore does not
label whole-bone HU as L1 trabecular attenuation or calibrated BMD. A validated
L1 trabecular ROI mask
can be supplied as an external anatomical label and measured with a raw-HU,
no-threshold definition; scanner/phase calibration remains a downstream study
requirement.

## Classification and cleanup order

For a named compatibility profile, mask construction uses this explicit order:

1. preserve the raw compartment labels;
2. optionally clip and/or denoise the intensity image for selected tissue
   classifiers only;
3. intersect the named compartment with the inclusive HU window;
4. optionally fill enclosed holes below the configured threshold; and
5. optionally remove connected components below the configured threshold.

The configuration records dimensionality (`2D` or `3D`), unit (`physical` or
`voxel`), connectivity (4/8 or 6/18/26), thresholds, and algorithm parameters.
In 2D, `voxel` means pixel count. Physical thresholds mean mm² in 2D and mm³ in
3D. Denoising can be scoped with `tissue.hu_denoise.apply_to`, so a fat-only
method does not silently alter muscle classification. Mean-HU outputs always
return to the untouched prepared CT.

## Executable literature profiles

Pass one profile as the ordinary final configuration override:

```bash
bodycomposition \
  -i ./data/images \
  -m BodyCompositionFast \
  -c config/tissue_profiles/derstine_2022_vat_m205_m51.yaml
```

| Profile | What it changes | Reproducibility interpretation |
|---|---|---|
| [`canonical_raw.yaml`](../config/tissue_profiles/canonical_raw.yaml) | -190/-30 fat, -29/150 muscle, no tissue smoothing/components/holes | Publication-compatible default |
| [`brown_2019_cscans_vat.yaml`](../config/tissue_profiles/brown_2019_cscans_vat.yaml) | VAT -150/-50 | Reported window compatibility |
| [`derstine_2022_vat_m205_m51.yaml`](../config/tissue_profiles/derstine_2022_vat_m205_m51.yaml) | VAT -205/-51 | One of the paired five-window definitions |
| [`derstine_2022_vat_m150_m50.yaml`](../config/tissue_profiles/derstine_2022_vat_m150_m50.yaml) | VAT -150/-50 | One of the paired five-window definitions |
| [`derstine_2022_vat_m195_m45.yaml`](../config/tissue_profiles/derstine_2022_vat_m195_m45.yaml) | VAT -195/-45 | One of the paired five-window definitions |
| [`derstine_2022_vat_m190_m30.yaml`](../config/tissue_profiles/derstine_2022_vat_m190_m30.yaml) | VAT -190/-30 | One of the paired five-window definitions |
| [`derstine_2022_vat_m250_m50.yaml`](../config/tissue_profiles/derstine_2022_vat_m250_m50.yaml) | VAT -250/-50 | One of the paired five-window definitions |
| [`boa_median_3x3.yaml`](../config/tissue_profiles/boa_median_3x3.yaml) | In-plane 1×3×3 median classification filter | BOA's optional, normally disabled filter |
| [`comp2comp_pixel_cleanup.yaml`](../config/tissue_profiles/comp2comp_pixel_cleanup.yaml) | 2D 8-connected IMAT minimum 10 pixels; bounded 5/50-pixel holes | Resolution-dependent Comp2Comp compatibility branch, not a recommended default |
| [`totalsegmentator_200mm3_cleanup.yaml`](../config/tissue_profiles/totalsegmentator_200mm3_cleanup.yaml) | 3D 26-connected minimum 200 mm³ | TotalSegmentator's optional general blob threshold applied as an explicit sensitivity analysis |
| [`draft_physical_cleanup_20_50mm3.yaml`](../config/tissue_profiles/draft_physical_cleanup_20_50mm3.yaml) | Local draft 20-mm³ SM/IMAT/VAT and 50-mm³ SAT rules | Retained only for validation against the raw default |
| [`lee_2018_fat_sensitivity.yaml`](../config/tissue_profiles/lee_2018_fat_sensitivity.yaml) | Fat -274/-49 plus 2D curvature anisotropic diffusion | Sensitivity implementation: the paper does not report enough filter parameters for a bitwise paper reproduction |
| [`ahmad_2023_adaptive_median_sensitivity.yaml`](../config/tissue_profiles/ahmad_2023_adaptive_median_sensitivity.yaml) | Parameterized 2D adaptive median preprocessing | Sensitivity implementation: input filtering and the unavailable target-specific learned model must not be presented as exact reproduction |

All profiles are merged over the canonical configuration and validated at
startup. The profile name is not a phenotype: it is preprocessing provenance.
The raw derived definitions remain available alongside a profile-specific
native compatibility mask.

## Anatomical methods that a parameter file cannot invent

SSAT/DSAT require a validated superficial-fascial plane; IPAT/RPAT require a
validated peritoneal/retroperitoneal boundary. The reviewed Ahmad model exposes
such labels, and the large DAFS study exposes multilevel outputs, but an exact
public model/implementation was not available for integration. The pipeline
therefore does not infer these boundaries from distance to skin or rename
ordinary SAT/VAT.

The generic derivation function accepts external validated labels named
`SSAT`, `DSAT`, `IPAT`, `RPAT`, or a validated trabecular ROI label and can
apply the same configurable HU rules (or no HU restriction):
[`derive_configured_tissue_masks`](../BodyComposition/measurement/tissues.py).
This makes the measurement method available when a validated compartment
source is supplied, without fabricating an anatomical segmentation that the
current models do not produce. Proprietary or insufficiently reported
algorithms remain explicitly non-reproducible rather than receiving guessed
defaults.

## Cutpoints and protocol effects

Sarcopenia, myopenia, and myosteatosis cutpoints are not pipeline defaults.
Published thresholds are tied to cohort, sex, BMI, outcome, vertebral level,
contrast phase, scanner, and reconstruction. The production pipeline exports
continuous measurements and acquisition/processing provenance; a study may
attach a versioned reference model downstream. Contrast- or scanner-specific
calibration must never be hidden by relabeling a raw HU measurement.
