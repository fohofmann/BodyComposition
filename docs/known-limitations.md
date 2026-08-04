# Model/data cards and known limitations

## Intended use

BodyComposition produces research measurements and technical QC from
deidentified CT. Intended users are imaging researchers who can validate
segmentation, physical geometry, cohort coverage, and clinical definitions.
The outputs are not diagnostic labels, treatment recommendations, or a
validated clinical report.

## Input domain

The release expects calibrated three-dimensional CT with meaningful physical
metadata. It accepts one NIfTI/DICOM CT or a directory of cases.
SimpleITK/GDCM conversion preserves and validates the reader geometry, but it
does not classify contrast phase or reconstruction, calibrate scanners,
de-identify DICOM, or remove burned-in pixel annotations. Directory analysis
processes every discovered CT series independently; it does not infer which
phase or reconstruction a study intended. Use an explicit Series Instance UID
when only one series should be processed.

CTDeepRot estimates proper rotations from anatomy, not arbitrary affine
corruption. Oblique, truncated, metal-affected, unusual-anatomy, pediatric,
postoperative, and non-human scans may be uncertain or out of domain. Every
automatic repair is flagged for manual review.

## Segmentation models

SPINEPS/VERIDAH and the internal vertebral/tissue models inherit their training
data, labeling conventions, scanner/protocol, population, and pathology
limitations. Variant enumeration (including T13/L6), transitional anatomy,
rib remnants, fractures, implants, resections, deformity, and limited coverage
require review. BodyComposition validates geometry and QC contracts but does
not make an upstream model clinically generalizable.

The default tissue models have frozen public source revisions and model cards.
Weights remain external to wheels and containers; a deployment is operational
only after `bodycomposition models sync` has downloaded and verified every
required asset from its declared source.

The optional BOA Task 542 model predicts body regions rather than the default
model's complete seven-compartment schema. HU-filtered SM, SAT, aVAT, and tVAT
can be derived through the documented adapter mapping, but native aVAT/tVAT,
heart, and lung channels are unavailable and remain missing. Cross-backend
equivalence has not been established; studies must retain backend identity and
must not pool outputs as if the source masks were interchangeable.

The container removes torchmetrics' bundled optional DISTS image-metric
checkpoint to enforce a strict no-model-weights image. BodyComposition does
not use or expose that metric.

The container definition targets Linux `amd64` and `arm64`. Its PyTorch wheels
use CUDA 13.0, which requires NVIDIA driver branch R580 or newer. GPU readiness
is validated at runtime rather than inferred from the host name. Native macOS
and Windows containers are not release targets; the Python package can run on
CPU where its frozen dependencies support the platform. The unreleased source
does not yet publish a public image automatically. Platform validation and
performance observations are reported separately in
[performance.md](performance.md).

## Body surface and circumference

The default body/trunk envelope is derived from predicted compartments,
not a validated skin model. Connected arms, sparse predictions, devices,
table/padding, and field-of-view truncation can affect contour and
circumference. The sacral maximum is called a pelvic circumference, not a
validated hip circumference. Observed waist/pelvic extrema may be retained from
an incomplete search, but their eligibility and coverage must be preserved.
The minimum-waist search retains an observed but ineligible value from the
bounded T10-to-L5 portion when one boundary anchor is outside the scan. It
remains null when no supported part of that range is bounded or no valid
closed contour exists. The pelvic maximum remains null without a bounded
sacral territory. A selected sacral contour that is closed and nonfragmented
but touches the lateral FOV remains a numeric, explicitly FOV-cropped
observation; it and ratios using it are ineligible as exact values and may
underestimate the truth. Fragmented or otherwise untraceable boundary
contours remain unavailable. Full-body contact, trunk contact, sacral-search
contact, and selected-value cropping are recorded as separate flags.
These measures remain research phenotypes.

## Longitudinal measurements

The package preserves physical superior position and native vertebral
territories and exports a versioned fixed-mm signature. Alignment is
translation-only: direct L3 when available, otherwise an explicitly
confidence-scored estimate from visible vertebral centroids. It does not
normalize patient height, stretch anatomy, or impute unscanned regions.
Comparisons must retain coverage, reference confidence, variants, and
anatomical missingness. A single-anchor inferred origin is low confidence.
Affine area and volume integration, including numeric longitudinal-bin
aggregation, uses each native slice slab's exact physical overlap and remains
physically correct for oblique images. Categorical territory and bin
annotations in the per-slice table use the slice centre for display. Cases
whose in-plane superior span exceeds 10 mm are flagged for review. That
threshold is an engineering trigger, not proof that smaller obliquity is exact
or clinically validated.

## Tissue interpretation

HU windows are named compatibility definitions, not universal biological
boundaries. Contrast, scanner, reconstruction, and calibration affect
attenuation. Complete muscle-compartment mean HU is a continuous measurement.
The default does not calculate IMAT, LAMA, NAMA, or binary muscle-quality
phenotypes. Any study-specific HU filter is a separately named technical
partition, not a biological diagnosis. Whole-bone area/HU is not BMD or a
validated trabecular ROI. Sarcopenia, myosteatosis, and outcome cutpoints are
deliberately absent.

## Fairness and validation

No claim is made that model performance or derived associations are uniform by
sex, age, body size, ancestry, site, scanner, protocol, contrast phase,
diagnosis, or disease severity. A deployment study must prespecify subgroup
performance, missingness, review burden, and error impact before using results.

## Privacy and security

File names and source paths are excluded from manifests, but CT pixels and
generated images remain sensitive medical data. Use pseudonymous IDs, access
controls, encrypted transport/storage, and governed linkage outside the
repository. See [SECURITY.md](../SECURITY.md).
