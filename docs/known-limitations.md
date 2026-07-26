# Model/data cards and known limitations

## Intended use

BodyComposition produces research measurements and technical QC from
deidentified CT. Intended users are imaging researchers who can validate
segmentation, physical geometry, cohort coverage, and clinical definitions.
The outputs are not diagnostic labels, treatment recommendations, or a
validated clinical report.

## Input domain

The release expects calibrated three-dimensional CT with meaningful physical
metadata. It accepts NIfTI or one explicitly selected DICOM CT series.
SimpleITK/GDCM conversion preserves and validates the reader geometry, but it
does not classify contrast phase or reconstruction, calibrate scanners,
de-identify DICOM, or remove burned-in pixel annotations. When multiple CT
series are present, the caller must supply the intended Series Instance UID.

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

The container removes torchmetrics' bundled optional DISTS image-metric
checkpoint to enforce a strict no-model-weights image. BodyComposition does
not use or expose that metric.

## Body surface and circumference

The default body/trunk envelope is derived from predicted tissue compartments,
not a validated skin model. Connected arms, sparse predictions, devices,
table/padding, and field-of-view truncation can affect contour and
circumference. The sacral maximum is called a pelvic circumference, not a
validated hip circumference. Observed waist/pelvic extrema may be retained from
an incomplete search, but their eligibility and coverage must be preserved.
Missing anchors and searches without any valid closed contour remain null.
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
