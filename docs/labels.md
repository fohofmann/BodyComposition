# Canonical labels

Model-native labels and downstream masks are distinct. A native artifact may
be retained for provenance/QC, but downstream stages consume only the
canonical physical-domain outputs listed here.

## Vertebral bodies

`masks/vertebral_bodies.nii.gz` contains corpora only. Full vertebrae and
posterior elements are not used for territories, measurement, or reports.

| Label | Anatomy |
| ---: | --- |
| 1–12 | T1–T12 |
| 13–17 | L1–L5 |
| 18 | L6 variant |
| 19 | sacrum |
| 20 | coccyx |
| 21 | T13 variant |

The result JSON records the native-to-anatomical mapping. T13 and L6 are not
compressed into a fixed template. Sacrum participates in the same territory
and three-bin summary contract as other supported levels.

## Tissue compartments

`masks/tissue_compartments.nii.gz` is the untouched model-predicted anatomical
compartment map:

| Label | Name |
| ---: | --- |
| 1 | SM (muscle compartment) |
| 2 | BONE |
| 3 | SAT |
| 4 | aVAT |
| 5 | tVAT |
| 6 | HEART |
| 7 | LUNG |
| 8 | IMAT (learned compartment) |

Native aVAT and tVAT remain separate. `total_vat` is a derived union in the
tables and does not overwrite either source label.

`masks/tissue_labels.nii.gz` is the configured HU/cleanup compatibility mask.
Canonical derived tissue classes are nevertheless calculated from the raw
compartment map and untouched prepared CT so excluded voxels remain auditable.
See [tissue_definitions.md](tissue_definitions.md).

## Body surface

`masks/body_surface.nii.gz` is aligned to the prepared CT. Label 1 is the
primary trunk and label 2 is remaining body support, normally extremities. The
full body support is their union. This is a measurement/QC envelope, not a
validated skin segmentation.

## Geometry

All canonical masks must match the orientation-prepared CT in size, spacing, origin,
direction, and physical domain. SimpleITK metadata/indices are `x-y-z`; arrays
returned by SimpleITK are `z-y-x`. Labels are resampled only with
nearest-neighbour interpolation.
