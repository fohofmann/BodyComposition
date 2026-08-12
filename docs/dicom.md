# DICOM input and conversion

BodyComposition accepts a three-dimensional NIfTI/DICOM CT or a directory of
cases. CLI and Python use the same SimpleITK/GDCM reader, discovery rules, and
physical-domain validation.

## Direct analysis

Point `analyze` at one DICOM study directory or at any file in the intended
series:

```bash
bodycomposition analyze ./input/case-001/dicom
bodycomposition analyze ./input/case-001/dicom/IM0001.dcm
```

Point the same command at a cohort root to analyze one selected CT stack per
recursively discovered DICOM study:

```bash
bodycomposition analyze ./import
```

For example, `import/patient1/DICOMS/*.dcm` and
`import/patient2/DICOMS/*.dcm` become two cases. Directory names and nesting
depth do not define identity; content does. Non-CT series are ignored. To run
only one selected CT from a directory, pass its UID:

```bash
bodycomposition analyze ./input/case-001/dicom \
  --series 1.2.826.0.1.3680043.10.123.456
```

Within one patient-study directory, all CT series are geometry-audited and the
best eligible axial stack is selected automatically. Eligibility requires at
least two planes, axial orientation, complete positions, uniform spacing,
consistent in-plane dimensions/pixel spacing, and consistent declared slice
thickness when that tag is present. Every retained instance must also declare
the same finite Rescale Intercept and the same positive Rescale Slope. Eligible
stacks rank by anatomical coverage, then finer slice spacing, then retained
plane count. An exact quality tie fails closed. Directory order, description,
phase labels, and raw file count are not tie-breakers. A raw Series Instance UID
is used only for local selection; case manifests store its SHA-256, not the UID
itself.

Directories created by cohort conversion contain
`bodycomposition-batch.json`; `analyze` uses that manifest automatically. A
directory containing several arbitrary NIfTI files is accepted only when every
file has a matching BodyComposition conversion sidecar. Otherwise an explicit
batch manifest is required so masks cannot be mistaken for source CTs.

During analysis, the selected stack is converted to a verified NIfTI in the
private attempt workspace. It is removed after the terminal case bundle is
promoted. The source DICOM content digest, byte size, pixel digest, physical
geometry, selection audit, allowlisted technical metadata, instance count,
metadata-completeness flags, and converter version remain in path-free
provenance.

## Standalone conversion and pre-staging

The converter remains available without running any model:

```bash
bodycomposition convert ./input/case-001/dicom ./input/case-001/ct.nii.gz
```

The command writes a transfer pair:

- `ct.nii.gz`, containing the CT voxel values and physical geometry; and
- `ct.bodycomposition.json`, containing hash-bound, allowlisted conversion
  provenance.

Transfer both files into the same destination directory without changing one
basename independently. Continue with the ordinary pipeline command; no
sidecar argument or import step is needed:

```bash
bodycomposition analyze ./input/case-001/ct.nii.gz
```

To pre-stage a complete nested DICOM cohort, make the output a directory:

```bash
bodycomposition convert ./import ./converted
```

For each discovered study, the best eligible CT stack is converted to a
pseudonymously named NIfTI/sidecar pair. The name derives from the hashed Study
Instance UID when available, so replacing the data or preserved metadata for
one study keeps the same logical case name. The directory also contains:

- `bodycomposition-batch.json`, ready for automatic or explicit analysis; and
- `bodycomposition-conversion.json`, a path- and identifier-minimized report
  of converted and failed study groups.

Transfer the complete directory and continue without another preparation step:

```bash
bodycomposition analyze ./converted
```

To refresh a pre-staged cohort after adding or replacing source DICOM, rerun
conversion with `--overwrite`, transfer the complete updated directory, and
run analysis with `--update` as described in [execution](execution.md).
Unchanged analysis identities are reused; changed source bytes or preserved
study metadata receive a new analysis identity.

If the sidecar is present, `analyze` validates its NIfTI byte size, file hash,
pixel hash, and geometry before accepting its metadata. When all source
instances declare one consistent positive DICOM `SliceThickness (0018,0050)`,
the sidecar retains it separately from the NIfTI slice-centre spacing. A
missing sidecar does not prevent analysis of an otherwise valid NIfTI, but the
original DICOM technical provenance is then unavailable. A present but stale
or modified sidecar is an input error rather than silently ignored.

An output ending in `.nii` or `.nii.gz` requests one conversion; a directory
output requests cohort conversion. A file output automatically selects the best
eligible stack from one patient-study input. Use `--series UID` only for an
explicit series override. When a Series UID itself contains clinically
adjudicated acquisitions, the advanced
`--acquisition-number VALUE` override selects that exact group and records the
decision; it must be paired with `--series UID`. Use `--overwrite` only for an
intentional replacement. Cohort conversion continues after a study failure,
returns a nonzero status when any study fails, and records the successful cases
in its generated analysis manifest.

Axial-only validation is automatic for every DICOM conversion and direct DICOM
analysis. When exactly one orientation group is a uniformly spaced axial
stack, explicit localizer/scout objects, Secondary Capture objects, and
non-monochrome or multi-sample accessory images are excluded. Other orientation
groups are excluded only when tagged `LOCALIZER` or
`DERIVED/SECONDARY/REFORMATTED`:

```bash
bodycomposition convert ./input/case-001/dicom ./input/case-001/ct.nii.gz \
  --series UID
```

No option is required. Missing geometry, untagged orientation mixtures,
duplicate positions, missing slice positions, inconsistent in-plane geometry
or declared thickness, coronal/sagittal stacks, and localizer-only series are
ineligible. Multiple eligible acquisitions or series are ranked by the
geometry rules above; exact ties fail. This preserves valid axial
reconstructions even when their own `ImageType` is
`DERIVED/SECONDARY/REFORMATTED`; tags alone never determine the retained
stack. The sidecar records every eligible/rejected candidate, ranking metrics,
the selection strategy, discovered/retained/excluded instance counts, and each
excluded file's content hash. Alternatives, exclusions, and bounded header
normalizations are marked for manual review so the decision remains visible in
downstream QC.

The direct reader currently supports classic single-frame CT series. A
single-file Enhanced CT multi-frame object is rejected with an explicit error.
Convert such an object to a validated NIfTI with calibrated HU and preserved
physical geometry before analysis.

```bash
bodycomposition convert ./input/case-001/dicom ./input/case-001/ct.nii.gz --json
```

The equivalent Python surface is:

```python
from BodyComposition import convert_dicom

result = convert_dicom(
    "./input/case-001/dicom",
    "./input/case-001/ct.nii.gz",
)

cohort = convert_dicom("./import", "./converted")
```

Discovery returns only Series Instance UID, modality, and instance count. It
does not return patient, study, description, date, accession, or source-path
metadata.

## Geometry and orientation contract

Conversion does not call `DICOMOrient`, transpose arrays, or infer an anatomical
orientation. GDCM orders and assembles the selected stack. Small numeric
rounding differences in per-instance orientation, pixel spacing, declared
slice thickness, or slice positions may be normalized to a median orthonormal
grid when they remain within fixed tolerances. The source DICOM is never
modified, every normalization is recorded in `selection.header_repairs`, and
material inconsistencies fail. The written NIfTI is read back and rejected
unless its pixels and full size, spacing, origin, and direction match that
assembled physical domain.

For classic CT input, finite Rescale Intercept and Rescale Slope values are
required on every retained instance. The slope must be positive and both values
must be consistent throughout the stack. GDCM applies that transform while
reading; the resulting NIfTI therefore contains the calibrated values used by
the measurement pipeline.

SimpleITK metadata, indices, and size remain `x-y-z`; SimpleITK arrays remain
`z-y-x`. Direct DICOM analysis then passes the converted image to the ordinary
orientation-prepared image boundary. CTDeepRot assesses the input there, and
only that stage may apply a supported lossless orientation repair. The
transfer sidecar carries the same DICOM orientation-completeness flags, so
pre-staged NIfTI analysis preserves this uncertainty handling.

## Privacy boundary

The NIfTI writer receives pixels and physical geometry without DICOM tags. The
sidecar contains its UTC conversion timestamp; DICOM acquisition, study,
series, content, and instance-creation dates; selected series/study/protocol
descriptions; scanner, contrast, exposure, reconstruction, and geometry fields;
hashed Series/Study/Frame-of-Reference/SOP UIDs; and the complete selection
audit. `InstanceCreationDate/Time` describes the retained DICOM objects when
present; it is not guaranteed to be the PACS export time.

The allowlist excludes patient name/ID, birth date, sex, accession, source
paths, and raw UIDs. However, descriptions and protocol fields are free text
and can contain identifiers at some institutions. Case manifests follow the
same boundary. The sidecar and CT are therefore not de-identified:

- source DICOM files are not modified;
- identifying text encoded in pixels is not detected or removed;
- a Series Instance UID is visible in local discovery, selection errors, and
  the standalone conversion result; and
- CT pixels and all derived images remain sensitive medical data.

Apply the institutionally approved DICOM de-identification profile, storage,
access-control, and retention rules before or around BodyComposition as
required by the project.
