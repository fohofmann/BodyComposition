# DICOM input and conversion

BodyComposition accepts a three-dimensional NIfTI/DICOM CT or a directory of
cases. CLI and Python use the same SimpleITK/GDCM reader, discovery rules, and
physical-domain validation.

## Direct analysis

Point `analyze` at one DICOM directory or at any file in the intended series:

```bash
bodycomposition analyze ./input/case-001/dicom
bodycomposition analyze ./input/case-001/dicom/IM0001.dcm
```

Point the same command at a cohort root to analyze every recursively discovered
CT series as an independent case:

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

No series is silently preferred by file count, description, phase, kernel, or
directory order. Without `--series`, every discovered CT series is processed.
A raw Series Instance UID is used only for local selection; case manifests
store its SHA-256, not the UID itself.

Directories created by cohort conversion contain
`bodycomposition-batch.json`; `analyze` uses that manifest automatically. A
directory containing several arbitrary NIfTI files is accepted only when every
file has a matching BodyComposition conversion sidecar. Otherwise an explicit
batch manifest is required so masks cannot be mistaken for source CTs.

During analysis, the selected series is converted to a verified NIfTI in the
private attempt workspace. It is removed after the terminal case bundle is
promoted. The source DICOM content digest, byte size, pixel digest, physical
geometry, instance count, metadata-completeness flags, and converter version
remain in path-free provenance.

## Standalone conversion and pre-staging

The converter remains available without running any model:

```bash
bodycomposition convert ./input/case-001/dicom ./input/case-001/ct.nii.gz
```

The command writes a transfer pair:

- `ct.nii.gz`, containing the CT voxel values and physical geometry; and
- `ct.bodycomposition.json`, containing hash-bound, privacy-safe conversion
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

Every discovered CT series is converted independently to a content-named
NIfTI/sidecar pair. The directory also contains:

- `bodycomposition-batch.json`, ready for automatic or explicit analysis; and
- `bodycomposition-conversion.json`, a path- and identifier-minimized report
  of converted and failed series.

Transfer the complete directory and continue without another preparation step:

```bash
bodycomposition analyze ./converted
```

If the sidecar is present, `analyze` validates its NIfTI byte size, file hash,
pixel hash, and geometry before accepting its metadata. A missing sidecar does
not prevent analysis of an otherwise valid NIfTI, but the original DICOM
technical provenance is then unavailable. A present but stale or modified
sidecar is an input error rather than silently ignored.

An output ending in `.nii` or `.nii.gz` requests one conversion; a directory
output requests cohort conversion. When a file output is combined with a
multi-series input, use `--series UID`. Use `--overwrite` only for an
intentional replacement. Cohort conversion continues after a case failure,
returns a nonzero status when any series fails, and records the successful
cases in its generated analysis manifest.

```bash
bodycomposition convert ./input/case-001/dicom ./input/case-001/ct.nii.gz --json
```

The equivalent Python surface is:

```python
from BodyComposition import convert_dicom, discover_dicom_series

series = discover_dicom_series("./input/case-001/dicom")
result = convert_dicom(
    "./input/case-001/dicom",
    "./input/case-001/ct.nii.gz",
    series_uid=series[0].series_instance_uid,
)

cohort = convert_dicom("./import", "./converted")
```

Discovery returns only Series Instance UID, modality, and instance count. It
does not return patient, study, description, date, accession, or source-path
metadata.

## Geometry and orientation contract

Conversion does not call `DICOMOrient`, transpose arrays, infer a replacement
direction, or repair metadata. GDCM orders and assembles the selected series;
the resulting SimpleITK image defines the LPS physical domain. The written
NIfTI is read back and rejected unless its pixels and full size, spacing,
origin, and direction match that domain.

SimpleITK metadata, indices, and size remain `x-y-z`; SimpleITK arrays remain
`z-y-x`. Direct DICOM analysis then passes the converted image to the ordinary
orientation-prepared image boundary. CTDeepRot assesses the input there, and
only that stage may apply a supported lossless orientation repair. The
transfer sidecar carries the same DICOM orientation-completeness flags, so
pre-staged NIfTI analysis preserves this uncertainty handling.

## Privacy boundary

The NIfTI writer receives pixels and physical geometry without DICOM patient or
study tags. The sidecar contains scanner manufacturer/model, hashed Series UID,
instance count, metadata-completeness flags, converter identity, source
content hash, and geometry binding. It excludes patient and study identifiers,
accession, dates, descriptions, free-text fields, source paths, and the raw
Series UID. Case manifests follow the same boundary. This reduces accidental
identifier propagation, but it is not de-identification:

- source DICOM files are not modified;
- identifying text encoded in pixels is not detected or removed;
- a Series Instance UID is visible in local discovery, selection errors, and
  the standalone conversion result; and
- CT pixels and all derived images remain sensitive medical data.

Apply the institutionally approved DICOM de-identification profile, storage,
access-control, and retention rules before or around BodyComposition as
required by the project.
