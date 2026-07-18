# DICOM input and conversion

BodyComposition accepts either a three-dimensional NIfTI CT or one DICOM CT
series. Both CLI and Python API use the same SimpleITK/GDCM reader and the same
physical-domain validation.

## Direct analysis

Point `analyze` at a DICOM directory or at any file in the intended series:

```bash
bodycomposition analyze ./input/case-001/dicom
bodycomposition analyze ./input/case-001/dicom/IM0001.dcm
```

One CT series is selected automatically. Non-CT series do not compete with a
single CT series. If a directory contains more than one CT series, selection
must be explicit; BodyComposition never chooses by file count, description,
phase, kernel, or arbitrary directory order:

```bash
bodycomposition analyze ./input/case-001/dicom \
  --series 1.2.826.0.1.3680043.10.123.456
```

For a batch manifest, add the optional `series_uid` field to that case. A raw
Series Instance UID is used only to select local input. The case manifest
stores its SHA-256, not the UID itself.

During analysis, the selected series is converted to a verified NIfTI in the
private attempt workspace. It is removed after the terminal case bundle is
promoted. The source DICOM content digest, byte size, pixel digest, physical
geometry, instance count, metadata-completeness flags, and converter version
remain in path-free provenance.

## Standalone conversion

The converter remains available without running any model:

```bash
bodycomposition convert ./input/case-001/dicom ./input/case-001/ct.nii.gz
```

Use `--series UID` for an ambiguous directory and `--overwrite` only for an
intentional atomic replacement. The command fails rather than partially
writing an output. Its JSON mode reports source/output hashes and geometry:

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
orientation-prepared image boundary. CTDeepRot assesses the input there, and only that
stage may apply a supported lossless orientation repair. Incomplete DICOM
Image Orientation Patient or Image Position Patient metadata is carried into
that assessment as uncertainty rather than silently repaired during reading.

## Privacy boundary

The NIfTI writer receives pixels and physical geometry without DICOM patient or
study tags. Case manifests omit source paths and raw DICOM UIDs. This reduces
accidental identifier propagation, but it is not de-identification:

- source DICOM files are not modified;
- identifying text encoded in pixels is not detected or removed;
- a Series Instance UID is visible in local discovery, selection errors, and
  the standalone conversion result; and
- CT pixels and all derived images remain sensitive medical data.

Apply the institutionally approved DICOM de-identification profile, storage,
access-control, and retention rules before or around BodyComposition as
required by the project.
