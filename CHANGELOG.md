# Changelog

## 1.0.0rc1 — unreleased

This is a deliberate breaking release candidate.

### Added

- one strict typed configuration with generated defaults/schema;
- one service shared by the Python API and CLI;
- strict direct DICOM CT input and standalone conversion through the shared
  SimpleITK/GDCM geometry and provenance boundary;
- content-addressed case identity, immutable manifests, artifact hashes,
  atomic promotion, deterministic resume, and ordered aggregate tables;
- CTDeepRot orientation integrity with safe repair and mandatory review flags;
- upstream SPINEPS/VERIDAH plus TPTBox/VibeSeg default vertebral backend;
- canonical native-mm slice, three-bin vertebral, and summary measurements;
- optional one-page case reports and ordered run/export collation with a
  conditional manual-review summary;
- external pinned model synchronization/verification;
- non-root, weight-free, digest-pinned container definition; and
- release audit, SBOM, vulnerability, clean-install, and CI tooling.

### Changed

- Python 3.11 and `uv.lock` are the reproducible environment boundary.
- SimpleITK physical metadata/indices are explicitly `x-y-z`; arrays are
  explicitly `z-y-x`.
- only vertebral-body/corpus masks feed downstream measurements and reports.
- failures continue by default and are included in aggregate tables.
- model downloads are explicit setup operations and are forbidden during
  inference.
- branch-aware coverage is measured across the complete package without broad
  subsystem omissions; model/GPU regressions remain separate integration gates.
- the standard CLI and Python API need only an input CT; code-defined settings,
  automatic device selection, and the `./output` result root are defaults.
- the pinned VibeSeg CPU path is guarded against TPTBox's CUDA-only memory
  telemetry without changing upstream inference or CUDA execution.

### Removed

- named pipeline registry and legacy Stanford/BOA/TotalSegmentator pipeline
  variants;
- working-directory global and named-pipeline YAML configuration;
- old command collection, bulk DICOM-conversion helper, mutable-memory Python
  API, and standalone measurement/report commands;
- nnU-Net v1 trainers and obsolete crop/postprocess/action layers; and
- backward-compatibility aliases.

### Remaining release validation

- Rebuild the publication artifacts and container from the approved release
  tag, and complete the deferred Linux x86-64 and scientific/cohort validation
  gates before making platform-performance or clinical-validity claims.
