# Changelog

## Unreleased

### Changed

- measurement schema 3.5.0 names through-plane sampling distances
  `slice_spacing_normal_mm` and `slice_spacing_superior_mm`; consistent DICOM
  `SliceThickness (0018,0050)` is preserved separately in conversion
  provenance and report metadata;
- scheduler workers treat `SIGTERM` as a local drain request: finish and
  publish the active case, then exit without claiming another case;
- local and Slurm directory runs can use `--update` without a run ID to advance
  the implicit `current` lineage, retain the previous generation, reuse
  unchanged case bundles, and process only added or source-changed studies;
- DICOM studies use the hashed Study Instance UID as stable automatic case
  identity when available, and the analysis identity now includes source CT
  content plus preserved DICOM study metadata;
- classic CT eligibility now requires a finite, consistent HU rescale transform,
  and unsupported single-file Enhanced CT input fails with an explicit message;
  and
- the minimum supported `pypdf` version is 6.15.0, including upstream fixes
  for crafted-PDF resource exhaustion.

## 1.0.0rc1 — 2026-08-05

This is a deliberate breaking release candidate.

### Added

- one strict typed configuration with generated defaults/schema;
- one service shared by the Python API and CLI;
- automatic single-case or deterministic cohort dispatch from one `analyze`
  entry point, with explicit manifests retained for governed and scheduler
  execution;
- strict direct DICOM CT input and standalone transfer-ready NIfTI conversion
  with a hash-bound, privacy-safe technical sidecar through the shared
  SimpleITK/GDCM geometry and provenance boundary;
- recursive DICOM cohort conversion with content-derived filenames, an
  analysis-ready manifest, and a privacy-minimized conversion report;
- content-addressed case identity, immutable manifests, artifact hashes,
  atomic promotion, deterministic resume, and ordered aggregate tables;
- CTDeepRot orientation integrity with safe repair and mandatory review flags;
- upstream SPINEPS/VERIDAH plus TPTBox/VibeSeg default vertebral backend;
- canonical native-mm slice, three-bin vertebral, summary, fixed-grid
  signature, and whole-volume native-compartment HU-distribution measurements;
- deterministic CSV mirrors of the canonical Parquet tables, an opt-out flag,
  and verified later CSV conversion from an existing result bundle;
- an explicit L3-only low-resource preset using the ResEncM models without
  resizing the in-plane field of view;
- an optional, externally synchronized BOA Task 542 tissue backend using the
  original Apache-2.0 model release, pinned nnU-Net inference, upstream-
  compatible 5-mm thickness staging, native labels, and explicit missing
  channels where BOA has no homologous compartment;
- optional one-page scientific case reports with aligned sagittal spine,
  tissue-area profile, native-compartment HU distributions, measurement table,
  axial overlay, and ordered run/export collation;
- a generic filesystem-backed scheduler-worker mode with shared input preflight,
  collision-safe whole-case claims, and early release of surplus workers;
- external pinned model synchronization/verification;
- non-root, weight-free, digest-pinned container definition; and
- reproducible single- or multi-platform Buildx orchestration with source
  provenance, strict final-layer asset auditing, OCI SBOM and vulnerability
  receipts; and
- release audit, Python dependency SBOM, clean-install, and CI tooling.

### Changed

- Python 3.11 and `uv.lock` are the reproducible environment boundary.
- the transitive `aiohttp` runtime is locked at 3.14.3, clearing the published
  advisories affecting 3.14.1 without changing a pipeline API or model.
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
- source CT voxel dtype and calibrated values are preserved independently of
  path names; integer casting is explicit and validated only for output labels.
- model-native inference results are named compartments; HU-filtered derived
  results are named tissues. The default applies only the declared skeletal
  muscle and adipose HU definitions and does not calculate IMAT, LAMA, or NAMA.
- the default comparison signature combines a 100-by-20-mm translation-only
  vertebral-reference grid with 5-HU whole-analyzed-volume distributions for
  native SM, SAT, aVAT, and tVAT compartments.
- JSON CLI mode reserves stdout for its single machine-readable payload and
  routes dependency progress to stderr.
- the default pipeline-execution budget is four hours so the automatic CPU
  fallback can complete the pinned model stack.
- TotalSegmentator task 297 rib/iliac landmark inference is now an explicit
  opt-in extension. Default anthropometry uses the observed T10–L5 minimum
  waist, sacral pelvic maximum, and their ratio without that model.
- the pinned VibeSeg CPU path is guarded against TPTBox's CUDA-only memory
  telemetry without changing upstream inference or CUDA execution.
- CTDeepRot checkpoint loading requires the exact validated PyTorch/torchvision
  pair and fails closed instead of retrying legacy pickle-loading APIs.
- case manifests retain distinct QC observations even when their stage, code,
  and summary text match, so an error-level finding cannot be hidden by an
  earlier warning for another anatomical level.
- informational scan-boundary and incomplete-coverage findings remain in the
  immutable case evidence without escalating manual review; only warnings and
  errors enter review queues and report notes. Vertebral fragmentation uses the
  same 18-neighbour, 5% largest-component rule as territory measurement.
- the pinned TotalSegmentator support models retain nnU-Net preprocessing,
  inference, interpolation, and multiclass decisions while resampling output
  logits in bounded channel groups, avoiding full-FOV target-grid memory
  spikes.
- explicit CUDA selection now fails the preflight gate when the accelerator is
  unavailable; the container records its CUDA 13.0 runtime requirement and
  whether it was built from a dirty source context.

### Removed

- named pipeline registry and the legacy Stanford/BOA/TotalSegmentator
  multi-stage pipeline variants;
- working-directory global and named-pipeline YAML configuration;
- old command collection, legacy standalone bulk-conversion helper, mutable-
  memory Python API, and standalone measurement/report commands;
- nnU-Net v1 trainers and obsolete crop/postprocess/action layers; and
- backward-compatibility aliases.

### Post-release work

- Repeat full and low-resource runtime and memory measurements from the frozen
  current default after task 297 became optional; historical candidate timings
  are not an accelerator ranking.
- Manual contour, anthropometric, and clinical-validity studies remain
  separate post-publication work.
