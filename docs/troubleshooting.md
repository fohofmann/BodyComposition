# Troubleshooting and `doctor`

Start with:

```bash
bodycomposition doctor --json
```

Exit code `3` means the selected environment/model readiness gate failed; `4`
means a case execution failed. JSON stdout is safe for automation and logs
remain on stderr. By default, the exit code reflects `operational_ready`. Use
`doctor --release` when the stricter publication gate must determine the exit
code; the JSON always reports both states.

## Required models are not ready

Run `models verify` to identify exact missing, size-mismatched, or
hash-mismatched files. Then run the explicit `models sync` command with a
writable external cache and network access. Inference itself will not download.

Inference and `models verify` never alter an invalid bundle. An explicit
`models sync` stages the pinned replacement separately, verifies every required
file, and only then promotes it atomically. Preserve the old bundle separately
before synchronization when it is needed for investigation.
`upstream_revision_unresolved` means a model definition lacks an immutable
public source and is a publication blocker rather than a corrupt-cache message;
released model definitions must not report it. See [models.md](models.md).

## Dirty source tree

Cohort/release execution is blocked when Git-tracked source differs from the
recorded commit. Review and commit the intended patch first. For local
development only, set `runtime.allow_dirty: true`; that choice is recorded and
cannot produce `release_ready=true`.

## CUDA unavailable

`--device cuda` fails if CUDA is not available. Use `--device auto` for CUDA →
CPU selection. MPS is not a released inference target. The launcher, not
BodyComposition, controls `CUDA_VISIBLE_DEVICES`. The released container uses
the frozen PyTorch CUDA 13.0 runtime and therefore requires NVIDIA driver branch
R580 or newer. In Docker, expose the accelerator with the NVIDIA Container
Toolkit; in Apptainer/Singularity, use `--nv`. Run `doctor --device cuda`
inside the same container invocation and with the same mounts as inference.

## Accelerator out of memory

The failed case records `accelerator_out_of_memory`. Restart the identical run;
the shared queue enables `single_segmentation_bundle` mode and unloads models
between stages. If it still fails, reduce competing processes or use a device
with more memory. The pipeline will not silently switch model, precision, or
device.

## Existing run belongs to another plan

A `run_id` is immutable. Use the original ordered cases and configuration, or
choose a new run ID. Do not delete a run manifest to force reuse.

## DICOM input is ambiguous or unreadable

`analyze DIRECTORY` treats every recursively discovered CT series as a
separate case; it never chooses by directory order or largest slice count. Add
`--series UID` only when one exact series was requested. Likewise, directory-
output conversion converts every CT series:

```bash
bodycomposition analyze /path/to/cohort
bodycomposition convert /path/to/cohort /path/to/converted
```

A single-file conversion can represent only one series. If its input contains
several CT series, the command reports the available UIDs and requires an
explicit selection before writing:

```bash
bodycomposition convert /path/to/dicom /path/to/ct.nii.gz --series UID
```

No CT series means GDCM found no consistently declared CT input. Verify the
export is complete and readable. A Series Instance UID split across multiple
directories is rejected; consolidate that series without renaming or changing
its instances. Conversion does not repair metadata, de-identify DICOM, or
remove burned-in annotations.

## Artifact digest changed

`results inspect` verifies all bundle files. A size/digest change, missing
file, unexpected file, or escaping path means the result is no longer the
immutable object described by its manifest. Restore the original bundle or
rerun; do not update hashes manually.

## Output is not writable

The output and model mounts must be writable by the executing UID. The image
runs as UID/GID `10001`. Inputs should be mounted read-only; output and the
model directory used by `models sync` must permit writes. Inference may use a
read-only verified model mount.

## Stale shared-worker claim

Workers heartbeat under `.bodycomposition/execution/<run-id>`. A stale claim is
reclaimed after the configured delay and the old token is fenced. After the
bounded attempt count, the case receives a terminal failure instead of looping.
Do not edit coordination JSON while workers are active.

## Report rendering fails

Scientific results remain authoritative. Post-hoc rendering requires the
explicit report-input manifest and verifies that CT, vertebral, and
measurement identities match. A mixed/stale bundle is rejected. Use
`reports inspect` to validate an existing PDF/manifest pair.
