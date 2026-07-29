# Parallel execution and accelerator policy

Every process runs the same complete case pipeline. The process does not spawn
GPU workers and does not choose a physical GPU. CUDA visibility is owned by the
launcher (for example Slurm, a container runtime, or `CUDA_VISIBLE_DEVICES`).

## Shared whole-case queue

An ordered batch manifest can be submitted once or started by several identical
processes:

```bash
bodycomposition batch \
  /shared/input/cohort.json \
  -o /shared/output \
  --device auto
```

When `--run-id` is omitted, the ordered cases and queue-compatible configuration
produce a deterministic run ID. Compatible workers therefore join the same
queue without a worker index, shard number, or model assignment. Workers may
use different devices, CPU-thread limits, case timeouts, model-cache paths,
vertebral runtime devices, and logging levels. Scientific settings, requested
output artifacts, reporting, determinism, and fail-fast behavior must remain
identical. If an explicit run ID is used, every worker must receive the same
value and the same queue-compatible configuration.

Each process atomically claims one complete patient, runs all stages, commits an
attempt bundle atomically, and then looks for another patient. Models remain
loaded once per process during normal execution. Completed and terminal cases
are detected before claims, so every worker exits as soon as the batch is
terminal; workers do not poll after all work is complete.

Input geometry and content summaries are prepared once under a shared lock and
then reused by compatible workers. The cache contains privacy-safe, path-free
input summaries; DICOM patient-header fields are read only by the worker that
actually owns that case. Every owner still verifies the input against the
cached content identity immediately before inference.

The output must be on a POSIX filesystem shared by all workers and must provide
advisory file locking plus atomic same-filesystem directory creation and
rename. When combined PDF reporting is enabled, it must also support relative
symbolic links. Coordination state is stored below:

```text
<output>/.bodycomposition/execution/<run-id>/
```

The first compatible worker records one canonical run start time beside the
immutable plan digest. Every later worker reuses that time. Once all cases are
terminal, a run-level lock admits exactly one finalizer: it writes the
aggregates, validates the complete ordered report snapshot, derives the run end
from the latest case terminal time, and atomically publishes
`run_manifest.json` last. Other workers read that completed manifest instead of
rewriting it. Thus all workers return the same run period and final output
references even when they finish concurrently.

## Slurm array

The example script contains only the worker command:

```bash
#!/bin/bash
set -euo pipefail

srun bodycomposition batch \
  "$1" \
  -o "$2" \
  --worker \
  --device auto
```

The user supplies the array size and site-specific resources when submitting
it. For example:

```bash
sbatch \
  --array=0-31%8 \
  --gpus-per-task=1 \
  --cpus-per-task=8 \
  --mem=48G \
  bodycomposition-array.sh \
  /shared/input/cohort.json \
  /shared/output
```

`0-31` and `%8` are examples, not pipeline requirements. Choose at most one
array element per case. The optional percent suffix limits simultaneous tasks;
omit it when Slurm should admit as many workers as the allocation permits.
Partition, account, time limit, container invocation, and mount directives are
site-specific and deliberately remain outside the application.

Slurm controls the CUDA device visible to each task. The pipeline uses that
visible device as `cuda` and never maps Slurm task IDs to physical GPU IDs.
Allocated CPU affinity is used for PyTorch and SimpleITK pre/post-processing;
`runtime.cpu_threads` can impose a smaller explicit bound when needed.

`--worker` affects only scheduler lifecycle. An array task exits successfully
when every unfinished case is already held by another live worker, releasing
its allocation instead of waiting for the slowest case. The worker that
completes the last case publishes the aggregates and run manifest. If the last
owner is terminated before it can finish, rerun the same idempotent command so
a new worker can reclaim the stale case. The ordinary command without
`--worker` waits for the complete `BatchResult` and remains the default for
synchronous Python and local CLI use.

The same pattern works with separate GPU jobs. For example, a shell or service
manager may start one command with `CUDA_VISIBLE_DEVICES=0` and another with
`CUDA_VISIBLE_DEVICES=1`. BodyComposition itself does not start or manage those
processes.

## Stale claims and bounded recovery

A live claim writes a heartbeat and the current pipeline stage. Another worker
may reclaim it when either the heartbeat expires or the absolute case time
limit is reached. The old claim token is fenced: a late worker cannot commit a
canonical result after takeover. Abandoned claims consume a bounded attempt;
after the limit, the case receives a normal terminal failure manifest instead
of being reclaimed forever.

The stale and absolute time limits are stored in each claim. A worker with a
short timeout therefore cannot incorrectly reclaim a live claim created by a
worker with a longer timeout. If a worker itself raises its configured
pipeline-execution timeout between stages, that failed attempt is retained for
audit, the process exits nonzero, and the case remains claimable within the
bounded attempt budget. Another already-running worker—or a restarted command
with a larger timeout—can then run the complete case.

Defaults can be adjusted operationally without changing the scientific
configuration:

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `BODYCOMPOSITION_HEARTBEAT_SECONDS` | 30 | Claim heartbeat interval |
| `BODYCOMPOSITION_STALE_AFTER_SECONDS` | 300 | Missing-heartbeat takeover delay |
| `BODYCOMPOSITION_MAX_CLAIM_SECONDS` | pipeline budget + 600 | Absolute live/hung claim limit |
| `BODYCOMPOSITION_MAX_ATTEMPTS` | 3 | Abandoned/OOM attempts before terminal failure |
| `BODYCOMPOSITION_POLL_SECONDS` | 2 | Wait while other workers own remaining cases |

Ordinary deterministic pipeline errors are not repeated automatically in the
same invocation. Invalid inputs and cancellations are also terminal. Start a
new run after correcting those causes.

## OOM restart strategy

A recognized accelerator out-of-memory exception is written immediately to the
shared runtime state before failure reporting or cleanup. The command then
returns a non-zero execution result. The marker is scoped to an OOM-relevant
hardware profile: backend, accelerator model, compute capability, total memory,
and GPU architecture size. Visible device ordinals and physical GPU UUIDs are
not included. Consequently, an A100 marker applies to an equivalent A100 Slurm
worker, but is ignored when the queue resumes on a B200 or a different-memory
GPU. Multiple hardware-specific markers can coexist in one shared queue.

On the next identical invocation on equivalent hardware, the pipeline
automatically enables its simple low-memory strategy:

- unload the orientation model before segmentation;
- disable VibeSeg model caching and unload the SPINEPS/VERIDAH bundle after its
  stage; and
- unload the tissue nnU-Net after its stage.

Thus only the currently active segmentation bundle is retained. There is no
CPU offload/eviction scheduler and no public model-order option. Normal runs
use a stage-aware cache: compatible models remain resident, but prior
segmentation bundles are released before the memory-intensive task-297/299
measurement-support predictor is loaded, and that support predictor is
released immediately after use. The support-model exporter applies the pinned
nnU-Net interpolation to bounded channel groups and updates the multiclass
argmax incrementally; it does not allocate every class on the full source grid
at once. These controls prevent the standard pipeline from requiring all model
families or the complete task-297 logit grid to fit in memory simultaneously.

The explicit `--low-resource` L3 preset enables the same model-unloading
strategy from the first attempt rather than waiting for an OOM marker. It also
changes the scientific analysis scope and model choices as documented in
[low-resource.md](low-resource.md); it is not an automatic fallback for a
full-CT request.

## Device fallback

`runtime.device: auto` resolves in this order:

1. the externally visible CUDA device;
2. CPU.

Apple MPS is intentionally not exposed as a released inference target because
the complete pinned CTDeepRot, SPINEPS/VERIDAH, tissue, and optional
measurement-support stack has not passed a controlled MPS validation gate.
Explicit `cuda` fails clearly when CUDA is unavailable; `auto` is the portable
default.
