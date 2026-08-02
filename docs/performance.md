# Performance and resource observations

This page reports measured behavior from release-candidate validation. It does
not define minimum hardware. Runtime and memory depend on CT dimensions and
coverage, storage, model-cache state, enabled models, and accelerator software.

## Current measurements

The GPU measurements used isolated Docker containers on one NVIDIA GB10,
Linux arm64, 20 available CPU cores, and 124,518.4 MiB of unified host memory.
Models were mounted from an external cache; no model weights were built into
the container. One process owned the GPU.

The CPU measurements used a 10-core Apple M1 Max with 64 GiB unified memory,
macOS 26.5.2, Python 3.11.14, and exact clean source `b972804`. The same pinned
external model cache and licensed public CT were used for both workflows. The
input was 512 × 512 × 75 with 0.703125 × 0.703125 × 5.0-mm spacing and SHA-256
`07f4fbd06c775508dfab2545c742740bcbc1fc49b69ba10403a195a5843b3757`.

| Compute | Workflow | Input and scope | Cases | Case-manifest time | Throughput |
| --- | --- | --- | ---: | ---: | ---: |
| NVIDIA GB10 | Default | Validation cohort, 33–486 slices per CT | 100 | mean 360.3 s; median 327.2 s; P5–P95 149.0–748.6 s | 9.98 cases/h |
| NVIDIA GB10 | Default | Public CT, 512 × 512 × 75, 5-mm slices | 1 | 220.1 s | — |
| NVIDIA GB10 | SPINEPS/VERIDAH + optional BOA tissue backend | Same public CT | 1 | 215.9 s | — |
| NVIDIA GB10 | `--low-resource` | Same public CT; L3-only analysis | 1 | 52.5 s | — |
| Apple M1 Max CPU | Default | Same public CT; full longitudinal analysis | 1 | 4,163.1 s (69 min 23 s) | — |
| Apple M1 Max CPU | `--low-resource` | Same public CT; L3-only analysis | 1 | 440.8 s (7 min 21 s) | — |

For the 100-case run, the sum of case-manifest durations was 10:00:34.9. The
resource-monitoring window was 10:01:14, which includes approximately 39
seconds of queue and process overhead and gives the reported throughput. Case
durations ranged from 91.9 to 776.5 seconds. The scan-length range is included
because a single short public CT is not representative of cohort throughput.

The public-CT values are direct complete-case timings from validated
candidates. The BOA row uses the same orientation, vertebral, measurement, and
landmark stages as the default; only the tissue backend differs. Formal tests
of exact source `9f24c03`, which additionally include
test assertions and identical-result reuse, took 242.0 seconds for the default
workflow and 63.6 seconds for `--low-resource`. Those test-wall times should
not be substituted for case-manifest durations.

For the M1 Max measurements, independent process monitors observed 4,169.8
seconds and 445.2 seconds of wall time, respectively. The small difference
from the case-manifest values covers process startup, shutdown, and monitor
overhead. Both immutable result bundles passed `results inspect`. These are
single-case feasibility observations, not cohort averages.

## Observed memory

| Platform and run | Peak summed process-tree RSS | Peak total system memory used | Increase from baseline | GPU-process memory |
| --- | ---: | ---: | ---: | ---: |
| Apple M1 Max, default public CT | 52.0 GiB | 42.6 GiB | 24.3 GiB | — |
| Apple M1 Max, low-resource public CT | 24.9 GiB | 37.3 GiB | 17.9 GiB | — |
| NVIDIA GB10, 100-case default run | — | 34,836.48 MiB | — | 16,422 MiB |

The GB10 is a unified-memory platform. The GPU-process value is the allocation
reported by the NVIDIA runtime, not a measured dedicated-VRAM requirement.
The 100-case run showed no material host or GPU growth across 2,247 samples.

The M1 Max monitor sampled the complete process tree every 0.25 seconds.
macOS did not expose a usable unique-set-size value through the monitoring
interface. Summed RSS can therefore count shared pages more than once, while
the baseline-to-peak system-used delta can include unrelated host activity.
Both are retained to make the limitation explicit. None of the observed
figures should be presented as a minimum RAM or VRAM requirement.

## Interpretation

The workflows produce different outputs. The default path uses
SPINEPS/VERIDAH for vertebral localization, ResEncL for full longitudinal
compartment segmentation, and the standard landmark stages. The low-resource
path uses the ResEncM vertebral-body and tissue models, crops tissue inference
to an L3-centred region without in-plane resizing, and exports L3-scoped
measurements. Its shorter runtime is therefore not an equivalent full-volume
speedup.

The 100-case figures describe sequential processing with one worker assigned
to one GPU. They should not be multiplied by worker count to predict
multi-accelerator throughput without measuring storage and CPU contention.

## Adding another accelerator

A100, B200, or other accelerator measurements should append rows to the tables
above only after running the same pinned container and model revisions. Record:

- accelerator and CPU platform, architecture, container/source identity, and
  CUDA/runtime versions;
- input dimensions and scan coverage;
- cold or warm model-cache state and number of GPU workers;
- case-manifest mean, median, P5–P95, and total wall-clock throughput; and
- peak host and GPU-process memory from a time-series monitor.

This keeps later comparisons reproducible without turning observations from
one platform into unsupported hardware requirements.
