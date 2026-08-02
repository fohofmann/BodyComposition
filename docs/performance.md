# Performance and resource observations

This page reports measured behavior from release-candidate validation. It does
not define minimum hardware. Runtime and memory depend on CT dimensions and
coverage, storage, model-cache state, enabled models, and accelerator software.

## Current measurements

The measurements below used isolated Docker containers on one NVIDIA GB10,
Linux arm64, 20 available CPU cores, and 124,518.4 MiB of host memory. Models
were mounted from an external cache; no model weights were built into the
container. One process owned the GPU.

| Accelerator | Workflow | Input and scope | Cases | Observed end-to-end time | Throughput |
| --- | --- | --- | ---: | ---: | ---: |
| NVIDIA GB10 | Default | Validation cohort, 33–486 slices per CT | 100 | mean 360.3 s; median 327.2 s; P5–P95 149.0–748.6 s | 9.98 cases/h |
| NVIDIA GB10 | Default | Public CT, 512 × 512 × 75, 5-mm slices | 1 | 220.1 s | — |
| NVIDIA GB10 | SPINEPS/VERIDAH + optional BOA tissue backend | Same public CT | 1 | 215.9 s | — |
| NVIDIA GB10 | `--low-resource` | Same public CT; L3-only analysis | 1 | 52.5 s | — |

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

## Observed memory

| Run | Peak host memory | Peak GPU-process memory | Repeated-run behavior |
| --- | ---: | ---: | --- |
| 100 cases, default workflow, one GPU worker | 34,836.48 MiB (27.98% of the available host memory) | 16,422 MiB | No material host or GPU growth across 2,247 samples |

The GB10 is a unified-memory platform. The GPU-process value is the allocation
reported by the NVIDIA runtime, not a measured dedicated-VRAM requirement.
The single-case low-resource validation did not record a separate defensible
peak-memory series, so no low-resource memory figure is reported. In
particular, neither 34.0 GiB of host memory nor 16.0 GiB of reported GPU-process
memory should be presented as the minimum required to run the pipeline.

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
