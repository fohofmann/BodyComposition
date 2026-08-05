# Containers

BodyComposition provides one weight-free, non-root image definition for Linux
`amd64` and `arm64`. The same image supports CPU or an externally exposed
NVIDIA GPU; the selected pipeline is controlled by the ordinary CLI.

Version `1.0.0rc1` is a reviewed pre-release. Source CI and container
publication are separate; the repository provides reproducible Buildx tooling
but does not publish registry images from its ordinary CI workflow.

## Use a published image

Ordinary users only need the published registry image:

```bash
docker pull ghcr.io/fohofmann/bodycomposition:1.0.0rc1
```

Verify that the selected tag is available before deployment. The local build
helper is for maintainers and users testing a checkout; it is not part of the
normal installed workflow.

## Build a checkout locally

Maintainers can build a clean checkout with:

```bash
uv run python scripts/build_container.py \
  --tag bodycomposition:1.0.0rc1
```

The helper records package version, Git revision, lock digest, source-context
digest, source date, platform, and the exact Buildx command. It refuses a dirty
release build; `--allow-dirty` is only for explicitly recorded development
images.

| Build mode | Option |
| --- | --- |
| Current host architecture, loaded locally | default or `--load` |
| Explicit architecture | `--platform linux/amd64` or `linux/arm64` |
| Both architectures in one registry tag | `--platform linux/amd64,linux/arm64 --push` |
| Rebuild base and dependency layers | `--no-cache` |
| Use already available base images | `--no-pull` |

A multi-platform tag is an OCI registry manifest pointing to one image per
architecture. Docker selects the correct image at pull time; this is Buildx
and registry behavior, not a runtime branch in the Dockerfile. An
Apptainer/Singularity `.sif` remains architecture-specific.

## Mounts

| Container path | Access during inference | Contents |
| --- | --- | --- |
| `/input` | read-only | NIfTI CTs or DICOM directories |
| `/models` | read-only | synchronized, verified external model assets |
| `/output` | writable | result bundles and coordination state |

The image runs as UID/GID `10001`. Writable host directories must permit that
identity. Model synchronization needs a temporary writable `/models` mount;
inference can then mount the verified cache read-only.

```bash
mkdir -p input models output

docker run --rm \
  -v "$PWD/models:/models" \
  bodycomposition:1.0.0rc1 models sync

docker run --rm \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/output:/output" \
  bodycomposition:1.0.0rc1 doctor --json

docker run --rm --gpus all \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/output:/output" \
  bodycomposition:1.0.0rc1 analyze /input/CT.nii.gz --device cuda --json
```

Remove `--gpus all` and use `--device cpu` for CPU-only execution. The frozen
container runtime uses CUDA 13.0 and requires NVIDIA driver branch R580 or
newer for CUDA execution. Always run `doctor` inside the same runtime and with
the same mounts used for inference.

## Registry publication

To publish both supported architectures to a registry you control:

```bash
uv run python scripts/build_container.py \
  --platform linux/amd64,linux/arm64 \
  --tag registry.example.org/bodycomposition:1.0.0rc1 \
  --push
```

Buildx adds provenance and SBOM attestations unless explicitly disabled.
Registry publication may be performed manually or by a dedicated release
workflow; no Dockerfile change is required.

## Apptainer and Slurm

Build a SIF matching the cluster architecture from an available registry tag:

```bash
apptainer build bodycomposition_1.0.0rc1.sif \
  docker://registry.example.org/bodycomposition:1.0.0rc1
```

Use `apptainer exec --nv` for GPU execution and bind the same input, model, and
output directories. Submit builds and inference through the site's scheduler
when they exceed login-node limits. BodyComposition does not encode partition,
account, memory, time, or array size; see the generic [Slurm worker
pattern](execution.md#slurm-array).

## Audit

Before publishing a locally loaded image:

```bash
uv run python scripts/container_checks.py bodycomposition:1.0.0rc1
```

The checks verify runtime metadata and the final image layers, reject embedded
models or medical-image assets, and create CycloneDX and vulnerability
receipts. Model weights and CTs must never be copied into an image.
