# Model setup, verification, and licensing

Model weights are external runtime assets. They are not committed, packaged in
the wheel/source archive, or copied into the container. The model directory is
selected by `models.root` or `BODYCOMPOSITION_MODEL_ROOT`.

```bash
bodycomposition models list --json
bodycomposition models sync
bodycomposition models verify --json
```

`sync` is explicit, idempotent, and separate from inference. It uses an exact
original upstream URL or immutable repository revision, stages into a
temporary directory, checks byte size and SHA-256, then promotes atomically.
An invalid existing bundle is never overwritten silently. `verify` performs
the same pinned checks without network access. Inference requires successful
verification and cannot trigger upstream download code.

If an application update changes only the local asset descriptor, an explicit
`models sync` may atomically refresh that descriptor without downloading when
every pinned upstream file still has its exact expected size and SHA-256.
Inference and `models verify` never perform this repair.

The container uses Hugging Face's standard HTTPS transfer path and a
ten-minute per-file download timeout. This avoids optional transport-helper
deadlocks observed with multi-gigabyte checkpoints on Linux arm64 while
retaining the same original repository, immutable revision, atomic staging,
and digest verification. It is a transport setting, not a model mirror or
scientific configuration.

## Released model IDs

| Model ID | Role | Source/license policy |
| --- | --- | --- |
| `ctdeeprot_2d_v1` | default CT orientation assessment | CTDeepRot code/conventions pinned to commit `492114b8f9f3a7f058d4e97c0dd3643fb8d39649`; BSD-3-Clause; exact upstream checkpoint sync only |
| `spineps_veridah_ct_v1` | default vertebral-body segmentation and labeling | upstream `SPINEPS==2.0.0`; VERIDAH is its pinned `ct_labeling` model; `TPTBox==0.7.5` supplies VibeSeg Dataset100 crop inference; upstream release sync only |
| `bodycomposition_resenc_l_v1` | default anatomical compartments | original `fhofmann/BodyCompositionCT-ResEncL` revision `b355aa7254f5307b6d18005dfbabae8c439d24fb`; CC-BY-4.0 weights |
| `bodycomposition_resenc_m_v1` | smaller selectable tissue model | original `fhofmann/BodyCompositionCT-ResEncM` revision `9c60c8f59a99442b9b8cc1a45abf6c4d81690c0d`; CC-BY-4.0 weights |
| `boa_body_regions_task542_v1` | optional BOA body-region tissue source | BOA v1.0.2 code commit `6e761702a738ba681e6f7a3a7c944b00b186c3a8`; original `v1.0.0-weights` Task 542 archive pinned by size and SHA-256; Apache-2.0 |
| `vertebral_bodies_resenc_l` | selectable corpus-only vertebral model | immutable public Hugging Face revision; CC-BY-SA-4.0 weights |
| `vertebral_bodies_resenc_m` | smaller selectable corpus-only vertebral model | immutable public Hugging Face revision; CC-BY-SA-4.0 weights |
| `totalsegmentator_total_task297_landmarks_v1` | optional ribs/hips for mid-waist landmarks | TotalSegmentator task 297 from the exact official `v2.0.0-weights` archive; Apache-2.0; direct `nnUNetv2==2.5.2` inference |
| `totalsegmentator_body_task299_v1` | optional body/trunk support | TotalSegmentator task 299 from the exact official `v2.0.0-weights` archive; Apache-2.0; direct `nnUNetv2==2.5.2` inference |

BodyComposition does not use TotalSegmentator's separately licensed
`tissue_types` or `vertebrae_body` tasks.

## Default cache layout

```text
<model-root>/
├── CTDeepRot/<commit>/net2d.pt
├── SPINEPS/spineps-veridah-ct-v1/
├── Dataset611_BodyComposition/<trainer>/...
├── Dataset601_VertebralBodies/<trainer>/...
├── Dataset542_BCA_inference/<trainer>/...
├── Dataset297_TotalSegmentator_total_3mm_1559subj/...
└── Dataset299_body_1559subj/...
```

Exact trainers, required files, sizes, hashes, upstream links, compatibility,
and citations are returned by `models list` and recorded path-free in case
provenance.

The task archives are native nnUNet result directories. BodyComposition calls
the pinned upstream nnUNet predictor directly, after model verification, and
reads the label mapping from the verified task's `dataset.json`. It does not
need TotalSegmentator's mutable runtime configuration or downloader, so the
model root remains read-only throughout inference.

The high-level TotalSegmentator distribution is intentionally not a runtime
dependency. Its current declared dependency range resolves alongside the
pinned SPINEPS environment but its inference imports require newer,
incompatible nnUNet/acvl-utils APIs. The narrow direct adapter uses the same
official model, plans, fold, checkpoint, preprocessing, and nnUNet inference
engine without copying or reimplementing segmentation code.

Task 297 has 118 mutually exclusive output classes. The pinned nnU-Net
single-array exporter otherwise materializes every class on the complete
source grid at once. BodyComposition therefore calls the unchanged upstream
preprocessor, predictor, and interpolation function, but resamples channels in
bounded groups and updates the multiclass argmax incrementally. This is
numerically equivalent to the upstream non-region softmax decision and is
covered by a direct equivalence regression; it changes peak memory, not the
model or label definition.

## Public tissue-model sources

The default ResEncL and optional ResEncM tissue assets are pinned to immutable
commits in their original public Hugging Face repositories. The repository
model cards, CC BY 4.0 licenses, inference metadata, checkpoint hashes, and
file sizes were verified before the revisions were recorded. Synchronization
downloads only the dataset metadata, plans, and checkpoint files required for
inference. It does not fetch training logs or evaluation artifacts.

Changing either revision is a model release and validation event. Do not point
the model manager at `main`, substitute a mirror, or update a digest merely to
accept changed bytes.

## Optional BOA Task 542 integration

BOA is selected only with `--tissue-backend boa_body_regions_task542_v1` or
the corresponding configuration/API option. The complete BOA application is
not installed: it adds unrelated reporting/PACS dependencies, owns mutable
download paths, and pins a different PyTorch stack. BodyComposition instead
uses the already pinned upstream `nnUNetPredictor` with the official Task 542
plans, trainer, five folds, and checkpoints.

Model synchronization downloads
`Dataset542_BCA_inference.zip` from BOA's original GitHub weight release. The
1,150,464,739-byte archive is fixed by SHA-256
`096d6cdb45524273026d30e3bd290e8fd10990dec02f0aa0669d214b71bf6d4f`;
every required extracted file is also verified. Synchronization is explicit,
atomic, and idempotent. Inference has no network path.

The adapter reproduces BOA v1.0.2's model boundary: an RAS model-space copy,
cubic resampling of slice thickness only to 5 mm, unchanged nnU-Net inference,
nearest-neighbour restoration, and the upstream connected-component policy.
The returned label image must match the orientation-prepared CT physical
domain exactly. This narrow compatibility code is regression-tested against
the pinned upstream behavior; no BOA, TotalSegmentator, or nnU-Net source is
vendored.

Task 542 predicts body regions, not the same seven compartments as the default
ResEnc models. Native labels therefore retain BOA's names. The standard HU
profile may derive SM and SAT tissue from the muscle and subcutaneous regions,
aVAT tissue from the abdominal cavity, and tVAT tissue from the union of
thoracic cavity and mediastinum. The latter is a versioned BodyComposition
mapping, not an upstream BOA tVAT label. Whole-volume native-compartment HU
distributions do not relabel BOA cavities as aVAT or tVAT; those two native
histogram channels are explicitly unavailable for this backend.

## SPINEPS/TPTBox integration boundary

Segmentation and labeling call the upstream Python APIs. BodyComposition wraps
them narrowly for stable paths, prepared-image geometry, QC, provenance, and
failure behavior. It does not vendor SPINEPS, VERIDAH, TPTBox, or VibeSeg.

The local synchronization/extraction boundary exists because upstream paths
may place crop weights inside `site-packages` and permit hidden downloads.
BodyComposition instead verifies all archives, extracts safely, and supplies
package-external paths. The upstream citation reminder is routed to normal
stderr logging so automation JSON remains valid; the citation and license
obligations remain in notices and provenance.

TPTBox 0.7.5's VibeSeg predictor invokes CUDA-only memory telemetry even for
its supported CPU device. BodyComposition therefore applies a locked,
CPU-only guard while serializing the complete in-process TPTBox/SPINEPS
boundary: GPU waiting is disabled, available host memory feeds the upstream
memory gate, and both telemetry functions are restored afterward. CUDA uses
the original behavior, and CPU activation is recorded in crop-model
provenance. The exact upstream tag, function, reason, and tests are documented
in `THIRD_PARTY_NOTICES.md`.

SPINEPS's pinned CT model declares an isotropic acquisition and its coarse
preflight rejects anisotropic files before inference. BodyComposition accepts
governed routine CT geometry, allows SPINEPS's own documented
reorientation/resampling path to its 0.8-mm model grid, and then requires every
returned mask to match the orientation-prepared CT physical domain. The
selected compatibility policy is recorded in vertebral provenance; no image is
stretched or independently reoriented by the adapter.

SPINEPS places derived intervertebral-disc and endplate instances in the same
upstream instance image using its documented +100 and +200 label ranges.
BodyComposition preserves that native file when requested, but removes those
auxiliary instances from its `vertebra_labels` output before QC and downstream
use. The downstream `vertebral_bodies` mask is then the vertebra-only instance
mask intersected with semantic vertebral corpus label 49 and sacral-body label
73. Thus discs and posterior elements are never used for body-composition
territories, while the sacrum remains an explicit anatomical level.

## Citation and redistribution

Code and weight licenses are separate. BodyComposition's Apache-2.0 license
does not relicense model weights. Cite every model used by a run. Complete
attribution, TPTBox's Apache-2.0 license, required papers, and redistribution
decisions are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
