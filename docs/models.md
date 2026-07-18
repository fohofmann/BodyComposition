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

## Released model IDs

| Model ID | Role | Source/license policy |
| --- | --- | --- |
| `ctdeeprot_2d_v1` | default CT orientation assessment | CTDeepRot code/conventions pinned to commit `492114b8f9f3a7f058d4e97c0dd3643fb8d39649`; BSD-3-Clause; exact upstream checkpoint sync only |
| `spineps_veridah_ct_v1` | default vertebral-body segmentation and labeling | upstream `SPINEPS==2.0.0`; VERIDAH is its pinned `ct_labeling` model; `TPTBox==0.7.5` supplies VibeSeg Dataset100 crop inference; upstream release sync only |
| `bodycomposition_resenc_l_v1` | default tissue compartments | internal nnU-Net v2 ResEncL; declared CC-BY-SA-4.0 weight terms; release revision unresolved |
| `bodycomposition_resenc_m_v1` | smaller selectable tissue model | internal nnU-Net v2 ResEncM; declared CC-BY-SA-4.0 weight terms; release revision unresolved |
| `vertebral_bodies_resenc_l` | selectable corpus-only vertebral model | immutable public Hugging Face revision; CC-BY-SA-4.0 weights |
| `vertebral_bodies_resenc_m` | smaller selectable corpus-only vertebral model | immutable public Hugging Face revision; CC-BY-SA-4.0 weights |
| `totalsegmentator_total_task297_landmarks_v1` | optional ribs/hips for mid-waist landmarks | TotalSegmentator 2.15.0 task 297, exact official release archive; Apache-2.0 |
| `totalsegmentator_body_task299_v1` | optional body/trunk support | TotalSegmentator 2.15.0 task 299, exact official release archive; Apache-2.0 |

BodyComposition does not use TotalSegmentator's separately licensed
`tissue_types` or `vertebrae_body` tasks.

## Default cache layout

```text
<model-root>/
├── CTDeepRot/<commit>/net2d.pt
├── SPINEPS/spineps-veridah-ct-v1/
├── Dataset611_BodyComposition/<trainer>/...
├── Dataset601_VertebralBodies/<trainer>/...
├── Dataset297_TotalSegmentator_total_3mm_1559subj/...
├── Dataset299_body_1559subj/...
└── totalsegmentator_config/
```

Exact trainers, required files, sizes, hashes, upstream links, compatibility,
and citations are returned by `models list` and recorded path-free in case
provenance.

## Current release blocker

The default ResEncL and optional ResEncM tissue assets have exact verified
local layouts and hashes, but their intended original Hugging Face repositories
are not public at immutable revisions. Consequently:

- a local cache containing the exact files can pass operational verification;
- `models sync` refuses to invent or use an unpinned source;
- `models verify` reports `upstream_revision_unresolved` as a release issue;
  and
- `doctor` cannot report `release_ready=true`.

The blocker is resolved by publishing/freezing the original model repositories
with model cards and the declared weight license, then recording their immutable
revisions. Rehosting another project's weights or embedding any weights in the
container is not an acceptable workaround.

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

## Citation and redistribution

Code and weight licenses are separate. BodyComposition's Apache-2.0 license
does not relicense model weights. Cite every model used by a run. Complete
attribution, the TPTBox wheel-metadata discrepancy, required papers, and
redistribution decisions are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
