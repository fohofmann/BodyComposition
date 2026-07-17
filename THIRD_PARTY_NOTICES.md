# Third-party notices

## CTDeepRot

BodyComposition contains an adapted, narrow implementation of the 2D feature
and proper-rotation conventions from
[CTDeepRot](https://github.com/JakubicekRoman/CTDeepRot), pinned to commit
`492114b8f9f3a7f058d4e97c0dd3643fb8d39649`.

CTDeepRot is the appropriate and required anatomy-based proper-rotation model
for BodyComposition's default orientation-integrity stage. The CTDeepRot model,
preprocessing method, and rotation conventions are the work of Roman
Jakubicek, Tomas Vicar, Jan Chmelik, and the CTDeepRot contributors;
BodyComposition supplies the integration, geometry handling, safety gates, and
review workflow. This acknowledgment identifies and credits the dependency and
does not imply that the CTDeepRot authors endorse BodyComposition.

The BSD-3-Clause conditions below are the legal redistribution requirements.
Scholarly citation is additionally required by BodyComposition project policy
whenever results are produced with the default orientation stage.

The pretrained `net2d.pt` checkpoint is not stored in the BodyComposition
repository or source distribution. The model manager downloads the exact file
from the pinned upstream commit and verifies SHA-256
`a2feb521cfe49c367c4e594f38fa76ba9982ba009e114fbf15fb2aae06c06d96`.
Because the upstream repository does not provide a separate
checkpoint-specific license notice, BodyComposition does not redistribute the
checkpoint in its repository, wheel, source distribution, or public container
artifacts. A future redistribution decision requires explicit confirmation of
the model-weight terms.

Required scholarly acknowledgment:

> Jakubicek R, Vicar T, Chmelik J. A Tool for Automatic Estimation of Patient
> Position in Spinal CT Data. IFMBE Proceedings. 2021;80:51–56.
> https://doi.org/10.1007/978-3-030-64610-3_7

BSD 3-Clause License

Copyright (c) 2020, Jakubicek Roman
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## SPINEPS, VERIDAH, TPTBox, and VIBESegmentator

The default vertebral backend calls the upstream `SPINEPS==2.0.0` Python API
and its pinned `ct_labeling` model (VERIDAH). It uses `TPTBox==0.7.5` for the
BIDS/image boundary and VibeSeg Dataset100 crop inference. BodyComposition does
not vendor or rewrite these projects. Its adapter adds explicit model paths,
disables inference-time downloads, validates SimpleITK geometry, preserves
native labels, and records QC/provenance.

The only locally implemented upstream boundary is model transfer and archive
extraction. The pinned upstream download paths are unsuitable for this package
because they can resolve VibeSeg weights inside `site-packages`, permit hidden
inference-time downloads, and do not provide BodyComposition's complete
archive-hash, safe-extraction, and atomic-promotion contract. No segmentation
or labeling algorithm is copied. The boundary is tested for the same pinned
SPINEPS tag commit `ad622b87d9e4b81fb6a88df2a8050bd40f7046d5` and TPTBox
0.7.5 APIs, including reuse of the precomputed upstream crop output without a
download.

SPINEPS, TPTBox, and the VIBESegmentator source repositories declare the
Apache License 2.0. The TPTBox 0.7.5 wheel metadata contains an inconsistent
AGPL classifier even though its included LICENSE and upstream repository state
Apache-2.0. The project decision uses the repeatedly declared Apache-2.0
project license for this pin. The wheel-metadata discrepancy is retained as a
non-blocking provenance note.

BodyComposition does not redistribute the pinned SPINEPS
semantic/instance/VERIDAH archives or VibeSeg Dataset100 archives. They are not
stored in this repository, wheel, source distribution, or public container.
The model directory is mounted at runtime, and the BodyComposition model-sync
command downloads the exact assets from their original upstream release URLs,
verifies their byte sizes and SHA-256 digests, and installs them atomically.
This upstream-sync-only policy is the release decision; permission to rehost or
bundle the weights is not required for the weight-free release. A future choice
to redistribute weights would require a separate review.

Required scholarly acknowledgments include:

- Möller H, Graf R, Schmitt J, et al. SPINEPS—automatic whole spine
  segmentation of T2-weighted MR images using a two-phase approach to
  multi-class semantic and instance segmentation. European Radiology. 2024.
  https://doi.org/10.1007/s00330-024-11155-y
- Möller H, Schoen H, Graf R, et al. VERIDAH: Solving Enumeration Anomaly
  Aware Vertebra Labeling across Imaging Sequences. arXiv:2601.14066. 2026.
- Graf R, Platzek P, Riedel EO, et al. VIBESegmentator: full body MRI
  segmentation for the NAKO and UK Biobank. European Radiology. 2025.
  https://doi.org/10.1007/s00330-025-12035-9

The complete Apache License 2.0 is available in the upstream repositories and
in the installed distributions. Copyright remains with their respective
authors and contributors. This acknowledgment does not imply endorsement of
BodyComposition by those projects.

## TotalSegmentator

BodyComposition pins `TotalSegmentator==2.15.0` and can call its upstream
Python API narrowly for optional canonical measurement support masks:

- the 1.5-mm `body` model (task 299), when explicitly selected instead of the
  default tissue-derived envelope, supplies `body_trunc` and
  `body_extremities`; and
- the 3-mm `total` model (task 297) runs once through the upstream fast API;
  the BodyComposition landmark adapter consumes only its bilateral hip and rib
  labels for anatomical mid-waist landmarks.

The upstream project and installed 2.15.0 distribution carry the Apache
License 2.0. TotalSegmentator's official task documentation lists `total` and
`body` as openly available for any usage under Apache-2.0. These are distinct
from TotalSegmentator subtasks such as `tissue_types` and `vertebrae_body`,
which the upstream project places under separate licensed/non-commercial
terms and which are not used for measurement stage's body surface or landmark inference.

BodyComposition does not vendor TotalSegmentator source or models and does not
place model weights in its repository, wheel, source distribution, or public
container. The model-sync command downloads the exact task-299 and task-297
archives from TotalSegmentator's original `v2.0.0-weights` GitHub release,
verifies the pinned archive byte size and SHA-256 before safe extraction, checks
the required extracted files, and atomically promotes the result into the
configured mounted model directory. This narrow transfer/extraction boundary is
implemented locally because the upstream downloader extracts directly into its
final directory, cannot verify the archive against a published pin before
extraction, and treats a partial existing directory as already downloaded. No
segmentation or labeling code is copied. Before inference, the adapter requires
the exact task-299 or task-297 model directory and checkpoint to exist; it does
not permit the upstream API's normal inference-time download behavior to satisfy
a missing asset.

Analyses using these masks must acknowledge TotalSegmentator and nnU-Net and
cite:

> Wasserthal J, Breit H-C, Meyer MT, et al. TotalSegmentator: Robust
> Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial
> Intelligence. 2023. https://doi.org/10.1148/ryai.230024
