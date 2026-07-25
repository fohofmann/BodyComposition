# Third-party notices

## SimpleITK and GDCM DICOM reader

BodyComposition uses the upstream SimpleITK dependency and its GDCM ImageIO
support for DICOM series discovery, assembly, and NIfTI writing. No SimpleITK,
ITK, or GDCM source is copied or modified in this repository. The locked
SimpleITK distribution is Apache-2.0 and carries its own LICENSE and NOTICE;
those files remain part of the installed upstream distribution. Conversion
preserves the reader geometry, and BodyComposition adds strict CT-series
selection, privacy-safe provenance, atomic output, and pixel/physical-domain
verification.

Suggested scholarly acknowledgment:

> Lowekamp BC, Chen DT, Ibanez L, Blezek D. The Design of SimpleITK. Frontiers
> in Neuroinformatics. 2013;7:45. https://doi.org/10.3389/fninf.2013.00045

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

## Internal tissue and alternative vertebral-body models

BodyComposition integrates nnU-Net v2 inference for the project-specific
BodyCompositionCT and VertebralBodiesCT model families. nnU-Net source is
Apache-2.0 and remains an upstream package dependency; no nnU-Net source is
vendored. Analyses using these models should cite:

> Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a
> self-configuring method for deep learning-based biomedical image
> segmentation. Nature Methods. 2021;18:203–211.
> https://doi.org/10.1038/s41592-020-01008-z

The VertebralBodiesCT ResEncL and ResEncM weights are published under
CC-BY-SA-4.0 at immutable Hugging Face revisions recorded by the model manager.
The BodyCompositionCT ResEncL and ResEncM weights, documentation, metadata, and
included training/evaluation records are published under CC-BY-4.0 in their
original Hugging Face repositories. The model manager records immutable
revisions, exact inference-file sizes, and SHA-256 digests and downloads those
assets only through explicit user model synchronization.

No BodyCompositionCT or VertebralBodiesCT weight is stored in this repository,
wheel, source distribution, or container. Weight licenses remain with their
authors and are not converted to Apache-2.0. Cite the model card and the
associated validation publication when using these assets:

> Hofmann FO, et al. Validation of body composition parameters extracted via
> deep learning-based segmentation from routine computed tomographies.
> Scientific Reports. 2025;15:11909.
> https://doi.org/10.1038/s41598-025-96238-6

The exact asset IDs, files, SHA-256 digests, revisions, and compatibility are
returned by `bodycomposition models list --json` and recorded in run
provenance.

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

## ReportLab, pypdf, and Bitstream Vera

Optional PDF case reports are rendered with ReportLab and collated/validated
with pypdf. Both projects are distributed under BSD-3-Clause terms; their
installed distributions include the complete license texts. BodyComposition
does not modify either library.

Generated reports embed subsets of the unmodified Bitstream Vera Sans regular
and bold fonts supplied with ReportLab. The required font notice is:

> Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. Bitstream Vera is
> a trademark of Bitstream, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
the fonts accompanying this license ("Fonts") and associated documentation
files (the "Font Software"), to reproduce and distribute the Font Software,
including without limitation the rights to use, copy, merge, publish,
distribute, and/or sell copies of the Font Software, and to permit persons to
whom the Font Software is furnished to do so, subject to the following
conditions:

The above copyright and trademark notices and this permission notice shall be
included in all copies of one or more of the Font Software typefaces.

The Font Software may be modified, altered, or added to, and in particular the
designs of glyphs or characters in the Fonts may be modified and additional
glyphs or characters may be added to the Fonts, only if the fonts are renamed
to names not containing either the words "Bitstream" or the word "Vera".

This License becomes null and void to the extent applicable to Fonts or Font
Software that has been modified and is distributed under the "Bitstream Vera"
names.

The Font Software may be sold as part of a larger software package but no copy
of one or more of the Font Software typefaces may be sold by itself.

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT,
TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL BITSTREAM OR THE GNOME FOUNDATION
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING ANY GENERAL,
SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF THE USE OR INABILITY TO USE
THE FONT SOFTWARE OR FROM OTHER DEALINGS IN THE FONT SOFTWARE.

Except as contained in this notice, the names of GNOME, the GNOME Foundation,
and Bitstream Inc. shall not be used in advertising or otherwise to promote the
sale, use, or other dealings in this Font Software without prior written
authorization from the GNOME Foundation or Bitstream Inc., respectively.

BodyComposition embeds the original, unmodified fonts and records their
SHA-256 digests in each report identity/manifest.

## TotalSegmentator

BodyComposition uses two official TotalSegmentator model archives for optional
canonical measurement support masks:

- the 1.5-mm `body` model (task 299), when explicitly selected instead of the
  default tissue-derived envelope, supplies `body_trunc` and
  `body_extremities`; and
- the 3-mm `total` model (task 297) supplies the bilateral hip and rib labels
  consumed by the BodyComposition anatomical mid-waist adapter.

The upstream project carries the Apache License 2.0. TotalSegmentator's
official task documentation lists `total` and `body` as openly available for
any usage under Apache-2.0. These are distinct from TotalSegmentator subtasks
such as `tissue_types` and `vertebrae_body`, which the upstream project places
under separate licensed/non-commercial terms and which are not used for
body-surface or landmark inference.

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
segmentation or labeling code is copied.

The verified archives are native nnUNet result directories. BodyComposition
pins `nnUNetv2==2.5.2` and invokes its upstream `nnUNetPredictor` directly with
the official task plans, fold 0, and `checkpoint_final.pth`; the label schema is
read from the verified upstream `dataset.json`. The high-level TotalSegmentator
Python distribution is not an inference dependency. This boundary avoids its
mutable configuration and downloader and keeps one dependency set compatible
with the pinned SPINEPS environment. Before predictor initialization, the
adapter requires the exact task-299 or task-297 asset to pass all pinned file
checks. The mounted model directory can remain read-only and inference cannot
download a missing asset.

Analyses using these masks must acknowledge TotalSegmentator and nnU-Net and
cite:

> Wasserthal J, Breit H-C, Meyer MT, et al. TotalSegmentator: Robust
> Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial
> Intelligence. 2023. https://doi.org/10.1148/ryai.230024
