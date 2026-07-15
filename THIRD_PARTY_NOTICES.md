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
